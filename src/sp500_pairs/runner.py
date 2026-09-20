from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform
import tomllib
import traceback
import uuid

import numpy as np
import pandas as pd

from .config import Config, ConfigError, load_config
from .data import MarketData, load_data
from .engine import simulate_pair
from .formation import select_pairs
from .corporate_actions import calendar_for
from .pretrade import reference_config
from .schedule import make_windows


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: Path, value):
    path.write_text(json.dumps(json_safe(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def source_fingerprint():
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode()); digest.update(path.read_bytes())
    return digest.hexdigest()


def metrics(daily: pd.DataFrame, trades: pd.DataFrame, cfg: Config) -> dict:
    initial = cfg["portfolio"]["initial_capital"]
    ann = cfg["evaluation"]["annualization"]
    returns = daily["net_return"]
    wealth = np.r_[initial, daily["nav"].to_numpy()]
    drawdown = wealth / np.maximum.accumulate(wealth) - 1
    cumulative = wealth[-1] / initial - 1
    annual_return = (1 + cumulative) ** (ann / len(daily)) - 1
    volatility = returns.std(ddof=1)
    rf = (1 + cfg["evaluation"]["risk_free_annual_rate"]) ** (1/ann) - 1
    return json_safe({
        "first_date": str(daily["date"].iloc[0].date()), "last_date": str(daily["date"].iloc[-1].date()),
        "calendar_observations": len(daily), "initial_capital": initial, "final_nav": wealth[-1],
        "cumulative_return": cumulative, "annualized_return": annual_return,
        "annualized_volatility": volatility * np.sqrt(ann),
        "sharpe": (returns.mean()-rf) / volatility * np.sqrt(ann) if volatility > 0 else np.nan,
        "max_drawdown": drawdown.min(),
        "calmar": annual_return / abs(drawdown.min()) if drawdown.min() < 0 else np.nan,
        "completed_trades": len(trades),
        "merger_announcement_exits": int(trades["exit_reason"].eq("merger_announcement").sum()) if len(trades) else 0,
        "win_rate": float((trades["net_pnl"] > 0).mean()) if len(trades) else None,
        "mean_reversion_exit_rate": float(trades["exit_reason"].eq("mean_reversion").mean()) if len(trades) else None,
        "mean_holding_days": float(trades["holding_days"].mean()) if len(trades) else None,
        "trading_cost": float(daily["trading_cost"].sum()), "borrow_cost": float(daily["borrow_cost"].sum()),
        "traded_notional": float(daily["turnover"].sum()),
        "average_gross_exposure_at_interval_start": float((daily["gross_before"] / daily["nav_start"]).mean()),
    })


def _concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
    columns = list(dict.fromkeys(column for frame in frames for column in frame.columns))
    nonempty = [f for f in frames if not f.empty]
    return pd.concat(nonempty, ignore_index=True) if nonempty else pd.DataFrame(columns=columns)


def save_plot(daily, output: Path, initial: float, data_mode: str = "in_memory"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nav = daily["nav"].to_numpy()
    running = np.maximum.accumulate(np.r_[initial, nav])[1:]
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    axes[0].plot(daily["date"], nav/initial, color="#2463A7", linewidth=1.4)
    axes[0].set_title("Synthetic data: software demonstration" if data_mode == "synthetic" else "Research backtest: supplied price data")
    axes[0].set_ylabel("NAV / initial capital")
    axes[1].fill_between(daily["date"], 100*(nav/running-1), 0, color="#D35B5B", alpha=.7)
    axes[1].set_ylabel("Drawdown (%)")
    for ax in axes:
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output / "equity_and_drawdown.png", dpi=160)
    plt.close(fig)


def run_backtest(cfg: Config, data: MarketData | None = None, data_dir=None,
                 output_dir: str | Path | None = None, write_outputs: bool = True,
                 progress=print, formation_cache: dict | None = None,
                 detector_factory=None, output_profile="full", selection_audit_dir=None) -> dict:
    """detector_factory(pair, formation_prices) may inspect formation only.

    Each detector then receives sequential close observations, including after an exit.
    The trading baseline stays fixed; a detector can request next-close exit and blocking.
    """
    if output_profile not in {"full", "compact"}:
        raise ConfigError("output_profile: full/compact")
    data = data if data is not None else load_data(cfg, data_dir)
    mergers = calendar_for(data, cfg)
    windows = make_windows(data.prices.index, cfg)
    label = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:6]
    if output_dir is not None:
        destination = Path(output_dir).expanduser().resolve()
    else:
        base = Path(cfg["run"]["output_directory"]).expanduser()
        if not base.is_absolute():
            base = cfg.project_root / base
        destination = base / label
    manifest = {
        "status": "running", "run_id": label, "model": "S&P500 log-price Engle-Granger fixed-parameter baseline",
        "data_mode": data.metadata.get("mode", "in_memory"), "data": data.metadata,
        "evaluation_label": "exploratory_not_holdout" if cfg["evaluation"]["active_split"] == "all" else cfg["evaluation"]["active_split"],
        "limited_window_run": bool(cfg["run"]["max_windows"]),
        "execution": "close t signal -> close t+1 fill; adjusted-close research proxy",
        "calendar": "rows of supplied price file; provider must supply the complete market calendar",
        "source_sha256": source_fingerprint(), "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scipy", "statsmodels", "matplotlib")},
        "detector": "none" if detector_factory is None else getattr(detector_factory, "__name__", type(detector_factory).__name__),
        "default_stop_reentry": "block_window; user-adjustable separately from stop_z",
        "output_profile": output_profile,
        "selection_audit_directory": str(selection_audit_dir) if selection_audit_dir is not None else None,
        "merger_policy_enabled": cfg["corporate_actions"]["enabled"],
        "merger_calendar_sha256": mergers.fingerprint,
        "merger_calendar_coverage": mergers.coverage,
    }
    if write_outputs:
        destination.mkdir(parents=True, exist_ok=False)
        write_json(destination / "manifest.json", manifest)
        write_json(destination / "effective_config.json", cfg.values)
        write_json(destination / "merger_events.json", mergers.records())
    tables = {key: [] for key in ("portfolio_daily", "pair_daily", "trades", "tested_pairs", "universe_audit", "direction_tests", "selected_pairs")}
    window_rows = []
    capital = cfg["portfolio"]["initial_capital"]
    formation_cache = formation_cache if formation_cache is not None else {}
    try:
        for number, window in enumerate(windows, 1):
            if progress:
                progress(f"[{number}/{len(windows)}] {window.id}: 포메이션 및 페어 선정")
            key_text = json.dumps({"data_object": getattr(formation_cache, "data_identity", id(data)), "window": window.id,
                "formation_start": str(window.formation_dates[0]), "formation_end": str(window.formation_dates[-1]),
                "data_settings": cfg["data"], "formation": cfg["formation"], "selection": cfg["selection"],
                "mergers": mergers.fingerprint,
                "pretrade": cfg["pretrade"],
                "pretrade_context": {"strategy": reference_config(cfg)["strategy"],
                    "costs": cfg["costs"], "evaluation": cfg["evaluation"], "cash_rate": cfg["portfolio"]["cash_annual_rate"]}
                    if cfg["pretrade"]["enabled"] else None}, sort_keys=True)
            if key_text not in formation_cache:
                formation_cache[key_text] = select_pairs(data, window, cfg, progress, feature_cache=formation_cache)
            pairs, tested, stock_audit, directions, info = formation_cache[key_text]
            tables["tested_pairs"].append(tested)
            tables["universe_audit"].append(stock_audit)
            tables["direction_tests"].append(directions)
            tables["selected_pairs"].append(pd.DataFrame([{"window": window.id, **asdict(pair)} for pair in pairs]))
            slot = capital / cfg["selection"]["max_pairs"]
            unused = (cfg["selection"]["max_pairs"] - len(pairs)) * slot
            cash_nav, previous_date = unused, window.formation_dates[-1]
            cash_rows = []
            for date in window.trading_dates:
                prior = cash_nav
                cash_nav *= (1 + cfg["portfolio"]["cash_annual_rate"]) ** ((date - previous_date).days / 365)
                cash_rows.append({"date": date, "nav_start": prior, "nav": cash_nav, "cash_interest": cash_nav-prior})
                previous_date = date
            pair_frames = []
            for pair in pairs:
                detector = detector_factory(pair, data.prices.loc[window.formation_dates, [pair.x, pair.y]].copy()) if detector_factory else None
                daily, trades = simulate_pair(pair, data.prices.loc[window.trading_dates, [pair.x, pair.y]], slot, cfg, window.id, detector, mergers)
                pair_frames.append(daily); tables["pair_daily"].append(daily); tables["trades"].append(trades)
            columns = ["nav_start", "nav", "gross_pnl", "trading_cost", "borrow_cost", "cash_interest", "turnover", "gross_before", "gross_notional"]
            agg = pd.DataFrame(0.0, index=window.trading_dates, columns=columns)
            if pair_frames:
                agg += pd.concat(pair_frames).groupby("date")[columns].sum().reindex(window.trading_dates)
            cash = pd.DataFrame(cash_rows).set_index("date")
            agg[["nav_start", "nav", "cash_interest"]] += cash[["nav_start", "nav", "cash_interest"]]
            agg["unused_slot_cash"] = cash["nav"]
            agg["window"] = window.id
            agg["net_return"] = agg["nav"] / agg["nav_start"] - 1
            agg.index.name = "date"
            tables["portfolio_daily"].append(agg.reset_index())
            end_capital = float(agg["nav"].iloc[-1])
            window_rows.append({"window": window.id, "formation_start": window.formation_dates[0],
                "formation_end": window.formation_dates[-1], "trading_start": window.trading_dates[0],
                "trading_end": window.trading_dates[-1], "nominal_start": window.nominal_start,
                "nominal_end": window.nominal_end, "formation_observations": len(window.formation_dates),
                "trading_observations": len(window.trading_dates), "capital_start": capital,
                "capital_end": end_capital, "slot_initial_capital": slot, **info})
            capital = end_capital
            if progress:
                progress(f"  선정 {len(pairs)}페어, 빈 슬롯 {cfg['selection']['max_pairs']-len(pairs)}개; 체결/회계 계산 완료")
        result = {key: _concat(frames) for key, frames in tables.items()}
        # An empty output still has useful headers and can be read with read_csv.
        empty_columns = {
            "trades": "window pair_id x y sector direction alpha beta entry_signal_date entry_date entry_signal_z entry_fill_z entry_x entry_y quantity_x quantity_y entry_nav entry_gross entry_cost gross_pnl borrow_cost exit_signal_date exit_date exit_signal_z exit_fill_z exit_x exit_y exit_reason exit_cost exit_nav holding_days net_pnl net_return merger_event_ids",
            "pair_daily": "date window pair_id x y sector zscore nav_start nav gross_pnl trading_cost borrow_cost cash_interest turnover net_return position_before position quantity_x quantity_y notional_x notional_y gross_before gross_notional holding_days blocked waiting_reset event detector_signal merger_blocked merger_event_ids next_order next_reason",
            "selected_pairs": "window id x y sector alpha beta mean std formation_end last_z last_x last_y",
            "tested_pairs": "window pair_id sector x y error pvalue tstat direction_tests pair_pvalue i1_pass selected exclusion decision_pvalue",
            "direction_tests": "window pair_id x y error pvalue tstat",
            "universe_audit": "window ticker sector coverage exclusion level_adf_p difference_adf_p i1_pass merger_event_ids",
        }
        for key, columns in empty_columns.items():
            if len(result[key].columns) == 0:
                result[key] = pd.DataFrame(columns=columns.split())
        result["windows"] = pd.DataFrame(window_rows)
        p = result["pair_daily"]
        if not p.empty and output_profile == "full":
            exposures = []
            for side in ("x", "y"):
                frame = p[["date", side, "notional_"+side]].rename(columns={side: "ticker", "notional_"+side: "net_notional"}).copy()
                frame["gross_independent_sleeves"] = frame["net_notional"].abs()
                exposures.append(frame)
            result["stock_exposure"] = pd.concat(exposures).groupby(["date", "ticker"], as_index=False)[["net_notional", "gross_independent_sleeves"]].sum()
        else:
            result["stock_exposure"] = pd.DataFrame(columns=["date", "ticker", "net_notional", "gross_independent_sleeves"])
        result["summary"] = metrics(result["portfolio_daily"], result["trades"], cfg)
        manifest.update(status="complete", completed_utc=datetime.now(timezone.utc).isoformat(), windows=len(windows))
        result["manifest"], result["output_directory"] = manifest, destination
        if write_outputs:
            for key, value in result.items():
                if isinstance(value, pd.DataFrame):
                    if output_profile == "compact" and key in {"pair_daily", "stock_exposure"}:
                        continue
                    target = destination
                    if selection_audit_dir is not None and key in {"tested_pairs", "universe_audit", "direction_tests"}:
                        target = Path(selection_audit_dir)
                        target.mkdir(parents=True, exist_ok=True)
                        if (target / (key + ".csv")).exists():
                            continue
                        temp = target / (key + ".csv.tmp")
                        value.to_csv(temp, index=False)
                        temp.replace(target / (key + ".csv"))
                    else:
                        value.to_csv(target / (key + ".csv"), index=False)
            write_json(destination / "summary.json", result["summary"])
            write_json(destination / "manifest.json", manifest)
            if output_profile == "full":
                save_plot(result["portfolio_daily"], destination, cfg["portfolio"]["initial_capital"], manifest["data_mode"])
        return result
    except Exception as exc:
        if write_outputs:
            manifest.update(status="failed", error=str(exc), error_type=type(exc).__name__)
            write_json(destination / "manifest.json", manifest)
            (destination / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


def run_grid(path: str | Path, data_dir=None, progress=print):
    path = Path(path).resolve()
    with path.open("rb") as handle:
        spec = tomllib.load(handle)
    base = load_config(path.parent / spec["base"])
    if base["evaluation"]["active_split"] == "test":
        raise ConfigError("최종 test 기간에서는 파라미터 탐색을 실행할 수 없습니다. train/validation 또는 탐색용 all을 사용하세요.")
    grid = spec["grid"]
    if not grid or any(not isinstance(v, list) or not v for v in grid.values()):
        raise ConfigError("grid에는 비어 있지 않은 값 목록이 필요합니다")
    forbidden = ("evaluation.", "data.", "run.")
    if any(k.startswith(forbidden) for k in grid):
        raise ConfigError("한 grid에서는 데이터·평가 구간·실행 범위를 고정하고 매매/선정 파라미터를 바꾸세요")
    updates_list = [dict(zip(grid, values)) for values in product(*grid.values())]
    # Validate the complete experiment before starting any expensive work.
    configs = [base.with_updates(updates) for updates in updates_list]
    data = load_data(base, data_dir)
    output = Path(base["run"]["output_directory"])
    if not output.is_absolute():
        output = base.project_root / output
    output /= "grid_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:6]
    output.mkdir(parents=True)
    rows, cache = [], {}
    for i, (cfg, updates) in enumerate(zip(configs, updates_list), 1):
        if progress:
            progress(f"실험 {i}/{len(configs)}: {updates}")
        result = run_backtest(cfg, data=data, output_dir=output/f"experiment_{i:03d}", progress=progress, formation_cache=cache)
        rows.append({"experiment": i, **updates, **result["summary"],
                     "evaluation_label": result["manifest"]["evaluation_label"],
                     "output_directory": f"experiment_{i:03d}"})
        pd.DataFrame(rows).to_csv(output / "comparison.csv", index=False)
    write_json(output / "grid.json", spec)
    return pd.DataFrame(rows), output
