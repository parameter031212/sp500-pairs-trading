"""Causal, rolling coefficient estimates for the pair-selection Sharpe screen."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .data import DataError
from .formation import Pair, fit_direction
from .corporate_actions import calendar_for


def reference_config(cfg):
    h = cfg["pretrade"]
    return cfg.with_updates({
        "strategy.entry_z": h["entry_z"], "strategy.exit_z": h["exit_z"],
        "strategy.stop_z": h["stop_z"], "strategy.stop_enabled": True,
    })


def historical_score(data, window, row, cfg, cache=None):
    """Trade the preceding 12 months in two six-month blocks by default.

    Each block fits coefficients using the preceding F months, then freezes them.
    The current formation fit is never used to trade its own estimation sample.
    Scoring uses a fixed reference strategy, not the trial's exit/stop thresholds.
    """
    from .engine import simulate_pair

    ref = reference_config(cfg)
    mergers = calendar_for(data, cfg)
    h, f, d = cfg["pretrade"], cfg["formation"], cfg["data"]
    identity = getattr(cache, "data_identity", id(data))
    settings = {k: v for k, v in f.items() if k not in {"correlation_filter", "min_return_correlation"}}
    key = "pretrade:" + json.dumps({"data": identity, "pair": row["pair_id"],
        "anchor": str(window.nominal_start), "formation_months": cfg["schedule"]["formation_months"],
        "formation": settings, "pretrade": {k: v for k, v in h.items() if k != "enabled"},
        "strategy": ref["strategy"], "costs": ref["costs"], "evaluation": ref["evaluation"],
        "cash_rate": ref["portfolio"]["cash_annual_rate"], "data_settings": d,
        "mergers": mergers.fingerprint}, sort_keys=True)
    if cache is not None and key in cache:
        return cache[key]
    anchor = window.nominal_start - pd.DateOffset(months=h["months"])
    result = {"pretrade_start": str(anchor.date()), "pretrade_end": str(window.formation_dates[-1].date()),
        "pretrade_sharpe": np.nan, "pretrade_entries": 0, "pretrade_trading_cost": 0.0,
        "pretrade_borrow_cost": 0.0, "pretrade_error": "", "pretrade_blocks": "[]"}
    returns, blocks = [], []
    capital = 1.0
    tickers = sorted([row["x"], row["y"]])
    try:
        for _ in range(h["months"] // h["block_months"]):
            stop = anchor + pd.DateOffset(months=h["block_months"])
            first = anchor - pd.DateOffset(months=cfg["schedule"]["formation_months"])
            calendar = data.prices.index
            if calendar.min() > first + pd.Timedelta(days=7):
                raise DataError(f"사전 모의거래의 {first.date()} 추정 시작 자료 부족")
            formation = data.prices.loc[(calendar >= first) & (calendar < anchor), tickers]
            trading = data.prices.loc[(calendar >= anchor) & (calendar < stop), tickers]
            if len(formation) < d["min_formation_observations"] or len(trading) < 2:
                raise DataError("사전 모의거래 관측치 부족")
            joint = formation.dropna()
            if len(joint) < d["min_formation_observations"] or len(joint) / len(formation) < d["min_formation_coverage"]:
                raise DataError("사전 추정 구간의 페어 가격 커버리지 부족")
            logs = np.log(formation)
            orientations = [tuple(tickers)]
            if f["orientation"] == "best_of_two":
                orientations.append(tuple(reversed(tickers)))
            fits = [fit_direction(logs, x, y, ref) for x, y in orientations]
            fitted = min(fits, key=lambda v: (v["pvalue"], v["x"], v["y"]))
            if fitted["error"]:
                raise DataError("사전 회귀 추정 실패: " + fitted["error"])
            if not f["beta_min"] <= fitted["beta"] <= f["beta_max"]:
                raise DataError("사전 회귀 헤지 비율 범위 이탈")
            x, y = fitted["x"], fitted["y"]
            px, py = float(formation[x].iloc[-1]), float(formation[y].iloc[-1])
            if not np.isfinite(px) or not np.isfinite(py):
                raise DataError("사전 추정 구간 마지막 가격 결측")
            z = (np.log(py) - fitted["alpha"] - fitted["beta"] * np.log(px) - fitted["mean"]) / fitted["std"]
            pair = Pair(row["pair_id"], x, y, row["sector"], fitted["alpha"], fitted["beta"],
                fitted["mean"], fitted["std"], formation.index[-1], z, px, py)
            daily, trades = simulate_pair(pair, trading, capital, ref, "pretrade_" + str(anchor.date()), merger_calendar=mergers)
            entries = int(daily["event"].eq("entry").sum())
            result["pretrade_entries"] += entries
            result["pretrade_trading_cost"] += float(daily["trading_cost"].sum())
            result["pretrade_borrow_cost"] += float(daily["borrow_cost"].sum())
            returns.extend(daily["net_return"].tolist())
            capital = float(daily["nav"].iloc[-1])
            blocks.append({"formation_start": str(formation.index[0].date()),
                "formation_end": str(formation.index[-1].date()), "trading_start": str(trading.index[0].date()),
                "trading_end": str(trading.index[-1].date()), "x": x, "y": y,
                "alpha": pair.alpha, "beta": pair.beta, "entries": entries, "completed_trades": len(trades)})
            anchor = stop
        values = np.asarray(returns)
        sd = values.std(ddof=1)
        ann = cfg["evaluation"]["annualization"]
        rf = (1 + cfg["evaluation"]["risk_free_annual_rate"]) ** (1 / ann) - 1
        result["pretrade_sharpe"] = float((values.mean() - rf) / sd * np.sqrt(ann)) if sd > 0 else np.nan
    except (DataError, ValueError, np.linalg.LinAlgError) as exc:
        result["pretrade_error"] = str(exc)
    result["pretrade_blocks"] = json.dumps(blocks, ensure_ascii=False)
    if cache is not None:
        cache[key] = result
    return result
