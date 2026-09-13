from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations
import inspect
import warnings

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import adfuller, coint

from .config import Config
from .data import MarketData, normalize_ticker
from .schedule import Window


@dataclass(frozen=True)
class Pair:
    id: str
    x: str
    y: str
    sector: str
    alpha: float
    beta: float
    mean: float
    std: float
    formation_end: pd.Timestamp
    last_z: float
    last_x: float
    last_y: float

    def zscore(self, px: float, py: float) -> float:
        if not np.isfinite(px) or not np.isfinite(py) or px <= 0 or py <= 0:
            return float("nan")
        return (np.log(py) - self.alpha - self.beta * np.log(px) - self.mean) / self.std


def _lag_kwargs(settings):
    return {"autolag": None if settings["autolag"] == "fixed" else settings["autolag"],
            "maxlag": None if settings["maxlag"] == -1 else settings["maxlag"]}


def adf_pvalue(values, regression="c", **kwargs):
    # Explicitly keep tuple returns on 0.15 while supporting the 0.14 signature.
    if "result_object" in inspect.signature(adfuller).parameters:
        kwargs["result_object"] = False
    return float(adfuller(values, regression=regression, **kwargs)[1])


def fit_direction(logs: pd.DataFrame, x: str, y: str, cfg: Config) -> dict:
    f = cfg["formation"]
    frame = logs[[x, y]].dropna()
    result = {"x": x, "y": y, "error": "", "pvalue": 1.0, "tstat": float("nan")}
    try:
        xv, yv = frame[x].to_numpy(), frame[y].to_numpy()
        design = np.column_stack([np.ones(len(frame)), xv])
        if np.linalg.matrix_rank(design) < 2 or np.std(yv) < 1e-12:
            raise ValueError("constant_series")
        alpha, beta = np.linalg.lstsq(design, yv, rcond=None)[0]
        spread = yv - alpha - beta * xv
        sd = float(spread.std(ddof=1))
        if sd <= 1e-10:
            raise ValueError("near_exact_collinearity")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            stat, pval, _ = coint(yv, xv, trend="c", method="aeg", return_results=False, **_lag_kwargs(f))
        if not np.isfinite(stat) or not np.isfinite(pval):
            raise ValueError("nonfinite_cointegration_result")
        phi = float(np.linalg.lstsq(np.column_stack([np.ones(len(spread)-1), spread[:-1]]), spread[1:], rcond=None)[0][1])
        half_life = float(-np.log(2) / np.log(phi)) if 0 < phi < 1 else float("nan")
        # Correlation only uses adjacent original market observations, not compressed gaps.
        corr = float(logs[[x, y]].diff().corr().iloc[0, 1])
        result.update(alpha=float(alpha), beta=float(beta), mean=float(spread.mean()), std=sd,
                      phi=phi, half_life=half_life, return_correlation=corr,
                      tstat=float(stat), pvalue=float(pval), n_observations=len(frame),
                      dropped_formation_rows=len(logs)-len(frame),
                      warnings="; ".join(str(w.message) for w in caught))
    except (ValueError, np.linalg.LinAlgError, ZeroDivisionError) as exc:
        result["error"] = str(exc)
    return result


def select_pairs(data: MarketData, window: Window, cfg: Config, progress=None):
    f, d, selection = cfg["formation"], cfg["data"], cfg["selection"]
    prices = data.prices.loc[window.formation_dates]
    universe, info = data.universe(window.formation_dates[-1])
    unknown = {s.casefold() for s in d["unknown_sectors"]}
    excluded = {normalize_ticker(t) for t in d["excluded_tickers"]}
    stock_rows, usable = [], []
    i1_map = {}
    for item in universe.itertuples():
        record = {"window": window.id, "ticker": item.ticker, "sector": item.sector,
                  "coverage": 0.0, "exclusion": "", "level_adf_p": np.nan, "difference_adf_p": np.nan, "i1_pass": False}
        if item.ticker in excluded:
            record["exclusion"] = "explicit_exclusion"
        elif item.ticker not in prices:
            record["exclusion"] = "no_price_column"
        elif item.sector.casefold() in unknown:
            record["exclusion"] = "unknown_sector"
        else:
            values = prices[item.ticker]
            record["coverage"] = float(values.notna().mean())
            if record["coverage"] < d["min_formation_coverage"] or values.count() < d["min_formation_observations"]:
                record["exclusion"] = "formation_coverage"
            else:
                logs = np.log(values.dropna().to_numpy())
                try:
                    record["level_adf_p"] = adf_pvalue(logs, **_lag_kwargs(f))
                    record["difference_adf_p"] = adf_pvalue(np.diff(logs), **_lag_kwargs(f))
                    record["i1_pass"] = record["level_adf_p"] >= f["i1_alpha"] and record["difference_adf_p"] < f["i1_alpha"]
                except (ValueError, np.linalg.LinAlgError):
                    record["i1_error"] = True
                # Diagnose first; apply optional I(1) screen after the entire pair-testing family is defined.
                usable.append((item.ticker, item.sector))
        i1_map[item.ticker] = record["i1_pass"]
        stock_rows.append(record)
    logs = np.log(prices[[t for t, _ in usable]])
    candidates = []
    groups = pd.DataFrame(usable, columns=["ticker", "sector"])
    for sector, group in groups.groupby("sector", sort=True):
        candidates.extend((x, y, sector) for x, y in combinations(sorted(group["ticker"]), 2))
    rows, directions = [], []
    for index, (a, b, sector) in enumerate(candidates):
        pair_id = f"{a}__{b}"
        orientations = [(a, b)] if f["orientation"] == "fixed" else [(a, b), (b, a)]
        fitted = []
        for x, y in orientations:
            result = fit_direction(logs, x, y, cfg)
            directions.append({"window": window.id, "pair_id": pair_id, **result})
            fitted.append(result)
        chosen = min(fitted, key=lambda row: (row["pvalue"], row["x"], row["y"]))
        row = {"window": window.id, "pair_id": pair_id, "sector": sector, **chosen,
               "direction_tests": len(orientations),
               "pair_pvalue": min(1.0, len(orientations) * chosen["pvalue"]),
               "i1_pass": bool(i1_map[a] and i1_map[b]), "selected": False, "exclusion": ""}
        rows.append(row)
        if progress and (index + 1) % 500 == 0:
            progress(f"{window.id}: {index+1:,}/{len(candidates):,} 후보 검정")
    tested = pd.DataFrame(rows)
    if tested.empty:
        return [], tested, pd.DataFrame(stock_rows), pd.DataFrame(directions), {**info, "tested_pairs": 0, "selected_pairs": 0}
    pvalues = tested["pair_pvalue"].to_numpy()
    # Every within-sector candidate enters this family, before correlation/HR/half-life filtering.
    tested["decision_pvalue"] = pvalues if f["multiple_testing"] == "none" else multipletests(pvalues, method=f["multiple_testing"])[1]
    for i, row in tested.iterrows():
        reasons = []
        if row["error"]:
            reasons.append("fit_error")
        else:
            if row["n_observations"] < d["min_formation_observations"] or row["n_observations"] / len(logs) < d["min_formation_coverage"]:
                reasons.append("pair_coverage")
            if row["decision_pvalue"] >= f["cointegration_alpha"]:
                reasons.append("cointegration")
            if f["correlation_filter"] and (not np.isfinite(row["return_correlation"]) or row["return_correlation"] <= f["min_return_correlation"]):
                reasons.append("correlation")
            if not f["beta_min"] <= row["beta"] <= f["beta_max"]:
                reasons.append("hedge_ratio")
            if f["half_life_filter"] and (not np.isfinite(row["half_life"]) or not f["half_life_min"] <= row["half_life"] <= f["half_life_max"]):
                reasons.append("half_life")
            if f["require_i1"] and not row["i1_pass"]:
                reasons.append("i1_screen")
            if not np.isfinite(prices[row["x"]].iloc[-1]) or not np.isfinite(prices[row["y"]].iloc[-1]):
                reasons.append("missing_formation_end_price")
        tested.at[i, "exclusion"] = ";".join(reasons)
    eligible = tested.loc[tested["exclusion"] == ""].sort_values(["tstat", "pair_id", "x"], kind="stable")
    stock_count, sector_count, pairs = Counter(), Counter(), []
    for i, row in eligible.iterrows():
        if len(pairs) >= selection["max_pairs"]:
            tested.at[i, "exclusion"] = "portfolio_full"
            continue
        if sector_count[row["sector"]] >= selection["max_per_sector"]:
            tested.at[i, "exclusion"] = "sector_cap"
            continue
        if max(stock_count[row["x"]], stock_count[row["y"]]) >= selection["max_per_stock"]:
            tested.at[i, "exclusion"] = "stock_cap"
            continue
        px, py = float(prices[row["x"]].iloc[-1]), float(prices[row["y"]].iloc[-1])
        last_z = (np.log(py) - row["alpha"] - row["beta"] * np.log(px) - row["mean"]) / row["std"]
        pairs.append(Pair(row["pair_id"], row["x"], row["y"], row["sector"], row["alpha"], row["beta"], row["mean"], row["std"], window.formation_dates[-1], last_z, px, py))
        stock_count.update([row["x"], row["y"]]); sector_count.update([row["sector"]])
        tested.at[i, "selected"] = True
        tested.at[i, "selection_rank"] = len(pairs)
    return pairs, tested, pd.DataFrame(stock_rows), pd.DataFrame(directions), {
        **info, "eligible_stocks": len(usable), "tested_pairs": len(tested),
        "selected_pairs": len(pairs), "empty_slots": selection["max_pairs"] - len(pairs),
    }
