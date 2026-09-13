from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from .config import Config
from .data import DataError
from .formation import Pair
from .monitoring import Detector, NoDetector, Observation


@dataclass(frozen=True)
class Order:
    action: str
    direction: int
    signal_date: pd.Timestamp
    signal_z: float
    reason: str


def entry_direction(z: float, settings: dict) -> int:
    if not np.isfinite(z) or abs(z) < settings["entry_z"]:
        return 0
    if settings["stop_enabled"] and abs(z) >= settings["stop_z"]:
        return 0
    return -1 if z > 0 else 1


def quantities(capital: float, beta: float, direction: int, px: float, py: float):
    """For log(B)-alpha-beta*log(A), beta always multiplies A's notional."""
    if not all(np.isfinite(v) for v in (capital, beta, px, py)) or min(capital, beta, px, py) <= 0 or direction not in (-1, 1):
        raise ValueError("자본/β/가격은 양수이며 방향은 ±1이어야 합니다")
    return -direction * capital * beta / (1 + beta) / px, direction * capital / (1 + beta) / py


def simulate_pair(pair: Pair, prices: pd.DataFrame, capital: float, cfg: Config,
                  window_id: str, detector: Detector | None = None):
    """Mark existing shares, then fill yesterday's order at this close, then decide.

    Missing held prices and insolvency fail explicitly. No stale-price liquidation,
    implicit daily rebalancing, or current-close entry earning previous returns.
    """
    settings, costs, portfolio = cfg["strategy"], cfg["costs"], cfg["portfolio"]
    detector = detector if detector is not None else NoDetector()
    if len(prices) < 2:
        raise DataError("페어 거래 구간은 최소 2개 관측치가 필요합니다")
    dates = pd.DatetimeIndex(prices.index)
    pxs, pys = prices[pair.x].to_numpy(float), prices[pair.y].to_numpy(float)
    nav = float(capital)
    qx = qy = 0.0
    position = 0
    entry_index = None
    previous_x, previous_y, previous_date = pair.last_x, pair.last_y, pair.formation_end
    direction = entry_direction(pair.last_z, settings)
    pending = Order("enter", direction, pair.formation_end, pair.last_z, "entry") if direction else None
    blocked = waiting_reset = False
    open_trade = None
    trades, daily = [], []
    fee_rate = (costs["commission_bps"] + costs["slippage_bps"]) / 10000

    for i, (date, px, py) in enumerate(zip(dates, pxs, pys)):
        old_nav = nav
        old_position = position
        days_elapsed = (date - previous_date).days
        finite = bool(np.isfinite(px) and np.isfinite(py) and px > 0 and py > 0)
        z = pair.zscore(px, py)
        gross_pnl = borrow = interest = turnover = trading_cost = 0.0
        event = ""
        detector_signal = False
        if position:
            if not finite:
                raise DataError(f"보유 중 가격 결측: {window_id} {pair.id} {date.date()}. 가격을 채우거나 가상 청산하지 않았습니다.")
            gross_before = abs(qx * previous_x) + abs(qy * previous_y)
            gross_pnl = qx * (px - previous_x) + qy * (py - previous_y)
            short_before = max(-qx * previous_x, 0.0) + max(-qy * previous_y, 0.0)
            borrow = short_before * costs["borrow_annual_bps"] / 10000 * days_elapsed / 365
            nav += gross_pnl - borrow
            open_trade["gross_pnl"] += gross_pnl
            open_trade["borrow_cost"] += borrow
        else:
            gross_before = 0.0
            interest = nav * ((1 + portfolio["cash_annual_rate"]) ** (days_elapsed / 365) - 1)
            nav += interest
        if not np.isfinite(nav) or nav <= 0:
            raise DataError(f"계좌 지급불능/비정상 NAV: {window_id} {pair.id} {date.date()} NAV={nav}; 손실을 0으로 잘라내지 않습니다")

        final_day = i == len(dates) - 1
        execution = pending
        pending = None
        if final_day:
            if position and (execution is None or execution.action != "exit"):
                execution = Order("exit", 0, previous_date, np.nan, "window_end")
            elif not position:
                execution = None
        if execution and execution.action == "enter" and not final_day:
            if not finite:
                event = "entry_cancelled_missing_price"
            elif position == 0 and not blocked:
                entry_nav = nav
                qx, qy = quantities(nav, pair.beta, execution.direction, px, py)
                turnover = abs(qx * px) + abs(qy * py)
                trading_cost = turnover * fee_rate
                nav -= trading_cost
                position = execution.direction
                entry_index = i
                open_trade = {
                    "window": window_id, "pair_id": pair.id, "x": pair.x, "y": pair.y,
                    "sector": pair.sector, "direction": position, "alpha": pair.alpha, "beta": pair.beta,
                    "entry_signal_date": execution.signal_date, "entry_date": date,
                    "entry_signal_z": execution.signal_z, "entry_fill_z": z,
                    "entry_x": px, "entry_y": py, "quantity_x": qx, "quantity_y": qy,
                    "entry_nav": entry_nav, "entry_gross": turnover,
                    "entry_cost": trading_cost, "gross_pnl": 0.0, "borrow_cost": 0.0,
                }
                event = "entry"
        elif execution and execution.action == "exit" and position:
            turnover = abs(qx * px) + abs(qy * py)
            trading_cost = turnover * fee_rate
            nav -= trading_cost
            open_trade.update({
                "exit_signal_date": execution.signal_date, "exit_date": date,
                "exit_signal_z": execution.signal_z, "exit_fill_z": z,
                "exit_x": px, "exit_y": py, "exit_reason": execution.reason,
                "exit_cost": trading_cost, "exit_nav": nav, "holding_days": i - entry_index + 1,
                "net_pnl": nav - open_trade["entry_nav"],
                "net_return": nav / open_trade["entry_nav"] - 1,
            })
            trades.append(open_trade)
            open_trade = None
            position, qx, qy, entry_index = 0, 0.0, 0.0, None
            event = execution.reason
            if event in {"stop_z", "time_exit"}:
                key = "reentry_after_stop" if event == "stop_z" else "reentry_after_time_exit"
                blocked = settings[key] == "block_window"
                waiting_reset = settings[key] == "after_reset"
            elif event == "detector":
                blocked = True
        if nav <= 0 or not np.isfinite(nav):
            raise DataError(f"비용 차감 후 지급불능: {window_id} {pair.id} {date.date()}")

        holding_days = i - entry_index + 1 if position else 0
        observation = Observation(date, window_id, pair.id, pair.x, pair.y, px, py, z,
                                  pair.alpha, pair.beta, pair.mean, pair.std,
                                  position, holding_days, nav)
        # Continue observation after liquidation and even when re-entry is blocked.
        if finite:
            detector_signal = bool(detector.on_close(observation))
        if detector_signal:
            blocked = True

        if not final_day:
            if position:
                reason = None
                if (position == 1 and z >= -settings["exit_z"]) or (position == -1 and z <= settings["exit_z"]):
                    reason = "mean_reversion"
                elif settings["stop_enabled"] and ((position == 1 and z <= -settings["stop_z"]) or (position == -1 and z >= settings["stop_z"])):
                    reason = "stop_z"
                elif detector_signal:
                    reason = "detector"
                elif settings["max_holding_enabled"] and holding_days >= settings["max_holding_days"] - 1:
                    reason = "time_exit"
                if reason:
                    pending = Order("exit", 0, date, z, reason)
            elif not event and finite and not blocked:
                if waiting_reset:
                    if abs(z) < settings["entry_z"]:
                        waiting_reset = False
                else:
                    direction = entry_direction(z, settings)
                    # Never open at the compulsory final liquidation close.
                    if direction and i + 1 < len(dates) - 1:
                        pending = Order("enter", direction, date, z, "entry")

        end_gross = abs(qx * px) + abs(qy * py) if position else 0.0
        daily.append({
            "date": date, "window": window_id, "pair_id": pair.id, "x": pair.x, "y": pair.y,
            "sector": pair.sector, "zscore": z, "nav_start": old_nav, "nav": nav,
            "gross_pnl": gross_pnl, "trading_cost": trading_cost, "borrow_cost": borrow,
            "cash_interest": interest, "turnover": turnover, "net_return": nav / old_nav - 1,
            "position_before": old_position, "position": position, "quantity_x": qx, "quantity_y": qy,
            "notional_x": qx * px if position else 0.0, "notional_y": qy * py if position else 0.0,
            "gross_before": gross_before, "gross_notional": end_gross,
            "holding_days": holding_days, "blocked": blocked, "waiting_reset": waiting_reset,
            "event": event, "detector_signal": detector_signal,
            "next_order": pending.action if pending else "", "next_reason": pending.reason if pending else "",
        })
        previous_x, previous_y, previous_date = px, py, date
    if position or open_trade is not None:
        raise AssertionError("거래 구간 종료 청산이 누락되었습니다")
    return pd.DataFrame(daily), pd.DataFrame(trades)
