from dataclasses import dataclass
import pandas as pd

from .config import Config
from .data import DataError


@dataclass(frozen=True)
class Window:
    id: str
    formation_dates: pd.DatetimeIndex
    trading_dates: pd.DatetimeIndex
    nominal_start: pd.Timestamp
    nominal_end: pd.Timestamp


def make_windows(calendar: pd.DatetimeIndex, cfg: Config) -> list[Window]:
    calendar = pd.DatetimeIndex(calendar).sort_values()
    start, end = cfg.date_range()
    # Preserve the original cohort grid when selecting an evaluation split.
    anchor = pd.Timestamp(cfg["schedule"]["start"])
    horizon_stop = pd.Timestamp(cfg["schedule"]["end"]) + pd.Timedelta(days=1)
    maximum = cfg["run"]["max_windows"]
    windows = []
    while anchor <= end:
        stop = min(anchor + pd.DateOffset(months=cfg["schedule"]["trading_months"]), horizon_stop)
        if anchor >= start and stop - pd.Timedelta(days=1) <= end:
            first = anchor - pd.DateOffset(months=cfg["schedule"]["formation_months"])
            if calendar.min() > first + pd.Timedelta(days=7):
                raise DataError(f"{anchor.date()} 구간의 포메이션 시작 자료가 부족합니다")
            if calendar.max() < stop - pd.Timedelta(days=7):
                raise DataError(f"{anchor.date()} 구간의 거래 종료 자료가 부족합니다")
            formation = calendar[(calendar >= first) & (calendar < anchor)]
            trading = calendar[(calendar >= anchor) & (calendar < stop)]
            if len(formation) < cfg["data"]["min_formation_observations"] or len(trading) < 2:
                raise DataError(f"{anchor.date()} 구간에 필요한 관측치가 부족합니다")
            windows.append(Window(anchor.strftime("%Y-%m-%d"), formation, trading, anchor, stop - pd.Timedelta(days=1)))
            if maximum and len(windows) >= maximum:
                break
        anchor = stop
    if not windows:
        raise DataError("선택한 날짜 범위에 거래 구간이 없습니다. 평가 분할을 거래 구간 경계에 맞추세요.")
    return windows
