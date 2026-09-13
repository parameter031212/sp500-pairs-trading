"""Optional detector interface. Observations contain only information available at that close."""
from dataclasses import dataclass
from typing import Protocol
import pandas as pd


@dataclass(frozen=True)
class Observation:
    date: pd.Timestamp
    window: str
    pair_id: str
    x: str
    y: str
    price_x: float
    price_y: float
    zscore: float
    alpha: float
    beta: float
    spread_mean: float
    spread_std: float
    position: int
    holding_days: int
    slot_nav: float


class Detector(Protocol):
    def on_close(self, observation: Observation) -> bool:
        """True requests next-close liquidation and blocks this pair for this window."""
        ...


class NoDetector:
    def on_close(self, observation: Observation) -> bool:
        return False
