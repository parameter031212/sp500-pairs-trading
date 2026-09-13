"""Shared statistical, execution, and accounting code for team experiments."""
__version__ = "0.1.0"

from .config import Config, load_config
from .data import MarketData, load_data, synthetic_data
from .runner import run_backtest

__all__ = ["Config", "load_config", "MarketData", "load_data", "synthetic_data", "run_backtest"]
