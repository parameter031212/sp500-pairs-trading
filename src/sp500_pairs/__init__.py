"""Shared statistical, execution, and accounting code for team experiments."""
__version__ = "0.1.0"

from .config import Config, load_config
from .data import MarketData, load_data, synthetic_data
from .runner import run_backtest
from .study import build_study_plan, run_selection_study

__all__ = ["Config", "load_config", "MarketData", "load_data", "synthetic_data", "run_backtest",
           "build_study_plan", "run_selection_study"]
