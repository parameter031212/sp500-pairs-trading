"""Keep the original default model distinct from the added B model."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import pandas as pd
from pandas.testing import assert_frame_equal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sp500_pairs.config import load_config
from sp500_pairs.corporate_actions import MergerCalendar, MergerEvent
from sp500_pairs.data import MarketData, synthetic_data
from sp500_pairs.engine import simulate_pair
from sp500_pairs.formation import select_pairs
from sp500_pairs.schedule import make_windows
from test_baseline import price_path


def merger(date, timing="date_only"):
    return MergerEvent("test_deal", "B", pd.Timestamp(date), timing, None, None,
                       "https://example.com/announcement", "")


class ModelBTests(unittest.TestCase):
    def test_original_defaults_and_b_overrides(self):
        original = load_config(ROOT / "configs/baseline.toml")
        model_b = load_config(ROOT / "configs/model_b.toml")
        self.assertEqual((original["selection"]["max_per_sector"], original["selection"]["max_per_stock"]), (4, 2))
        self.assertEqual((model_b["selection"]["max_per_sector"], model_b["selection"]["max_per_stock"]), (0, 0))
        self.assertFalse(original["corporate_actions"]["enabled"])
        self.assertTrue(model_b["corporate_actions"]["enabled"])
        self.assertEqual(model_b["formation"]["multiple_testing"], "none")
        self.assertTrue(model_b["formation"]["correlation_filter"])
        self.assertEqual(model_b["formation"]["min_return_correlation"], 0.5)

    def test_b_removes_concentration_caps_without_changing_original_selection(self):
        source = synthetic_data(17)
        tickers = list(source.prices.columns[:4])
        data = MarketData(source.prices[tickers],
                          pd.DataFrame({"date": ["2014-01-01"], "tickers": [",".join(tickers)]}),
                          pd.DataFrame({"ticker": tickers, "sector": ["One sector"] * 4}))

        def passing_pair(logs, x, y, cfg):
            return {"x": x, "y": y, "error": "", "pvalue": .001, "tstat": -5.,
                    "alpha": 0., "beta": 1., "mean": 0., "std": .01, "phi": .93,
                    "half_life": 10., "return_correlation": .8,
                    "n_observations": len(logs), "dropped_formation_rows": 0, "warnings": ""}

        with patch("sp500_pairs.formation.fit_direction", side_effect=passing_pair):
            original = load_config(ROOT / "configs/baseline.toml").with_updates({"schedule.end": "2016-06-30"})
            old_pairs = select_pairs(data, make_windows(data.prices.index, original)[0], original)[0]
            model_b = load_config(ROOT / "configs/model_b.toml").with_updates({"schedule.end": "2016-06-30"})
            new_pairs = select_pairs(data, make_windows(data.prices.index, model_b)[0], model_b)[0]
        self.assertLess(len(old_pairs), 6)
        self.assertEqual(len(new_pairs), 6)
        data.mergers = MergerCalendar([MergerEvent(
            "known_deal", tickers[0], pd.Timestamp("2015-12-01"), "date_only",
            None, None, "https://example.com/announcement", "")])
        data.metadata.pop("merger_calendar_from_config", None)
        with patch("sp500_pairs.formation.fit_direction", side_effect=passing_pair):
            blocked_pairs, _, audit, _, _ = select_pairs(
                data, make_windows(data.prices.index, model_b)[0], model_b)
        self.assertEqual(audit.set_index("ticker").loc[tickers[0], "exclusion"], "merger_announced")
        self.assertTrue(all(tickers[0] not in (pair.x, pair.y) for pair in blocked_pairs))

    def test_merger_announcement_exits_b_only_after_news_becomes_known(self):
        pair, prices = price_path([2.5] * 8)
        calendar = MergerCalendar([merger("2020-01-06")])
        model_b = load_config(ROOT / "configs/model_b.toml")
        daily, trades = simulate_pair(pair, prices, 1000., model_b, "test", merger_calendar=calendar)
        self.assertEqual(trades.iloc[0].exit_reason, "merger_announcement")
        self.assertEqual(trades.iloc[0].exit_signal_date, pd.Timestamp("2020-01-07"))
        self.assertEqual(trades.iloc[0].exit_date, pd.Timestamp("2020-01-08"))
        self.assertEqual(trades.iloc[0].merger_event_ids, "test_deal")
        self.assertEqual(daily.iloc[4:].position.abs().sum(), 0)

        original = load_config(ROOT / "configs/baseline.toml")
        ignored = simulate_pair(pair, prices, 1000., original, "test", merger_calendar=calendar)
        no_event = simulate_pair(pair, prices, 1000., original, "test", merger_calendar=MergerCalendar())
        for left, right in zip(ignored, no_event):
            assert_frame_equal(left, right)


if __name__ == "__main__":
    unittest.main()
