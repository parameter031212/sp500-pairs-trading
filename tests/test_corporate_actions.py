from dataclasses import replace
from pathlib import Path
import csv
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from test_baseline import config, price_path
from test_study import cfg, fake_fit, fixture_data
from sp500_pairs.corporate_actions import MergerCalendar, MergerEvent, configured_calendar, read_merger_calendar
from sp500_pairs.data import DataError
from sp500_pairs.engine import simulate_pair
from sp500_pairs.formation import select_pairs
from sp500_pairs.pretrade import historical_score
from sp500_pairs.schedule import make_windows


def event(date, timing="before_close", last=None, release=None):
    return MergerEvent("test_deal", "B", pd.Timestamp(date), timing,
        pd.Timestamp(last) if last else None, pd.Timestamp(release) if release else None,
        "https://example.com/announcement", "https://example.com/halt" if last else "",
        "https://example.com/cancel" if release else "")


class MergerExecutionTests(unittest.TestCase):
    def test_disabled_policy_ignores_even_an_explicit_calendar(self):
        pair, prices = price_path([2.5] * 7)
        settings = config(**{"corporate_actions.enabled": False})
        original = simulate_pair(pair, prices, 1000., settings, "test")
        supplied = simulate_pair(pair, prices, 1000., settings, "test",
            merger_calendar=MergerCalendar([event("2019-12-01")]))
        for left, right in zip(original, supplied):
            assert_frame_equal(left, right)

    def test_held_pair_exits_next_close_with_both_legs_costs_and_pnl(self):
        for sign in (-1, 1):
            pair, prices = price_path(sign * np.array([2.5, 2.4, 2.6, 2.9, 2.5, 2.5]))
            settings = config(**{"corporate_actions.enabled": True, "costs.borrow_annual_bps": 365.})
            calendar = MergerCalendar([event(prices.index[2])])
            daily, trades = simulate_pair(pair, prices, 1000., settings, "test", merger_calendar=calendar)
            trade = trades.iloc[0]
            self.assertEqual(len(trades), 1)
            self.assertEqual(trade.exit_reason, "merger_announcement")
            self.assertEqual(trade.merger_event_ids, "test_deal")
            self.assertEqual(trade.exit_signal_date, prices.index[2])
            self.assertEqual(trade.exit_date, prices.index[3])
            self.assertTrue(daily.iloc[3:][["quantity_x", "quantity_y", "position"]].eq(0).all().all())
            self.assertAlmostEqual(trade.gross_pnl, trade.quantity_y * (trade.exit_y - trade.entry_y))
            self.assertGreater(trade.borrow_cost, 0)
            self.assertGreater(trade.exit_cost, 0)
            self.assertAlmostEqual(trade.net_pnl, trade.gross_pnl - trade.borrow_cost - trade.entry_cost - trade.exit_cost)
            np.testing.assert_allclose(daily.nav - daily.nav_start,
                daily.gross_pnl + daily.cash_interest - daily.trading_cost - daily.borrow_cost, atol=1e-10)
            baseline, _ = simulate_pair(pair, prices, 1000., settings, "test", merger_calendar=MergerCalendar())
            assert_frame_equal(daily.iloc[:2], baseline.iloc[:2])

    def test_pending_entry_is_cancelled_without_any_trading_cost(self):
        pair, prices = price_path([2.5] * 7)
        daily, trades = simulate_pair(pair, prices, 1000., config(), "test",
            merger_calendar=MergerCalendar([event(prices.index[1])]))
        self.assertTrue(trades.empty)
        self.assertEqual(daily.event.iloc[1], "entry_cancelled_merger")
        self.assertEqual(daily.trading_cost.sum(), 0)
        self.assertTrue(daily.position.eq(0).all())

    def test_unknown_or_after_close_timing_is_recognized_next_session(self):
        pair, prices = price_path([2.5] * 8)
        # Friday news: Monday recognizes, Tuesday executes. No weekend synthetic fill.
        for timing in ("date_only", "after_close"):
            daily, trades = simulate_pair(pair, prices, 1000., config(), "test",
                merger_calendar=MergerCalendar([event("2020-01-03", timing)]))
            self.assertFalse(daily.merger_blocked.iloc[1])
            self.assertEqual(trades.iloc[0].entry_date, pd.Timestamp("2020-01-03"))
            self.assertEqual(trades.iloc[0].exit_signal_date, pd.Timestamp("2020-01-06"))
            self.assertEqual(trades.iloc[0].exit_date, pd.Timestamp("2020-01-07"))

    def test_known_before_formation_cancels_initial_signal_and_future_event_is_inert(self):
        pair, prices = price_path([2.5] * 7)
        pair = replace(pair, last_z=2.5)
        _, trades = simulate_pair(pair, prices, 1000., config(), "test",
            merger_calendar=MergerCalendar([event(pair.formation_end)]))
        self.assertTrue(trades.empty)
        baseline = simulate_pair(pair, prices, 1000., config(), "test", merger_calendar=MergerCalendar())
        future = simulate_pair(pair, prices, 1000., config(), "test",
            merger_calendar=MergerCalendar([event("2030-01-01", last="2030-05-01")]))
        for original, changed in zip(baseline, future):
            assert_frame_equal(original, changed)

    def test_stale_csv_price_after_verified_halt_is_not_executable(self):
        pair, prices = price_path([2.5] * 7)
        calendar = MergerCalendar([event("2020-01-06", last="2020-01-06")])
        with self.assertRaisesRegex(DataError, "거래정지 이후 보유"):
            simulate_pair(pair, prices, 1000., config(), "test", merger_calendar=calendar)

    def test_missing_held_price_is_not_fabricated_even_with_merger_signal(self):
        pair, prices = price_path([2.5] * 7)
        prices.loc[prices.index[3], "B"] = np.nan
        with self.assertRaisesRegex(DataError, "보유 중 가격 결측"):
            simulate_pair(pair, prices, 1000., config(), "test",
                merger_calendar=MergerCalendar([event(prices.index[2])]))

    def test_cancellation_before_exit_fill_preserves_reason_and_window_block(self):
        pair, prices = price_path([2.5] * 7)
        calendar = MergerCalendar([event("2020-01-06", release="2020-01-07")])
        daily, trades = simulate_pair(pair, prices, 1000., config(), "test", merger_calendar=calendar)
        self.assertEqual(trades.iloc[0].exit_date, pd.Timestamp("2020-01-07"))
        self.assertEqual(trades.iloc[0].merger_event_ids, "test_deal")
        self.assertTrue(daily.iloc[2:].merger_blocked.all())
        self.assertEqual(calendar.active("2020-01-07"), ())


class MergerSelectionTests(unittest.TestCase):
    def test_formation_uses_only_known_events_and_invalidates_cached_selection(self):
        data, settings, cache = fixture_data(), cfg(), {}
        window = make_windows(data.prices.index, settings)[0]
        data.mergers = MergerCalendar()
        with patch("sp500_pairs.formation.fit_direction", side_effect=fake_fit):
            original = select_pairs(data, window, settings, feature_cache=cache)
            self.assertEqual(len(original[0]), 1)
            data.mergers = MergerCalendar([event("2017-02-01")])
            future = select_pairs(data, window, settings, feature_cache=cache)
            self.assertEqual(original[0], future[0])
            assert_frame_equal(original[1], future[1])
            data.mergers = MergerCalendar([event("2016-06-01")])
            known = select_pairs(data, window, settings, feature_cache=cache)
            self.assertEqual(known[0], [])
            self.assertEqual(known[2].set_index("ticker").loc["B", "exclusion"], "merger_announced")
            data.mergers = MergerCalendar([event("2016-06-01", release="2016-12-01")])
            released = select_pairs(data, window, settings, feature_cache=cache)
            self.assertEqual(original[0], released[0])

    def test_pretrade_obeys_historical_announcements_and_refreshes_score_cache(self):
        data, settings, cache = fixture_data(), cfg(), {}
        window = make_windows(data.prices.index, settings)[0]
        row = {"pair_id": "A__B", "x": "A", "y": "B", "sector": "test"}
        data.mergers = MergerCalendar()
        with patch("sp500_pairs.pretrade.fit_direction", side_effect=fake_fit):
            original = historical_score(data, window, row, settings, cache)
            data.mergers = MergerCalendar([event("2017-02-01")])
            future = historical_score(data, window, row, settings, cache)
            self.assertEqual(original, future)
            data.mergers = MergerCalendar([event("2016-06-01")])
            known = historical_score(data, window, row, settings, cache)
            self.assertEqual(known["pretrade_error"], "")
            self.assertLess(known["pretrade_entries"], original["pretrade_entries"])
            self.assertNotEqual(known["pretrade_sharpe"], original["pretrade_sharpe"])

    def test_registry_reloads_when_file_changes_and_validates_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.csv"
            fields = ["event_id", "ticker", "announcement_date", "timing", "last_trading_date", "release_date", "source", "trading_source", "release_source"]
            row = dict(event_id="deal", ticker="B", announcement_date="2020-01-03", timing="date_only",
                last_trading_date="", release_date="", source="https://example.com/release", trading_source="", release_source="")
            def write():
                with path.open("w") as f:
                    writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerow(row)
            write()
            settings = config(**{"corporate_actions.enabled": True, "corporate_actions.events_file": str(path)})
            old = configured_calendar(settings)
            row["event_id"] = "updated_deal"
            write()
            self.assertNotEqual(old.fingerprint, configured_calendar(settings).fingerprint)
            row["release_date"] = "2020-05-01"
            write()
            with self.assertRaisesRegex(DataError, "공식 출처"):
                read_merger_calendar(path)
            row["release_source"] = "https://example.com/cancel"
            write()
            self.assertFalse(read_merger_calendar(path).active("2020-05-01"))


if __name__ == "__main__":
    unittest.main()
