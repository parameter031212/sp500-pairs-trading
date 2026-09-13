from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from statsmodels.tsa.stattools import adfuller, coint

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sp500_pairs.config import ConfigError, load_config
from sp500_pairs.data import DataError, MarketData, synthetic_data
from sp500_pairs.engine import entry_direction, quantities, simulate_pair
from sp500_pairs.formation import Pair, adf_pvalue, fit_direction, select_pairs
from sp500_pairs.monitoring import NoDetector
from sp500_pairs.runner import run_backtest, run_grid
from sp500_pairs.schedule import make_windows


def config(**updates):
    return load_config(ROOT / "configs" / "demo.toml").with_updates(updates)


def price_path(zs, beta=1.0):
    dates = pd.bdate_range("2020-01-02", periods=len(zs))
    xs = np.full(len(zs), 100.0)
    ys = np.exp(beta * np.log(xs) + np.asarray(zs) * .01)
    pair = Pair("A__B", "A", "B", "sector", 0., beta, 0., .01,
                pd.Timestamp("2020-01-01"), 0., 100., 100.**beta)
    return pair, pd.DataFrame({"A": xs, "B": ys}, index=dates)


class ExecutionTests(unittest.TestCase):
    def test_signal_fill_and_holding_return_use_distinct_closes(self):
        pair, prices = price_path([-2.5, -2.4, -1., .1, .2, .3, .2])
        cfg = config(**{"strategy.stop_enabled": False})
        daily, trades = simulate_pair(pair, prices, 1000, cfg, "test")
        trade = trades.iloc[0]
        self.assertEqual(trade.entry_signal_date, prices.index[0])
        self.assertEqual(trade.entry_date, prices.index[1])
        self.assertEqual(trade.exit_signal_date, prices.index[3])
        self.assertEqual(trade.exit_date, prices.index[4])
        self.assertAlmostEqual(daily.iloc[1].gross_pnl, 0)
        self.assertAlmostEqual(daily.iloc[1].nav, 999.5)
        expected = trade.quantity_y * (prices.B.iloc[4] - prices.B.iloc[1])
        self.assertAlmostEqual(trade.gross_pnl, expected)
        self.assertAlmostEqual(trade.net_pnl, expected - trade.entry_cost - trade.exit_cost)

    def test_beta_stays_on_x_leg_for_both_trade_directions(self):
        for direction in [-1, 1]:
            qx, qy = quantities(900, 2, direction, 50, 100)
            self.assertAlmostEqual(abs(qx*50), 600)
            self.assertAlmostEqual(abs(qy*100), 300)
            self.assertEqual(np.sign(qx), -direction)
            self.assertEqual(np.sign(qy), direction)

    def test_cost_is_on_total_actual_traded_leg_notionals(self):
        pair, prices = price_path([2.5, 2.4, 1, .1, -.1, -.2, -.3])
        daily, trades = simulate_pair(pair, prices, 1000, config(), "test")
        trade = trades.iloc[0]
        self.assertAlmostEqual(trade.entry_cost, .5)
        exit_gross = abs(trade.quantity_x*trade.exit_x) + abs(trade.quantity_y*trade.exit_y)
        self.assertAlmostEqual(trade.exit_cost, exit_gross * .0005)
        np.testing.assert_allclose(daily.nav-daily.nav_start,
            daily.gross_pnl+daily.cash_interest-daily.trading_cost-daily.borrow_cost, atol=1e-10)

    def test_exit_threshold_is_tunable_and_has_signed_meaning(self):
        pair, prices = price_path([-2.5, -2.4, -.4, -.3, .1, .2, .3])
        _, early = simulate_pair(pair, prices, 1000, config(**{"strategy.exit_z": .5}), "test")
        _, mean = simulate_pair(pair, prices, 1000, config(), "test")
        self.assertEqual(early.iloc[0].exit_date, prices.index[3])
        self.assertEqual(mean.iloc[0].exit_date, prices.index[5])

    def test_favorable_jump_is_not_misclassified_as_absolute_z_stop(self):
        pair, prices = price_path([2.5, 2.4, -4., -4., 0., 0.])
        _, trades = simulate_pair(pair, prices, 1000, config(), "test")
        self.assertEqual(trades.iloc[0].exit_reason, "mean_reversion")

    def test_stop_threshold_and_switch_change_actual_orders(self):
        pair, prices = price_path([2.5, 2.6, 4, 4, 4, 4, 4])
        _, standard = simulate_pair(pair, prices, 1000, config(), "test")
        _, higher = simulate_pair(pair, prices, 1000, config(**{"strategy.stop_z": 5.}), "test")
        _, disabled = simulate_pair(pair, prices, 1000, config(**{"strategy.stop_enabled": False}), "test")
        self.assertEqual(standard.iloc[0].exit_reason, "stop_z")
        self.assertEqual(standard.iloc[0].exit_date, prices.index[3])
        self.assertEqual(higher.iloc[0].exit_reason, "window_end")
        self.assertEqual(disabled.iloc[0].exit_reason, "window_end")
        self.assertEqual(entry_direction(4., config()["strategy"]), 0)
        self.assertEqual(entry_direction(4., config(**{"strategy.stop_enabled": False})["strategy"]), -1)

    def test_holding_limit_executes_on_inclusive_target_day(self):
        pair, prices = price_path([-2.5] * 10)
        _, trades = simulate_pair(pair, prices, 1000, config(**{"strategy.max_holding_days": 4}), "test")
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades.iloc[0].holding_days, 4)
        self.assertEqual(trades.iloc[0].exit_date, prices.index[4])
        self.assertEqual(trades.iloc[0].exit_reason, "time_exit")

    def test_stop_reentry_requires_reset_if_enabled(self):
        pair, prices = price_path([2.5, 2.6, 4, 4, 0, 2.5, 2.5, 1, 0, 0, 0])
        _, blocked = simulate_pair(pair, prices, 1000, config(), "test")
        _, reset = simulate_pair(pair, prices, 1000, config(**{"strategy.reentry_after_stop": "after_reset"}), "test")
        self.assertEqual(len(blocked), 1)
        self.assertEqual(len(reset), 2)
        self.assertEqual(reset.iloc[1].entry_date, prices.index[6])

    def test_mean_exit_allows_new_trade_without_same_close_reversal(self):
        pair, prices = price_path([2.5, 2.5, 0, 0, -2.5, -2.5, 0, 0, 0])
        _, trades = simulate_pair(pair, prices, 1000, config(), "test")
        self.assertEqual(len(trades), 2)
        self.assertGreater(trades.iloc[1].entry_signal_date, trades.iloc[0].exit_date)

    def test_gap_does_not_reprice_or_cancel_yesterdays_order_with_hindsight(self):
        pair, prices = price_path([-2.5, .5, 1., 1., 1.])
        _, trades = simulate_pair(pair, prices, 1000, config(), "test")
        self.assertEqual(trades.iloc[0].entry_date, prices.index[1])
        self.assertEqual(trades.iloc[0].exit_date, prices.index[2])
        self.assertAlmostEqual(trades.iloc[0].entry_y, prices.B.iloc[1])

    def test_missing_held_price_fails_instead_of_fictional_liquidation(self):
        pair, prices = price_path([-2.5, -2.5, -2., -1., 0., 0.])
        prices.loc[prices.index[2], "B"] = np.nan
        with self.assertRaisesRegex(DataError, "보유 중 가격 결측"):
            simulate_pair(pair, prices, 1000, config(), "test")

    def test_borrow_uses_elapsed_calendar_days_and_previous_short_notional(self):
        pair, prices = price_path([-2.5] * 6)
        cfg = config(**{"costs.borrow_annual_bps": 365.0})
        daily, _ = simulate_pair(pair, prices, 1000, cfg, "test")
        # Friday entry, Monday mark: three calendar days on the 500 short X notional.
        self.assertEqual((prices.index[2]-prices.index[1]).days, 3)
        self.assertAlmostEqual(daily.iloc[2].borrow_cost, 500 * .0365 * 3 / 365)

    def test_noop_detector_is_identical_and_monitoring_continues_after_exit(self):
        pair, prices = price_path([-2.5, -2.5, 0., 0., 0., 0., 0.])
        baseline, _ = simulate_pair(pair, prices, 1000, config(), "test")
        observed = []
        class RecordingDetector:
            def on_close(self, obs):
                observed.append(obs)
                return False
        extended, _ = simulate_pair(pair, prices, 1000, config(), "test", RecordingDetector())
        assert_frame_equal(baseline, extended)
        self.assertEqual(len(observed), len(prices))
        self.assertEqual(observed[-1].position, 0)

    def test_detector_exit_is_lagged_and_blocks_reentry(self):
        pair, prices = price_path([-2.5]*8)
        class Alarm:
            def on_close(self, obs):
                return obs.date == prices.index[2]
        daily, trades = simulate_pair(pair, prices, 1000, config(), "test", Alarm())
        self.assertEqual(trades.iloc[0].exit_date, prices.index[3])
        self.assertEqual(trades.iloc[0].exit_reason, "detector")
        self.assertEqual(len(trades), 1)


class FormationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = synthetic_data(10)

    def test_coint_pvalue_matches_eg_not_generic_residual_adf(self):
        cfg = config()
        logs = np.log(self.data.prices.iloc[:252, :2])
        x, y = logs.columns
        fitted = fit_direction(logs, x, y, cfg)
        expected = coint(logs[y], logs[x], trend="c", autolag="AIC")
        self.assertAlmostEqual(fitted["pvalue"], expected[1])
        residual = logs[y] - fitted["alpha"] - fitted["beta"]*logs[x]
        naive_p = adf_pvalue(residual, regression="n", autolag="AIC")
        self.assertGreater(abs(naive_p - fitted["pvalue"]), 1e-8)

    def test_future_prices_and_future_membership_do_not_change_selection(self):
        cfg = config(**{"run.max_windows": 1})
        window = make_windows(self.data.prices.index, cfg)[0]
        original = select_pairs(self.data, window, cfg)
        changed_prices = self.data.prices.copy()
        changed_prices.loc[changed_prices.index > window.formation_dates[-1]] = np.nan
        history = pd.concat([self.data.membership[["date", "tickers"]], pd.DataFrame([{"date": "2016-02-01", "tickers": "S00"}])], ignore_index=True)
        changed = MarketData(changed_prices, history, self.data.sectors)
        alternative = select_pairs(changed, window, cfg)
        self.assertEqual(original[0], alternative[0])
        assert_frame_equal(original[1], alternative[1])

    def test_direction_search_adjusts_pvalue_before_pair_acceptance(self):
        data = MarketData(self.data.prices.iloc[:, :2], self.data.membership[["date", "tickers"]], self.data.sectors)
        cfg = config(**{"run.max_windows": 1, "formation.orientation": "best_of_two"})
        window = make_windows(data.prices.index, cfg)[0]
        def fake(logs, x, y, cfg):
            return {"x": x, "y": y, "error": "", "pvalue": .04 if x < y else .2,
                    "tstat": -4., "alpha": 0., "beta": 1., "mean": 0., "std": .01,
                    "phi": .93, "half_life": 10., "return_correlation": .8,
                    "n_observations": len(logs), "dropped_formation_rows": 0, "warnings": ""}
        with patch("sp500_pairs.formation.fit_direction", side_effect=fake):
            pairs, tested, _, directions, _ = select_pairs(data, window, cfg)
        self.assertEqual(len(directions), 2)
        self.assertAlmostEqual(tested.iloc[0].pair_pvalue, .08)
        self.assertEqual(len(pairs), 0)

    def test_selection_caps_and_fixed_orientation(self):
        cfg = config(**{"selection.max_pairs": 5, "selection.max_per_stock": 1,
                        "selection.max_per_sector": 1, "formation.half_life_filter": False})
        window = make_windows(self.data.prices.index, cfg)[0]
        pairs, _, _, _, _ = select_pairs(self.data, window, cfg)
        self.assertTrue(pairs)
        self.assertLessEqual(len(pairs), 5)
        tickers = [t for pair in pairs for t in (pair.x, pair.y)]
        self.assertEqual(len(tickers), len(set(tickers)))
        self.assertEqual(len(pairs), len({pair.sector for pair in pairs}))
        self.assertTrue(all(p.x < p.y for p in pairs))


class PortfolioAndConfigTests(unittest.TestCase):
    def test_toml_inheritance_preserves_defaults(self):
        cfg = load_config(ROOT / "configs" / "example_experiment.toml")
        self.assertEqual(cfg["strategy"]["stop_z"], 4.)
        self.assertEqual(cfg["strategy"]["reentry_after_stop"], "block_window")
        self.assertEqual(cfg["selection"]["max_pairs"], 20)

    def test_invalid_parameters_fail_early(self):
        for change in [{"strategy.stop_z": 1.}, {"strategy.exit_z": 2.},
                       {"selection.max_pairs": 2.5}, {"schedule.trading_months": 2.0}, {"strategy.stop_enable": False},
                       {"strategy.stop_enabled": "false"}, {"formation.autolag": "wrong"}]:
            with self.subTest(change=change), self.assertRaises(ConfigError):
                config(**change)

    def test_tuning_test_split_is_not_allowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            base = folder / "base.toml"
            base.write_text(f'extends = "{ROOT / "configs" / "demo.toml"}"\n[evaluation]\nactive_split="test"\ntest_start="2018-01-01"\ntest_end="2018-12-31"\n', encoding="utf-8")
            grid = folder / "grid.toml"
            grid.write_text('base="base.toml"\n[grid]\n"strategy.entry_z"=[1.5,2.0]\n', encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "test"):
                run_grid(grid, progress=None)

    def test_calendar_month_windows_do_not_overlap(self):
        cfg = load_config(ROOT / "configs" / "baseline.toml")
        windows = make_windows(pd.bdate_range("2014-01-01", "2025-12-31"), cfg)
        self.assertEqual(len(windows), 18)
        self.assertEqual(windows[0].nominal_start, pd.Timestamp("2016-01-01"))
        self.assertEqual(windows[0].nominal_end, pd.Timestamp("2016-06-30"))
        for left, right in zip(windows, windows[1:]):
            self.assertLess(left.trading_dates[-1], right.trading_dates[0])
            self.assertLess(right.formation_dates[-1], right.trading_dates[0])

    def test_unallocated_slots_stay_cash_and_ledger_reconciles(self):
        data = synthetic_data(17)
        cfg = config(**{"selection.max_pairs": 20, "selection.max_per_sector": 1,
                        "run.max_windows": 1, "formation.half_life_filter": False})
        result = run_backtest(cfg, data=data, write_outputs=False, progress=None)
        window = result["windows"].iloc[0]
        self.assertGreater(window.selected_pairs, 0)
        self.assertLessEqual(window.selected_pairs, 6)
        daily, pair_daily = result["portfolio_daily"], result["pair_daily"]
        expected_cash = (20-window.selected_pairs) * cfg["portfolio"]["initial_capital"] / 20
        np.testing.assert_allclose(daily.unused_slot_cash, expected_cash)
        nav_of_selected = pair_daily.groupby("date")["nav"].sum().reindex(daily.date).to_numpy()
        np.testing.assert_allclose(daily.nav, expected_cash+nav_of_selected)
        np.testing.assert_allclose(daily.nav-daily.nav_start,
            daily.gross_pnl+daily.cash_interest-daily.trading_cost-daily.borrow_cost, atol=1e-8)
        self.assertTrue((pair_daily.groupby("pair_id").tail(1).position == 0).all())

    def test_no_selected_pairs_keeps_all_capital_and_all_days(self):
        cfg = config(**{"run.max_windows": 1, "formation.min_return_correlation": 1.0})
        result = run_backtest(cfg, data=synthetic_data(7), write_outputs=False, progress=None)
        self.assertEqual(result["summary"]["completed_trades"], 0)
        self.assertAlmostEqual(result["summary"]["cumulative_return"], 0)
        self.assertEqual(len(result["portfolio_daily"]), len(make_windows(synthetic_data(7).prices.index, cfg)[0].trading_dates))
        self.assertIsNone(result["summary"]["sharpe"])

    def test_empty_outputs_remain_readable_csv_files(self):
        cfg = config(**{"run.max_windows": 1, "formation.min_return_correlation": 1.0})
        with tempfile.TemporaryDirectory() as temp, patch("sp500_pairs.runner.save_plot"):
            out = Path(temp) / "empty_run"
            run_backtest(cfg, data=synthetic_data(7), output_dir=out, progress=None)
            for path in out.glob("*.csv"):
                frame = pd.read_csv(path)
                self.assertGreater(len(frame.columns), 0, path.name)
            self.assertIn("net_pnl", pd.read_csv(out / "trades.csv").columns)


if __name__ == "__main__":
    unittest.main()
