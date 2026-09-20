from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sp500_pairs.config import ConfigError, load_config
from sp500_pairs.corporate_actions import MergerCalendar, MergerEvent
from sp500_pairs.data import DataError, MarketData
from sp500_pairs.engine import simulate_pair
from sp500_pairs.formation import Pair, select_pairs
from sp500_pairs.pretrade import historical_score
from sp500_pairs.runner import run_backtest
from sp500_pairs.schedule import make_windows
from sp500_pairs.study import SelectionCache, build_study_plan, experiment_config, run_selection_study, study_history_requirements


def fixture_data():
    dates = pd.bdate_range("2013-01-01", "2018-12-31")
    z = np.resize(np.array([0., 2.6, 2.5, 1.5, .4, .2, -.1, -.4, -.6, -.7, 0., 0.]), len(dates))
    xs = 100. * np.exp(np.cumsum(np.random.default_rng(123).normal(.0001, .001, len(dates))))
    return MarketData(pd.DataFrame({"A": xs, "B": xs * np.exp(z * .01)}, index=dates),
        pd.DataFrame([{"date": "2013-01-01", "tickers": "A,B"}]),
        pd.DataFrame([{"ticker": "A", "sector": "test"}, {"ticker": "B", "sector": "test"}]),
        {"mode": "synthetic", "warnings": ["Test fixture only"]})


def cfg(**updates):
    return load_config(ROOT / "configs/selection_study.toml").with_updates({
        "schedule.start": "2017-01-01", "schedule.end": "2017-06-30", "run.max_windows": 1,
        **updates})


def fake_fit(logs, x, y, config):
    return {"x": x, "y": y, "error": "", "pvalue": .001, "tstat": -5.,
        "alpha": 0., "beta": 1., "mean": 0., "std": .01, "phi": .93,
        "half_life": 10., "return_correlation": .8, "n_observations": len(logs.dropna()),
        "dropped_formation_rows": 0, "warnings": ""}


class ExitAndScoringTests(unittest.TestCase):
    def test_all_three_targets_are_directional_crossings_and_next_close_fills(self):
        for sign in (1, -1):
            z = sign * np.array([2.5, 2.4, .4, .2, 0., -.4, -.6, -.7, -.8])
            dates = pd.bdate_range("2017-01-02", periods=len(z))
            prices = pd.DataFrame({"A": 100., "B": 100. * np.exp(.01 * z)}, index=dates)
            pair = Pair("A__B", "A", "B", "test", 0., 1., 0., .01,
                dates[0] - pd.Timedelta(days=1), 0., 100., 100.)
            for target, fill_index in ((.5, 3), (0., 5), (-.5, 7)):
                with self.subTest(sign=sign, target=target):
                    daily, trades = simulate_pair(pair, prices, 1000., cfg(**{"strategy.exit_z": target}), "test")
                    self.assertEqual(trades.iloc[0].exit_reason, "mean_reversion")
                    self.assertEqual(trades.iloc[0].exit_date, dates[fill_index])
                    self.assertEqual(trades.iloc[0].exit_signal_date, dates[fill_index - 1])

    def test_score_fits_before_each_block_ignores_future_and_trial_strategy(self):
        data, settings = fixture_data(), cfg()
        window = make_windows(data.prices.index, settings)[0]
        row = {"pair_id": "A__B", "x": "A", "y": "B", "sector": "test", "alpha": 999., "beta": 999.}
        seen = []
        def fitting(logs, x, y, config):
            seen.append(logs.index.copy())
            return fake_fit(logs, x, y, config)
        with patch("sp500_pairs.pretrade.fit_direction", side_effect=fitting):
            score = historical_score(data, window, row, settings)
        self.assertEqual(score["pretrade_error"], "")
        self.assertEqual(len(seen), 2)
        self.assertEqual(score["pretrade_start"], "2016-01-01")
        self.assertGreaterEqual(score["pretrade_entries"], 3)
        self.assertGreater(score["pretrade_sharpe"], 0.)
        self.assertGreater(score["pretrade_trading_cost"], 0.)
        blocks = json.loads(score["pretrade_blocks"])
        for dates, block in zip(seen, blocks):
            self.assertLess(dates[-1], pd.Timestamp(block["trading_start"]))
            self.assertLess(pd.Timestamp(block["trading_end"]), window.nominal_start)
            self.assertEqual(str(dates[-1].date()), block["formation_end"])
        changed = data.prices.copy()
        changed.loc[changed.index >= window.nominal_start] = np.nan
        alternative = MarketData(changed, data.membership[["date", "tickers"]], data.sectors)
        with patch("sp500_pairs.pretrade.fit_direction", side_effect=fake_fit):
            other = historical_score(alternative, window, row, settings.with_updates({
                "strategy.exit_z": -.5, "strategy.stop_z": 7.}))
        self.assertEqual(score, other)

    def test_sharpe_is_net_of_costs_and_cost_changes_invalidate_score_cache(self):
        data, settings, cache = fixture_data(), cfg(), {}
        window = make_windows(data.prices.index, settings)[0]
        row = {"pair_id": "A__B", "x": "A", "y": "B", "sector": "test"}
        with patch("sp500_pairs.pretrade.fit_direction", side_effect=fake_fit):
            cheap = historical_score(data, window, row, settings, cache)
            expensive = historical_score(data, window, row, settings.with_updates({"costs.commission_bps": 100.}), cache)
        self.assertGreater(expensive["pretrade_trading_cost"], cheap["pretrade_trading_cost"])
        self.assertLess(expensive["pretrade_sharpe"], cheap["pretrade_sharpe"])

    def test_models_apply_correlation_and_positive_sharpe_and_minimum_entries(self):
        data, settings = fixture_data(), cfg()
        window = make_windows(data.prices.index, settings)[0]
        plan = build_study_plan()
        for corr, sharpe, entries, expected in ((.4, 1., 3, [1, 0, 1, 0]),
                (.8, -1., 3, [1, 1, 0, 0]), (.8, 1., 2, [1, 1, 0, 0])):
            def fit(logs, x, y, config):
                return fake_fit(logs, x, y, config) | {"return_correlation": corr}
            counts, feature_cache = [], {}
            with patch("sp500_pairs.formation.fit_direction", side_effect=fit) as mocked, patch(
                    "sp500_pairs.pretrade.historical_score", return_value={"pretrade_error": "",
                        "pretrade_entries": entries, "pretrade_sharpe": sharpe}):
                for model in "ABCD":
                    row = plan.loc[plan.model == model].iloc[0]
                    pairs, *_ = select_pairs(data, window, experiment_config(settings, row), feature_cache=feature_cache)
                    counts.append(len(pairs))
                self.assertEqual(mocked.call_count, 1) # A/B/C/D share the same EG tests.
            self.assertEqual(counts, expected)

    def test_missing_held_history_is_unavailable_instead_of_a_filled_price(self):
        data, settings = fixture_data(), cfg()
        window = make_windows(data.prices.index, settings)[0]
        with patch("sp500_pairs.pretrade.fit_direction", side_effect=fake_fit), patch(
                "sp500_pairs.engine.simulate_pair", side_effect=DataError("보유 중 가격 결측")):
            result = historical_score(data, window, {"pair_id": "A__B", "x": "A", "y": "B", "sector": "test"}, settings)
        self.assertIn("보유 중 가격 결측", result["pretrade_error"])
        self.assertTrue(np.isnan(result["pretrade_sharpe"]))


class OrderedStudyTests(unittest.TestCase):
    def test_history_check_reports_longest_lookback_and_common_semester_start(self):
        calendar = pd.bdate_range("2005-09-27", "2025-12-31")
        settings = cfg(**{"schedule.start": "2007-01-01", "schedule.end": "2025-12-31"})
        history = study_history_requirements(settings, calendar)
        longest = history[(history.model == "C") & (history.formation_months == 24)].iloc[0]
        self.assertEqual(longest.required_price_start, "2004-01-01")
        self.assertEqual(longest.history_months, 36)
        self.assertEqual(history.suggested_first_window.max(), "2009-01-01")
        # 2008 requires January 2005, not merely the available September 2005.
        later = study_history_requirements(settings.with_updates({"schedule.start": "2008-01-01"}), calendar)
        self.assertEqual(later[later.history_months == 36].required_price_start.iloc[0], "2005-01-01")
        self.assertFalse(later.history_available.all())
        ready = settings.with_updates({"schedule.start": "2009-01-01"})
        self.assertTrue(study_history_requirements(ready, calendar).history_available.all())
        for months in (12, 24):
            self.assertTrue(make_windows(calendar, ready.with_updates({"schedule.formation_months": months})))

    def test_history_requirements_follow_selected_models_and_active_split_grid(self):
        calendar = pd.bdate_range("2005-09-27", "2025-12-31")
        settings = cfg(**{"schedule.start": "2007-01-01", "schedule.end": "2025-12-31"})
        ab = study_history_requirements(settings, calendar, models=["A", "B"])
        self.assertEqual(ab.history_months.max(), 24)
        self.assertEqual(ab.suggested_first_window.max(), "2008-01-01")
        split = settings.with_updates({"evaluation.active_split": "validation",
            "evaluation.validation_start": "2008-10-01", "evaluation.validation_end": "2010-12-31"})
        report = study_history_requirements(split, calendar)
        self.assertEqual(set(report.first_window), {"2009-01-01"})
        self.assertTrue(report.history_available.all())

    def test_full_144_runs_are_ordered_and_resume_reuses_disk_selection(self):
        data, settings = fixture_data(), cfg()
        # A known deal closes the holding before terminal missing data in every trial.
        held = data.prices.index >= "2017-02-01"
        data.prices.loc[held, "B"] = data.prices.loc[held, "A"] * np.exp(.025)
        data.prices.loc[data.prices.index > "2017-02-10", "B"] = np.nan
        data.mergers = MergerCalendar([MergerEvent("fixture_deal", "B", pd.Timestamp("2017-02-07"),
            "before_close", pd.Timestamp("2017-02-10"), None, "https://example.com/announcement", "https://example.com/halt")])
        with tempfile.TemporaryDirectory() as tmp, patch("sp500_pairs.formation.fit_direction", side_effect=fake_fit), patch(
                "sp500_pairs.pretrade.fit_direction", side_effect=fake_fit):
            out = Path(tmp) / "study"
            first, _ = run_selection_study(settings, data=data, output_dir=out, max_experiments=1, progress=None)
            self.assertEqual(first.status.eq("complete").sum(), 1)
            with patch("sp500_pairs.formation._candidate_features", side_effect=AssertionError("recomputed formation")):
                second, _ = run_selection_study(settings, data=data, output_dir=out, max_experiments=1, progress=None)
            self.assertEqual(second.status.eq("complete").sum(), 2)
            table, _ = run_selection_study(settings, data=data, output_dir=out, progress=None)
            self.assertEqual(len(table), 144)
            self.assertEqual(table.status.eq("complete").sum(), 144)
            self.assertEqual(table.model.tolist(), list("A" * 36 + "B" * 36 + "C" * 36 + "D" * 36))
            self.assertEqual(table.name.nunique(), 144)
            self.assertTrue((table.completed_trades > 0).all())
            self.assertTrue((table.merger_announcement_exits == 1).all())
            saved_events = json.loads((out / table.output_directory.iloc[0] / "merger_events.json").read_text())
            self.assertEqual(saved_events, data.mergers.records())
            self.assertEqual(set(table.exit_z), {0., .5})
            self.assertEqual(set(table.stop_z), {3., 3.5, 4., 4.5, 5., 5.5, 6., 6.5, 7.})
            self.assertEqual(set(table.formation_months), {12, 24})
            self.assertEqual(json.loads((out / "study_manifest.json").read_text())["status"], "complete")
            self.assertEqual(len(list((out / "selection_audits").glob("*/tested_pairs.csv"))), 8)
            paths = table[table.model == "C"].iloc[[0, 9]].output_directory
            selected = [pd.read_csv(out / p / "selected_pairs.csv") for p in paths]
            assert_frame_equal(selected[0], selected[1])
            for target in (0., .5):
                row = table[(table.model == "A") & (table.formation_months == 12) & (table.exit_z == target)].iloc[0]
                trades = pd.read_csv(out / row.output_directory / "trades.csv")
                self.assertTrue(trades.exit_reason.eq("mean_reversion").any())
            with patch("sp500_pairs.study.run_backtest", side_effect=AssertionError("reran completed trial")):
                rerun, _ = run_selection_study(settings, data=data, output_dir=out, progress=None)
            self.assertEqual(rerun.status.eq("complete").sum(), 144)
            data.mergers = MergerCalendar()
            with self.assertRaisesRegex(ConfigError, "다릅니다"):
                run_selection_study(settings, data=data, output_dir=out, progress=None)

    def test_failure_is_saved_next_trial_continues_and_explicit_retry_works(self):
        data, settings = fixture_data(), cfg()
        calls = []
        def first_fails(config, **kwargs):
            calls.append(config["strategy"]["stop_z"])
            if len(calls) == 1:
                raise DataError("보유 중 가격 결측")
            return run_backtest(config, **kwargs)
        with tempfile.TemporaryDirectory() as tmp, patch("sp500_pairs.formation.fit_direction", side_effect=fake_fit):
            out = Path(tmp) / "study"
            with patch("sp500_pairs.study.run_backtest", side_effect=first_fails):
                table, _ = run_selection_study(settings, data=data, output_dir=out, models=["A"], max_experiments=2, progress=None)
            self.assertEqual(table.status.iloc[:2].tolist(), ["failed", "complete"])
            self.assertIn("보유 중 가격 결측", table.error.iloc[0])
            self.assertTrue(pd.isna(table.sharpe.iloc[0]))
            table, _ = run_selection_study(settings, data=data, output_dir=out, models=["A"], max_experiments=1, retry_failed=True, progress=None)
            self.assertEqual(table.status.iloc[:2].tolist(), ["complete", "complete"])
            self.assertEqual(table.attempts.iloc[0], 2)
            with self.assertRaisesRegex(ConfigError, "다릅니다"):
                run_selection_study(settings.with_updates({"costs.commission_bps": 6.}), data=data, output_dir=out, progress=None)

    def test_interrupt_and_missing_completed_files_are_rerunnable(self):
        data, settings = fixture_data(), cfg()
        with tempfile.TemporaryDirectory() as tmp, patch("sp500_pairs.formation.fit_direction", side_effect=fake_fit):
            out = Path(tmp) / "study"
            with patch("sp500_pairs.study.run_backtest", side_effect=KeyboardInterrupt), self.assertRaises(KeyboardInterrupt):
                run_selection_study(settings, data=data, output_dir=out, models=["A"], progress=None)
            manifest = json.loads((out / "study_manifest.json").read_text())
            self.assertEqual(manifest["experiments"][0]["status"], "interrupted")
            table, _ = run_selection_study(settings, data=data, output_dir=out, models=["A"], max_experiments=1, progress=None)
            (out / table.output_directory.iloc[0] / "summary.json").unlink()
            table, _ = run_selection_study(settings, data=data, output_dir=out, models=["A"], max_experiments=1, progress=None)
            self.assertEqual(table.status.iloc[0], "complete")
            self.assertEqual(table.attempts.iloc[0], 3)

    def test_test_split_and_insufficient_sharpe_warmup_fail_before_trials(self):
        data = fixture_data()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ConfigError, "test"):
                run_selection_study(cfg(**{"evaluation.active_split": "test", "evaluation.test_start": "2017-01-01",
                    "evaluation.test_end": "2017-06-30"}), data=data, output_dir=tmp, progress=None)
            truncated = MarketData(data.prices.loc["2015-01-01":], data.membership[["date", "tickers"]], data.sectors)
            out = Path(tmp) / "not_started"
            with patch("sp500_pairs.study.run_backtest", side_effect=AssertionError("started a trial")), self.assertRaisesRegex(DataError, "샤프 필터") as caught:
                run_selection_study(cfg(), data=truncated, output_dir=out, progress=None)
            self.assertIn("2014-01-01", str(caught.exception))
            self.assertIn("2015-01-01", str(caught.exception))
            self.assertIn("2018-01-01", str(caught.exception))
            self.assertFalse(out.exists())

    def test_json_cache_restores_pair_timestamps_frames_and_unavailable_scores(self):
        pair = Pair("A__B", "A", "B", "test", .1, 1., 0., .01, pd.Timestamp("2016-12-30"), 0., 100., 100.)
        frame = pd.DataFrame({"selected": [True, False], "score": [np.nan, np.nextafter(.05, 0.)], "reason": ["", "error"]})
        value = ([pair], frame, {"sharpe": np.nan})
        with tempfile.TemporaryDirectory() as tmp:
            cache = SelectionCache(tmp, "data")
            cache["selection"] = value
            restored = SelectionCache(tmp, "data")["selection"]
            self.assertEqual(restored[0], [pair])
            assert_frame_equal(restored[1], frame, check_exact=True)
            self.assertTrue(np.isnan(restored[2]["sharpe"]))


if __name__ == "__main__":
    unittest.main()
