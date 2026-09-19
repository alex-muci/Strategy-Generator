"""
Tests for the pieces between "a template was backtested" and "a template was
chosen" that the engine tests never reach directly:

  * robustness.evaluate_template -- the contract both entry points build on --
    and the POOLED form of the PBO (block statistics merged across templates),
    which is the only form production uses;
  * the walk-forward objective for every --metric / --selection the CLI offers;
  * the portfolio filters (--require-pardo, min WFE), the subset rules, and the
    promise that the nested selection never sees the future.

Run with:  python -m unittest discover -s tests -v
"""

from __future__ import annotations
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import synthetic_ohlc  # noqa: E402
from generator import generate_templates, param_grid_for  # noqa: E402
from walkforward import (  # noqa: E402
    walk_forward, grid_combos, score_stats, select_params, matrix_cells, matrix_row, matrix_frame,
    walk_forward_matrix,
)
from robustness import (  # noqa: E402
    evaluate_template, trial_returns, cscv_block_stats, merge_block_stats, cscv_pbo,
    expected_max_sharpe, min_backtest_length,
)
from portfolio import (  # noqa: E402
    select_portfolio, select_subset, portfolio_weights, walk_forward_portfolio, candidate_table,
    returns_frame,
)


class EvaluateTemplateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = synthetic_ohlc(900, seed=13)
        cls.tpl = generate_templates("quick", max_templates=1)[0]
        cls.grid = param_grid_for(cls.tpl)
        cls.res = evaluate_template(cls.df, cls.tpl, cls.grid, train_bars=300, test_bars=100,
                                    cpcv_groups=6, cpcv_k=2, cscv_partitions_n=8)

    def test_it_is_the_walk_forward_plus_the_stress_tests(self):
        wfa = walk_forward(self.df, self.tpl, self.grid, train_bars=300, test_bars=100)
        pd.testing.assert_series_equal(self.res["oos_returns"], wfa["oos_returns"])
        self.assertEqual(self.res["boundaries"], wfa["boundaries"])
        self.assertEqual(self.res["summary"], wfa["summary"])
        self.assertIs(self.res["template"], self.tpl)

    def test_trials_and_blocks_cover_the_whole_grid(self):
        n = len(grid_combos(self.grid)[0])
        self.assertEqual(self.res["n_trials"], n)
        b = self.res["trial_blocks"]
        self.assertEqual(b["sums"].shape, (8, n))
        self.assertEqual(b["sumsq"].shape, (8, n))
        self.assertEqual(b["counts"].sum(), len(self.df))

    def test_cpcv_summary_travels_without_the_path_matrix(self):
        cp = self.res["cpcv"]
        self.assertEqual(cp["n_paths"], 5)                      # C(6,2) * 2 / 6
        self.assertEqual(len(cp["path_sharpes"]), 5)
        self.assertNotIn("path_returns", cp, "the T x paths matrix must stay in the worker")
        self.assertAlmostEqual(cp["sharpe_mean"], float(np.mean(cp["path_sharpes"])))
        self.assertTrue(0.0 <= cp["prob_sharpe_negative"] <= 1.0)

    def test_pooled_pbo_equals_pbo_of_the_stacked_trials(self):
        """main.py never holds the trials of every template at once: it merges
        their block statistics. That has to be the same PBO."""
        combos, _ = grid_combos(self.grid)
        R1, _ = trial_returns(self.df, self.tpl, combos)
        tpl2 = generate_templates("quick")[5]
        R2, _ = trial_returns(self.df, tpl2, grid_combos(param_grid_for(tpl2))[0])
        pooled = cscv_pbo(blocks=merge_block_stats([cscv_block_stats(R1, 8), cscv_block_stats(R2, 8)]))
        stacked = cscv_pbo(np.hstack([R1, R2]), n_partitions=8)
        self.assertEqual(pooled["n_trials"], R1.shape[1] + R2.shape[1])
        self.assertEqual(pooled["n_combinations"], 70)          # C(8,4)
        self.assertAlmostEqual(pooled["pbo"], stacked["pbo"])
        np.testing.assert_allclose(pooled["logits"], stacked["logits"])
        np.testing.assert_allclose(pooled["oos_sharpe"], stacked["oos_sharpe"], atol=1e-9)

    def test_an_odd_partition_count_is_made_even(self):
        R = np.random.default_rng(0).normal(0, 0.01, (400, 3))
        self.assertEqual(cscv_block_stats(R, 7)["sums"].shape[0], 8)


class DeflationEdgeTests(unittest.TestCase):
    def test_expected_max_sharpe(self):
        self.assertEqual(expected_max_sharpe(1, 0.01), 0.0)
        self.assertEqual(expected_max_sharpe(50, 0.0), 0.0)
        self.assertEqual(expected_max_sharpe(50, float("nan")), 0.0)
        a, b = expected_max_sharpe(10, 0.01), expected_max_sharpe(1000, 0.01)
        self.assertTrue(0 < a < b, "more trials, higher bar")
        self.assertAlmostEqual(expected_max_sharpe(10, 0.04), 2 * a)

    def test_min_backtest_length(self):
        self.assertEqual(min_backtest_length(1, 1.0), np.inf)
        self.assertEqual(min_backtest_length(100, 0.0), np.inf)
        self.assertAlmostEqual(min_backtest_length(100, 1.0, ppy=252), 2 * np.log(100))
        self.assertAlmostEqual(min_backtest_length(100, 2.0, ppy=252), 2 * np.log(100) / 4)
        # years of track record do not depend on how finely the year is cut
        self.assertAlmostEqual(min_backtest_length(100, 1.0, ppy=252), min_backtest_length(100, 1.0, ppy=1764))


class ObjectiveTests(unittest.TestCase):
    STATS = dict(n_trades=12, sharpe=1.5, max_drawdown=-0.10, total_return=0.30, profit_factor=2.5)

    def test_every_cli_metric(self):
        self.assertEqual(score_stats(self.STATS, "sharpe", 5), 1.5)
        self.assertAlmostEqual(score_stats(self.STATS, "return_over_dd", 5), 3.0)
        self.assertEqual(score_stats(self.STATS, "profit_factor", 5), 2.5)

    def test_degenerate_inputs(self):
        self.assertEqual(score_stats(self.STATS, "sharpe", 13), -np.inf, "too few trades")
        no_dd = dict(self.STATS, max_drawdown=0.0)
        self.assertTrue(np.isfinite(score_stats(no_dd, "return_over_dd", 5)))
        never_lost = dict(self.STATS, profit_factor=np.inf)
        self.assertEqual(score_stats(never_lost, "profit_factor", 5), 10.0, "capped, or it always wins")
        with self.assertRaises(ValueError):
            score_stats(self.STATS, "sortino", 5)

    def test_best_takes_the_peak_and_plateau_the_neighbourhood(self):
        idx = np.arange(5).reshape(5, 1)
        scores = np.array([0.0, 9.0, 0.0, 5.0, 5.5])       # a spike at 1, a plateau at 3-4
        self.assertEqual(select_params(scores, idx, "best"), 1)
        self.assertEqual(select_params(scores, idx, "plateau"), 4)
        self.assertIsNone(select_params(np.full(5, -np.inf), idx, "best"))
        with self.assertRaises(ValueError):
            select_params(scores, idx, "median")

    def test_every_metric_and_selection_walks_forward(self):
        df = synthetic_ohlc(800, seed=19)
        tpl = generate_templates("quick", max_templates=1)[0]
        grid = param_grid_for(tpl)
        seen = set()
        for metric in ("sharpe", "return_over_dd", "profit_factor"):
            for selection in ("plateau", "best"):
                out = walk_forward(df, tpl, grid, train_bars=300, test_bars=100, metric=metric, selection=selection)
                self.assertEqual(out["summary"]["n_windows"], 5)
                self.assertEqual(len(out["oos_returns"]), 500)
                seen.add(tuple(str(w["params"]) for w in out["windows"]))
        self.assertGreater(len(seen), 1, "the objective never changed a single choice")

    def test_a_template_with_nothing_to_fit_still_walks_forward(self):
        """The online (hedge) channel learns its lookbacks; with a channel exit
        and no filters its grid is empty -- one trial, zero dimensions."""
        tpl = next(t for t in generate_templates("online") if not param_grid_for(t))
        combos, idx = grid_combos({})
        self.assertEqual((combos, idx.shape), ([{}], (1, 0)))
        out = walk_forward(synthetic_ohlc(900, seed=4), tpl, {}, train_bars=400, test_bars=100)
        self.assertEqual(out["summary"]["n_windows"], 5)
        self.assertTrue(all(w["params"] in ({}, None) for w in out["windows"]))


class MatrixTests(unittest.TestCase):
    def test_cells_are_the_ones_that_fit(self):
        self.assertEqual(matrix_cells(313), [])
        self.assertEqual(matrix_cells(314), [(250, 63)])
        self.assertEqual(len(matrix_cells(5000)), 12)
        self.assertEqual(matrix_cells(1000, (100, 990), (50,)), [(100, 50)])

    def test_rows_assemble_into_the_matrix(self):
        df = synthetic_ohlc(700, seed=8)
        tpl = generate_templates("quick", max_templates=1)[0]
        grid = param_grid_for(tpl)
        whole = walk_forward_matrix(df, tpl, grid, train_lengths=(250, 375), test_lengths=(63, 125))
        rows = [matrix_row(df, tpl, grid, tr, te) for tr, te in reversed(matrix_cells(700, (250, 375), (63, 125)))]
        pd.testing.assert_frame_equal(matrix_frame(rows), whole)
        self.assertEqual(list(whole.index), [(250, 63), (250, 125), (375, 63), (375, 125)])
        self.assertTrue(matrix_frame([]).empty)
        self.assertTrue(walk_forward_matrix(df.iloc[:200], tpl, grid).empty)


def _fake_results(specs: dict, T=800, seed=0):
    """{name: (mean, overrides of the summary)} -> walk_forward()-shaped dicts."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=T)
    out = {}
    for name, (mu, over) in specs.items():
        r = pd.Series(rng.normal(mu, 0.01, T), index=idx)
        summary = dict(oos_cagr=0.0, oos_max_drawdown=0.0, wfe=1.0, pct_profitable_windows=0.6, n_windows=8,
                       n_trades_oos=80, param_change_rate=0.2, pardo_pass=True)
        summary.update(over)
        out[name] = dict(oos_returns=r, oos_equity=1e5 * (1 + r).cumprod(), boundaries=list(idx[::100]),
                         windows=[], summary=summary)
    return out


class PortfolioFilterTests(unittest.TestCase):
    def setUp(self):
        self.res = _fake_results({
            "solid": (0.0015, {}),
            "fails_pardo": (0.0020, dict(pardo_pass=False)),
            "low_wfe": (0.0018, dict(wfe=0.2)),
            "no_wfe": (0.0017, dict(wfe=np.nan)),
            "few_trades": (0.0025, dict(n_trades_oos=3)),
            "few_windows": (0.0025, dict(n_windows=2)),
        })

    def test_the_default_filter_only_asks_for_trades_and_windows(self):
        q = select_portfolio(self.res, min_sharpe=0.0)["qualifying"]
        self.assertEqual(sorted(q), ["fails_pardo", "low_wfe", "no_wfe", "solid"])

    def test_require_pardo(self):
        port = select_portfolio(self.res, min_sharpe=0.0, require_pardo=True)
        self.assertNotIn("fails_pardo", port["qualifying"])
        self.assertIn("solid", port["qualifying"])

    def test_min_wfe_rejects_an_undefined_wfe_too(self):
        q = select_portfolio(self.res, min_sharpe=0.0, min_wfe=0.5)["qualifying"]
        self.assertEqual(sorted(q), ["fails_pardo", "solid"])

    def test_nothing_qualifies(self):
        port = select_portfolio(self.res, min_sharpe=99.0)
        self.assertEqual((port["selected"], port["qualifying"]), ([], []))
        self.assertEqual(len(port["portfolio_returns"]), 0)
        self.assertTrue(port["corr_matrix"].empty)
        self.assertEqual(len(port["candidate_stats"]), 6, "the ranking table is still complete")

    def test_candidate_table_scores_a_missing_series_zero(self):
        self.res["empty"] = dict(self.res["solid"], oos_returns=pd.Series(dtype=float))
        table = candidate_table(self.res, returns_frame(self.res))
        self.assertEqual(table.loc["empty", "oos_sharpe"], 0.0)
        self.assertGreater(table.loc["solid", "oos_sharpe"], 0.0)


class SubsetTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(2)
        idx = pd.bdate_range("2015-01-01", periods=1000)
        base = rng.normal(0.001, 0.01, 1000)
        self.rets = pd.DataFrame({
            "best": base,
            "twin": base + rng.normal(0, 0.0005, 1000),          # ~1.0 correlated with best
            "other": rng.normal(0.0008, 0.01, 1000),
            "flat": np.zeros(1000),
        }, index=idx)

    def test_trivial_cases(self):
        self.assertEqual(select_subset(self.rets, []), [])
        self.assertEqual(select_subset(self.rets, ["other"]), ["other"])

    def test_greedy_skips_a_correlated_twin_and_a_flat_stream(self):
        sel = select_subset(self.rets, ["best", "twin", "other", "flat"], "greedy", corr_ceiling=0.6)
        sharpe = (self.rets.mean() / self.rets.std()).drop("flat")
        self.assertEqual(sel[0], sharpe.idxmax())
        self.assertIn("other", sel)
        self.assertEqual(len({"best", "twin"} & set(sel)), 1)
        # undefined correlation is not "uncorrelated"
        self.assertNotIn("flat", sel)

    def test_max_strategies_caps_both_methods(self):
        for method in ("greedy", "cluster"):
            sel = select_subset(self.rets, ["best", "twin", "other"], method, max_strategies=1, corr_ceiling=1.0)
            self.assertEqual(len(sel), 1, method)

    def test_cluster_keeps_one_of_the_twins(self):
        sel = select_subset(self.rets, ["best", "twin", "other"], "cluster", max_strategies=2)
        self.assertEqual(len(sel), 2)
        self.assertIn("other", sel)

    def test_weights(self):
        w = portfolio_weights(self.rets[["best", "other"]], "equal")
        self.assertEqual(list(w), [0.5, 0.5])
        self.assertEqual(len(portfolio_weights(self.rets[[]], "equal")), 0)
        h = portfolio_weights(self.rets[["best", "twin", "other"]], "hrp")
        self.assertAlmostEqual(h.sum(), 1.0)
        self.assertTrue((h > 0).all())


class NestedSelectionTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.idx = pd.bdate_range("2015-01-01", periods=1000)
        self.rets = pd.DataFrame(rng.normal(0.0005, 0.01, (1000, 5)), columns=list("abcde"), index=self.idx)
        self.bounds = list(self.idx[::100])

    def test_a_selection_never_sees_its_own_future(self):
        """Rewrite everything from bar 700 on: every choice made before it, and
        the portfolio return of every bar before it, must not move."""
        before = walk_forward_portfolio(self.rets, self.bounds, min_sharpe=-10)
        changed = self.rets.copy()
        changed.iloc[700:] = np.random.default_rng(99).normal(-0.002, 0.03, (300, 5))
        after = walk_forward_portfolio(changed, self.bounds, min_sharpe=-10)
        cut = self.idx[700]
        early = [s for s in before["selections"] if s["period_start"] <= cut]
        self.assertEqual(early, [s for s in after["selections"] if s["period_start"] <= cut])
        self.assertGreaterEqual(len(early), 3)
        pd.testing.assert_series_equal(before["portfolio_returns"].loc[:self.idx[699]],
                                       after["portfolio_returns"].loc[:self.idx[699]])
        self.assertFalse(before["portfolio_returns"].loc[cut:].equals(after["portfolio_returns"].loc[cut:]))

    def test_it_waits_for_history_and_then_covers_every_bar_once(self):
        out = walk_forward_portfolio(self.rets, self.bounds, min_history_windows=4, min_sharpe=-10)
        pr = out["portfolio_returns"]
        self.assertEqual(pr.index[0], self.bounds[4])
        self.assertTrue(pr.index.equals(self.rets.loc[self.bounds[4]:].index))
        self.assertEqual(len(out["selections"]), len(self.bounds) - 4)

    def test_it_holds_cash_when_nothing_qualifies(self):
        out = walk_forward_portfolio(self.rets, self.bounds, min_sharpe=99.0)
        self.assertTrue((out["portfolio_returns"] == 0.0).all())
        self.assertTrue(all(s["selected"] == [] for s in out["selections"]))
        self.assertEqual(out["sharpe"], 0.0)

    def test_too_little_history_is_an_empty_portfolio(self):
        out = walk_forward_portfolio(self.rets, self.bounds[:3], min_sharpe=-10)
        self.assertEqual((len(out["portfolio_returns"]), len(out["portfolio_equity"]), out["sharpe"]), (0, 0, 0.0))


if __name__ == "__main__":
    unittest.main()
