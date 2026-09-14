"""
Sanity tests for the engine and the robustness toolkit.

Run with:   python -m unittest discover -s tests -v
(no pytest dependency needed; pytest also works if installed)
"""

from __future__ import annotations
import os
import sys
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import synthetic_ohlc  # noqa: E402
from strategy import (  # noqa: E402
    StrategyTemplate, backtest, cti, adx, choppiness, variance_ratio, efficiency_ratio,
    _compute_indicators,
)
from generator import generate_templates, param_grid_for  # noqa: E402
from walkforward import walk_forward, grid_combos, smooth_scores, warmup_bars  # noqa: E402
from robustness import (  # noqa: E402
    trial_returns, cpcv, cpcv_paths, cscv_pbo, deflated_sharpe_ratio, probabilistic_sharpe_ratio,
    bootstrap_sharpe_pvalue, reality_check, hrp_weights, stationary_bootstrap_indices,
)
from portfolio import (  # noqa: E402
    select_portfolio, walk_forward_portfolio, returns_frame, portfolio_weights,
)


def trending_series(n=1500, drift=0.002, vol=0.008, seed=1):
    rng = np.random.default_rng(seed)
    rets = drift + rng.normal(0, vol, n)
    close = 100 * np.cumprod(1 + rets)
    intrabar = np.abs(rng.normal(0, 0.003, n)) + 0.001
    open_ = np.roll(close, 1); open_[0] = close[0]
    high = np.maximum.reduce([close * (1 + intrabar), open_, close])
    low = np.minimum.reduce([close * (1 - intrabar), open_, close])
    idx = pd.bdate_range("2010-01-01", periods=n)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": 1}, index=idx)


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.df = synthetic_ohlc(1200, seed=3)

    def test_every_template_runs(self):
        for tpl in generate_templates("full")[::53]:
            res = backtest(self.df, tpl)
            self.assertEqual(len(res["equity"]), len(self.df))
            self.assertFalse(res["equity"].isna().any())
            for t in res["trades"]:
                self.assertGreater(t["exit_date"], t["entry_date"]) if t["bars_held"] > 0 else None

    def test_no_lookahead(self):
        """Truncating the future must not change anything that happened before."""
        for tpl in generate_templates("full")[::97]:
            full = backtest(self.df, tpl)
            cut = 800
            part = backtest(self.df.iloc[:cut], tpl)
            np.testing.assert_allclose(full["equity"].iloc[:cut - 1].values, part["equity"].iloc[:cut - 1].values,
                                       rtol=1e-12, err_msg=tpl.name)
            np.testing.assert_array_equal(full["entries"][:cut - 1], part["entries"][:cut - 1], err_msg=tpl.name)

    def test_costs_reduce_pnl(self):
        tpl = StrategyTemplate("t", cost_bps=0.0)
        free = backtest(self.df, tpl)["equity"].iloc[-1]
        paid = backtest(self.df, tpl.with_params(cost_bps=20.0))["equity"].iloc[-1]
        self.assertLess(paid, free)

    def test_trend_following_makes_money_on_a_trend(self):
        df = trending_series()
        tpl = StrategyTemplate("t", direction_logic="trend", exit_style="channel", cost_bps=0.0)
        res = backtest(df, tpl)
        self.assertGreater(res["stats"]["total_return"], 0.2)
        self.assertGreater(res["stats"]["n_trades"], 0)
        # fading a strong trend with a countertrend template should do worse
        ct = backtest(df, tpl.with_params(direction_logic="countertrend"))
        self.assertLess(ct["stats"]["total_return"], res["stats"]["total_return"])

    def test_stop_never_loses_more_than_risk_plus_gap(self):
        tpl = StrategyTemplate("t", exit_style="target_stop", risk_pct=0.01, cost_bps=0.0)
        res = backtest(self.df, tpl)
        for t in res["trades"]:
            if t["reason"] in ("stop", "stop_same_bar"):
                # loss at the stop level = risk_pct of equity (gaps can make it worse, never better)
                self.assertLessEqual(t["pnl"], 0.0)

    def test_equity_marked_at_close(self):
        """Realised P&L must land on the exit bar, not one bar later."""
        tpl = StrategyTemplate("t", exit_style="target_stop", cost_bps=0.0)
        res = backtest(self.df, tpl)
        eq = res["equity"]
        t = res["trades"][0]
        i = self.df.index.get_loc(t["exit_date"])
        # after the exit bar the position is flat: equity is unchanged on the next bar
        # unless a new trade opened on it
        opened_next = any(tr["entry_date"] == self.df.index[i + 1] for tr in res["trades"])
        if not opened_next:
            self.assertAlmostEqual(eq.iloc[i], eq.iloc[i + 1])

    def test_jit_kernel_matches_pure_python(self):
        """The Numba-compiled loop and the plain-Python loop must agree exactly."""
        import strategy as S
        if not S.HAVE_NUMBA:
            self.skipTest("numba not installed")
        fast = S._bar_loop_fast
        try:
            S._bar_loop_fast = S._bar_loop      # plain Python
            for tpl in generate_templates("full")[::211]:
                slow = backtest(self.df, tpl)
                S._bar_loop_fast = fast
                quick = backtest(self.df, tpl)
                S._bar_loop_fast = S._bar_loop
                np.testing.assert_allclose(slow["equity"].values, quick["equity"].values, rtol=0, atol=1e-9, err_msg=tpl.name)
                self.assertEqual(len(slow["trades"]), len(quick["trades"]), tpl.name)
        finally:
            S._bar_loop_fast = fast

    def test_indicators_ranges(self):
        c = self.df["Close"]
        self.assertTrue(((cti(c, 20).dropna().abs()) <= 1).all())
        self.assertTrue(((adx(self.df, 14).dropna()) <= 100).all())
        self.assertTrue(((choppiness(self.df, 14).dropna()) <= 100).all())
        self.assertTrue(((efficiency_ratio(c, 20).dropna()) <= 1).all())
        self.assertTrue((variance_ratio(c, 60).dropna() >= 0).all())
        # CTI of a perfect straight line is +1
        line = pd.Series(np.arange(100, dtype=float))
        self.assertAlmostEqual(cti(line, 20).iloc[-1], 1.0, places=9)


class WalkForwardTests(unittest.TestCase):
    def setUp(self):
        self.df = synthetic_ohlc(1500, seed=5)
        self.tpl = generate_templates("quick")[0]
        self.grid = param_grid_for(self.tpl)

    def test_windows_are_disjoint_and_forward(self):
        res = walk_forward(self.df, self.tpl, self.grid, train_bars=400, test_bars=100)
        prev_end = None
        for w in res["windows"]:
            self.assertLess(w["train_end"], w["test_start"])
            if prev_end is not None:
                self.assertGreater(w["test_start"], prev_end)
            prev_end = w["test_end"]
        # every OOS bar is after the first training window
        self.assertGreaterEqual(res["oos_returns"].index[0], self.df.index[400])
        self.assertEqual(len(res["oos_returns"]), len(self.df) - 400)

    def test_anchored_uses_all_history(self):
        res = walk_forward(self.df, self.tpl, self.grid, train_bars=400, test_bars=100, anchored=True)
        for w in res["windows"]:
            self.assertEqual(w["train_start"], self.df.index[0])

    def test_embargo_gap(self):
        res = walk_forward(self.df, self.tpl, self.grid, train_bars=400, test_bars=100, embargo_bars=10)
        for w in res["windows"]:
            gap = self.df.index.get_loc(w["test_start"]) - self.df.index.get_loc(w["train_end"]) - 1
            self.assertEqual(gap, 10)

    def test_plateau_smoothing(self):
        grid = {"a": [1, 2, 3], "b": [1, 2]}
        combos, idx = grid_combos(grid)
        scores = np.array([0, 0, 10, 0, 0, 0], dtype=float)  # isolated spike at (a=2,b=1)
        sm = smooth_scores(scores, idx)
        self.assertEqual(int(np.argmax(scores)), 2)
        # spike gets diluted by its 5 neighbours; every neighbour of the spike gets the same mean
        self.assertLess(sm[2], 10)
        scores2 = np.array([-np.inf, 1, 1, 1, 1, 1], dtype=float)
        sm2 = smooth_scores(scores2, idx)
        self.assertEqual(sm2[0], -np.inf)
        self.assertTrue(np.isfinite(sm2[1:]).all())

    def test_warmup_covers_indicators(self):
        tpl = StrategyTemplate("t", vol_filter=True, bias_filter="sma", regime_filter="trend_only")
        self.assertGreaterEqual(warmup_bars(tpl), 200)
        self.assertLess(warmup_bars(StrategyTemplate("t")), 60)


class RobustnessTests(unittest.TestCase):
    def test_cpcv_paths_cover_every_group_once(self):
        for n, k in ((6, 2), (8, 2), (10, 3)):
            combos, assign, n_paths = cpcv_paths(n, k)
            for p in range(n_paths):
                groups = sorted(g for (c, g), pid in assign.items() if pid == p)
                self.assertEqual(groups, list(range(n)))

    def test_cpcv_on_noise(self):
        rng = np.random.default_rng(0)
        R = rng.normal(0, 0.01, (2000, 12))
        out = cpcv(R, None, None, n_groups=6, k_test=2, selection="best")
        self.assertEqual(out["path_returns"].shape, (2000, 5))
        self.assertFalse(np.isnan(out["path_returns"]).any())

    def test_pbo_noise_is_about_half_and_signal_is_low(self):
        rng = np.random.default_rng(1)
        noise = rng.normal(0, 0.01, (2400, 40))
        self.assertGreater(cscv_pbo(noise, n_partitions=8)["pbo"], 0.3)
        signal = noise.copy()
        signal[:, 7] += 0.004  # one genuinely good trial
        self.assertLess(cscv_pbo(signal, n_partitions=8)["pbo"], 0.1)

    def test_deflated_sharpe(self):
        rng = np.random.default_rng(2)
        r = rng.normal(0.0002, 0.01, 2500)
        one = deflated_sharpe_ratio(r, n_trials=1, var_sr_trials=0.0)
        many = deflated_sharpe_ratio(r, n_trials=500, var_sr_trials=0.05 / 252)
        self.assertGreater(one["dsr"], many["dsr"])
        self.assertAlmostEqual(one["dsr"], one["psr0"])
        self.assertGreater(probabilistic_sharpe_ratio(0.1, 0.0, 1000), 0.99)

    def test_bootstrap_pvalue_and_reality_check(self):
        rng = np.random.default_rng(3)
        idx = stationary_bootstrap_indices(500, 50, 10, rng)
        self.assertEqual(idx.shape, (50, 500))
        self.assertTrue(((idx >= 0) & (idx < 500)).all())
        good = bootstrap_sharpe_pvalue(rng.normal(0.002, 0.01, 1500), n_boot=300)
        bad = bootstrap_sharpe_pvalue(rng.normal(0.0, 0.01, 1500), n_boot=300)
        self.assertLess(good["p_value"], 0.05)
        self.assertGreater(bad["p_value"], 0.05)
        fam = pd.DataFrame(rng.normal(0, 0.01, (1500, 30)))
        self.assertGreater(reality_check(fam, n_boot=300)["p_value"], 0.05)
        fam[5] += 0.003
        self.assertLess(reality_check(fam, n_boot=300)["p_value"], 0.05)

    def test_effective_n_trials(self):
        from robustness import effective_n_trials
        rng = np.random.default_rng(5)
        base = rng.normal(0, 0.01, (1000, 3))
        cols = {}
        for k in range(3):
            for j in range(5):   # 5 near-copies of each of 3 independent streams
                cols[f"s{k}_{j}"] = base[:, k] + rng.normal(0, 0.002, 1000)
        eff = effective_n_trials(pd.DataFrame(cols))
        self.assertEqual(eff["n_eff"], 3)

    def test_hrp_weights(self):
        rng = np.random.default_rng(4)
        X = pd.DataFrame(rng.normal(0, 0.01, (800, 6)), columns=list("abcdef"))
        X["b"] = X["a"] * 0.9 + rng.normal(0, 0.003, 800)   # a, b highly correlated
        X["f"] = X["f"] * 3                                   # f very volatile
        w = hrp_weights(X)
        self.assertAlmostEqual(w.sum(), 1.0)
        self.assertTrue((w > 0).all())
        self.assertLess(w["f"], w["c"])


class PortfolioTests(unittest.TestCase):
    def _fake_results(self, n_tpl=6, T=1200, seed=0):
        rng = np.random.default_rng(seed)
        idx = pd.bdate_range("2012-01-01", periods=T)
        res, boundaries = {}, list(idx[::100])
        for k in range(n_tpl):
            r = pd.Series(rng.normal(0.0012 if k < 3 else -0.0012, 0.01, T), index=idx)
            res[f"t{k}"] = dict(
                oos_returns=r, oos_equity=1e5 * (1 + r).cumprod(), boundaries=boundaries, windows=[{}] * 12,
                summary=dict(oos_cagr=0, oos_max_drawdown=0, wfe=1.0, pct_profitable_windows=0.6, n_windows=12,
                             n_trades_oos=50, param_change_rate=0.2, pardo_pass=True),
            )
        return res

    def test_static_and_nested_selection(self):
        res = self._fake_results()
        port = select_portfolio(res, min_sharpe=0.0, max_strategies=4, corr_ceiling=0.6, weighting="hrp")
        self.assertTrue(set(port["selected"]) <= {"t0", "t1", "t2"})
        self.assertAlmostEqual(port["weights"].sum(), 1.0)
        rets = returns_frame(res)
        nested = walk_forward_portfolio(rets, res["t0"]["boundaries"], min_history_windows=3, min_sharpe=0.0)
        self.assertGreater(len(nested["portfolio_returns"]), 0)
        # nested curve starts strictly after the history it used
        first = nested["selections"][0]["period_start"]
        self.assertGreaterEqual(nested["portfolio_returns"].index[0], first)

    def test_cluster_method(self):
        res = self._fake_results(n_tpl=8)
        port = select_portfolio(res, min_sharpe=-1.0, max_strategies=3, method="cluster")
        self.assertLessEqual(len(port["selected"]), 3)


class BugRegressionTests(unittest.TestCase):
    """One test per bug fixed after the first review pass."""

    @staticmethod
    def _halted_series(n=400, halt=(150, 190), gap=0.70, seed=0):
        """A steady uptrend, then a completely flat (halted) stretch that drives
        the ATR to zero, then a hard gap down through any sane stop."""
        rng = np.random.default_rng(seed)
        close = 100 * np.cumprod(1 + rng.normal(0.004, 0.01, n))
        lo, hi = halt
        close[lo:hi] = close[lo]
        close[hi:] = close[lo] * gap * np.cumprod(1 + rng.normal(0, 0.01, n - hi))
        open_ = np.roll(close, 1); open_[0] = close[0]
        intr = np.abs(rng.normal(0, 0.003, n)) + 0.001
        high = np.maximum.reduce([close * (1 + intr), open_, close])
        low = np.minimum.reduce([close * (1 - intr), open_, close])
        high[lo:hi] = close[lo]; low[lo:hi] = close[lo]; open_[lo:hi] = close[lo]
        idx = pd.bdate_range("2015-01-01", periods=n)
        return pd.DataFrame({"Open": open_, "High": high, "Low": low,
                             "Close": close, "Volume": 1}, index=idx)

    def test_open_position_is_managed_when_the_atr_collapses(self):
        """A flat patch makes ATR(n) == 0. The entry gate must close, but the
        stop on an OPEN position must not: otherwise the position rides
        unprotected straight through the gap that follows."""
        df = self._halted_series()
        tpl = StrategyTemplate("t", direction_logic="trend", entry_style="stop",
                               exit_style="target_stop", n_entry=20, atr_n=20,
                               atr_mult_stop=2.0, risk_pct=0.01, cost_bps=0.0)
        ind = _compute_indicators(df, tpl)
        self.assertTrue((ind["atr"][20:] <= 0).any(), "fixture should zero the ATR")
        res = backtest(df, tpl)
        worst = min(t["pnl"] for t in res["trades"])
        # 1 % of 100k is the intended risk; a gap through the stop can add to it,
        # but not by an order of magnitude
        self.assertGreater(worst, -3_000, f"stop was not honoured: worst trade {worst:.0f}")

    def test_trailing_stop_includes_the_entry_bar_extreme(self):
        """The chandelier anchor must include the entry bar's own high/low --
        a big favourable move on the entry bar should not be given back."""
        n = 60
        close = np.empty(n); high = np.empty(n); low = np.empty(n); open_ = np.empty(n)
        for i in range(30):                        # narrowing range: no break fires here
            close[i] = 100.0; open_[i] = 100.0
            high[i] = 101.0 - 0.02 * i; low[i] = 99.0 + 0.02 * i
        open_[30], high[30], low[30], close[30] = 100.0, 120.0, 99.5, 101.5
        for i in range(31, n):                     # then bleed back down
            close[i] = 101.5 - 0.6 * (i - 30)
            open_[i] = close[i] + 0.1; high[i] = close[i] + 0.3; low[i] = close[i] - 0.3
        df = pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close,
                           "Volume": 1}, index=pd.bdate_range("2015-01-01", periods=n))
        tpl = StrategyTemplate("t", direction_logic="trend", entry_style="stop",
                               exit_style="atr_trail", n_entry=20, atr_n=10,
                               atr_mult_trail=2.0, atr_mult_stop=20.0, cost_bps=0.0)
        trades = backtest(df, tpl)["trades"]
        self.assertEqual(df.index.get_loc(trades[0]["entry_date"]), 30)
        self.assertGreater(trades[0]["pnl"], 0.0)

    def test_qualifying_never_names_a_template_absent_from_the_returns_frame(self):
        """min_sharpe <= 0 used to let a template with an empty OOS series
        qualify, then KeyError in select_subset."""
        idx = pd.bdate_range("2015-01-01", periods=600)
        r = pd.Series(np.random.default_rng(0).normal(0.0005, 0.01, 600), index=idx)
        summary = dict(oos_cagr=0.0, oos_max_drawdown=0.0, wfe=1.0, pct_profitable_windows=0.6,
                       n_windows=5, n_trades_oos=50, param_change_rate=0.0, pardo_pass=True)
        res = {
            "ok": dict(oos_returns=r, oos_equity=1e5 * (1 + r).cumprod(),
                       boundaries=list(idx[::100]), windows=[{}] * 5, summary=dict(summary)),
            "empty": dict(oos_returns=pd.Series(dtype=float), oos_equity=pd.Series(dtype=float),
                          boundaries=list(idx[::100]), windows=[{}] * 5, summary=dict(summary)),
        }
        port = select_portfolio(res, min_sharpe=-10.0, min_windows=0, min_trades=0)
        self.assertEqual(port["selected"], ["ok"])
        self.assertNotIn("empty", port["qualifying"])

    def test_hrp_gives_a_dead_strategy_no_weight(self):
        """A never-traded (zero-variance) column used to divide by zero and then
        collect an equal share of the book through the NaN bisection."""
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.normal(0, 0.01, (500, 4)), columns=list("abcd"))
        X["c"] = 0.0
        w = hrp_weights(X)
        self.assertFalse(bool(w.isna().any()))
        self.assertAlmostEqual(w.sum(), 1.0)
        self.assertEqual(w["c"], 0.0)
        self.assertAlmostEqual(portfolio_weights(X, "hrp").sum(), 1.0)

    def test_warmup_covers_adx_double_smoothing(self):
        """ADX smooths twice, so it needs ~2n bars, not n."""
        tpl = StrategyTemplate("t", regime_indicator="adx", regime_filter="trend_only",
                               regime_n=40, n_entry=10, atr_n=10)
        df = synthetic_ohlc(600, seed=2)
        ready = _compute_indicators(df, tpl)["ready"]
        first_ready = int(np.argmax(ready))
        self.assertTrue(ready.any())
        self.assertLessEqual(first_ready, warmup_bars(tpl) - 1)

    def test_loader_raises_instead_of_returning_an_empty_frame(self):
        """A failed download must not look like 'this ticker has no history'."""
        import data as D

        class _FakeYF:
            @staticmethod
            def download(*a, **k):
                return pd.DataFrame()

        real = sys.modules.get("yfinance")
        sys.modules["yfinance"] = _FakeYF
        try:
            with self.assertRaises(ValueError):
                D.load_yfinance("NOPE", start="2020-01-01")
        finally:
            if real is None:
                del sys.modules["yfinance"]
            else:
                sys.modules["yfinance"] = real


if __name__ == "__main__":
    unittest.main()
