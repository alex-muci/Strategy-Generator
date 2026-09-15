"""
Walk-forward correctness: what an out-of-sample window is allowed to see,
how it is warmed up, how its efficiency is measured, and that the live refit
makes the same choice the research loop made.

Each test here pins a bug that was found in review:

* a window's warm-up buffer was allowed to TRADE, so positions chosen by
  in-sample-fitted parameters were opened before the window and their P&L
  counted as out-of-sample;
* the buffer was too short for exponentially smoothed indicators (Keltner
  EMA, Wilder ADX), so the window's indicators were not the ones a trader
  with full history would have seen;
* Pardo's WFE compared the COMPOUNDED total OOS return, annualized, with
  per-window IS returns, so it grew with the length of the history alone;
* a trend template's channel exit and hard stop were checked in a fixed
  order, so a bar through both filled at the further level;
* the indicator cache was keyed on closes only, so a frame with the same
  closes and different wicks came back with the wrong ATR and channels.

Run with:   python -m unittest discover -s tests -v
"""

from __future__ import annotations
import os
import sys
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import synthetic_ohlc  # noqa: E402
from strategy import StrategyTemplate, backtest, _compute_indicators, atr  # noqa: E402
from generator import generate_templates, param_grid_for  # noqa: E402
from walkforward import (  # noqa: E402
    walk_forward, walk_forward_matrix, window_backtest, warmup_bars, summarize_walk_forward,
    grid_combos, _ema_settle_bars,
)
from live import refit_params  # noqa: E402


def _bars(open_, high, low, close, start="2016-01-01"):
    idx = pd.bdate_range(start, periods=len(close))
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": 1}, index=idx)


class WarmupGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = synthetic_ohlc(1500, seed=23)

    def test_first_trade_bar_gates_entries_and_equity(self):
        tpl = StrategyTemplate("t")
        k = 400
        res = backtest(self.df, tpl, first_trade_bar=k)
        self.assertEqual(int(res["entries"][:k].sum()), 0)
        np.testing.assert_array_equal(res["equity"].to_numpy()[:k], 100_000.0)
        self.assertGreater(int(res["entries"][k:].sum()), 0)
        self.assertTrue(all(t["entry_date"] >= self.df.index[k] for t in res["trades"]))
        # bar k itself may trade: the gate is on decisions, not on indicators
        cold = backtest(self.df, tpl)
        self.assertGreater(int(cold["entries"][:k].sum()), 0)

    def test_warm_window_matches_a_full_history_run_for_every_template(self):
        """The whole reason for the buffer: a window that only saw
        `warmup_bars` of history before it must behave exactly like one that
        saw everything. Exponential indicators make this non-trivial."""
        k, n = 700, len(self.df)
        for tpl in generate_templates("full")[::37]:
            for params in ({}, {"n_entry": 60, "n_exit": 20, "regime_n": 30}):
                t = tpl.with_params(**params)
                full = backtest(self.df, t, first_trade_bar=k)
                win = window_backtest(self.df, t, k, n)
                self.assertLess(warmup_bars(t), k, t.name)
                # an EMA never forgets its seed completely: the buffer leaves
                # ~1e-4 of it, about a dollar on 100k here, and no trade differs
                np.testing.assert_allclose(win["equity"].to_numpy(), full["equity"].to_numpy()[k:],
                                           rtol=1e-5, err_msg=f"{t.name} {params}")
                self.assertEqual(len(win["trades"]), len(full["trades"]), t.name)
                self.assertEqual(win["window_start"], self.df.index[k])
                self.assertEqual(win["stats"]["n_bars"], n - k)
        # and the test has teeth: after the old, too-short buffer (2n + 15 bars)
        # the ADX and a Keltner channel still carry their seed
        for t, key in ((StrategyTemplate("t", regime_filter="trend_only", regime_indicator="adx", regime_n=14),
                        "regime"),
                       (StrategyTemplate("t", channel_type="keltner", n_entry=60), "upper")):
            truth = _compute_indicators(self.df, t)[key][k]
            old = 2 * 14 + 15 if key == "regime" else 60 + 5
            stale = _compute_indicators(self.df.iloc[k - old:], t)[key][old]
            warm = _compute_indicators(self.df.iloc[k - warmup_bars(t):], t)[key][warmup_bars(t)]
            self.assertGreater(abs(stale / truth - 1), 1e-3, key)
            self.assertLess(abs(warm / truth - 1), 1e-5, key)

    def test_ema_settle_bars_is_the_decay_time_of_the_seed(self):
        alpha = 2.0 / 21
        b = _ema_settle_bars(alpha, 1e-4)
        self.assertLess((1 - alpha) ** b, 1e-4)
        self.assertGreater((1 - alpha) ** (b - 1), 1e-4)
        self.assertGreater(warmup_bars(StrategyTemplate("t", channel_type="keltner", n_entry=60)),
                           warmup_bars(StrategyTemplate("t", channel_type="donchian", n_entry=60)))
        self.assertGreater(warmup_bars(StrategyTemplate("t", regime_filter="trend_only", regime_indicator="adx")),
                           warmup_bars(StrategyTemplate("t", regime_filter="trend_only", regime_indicator="er")))


class WalkForwardWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = synthetic_ohlc(1300, seed=29)

    def test_no_trade_opened_before_a_test_window_counts_in_it(self):
        """A position entered on the warm-up bars was chosen by parameters
        fitted on exactly those bars: its P&L inside the window is in-sample."""
        df = self.df
        for tpl in (StrategyTemplate("chan", exit_style="channel"),
                    StrategyTemplate("time", exit_style="time_stop", max_hold_bars=40, entry_style="pullback"),
                    StrategyTemplate("kel", channel_type="keltner", exit_style="atr_trail")):
            res = walk_forward(df, tpl, param_grid_for(tpl), train_bars=400, test_bars=100)
            live = [w for w in res["windows"] if not w["skipped"]]
            self.assertGreater(len(live), 4, tpl.name)
            for w in live:
                chosen = tpl.with_params(**w["params"])
                ts, te = df.index.get_loc(w["test_start"]), df.index.get_loc(w["test_end"]) + 1
                win = window_backtest(df, chosen, ts, te)
                self.assertTrue(all(t["entry_date"] >= w["test_start"] for t in win["trades"]), tpl.name)
                self.assertEqual(w["oos_stats"]["n_trades"], len(win["trades"]))
                # flat on the first bar unless it opened a trade on that very bar
                if not win["entries"][0]:
                    self.assertEqual(float(win["returns"].iloc[0]), 0.0)
                pd.testing.assert_series_equal(win["returns"], res["oos_returns"].loc[w["test_start"]:w["test_end"]],
                                               check_names=False)
                self.assertEqual(w["is_stats"]["n_bars"], 400)

    def test_walk_forward_handles_every_template(self):
        df = self.df.iloc[:900]
        for tpl in generate_templates("full")[::151]:
            grid = param_grid_for(tpl)
            res = walk_forward(df, tpl, grid, train_bars=300, test_bars=100, min_trades=1)
            self.assertIs(res["template"], tpl)
            self.assertEqual(len(res["windows"]), 6)
            self.assertEqual(res["boundaries"], [w["test_start"] for w in res["windows"]])
            pd.testing.assert_index_equal(res["oos_returns"].index, df.index[300:])
            self.assertFalse(res["oos_returns"].isna().any(), tpl.name)
            self.assertFalse(res["oos_equity"].isna().any(), tpl.name)
            for w in res["windows"]:
                self.assertEqual(df.index.get_loc(w["test_start"]) - df.index.get_loc(w["train_end"]), 1)
                if w["skipped"]:
                    self.assertTrue((res["oos_returns"].loc[w["test_start"]:w["test_end"]] == 0).all())
                    continue
                self.assertEqual(set(w["params"]), set(grid), tpl.name)
                for k, v in w["params"].items():
                    self.assertIn(v, grid[k], f"{tpl.name}: {k}={v} is not on the grid")
                self.assertTrue(np.isfinite(w["is_score"]))
                self.assertTrue(np.isnan(w["wfe"]) or np.isfinite(w["wfe"]))
            s = res["summary"]
            for key in ("oos_sharpe", "oos_cagr", "oos_max_drawdown", "oos_total_return", "pct_profitable_windows"):
                self.assertTrue(np.isfinite(s[key]), f"{tpl.name}: {key}")
            self.assertLessEqual(s["n_live_windows"], s["n_windows"])

    def test_refit_chooses_what_the_walk_forward_chose(self):
        """live.refit_params IS the in-sample step: on the history up to a
        window boundary it must land on the same parameters and score."""
        tpl = generate_templates("quick")[3]
        grid = param_grid_for(tpl)
        res = walk_forward(self.df, tpl, grid, train_bars=400, test_bars=100)
        live = [w for w in res["windows"] if not w["skipped"]]
        self.assertGreater(len(live), 3)
        for w in live[1:4]:
            ts = self.df.index.get_loc(w["test_start"])
            fit = refit_params(self.df.iloc[:ts], tpl, grid, train_bars=400)
            self.assertEqual(fit["params"], w["params"])
            self.assertAlmostEqual(fit["is_score"], w["is_score"])
            self.assertEqual(fit["is_stats"], w["is_stats"])
            self.assertEqual(fit["train_start"], w["train_start"])

    def test_walk_forward_matrix_covers_the_feasible_cells(self):
        tpl = generate_templates("quick")[0]
        m = walk_forward_matrix(self.df.iloc[:800], tpl, {"n_entry": [20, 40]},
                                train_lengths=(200, 400), test_lengths=(100, 700))
        self.assertEqual(sorted(m.index.tolist()), [(200, 100), (400, 100)])
        self.assertTrue(np.isfinite(m["oos_sharpe"]).all())


class WalkForwardEfficiencyTests(unittest.TestCase):
    @staticmethod
    def _windows(n_win, is_annual=0.10, oos_annual=0.10, train=500, test=125):
        oos = pd.Series(np.full(test * n_win, (1 + oos_annual) ** (1 / 252) - 1))
        is_stats = dict(total_return=is_annual * train / 252, sharpe=1.0, n_bars=train)
        oos_stats = dict(total_return=float((1 + oos.iloc[:test]).prod() - 1), sharpe=1.0,
                         n_trades=10, n_bars=test)
        wins = [dict(skipped=False, is_stats=is_stats, oos_stats=oos_stats, params_changed=False)] * n_win
        return wins, oos

    def test_wfe_does_not_grow_with_the_length_of_the_history(self):
        """Identical IS and OOS annual returns mean WFE = 1, whether the OOS
        history is two years or thirty. Compounding the OOS total used to
        turn thirty flat years into a WFE of 6."""
        short = summarize_walk_forward(*self._windows(4))["wfe"]
        long_ = summarize_walk_forward(*self._windows(64))["wfe"]
        self.assertAlmostEqual(short, 1.0, delta=0.03)
        self.assertAlmostEqual(long_, 1.0, delta=0.03)
        self.assertAlmostEqual(short, long_, places=9)
        half = summarize_walk_forward(*self._windows(40, oos_annual=0.05))["wfe"]
        self.assertAlmostEqual(half, 0.5, delta=0.02)
        self.assertTrue(np.isnan(summarize_walk_forward(*self._windows(8, is_annual=-0.05))["wfe"]))

    def test_pardo_criteria(self):
        wins, oos = self._windows(6)
        s = summarize_walk_forward(wins, oos)
        self.assertTrue(s["pardo_pass"])
        self.assertEqual(s["pct_profitable_windows"], 1.0)
        wins, oos = self._windows(6, oos_annual=0.04)
        self.assertFalse(summarize_walk_forward(wins, oos)["pardo_pass"])   # WFE 0.4 < 0.5
        self.assertFalse(summarize_walk_forward(wins[:2], oos)["pardo_pass"])  # too few windows


class ExitOrderingTests(unittest.TestCase):
    def test_channel_exit_fills_at_the_nearer_level(self):
        """Long, channel exit above the hard stop, one bar through both: the
        channel is reached first on the way down, so that is the fill."""
        n = 45
        o = np.full(n, 100.0); h = np.full(n, 101.0); lo = np.full(n, 99.0); c = np.full(n, 100.0)
        for i in range(30):                          # narrowing range, no break
            h[i] = 101.0 - 0.02 * i; lo[i] = 99.0 + 0.02 * i
        o[30], h[30], lo[30], c[30] = 100.0, 120.0, 99.8, 118.0   # break: long at upper[29]
        for i in range(31, 44):                      # grind higher: the 10-bar low rises to ~117
            c[i] = 118.0 + 0.2 * (i - 30); o[i] = c[i] - 0.1; h[i] = c[i] + 0.3; lo[i] = c[i] - 0.3
        o[44], h[44], lo[44], c[44] = 120.5, 121.0, 80.0, 85.0    # crash through channel AND stop
        df = _bars(o, h, lo, c)
        tpl = StrategyTemplate("t", direction_logic="trend", entry_style="stop", exit_style="channel",
                               n_entry=20, n_exit=10, atr_n=10, atr_mult_stop=3.0, cost_bps=0.0)
        ind = _compute_indicators(df, tpl)
        res = backtest(df, tpl)
        self.assertEqual(len(res["trades"]), 1)
        t = res["trades"][0]
        self.assertEqual((t["side"], df.index.get_loc(t["entry_date"]), df.index.get_loc(t["exit_date"])), (1, 30, 44))
        channel = float(ind["lower_x"][43])
        stop = t["entry_price"] - 3.0 * float(ind["atr"][29])
        self.assertGreater(channel, stop)
        self.assertGreater(df["Open"].iloc[44], channel)
        self.assertEqual(t["reason"], "channel")
        self.assertAlmostEqual(t["exit_price"], channel, places=9)
        # ...and when the stop is the nearer level, the stop fills (a tighter stop)
        tight = backtest(df, tpl.with_params(atr_mult_stop=0.5))["trades"]
        self.assertEqual(tight[-1]["reason"] if tight else None, "stop")

    def test_short_side_channel_exit_mirrors_the_long(self):
        n = 45
        o = np.full(n, 100.0); h = np.full(n, 101.0); lo = np.full(n, 99.0); c = np.full(n, 100.0)
        for i in range(30):
            h[i] = 101.0 - 0.02 * i; lo[i] = 99.0 + 0.02 * i
        o[30], h[30], lo[30], c[30] = 100.0, 100.2, 80.0, 82.0
        for i in range(31, 44):
            c[i] = 82.0 - 0.2 * (i - 30); o[i] = c[i] + 0.1; h[i] = c[i] + 0.3; lo[i] = c[i] - 0.3
        o[44], h[44], lo[44], c[44] = 79.5, 120.0, 79.0, 115.0
        df = _bars(o, h, lo, c)
        tpl = StrategyTemplate("t", direction_logic="trend", entry_style="stop", exit_style="channel",
                               n_entry=20, n_exit=10, atr_n=10, atr_mult_stop=3.0, cost_bps=0.0)
        ind = _compute_indicators(df, tpl)
        t = backtest(df, tpl)["trades"][0]
        self.assertEqual((t["side"], t["reason"]), (-1, "channel"))
        self.assertAlmostEqual(t["exit_price"], float(ind["upper_x"][43]), places=9)


class IndicatorCacheTests(unittest.TestCase):
    def test_cache_key_sees_high_and_low(self):
        df = synthetic_ohlc(600, seed=2)
        tpl = StrategyTemplate("t")
        a = _compute_indicators(df, tpl)
        wick = df.copy()
        wick["High"] = wick["High"] * 1.05
        b = _compute_indicators(wick, tpl)
        self.assertFalse(np.allclose(a["atr"][30:], b["atr"][30:]))
        np.testing.assert_allclose(b["atr"][30:], atr(wick, tpl.atr_n).to_numpy()[30:])
        self.assertFalse(np.allclose(a["upper"][50:], b["upper"][50:]))
        # same frame twice is still served from the cache (identical arrays)
        c = _compute_indicators(wick, tpl)
        self.assertIs(b["atr"], c["atr"])


if __name__ == "__main__":
    unittest.main()
