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
    _compute_indicators, annualized_sharpe,
    hedge_weights, hedge_channel, hedge_warmup, hedge_direction, hedge_experts, donchian,
    HEDGE_LADDER, HEDGE_MEMORY, _adahedge_loop,
    hedge_learner_params, hedge_buffer, HEDGE_LEARNERS, HEDGE_FLOOR, _IND_CACHE,
)
import strategy as S  # noqa: E402
from generator import generate_templates, param_grid_for  # noqa: E402
from walkforward import (  # noqa: E402
    walk_forward, grid_combos, smooth_scores, warmup_bars, summarize_walk_forward, window_backtest,
)
from robustness import (  # noqa: E402
    trial_returns, cpcv, cpcv_paths, cscv_pbo, deflated_sharpe_ratio, probabilistic_sharpe_ratio,
    bootstrap_sharpe_pvalue, reality_check, hrp_weights, stationary_bootstrap_indices,
    min_backtest_length,
)
from portfolio import (  # noqa: E402
    select_portfolio, walk_forward_portfolio, returns_frame, portfolio_weights,
)


def _ohlc_from_close(close, rng):
    n = len(close)
    intrabar = np.abs(rng.normal(0, 0.003, n)) + 0.001
    open_ = np.roll(close, 1); open_[0] = close[0]
    high = np.maximum.reduce([close * (1 + intrabar), open_, close])
    low = np.minimum.reduce([close * (1 - intrabar), open_, close])
    idx = pd.bdate_range("2010-01-01", periods=n)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": 1}, index=idx)


def regime_series(seed=0, kinds=("trend", "mr", "trend"), n_each=1000):
    """Alternating regimes: steady up-trend / AR(1) mean reversion."""
    rng = np.random.default_rng(seed)
    parts, lvl = [], 100.0
    for kind in kinds:
        if kind == "trend":
            c = lvl * np.cumprod(1 + 0.002 + rng.normal(0, 0.008, n_each))
        else:
            x = np.zeros(n_each)
            for i in range(1, n_each):
                x[i] = 0.85 * x[i - 1] + rng.normal(0, 0.02)
            c = lvl * np.exp(x)
        parts.append(c)
        lvl = c[-1]
    return _ohlc_from_close(np.concatenate(parts), rng)


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
            sample = generate_templates("full")[::211]
            sample += [t.with_params(vol_target=0.15) for t in generate_templates("full")[::503]]
            for tpl in sample:
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


class VolTargetSizingTests(unittest.TestCase):
    """`vol_target` > 0 sizes an entry to a constant annualized volatility
    instead of risk_pct on the ATR stop. Off, nothing may change."""

    K = 100        # first_trade_bar: past every warm-up used below

    @classmethod
    def setUpClass(cls):
        cls.df = synthetic_ohlc(1200, seed=3)
        cls.close = cls.df["Close"]

    def setUp(self):
        import strategy as S
        self._ppy = S.periods_per_year()

    def tearDown(self):
        import strategy as S
        S.set_periods_per_year(self._ppy)

    @staticmethod
    def _rvol(df, n):
        return df["Close"].pct_change().rolling(n).std()

    def _flat_entries(self, res):
        """(bar, trade) for every trade opened from a flat book, so the cash
        the engine sized on is the previous bar's equity."""
        exits = {t["exit_date"] for t in res["trades"]}
        out = []
        for t in res["trades"]:
            i = self.df.index.get_loc(t["entry_date"])
            if self.df.index[i] not in exits or t["exit_date"] == t["entry_date"]:
                out.append((i, t))
        return out

    def test_off_is_byte_identical_and_the_lookback_is_inert(self):
        tpl = StrategyTemplate("t")
        a = backtest(self.df, tpl)
        b = backtest(self.df, tpl.with_params(vol_target=0.0, vol_target_n=17))
        np.testing.assert_array_equal(a["equity"].values, b["equity"].values)
        self.assertNotIn("rvol", a["indicators"])
        self.assertNotIn("rvol", b["indicators"])

    def test_entry_notional_is_the_target_over_the_realized_vol(self):
        import strategy as S
        tpl = StrategyTemplate("t", vol_target=0.15, vol_target_n=60, cost_bps=0.0, max_leverage=1e9)
        res = backtest(self.df, tpl, first_trade_bar=self.K)
        rv = self._rvol(self.df, 60)
        eq = res["equity"]
        checked = 0
        for i, t in self._flat_entries(res):
            notional = t["shares"] * t["entry_price"] / eq.iloc[i - 1]
            self.assertAlmostEqual(notional, (0.15 / np.sqrt(S.periods_per_year())) / rv.iloc[i - 1],
                                   places=8, msg=str(t["entry_date"]))
            checked += 1
        self.assertGreater(checked, 5)

    def test_a_target_equal_to_the_realized_vol_gives_unit_notional(self):
        """The whole point: at the asset's own vol the strategy holds ~1x, the
        buy-and-hold scale."""
        import strategy as S
        base = StrategyTemplate("t", vol_target=0.15, vol_target_n=60, cost_bps=0.0, max_leverage=1e9)
        first = backtest(self.df, base, first_trade_bar=self.K)
        i0, t0 = self._flat_entries(first)[0]
        rv = self._rvol(self.df, 60)
        tuned = base.with_params(vol_target=float(rv.iloc[i0 - 1] * np.sqrt(S.periods_per_year())))
        res = backtest(self.df, tuned, first_trade_bar=self.K)
        np.testing.assert_array_equal(res["entries"], first["entries"])
        t = [tr for tr in res["trades"] if tr["entry_date"] == t0["entry_date"]][0]
        self.assertAlmostEqual(t["shares"] * t["entry_price"] / res["equity"].iloc[i0 - 1], 1.0, places=8)

    def test_when_you_trade_does_not_depend_on_how_much(self):
        base = StrategyTemplate("t", cost_bps=0.0, max_leverage=1e9)
        a = backtest(self.df, base, first_trade_bar=self.K)
        b = backtest(self.df, base.with_params(vol_target=0.15), first_trade_bar=self.K)
        np.testing.assert_array_equal(a["entries"], b["entries"])
        self.assertEqual([t["entry_date"] for t in a["trades"]], [t["entry_date"] for t in b["trades"]])
        self.assertEqual([t["exit_date"] for t in a["trades"]], [t["exit_date"] for t in b["trades"]])
        self.assertGreater(len(a["trades"]), 5)

    def test_the_leverage_cap_still_binds(self):
        tpl = StrategyTemplate("t", vol_target=3.0, max_leverage=2.0, cost_bps=0.0)
        res = backtest(self.df, tpl, first_trade_bar=self.K)
        # the cap is on the notional AT ENTRY: the size is then fixed, so a
        # losing position drifts above it mark-to-market (under either rule)
        lev = [t["shares"] * t["entry_price"] / res["equity"].iloc[i - 1] for i, t in self._flat_entries(res)]
        self.assertLessEqual(max(lev), 2.0 + 1e-9)
        self.assertGreater(sum(abs(x - 2.0) < 1e-6 for x in lev), 5)

    def test_the_stop_is_still_atr_mult_stop_away(self):
        tpl = StrategyTemplate("t", exit_style="target_stop", vol_target=0.15, cost_bps=0.0)
        res = backtest(self.df, tpl, first_trade_bar=self.K)
        atr_v = res["indicators"]["atr"]
        checked = 0
        for t in res["trades"]:
            if t["reason"] != "stop":
                continue
            i = self.df.index.get_loc(t["entry_date"])
            j = self.df.index.get_loc(t["exit_date"])
            # the stop was set off the ATR on the bar before entry, filled at the
            # level or at the open if it gapped through
            level = t["entry_price"] - t["side"] * tpl.atr_mult_stop * atr_v[i - 1]
            open_j = float(self.df["Open"].iloc[j])
            expected = min(open_j, level) if t["side"] == 1 else max(open_j, level)
            self.assertAlmostEqual(t["exit_price"], expected, places=8)
            checked += 1
        self.assertGreater(checked, 3)

    def test_no_entry_until_the_realized_vol_is_formed(self):
        tpl = StrategyTemplate("t", vol_target=0.15, vol_target_n=300)
        res = backtest(self.df, tpl)
        ready = res["indicators"]["ready"]
        # pct_change eats bar 0, so the 300-return window is first full ON bar
        # 300; bar 301 is the first that can decide off it
        self.assertEqual(int(ready[:300].sum()), 0)
        self.assertTrue(ready[300])
        self.assertEqual(int(res["entries"][:301].sum()), 0)
        self.assertGreater(int(res["entries"][301:].sum()), 0)
        base_ready = backtest(self.df, tpl.with_params(vol_target=0.0))["indicators"]["ready"]
        np.testing.assert_array_equal(ready, base_ready & ~np.isnan(res["indicators"]["rvol"]))

    def test_a_zero_realized_vol_starts_nothing_new(self):
        """Twenty identical closes make the realized vol exactly zero: the
        engine must not divide by it (and must not size to infinity), so no
        new business starts until a return shows up again -- the same rule the
        ATR already follows. The ATR rule keeps trading on the same bars."""
        rng = np.random.default_rng(5)
        pre = _ohlc_from_close(100 * np.cumprod(1 + rng.normal(0, 0.01, 200)), rng)
        L = 1.02 * float(pre["High"].max())              # a break into a dead-flat patch
        flat = pd.DataFrame({"Open": L, "High": L * 1.001, "Low": L * 0.999, "Close": L, "Volume": 1},
                            index=pd.bdate_range(pre.index[-1] + pd.Timedelta(days=1), periods=120))
        df = pd.concat([pre, flat])
        vol = StrategyTemplate("v", exit_style="time_stop", max_hold_bars=10, vol_target=0.15, vol_target_n=20)
        base = vol.with_params(vol_target=0.0)
        rv, rb = backtest(df, vol), backtest(df, base)
        rvol = rv["indicators"]["rvol"]
        dead = np.where(rvol == 0.0)[0]                   # bars whose 20 returns are all zero
        dead = dead[dead + 1 < len(df)]                   # the last bar has no next bar to decide on
        self.assertGreater(len(dead), 40)
        self.assertTrue(rv["indicators"]["ready"][dead].all(), "the block is a_ok, not the warm-up")
        self.assertTrue((rv["indicators"]["atr"][dead] > 0).all(), "the wicks keep the ATR alive")
        # the bars that DECIDE off a dead vol are the ones after it
        self.assertEqual(int(rv["entries"][dead + 1].sum()), 0)
        self.assertGreater(int(rb["entries"][dead + 1].sum()), 0)
        # ... and the vol-target template did trade while the vol was alive
        self.assertGreater(int(rv["entries"][:dead[0] + 1].sum()), 0)

    def test_the_target_is_read_at_the_bar_frequency_in_force(self):
        """The cached realized vol is per-bar; the annualized target is scaled
        by periods_per_year() at call time, so a worker's setting is honoured."""
        import strategy as S
        tpl = StrategyTemplate("t", vol_target=0.15, cost_bps=0.0, max_leverage=1e9)
        S.set_periods_per_year(252)
        daily = backtest(self.df, tpl, first_trade_bar=self.K)
        S.set_periods_per_year(252 * 7)
        hourly = backtest(self.df, tpl, first_trade_bar=self.K)
        np.testing.assert_array_equal(daily["entries"], hourly["entries"])
        self.assertAlmostEqual(daily["trades"][0]["shares"] / hourly["trades"][0]["shares"], np.sqrt(7.0), places=6)


class HedgeChannelTests(unittest.TestCase):
    """The online-learned channel: parameter-free, causal, bounded memory,
    and it learns the period and the side."""

    def setUp(self):
        self.df = synthetic_ohlc(1200, seed=7)

    def test_adahedge_follows_the_leader(self):
        loss = np.full((300, 4), 0.6)
        loss[:, 1] = 0.3                       # expert 1 is always better
        W, eta = _adahedge_loop(loss, HEDGE_MEMORY)
        np.testing.assert_allclose(W.sum(axis=1), 1.0)
        self.assertGreater(W[-1, 1], 0.99)
        rng = np.random.default_rng(0)
        noisy = rng.uniform(0, 1, (3000, 4))
        noisy[:, 2] -= 0.08                    # a small but persistent edge
        W, eta = _adahedge_loop(np.clip(noisy, 0, 1), HEDGE_MEMORY)
        self.assertEqual(int(np.argmax(W[-1])), 2)
        self.assertTrue(np.isfinite(eta[-1]))  # learning rate shrank from FTL to a finite eta

    def test_a_written_off_expert_still_ends_follow_the_leader(self):
        """While the mixability gap is tiny, eta is huge and a trailing
        expert's weight underflows to exactly 0. When that expert then beats
        the leader by a mile, a mix loss summed over the WEIGHTS is -log(0):
        the round's gap was discarded, eta stayed at 74,000 and the learner
        flipped all-in, on the one round built to teach it caution."""
        c = 1e-5
        loss = np.array([(0, c), (0, c), (0, c), (0, 10 * c), (0, 300 * c), (0, 3000 * c),   # expert 0 leads ...
                         (1.0, 0.0)])                                                        # ... and is routed
        W, eta = _adahedge_loop(loss, HEDGE_MEMORY)
        np.testing.assert_array_equal(W[5], [1.0, 0.0])      # the precondition: written off completely
        self.assertGreater(eta[5], 745.0)                    # and exp(-eta * 1) underflows as well
        # the last round's mix loss is the written-off expert's 3313c deficit
        # (it is the better of "leader loses 1" and "catch up 3313c, lose 0"),
        # the Hedge loss is 1, so delta gains 1 - 3313c on top of what it had
        delta = np.log(2) / eta[5] + 1.0 - 3313 * c
        self.assertAlmostEqual(eta[6], np.log(2) / delta, places=9)
        # expert 1 now leads by exactly that gap: weights 2/3 and 1/3, not 1 and 0
        np.testing.assert_allclose(W[6], [1 / 3, 2 / 3], atol=1e-5)

    def test_bounded_memory_tracks_a_change_of_leader(self):
        loss = np.full((2000, 4), 0.6)
        loss[:1000, 0] = 0.4                   # expert 0 leads for 1000 rounds...
        loss[1000:, 3] = 0.4                   # ...then expert 3 does
        W, _ = _adahedge_loop(loss, HEDGE_MEMORY)
        self.assertGreater(W[999, 0], 0.9)
        self.assertGreater(W[1000 + HEDGE_MEMORY, 3], 0.9)   # the old leader has left the memory
        # row t depends on rows t-memory+1..t only
        W2, _ = _adahedge_loop(loss[500:], HEDGE_MEMORY)
        np.testing.assert_allclose(W[500 + HEDGE_MEMORY:], W2[HEDGE_MEMORY:])

    def test_weights_are_causal_and_normalised(self):
        W = hedge_weights(self.df, 20, "trend")
        np.testing.assert_allclose(W.sum(axis=1), 1.0)
        self.assertTrue((W >= 0).all())
        Wc = hedge_weights(self.df.iloc[:700], 20, "trend")
        np.testing.assert_allclose(W[:700], Wc)  # the future does not change the past

    def test_channel_is_a_mixture_of_the_experts(self):
        up, lo, mid = hedge_channel(self.df, 20, "trend")
        warm = hedge_warmup(20)
        self.assertEqual(int(up.isna().sum()), warm)
        ups = np.stack([donchian(self.df, n)[0].to_numpy() for n in HEDGE_LADDER], axis=1)[warm:]
        los = np.stack([donchian(self.df, n)[1].to_numpy() for n in HEDGE_LADDER], axis=1)[warm:]
        self.assertTrue((up.to_numpy()[warm:] <= ups.max(axis=1) + 1e-9).all())
        self.assertTrue((up.to_numpy()[warm:] >= ups.min(axis=1) - 1e-9).all())
        self.assertTrue((lo.to_numpy()[warm:] <= los.max(axis=1) + 1e-9).all())
        self.assertTrue((lo.to_numpy()[warm:] >= los.min(axis=1) - 1e-9).all())

    def test_no_lookback_in_the_grid(self):
        for tpl in generate_templates("online"):
            grid = param_grid_for(tpl)
            for key in ("n_entry", "n_exit", "channel_k"):
                self.assertNotIn(key, grid, tpl.name)
        tpl = generate_templates("online")[0]
        self.assertEqual(param_grid_for(tpl), {})
        res = walk_forward(self.df, tpl, param_grid_for(tpl), train_bars=400, test_bars=100)
        self.assertEqual(len(res["oos_returns"]), len(self.df) - 400)
        self.assertGreaterEqual(warmup_bars(tpl), hedge_warmup(tpl.atr_n))
        self.assertGreaterEqual(warmup_bars(StrategyTemplate("t", direction_logic="learned")), hedge_warmup(20))

    def test_learns_the_trend_lookback_and_the_fade(self):
        # on a strong trend the trend learner concentrates on the slowest expert...
        df = trending_series()
        W = hedge_weights(df, 20, "trend")
        self.assertGreater(W[-500:, 2:].sum(axis=1).mean(), 0.9)   # the two slowest experts
        tr = backtest(df, StrategyTemplate("t", channel_type="hedge", cost_bps=0.0))
        ct = backtest(df, StrategyTemplate("t", channel_type="hedge", direction_logic="countertrend", cost_bps=0.0))
        self.assertGreater(tr["stats"]["total_return"], 0.2)
        self.assertLess(ct["stats"]["total_return"], tr["stats"]["total_return"])
        # ...and on a mean-reverting series the countertrend learner picks a fast
        # expert and fading it makes money where following it loses
        mr = regime_series(kinds=("mr",), n_each=2000)
        Wc = hedge_weights(mr, 20, "countertrend")
        self.assertGreater(Wc[-1000:, :2].sum(axis=1).mean(), 0.6)  # the two fastest experts
        fade = backtest(mr, StrategyTemplate("t", channel_type="hedge", direction_logic="countertrend",
                                             exit_style="time_stop", max_hold_bars=5, cost_bps=0.0))
        follow = backtest(mr, StrategyTemplate("t", channel_type="hedge", direction_logic="trend",
                                               exit_style="time_stop", max_hold_bars=5, cost_bps=0.0))
        self.assertGreater(fade["stats"]["total_return"], 0.0)
        self.assertLess(follow["stats"]["total_return"], fade["stats"]["total_return"])

    def test_learned_direction_follows_then_fades(self):
        df = regime_series()
        d = hedge_direction(df, 20)
        lookbacks, sides = hedge_experts("learned")
        self.assertEqual(len(lookbacks), 2 * len(HEDGE_LADDER))
        self.assertTrue(np.isnan(d[:hedge_warmup(20)]).all())
        self.assertGreater((d[500:1000] > 0).mean(), 0.9)     # trend regime: follow
        self.assertGreater((d[1400:2000] < 0).mean(), 0.9)    # mean reversion: fade
        self.assertGreater((d[2500:3000] > 0).mean(), 0.9)    # trend again: follow
        learned = backtest(df, StrategyTemplate("t", channel_type="hedge", direction_logic="learned", cost_bps=0.0))
        follow = backtest(df, StrategyTemplate("t", channel_type="hedge", direction_logic="trend", cost_bps=0.0))
        fade = backtest(df, StrategyTemplate("t", channel_type="hedge", direction_logic="countertrend", cost_bps=0.0))
        self.assertGreater(learned["stats"]["total_return"], follow["stats"]["total_return"])
        self.assertGreater(learned["stats"]["total_return"], fade["stats"]["total_return"])
        # a trade keeps the exit logic of the side it was opened under
        reasons = {t["reason"] for t in learned["trades"]}
        self.assertTrue({"channel", "midline"} & reasons)



class DiscountedHedgeTests(unittest.TestCase):
    """The learner's memory: an exponentially discounted one by default, the
    optional weight floor on top, and the original hard window kept as an
    option. All three keep the learner a function of a fixed number of past
    bars, which the walk-forward's warm-up buffer relies on."""

    def setUp(self):
        self.df = synthetic_ohlc(1600, seed=7)

    def test_the_settings(self):
        self.assertEqual(StrategyTemplate("t").hedge_learner, "window")
        self.assertEqual(hedge_learner_params("window"), (HEDGE_MEMORY, 1.0, 0.0))
        mem, gamma, floor = hedge_learner_params("discounted")
        self.assertAlmostEqual(gamma ** S.HEDGE_HALF_LIFE, 0.5)          # a half-life, in bars
        self.assertAlmostEqual(gamma ** mem, 2.0 ** -S.HEDGE_HORIZON)    # the memory cut-off weighs 1/16
        self.assertEqual(floor, 0.0)
        self.assertEqual(hedge_learner_params("floored"), (mem, gamma, HEDGE_FLOOR))
        with self.assertRaises(ValueError):
            hedge_learner_params("forever")
        with self.assertRaises(AssertionError):
            StrategyTemplate("t", hedge_learner="forever").validate()
        for learner in HEDGE_LEARNERS:
            self.assertGreater(hedge_buffer(20, learner), hedge_warmup(20, learner))
            t = StrategyTemplate("t", channel_type="hedge", hedge_learner=learner)
            self.assertGreaterEqual(warmup_bars(t), hedge_buffer(20, learner))

    def test_the_undiscounted_unfloored_run_is_plain_adahedge(self):
        rng = np.random.default_rng(3)
        loss = rng.uniform(0, 1, (600, 4))
        W0, e0 = _adahedge_loop(loss, HEDGE_MEMORY)
        W1, e1 = _adahedge_loop(loss, *hedge_learner_params("window"))
        np.testing.assert_array_equal(W0, W1)
        np.testing.assert_array_equal(e0, e1)

    def test_an_old_bar_fades_instead_of_dropping_off_a_cliff(self):
        """Tip one round towards expert 0 and follow the change it makes to the
        weights as the round ages. Under the hard window its influence is as
        large as ever the day before it leaves the memory (larger: the rest of
        the window has been re-dealt around it), then 0 the next day. Under
        discounting it has faded to a small fraction by then."""
        rng = np.random.default_rng(0)
        base = rng.uniform(0, 1, (1400, 4))
        effect = {}
        for learner in ("window", "discounted"):
            mem, gamma, floor = hedge_learner_params(learner)
            Wb, _ = _adahedge_loop(base, mem, gamma, floor)
            runs = []
            for s in range(200, 260, 5):
                tipped = base.copy()
                tipped[s] = [0.0, 1.0, 1.0, 1.0]
                Wt, _ = _adahedge_loop(tipped, mem, gamma, floor)
                runs.append(np.abs(Wt - Wb).sum(axis=1)[s:s + mem + 1])
            e = np.mean(runs, axis=0)
            self.assertEqual(e[mem], 0.0)                     # a fixed number of past bars, either way
            effect[learner] = (e[:10].mean(), e[mem - 10:mem].mean())
        young, old = effect["window"]
        self.assertGreater(old, young)                        # the cliff: full weight to the last day
        young, old = effect["discounted"]
        self.assertLess(old, 0.25 * young)                    # faded long before the cut-off
        self.assertLess(old, 0.25 * effect["window"][1])

    def test_discounting_tracks_a_change_of_leader_sooner(self):
        rng = np.random.default_rng(1)
        loss = rng.uniform(0, 1, (2000, 4))
        loss[:1000, 0] -= 0.1                  # expert 0 leads for 1000 noisy rounds...
        loss[1000:, 3] -= 0.1                  # ...then expert 3 does
        loss = np.clip(loss, 0, 1)
        took = {}
        for learner in HEDGE_LEARNERS:
            W, _ = _adahedge_loop(loss, *hedge_learner_params(learner))
            self.assertGreater(W[999, 0], 0.9, learner)
            took[learner] = int(np.argmax(W[1000:, 3] > 0.5))
            self.assertGreater(W[-200:, 3].mean(), 0.5, learner)   # and settled on it
        self.assertLess(took["discounted"], took["window"])
        self.assertLess(took["floored"], took["discounted"])  # a written-off expert is never far behind

    def test_the_floor_keeps_every_expert_in_play(self):
        rng = np.random.default_rng(2)
        loss = rng.uniform(0, 1, (1500, 4))
        loss[:, 0] -= 0.2                      # a clear leader: the others would sink towards 0
        loss = np.clip(loss, 0, 1)
        Wd, _ = _adahedge_loop(loss, *hedge_learner_params("discounted"))
        Wf, eta = _adahedge_loop(loss, *hedge_learner_params("floored"))
        np.testing.assert_allclose(Wf.sum(axis=1), 1.0)
        finite = np.isfinite(eta)              # follow-the-leader rounds have no scale to floor on
        self.assertGreater(finite[100:].mean(), 0.99)
        ratio = Wf.min(axis=1) / Wf.max(axis=1)
        self.assertTrue((ratio[finite] >= HEDGE_FLOOR * (1 - 1e-9)).all())
        self.assertLess((Wd.min(axis=1) / Wd.max(axis=1))[500:].min(), HEDGE_FLOOR / 100)
        self.assertEqual(int(np.argmax(Wf[-1])), 0)

    def test_state_is_a_function_of_the_last_memory_rounds(self):
        rng = np.random.default_rng(4)
        loss = rng.uniform(0, 1, (1500, 4))
        for learner in HEDGE_LEARNERS:
            mem, gamma, floor = hedge_learner_params(learner)
            W, _ = _adahedge_loop(loss, mem, gamma, floor)
            W2, _ = _adahedge_loop(loss[400:], mem, gamma, floor)
            np.testing.assert_allclose(W[400 + mem:], W2[mem:], err_msg=learner)

    def test_a_warm_window_matches_a_full_history_run_with_every_learner(self):
        k, n = 1000, len(self.df)
        for learner in HEDGE_LEARNERS:
            for tpl in (StrategyTemplate("t", channel_type="hedge", hedge_learner=learner),
                        StrategyTemplate("t", direction_logic="learned", exit_style="atr_trail",
                                         hedge_learner=learner)):
                full = backtest(self.df, tpl, first_trade_bar=k)
                win = window_backtest(self.df, tpl, k, n)
                np.testing.assert_allclose(win["equity"].to_numpy(), full["equity"].to_numpy()[k:],
                                           rtol=1e-9, err_msg=f"{learner} {tpl.direction_logic}")
                self.assertEqual(len(win["trades"]), len(full["trades"]))

    def test_the_learner_is_in_the_indicator_cache_key(self):
        tpl = StrategyTemplate("t", direction_logic="learned", channel_type="hedge", cost_bps=0.0)
        runs = {}
        for learner in ("window", "discounted", "window"):
            runs.setdefault(learner, []).append(backtest(self.df, tpl.with_params(hedge_learner=learner))["equity"])
        pd.testing.assert_series_equal(runs["window"][0], runs["window"][1])
        self.assertFalse(runs["window"][0].equals(runs["discounted"][0]))
        # and so are its settings, which an experiment may change between runs
        before = runs["discounted"][0]
        half_life = S.HEDGE_HALF_LIFE
        try:
            S.HEDGE_HALF_LIFE = 40
            after = backtest(self.df, tpl.with_params(hedge_learner="discounted"))["equity"]
        finally:
            S.HEDGE_HALF_LIFE = half_life
            _IND_CACHE.clear()
        self.assertFalse(before.equals(after))

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
        # bar 0 has nothing to warm up on: every window is scored from the end
        # of the grid's longest warm-up, the earliest bar that can be
        warm = max(warmup_bars(self.tpl.with_params(**p)) for p in grid_combos(self.grid)[0])
        for w in res["windows"]:
            self.assertEqual(w["train_start"], self.df.index[warm])

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


class AnnualizationTests(unittest.TestCase):
    """PERIODS_PER_YEAR used to be imported by value into four modules (and
    baked into default arguments), so a non-daily bar frequency could not be
    set at all. Everything must now read it through strategy.periods_per_year()."""

    def setUp(self):
        import strategy as S
        self._saved = S.periods_per_year()

    def tearDown(self):
        import strategy as S
        S.set_periods_per_year(self._saved)

    def test_sharpe_scales_with_the_bar_frequency(self):
        import strategy as S
        rng = np.random.default_rng(0)
        r = pd.Series(rng.normal(0.0004, 0.01, 3000))
        daily = annualized_sharpe(r)
        S.set_periods_per_year(S.periods_per_year_for_interval("1h"))
        hourly = annualized_sharpe(r)
        self.assertAlmostEqual(hourly / daily, np.sqrt(7.0), places=6)

    def test_every_module_honours_the_setting(self):
        """walkforward, robustness and portfolio must all see the change."""
        import strategy as S
        import robustness as R
        from robustness import _sharpe_cols

        rng = np.random.default_rng(1)
        X = rng.normal(0.0004, 0.01, (2000, 3))
        mask = np.ones(2000, dtype=bool)
        sr_daily = _sharpe_cols(X, mask).copy()
        win_daily = summarize_walk_forward(
            [dict(skipped=False, is_stats=dict(total_return=0.1, sharpe=1.0, n_bars=500),
                  oos_stats=dict(total_return=0.05, sharpe=0.8, n_trades=10, n_bars=125),
                  params_changed=False)] * 4,
            pd.Series(X[:, 0]),
        )
        boot_daily = R.bootstrap_sharpe_pvalue(X[:, 0], n_boot=50)["sharpe"]
        dsr_daily = R.deflated_sharpe_ratio(X[:, 0], n_trials=10, var_sr_trials=1e-4)["sharpe_annual"]
        mbtl_daily = min_backtest_length(100, 1.0)

        S.set_periods_per_year(S.periods_per_year_for_interval("1h"))
        k = np.sqrt(7.0)
        np.testing.assert_allclose(_sharpe_cols(X, mask), sr_daily * k, rtol=1e-9)
        np.testing.assert_allclose(R.bootstrap_sharpe_pvalue(X[:, 0], n_boot=50)["sharpe"],
                                   boot_daily * k, rtol=1e-9)
        np.testing.assert_allclose(
            R.deflated_sharpe_ratio(X[:, 0], n_trials=10, var_sr_trials=1e-4)["sharpe_annual"],
            dsr_daily * k, rtol=1e-9)
        # MinBTL is in years: the same Sharpe needs the same wall-clock time,
        # but 7x as many bars, so the YEARS figure is unchanged
        self.assertAlmostEqual(min_backtest_length(100, 1.0), mbtl_daily, places=9)
        win_hourly = summarize_walk_forward(
            [dict(skipped=False, is_stats=dict(total_return=0.1, sharpe=1.0, n_bars=500),
                  oos_stats=dict(total_return=0.05, sharpe=0.8, n_trades=10, n_bars=125),
                  params_changed=False)] * 4,
            pd.Series(X[:, 0]),
        )
        self.assertAlmostEqual(win_hourly["oos_sharpe"] / win_daily["oos_sharpe"], k, places=9)
        self.assertGreater(win_hourly["oos_cagr"], win_daily["oos_cagr"])

    def test_unknown_interval_is_rejected(self):
        import strategy as S
        with self.assertRaises(ValueError):
            S.periods_per_year_for_interval("3d")
        with self.assertRaises(ValueError):
            S.set_periods_per_year(0)


if __name__ == "__main__":
    unittest.main()
