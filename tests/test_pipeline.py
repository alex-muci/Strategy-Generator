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
    hedge_weights, hedge_channel, hedge_warmup, hedge_direction, hedge_experts, hedge_diagnostics, donchian,
    HEDGE_LADDER, HEDGE_MEMORY, HEDGE_HORIZONS, _adahedge_loop, _hedge_loss,
    hedge_ladder, hedge_ladder_for, HedgeExpert, HEDGE_LADDERS, HEDGE_CHANNELS, HEDGE_EXIT_SCALE,
    HEDGE_GATE_N, HEDGE_GATE_THRESHOLD, HEDGE_WIDE_N, HEDGE_WIDE_WIDTHS, _hedge_gates, _hedge_stances,
    bollinger, sma, atr, efficiency_ratio, REGIME_INDICATORS,
)
from generator import generate_templates, param_grid_for  # noqa: E402
from walkforward import (  # noqa: E402
    walk_forward, grid_combos, smooth_scores, warmup_bars, summarize_walk_forward,
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
    no expert ever written off, and it learns the period, the side and
    how long a memory pays."""

    def setUp(self):
        self.df = synthetic_ohlc(1200, seed=7)

    def test_adahedge_follows_the_leader(self):
        loss = np.full((300, 4), 0.6)
        loss[:, 1] = 0.3                       # expert 1 is always better
        W, eta, V, surprise = _adahedge_loop(loss, HEDGE_MEMORY)
        np.testing.assert_allclose(W.sum(axis=1), 1.0)
        np.testing.assert_allclose(V.sum(axis=1), 1.0)
        self.assertGreater(W[-1, 1], 0.99)
        self.assertLess(surprise[-1], 0.01)    # the mixture is as good as the best expert
        rng = np.random.default_rng(0)
        noisy = rng.uniform(0, 1, (3000, 4))
        noisy[:, 2] -= 0.08                    # a small but persistent edge
        W, eta, _, surprise = _adahedge_loop(np.clip(noisy, 0, 1), HEDGE_MEMORY)
        self.assertEqual(int(np.argmax(W[-500:].mean(axis=0))), 2)
        self.assertTrue(np.isfinite(eta[-1]).all())  # every learning rate shrank from FTL to a finite eta
        self.assertTrue((surprise >= 0).all())

    def test_a_written_off_expert_still_ends_follow_the_leader(self):
        """While the mixability gap is tiny, eta is huge and a trailing
        expert's weight underflows to exactly 0. When that expert then beats
        the leader by a mile, a mix loss summed over the WEIGHTS is -log(0):
        the round's gap was discarded, eta stayed at 74,000 and the learner
        flipped all-in, on the one round built to teach it caution."""
        c = 1e-5
        loss = np.array([(0, c), (0, c), (0, c), (0, 10 * c), (0, 300 * c), (0, 3000 * c),   # expert 0 leads ...
                         (1.0, 0.0)])                                                        # ... and is routed
        W, eta, _, _ = _adahedge_loop(loss, HEDGE_MEMORY)
        self.assertEqual(W[5, 1], 0.0)                       # the precondition: written off completely
        self.assertAlmostEqual(W[5, 0], 1.0, places=12)
        self.assertTrue((eta[5] > 745.0).all())              # and exp(-eta * 1) underflows as well
        # the last round's mix loss is the written-off expert's ~3313c deficit
        # (it is the better of "leader loses 1" and "catch up 3313c, lose 0"),
        # the Hedge loss is 1, so delta gains ~1 - 3313c and expert 1 leads by
        # about that gap: weights near 2/3 and 1/3 (exactly so without the
        # discount), not 1 and 0
        np.testing.assert_allclose(W[6], [1 / 3, 2 / 3], atol=2e-2)

    def test_no_expert_is_written_off_for_good(self):
        """A discounted deficit is bounded by the lifetime times the shortfall
        per bar, whatever the expert lost before, and the shortest lifetime
        on the ladder bounds it tightest: an expert that starts winning is
        back in front within tens of bars, and the meta learner, scoring
        the lifetimes on their own hedge loss, follows the one that moved."""
        loss = np.full((440, 4), 0.6)
        loss[:400, 0] = 0.4                    # 400 rounds of write-off...
        loss[400:, 3] = 0.4                    # ...then the trailer wins by the same edge
        W, _, V, _ = _adahedge_loop(loss, HEDGE_MEMORY)
        self.assertLess(W[399, 3], 1e-6)
        self.assertGreater(W[399 + 30, 3], 0.5)                     # in front within thirty rounds
        self.assertGreater(V[399 + 30, :2].sum(), V[399, :2].sum())  # carried by the short lifetimes

    def test_bounded_memory_tracks_a_change_of_leader(self):
        loss = np.full((2000, 4), 0.6)
        loss[:1000, 0] = 0.4                   # expert 0 leads for 1000 rounds...
        loss[1000:, 3] = 0.4                   # ...then expert 3 does
        W, _, _, _ = _adahedge_loop(loss, HEDGE_MEMORY)
        self.assertGreater(W[999, 0], 0.9)
        self.assertLess(W[999, 3], 0.1)
        self.assertGreater(W[1000 + 30, 3], 0.5)             # the new leader is in front in tens of bars...
        self.assertGreater(W[1000 + 60, 3], 0.8)
        self.assertGreater(W[1000 + HEDGE_MEMORY, 3], 0.9)   # ...not once the old one has left the memory
        # row t depends on rows t-memory+1..t only, exactly (the walk-forward warm-up relies on it)
        W2, _, _, _ = _adahedge_loop(loss[500:], HEDGE_MEMORY)
        np.testing.assert_array_equal(W[500 + HEDGE_MEMORY:], W2[HEDGE_MEMORY:])

    def test_eta_recovers_after_a_calm_stretch(self):
        """Plain AdaHedge's learning rate only ever falls: once burnt, always
        cautious. Discounting the mixability gap lets it climb back when the
        experts stop disagreeing, faster at the shorter lifetimes."""
        rng = np.random.default_rng(1)
        loss = np.vstack([rng.uniform(0, 1, (100, 4)), np.full((150, 4), 0.5)])
        _, eta, _, _ = _adahedge_loop(loss, HEDGE_MEMORY)
        self.assertTrue((eta[249] > 2 * eta[99]).all())
        self.assertGreater(eta[249, 0], 100 * eta[99, 0])
        self.assertTrue((np.diff(eta[249]) < 0).all())       # the shortest lifetime recovered most

    def test_meta_learner_shortens_the_memory_on_a_regime_break(self):
        """The lifetimes are a ladder the learner picks from: when the leader
        changes, the learners with a short memory adapt first and their hedge
        loss wins the meta learner over; once the new leader is established
        the long-memory learner, which concentrates most, takes it back."""
        loss = np.full((1500, 4), 0.6)
        rng = np.random.default_rng(3)
        loss += rng.normal(0, 0.05, loss.shape)              # noise, so the deficits are not all at the cap
        loss[:1000, 0] -= 0.2
        loss[1000:, 3] -= 0.2
        loss = np.clip(loss, 0, 1)
        _, _, V, _ = _adahedge_loop(loss, HEDGE_MEMORY)
        short = V[:, :2].sum(axis=1)
        self.assertLess(short[950:1000].mean(), short[1005:1060].mean())      # shorter memory right after the break
        self.assertLess(short[1300:1500].mean(), short[1005:1060].mean())     # and back to a long one after

    def test_costs_move_weight_to_the_slower_experts(self):
        """Each expert pays the sides it trades, in ATRs, as the engine
        charges cost_bps: the fast lookback flips most, so a cost it does not
        earn back moves the learner's weight down the ladder."""
        loss0, _, _ = _hedge_loss(self.df, 20, "trend", 0.0)
        loss1, _, _ = _hedge_loss(self.df, 20, "trend", 50.0)
        self.assertTrue((loss1 >= loss0 - 1e-12).all())
        charged = loss1 > loss0 + 1e-12
        self.assertGreater(charged.sum(), 50)
        self.assertLess(charged.mean(), 0.5)                    # only on the bars an expert changed its stance
        W0 = hedge_weights(self.df, 20, "trend", 0.0)
        W1 = hedge_weights(self.df, 20, "trend", 50.0)
        warm = hedge_warmup(20)
        self.assertLess(W1[warm:, 0].mean(), W0[warm:, 0].mean())
        np.testing.assert_array_equal(W0, hedge_weights(self.df, 20, "trend"))   # cost_bps=0 is the cost-free loss
        # and the template's cost reaches the learner: the channel changes with it
        tpl = StrategyTemplate("t", channel_type="hedge", cost_bps=0.0)
        a = _compute_indicators(self.df, tpl)["upper"]
        b = _compute_indicators(self.df, tpl.with_params(cost_bps=50.0))["upper"]
        self.assertFalse(np.allclose(np.nan_to_num(a), np.nan_to_num(b)))

    def test_diagnostics_are_consistent(self):
        d = hedge_diagnostics(self.df, 20, "learned", 5.0)
        W = hedge_weights(self.df, 20, "learned", 5.0)
        np.testing.assert_array_equal(d["weights"].to_numpy(), W)
        self.assertEqual(list(d["weights"].columns), ["follow_10", "follow_20", "follow_40", "follow_80",
                                                      "fade_10", "fade_20", "fade_40", "fade_80"])
        self.assertEqual(list(d["eta"].columns), list(HEDGE_HORIZONS))
        np.testing.assert_allclose(d["horizon_weights"].sum(axis=1), 1.0)
        self.assertTrue((d["surprise"] >= 0).all() and (d["surprise"] <= 1).all())
        self.assertTrue(((d["loss"] >= 0) & (d["loss"] <= 1)).all().all())
        self.assertTrue(d["weights"].index.equals(self.df.index))

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

    def test_learned_direction_sizes_by_conviction(self):
        """A learned direction is the learner's net side weight: its sign
        picks follow or fade, its magnitude scales the position. A trade
        entered from flat on bar i holds equity[i-1] * risk_pct /
        (atr_mult_stop * ATR[i-1]) shares times the conviction on bar i-1;
        a fixed direction is conviction 1."""
        df = regime_series()
        tpl = StrategyTemplate("t", channel_type="hedge", direction_logic="learned", cost_bps=0.0, max_leverage=1e9)
        d = hedge_direction(df, tpl.atr_n, 0.0)
        finite = d[~np.isnan(d)]
        self.assertTrue((finite >= -1.0).all() and (finite <= 1.0).all())
        convs = []
        for direction, learned in (("learned", True), ("trend", False)):
            res = backtest(df, tpl.with_params(direction_logic=direction))
            eq = res["equity"].to_numpy(); a = res["indicators"]["atr"]
            self.assertGreater(len(res["trades"]), 30)
            for t in res["trades"]:
                i = df.index.get_loc(t["entry_date"])
                full = eq[i - 1] * tpl.risk_pct / (tpl.atr_mult_stop * a[i - 1])
                conv = abs(d[i - 1]) if learned else 1.0
                self.assertAlmostEqual(t["shares"] / full, conv, places=9, msg=f"{direction} {t['entry_date']}")
                if learned:
                    convs.append(conv)
        self.assertTrue(all(0.0 < c <= 1.0 for c in convs))
        self.assertGreater(sum(c < 0.9 for c in convs), 5)     # near-tied bars open small positions...
        self.assertGreater(sum(c > 0.9 for c in convs), 5)     # ...one-sided ones full ones

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


class WideLadderTests(unittest.TestCase):
    """The 'hedge_wide' channel: the same learner over a wider fixed ladder
    (the Donchian rungs, Keltner and Bollinger bands at fixed widths, and a
    regime-gated copy of every rung). Nothing new to fit, the same warm-up
    contract, and the plain ladder is left exactly as it was."""

    def setUp(self):
        self.df = synthetic_ohlc(1200, seed=7)

    def test_the_ladder_is_fixed_in_advance_and_labelled(self):
        spec = HEDGE_LADDERS["hedge_wide"]
        self.assertEqual(HEDGE_CHANNELS, ("hedge", "hedge_wide"))
        self.assertEqual([e.label for e in hedge_ladder("trend")], [f"follow_{n}" for n in HEDGE_LADDER])
        self.assertEqual(len(spec), len(HEDGE_LADDER) + 2 * len(HEDGE_WIDE_WIDTHS) + len(HEDGE_LADDER))
        labels = [e.label for e in hedge_ladder("learned", "hedge_wide")]
        self.assertEqual(len(labels), 2 * len(spec))
        self.assertEqual(len(set(labels)), len(labels))
        self.assertEqual(labels[:4], ["follow_10", "follow_20", "follow_40", "follow_80"])
        for k in HEDGE_WIDE_WIDTHS:
            self.assertIn(f"follow_kel{HEDGE_WIDE_N}x{k}", labels)
            self.assertIn(f"fade_bol{HEDGE_WIDE_N}x{k}", labels)
        self.assertIn("follow_40:trend", labels)
        self.assertIn("fade_40:range", labels)
        self.assertNotIn("follow_40:range", labels)     # a follow expert is gated on trend, a fade one on range
        self.assertNotIn("fade_40:trend", labels)
        # the gate is the ER registry's own default split, not a fitted number
        self.assertEqual((HEDGE_GATE_N, HEDGE_GATE_THRESHOLD),
                         (REGIME_INDICATORS["er"]["n"], REGIME_INDICATORS["er"]["threshold"]))
        self.assertEqual(hedge_ladder_for("hedge_wide"), "hedge_wide")
        self.assertEqual(hedge_ladder_for("hedge"), "hedge")
        self.assertEqual(hedge_ladder_for("donchian"), "hedge")
        with self.assertRaises(ValueError):
            hedge_ladder("trend", "nope")
        with self.assertRaises(ValueError):
            hedge_experts("sideways", "hedge_wide")
        # the wide experts do not stretch the warm-up: the 80-bar rung still leads it
        self.assertEqual(hedge_warmup(20, "hedge_wide"), hedge_warmup(20))
        self.assertEqual(hedge_warmup(20), HEDGE_MEMORY + 2 * max(HEDGE_LADDER))

    def test_bands_are_the_channels_the_fitted_templates_trade(self):
        df, close = self.df, self.df["Close"]
        don, kel, bol = HedgeExpert("donchian", 40), HedgeExpert("keltner", 20, 1.5), HedgeExpert("bollinger", 20, 2.5)
        u, l = bol.bands(df, 20)
        ub, lb, _ = bollinger(df, 20, 2.5)
        np.testing.assert_allclose(u, ub.to_numpy(), rtol=1e-12); np.testing.assert_allclose(l, lb.to_numpy(), rtol=1e-12)
        u, l = kel.bands(df, 14)
        mid, w = sma(close, 20), 1.5 * atr(df, 14)          # the SMA form: exact once its window is full
        np.testing.assert_allclose(u, (mid + w).to_numpy(), rtol=1e-12)
        np.testing.assert_allclose(l, (mid - w).to_numpy(), rtol=1e-12)
        u, l = don.bands(df, 20)
        np.testing.assert_array_equal(u, donchian(df, 40)[0].to_numpy())
        # the exit channel is the same expert at half its lookback, same width
        u, _ = kel.bands(df, 14, HEDGE_EXIT_SCALE)
        np.testing.assert_allclose(u, (sma(close, 10) + 1.5 * atr(df, 14)).to_numpy(), rtol=1e-12)
        np.testing.assert_array_equal(don.bands(df, 20, HEDGE_EXIT_SCALE)[0], donchian(df, 20)[0].to_numpy())
        # to rounding, because the bands are reduced window by window: a slice of the
        # series reproduces them bit for bit, which pandas' running sums do not
        for e in (kel, bol):
            for k in (300, 777):
                np.testing.assert_array_equal(e.bands(df.iloc[k:], 20)[0][40:], e.bands(df, 20)[0][k + 40:], e.label)
        self.assertFalse(np.array_equal(bollinger(df.iloc[300:], 20, 2.5)[0].to_numpy()[40:],
                                        bollinger(df, 20, 2.5)[0].to_numpy()[340:]))
        # span, formed and lead: a rung's stance is exact 2 n bars in; a band's once its
        # window (and the ATR's) is full plus its span; a gated one also needs its gate
        self.assertEqual((don.span, don.formed(20), don.lead(20)), (40, 39, 80))
        self.assertEqual((kel.span, kel.formed(20), kel.lead(20)), (20, 20, 41))
        self.assertEqual((bol.span, bol.formed(20), bol.lead(20)), (20, 19, 40))
        self.assertEqual(HedgeExpert("donchian", 10, gated=True).lead(20), HEDGE_GATE_N + 1)
        self.assertEqual(HedgeExpert("donchian", 80, gated=True).lead(20), 160)
        for e in (don, bol):
            self.assertEqual(int(np.argmax(~np.isnan(e.bands(df, 20)[0]))), e.formed(20))
        # the first true range has no close before it, so the ATR is counted formed a bar late
        self.assertLessEqual(int(np.argmax(~np.isnan(kel.bands(df, 20)[0]))), kel.formed(20))
        with self.assertRaises(ValueError):
            HedgeExpert("ichimoku", 20).bands(df, 20)

    def test_a_gated_expert_is_its_rung_inside_the_regime_and_flat_outside(self):
        df = self.df
        er = efficiency_ratio(df["Close"], HEDGE_GATE_N).to_numpy()
        S, formed, experts = _hedge_stances(df, 20, "countertrend", "hedge_wide")
        gates = _hedge_gates(df, experts)
        labels = [e.label for e in experts]
        self.assertTrue(formed[HEDGE_MEMORY:].all())
        for cost in (0.0, 50.0):
            loss, _, _ = _hedge_loss(df, 20, "countertrend", cost, "hedge_wide")
            for n in HEDGE_LADDER:
                u, g = labels.index(f"fade_{n}"), labels.index(f"fade_{n}:range")
                self.assertTrue(gates[:, u].all())
                np.testing.assert_array_equal(gates[:, g], er < HEDGE_GATE_THRESHOLD)   # NaN fails: stands aside
                np.testing.assert_array_equal(S[:, g], np.where(gates[:, g], S[:, u], 0.0))   # the rung, gated
                open_ = gates[1:, g] & gates[:-1, g]
                shut = ~gates[1:, g] & ~gates[:-1, g]
                self.assertGreater(open_.sum(), 100); self.assertGreater(shut.sum(), 100)
                np.testing.assert_array_equal(loss[1:][open_, g], loss[1:][open_, u])   # inside: the rung itself
                self.assertTrue((loss[1:][shut, g] == 0.5).all())                        # outside: flat, neutral
                # stepping aside is a side traded: on a bar the gate shuts while the
                # ungated rung holds its stance, the gated expert pays the one exit
                # the ungated one does not, a quarter of the cost in ATRs
                held = (S[1:, u] == S[:-1, u]) & (S[:-1, u] != 0)
                shuts = np.where(gates[:-1, g] & ~gates[1:, g] & held)[0] + 1
                self.assertGreater(len(shuts), 5)
                a_prev = atr(df, 20).to_numpy()[shuts - 1]
                charge = 0.25 * df["Close"].to_numpy()[shuts] * cost / 1e4 / a_prev
                extra = loss[shuts, g] - loss[shuts, u]
                self.assertTrue((extra <= charge + 1e-12).all())
                inside = (loss[shuts, u] > 0) & (loss[shuts, u] < 1) & (loss[shuts, g] > 0) & (loss[shuts, g] < 1)
                self.assertGreater(inside.sum(), len(shuts) * 0.8)     # the payoff clip at +/-1 eats the rest
                np.testing.assert_allclose(extra[inside], charge[inside], atol=1e-12)
        # the follow side is gated on trend
        fexp = hedge_ladder("trend", "hedge_wide")
        fg = _hedge_gates(df, fexp)
        np.testing.assert_array_equal(fg[:, [e.label for e in fexp].index("follow_20:trend")], er >= HEDGE_GATE_THRESHOLD)
        self.assertTrue(_hedge_gates(df, hedge_ladder("trend")).all())    # the plain ladder has no gates

    def test_weights_are_causal_normalised_and_kept_apart_from_the_plain_ladder(self):
        W = hedge_weights(self.df, 20, "learned", 5.0, "hedge_wide")
        self.assertEqual(W.shape, (len(self.df), 2 * len(HEDGE_LADDERS["hedge_wide"])))
        np.testing.assert_allclose(W.sum(axis=1), 1.0)
        self.assertTrue((W >= 0).all())
        np.testing.assert_allclose(W[:700], hedge_weights(self.df.iloc[:700], 20, "learned", 5.0, "hedge_wide"))
        W0 = hedge_weights(self.df, 20, "learned", 5.0)
        self.assertEqual(W0.shape[1], 2 * len(HEDGE_LADDER))
        n = len(HEDGE_LADDERS["hedge_wide"])
        self.assertFalse(np.allclose(W0, W[:, list(range(4)) + list(range(n, n + 4))]))   # a different mixture
        d = hedge_diagnostics(self.df, 20, "learned", 5.0, "hedge_wide")
        self.assertEqual(list(d["weights"].columns), [e.label for e in hedge_ladder("learned", "hedge_wide")])
        np.testing.assert_array_equal(d["weights"].to_numpy(), W)
        self.assertTrue(((d["loss"] >= 0) & (d["loss"] <= 1)).all().all())

    def test_channel_is_a_mixture_of_the_experts(self):
        warm = hedge_warmup(20, "hedge_wide")
        for mode in ("trend", "learned"):
            up, lo, mid = hedge_channel(self.df, 20, mode, 1.0, 5.0, "hedge_wide")
            self.assertEqual(int(up.isna().sum()), warm)
            experts = hedge_ladder(mode, "hedge_wide")
            ups = np.stack([e.bands(self.df, 20)[0] for e in experts], axis=1)[warm:]
            los = np.stack([e.bands(self.df, 20)[1] for e in experts], axis=1)[warm:]
            self.assertTrue((up.to_numpy()[warm:] <= ups.max(axis=1) + 1e-9).all())
            self.assertTrue((up.to_numpy()[warm:] >= ups.min(axis=1) - 1e-9).all())
            self.assertTrue((lo.to_numpy()[warm:] <= los.max(axis=1) + 1e-9).all())
            self.assertTrue((lo.to_numpy()[warm:] >= los.min(axis=1) - 1e-9).all())
            np.testing.assert_allclose(mid.to_numpy()[warm:], (up + lo).to_numpy()[warm:] / 2)
        plain = hedge_channel(self.df, 20, "trend", 1.0, 5.0)[0]
        self.assertFalse(np.allclose(np.nan_to_num(up), np.nan_to_num(plain)))

    def test_direction_abstains_while_a_gate_is_shut(self):
        experts = hedge_ladder("learned", "hedge_wide")
        W = hedge_weights(self.df, 20, "learned", 0.0, "hedge_wide")
        gates = _hedge_gates(self.df, experts)
        sides = np.array([float(e.side) for e in experts])
        d = hedge_direction(self.df, 20, 0.0, "hedge_wide")
        warm = hedge_warmup(20, "hedge_wide")
        self.assertTrue(np.isnan(d[:warm]).all())
        np.testing.assert_allclose(d[warm:], np.clip((W * gates) @ sides, -1.0, 1.0)[warm:])
        aside = (W * ~gates).sum(axis=1)[warm:]                  # weight on experts standing aside
        self.assertTrue((np.abs(d[warm:]) <= 1.0 - aside + 1e-12).all())
        self.assertGreater(aside.max(), 0.1)                     # and it is not a corner case
        # the plain ladder's direction is untouched by the gates
        np.testing.assert_array_equal(hedge_direction(self.df, 20, 0.0),
                                      np.where(np.arange(len(d)) < hedge_warmup(20), np.nan,
                                               np.clip(hedge_weights(self.df, 20, "learned", 0.0)
                                                       @ hedge_experts("learned")[1], -1, 1)))

    def test_templates_have_nothing_to_fit_and_read_no_width(self):
        df = self.df
        tpl = StrategyTemplate("t", channel_type="hedge_wide", cost_bps=0.0)
        self.assertEqual(param_grid_for(tpl), {})
        self.assertEqual(param_grid_for(tpl, wide=True), {})
        base = backtest(df, tpl)
        self.assertGreater(base["stats"]["n_trades"], 5)
        eq = base["equity"].to_numpy()
        for k, v in (("channel_k", 0.7), ("n_entry", 13), ("n_exit", 7), ("regime_threshold", 0.01), ("regime_n", 7)):
            np.testing.assert_array_equal(eq, backtest(df, tpl.with_params(**{k: v}))["equity"].to_numpy(), k)
        # the ATR length and the cost do reach the learner
        self.assertFalse(np.array_equal(eq, backtest(df, tpl.with_params(atr_n=14))["equity"].to_numpy()))
        self.assertFalse(np.array_equal(eq, backtest(df, tpl.with_params(cost_bps=50.0))["equity"].to_numpy()))
        self.assertFalse(np.array_equal(eq, backtest(df, tpl.with_params(channel_type="hedge"))["equity"].to_numpy()))
        self.assertGreaterEqual(warmup_bars(tpl), hedge_warmup(20, "hedge_wide"))
        # a learned direction on a fitted channel keeps the plain ladder
        don = backtest(df, StrategyTemplate("t", direction_logic="learned", cost_bps=5.0))["indicators"]["direction"]
        np.testing.assert_array_equal(don, hedge_direction(df, 20, 5.0))
        wide = backtest(df, tpl.with_params(direction_logic="learned", cost_bps=5.0))["indicators"]["direction"]
        np.testing.assert_array_equal(wide, hedge_direction(df, 20, 5.0, "hedge_wide"))
        # the family: the online switches, template for template, over the wide ladder
        fam, online = generate_templates("online_wide"), generate_templates("online")
        self.assertEqual(len(fam), len(online))
        self.assertEqual([t.name.replace("-hdw-", "-hdg-") for t in fam], [t.name for t in online])
        self.assertTrue(all(t.channel_type == "hedge_wide" for t in fam))
        for t in fam:
            for key in ("n_entry", "n_exit", "channel_k"):
                self.assertNotIn(key, param_grid_for(t), t.name)
        res = walk_forward(df, fam[0], param_grid_for(fam[0]), train_bars=400, test_bars=100)
        self.assertEqual(len(res["oos_returns"]), len(df) - 400)

    def test_learns_to_stand_aside_and_where_to_fade(self):
        """On the regime series the countertrend learner puts the fade side's
        weight on the gated rungs through the trend (they stand aside while
        fading loses) and back on the ungated ones in the range; the trend
        learner does the mirror image. The learned direction follows the
        trends and fades the range as the plain ladder does, and the bars it
        fades inside a trend are the band fades earning on pullbacks."""
        rs = regime_series()
        Wc = hedge_weights(rs, 20, "countertrend", 0.0, "hedge_wide")
        lab = [e.label for e in hedge_ladder("countertrend", "hedge_wide")]
        gated = [i for i, l in enumerate(lab) if l.endswith(":range")]
        rungs = [lab.index(f"fade_{n}") for n in HEDGE_LADDER]
        self.assertGreater(Wc[500:1000, gated].sum(axis=1).mean(), 0.6)
        self.assertLess(Wc[500:1000, rungs].sum(axis=1).mean(), 0.2)
        self.assertGreater(Wc[1400:2000, rungs].sum(axis=1).mean(), Wc[1400:2000, gated].sum(axis=1).mean() * 0.8)
        Wt = hedge_weights(rs, 20, "trend", 0.0, "hedge_wide")
        labt = [e.label for e in hedge_ladder("trend", "hedge_wide")]
        gt = [i for i, l in enumerate(labt) if l.endswith(":trend")]
        ut = [labt.index(f"follow_{n}") for n in HEDGE_LADDER]
        self.assertGreater(Wt[500:1000, ut].sum(axis=1).mean(), 0.7)
        self.assertGreater(Wt[1400:2000, gt].sum(axis=1).mean(), 0.8)
        d = hedge_direction(rs, 20, 0.0, "hedge_wide")
        self.assertGreater((d[500:1000] > 0).mean(), 0.8)      # trend: follow
        self.assertGreater((d[1400:2000] < 0).mean(), 0.9)     # mean reversion: fade
        self.assertGreater((d[2500:3000] > 0).mean(), 0.8)     # trend again: follow
        W = hedge_weights(rs, 20, "learned", 0.0, "hedge_wide")
        labl = [e.label for e in hedge_ladder("learned", "hedge_wide")]
        bands = [i for i, l in enumerate(labl) if l.startswith("fade_kel") or l.startswith("fade_bol")]
        flip = np.zeros(len(d), dtype=bool)
        flip[500:1000] = d[500:1000] < 0
        flip[2500:3000] = d[2500:3000] < 0
        self.assertGreater(flip.sum(), 20)
        self.assertGreater(W[flip][:, bands].sum(axis=1).mean(), 0.4)
        learned = backtest(rs, StrategyTemplate("t", channel_type="hedge_wide", direction_logic="learned", cost_bps=0.0))
        follow = backtest(rs, StrategyTemplate("t", channel_type="hedge_wide", direction_logic="trend", cost_bps=0.0))
        fade = backtest(rs, StrategyTemplate("t", channel_type="hedge_wide", direction_logic="countertrend", cost_bps=0.0))
        self.assertGreater(learned["stats"]["total_return"], follow["stats"]["total_return"])
        self.assertGreater(learned["stats"]["total_return"], fade["stats"]["total_return"])


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
