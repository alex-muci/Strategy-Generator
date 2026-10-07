"""
Sanity tests for the engine and the robustness toolkit.

Run with:   python -m unittest discover -s tests -v
(no pytest dependency needed; pytest also works if installed)
"""

from __future__ import annotations
import os
import sys
import unittest
from dataclasses import replace
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
    hedge_learner, HEDGE_SLOW_MEMORY, HEDGE_SLOW_HORIZONS,
    HEDGE_SPLIT_FOLLOW, HEDGE_SPLIT_FADE, HEDGE_SPLIT_LEARNERS, hedge_group_weights,
    HEDGE_WIDE_K, hedge_position, hedge_position_sized, hedge_scored_sides, _allowed, _hedge_stances,
    hedge_active, _window_mean, EXIT_STYLES, _hedge_cached,
    sma, atr, hedge_stance, _buffered_level, STANCE_BUFFER, STANCE_SETTLE,
    hedge_rungs, set_hedge_share, HEDGE_SHARES, HEDGE_SHARE_SWITCHES,
)
import strategy as S  # noqa: E402
from generator import generate_templates, param_grid_for  # noqa: E402
from contextlib import contextmanager  # noqa: E402


@contextmanager
def hedge_share(share):
    """Run the block with every hedge learner forgetting by `share`."""
    old = S.HEDGE_SHARE
    set_hedge_share(share)
    try:
        yield
    finally:
        set_hedge_share(old)

# the online family as it was before the redesign (every direction on the plain ladder, six regime
# filters, an SMA bias or none): the tests below check the plain ladder through it
OLD_ONLINE = dict(
    direction_logics=["trend", "countertrend", "learned"], channel_types=["hedge"],
    entry_styles=["stop", "close_confirm"],
    regimes=[("er", "none"), ("er", "trend_only"), ("er", "range_only"),
             ("vr", "trend_only"), ("vr", "range_only"), ("chop", "range_only")],
    bias_filters=["none", "sma"],
)
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


def _flat_entries(df, res):
    """(bar, trade) for every trade opened from a flat book, so the cash the
    engine sized on is the previous bar's equity."""
    exits = {t["exit_date"] for t in res["trades"]}
    out = []
    for t in res["trades"]:
        i = df.index.get_loc(t["entry_date"])
        if df.index[i] not in exits or t["exit_date"] == t["entry_date"]:
            out.append((i, t))
    return out


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

    def test_jit_kernel_matches_pure_python_on_futures_paths(self):
        """The same, on the paths the share sweep above never reaches: a
        margined future with per-unit and roll costs on gappy prices (rolls,
        a target the open gaps through, ruin to a deficit, the last-bar
        ruin), with and without fixed capital, with the order log on."""
        import strategy as S
        if not S.HAVE_NUMBA:
            self.skipTest("numba not installed")
        rng = np.random.default_rng(3)
        df = synthetic_ohlc(600, seed=9).copy()
        gap = rng.random(len(df)) < 0.03                   # 3 % of bars gap 5 % either way
        f = np.where(gap, np.exp(rng.normal(0, 0.05, len(df))), 1.0).cumprod()
        for c in ("Open", "High", "Low", "Close"):
            df[c] = df[c] * f
        # a crash and a spike beyond any margin: longs are ruined by one, shorts by the other
        for k, jump in ((250, 0.4), (420, 2.5), (len(df) - 1, 0.5)):
            df.iloc[k:, [df.columns.get_loc(c) for c in ("Open", "High", "Low", "Close")]] *= jump
        df["Roll"] = (rng.random(len(df)) < 0.05).astype(float)
        fast = S._bar_loop_fast
        base = dict(point_value=50.0, margin_per_unit=4000.0, cost_bps=0.0, cost_per_unit=2.5,
                    roll_cost_per_unit=40.0, risk_pct=0.05, max_leverage=2.0)
        sample = [t.with_params(**base) for t in generate_templates("full")[::241]]
        sample += [t.with_params(whole_units=True) for t in sample[::3]]
        try:
            for tpl in sample:
                for fixed in (False, True):
                    S._bar_loop_fast = S._bar_loop
                    slow = backtest(df, tpl, first_trade_bar=60, log_orders=True, fixed_capital=fixed)
                    S._bar_loop_fast = fast
                    quick = backtest(df, tpl, first_trade_bar=60, log_orders=True, fixed_capital=fixed)
                    np.testing.assert_allclose(slow["equity"].values, quick["equity"].values, rtol=1e-12, atol=1e-6,
                                               err_msg=f"{tpl.name} fixed={fixed}")
                    pd.testing.assert_frame_equal(slow["orders"], quick["orders"], check_exact=False, rtol=1e-12)
                    self.assertEqual([t["reason"] for t in slow["trades"]], [t["reason"] for t in quick["trades"]])
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
        # the engine's realized vol on a cash asset: per-bar std of pct changes
        return df["Close"].pct_change().rolling(n).std()

    @staticmethod
    def _rvol_pts(df, n):
        # ... and on a future or spread (a margin given): std of close DIFFERENCES (price points)
        return df["Close"].diff().rolling(n).std()

    def _flat_entries(self, res):
        return _flat_entries(self.df, res)

    def test_off_is_byte_identical_and_the_lookback_is_inert(self):
        tpl = StrategyTemplate("t")
        a = backtest(self.df, tpl)
        b = backtest(self.df, tpl.with_params(vol_target=0.0, vol_target_n=17))
        np.testing.assert_array_equal(a["equity"].values, b["equity"].values)
        self.assertNotIn("rvol", a["indicators"])
        self.assertNotIn("rvol", b["indicators"])

    def test_entry_dollar_vol_is_the_target_times_equity(self):
        """The position's dollar volatility is the target x equity under both
        rules: a cash asset's units x price x pct vol, a margined future's
        units x point_value x sigma_points (whatever its price level)."""
        import strategy as S
        target_bar = 0.15 / np.sqrt(S.periods_per_year())
        cash = StrategyTemplate("t", vol_target=0.15, vol_target_n=60, cost_bps=0.0, max_leverage=1e9)
        lot = cash.with_params(point_value=1000.0, margin_per_unit=1.0)
        for tpl, rv, per_unit in ((cash, self._rvol(self.df, 60), lambda t, i, rv: t["entry_price"] * rv.iloc[i - 1]),
                                  (lot, self._rvol_pts(self.df, 60), lambda t, i, rv: 1000.0 * rv.iloc[i - 1])):
            res = backtest(self.df, tpl, first_trade_bar=self.K)
            eq = res["equity"]
            checked = 0
            for i, t in self._flat_entries(res):
                self.assertAlmostEqual(t["shares"] * per_unit(t, i, rv) / eq.iloc[i - 1], target_bar, places=8,
                                       msg=f"{tpl.margin_per_unit} {t['entry_date']}")
                checked += 1
            self.assertGreater(checked, 5)

    def test_a_target_equal_to_the_realized_vol_gives_unit_notional(self):
        """The whole point: at the asset's own (pct) vol a cash asset is held at
        exactly 1x notional, the buy-and-hold scale."""
        import strategy as S
        base = StrategyTemplate("t", vol_target=0.15, vol_target_n=60, cost_bps=0.0, max_leverage=1e9)
        first = backtest(self.df, base, first_trade_bar=self.K)
        i0, t0 = self._flat_entries(first)[0]
        own_vol = float(self._rvol(self.df, 60).iloc[i0 - 1] * np.sqrt(S.periods_per_year()))
        res = backtest(self.df, base.with_params(vol_target=own_vol), first_trade_bar=self.K)
        np.testing.assert_array_equal(res["entries"], first["entries"])
        t = [tr for tr in res["trades"] if tr["entry_date"] == t0["entry_date"]][0]
        self.assertAlmostEqual(t["shares"] * t["entry_price"] / res["equity"].iloc[i0 - 1], 1.0, places=8)

    def test_a_cash_asset_is_sized_on_its_pct_vol_and_a_future_on_points(self):
        """Without a margin the rule is the pct-of-notional one shares have always
        used (units = equity x target / pct vol / fill price); with a margin it
        is the points rule. The two differ on a trending share, which is why
        the cash asset keeps the pct rule."""
        import strategy as S
        target_bar = 0.15 / np.sqrt(S.periods_per_year())
        tpl = StrategyTemplate("t", vol_target=0.15, vol_target_n=60, cost_bps=0.0, max_leverage=1e9)
        res = backtest(self.df, tpl, first_trade_bar=self.K)
        rv_pct = self._rvol(self.df, 60)
        eq = res["equity"]
        entries = self._flat_entries(res)
        self.assertGreater(len(entries), 5)
        for i, t in entries:
            old_units = eq.iloc[i - 1] * target_bar / rv_pct.iloc[i - 1] / t["entry_price"]
            self.assertAlmostEqual(t["shares"] / old_units, 1.0, places=10)
        fut = backtest(self.df, tpl.with_params(margin_per_unit=1.0), first_trade_bar=self.K)
        self.assertNotEqual([t["shares"] for t in fut["trades"]], [t["shares"] for t in res["trades"]])

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
        W, eta, _, _ = _adahedge_loop(loss, HEDGE_MEMORY, share="discount")
        self.assertEqual(W[5, 1], 0.0)                       # the precondition: written off completely
        self.assertAlmostEqual(W[5, 0], 1.0, places=12)
        self.assertTrue((eta[5] > 745.0).all())              # and exp(-eta * 1) underflows as well
        # the last round's mix loss is the written-off expert's ~3313c deficit
        # (it is the better of "leader loses 1" and "catch up 3313c, lose 0"),
        # the Hedge loss is 1, so delta gains ~1 - 3313c and expert 1 leads by
        # about that gap: weights near 2/3 and 1/3 (exactly so without the
        # discount), not 1 and 0
        np.testing.assert_allclose(W[6], [1 / 3, 2 / 3], atol=2e-2)
        # fixed share never writes an expert off (every rung keeps alpha / N on
        # it), and the same round still ends follow-the-leader
        W, eta, _, _ = _adahedge_loop(loss, HEDGE_MEMORY, share="fixed_share")
        self.assertGreater(W[5, 1], 0.0)
        self.assertTrue(np.isfinite(W).all() and np.isfinite(eta).all())
        self.assertGreater(W[6, 1], 0.5)

    def test_no_expert_is_written_off_for_good(self):
        """(Discounting.) A discounted deficit is bounded by the lifetime times the shortfall
        per bar, whatever the expert lost before, and the shortest lifetime
        on the ladder bounds it tightest: an expert that starts winning is
        back in front within tens of bars, and the meta learner, scoring
        the lifetimes on their own hedge loss, follows the one that moved."""
        loss = np.full((440, 4), 0.6)
        loss[:400, 0] = 0.4                    # 400 rounds of write-off...
        loss[400:, 3] = 0.4                    # ...then the trailer wins by the same edge
        W, _, V, _ = _adahedge_loop(loss, HEDGE_MEMORY, share="discount")
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
        """(Discounting.) Plain AdaHedge's learning rate only ever falls: once
        burnt, always cautious. Discounting the mixability gap lets it climb
        back when the experts stop disagreeing, faster at the shorter
        lifetimes. (Under fixed share the gap is not discounted: eta recovers
        when the burn leaves the memory.)"""
        rng = np.random.default_rng(1)
        loss = np.vstack([rng.uniform(0, 1, (100, 4)), np.full((150, 4), 0.5)])
        _, eta, _, _ = _adahedge_loop(loss, HEDGE_MEMORY, share="discount")
        self.assertTrue((eta[249] > 2 * eta[99]).all())
        self.assertGreater(eta[249, 0], 100 * eta[99, 0])
        self.assertTrue((np.diff(eta[249]) < 0).all())       # the shortest lifetime recovered most

    def test_meta_learner_shortens_the_memory_on_a_regime_break(self):
        """(Discounting.) The lifetimes are a ladder the learner picks from: when the leader
        changes, the learners with a short memory adapt first and their hedge
        loss wins the meta learner over; once the new leader is established
        the long-memory learner, which concentrates most, takes it back."""
        loss = np.full((1500, 4), 0.6)
        rng = np.random.default_rng(3)
        loss += rng.normal(0, 0.05, loss.shape)              # noise, so the deficits are not all at the cap
        loss[:1000, 0] -= 0.2
        loss[1000:, 3] -= 0.2
        loss = np.clip(loss, 0, 1)
        _, _, V, _ = _adahedge_loop(loss, HEDGE_MEMORY, share="discount")
        short = V[:, :2].sum(axis=1)
        self.assertLess(short[950:1000].mean(), short[1005:1060].mean())      # shorter memory right after the break
        self.assertLess(short[1300:1500].mean(), short[1005:1060].mean())     # and back to a long one after

    def test_fixed_share_is_the_default_and_a_switch(self):
        """Fixed share is the default way to forget; discounting is the other
        setting, and the two give different learners on the same losses.
        hedge_rungs maps a ladder to its rungs: the lifetimes when
        discounting, the expected switches per memory under fixed share."""
        self.assertEqual(S.HEDGE_SHARE, "fixed_share")
        self.assertEqual(set(HEDGE_SHARES), {"fixed_share", "discount"})
        g, a, gm, r = hedge_rungs(HEDGE_MEMORY, HEDGE_HORIZONS, "fixed_share")
        np.testing.assert_array_equal(g, 1.0)
        np.testing.assert_allclose(a, np.array(HEDGE_SHARE_SWITCHES) / HEDGE_MEMORY)
        np.testing.assert_allclose(r, HEDGE_MEMORY / np.array(HEDGE_SHARE_SWITCHES))
        g, a, gm2, r = hedge_rungs(HEDGE_MEMORY, HEDGE_HORIZONS, "discount")
        np.testing.assert_allclose(g, 1.0 - 1.0 / np.array(HEDGE_HORIZONS))
        np.testing.assert_array_equal(a, 0.0)
        self.assertEqual(gm, gm2)                                  # the meta learner is discounted alike
        with self.assertRaises(ValueError):
            hedge_rungs(HEDGE_MEMORY, HEDGE_HORIZONS, "both")
        with self.assertRaises(ValueError):          # alpha = m / memory must stay below 1
            hedge_rungs(6, HEDGE_HORIZONS, "fixed_share")
        with self.assertRaises(ValueError):
            set_hedge_share("discounted")
        self.assertEqual(S.HEDGE_SHARE, "fixed_share")
        rng = np.random.default_rng(2)
        loss = rng.uniform(0, 1, (400, 4))
        np.testing.assert_array_equal(_adahedge_loop(loss)[0], _adahedge_loop(loss, share="fixed_share")[0])
        self.assertGreater(np.abs(_adahedge_loop(loss, share="discount")[0] - _adahedge_loop(loss)[0]).max(), 0.01)
        # everything built on the learner follows the setting, cache included
        df = self.df
        W = hedge_weights(df, 20, "learned", 5.0)
        with hedge_share("discount"):
            Wd = hedge_weights(df, 20, "learned", 5.0)
            d = hedge_diagnostics(df, 20, "learned", 5.0)
            self.assertEqual(list(d["eta"].columns), list(HEDGE_HORIZONS))
        self.assertGreater(np.abs(W - Wd).max(), 0.01)
        np.testing.assert_array_equal(hedge_weights(df, 20, "learned", 5.0), W)
        self.assertEqual(list(hedge_diagnostics(df, 20, "learned", 5.0)["eta"].columns),
                         list(HEDGE_MEMORY / np.array(HEDGE_SHARE_SWITCHES)))

    def test_fixed_share_keeps_every_expert_and_revives_it_at_once(self):
        """Fixed share spreads alpha of the weight evenly after every update,
        so no expert's weight falls under (smallest alpha) / N, and a
        written-off expert that starts winning is in front within a few
        rounds, however long it trailed: the time to revive depends on
        log(1/alpha), not on the deficit."""
        rng = np.random.default_rng(0)
        loss = rng.uniform(0, 1, (600, 6))
        loss[:, 2] = np.clip(loss[:, 2] - 0.3, 0, 1)            # a strong leader the others trail
        W = _adahedge_loop(loss, HEDGE_MEMORY, share="fixed_share")[0]
        floor = min(HEDGE_SHARE_SWITCHES) / HEDGE_MEMORY / 6
        self.assertGreaterEqual(W.min(), floor * (1 - 1e-9))
        np.testing.assert_allclose(W.sum(axis=1), 1.0)
        lead = []
        for pre in (100, 240):
            loss = np.full((pre + 60, 4), 0.6)
            loss[:pre, 0] = 0.4
            loss[pre:, 3] = 0.4
            W = _adahedge_loop(loss, HEDGE_MEMORY, share="fixed_share")[0]
            Wd = _adahedge_loop(loss, HEDGE_MEMORY, share="discount")[0]
            self.assertGreater(W[pre - 1, 3], 1e-3)                # never written off ...
            self.assertLess(Wd[pre - 1, 3], 1e-60)                 # ... where discounting has
            k = int(np.argmax(W[pre - 1:, 3] > 0.5))
            lead.append(k)
            self.assertLess(k, int(np.argmax(Wd[pre - 1:, 3] > 0.5)))   # in front sooner than discounted
        self.assertLessEqual(max(lead), 10)
        self.assertLessEqual(abs(lead[0] - lead[1]), 1)            # whatever it lost before

    def test_fixed_share_with_alpha_zero_is_plain_adahedge(self):
        """A rung with alpha = 0 and no discount is the plain algorithm: eta
        only falls inside the window, and the weights are exp(-eta d) over
        the cumulative deficits."""
        rng = np.random.default_rng(4)
        loss = rng.uniform(0, 1, (120, 3))
        W, ETA, _, _, _ = S._hedge_fast(loss, 1000, np.ones(1), np.zeros(1), 1.0)
        self.assertTrue((np.diff(ETA[1:, 0]) <= 1e-12).all())
        L = loss.cumsum(axis=0)
        d = L - L.min(axis=1, keepdims=True)
        w = np.exp(-ETA[:, :1] * d)
        np.testing.assert_allclose(W[1:], (w / w.sum(axis=1, keepdims=True))[1:], rtol=1e-9)

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
        self.assertEqual(list(d["eta"].columns), list(hedge_rungs(HEDGE_MEMORY, HEDGE_HORIZONS)[3]))
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
        for tpl in generate_templates("online", **OLD_ONLINE):
            grid = param_grid_for(tpl)
            for key in ("n_entry", "n_exit", "channel_k"):
                self.assertNotIn(key, grid, tpl.name)
        tpl = generate_templates("online", **OLD_ONLINE)[0]
        self.assertEqual(param_grid_for(tpl), {})
        res = walk_forward(self.df, tpl, param_grid_for(tpl), train_bars=400, test_bars=100)
        self.assertEqual(len(res["oos_returns"]), len(self.df) - 400)
        self.assertGreaterEqual(warmup_bars(tpl), hedge_warmup(tpl.atr_n))
        self.assertGreaterEqual(warmup_bars(StrategyTemplate("t", direction_logic="learned")), hedge_warmup(20))

    def test_learns_the_trend_lookback_and_the_fade(self):
        # on a strong trend the trend learner concentrates on the slowest expert...
        df = trending_series()
        W = hedge_weights(df, 20, "trend")
        self.assertGreater(W[-500:, 2:].sum(axis=1).mean(), 0.8)   # the two slowest experts: 0.84 here; fixed
                                                                   # share keeps 16 % on the two fast ones
                                                                   # (discounted: 0.95, 5 %)
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
        # by Sharpe: the learned direction is sized by its conviction, which
        # fixed share keeps lower than discounting, so its total return
        # measures the size as much as the skill (Sharpe 1.83 against 1.10
        # and -0.60 here, 1.73 discounted). By total return the learned
        # direction no longer beats trend-only on this series under fixed
        # share (0.652 against 0.664; discounted 0.696 against 0.651).
        sr = lambda r: annualized_sharpe(r["returns"])   # noqa: E731
        self.assertGreater(sr(learned), sr(follow) + 0.3)
        self.assertGreater(sr(learned), sr(fade))
        # a trade keeps the exit logic of the side it was opened under
        reasons = {t["reason"] for t in learned["trades"]}
        self.assertTrue({"channel", "midline"} & reasons)


class WideLadderTests(unittest.TestCase):
    """The 'hedge_wide' channel: the same learner over a wider fixed ladder
    (the Donchian rungs and a Keltner band at every rung), scored on the
    legs the template can trade and sized by the committee's own position.
    Nothing new to fit, the same warm-up contract, and the plain ladder is
    left exactly as it was. The behavioural thresholds below are pinned on
    the series the design was checked on (the regime series, one synthetic
    decline, `synthetic_ohlc(1200, seed=7)`): regression tests of what it
    does there, not evidence that it does it elsewhere."""

    def setUp(self):
        self.df = synthetic_ohlc(1200, seed=7)

    def test_the_ladder_is_fixed_in_advance_and_labelled(self):
        spec = HEDGE_LADDERS["hedge_wide"]
        self.assertEqual(HEDGE_CHANNELS, ("hedge", "hedge_wide", "hedge_slow", "hedge_wide_slow", "hedge_split"))
        self.assertEqual([e.label for e in hedge_ladder("trend")], [f"follow_{n}" for n in HEDGE_LADDER])
        self.assertEqual(len(spec), 2 * len(HEDGE_LADDER))
        for mode in ("trend", "countertrend", "learned"):
            self.assertEqual(len(hedge_ladder(mode, "hedge_wide")), (2 if mode == "learned" else 1) * len(spec))
        labels = [e.label for e in hedge_ladder("learned", "hedge_wide")]
        self.assertEqual(len(set(labels)), len(labels))
        self.assertEqual(labels[:4], ["follow_10", "follow_20", "follow_40", "follow_80"])
        for n in HEDGE_LADDER:
            self.assertIn(f"follow_kel{n}x{HEDGE_WIDE_K:g}", labels)
            self.assertIn(f"fade_kel{n}x{HEDGE_WIDE_K:g}", labels)
        self.assertTrue(hedge_position_sized("hedge_wide") and not hedge_position_sized("hedge"))
        self.assertEqual(hedge_ladder_for("hedge_wide"), "hedge_wide")
        self.assertEqual(hedge_ladder_for("hedge"), "hedge")
        self.assertEqual(hedge_ladder_for("donchian"), "hedge")
        with self.assertRaises(ValueError):
            hedge_ladder("trend", "nope")
        with self.assertRaises(ValueError):
            hedge_position_sized("nope")
        with self.assertRaises(ValueError):
            hedge_experts("sideways", "hedge_wide")
        with self.assertRaises(ValueError):
            hedge_scored_sides("hedge_wide", "long")
        # the wide experts do not stretch the warm-up: the 80-bar rung still leads it
        self.assertEqual(hedge_warmup(20, "hedge_wide"), hedge_warmup(20))
        self.assertEqual(hedge_warmup(20), HEDGE_MEMORY + 2 * max(HEDGE_LADDER))

    def test_slow_ladders_are_the_same_experts_with_a_long_memory(self):
        self.assertEqual(HEDGE_LADDERS["hedge_slow"], HEDGE_LADDERS["hedge"])
        self.assertEqual(HEDGE_LADDERS["hedge_wide_slow"], HEDGE_LADDERS["hedge_wide"])
        self.assertEqual(hedge_learner("hedge"), (HEDGE_MEMORY, HEDGE_HORIZONS))
        self.assertEqual(hedge_learner("hedge_slow"), (HEDGE_SLOW_MEMORY, HEDGE_SLOW_HORIZONS))
        self.assertTrue(hedge_position_sized("hedge_wide_slow") and not hedge_position_sized("hedge_slow"))
        self.assertEqual(hedge_warmup(20, "hedge_slow"), HEDGE_SLOW_MEMORY + 2 * max(HEDGE_LADDER))
        # the fast ladders are what they were: the slow memory is the only difference
        df = synthetic_ohlc(1400, seed=3)
        for fast, slow in (("hedge", "hedge_slow"), ("hedge_wide", "hedge_wide_slow")):
            loss, _, _ = _hedge_loss(df, 20, "learned", 5.0, fast)
            loss_s, _, _ = _hedge_loss(df, 20, "learned", 5.0, slow)
            np.testing.assert_array_equal(loss, loss_s)
            np.testing.assert_array_equal(hedge_weights(df, 20, "learned", 5.0, fast),
                                          _adahedge_loop(loss, HEDGE_MEMORY)[0])
            np.testing.assert_array_equal(hedge_weights(df, 20, "learned", 5.0, slow),
                                          _adahedge_loop(loss, HEDGE_SLOW_MEMORY, HEDGE_SLOW_HORIZONS)[0])
        d = hedge_diagnostics(df, 20, "learned", 5.0, "hedge_wide_slow")
        self.assertEqual(list(d["eta"].columns), list(HEDGE_SLOW_MEMORY / np.array(HEDGE_SHARE_SWITCHES)))
        with hedge_share("discount"):
            d = hedge_diagnostics(df, 20, "learned", 5.0, "hedge_wide_slow")
            self.assertEqual(list(d["eta"].columns), list(HEDGE_SLOW_HORIZONS))
        # a window warmed on hedge_warmup bars matches the full-history run
        warm = hedge_warmup(20, "hedge_slow")
        full = hedge_weights(df, 20, "learned", 5.0, "hedge_slow")
        part = hedge_weights(df.iloc[300:], 20, "learned", 5.0, "hedge_slow")
        np.testing.assert_allclose(full[300 + warm:], part[warm:], atol=1e-12)

    def test_bands_are_the_channels_the_fitted_templates_trade(self):
        df, close = self.df, self.df["Close"]
        don, kel = HedgeExpert("donchian", 40), HedgeExpert("keltner", 20, 2.0)
        u, l = kel.bands(df, 14)
        mid, w = sma(close, 20), 2.0 * atr(df, 14)          # the SMA form: exact once its window is full
        np.testing.assert_allclose(u, (mid + w).to_numpy(), rtol=1e-12)
        np.testing.assert_allclose(l, (mid - w).to_numpy(), rtol=1e-12)
        u, l = don.bands(df, 20)
        np.testing.assert_array_equal(u, donchian(df, 40)[0].to_numpy())
        # the exit channel is the same expert at half its lookback, same width
        u, _ = kel.bands(df, 14, HEDGE_EXIT_SCALE)
        np.testing.assert_allclose(u, (sma(close, 10) + 2.0 * atr(df, 14)).to_numpy(), rtol=1e-12)
        np.testing.assert_array_equal(don.bands(df, 20, HEDGE_EXIT_SCALE)[0], donchian(df, 20)[0].to_numpy())
        # to rounding, because the SMA is reduced window by window: a slice of the
        # series reproduces the band bit for bit, which pandas' running sum does not
        for k in (300, 777):
            np.testing.assert_array_equal(kel.bands(df.iloc[k:], 20)[0][40:], kel.bands(df, 20)[0][k + 40:])
        x = close.to_numpy()
        m = _window_mean(x, 20)
        self.assertTrue(np.isnan(m[:19]).all() and not np.isnan(m[19:]).any())
        for k in range(1, 200, 13):
            np.testing.assert_array_equal(_window_mean(x[k:], 20)[19:], m[k + 19:])
        np.testing.assert_allclose(m, close.rolling(20).mean().to_numpy(), rtol=1e-12)
        # span, formed and lead: a rung's stance is exact 2 n bars in; a band's once its
        # window (and the ATR's) is full plus its span
        self.assertEqual((don.span, don.formed(20), don.lead(20)), (40, 39, 80))
        self.assertEqual((kel.span, kel.formed(20), kel.lead(20)), (20, 20, 41))
        self.assertEqual(HedgeExpert("keltner", 10, 2.0).lead(20), 31)
        self.assertEqual(HedgeExpert("keltner", 80, 2.0).lead(20), 160)
        self.assertEqual(int(np.argmax(~np.isnan(don.bands(df, 20)[0]))), don.formed(20))
        # the first true range has no close before it, so the ATR is counted formed a bar late
        self.assertLessEqual(int(np.argmax(~np.isnan(kel.bands(df, 20)[0]))), kel.formed(20))
        with self.assertRaises(ValueError):
            HedgeExpert("bollinger", 20, 2.0).bands(df, 20)

    def test_the_trade_weight_is_the_played_mixture_against_cash(self):
        """One more aggregation on top of the learner: the mixture it actually
        played, scored on its own realised loss, against the neutral loss.
        Unit rate, discounted at the longest lifetime, restarted with every
        window: a mixture that beats 0.5 by 0.1 for 20 rounds is 7:1 on, one
        that loses to it is a few percent, and a tie stays a tie."""
        T = 400
        for level, lo, hi in ((0.5, 0.5, 0.5), (0.4, 0.99, 1.0), (0.6, 0.0, 0.01)):
            loss = np.full((T, 4), level)
            W, eta, V, surprise, trade = _adahedge_loop(loss, HEDGE_MEMORY, with_trade=True)
            self.assertEqual(W.shape, (T, 4))
            self.assertTrue(((trade >= 0) & (trade <= 1)).all())
            self.assertTrue((trade[-100:] >= lo).all() and (trade[-100:] <= hi).all(), level)
            self.assertEqual(len(_adahedge_loop(loss, HEDGE_MEMORY)), 4)     # the plain call is what it was
        loss = np.full((T, 4), 0.5); loss[:, 1] = 0.4                        # one expert pays: the mixture finds it
        trade = _adahedge_loop(loss, HEDGE_MEMORY, with_trade=True)[4]
        self.assertGreater(trade[-1], 0.99)
        loss = np.full((T, 4), 0.5); loss[-20:, :] = 0.4                     # 20 rounds of 0.1 better than cash
        gm = 1.0 - 1.0 / max(HEDGE_HORIZONS)
        expect = 1.0 / (1.0 + np.exp(-0.1 * sum(gm ** k for k in range(20))))
        self.assertAlmostEqual(_adahedge_loop(loss, HEDGE_MEMORY, with_trade=True)[4][-1], expect, places=12)
        # on the series: in [0, 1], causal, and the regime series' trend learner is all
        # in through the trends and mostly out through the range
        d = hedge_diagnostics(self.df, 20, "learned", 5.0, "hedge_wide")
        u = d["trade_weight"].to_numpy()
        self.assertTrue(((u >= 0) & (u <= 1)).all())
        np.testing.assert_array_equal(u[:700], hedge_diagnostics(self.df.iloc[:700], 20, "learned", 5.0, "hedge_wide")["trade_weight"].to_numpy())
        rs = regime_series()
        ut = hedge_diagnostics(rs, 20, "trend", 5.0, "hedge_wide")["trade_weight"].to_numpy()
        uc = hedge_diagnostics(rs, 20, "countertrend", 5.0, "hedge_wide")["trade_weight"].to_numpy()
        self.assertGreater(ut[500:1000].mean(), 0.8); self.assertLess(ut[1400:2000].mean(), 0.4); self.assertGreater(ut[2500:3000].mean(), 0.8)
        self.assertLess(uc[500:1000].mean(), 0.4); self.assertGreater(uc[1400:2000].mean(), 0.8)
        # the position: the committee's stance times the trade weight, clipped
        S, _, _ = _hedge_stances(rs, 20, "trend", "hedge_wide")
        W = hedge_weights(rs, 20, "trend", 5.0, "hedge_wide")
        pos = hedge_position(rs, 20, "trend", 5.0, "hedge_wide")
        warm = hedge_warmup(20, "hedge_wide")
        self.assertTrue(np.isnan(pos[:warm]).all())
        np.testing.assert_allclose(pos[warm:], (ut * np.clip((W * S).sum(axis=1), -1, 1))[warm:], rtol=0, atol=1e-12)
        self.assertGreater(pos[500:1000].mean(), 0.7)          # long, and sure of it, through the up-trend
        self.assertLess(np.abs(pos[1400:2000]).mean(), 0.25)   # split and losing through the range

    def test_experts_are_scored_on_the_legs_the_template_trades(self):
        df = self.df
        Sb, _, experts = _hedge_stances(df, 20, "learned", "hedge_wide", "both")
        Sl, _, _ = _hedge_stances(df, 20, "learned", "hedge_wide", "long_only")
        Ss, _, _ = _hedge_stances(df, 20, "learned", "hedge_wide", "short_only")
        self.assertTrue((Sl >= 0).all() and (Ss <= 0).all())
        np.testing.assert_array_equal(Sl, np.where(Sb > 0, Sb, 0.0))    # the long leg of the two-sided stance
        np.testing.assert_array_equal(Ss, np.where(Sb < 0, Sb, 0.0))
        self.assertGreater((Sb < 0).sum(), 1000)                          # and the leg dropped was not empty
        self.assertEqual(_allowed("both"), (True, True))
        self.assertEqual(_allowed("long_only"), (True, False))
        self.assertEqual(_allowed("short_only"), (False, True))
        # a long-only fade expert is "buy new lows": its loss no longer counts short legs
        lb, _, _ = _hedge_loss(df, 20, "learned", 5.0, "hedge_wide", "both")
        ll, _, _ = _hedge_loss(df, 20, "learned", 5.0, "hedge_wide", "long_only")
        flat = (Sl[:-1] == 0) & (Sl[1:] == 0)
        self.assertTrue((ll[1:][flat] == 0.5).all())
        held = (Sl[:-1] == Sb[:-1]) & (Sl[1:] == Sb[1:])
        np.testing.assert_array_equal(ll[1:][held], lb[1:][held])
        self.assertFalse(np.array_equal(ll, lb))
        # a reversal the two-sided expert makes (long to short, two sides traded)
        # is one side traded for the long-only one (long to flat): on that bar it
        # is charged one side less, a quarter of the cost in ATRs, whatever the
        # expert's kind
        labels = [e.label for e in experts]
        a_prev = atr(df, 20).to_numpy(); close = df["Close"].to_numpy()
        for j in (labels.index("follow_20"), labels.index("fade_kel20x2"), labels.index("fade_10")):
            cut = np.where((Sb[:-1, j] == 1) & (Sb[1:, j] == -1) & (Sl[1:, j] == 0))[0] + 1
            self.assertGreater(len(cut), 5, labels[j])
            charge = 0.25 * close[cut] * 5.0 / 1e4 / a_prev[cut - 1]
            inside = (lb[cut, j] > 0) & (lb[cut, j] < 1) & (ll[cut, j] > 0) & (ll[cut, j] < 1)
            self.assertGreater(inside.sum(), len(cut) * 0.8, labels[j])   # the payoff clip at +/-1 eats the rest
            np.testing.assert_allclose((lb[cut, j] - ll[cut, j])[inside], charge[inside], atol=1e-12, err_msg=labels[j])
            self.assertTrue(((lb[cut, j] - ll[cut, j]) <= charge + 1e-12).all(), labels[j])
        # the weights, direction and channel follow: different under a one-sided template
        Wb = hedge_weights(df, 20, "learned", 5.0, "hedge_wide")
        Wl = hedge_weights(df, 20, "learned", 5.0, "hedge_wide", "long_only")
        self.assertFalse(np.allclose(Wb, Wl))
        self.assertFalse(np.allclose(np.nan_to_num(hedge_direction(df, 20, 5.0, "hedge_wide")),
                                     np.nan_to_num(hedge_direction(df, 20, 5.0, "hedge_wide", "long_only"))))
        # ... and the plain ladder, without cash, scores both legs whatever the template trades
        self.assertEqual(hedge_scored_sides("hedge", "long_only"), "both")
        self.assertEqual(hedge_scored_sides("hedge_wide", "long_only"), "long_only")
        np.testing.assert_array_equal(hedge_weights(df, 20, "learned", 5.0, "hedge", "long_only"),
                                      hedge_weights(df, 20, "learned", 5.0))
        np.testing.assert_array_equal(hedge_direction(df, 20, 5.0, "hedge", "short_only"), hedge_direction(df, 20, 5.0))
        plain = backtest(df, StrategyTemplate("t", direction_logic="learned", sides="long_only"))["indicators"]["direction"]
        np.testing.assert_array_equal(plain, hedge_direction(df, 20, 5.0))

    def test_weights_are_causal_normalised_and_kept_apart_from_the_plain_ladder(self):
        W = hedge_weights(self.df, 20, "learned", 5.0, "hedge_wide")
        self.assertEqual(W.shape, (len(self.df), 2 * len(HEDGE_LADDERS["hedge_wide"])))
        np.testing.assert_allclose(W.sum(axis=1), 1.0)
        self.assertTrue((W >= 0).all())
        np.testing.assert_allclose(W[:700], hedge_weights(self.df.iloc[:700], 20, "learned", 5.0, "hedge_wide"))
        W0 = hedge_weights(self.df, 20, "learned", 5.0)
        self.assertEqual(W0.shape[1], 2 * len(HEDGE_LADDER))
        labels = [e.label for e in hedge_ladder("learned", "hedge_wide")]
        cols = [labels.index(e.label) for e in hedge_ladder("learned")]
        self.assertFalse(np.allclose(W0, W[:, cols]))   # a different mixture
        d = hedge_diagnostics(self.df, 20, "learned", 5.0, "hedge_wide")
        self.assertEqual(list(d["weights"].columns), labels)
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
            W = hedge_weights(self.df, 20, mode, 5.0, "hedge_wide")
            np.testing.assert_allclose(up.to_numpy()[warm:], (W[warm:] * ups).sum(axis=1), rtol=1e-12)
        plain = hedge_channel(self.df, 20, "trend", 1.0, 5.0)[0]
        self.assertFalse(np.allclose(np.nan_to_num(up), np.nan_to_num(plain)))

    def test_conviction_is_the_committee_position_on_the_legs_the_template_trades(self):
        df, warm = self.df, hedge_warmup(20, "hedge_wide")
        experts = hedge_ladder("learned", "hedge_wide")
        esides = np.array([float(e.side) for e in experts])
        for sides in ("both", "long_only", "short_only"):
            W = hedge_weights(df, 20, "learned", 5.0, "hedge_wide", sides)
            d = hedge_direction(df, 20, 5.0, "hedge_wide", sides)
            self.assertTrue(np.isnan(d[:warm]).all() and np.isfinite(d[warm:]).all())
            net = W @ esides
            pos = hedge_position(df, 20, "learned", 5.0, "hedge_wide", sides)
            u = hedge_diagnostics(df, 20, "learned", 5.0, "hedge_wide", sides)["trade_weight"].to_numpy()
            np.testing.assert_array_equal(np.sign(d[warm:]), np.where(net < 0, -1.0, 1.0)[warm:])   # the direction is the net side weight's
            np.testing.assert_allclose(np.abs(d[warm:]), np.abs(pos[warm:]))                        # the size is the position's
            self.assertTrue((np.abs(d[warm:]) <= u[warm:] + 1e-12).all())                          # never more than the trade weight
            S, _, _ = _hedge_stances(df, 20, "learned", "hedge_wide", sides)
            if sides == "long_only":
                self.assertTrue((pos[warm:] >= 0).all())      # a long-only committee is long or flat, never short
                self.assertTrue((S <= 0).sum() > 0 and (S < 0).sum() == 0)
            elif sides == "short_only":
                self.assertTrue((pos[warm:] <= 0).all())
            # graded, not a flag; fixed share's floor keeps the smallest conviction
            # near 6 % on this series (0.3 % discounted)
            self.assertGreater(np.abs(d[warm:]).max(), 0.5); self.assertLess(np.abs(d[warm:]).min(), 0.1)
        # the plain ladder's direction is untouched
        np.testing.assert_array_equal(hedge_direction(df, 20, 0.0),
                                      np.where(np.arange(len(df)) < hedge_warmup(20), np.nan,
                                               np.clip(hedge_weights(df, 20, "learned", 0.0)
                                                       @ hedge_experts("learned")[1], -1, 1)))
        np.testing.assert_array_equal(hedge_direction(df, 20, 0.0, "hedge", "long_only"), hedge_direction(df, 20, 0.0))

    def test_the_position_stands_aside_where_the_side_loses(self):
        """A long-only learner has no short leg to earn on in a decline: every
        long stance loses, the played mixture loses to cash and the trade
        weight takes the size down, where the plain ladder's net side weight
        would keep buying dips at whatever conviction the fade experts had
        over the follow ones. Long only through a bull market the committee
        is long and paid, and the size is near full."""
        rng = np.random.default_rng(5)
        rets = np.r_[0.0008 + rng.normal(0, 0.009, 700), -0.0015 + rng.normal(0, 0.012, 800)]
        df = _ohlc_from_close(100 * np.cumprod(1 + rets), rng)
        d = hedge_direction(df, 20, 5.0, "hedge_wide", "long_only")
        u = hedge_diagnostics(df, 20, "learned", 5.0, "hedge_wide", "long_only")["trade_weight"].to_numpy()
        self.assertGreater(np.abs(d[410:700]).mean(), 0.5)
        self.assertLess(np.abs(d[900:]).mean(), 0.3)
        self.assertLess(u[900:].mean(), 0.35)
        plain = hedge_direction(df, 20, 5.0, "hedge", "long_only")
        # the plain ladder keeps twice the size (0.32 against 0.16; discounted
        # 0.53 against 0.13): a ratio, as fixed share shrinks both
        self.assertGreater(np.abs(plain[900:]).mean(), 1.5 * np.abs(d[900:]).mean())
        res = backtest(df, StrategyTemplate("t", channel_type="hedge_wide", direction_logic="learned", sides="long_only"))
        ref = backtest(df, StrategyTemplate("t", channel_type="hedge", direction_logic="learned", sides="long_only"))
        leg = lambda r: float(r["equity"].iloc[-1] / r["equity"].iloc[900] - 1)
        self.assertGreater(leg(res), leg(ref))
        self.assertGreater(leg(res), -0.05)

    def test_a_fixed_direction_is_sized_by_the_committee_position(self):
        """A trend / countertrend template on the wide ladder (the `full`
        family has them) scales its entries by the magnitude of its
        committee's position, the way a learned direction does: a trade
        entered from flat on bar i holds equity[i-1] * risk_pct /
        (atr_mult_stop * ATR[i-1]) shares times that magnitude on bar i-1.
        The plain ladder's direction stays the constant it was."""
        df = regime_series()
        for dl, sign in (("trend", 1.0), ("countertrend", -1.0)):
            tpl = StrategyTemplate("t", channel_type="hedge_wide", direction_logic=dl, cost_bps=0.0, max_leverage=1e9)
            active = hedge_active(df, 20, dl, 0.0, "hedge_wide")
            warm = hedge_warmup(20, "hedge_wide")
            self.assertTrue(np.isnan(active[:warm]).all())
            np.testing.assert_array_equal(active[warm:], np.abs(hedge_position(df, 20, dl, 0.0, "hedge_wide"))[warm:])
            res = backtest(df, tpl)
            np.testing.assert_array_equal(res["indicators"]["direction"], sign * active)
            eq = res["equity"].to_numpy(); a = res["indicators"]["atr"]
            flat = _flat_entries(df, res)      # entered from a flat book: the cash sized on is equity[i-1]
            self.assertGreater(len(flat), 30)
            for i, t in flat:
                full = eq[i - 1] * tpl.risk_pct / (tpl.atr_mult_stop * a[i - 1])
                self.assertAlmostEqual(t["shares"] / full, active[i - 1], places=9, msg=f"{dl} {t['entry_date']}")
            # small in the regime that is not the template's, full in the one that is;
            # and the entries there, which are breaks the committee is not yet in, are
            # sized below the bar average but not starved
            wrong = slice(1400, 2000) if dl == "trend" else slice(500, 1000)
            right = slice(500, 1000) if dl == "trend" else slice(1400, 2000)
            self.assertLess(active[wrong].mean(), 0.3)
            self.assertGreater(active[right].mean(), 0.7)
            entered = [active[i - 1] for i, _ in flat if right.start <= i < right.stop]
            self.assertGreater(len(entered), 5)
            self.assertGreater(np.median(entered), 0.4)
            # a ladder that is not position-sized is fully active and its direction is the constant it was
            np.testing.assert_array_equal(hedge_active(df, 20, dl, 0.0)[warm:], 1.0)
            plain = backtest(df, tpl.with_params(channel_type="hedge"))["indicators"]["direction"]
            np.testing.assert_array_equal(plain, np.full(len(df), sign))
        # ... and it loses far less than the plain ladder where the regime is not its own
        bars = np.arange(len(df))
        in_trend = ((bars >= 410) & (bars < 1000)) | (bars >= 2000)
        for dl, wrong in (("trend", ~in_trend & (bars >= 1000)), ("countertrend", in_trend)):
            wide = backtest(df, StrategyTemplate("t", channel_type="hedge_wide", direction_logic=dl))
            plain = backtest(df, StrategyTemplate("t", channel_type="hedge", direction_logic=dl))
            def seg(res):
                r = res["equity"].pct_change().fillna(0).to_numpy()[wrong]
                return float(np.prod(1 + r) - 1)
            self.assertLess(seg(plain), -0.2)
            self.assertGreater(seg(wide), seg(plain) + 0.08)

    def test_templates_have_nothing_to_fit_and_read_no_width(self):
        df = self.df
        tpl = StrategyTemplate("t", channel_type="hedge_wide", direction_logic="learned", cost_bps=0.0)
        self.assertEqual(param_grid_for(tpl), {})
        self.assertEqual(param_grid_for(tpl, wide=True), {})
        base = backtest(df, tpl)
        self.assertGreater(base["stats"]["n_trades"], 5)
        eq = base["equity"].to_numpy()
        for k, v in (("channel_k", 0.7), ("n_entry", 13), ("n_exit", 7), ("regime_threshold", 0.01), ("regime_n", 7)):
            np.testing.assert_array_equal(eq, backtest(df, tpl.with_params(**{k: v}))["equity"].to_numpy(), k)
        # the ATR length, the cost and the sides do reach the learner
        self.assertFalse(np.array_equal(eq, backtest(df, tpl.with_params(atr_n=14))["equity"].to_numpy()))
        self.assertFalse(np.array_equal(eq, backtest(df, tpl.with_params(cost_bps=50.0))["equity"].to_numpy()))
        self.assertFalse(np.array_equal(eq, backtest(df, tpl.with_params(channel_type="hedge"))["equity"].to_numpy()))
        self.assertGreaterEqual(warmup_bars(tpl), hedge_warmup(20, "hedge_wide"))
        # a learned direction on a fitted channel keeps the plain ladder
        don = backtest(df, StrategyTemplate("t", direction_logic="learned", cost_bps=5.0))["indicators"]["direction"]
        np.testing.assert_array_equal(don, hedge_direction(df, 20, 5.0))
        for sd in ("both", "long_only"):
            wide = backtest(df, tpl.with_params(cost_bps=5.0, sides=sd))["indicators"]["direction"]
            np.testing.assert_array_equal(wide, hedge_direction(df, 20, 5.0, "hedge_wide", sd))
        # the family: the learned direction over the wide ladder, two entries by four
        # exits, and nothing stacked on top of the learner
        fam = generate_templates("online", direction_logics=["learned"], channel_types=["hedge_wide"],
                                 entry_styles=["stop", "close_confirm"])
        self.assertEqual(len(fam), 8)
        self.assertEqual({t.direction_logic for t in fam}, {"learned"})
        self.assertEqual({t.channel_type for t in fam}, {"hedge_wide"})
        self.assertEqual({t.regime_filter for t in fam}, {"none"})
        self.assertEqual({t.bias_filter for t in fam}, {"none"})
        self.assertEqual({t.vol_filter for t in fam}, {False})
        self.assertEqual({(t.entry_style, t.exit_style) for t in fam},
                         {(e, x) for e in ("stop", "close_confirm") for x in EXIT_STYLES})
        self.assertEqual(fam[0].name, "LN-hdw-stop-chan-noreg-noV-noB")
        for t in fam:
            self.assertEqual({k for k in param_grid_for(t)} & {"n_entry", "n_exit", "channel_k", "regime_threshold"}, set())
        res = walk_forward(df, fam[0], param_grid_for(fam[0]), train_bars=400, test_bars=100)
        self.assertEqual(len(res["oos_returns"]), len(df) - 400)

    def test_learns_to_follow_trends_and_fade_the_range(self):
        """On the regime series the learned direction follows the trends and
        fades the range, the fade weight sits on the Donchian and Keltner
        fade experts in the range, and the learned template beats a fixed
        fade one."""
        rs = regime_series()
        d = hedge_direction(rs, 20, 0.0, "hedge_wide")
        self.assertGreater((d[500:1000] > 0).mean(), 0.8)      # trend: follow
        self.assertGreater((d[1400:2000] < 0).mean(), 0.9)     # mean reversion: fade
        self.assertGreater((d[2500:3000] > 0).mean(), 0.8)     # trend again: follow
        W = hedge_weights(rs, 20, "learned", 0.0, "hedge_wide")
        labels = [e.label for e in hedge_ladder("learned", "hedge_wide")]
        fades = [i for i, l in enumerate(labels) if l.startswith("fade_")]
        follows = [i for i, l in enumerate(labels) if l.startswith("follow_")]
        self.assertGreater(W[1400:2000, fades].sum(axis=1).mean(), 0.7)
        self.assertGreater(W[500:1000, follows].sum(axis=1).mean(), 0.7)
        learned = backtest(rs, StrategyTemplate("t", channel_type="hedge_wide", direction_logic="learned", cost_bps=0.0))
        fade = backtest(rs, StrategyTemplate("t", channel_type="hedge_wide", direction_logic="countertrend", cost_bps=0.0))
        self.assertGreater(learned["stats"]["total_return"], fade["stats"]["total_return"])
        self.assertGreater(learned["stats"]["total_return"], 0.3)
        # long only, the learner is scored on buying breaks alone: in the trends buying
        # new highs and buying dips both pay, the committee is long and paid, so the
        # size is near full rather than the near tie the net side weight would make it
        dl = hedge_direction(rs, 20, 0.0, "hedge_wide", "long_only")
        self.assertGreater(np.nanmean(np.abs(dl[500:1000])), 0.7)
        long_ = backtest(rs, StrategyTemplate("t", channel_type="hedge_wide", direction_logic="learned", cost_bps=0.0,
                                              sides="long_only"))
        self.assertGreater(long_["stats"]["total_return"], 0.3)


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


class StanceEntryTests(unittest.TestCase):
    """entry_style 'stance': the template holds the side of the learner's
    committee (hedge_stance), held under a no-trade band, ordered at the
    next open."""

    @classmethod
    def setUpClass(cls):
        cls.df = synthetic_ohlc(1500, seed=5)

    def _tpl(self, **kw):
        kw = {"channel_type": "hedge_slow", **kw}
        return StrategyTemplate("st", direction_logic="learned", entry_style="stance", **kw)

    def test_causal_and_accounted(self):
        tpl = self._tpl()
        full = backtest(self.df, tpl, fixed_capital=True)
        part = backtest(self.df.iloc[:1200], tpl, fixed_capital=True)
        # nothing on bar t depends on a later bar
        np.testing.assert_allclose(full["equity"].to_numpy()[:1200], part["equity"].to_numpy(), atol=1e-8)
        self.assertGreater(len(full["trades"]), 5)
        # the closed trades and the open one are the whole P&L
        pnl = sum(t["pnl"] for t in full["trades"])
        op = full["open_position"]
        if op is not None:
            pnl += op["unrealized"] - op["entry_cost"]
        self.assertAlmostEqual(full["equity"].iloc[-1] - 100_000.0, pnl, places=6)
        # sizes follow the buffered stance, and the exit rule is inert
        st = backtest(self.df, tpl.with_params(exit_style="time_stop", max_hold_bars=3), fixed_capital=True)
        np.testing.assert_array_equal(full["equity"].to_numpy(), st["equity"].to_numpy())

    def test_sides_and_warmup(self):
        tpl = self._tpl(sides="long_only")
        res = backtest(self.df, tpl)
        self.assertTrue(all(t["side"] == 1 for t in res["trades"]))
        warm = hedge_warmup(20, "hedge_slow")
        self.assertTrue(all(self.df.index.get_loc(t["entry_date"]) > warm for t in res["trades"]))
        res = backtest(self.df, tpl, first_trade_bar=1300)
        self.assertTrue(all(self.df.index.get_loc(t["entry_date"]) >= 1300 for t in res["trades"]))

    def test_the_level_is_held_under_the_no_trade_band(self):
        """The held level stays while the stance is within STANCE_BUFFER of it
        and moves to the near edge of the band otherwise (no quarters), a NaN
        stance holds it and reports 0, and `sides` zeroes the forbidden side."""
        tpl = self._tpl()
        p = hedge_stance(self.df, tpl.atr_n, "learned", tpl.cost_bps, "hedge_slow", "both", 0.0)
        self.assertTrue(np.isnan(p[:5]).all() and np.isfinite(p[-1]))
        q = _buffered_level(p, STANCE_BUFFER)
        self.assertTrue((q[np.isnan(p)] == 0).all())
        ok = ~np.isnan(p)
        self.assertTrue((np.abs(p[ok] - q[ok]) <= STANCE_BUFFER + 1e-12).all())
        prev = 0.0
        moves = 0
        for pi, qi in zip(p[ok], q[ok]):
            if abs(pi - prev) <= STANCE_BUFFER:
                self.assertEqual(qi, prev)
            else:
                self.assertAlmostEqual(qi, pi - np.sign(pi - prev) * STANCE_BUFFER, places=12)
                moves += 1
            prev = qi
        self.assertGreater(moves, 5)
        self.assertGreater(len(set(np.round(q[ok], 6))), 8)     # not a grid of quarters
        # a stance wobbling across the old x.125 rounding boundary does not trade
        w = _buffered_level(np.array([0.0, 0.12, 0.13, 0.12, 0.13]), STANCE_BUFFER)
        np.testing.assert_allclose(w, [0.0, 0.0, 0.005, 0.005, 0.005], atol=1e-12)
        # a NaN holds the level but reports 0
        np.testing.assert_allclose(_buffered_level(np.array([0.5, np.nan, 0.45]), 0.125), [0.375, 0.0, 0.375])
        # a one-sided band goes flat on a bar with no stance on its side and
        # bands from there: never below 0 for long-only (the target's -0.6 is
        # flat, not a level of -0.475 to climb back from)
        z = np.array([0.6, -0.6, 0.2, -0.2])
        np.testing.assert_allclose(_buffered_level(z, 0.125, "long_only"), [0.475, 0.0, 0.075, 0.0])
        np.testing.assert_allclose(_buffered_level(z, 0.125, "short_only"), [0.0, -0.475, 0.0, -0.075])

    def test_a_one_sided_band_has_no_memory_of_the_forbidden_side(self):
        """A long-only band goes flat on every bar whose stance has nothing on
        the long side (negative or exactly 0) and bands from there, so what it
        holds after such a bar does not depend on anything before it: the
        bars before are a common restart, which is what lets a walk-forward
        window warmed on a short buffer reproduce the full history (a band
        clamped at 0 could keep any sliver in [0, buf] for as long as the
        stance stayed at 0). Short-only is the mirror."""
        p = np.array([0.3, -0.5, 0.2, -0.5, -0.5, 0.2, 0.5])
        lo = _buffered_level(p, STANCE_BUFFER, "long_only")
        np.testing.assert_allclose(lo, [0.175, 0.0, 0.075, 0.0, 0.0, 0.075, 0.375], atol=1e-12)
        self.assertGreaterEqual(lo.min(), 0.0)
        # different histories before a flat bar give the same levels after it
        q = np.array([0.9, 0.0, 0.2, -0.5, -0.5, 0.2, 0.5])
        np.testing.assert_allclose(_buffered_level(q, STANCE_BUFFER, "long_only")[1:], lo[1:], atol=1e-12)
        np.testing.assert_allclose(_buffered_level(-p, STANCE_BUFFER, "short_only"), -lo, atol=1e-12)
        # NaN holds and reports 0, and the caller's target is not touched
        t = np.array([np.nan, -0.5, 0.2])
        np.testing.assert_allclose(_buffered_level(t, 0.125, "long_only"), [0.0, 0.0, 0.075], atol=1e-12)
        np.testing.assert_array_equal(t[1:], [-0.5, 0.2])

    def test_a_warmed_window_reproduces_the_full_history(self):
        """The held level remembers where it was, so warmup_bars adds
        STANCE_SETTLE: a window warmed on it returns what a full-history run
        (first trade at the window start) does. Checked on the slow ladder and
        the shipped split ladder, on every side, and on a calendar spread
        (per-unit costs, whole contracts, Roll)."""
        from walkforward import warmup_bars, window_backtest
        df = synthetic_ohlc(3000, seed=2)
        cases = [("hedge_slow", sides, df, {}) for sides in ("both", "long_only")]
        cases += [("hedge_split", sides, df, {}) for sides in ("both", "long_only", "short_only")]
        from extra_utils.online_study import synth
        sp = synth.calendar_spread(3000, 1)
        inst = {k: v for k, v in sp.attrs["instrument"].items() if k != "tick"}
        cases += [("hedge_split", sides, sp, dict(inst, cost_bps=0.0, whole_units=True)) for sides in ("both", "short_only")]
        for ch, sides, d, kw in cases:
            tpl = self._tpl(channel_type=ch, sides=sides, **kw)
            self.assertEqual(warmup_bars(tpl),
                             hedge_warmup(tpl.atr_n, "hedge_slow" if ch == "hedge_slow" else "hedge_split")
                             + STANCE_SETTLE + 5)
            for st in (1800, 2100, 2500):
                full = backtest(d, tpl, first_trade_bar=st)
                win = window_backtest(d, tpl, st, st + 300, initial_equity=100_000.0)
                np.testing.assert_allclose(win["returns"].to_numpy(), full["returns"].to_numpy()[st:st + 300],
                                           rtol=0, atol=1e-12, err_msg=f"{ch} {sides} {st} {'spread' if kw else ''}")

    def test_switches_it_would_ignore_are_refused(self):
        for kw in (dict(regime_filter="trend_only"), dict(bias_filter="sma"), dict(vol_filter=True),
                   dict(channel_type="donchian")):
            with self.assertRaises(AssertionError, msg=str(kw)):
                self._tpl(**kw).validate()
        self._tpl().validate()

    def test_spread_contracts_and_rolls(self):
        """A calendar spread: prices through zero, a margin, whole contracts
        and a roll cost. Adding a constant to every price changes nothing,
        sizes are whole, and every cost (rolls included) is on a trade."""
        df = self.df.copy()
        df[["Open", "High", "Low", "Close"]] -= float(df["Close"].median())   # crosses zero
        df["Roll"] = 0.0
        df.iloc[::63, df.columns.get_loc("Roll")] = 1.0
        # 2 % risk: whole contracts need a committee position worth at least
        # one, and fixed share's committee is the less concentrated (at 1 %
        # it trades once here, discounting 7 times)
        tpl = self._tpl(channel_type="hedge_split", cost_bps=0.0, cost_per_unit=2.5, point_value=100.0,
                        margin_per_unit=500.0, whole_units=True, roll_cost_per_unit=4.0, risk_pct=0.02)
        res = backtest(df, tpl, fixed_capital=True)
        self.assertGreater(len(res["trades"]), 5)
        self.assertTrue(all(t["shares"] == int(t["shares"]) and t["shares"] >= 1 for t in res["trades"]))
        pnl = sum(t["pnl"] for t in res["trades"])
        op = res["open_position"]
        if op is not None:
            pnl += op["unrealized"] - op["entry_cost"]
        self.assertAlmostEqual(res["equity"].iloc[-1] - 100_000.0, pnl, places=6)
        shifted = df.copy()
        shifted[["Open", "High", "Low", "Close"]] += 37.0
        np.testing.assert_allclose(backtest(shifted, tpl, fixed_capital=True)["equity"].to_numpy(),
                                   res["equity"].to_numpy(), atol=1e-6)
        no_roll = backtest(df, tpl.with_params(roll_cost_per_unit=0.0), fixed_capital=True)
        self.assertGreater(no_roll["equity"].iloc[-1], res["equity"].iloc[-1])


class SplitLadderTests(unittest.TestCase):
    """The 'hedge_split' ladder: a follow group (20-80 bar breaks held as
    long as their lookback, slow learner), a fade group (5-20 bar breaks
    held 1 or 3 bars, fast learner) and a top learner between the two."""

    @classmethod
    def setUpClass(cls):
        cls.df = synthetic_ohlc(1600, seed=11)

    def test_hold_is_decoupled_from_the_lookback(self):
        e = HedgeExpert("donchian", 10, hold=3)
        self.assertEqual(e.span, 3)
        self.assertEqual(HedgeExpert("donchian", 10).span, 10)
        self.assertEqual(e.lead(20), 3 + 9 + 1)
        self.assertEqual(replace(e, side=-1).label, "fade_10_h3")
        self.assertEqual(HedgeExpert("keltner", 20, 2.0, side=1).label, "follow_kel20x2")
        self.assertTrue(all(x.span == x.n for x in HEDGE_SPLIT_FOLLOW))
        self.assertTrue(all(x.span in (1, 3) and x.span < x.n for x in HEDGE_SPLIT_FADE))
        # the exit channel rounds half up: a 5-bar rung exits on a 3-bar channel
        np.testing.assert_array_equal(HedgeExpert("donchian", 5).bands(self.df, 20, HEDGE_EXIT_SCALE)[0],
                                      donchian(self.df, 3)[0].to_numpy())
        # a fade held 1 bar is short on the bar that breaks up, long on the
        # bar that breaks down, flat otherwise; held 3 bars it keeps the most
        # recent break's side for 3 bars
        df = self.df
        S, _, ex = _hedge_stances(df, 20, "countertrend", "hedge_split")
        high, low = df["High"].to_numpy(), df["Low"].to_numpy()
        for j, x in enumerate(ex):
            u, lo = x.bands(df, 20)
            up = np.r_[False, high[1:] >= u[:-1]]
            dn = np.r_[False, low[1:] <= lo[:-1]]
            brk = np.where(up & ~dn, -1.0, np.where(dn & ~up, 1.0, 0.0))
            want = np.zeros(len(df))
            for t in range(len(df)):
                for back in range(x.span):
                    if t - back >= 0 and brk[t - back] != 0:
                        want[t] = brk[t - back]
                        break
            np.testing.assert_array_equal(S[:, j], want, err_msg=x.label)
            self.assertGreater((want != 0).sum(), 10)

    def test_ladder_modes_and_labels(self):
        self.assertTrue(hedge_position_sized("hedge_split"))
        self.assertEqual(hedge_ladder_for("hedge_split"), "hedge_split")
        nf, nr = len(HEDGE_SPLIT_FOLLOW), len(HEDGE_SPLIT_FADE)
        self.assertEqual([x.side for x in hedge_ladder("trend", "hedge_split")], [1] * nf)
        self.assertEqual([x.side for x in hedge_ladder("countertrend", "hedge_split")], [-1] * nr)
        labels = [x.label for x in hedge_ladder("learned", "hedge_split")]
        self.assertEqual(len(labels), nf + nr)
        self.assertEqual(len(set(labels)), len(labels))
        d = hedge_diagnostics(self.df, 20, "learned", 5.0, "hedge_split")
        self.assertEqual(list(d["eta"].columns), list(hedge_rungs(*HEDGE_SPLIT_LEARNERS["top"])[3]))
        d = hedge_diagnostics(self.df, 20, "countertrend", 5.0, "hedge_split")
        self.assertEqual(list(d["eta"].columns), list(hedge_rungs(*HEDGE_SPLIT_LEARNERS["fade"])[3]))

    def test_groups_run_their_own_learners(self):
        """The learned weights are the top weight times each group's own
        weights, and a group's own weights are what its fixed direction
        plays."""
        df, nf = self.df, len(HEDGE_SPLIT_FOLLOW)
        W = hedge_weights(df, 20, "learned", 5.0, "hedge_split")
        Wf = hedge_weights(df, 20, "trend", 5.0, "hedge_split")
        Wr = hedge_weights(df, 20, "countertrend", 5.0, "hedge_split")
        np.testing.assert_allclose(W.sum(axis=1), 1.0, atol=1e-12)
        g = W[:, :nf].sum(axis=1, keepdims=True)
        np.testing.assert_allclose(W[:, :nf], g * Wf, atol=1e-12)
        np.testing.assert_allclose(W[:, nf:], (1.0 - g) * Wr, atol=1e-12)
        loss, _, _ = _hedge_loss(df, 20, "trend", 5.0, "hedge_split")
        np.testing.assert_array_equal(Wf, _adahedge_loop(loss, *HEDGE_SPLIT_LEARNERS["follow"])[0])
        gw = hedge_group_weights(df, 20, 5.0)
        warm = hedge_warmup(20, "hedge_split")
        self.assertTrue(gw.iloc[:warm].isna().all().all() and not gw.iloc[warm:].isna().any().any())
        np.testing.assert_allclose(gw.sum(axis=1).iloc[warm:], 1.0, atol=1e-12)

    def test_warmup_contract_and_causality(self):
        df = self.df
        for atr_n in (14, 20):
            warm = hedge_warmup(atr_n, "hedge_split")
            self.assertGreaterEqual(warm, HEDGE_SPLIT_LEARNERS["follow"][0] + HEDGE_SPLIT_LEARNERS["top"][0])
            for mode in ("trend", "countertrend", "learned"):
                full = hedge_weights(df, atr_n, mode, 5.0, "hedge_split")
                for k in (37, 250):
                    part = hedge_weights(df.iloc[k:], atr_n, mode, 5.0, "hedge_split")
                    np.testing.assert_allclose(full[k + warm:], part[warm:], atol=1e-12, err_msg=f"{mode} {k}")
                # a prefix of the series gives the prefix of the weights
                head = hedge_weights(df.iloc[:1100], atr_n, mode, 5.0, "hedge_split")
                np.testing.assert_array_equal(full[:1100], head)
        full = hedge_position(df, 20, "learned", 5.0, "hedge_split", "long_only")
        part = hedge_position(df.iloc[100:], 20, "learned", 5.0, "hedge_split", "long_only")
        warm = hedge_warmup(20, "hedge_split")
        np.testing.assert_allclose(full[100 + warm:], part[warm:], atol=1e-12)

    def test_top_learner_moves_to_the_group_that_pays(self):
        """On a trend / mean-reversion / trend regime series the fade group
        holds most of the weight inside the mean-reverting stretch and
        little in the trends (pinned on these seeds: about 0.9 against 0.3
        or less)."""
        for seed in (0, 1):
            g = hedge_group_weights(regime_series(seed), 20, 5.0)["fade"].to_numpy()
            mr, tr1, tr2 = np.nanmean(g[1200:2000]), np.nanmean(g[800:1000]), np.nanmean(g[2200:3000])
            self.assertGreater(mr, 0.8)
            self.assertLess(max(tr1, tr2), 0.35)

    def test_templates_run_on_the_split_ladder(self):
        df = self.df
        warm = hedge_warmup(20, "hedge_split")
        for es in ("stop", "close_confirm", "stance"):
            for dl in ("trend", "countertrend", "learned"):
                tpl = StrategyTemplate("sp", direction_logic=dl, channel_type="hedge_split", entry_style=es)
                full = backtest(df, tpl, fixed_capital=True)
                part = backtest(df.iloc[:1300], tpl, fixed_capital=True)
                np.testing.assert_allclose(full["equity"].to_numpy()[:1300], part["equity"].to_numpy(), atol=1e-8)
                self.assertTrue(all(df.index.get_loc(t["entry_date"]) > warm for t in full["trades"]))
        names = [t.name for t in generate_templates(
            "online", direction_logics=["trend", "countertrend", "learned"],
            entry_styles=["stop", "close_confirm", "stance"])]
        self.assertEqual(len(names), 27)
        self.assertEqual(sum("-stance-" in n for n in names), 3)
        self.assertFalse(any(t.channel_type == "hedge_split" for t in generate_templates("full")))


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


class VarianceRatioByInstrumentTests(unittest.TestCase):
    def test_a_cash_asset_reads_log_returns_and_a_future_point_changes(self):
        """The `vr` regime filter is the classic log-return ratio on a share and
        the shift-invariant point-change ratio once a margin makes it a future."""
        from strategy import _compute_indicators
        df = synthetic_ohlc(800, seed=4, trend_drift=0.002)
        cash = StrategyTemplate("t", regime_filter="trend_only", regime_indicator="vr")
        fut = cash.with_params(point_value=1000.0, margin_per_unit=6000.0, cost_bps=0.0)
        np.testing.assert_array_equal(_compute_indicators(df, cash)["regime"],
                                      variance_ratio(df["Close"], 60, log_returns=True).to_numpy())
        np.testing.assert_array_equal(_compute_indicators(df, fut)["regime"], variance_ratio(df["Close"], 60).to_numpy())
        self.assertFalse(np.allclose(np.nan_to_num(_compute_indicators(df, cash)["regime"]),
                                     np.nan_to_num(_compute_indicators(df, fut)["regime"])))
