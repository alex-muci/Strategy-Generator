"""
Tests for the forecast channels (strategy.py, "The forecast channels"): the
warm-up contract, no look-ahead, shift invariance, what the forecaster learns
on planted series, validate(), and the MMI / Hurst indicators.

Run with:   python -m unittest discover -s tests -v     (or pytest)
"""

from __future__ import annotations
import os
import sys
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategy import (  # noqa: E402
    StrategyTemplate, backtest, forecast_position, forecast_diagnostics, forecast_warmup, forecast_stance,
    mmi, hurst, FORECAST_CHANNELS, REGIME_INDICATORS, FC_BUFFER, FC_MEMORY, FC_SETTLE, NORM_N, B_TREND,
)
from generator import generate_templates, param_grid_for  # noqa: E402
from walkforward import warmup_bars  # noqa: E402


def _ohlc(c, seed):
    """OHLC round a close path: the open near the last close, the wicks a
    fraction of a typical bar's move (in the series' own units, so a path
    through zero is as good as one far above it)."""
    rng = np.random.default_rng(seed)
    c = np.asarray(c, dtype=float)
    s = float(np.std(np.diff(c)))
    o = np.r_[c[0], c[:-1]] + rng.normal(0, 0.1 * s, len(c))
    h = np.maximum(o, c) + np.abs(rng.normal(0, 0.3 * s, len(c)))
    l = np.minimum(o, c) - np.abs(rng.normal(0, 0.3 * s, len(c)))
    return pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c, "Volume": 1},
                        index=pd.bdate_range("2010-01-01", periods=len(c)))


def random_walk(T, seed):
    return _ohlc(np.cumsum(np.random.default_rng(seed).normal(0, 1, T)), seed)


def planted_trend(T, seed, phi=0.98, drift_sd=0.25):
    """Increments = a slow AR(1) latent drift plus unit noise."""
    rng = np.random.default_rng(seed)
    d, r = 0.0, np.empty(T)
    for i in range(T):
        d = phi * d + rng.normal(0, drift_sd * np.sqrt(1 - phi ** 2))
        r[i] = d + rng.normal(0, 1)
    return _ohlc(np.cumsum(r), seed)


def _ou_path(T, rng, half_life=10, sd=3.0, z0=0.0):
    a = 0.5 ** (1.0 / half_life)
    z, out = z0, np.empty(T)
    for i in range(T):
        z = a * z + rng.normal(0, sd * np.sqrt(1 - a * a))
        out[i] = z
    return out


def ou_spread(T, seed, half_life=10):
    """A mean-reverting spread around ZERO (it crosses it all the time)."""
    return _ohlc(_ou_path(T, np.random.default_rng(seed), half_life), seed)


def regime_switch(seed, block=1500, blocks=4):
    """Trend blocks (slow latent drift) alternating with OU blocks."""
    rng = np.random.default_rng(seed)
    parts, x = [], 0.0
    for b in range(blocks):
        if b % 2 == 0:
            d = 0.0
            for _ in range(block):
                d = 0.98 * d + rng.normal(0, 0.25 * np.sqrt(1 - 0.98 ** 2))
                x += d + rng.normal(0, 1)
                parts.append(x)
        else:
            z = _ou_path(block, rng, 10, 3.0)
            parts.extend(x + z)
            x = x + z[-1]
    return _ohlc(np.array(parts), seed)


def futures(tpl, **kw):
    """A spread / future: sized on margin, points not percent, free of cost."""
    p = dict(margin_per_unit=1000.0, cost_bps=0.0, cost_per_unit=0.0, max_leverage=5.0, vol_target=0.15)
    p.update(kw)
    return tpl.with_params(**p)


def sharpe(res, skip):
    r = res["returns"].to_numpy()[skip:]
    return float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0.0


# the family has the plain channel only; the context channel is a switch and is tested here too
TEMPLATES = generate_templates("online_forecast", channel_types=["forecast", "forecast_ctx"])


class ForecastFamilyTests(unittest.TestCase):
    def test_the_family_is_three_templates(self):
        fam = generate_templates("online_forecast")
        self.assertEqual([t.channel_type for t in fam], ["forecast"] * 3)
        self.assertEqual({t.direction_logic for t in fam}, {"trend", "countertrend", "learned"})
        self.assertEqual(len(TEMPLATES), 6)
        self.assertEqual({t.channel_type for t in TEMPLATES}, set(FORECAST_CHANNELS))
        self.assertEqual({t.direction_logic for t in TEMPLATES}, {"trend", "countertrend", "learned"})
        self.assertTrue(all(t.entry_style == "stance" for t in TEMPLATES))
        self.assertTrue(all(param_grid_for(t) == {} for t in TEMPLATES))
        self.assertTrue(all("-fc-" in t.name or "-fcx-" in t.name for t in TEMPLATES))

    def test_online_is_the_split_follow_group(self):
        fam = generate_templates("online")
        self.assertEqual(len(fam), 9)
        self.assertEqual({t.direction_logic for t in fam}, {"trend"})
        self.assertEqual({t.channel_type for t in fam}, {"hedge_split"})
        self.assertEqual(sum(t.entry_style == "stance" for t in fam), 1)

    def test_full_family_leaves_the_forecast_channels_out(self):
        self.assertFalse({t.channel_type for t in generate_templates("full")} & set(FORECAST_CHANNELS))

    def test_warmup_numbers(self):
        # F0 = 3 * 128 + 1 = 385 for the features, + FC_MEMORY + 1 for the ridge when something is learned
        self.assertEqual(forecast_warmup("forecast", "trend"), 385)
        for ch in FORECAST_CHANNELS:
            for dl in ("countertrend", "learned"):
                self.assertEqual(forecast_warmup(ch, dl), 385 + FC_MEMORY + 1)
        self.assertEqual(forecast_warmup("forecast_ctx", "trend"), 385 + FC_MEMORY + 1)   # SLOW*R is learned
        for t in TEMPLATES:
            self.assertEqual(warmup_bars(t), forecast_warmup(t.channel_type, t.direction_logic) + FC_SETTLE + 5)
        self.assertEqual(warmup_bars(next(t for t in TEMPLATES if t.name.startswith("TR-fc-"))), 385 + FC_SETTLE + 5)

    def test_the_first_position_is_on_the_warmup_bar(self):
        df = random_walk(1500, 1)
        for t in TEMPLATES:
            p = forecast_position(df, t.channel_type, t.direction_logic)
            w = forecast_warmup(t.channel_type, t.direction_logic)
            self.assertTrue(np.isnan(p[:w]).all(), t.name)
            self.assertTrue(np.isfinite(p[w:]).all(), t.name)
            self.assertLessEqual(np.abs(p[w:]).max(), 1.0)


class WarmupContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = regime_switch(5, block=600, blocks=4)      # 2400 bars, both regimes

    def test_a_warmed_slice_reproduces_the_full_history_for_every_template(self):
        k, n = 1700, len(self.df)
        for tpl in TEMPLATES:
            t = futures(tpl)
            w = forecast_warmup(t.channel_type, t.direction_logic)
            full_p = forecast_position(self.df, t.channel_type, t.direction_logic)
            sl = self.df.iloc[k - w:]
            slice_p = forecast_position(sl, t.channel_type, t.direction_logic)
            np.testing.assert_allclose(slice_p[w:], full_p[k:], rtol=0, atol=1e-12, err_msg=t.name)
            # the backtest: warmed on warmup_bars, the equity from k on is the full run's
            b = warmup_bars(t)
            self.assertLess(b, k)
            full = backtest(self.df, t, first_trade_bar=k, fixed_capital=True)
            win = backtest(self.df.iloc[k - b:], t, first_trade_bar=b, fixed_capital=True)
            np.testing.assert_allclose(win["equity"].to_numpy()[b:], full["equity"].to_numpy()[k:],
                                       rtol=0, atol=1e-8 * 100_000.0, err_msg=t.name)
            self.assertEqual(len(win["trades"]), len(full["trades"]), t.name)
            self.assertGreater(len(win["trades"]), 10, t.name)


class NoLookAheadTests(unittest.TestCase):
    def test_changing_bars_after_t_leaves_the_past_alone(self):
        df = ou_spread(1800, 7)
        t0 = 1500
        bad = df.copy()
        rng = np.random.default_rng(0)
        for col in ("Open", "High", "Low", "Close"):
            bad.iloc[t0 + 1:, bad.columns.get_loc(col)] += rng.normal(0, 25.0, len(bad) - t0 - 1)
        for tpl in TEMPLATES:
            a = forecast_position(df, tpl.channel_type, tpl.direction_logic)
            b = forecast_position(bad, tpl.channel_type, tpl.direction_logic)
            np.testing.assert_array_equal(a[:t0 + 1], b[:t0 + 1], err_msg=tpl.name)
            self.assertFalse(np.array_equal(a[t0 + 1:], b[t0 + 1:], equal_nan=True))
            t = futures(tpl)
            ea = backtest(df, t, fixed_capital=True)["equity"].to_numpy()
            eb = backtest(bad, t, fixed_capital=True)["equity"].to_numpy()
            np.testing.assert_array_equal(ea[:t0 + 1], eb[:t0 + 1], err_msg=tpl.name)

    def test_the_order_at_the_open_uses_the_previous_target(self):
        # the held level of bar i-1's target is what a position opened at bar i holds
        df = ou_spread(1600, 8)
        t = futures(TEMPLATES[4])
        p = forecast_position(df, t.channel_type, t.direction_logic)
        q = forecast_stance(p, "both")
        res = backtest(df, t, fixed_capital=True)
        first = res["trades"][0]
        i = df.index.get_loc(first["entry_date"])
        self.assertEqual(int(np.sign(q[i - 1])), first["side"])
        self.assertGreaterEqual(i - 1, forecast_warmup(t.channel_type, t.direction_logic))


class ShiftInvarianceTests(unittest.TestCase):
    def test_adding_a_constant_leaves_the_position_alone(self):
        df = ou_spread(1500, 9)                   # crosses zero
        self.assertLess(df["Low"].min(), 0.0)
        self.assertGreater(df["High"].max(), 0.0)
        for shift in (1000.0, -37.5):
            sh = df + shift
            sh["Volume"] = 1
            for tpl in TEMPLATES:
                a = forecast_position(df, tpl.channel_type, tpl.direction_logic)
                b = forecast_position(sh, tpl.channel_type, tpl.direction_logic)
                np.testing.assert_allclose(b, a, rtol=0, atol=1e-8, err_msg=f"{tpl.name} {shift}")
        t = futures(TEMPLATES[4])
        e0 = backtest(df, t, fixed_capital=True)["equity"].to_numpy()
        sh = df + 1000.0
        sh["Volume"] = 1
        e1 = backtest(sh, t, fixed_capital=True)["equity"].to_numpy()
        np.testing.assert_allclose(e1, e0, rtol=0, atol=1e-6)


class SyntheticBehaviourTests(unittest.TestCase):
    def _by(self, df, skip):
        out = {}
        for tpl in TEMPLATES:
            t = futures(tpl)
            res = backtest(df, t, fixed_capital=True, first_trade_bar=skip)
            out[tpl.name] = (sharpe(res, skip), forecast_position(df, tpl.channel_type, tpl.direction_logic))
        return out

    def test_random_walk_no_edge_and_a_quiet_countertrend(self):
        df = random_walk(3000, 11)
        out = self._by(df, 900)
        for name, (sh, p) in out.items():
            self.assertLess(abs(sh), 1.0, name)           # noise: a Sharpe of 1 over 2100 bars is 2 sigma
        ct = np.nanmean(np.abs(out["CT-fc-stance-chan-noreg-noV-noB"][1]))
        trend_on_trend = np.nanmean(np.abs(self._by(planted_trend(3000, 12), 900)["CT-fc-stance-chan-noreg-noV-noB"][1]))
        self.assertLess(ct, 0.6)
        self.assertLess(ct, trend_on_trend)      # it moves much more when there is something to learn

    def test_the_trend_template_does_not_churn(self):
        # turnover guard: the kernel feature moves the trend position by well under
        # 0.04 of a full size a bar (the five-indicator composite it replaced: ~0.10)
        for seed in (21, 22):
            p = forecast_position(random_walk(3000, seed), "forecast", "trend")
            self.assertLess(float(np.nanmean(np.abs(np.diff(p)))), 0.04)

    def test_planted_slow_trend_is_followed(self):
        df = planted_trend(3000, 12)
        out = self._by(df, 900)
        for name in ("TR-fc-stance-chan-noreg-noV-noB", "TR-fcx-stance-chan-noreg-noV-noB",
                     "LN-fc-stance-chan-noreg-noV-noB", "LN-fcx-stance-chan-noreg-noV-noB"):
            self.assertGreater(out[name][0], 0.5, name)

    def test_ou_spread_through_zero_learns_reversion(self):
        df = ou_spread(3000, 13)
        out = self._by(df, 900)
        for name in ("CT-fc-stance-chan-noreg-noV-noB", "LN-fc-stance-chan-noreg-noV-noB"):
            sh = out[name][0]
            self.assertGreater(sh, 0.0, name)
            d = forecast_diagnostics(df, next(t for t in TEMPLATES if t.name == name))
            b = d["betas"]["FAST"].iloc[forecast_warmup("forecast"):]
            self.assertLess(float(b.mean()), -0.005, name)
            self.assertLess(float((b > 0).mean()), 0.1, name)
        self.assertLess(out["TR-fc-stance-chan-noreg-noV-noB"][0], -0.5)     # the trend prior loses

    def test_regime_switch_flips_the_fast_coefficient(self):
        df = regime_switch(14)                                  # trend, OU, trend, OU (1500 bars each)
        tpl = next(t for t in TEMPLATES if t.name.startswith("LN-fc-"))
        b = forecast_diagnostics(df, tpl)["betas"]["FAST"]
        ends = [b.iloc[1499], b.iloc[2999], b.iloc[4499], b.iloc[5999]]   # the last bar of each block
        self.assertGreater(ends[0], 0.0)
        self.assertLess(ends[1], 0.0)
        self.assertGreater(ends[2], 0.0)
        self.assertLess(ends[3], 0.0)


class BufferTests(unittest.TestCase):
    def test_the_level_moves_only_outside_the_band_to_its_near_edge(self):
        p = np.array([np.nan, 0.05, 0.12, 0.3, 0.25, 0.1, -0.5, -0.45, 0.0])
        q = forecast_stance(p, "both")
        want = [0.0, 0.0, 0.02, 0.2, 0.2, 0.2, -0.4, -0.4, -0.1]
        np.testing.assert_allclose(q, want, atol=1e-12)
        self.assertEqual(FC_BUFFER, 0.1)
        # a one-sided template goes flat on any bar whose target has nothing on
        # its side (-0.5, -0.45 and 0.0 for long-only) and bands from there: no
        # memory of the short side, and two runs started apart agree from the
        # first such bar (the old rule banded the raw target and zeroed the
        # level afterwards, which encoded the short-side memory)
        np.testing.assert_allclose(forecast_stance(p, "long_only"),
                                   [0.0, 0.0, 0.02, 0.2, 0.2, 0.2, 0.0, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(forecast_stance(p, "short_only"),
                                   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.4, -0.4, 0.0], atol=1e-12)


class ValidateTests(unittest.TestCase):
    def test_forecast_channels_refuse_non_stance_entries_and_filters(self):
        for ch in FORECAST_CHANNELS:
            ok = StrategyTemplate("t", channel_type=ch, entry_style="stance")
            ok.validate()
            for bad in (dict(entry_style="stop"), dict(entry_style="close_confirm"), dict(entry_style="pullback"),
                        dict(regime_filter="trend_only"), dict(vol_filter=True), dict(bias_filter="sma")):
                with self.assertRaises(AssertionError, msg=f"{ch} {bad}"):
                    StrategyTemplate("t", channel_type=ch, **{"entry_style": "stance", **bad}).validate()

    def test_hedge_stance_still_valid(self):
        StrategyTemplate("t", channel_type="hedge_slow", entry_style="stance").validate()


class IndicatorTests(unittest.TestCase):
    def test_mmi_and_hurst_registered(self):
        for name in ("mmi", "hurst"):
            spec = REGIME_INDICATORS[name]
            self.assertIn(spec["threshold"], spec["thresholds"])
            self.assertGreater(len(spec["thresholds"]), 1)
        df = random_walk(800, 1)
        self.assertTrue(np.isfinite(REGIME_INDICATORS["mmi"]["fn"](df, 100).iloc[-1]))

    def test_mmi(self):
        rng = np.random.default_rng(3)
        walk = pd.Series(np.cumsum(rng.normal(0, 1, 4000)))
        self.assertAlmostEqual(float(mmi(walk).dropna().mean()), 75.0, delta=2.0)
        # persistent differences (a smooth trend): lower
        d = np.zeros(4000)
        for i in range(1, 4000):
            d[i] = 0.9 * d[i - 1] + rng.normal()
        smooth = pd.Series(np.cumsum(d))
        self.assertLess(float(mmi(smooth).dropna().mean()), 65.0)
        # reverting differences (an alternating series): higher
        e = rng.normal(0, 1, 4000)
        for i in range(1, 4000):
            e[i] += -0.7 * e[i - 1]
        alt = pd.Series(np.cumsum(e))
        self.assertGreater(float(mmi(alt).dropna().mean()), 75.0)
        # a constant added to every price changes nothing
        np.testing.assert_allclose(mmi(walk + 500.0).to_numpy(), mmi(walk).to_numpy(), equal_nan=True)

    def test_hurst(self):
        rng = np.random.default_rng(4)
        walk = pd.Series(np.cumsum(rng.normal(0, 1, 4000)))
        self.assertAlmostEqual(float(hurst(walk).dropna().mean()), 0.5, delta=0.05)
        d = np.zeros(4000)
        for i in range(1, 4000):
            d[i] = 0.7 * d[i - 1] + rng.normal()
        self.assertGreater(float(hurst(pd.Series(np.cumsum(d))).dropna().mean()), 0.6)
        ou = pd.Series(_ou_path(4000, rng, 10, 3.0) - 5.0)       # through zero
        self.assertLess(float(hurst(ou).dropna().mean()), 0.45)
        self.assertEqual(int(hurst(walk).first_valid_index()), 109)


if __name__ == "__main__":
    unittest.main()
