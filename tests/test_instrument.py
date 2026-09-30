"""
The engine on an instrument that is not a share: a contract multiplier
(point_value), a cost per unit, a margin per unit, and prices that sit at or
below zero (a futures calendar spread).

The property everything here hangs on is SHIFT INVARIANCE: with the instrument
described in currency terms (margin and cost per unit, no bps of notional),
adding any constant to every price -- including one that drags the whole
series below zero -- must leave the trades, the P&L and the equity curve
unchanged. An engine that divides by a price anywhere fails it.
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
    StrategyTemplate, backtest, validate_instrument, atr, channel, hedge_direction, periods_per_year,
    REGIME_INDICATORS, REGIME_INDICATOR_NAMES,
)
from generator import generate_templates, param_grid_for  # noqa: E402
from walkforward import walk_forward  # noqa: E402


def _shift(df: pd.DataFrame, c: float) -> pd.DataFrame:
    out = df.copy()
    for col in ("Open", "High", "Low", "Close"):
        out[col] = df[col] + c
    return out


def _neg(df: pd.DataFrame) -> pd.DataFrame:
    """The whole series below zero."""
    return _shift(df, -(float(df["High"].max()) + 5.0))


def _flip(df: pd.DataFrame) -> pd.DataFrame:
    """p -> -p: highs and lows swap."""
    return pd.DataFrame({"Open": -df["Open"], "High": -df["Low"], "Low": -df["High"], "Close": -df["Close"],
                         "Volume": df.get("Volume", 1)}, index=df.index)


def _spread(tpl: StrategyTemplate, **kw) -> StrategyTemplate:
    """A Brent-like spread instrument: 1000 bbl per lot, $15 a lot a side, $3000 margin."""
    return tpl.with_params(cost_bps=0.0, cost_per_unit=15.0, margin_per_unit=3000.0, point_value=1000.0, **kw)


def _sample() -> list:
    """Templates spanning every switch, plus every sizing rule."""
    base = [t for t in generate_templates("full") if t.sides == "both"][::37]
    extra = [
        StrategyTemplate("hedge_learned", channel_type="hedge", direction_logic="learned", exit_style="atr_trail"),
        StrategyTemplate("wide_learned", channel_type="hedge_wide", direction_logic="learned", entry_style="pullback"),
        StrategyTemplate("wide_trend", channel_type="hedge_wide", direction_logic="trend", exit_style="channel"),
        StrategyTemplate("ct_boll", direction_logic="countertrend", channel_type="bollinger", exit_style="channel"),
        StrategyTemplate("vr_regime", regime_filter="trend_only", regime_indicator="vr"),
        StrategyTemplate("volf_bias", vol_filter=True, bias_filter="sma", exit_style="target_stop"),
        StrategyTemplate("vt", vol_target=0.15, exit_style="time_stop"),
        StrategyTemplate("vt_learned", vol_target=0.15, channel_type="hedge", direction_logic="learned"),
        StrategyTemplate("vt_pull", vol_target=0.15, entry_style="pullback", exit_style="atr_trail"),
    ]
    return base + extra


def _held_bars(res: dict, trade: dict, df: pd.DataFrame):
    i = df.index.get_loc(trade["entry_date"])
    j = df.index.get_loc(trade["exit_date"])
    return i, j


class ShiftInvarianceTests(unittest.TestCase):
    K = 120

    @classmethod
    def setUpClass(cls):
        cls.df = synthetic_ohlc(1100, seed=5)
        cls.sample = _sample()

    def _compare(self, a: dict, b: dict, c: float, name: str) -> int:
        np.testing.assert_allclose(a["equity"].to_numpy(), b["equity"].to_numpy(), rtol=1e-9,
                                   err_msg=f"{name}: equity differs after a shift of {c}")
        np.testing.assert_array_equal(a["entries"], b["entries"], err_msg=name)
        self.assertEqual(len(a["trades"]), len(b["trades"]), name)
        for x, y in zip(a["trades"], b["trades"]):
            self.assertEqual((x["entry_date"], x["exit_date"], x["side"], x["reason"]),
                             (y["entry_date"], y["exit_date"], y["side"], y["reason"]), name)
            self.assertAlmostEqual(x["shares"], y["shares"], delta=1e-9 * abs(x["shares"]), msg=name)
            # a price shifted by c carries c's rounding into a P&L in currency (x point_value)
            self.assertAlmostEqual(x["pnl"], y["pnl"], delta=1e-6 + 1e-9 * abs(x["pnl"]), msg=name)
            self.assertAlmostEqual(x["cost"], y["cost"], places=6, msg=name)
            self.assertAlmostEqual(x["entry_price"] + c, y["entry_price"], places=6, msg=name)
            self.assertAlmostEqual(x["exit_price"] + c, y["exit_price"], places=6, msg=name)
        return len(a["trades"])

    def test_backtest_is_shift_invariant(self):
        """Every template, every sizing rule, with the cap and the per-unit
        costs on: the same run on prices shifted below zero, across zero and
        far above their level. Any division by a price shows up here."""
        df = self.df
        shifts = [-(float(df["High"].max()) + 5.0), -float(df["Close"].median()), 500.0]
        checked = 0
        for tpl in self.sample:
            t = _spread(tpl)
            a = backtest(df, t, first_trade_bar=self.K)
            for c in shifts:
                b = backtest(_shift(df, c), t, first_trade_bar=self.K)
                checked += self._compare(a, b, c, f"{tpl.name} shift {c:+.1f}")
        self.assertGreater(checked, 300)

    def test_the_cap_and_the_costs_are_exercised_by_the_sample(self):
        """The invariance test must not pass by never sizing to the cap or
        never paying a cost: with the margin cap forced to bind, the shifted
        run still matches, and the costs are visibly paid."""
        df = self.df
        tpl = _spread(StrategyTemplate("t", vol_target=3.0), max_leverage=0.5)
        a = backtest(df, tpl, first_trade_bar=self.K)
        b = backtest(_neg(df), tpl, first_trade_bar=self.K)
        c = -(float(df["High"].max()) + 5.0)
        self._compare(a, b, c, "cap binding")
        eq = a["equity"]
        capped = [t for t in a["trades"]
                  if abs(t["shares"] * 3000.0 - 0.5 * eq.iloc[df.index.get_loc(t["entry_date"]) - 1]) < 1e-6]
        self.assertGreater(len(capped), 3)
        self.assertTrue(all(t["cost"] == 2 * 15.0 * t["shares"] for t in a["trades"]))

    def test_walk_forward_is_shift_invariant(self):
        """The whole walk-forward (grid search, plateau selection, stitched OOS
        returns) picks the same parameters and produces the same returns on
        the shifted series."""
        df = self.df
        tpl = _spread(StrategyTemplate("wf", exit_style="atr_trail"))
        grid = param_grid_for(tpl)
        a = walk_forward(df, tpl, grid, 400, 120)
        b = walk_forward(_shift(df, -float(df["Close"].median()) - 30.0), tpl, grid, 400, 120)
        self.assertGreater(len(a["windows"]), 2)
        self.assertEqual([w["params"] for w in a["windows"]], [w["params"] for w in b["windows"]])
        np.testing.assert_allclose(a["oos_returns"].to_numpy(), b["oos_returns"].to_numpy(), rtol=1e-9)
        self.assertGreater(a["summary"]["n_trades_oos"], 5)

    def test_indicators_are_shift_invariant(self):
        """Every regime indicator (the variance ratio included, now on price
        differences), the ATR and every channel, fitted and learned, move
        with the shift or not at all."""
        df = self.df
        c = -float(df["Close"].median()) - 20.0
        sh = _shift(df, c)
        for name in REGIME_INDICATOR_NAMES:
            spec = REGIME_INDICATORS[name]
            x = spec["fn"](df, spec["n"]).to_numpy()
            y = spec["fn"](sh, spec["n"]).to_numpy()
            self.assertGreater(np.isfinite(y).sum(), len(y) // 2, name)
            np.testing.assert_allclose(x, y, rtol=1e-8, atol=1e-10, err_msg=name)
        np.testing.assert_allclose(atr(df, 14).to_numpy(), atr(sh, 14).to_numpy(), rtol=1e-9)
        for kind in ("donchian", "keltner", "bollinger", "hedge", "hedge_wide"):
            for mode in ("trend", "countertrend"):
                a = channel(df, kind, 40, 2.0, 20, mode=mode, cost_pts=0.02)
                b = channel(sh, kind, 40, 2.0, 20, mode=mode, cost_pts=0.02)
                for u, v in zip(a, b):
                    np.testing.assert_allclose(u.to_numpy() + c, v.to_numpy(), rtol=1e-9, atol=1e-9,
                                               err_msg=f"{kind} {mode}")
        for ladder in ("hedge", "hedge_wide"):
            np.testing.assert_allclose(hedge_direction(df, 20, 0.0, ladder, "both", 0.02),
                                       hedge_direction(sh, 20, 0.0, ladder, "both", 0.02), rtol=1e-9, atol=1e-12)

    def test_a_learner_with_bps_costs_is_not_shift_invariant_but_per_unit_costs_are(self):
        """The reason a spread is costed per unit: a cost in basis points of
        the price is a different cost at every price level."""
        df = self.df
        sh = _shift(df, 300.0)
        per_unit = (hedge_direction(df, 20, 0.0, "hedge", "both", 0.05), hedge_direction(sh, 20, 0.0, "hedge", "both", 0.05))
        np.testing.assert_allclose(per_unit[0], per_unit[1], rtol=1e-9, atol=1e-12)
        bps = (hedge_direction(df, 20, 5.0, "hedge", "both"), hedge_direction(sh, 20, 5.0, "hedge", "both"))
        self.assertGreater(float(np.nanmax(np.abs(bps[0] - bps[1]))), 1e-6)


class ZeroCrossingTests(unittest.TestCase):
    """Hand-built bars with a trade that opens below zero and closes above it."""

    def _path(self):
        n = 35
        o = np.full(n, -2.0); c = o.copy(); h = o + 0.1; l = o - 0.1
        # bar 30 confirms a break of the 20-bar high (-1.9) at the close; the
        # market order fills at bar 31's open, -1.0; the time exit (3 bars) is
        # a market order at bar 34's open, +1.5
        o[30], h[30], l[30], c[30] = -2.0, -1.5, -2.1, -1.6
        o[31], h[31], l[31], c[31] = -1.0, -0.4, -1.2, -0.6
        o[32], h[32], l[32], c[32] = -0.5, 0.3, -0.7, 0.1
        o[33], h[33], l[33], c[33] = 0.2, 0.9, -0.1, 0.8
        o[34], h[34], l[34], c[34] = 1.5, 1.8, 1.2, 1.6
        return pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c},
                            index=pd.bdate_range("2025-01-01", periods=n))

    def _tpl(self, **kw):
        # the margin cap pins the size to exactly 2 lots: 0.06 x 100000 / 3000
        return StrategyTemplate("z", entry_style="close_confirm", exit_style="time_stop", max_hold_bars=3,
                                n_entry=20, atr_n=5, atr_mult_stop=50.0, risk_pct=0.5, max_leverage=0.06,
                                point_value=1000.0, margin_per_unit=3000.0, cost_per_unit=15.0, cost_bps=0.0, **kw)

    def test_known_long_trade_through_zero(self):
        res = backtest(self._path(), self._tpl(), first_trade_bar=30)
        self.assertEqual(len(res["trades"]), 1)
        t = res["trades"][0]
        self.assertEqual((t["side"], t["reason"], t["bars_held"]), (1, "time", 3))
        self.assertAlmostEqual(t["entry_price"], -1.0)
        self.assertAlmostEqual(t["exit_price"], 1.5)
        self.assertAlmostEqual(t["shares"], 2.0, places=9)
        self.assertAlmostEqual(t["cost"], 60.0)
        # 2 lots x 1000 bbl x (1.5 - (-1.0)) - 4 x 15
        self.assertAlmostEqual(t["pnl"], 2 * 1000.0 * 2.5 - 60.0)
        self.assertAlmostEqual(float(res["equity"].iloc[-1]), 100_000.0 + 4940.0)
        # marked to market through zero: the close of bar 32 is +0.1
        eq = res["equity"]
        self.assertAlmostEqual(float(eq.iloc[32]), 100_000.0 - 30.0 + 2 * 1000.0 * (0.1 - (-1.0)))
        self.assertAlmostEqual(res["stats"]["total_return"], 0.0494)

    def test_known_short_trade_is_its_mirror(self):
        res = backtest(_flip(self._path()), self._tpl(sides="short_only"), first_trade_bar=30)
        self.assertEqual(len(res["trades"]), 1)
        t = res["trades"][0]
        self.assertEqual(t["side"], -1)
        self.assertAlmostEqual(t["entry_price"], 1.0)
        self.assertAlmostEqual(t["exit_price"], -1.5)
        self.assertAlmostEqual(t["pnl"], 4940.0)
        self.assertAlmostEqual(float(res["equity"].iloc[-1]), 104_940.0)

    def test_per_bar_pnl_is_side_times_units_times_point_value_times_the_close_change(self):
        """The user's formula, bar by bar: while a trade is held, the equity
        moves by side x units x point_value x (close_t - close_{t-1}); the
        entry and exit bars carry the fill and half the cost each; the whole
        adds up to the trade's P&L."""
        df = _neg(synthetic_ohlc(1200, seed=3))
        checked = 0
        for tpl in (StrategyTemplate("t"), StrategyTemplate("ct", direction_logic="countertrend", exit_style="channel"),
                    StrategyTemplate("v", vol_target=0.15, exit_style="atr_trail"),
                    StrategyTemplate("h", channel_type="hedge", direction_logic="learned")):
            t = _spread(tpl)
            res = backtest(df, t, first_trade_bar=120)
            eq = res["equity"].to_numpy()
            close = df["Close"].to_numpy()
            for tr in res["trades"]:
                i, j = _held_bars(res, tr, df)
                if i == j:
                    continue      # a same-bar stop: fill and exit on one bar
                s, u, pv = tr["side"], tr["shares"], t.point_value
                half = tr["cost"] / 2
                self.assertAlmostEqual(eq[i] - eq[i - 1], s * u * pv * (close[i] - tr["entry_price"]) - half, places=6)
                for k in range(i + 1, j):
                    self.assertAlmostEqual(eq[k] - eq[k - 1], s * u * pv * (close[k] - close[k - 1]), places=6)
                self.assertAlmostEqual(eq[j] - eq[j - 1], s * u * pv * (tr["exit_price"] - close[j - 1]) - half, places=6)
                self.assertAlmostEqual(eq[j] - eq[i - 1], tr["pnl"], places=6)
                checked += 1
        self.assertGreater(checked, 40)


class PointValueTests(unittest.TestCase):
    K = 120

    @classmethod
    def setUpClass(cls):
        cls.df = synthetic_ohlc(1000, seed=8)

    def test_point_value_scales_the_units_not_the_equity(self):
        """A lot of 1000 barrels at $15 a side on $3000 margin is 1000 shares
        of a $0.015 cost on $3 margin: same equity, a thousandth of the units,
        under both sizing rules."""
        for tpl in (StrategyTemplate("t"), StrategyTemplate("v", vol_target=0.2), StrategyTemplate("c", vol_target=3.0, max_leverage=0.5)):
            lots = tpl.with_params(cost_bps=0.0, point_value=1000.0, cost_per_unit=15.0, margin_per_unit=3000.0)
            shares = tpl.with_params(cost_bps=0.0, point_value=1.0, cost_per_unit=0.015, margin_per_unit=3.0)
            a = backtest(self.df, lots, first_trade_bar=self.K)
            b = backtest(self.df, shares, first_trade_bar=self.K)
            np.testing.assert_allclose(a["equity"].to_numpy(), b["equity"].to_numpy(), rtol=1e-9, err_msg=tpl.name)
            self.assertGreater(len(a["trades"]), 5)
            for x, y in zip(a["trades"], b["trades"]):
                self.assertAlmostEqual(x["shares"] * 1000.0, y["shares"], delta=1e-9 * y["shares"])
                self.assertAlmostEqual(x["pnl"], y["pnl"], places=6)

    def test_the_default_instrument_is_byte_identical_to_the_cash_engine(self):
        """point_value 1, no per-unit cost, no margin: the same arithmetic as
        before the fields existed, under the ATR rule and with costs on."""
        base = StrategyTemplate("t", cost_bps=5.0)
        a = backtest(self.df, base)
        b = backtest(self.df, base.with_params(point_value=1.0, cost_per_unit=0.0, margin_per_unit=0.0))
        np.testing.assert_array_equal(a["equity"].to_numpy(), b["equity"].to_numpy())
        # and by hand: a share at price p costs 5 bps of p per side, and the
        # cap is on the notional
        t = a["trades"][0]
        self.assertAlmostEqual(t["cost"], 5e-4 * t["shares"] * (t["entry_price"] + t["exit_price"]), places=9)

    def test_unrealized_and_open_position_carry_the_point_value(self):
        df = self.df
        tpl = StrategyTemplate("t", exit_style="time_stop", max_hold_bars=500, point_value=50.0, cost_bps=0.0,
                               margin_per_unit=1000.0)
        res = backtest(df, tpl, first_trade_bar=self.K)
        pos = res["open_position"]
        self.assertIsNotNone(pos)
        self.assertAlmostEqual(pos["unrealized"], pos["side"] * pos["shares"] * 50.0 * (df["Close"].iloc[-1] - pos["entry_price"]))
        self.assertAlmostEqual(float(res["equity"].iloc[-1]), 100_000.0 + sum(t["pnl"] for t in res["trades"]) + pos["unrealized"] - pos["entry_cost"])


class CostAndMarginTests(unittest.TestCase):
    K = 120

    @classmethod
    def setUpClass(cls):
        cls.pos = synthetic_ohlc(1000, seed=21)
        cls.neg = _neg(cls.pos)

    def test_negative_prices_without_a_margin_are_refused(self):
        with self.assertRaises(ValueError) as cm:
            backtest(self.neg, StrategyTemplate("t", cost_bps=0.0, cost_per_unit=1.0))
        self.assertIn("margin_per_unit", str(cm.exception))
        with self.assertRaises(ValueError) as cm:
            validate_instrument(self.neg, StrategyTemplate("t", cost_bps=5.0, margin_per_unit=100.0))
        self.assertIn("cost_bps", str(cm.exception))
        # a positive series needs neither
        validate_instrument(self.pos, StrategyTemplate("t"))
        with self.assertRaises(ValueError):
            validate_instrument(self.pos, StrategyTemplate("t", point_value=0.0))

    def test_with_a_margin_the_spread_trades_and_nothing_is_rejected(self):
        """Before: a negative fill price made the notional cap negative and
        every entry was silently rejected."""
        tpl = _spread(StrategyTemplate("t"))
        res = backtest(self.neg, tpl, first_trade_bar=self.K, log_orders=True)
        self.assertGreater(len(res["trades"]), 5)
        self.assertEqual(int((res["orders"]["kind"] == "entry_reject").sum()), 0)
        self.assertTrue(all(t["entry_price"] < 0 for t in res["trades"]))

    def test_costs_are_never_negative(self):
        tpl = _spread(StrategyTemplate("t"))
        with_cost = backtest(self.neg, tpl, first_trade_bar=self.K)
        free = backtest(self.neg, tpl.with_params(cost_per_unit=0.0), first_trade_bar=self.K)
        self.assertTrue(all(t["cost"] > 0 for t in with_cost["trades"]))
        self.assertEqual([t["entry_date"] for t in with_cost["trades"]], [t["entry_date"] for t in free["trades"]])
        # the first trade is sized on the same cash in both runs: its cost is
        # exactly the equity gap at its exit (later sizes compound the gap)
        first = with_cost["trades"][0]
        j = self.neg.index.get_loc(first["exit_date"])
        self.assertAlmostEqual(float(free["equity"].iloc[j] - with_cost["equity"].iloc[j]), first["cost"], places=6)
        self.assertLess(float(with_cost["equity"].iloc[-1]), float(free["equity"].iloc[-1]))

    def test_the_cap_binds_on_the_margin(self):
        tpl = _spread(StrategyTemplate("t", vol_target=3.0), max_leverage=0.5)
        res = backtest(self.neg, tpl, first_trade_bar=self.K)
        eq = res["equity"]
        at_cap = 0
        for t in res["trades"]:
            i = self.neg.index.get_loc(t["entry_date"])
            self.assertLessEqual(t["shares"] * 3000.0, 0.5 * eq.iloc[i - 1] + 1e-6)
            at_cap += abs(t["shares"] * 3000.0 - 0.5 * eq.iloc[i - 1]) < 1e-6
        self.assertGreater(at_cap, 5)

    def test_without_a_margin_the_cap_is_on_the_notional(self):
        tpl = StrategyTemplate("t", vol_target=3.0, max_leverage=2.0, cost_bps=0.0, point_value=50.0)
        res = backtest(self.pos, tpl, first_trade_bar=self.K)
        eq = res["equity"]
        for t in res["trades"]:
            i = self.pos.index.get_loc(t["entry_date"])
            self.assertLessEqual(t["shares"] * 50.0 * t["entry_price"], 2.0 * eq.iloc[i - 1] + 1e-6)

    def test_a_fill_at_exactly_zero_without_a_margin_cannot_happen_and_with_one_is_fine(self):
        """A channel at 0.00 is a real level on a spread. Through zero with a
        margin the fill sizes as any other; the positivity check refuses the
        run without one before the loop divides by it."""
        df = _shift(self.pos, -float(self.pos["Low"].min()))    # the low touches exactly 0
        self.assertEqual(float(df["Low"].min()), 0.0)
        with self.assertRaises(ValueError):
            backtest(df, StrategyTemplate("t", cost_bps=0.0))
        res = backtest(df, _spread(StrategyTemplate("t")), first_trade_bar=self.K)
        self.assertGreater(len(res["trades"]), 5)
        self.assertTrue(np.isfinite(res["equity"].to_numpy()).all())


class VolTargetDollarVolTests(unittest.TestCase):
    K = 120

    def test_units_hit_the_dollar_vol_at_any_price_level(self):
        """units x point_value x sigma_points x sqrt(bars/year) = vol_target x
        equity, on the positive series and on the same series below zero."""
        pos = synthetic_ohlc(1000, seed=4)
        tpl = _spread(StrategyTemplate("v", vol_target=0.15, vol_target_n=60), max_leverage=1e9)
        for df in (pos, _neg(pos)):
            res = backtest(df, tpl, first_trade_bar=self.K)
            sig = df["Close"].diff().rolling(60).std()
            eq = res["equity"]
            n = 0
            for t in res["trades"]:
                i = df.index.get_loc(t["entry_date"])
                dollar_vol = t["shares"] * 1000.0 * sig.iloc[i - 1] * np.sqrt(periods_per_year())
                self.assertAlmostEqual(dollar_vol / eq.iloc[i - 1], 0.15, places=8)
                n += 1
            self.assertGreater(n, 5)

    def test_no_proxy_for_the_price_level_is_needed(self):
        """The same target on a series whose level is arbitrary (shifted by
        a large constant) sizes the same units: the rule never sees the level."""
        df = synthetic_ohlc(800, seed=6)
        tpl = _spread(StrategyTemplate("v", vol_target=0.2), max_leverage=1e9)
        a = backtest(df, tpl, first_trade_bar=self.K)["trades"]
        b = backtest(_shift(df, 10_000.0), tpl, first_trade_bar=self.K)["trades"]
        self.assertGreater(len(a), 5)
        for x, y in zip(a, b):
            self.assertAlmostEqual(x["shares"], y["shares"], delta=1e-9 * x["shares"])


if __name__ == "__main__":
    unittest.main()
