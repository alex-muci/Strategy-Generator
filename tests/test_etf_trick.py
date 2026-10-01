"""
extra_utils/ETF_trick_spreads.py: one continuous series out of successive
listed calendar spreads, and the way it hands off to the engine.

The trick is run in POINTS (point_value=1, contracts=1, side=+1, k0=0,
roll_cost=0): the output is then the listed spread itself between rolls,
shifted by a constant, with each roll's gap folded in, and the engine gets the
contract multiplier once, as point_value. The cost of a roll is charged by the
engine to what it holds through a Roll bar (roll_cost_per_unit), long or short:
folded into the price it would be a gain to a short.
"""
from __future__ import annotations
import os
import sys
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from extra_utils.ETF_trick_spreads import etf_trick_listed_spreads  # noqa: E402
from strategy import StrategyTemplate, backtest  # noqa: E402
from data import load_csv  # noqa: E402


def _spread_ohlc(n: int, seed: int, level: float = -1.0, scale: float = 0.1):
    """A random-walk spread around `level` (through zero), with wicks."""
    rng = np.random.default_rng(seed)
    close = level + np.cumsum(rng.normal(0, scale, n))
    open_ = np.roll(close, 1); open_[0] = close[0]
    wick = np.abs(rng.normal(0, scale / 2, n))
    high = np.maximum(open_, close) + wick
    low = np.minimum(open_, close) - wick
    return open_, high, low, close


def _listed(dates, cols, seed=0):
    """O/H/L/C frames of several listed spreads on common dates."""
    O, H, L, C = ({} for _ in range(4))
    for k, col in enumerate(cols):
        o, h, l, c = _spread_ohlc(len(dates), seed + k, level=-1.0 + 0.5 * k)
        O[col], H[col], L[col], C[col] = o, h, l, c
    return tuple(pd.DataFrame(d, index=dates) for d in (O, H, L, C))


class EtfTrickTests(unittest.TestCase):
    def setUp(self):
        self.dates = pd.bdate_range("2024-01-01", periods=300)

    def test_a_single_unexpired_spread_is_a_pure_shift(self):
        """One listed spread, expiring after the data: from the bar after the
        entry the output is the input minus the first close (bar 0 is the
        flat bar the position is entered at the close of)."""
        O, H, L, C = _listed(self.dates, ["Z25-Z26"])
        out = etf_trick_listed_spreads(O, H, L, C, expiry={"Z25-Z26": "2030-12-18"},
                                       point_value=1.0, contracts=1.0, side=1, k0=0.0)
        c0 = float(C.iloc[0, 0])
        for col, src in (("Open", O), ("High", H), ("Low", L), ("Close", C)):
            np.testing.assert_allclose(out[col].to_numpy()[1:], src.iloc[1:, 0].to_numpy() - c0, rtol=1e-12, atol=1e-12)
        self.assertEqual(out.iloc[0][["Open", "High", "Low", "Close"]].tolist(), [0.0, 0.0, 0.0, 0.0])
        self.assertTrue((out["Held"] == "Z25-Z26").all())
        # the output crosses zero, as the spread does: nothing was made positive
        self.assertLess(float(out["Low"].min()), 0.0)
        self.assertGreater(float(out["High"].max()), 0.0)

    def test_the_short_side_swaps_high_and_low(self):
        O, H, L, C = _listed(self.dates, ["A"])
        out = etf_trick_listed_spreads(O, H, L, C, expiry={"A": "2030-12-18"}, side=-1)
        c0 = float(C.iloc[0, 0])
        np.testing.assert_allclose(out["High"].to_numpy()[1:], -(L.iloc[1:, 0].to_numpy() - c0), atol=1e-12)
        np.testing.assert_allclose(out["Low"].to_numpy()[1:], -(H.iloc[1:, 0].to_numpy() - c0), atol=1e-12)
        self.assertTrue((out["High"] >= out["Low"]).all())

    def test_the_roll_carries_the_pnl_and_charges_the_roll_cost_once(self):
        """Two spreads; the first expires inside the data. The held spread
        switches once, at the close of the first bar with <= roll_days left,
        the daily change of the output is the held spread's own change
        everywhere except the roll bar, which also pays the roll cost."""
        O, H, L, C = _listed(self.dates, ["Z25-Z26", "Z26-Z27"])
        expiry = {"Z25-Z26": str(self.dates[150].date()), "Z26-Z27": "2030-12-18"}
        out = etf_trick_listed_spreads(O, H, L, C, expiry=expiry, roll_days=10, roll_cost=0.25)
        held = out["Held"]
        switch = int(np.argmax((held != held.iloc[0]).to_numpy()))
        self.assertEqual(switch, 150 - 10 + 1)           # rolled at the close of the bar with 10 days left
        self.assertEqual(held.iloc[switch - 1], "Z25-Z26")
        # the Roll column marks that close, and only it
        self.assertEqual(list(np.flatnonzero(out["Roll"].to_numpy())), [switch - 1])
        self.assertTrue((held.iloc[switch:] == "Z26-Z27").all())
        d_out = out["Close"].diff().to_numpy()
        d1 = C["Z25-Z26"].diff().to_numpy()
        d2 = C["Z26-Z27"].diff().to_numpy()
        # up to and including the roll bar the output moves with the old spread
        np.testing.assert_allclose(d_out[1:switch], d1[1:switch], atol=1e-12)
        # the roll happened at the roll bar's close (re-anchored to the new
        # spread's close there), so the next bar moves with the new spread
        # and carries the roll cost as a gap
        self.assertAlmostEqual(d_out[switch], d2[switch] - 0.25, places=12)
        np.testing.assert_allclose(d_out[switch + 1:], d2[switch + 1:], atol=1e-12)
        # total: every held move, one roll cost
        self.assertAlmostEqual(float(out["Close"].iloc[-1]),
                               float(C["Z25-Z26"].iloc[switch - 1] - C["Z25-Z26"].iloc[0]
                                     + C["Z26-Z27"].iloc[-1] - C["Z26-Z27"].iloc[switch - 1]) - 0.25, places=10)

    def test_a_no_trade_day_is_a_flat_bar_and_the_gap_lands_on_the_next_print(self):
        O, H, L, C = _listed(self.dates, ["A"])
        for f in (O, H, L, C):
            f.iloc[40, 0] = 0.0                              # all-zero OHLC: no trade
        out = etf_trick_listed_spreads(O, H, L, C, expiry={"A": "2030-12-18"})
        prev = float(out["Close"].iloc[39])
        self.assertEqual(out.iloc[40][["Open", "High", "Low", "Close"]].tolist(), [prev] * 4)
        self.assertAlmostEqual(float(out["Close"].iloc[41]), prev + float(C.iloc[41, 0] - C.iloc[39, 0]), places=12)

    def test_the_engine_holds_through_the_roll_without_charging_the_roll_twice(self):
        """Feed the points series to the engine with the multiplier as
        point_value: a trade held across the roll earns point_value times the
        series' own move, less the engine's per-unit costs; the roll cost is
        already inside the series."""
        O, H, L, C = _listed(self.dates, ["Z25-Z26", "Z26-Z27"], seed=3)
        expiry = {"Z25-Z26": str(self.dates[150].date()), "Z26-Z27": "2030-12-18"}
        df = etf_trick_listed_spreads(O, H, L, C, expiry=expiry, roll_days=10, roll_cost=0.25)
        df = df[["Open", "High", "Low", "Close"]]
        # a template that is long from the first tradeable bar to a time exit 100 bars later
        tpl = StrategyTemplate("hold", entry_style="close_confirm", exit_style="time_stop", max_hold_bars=100,
                               n_entry=5, atr_n=5, atr_mult_stop=100.0, sides="long_only", regime_filter="none",
                               point_value=1000.0, cost_per_unit=15.0, cost_bps=0.0, margin_per_unit=3000.0,
                               risk_pct=0.5, max_leverage=0.03)     # the margin cap pins the size at 1 lot
        res = backtest(df, tpl, first_trade_bar=100)
        held = [t for t in res["trades"] if t["bars_held"] == 100]
        self.assertGreaterEqual(len(held), 1)
        t = held[0]
        i, j = df.index.get_loc(t["entry_date"]), df.index.get_loc(t["exit_date"])
        self.assertTrue(i < 141 < j, "the trade must straddle the roll bar")
        self.assertAlmostEqual(t["shares"], 1.0, places=9)
        self.assertAlmostEqual(t["pnl"], 1000.0 * (t["exit_price"] - t["entry_price"]) - 2 * 15.0, places=8)
        self.assertAlmostEqual(t["entry_price"], float(df["Open"].iloc[i]))
        self.assertAlmostEqual(t["exit_price"], float(df["Open"].iloc[j]))
        # tied back to the LISTED spreads: the old one from the entry open to
        # the roll close, the new one from the roll close to the exit open,
        # one roll cost, the engine's two per-unit costs, nothing else
        roll = 141 - 1
        listed = (float(C["Z25-Z26"].iloc[roll]) - float(O["Z25-Z26"].iloc[i])
                  + float(O["Z26-Z27"].iloc[j]) - float(C["Z26-Z27"].iloc[roll]))
        self.assertAlmostEqual(t["pnl"], 1000.0 * (listed - 0.25) - 30.0, places=8)



class RollCostTests(unittest.TestCase):
    """A roll costs the position held through it, whatever its side. The two
    templates below take the SAME bars on opposite sides: a long that follows
    the first upside break and a short that fades it, one lot each, held 100
    bars across the roll. Their P&Ls are mirror images apart from the costs,
    so long + short = -(every cost both paid)."""

    def setUp(self):
        self.dates = pd.bdate_range("2024-01-01", periods=300)
        O, H, L, C = _listed(self.dates, ["Z25-Z26", "Z26-Z27"], seed=3)
        self.expiry = {"Z25-Z26": str(self.dates[150].date()), "Z26-Z27": "2030-12-18"}
        self.listed = (O, H, L, C)

    def _pair(self, df, **kw):
        base = dict(entry_style="close_confirm", exit_style="time_stop", max_hold_bars=100, n_entry=5,
                    atr_n=5, atr_mult_stop=100.0, point_value=1000.0, cost_per_unit=15.0, cost_bps=0.0,
                    margin_per_unit=3000.0, risk_pct=0.5, max_leverage=0.03)   # the margin cap: 1 lot
        out = []
        for logic, sides in (("trend", "long_only"), ("countertrend", "short_only")):
            res = backtest(df, StrategyTemplate(logic, direction_logic=logic, sides=sides, **(base | kw)),
                           first_trade_bar=100)
            t = [t for t in res["trades"] if t["bars_held"] == 100][0]
            out.append(t)
        long_, short_ = out
        self.assertEqual((long_["side"], short_["side"]), (1, -1))
        self.assertEqual((long_["entry_date"], long_["exit_date"]), (short_["entry_date"], short_["exit_date"]))
        i, j = df.index.get_loc(long_["entry_date"]), df.index.get_loc(long_["exit_date"])
        self.assertTrue(i < 141 <= j, "the trades must straddle the roll bar")
        return long_, short_

    def test_folded_into_the_price_a_short_collects_the_roll_cost(self):
        """The issue's probe: the cost inside the series is a price move."""
        df = etf_trick_listed_spreads(*self.listed, expiry=self.expiry, roll_days=10, roll_cost=1.0)
        long_, short_ = self._pair(df[["Open", "High", "Low", "Close"]])
        # long + short pay only the four per-unit fills: the 1,000 roll cost
        # the long lost, the short gained
        self.assertAlmostEqual(long_["pnl"] + short_["pnl"], -4 * 15.0, places=6)

    def test_charged_by_the_engine_both_sides_pay_it(self):
        df = etf_trick_listed_spreads(*self.listed, expiry=self.expiry, roll_days=10)   # no cost in the price
        self.assertEqual(int(df["Roll"].sum()), 1)
        long_, short_ = self._pair(df[["Open", "High", "Low", "Close", "Roll"]], roll_cost_per_unit=1000.0)
        self.assertAlmostEqual(long_["pnl"] + short_["pnl"], -4 * 15.0 - 2 * 1000.0, places=6)
        for t in (long_, short_):
            self.assertAlmostEqual(t["cost"], 2 * 15.0 + 1000.0, places=9)
            self.assertAlmostEqual(t["pnl"], t["side"] * 1000.0 * (t["exit_price"] - t["entry_price"])
                                   - 2 * 15.0 - 1000.0, places=6)
        # without the Roll column the cost cannot be charged: refused, not ignored
        with self.assertRaises(ValueError):
            backtest(df[["Open", "High", "Low", "Close"]],
                     StrategyTemplate("t", cost_bps=0.0, margin_per_unit=3000.0, roll_cost_per_unit=1.0))

    def test_flat_through_the_roll_pays_nothing_and_the_log_shows_it(self):
        df = etf_trick_listed_spreads(*self.listed, expiry=self.expiry, roll_days=10)
        df = df[["Open", "High", "Low", "Close", "Roll"]]
        tpl = StrategyTemplate("t", entry_style="close_confirm", exit_style="time_stop", max_hold_bars=5,
                               n_entry=5, atr_n=5, cost_bps=0.0, margin_per_unit=3000.0, point_value=1000.0,
                               roll_cost_per_unit=1000.0)
        res = backtest(df, tpl, first_trade_bar=100, log_orders=True)
        held_at_roll = [t for t in res["trades"]
                        if df.index.get_loc(t["entry_date"]) <= 140 < df.index.get_loc(t["exit_date"])]
        rolls = res["orders"][res["orders"]["kind"] == "roll"]
        self.assertEqual(len(rolls), len(held_at_roll) + (res["open_position"] is not None
                                                         and df.index.get_loc(res["open_position"]["entry_date"]) <= 140))
        for t in res["trades"]:
            paid = t["cost"] - 0.0
            self.assertAlmostEqual(paid, 1000.0 * t["shares"] if t in held_at_roll else 0.0, places=6)

    def test_the_roll_column_survives_a_csv(self):
        import tempfile
        df = etf_trick_listed_spreads(*self.listed, expiry=self.expiry, roll_days=10)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "spread.csv")
            df[["Open", "High", "Low", "Close", "Roll"]].to_csv(path)
            back = load_csv(path)
        # (the trick's first bar is the all-zero entry bar, which load_csv drops as a no-trade row)
        np.testing.assert_array_equal(back["Roll"].to_numpy(), df["Roll"].reindex(back.index).to_numpy())
        self.assertEqual(float(back["Roll"].sum()), 1.0)


if __name__ == "__main__":
    unittest.main()
