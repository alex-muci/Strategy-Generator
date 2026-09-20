"""
tests/test_futures_map.py
-------------------------
The ETF book restated in whole futures contracts: hedge ratio, rounding, the
no-trade buffer, level translation, and the quotes that must be refused.
"""

from __future__ import annotations
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from futures_map import (  # noqa: E402
    CONTRACTS, hedge_ratio, target_contracts, fut_level, to_contracts, translate_orders,
)


def _frame(close) -> pd.DataFrame:
    close = pd.Series(np.asarray(close, dtype=float),
                      index=pd.bdate_range("2024-01-01", periods=len(close)))
    return pd.DataFrame(dict(Open=close, High=close, Low=close, Close=close))


def _walk(n=300, vol=0.01, seed=0, start=100.0):
    r = np.random.default_rng(seed).normal(0, vol, n)
    return start * np.cumprod(1 + r), r


def _book(**notional) -> pd.DataFrame:
    return pd.DataFrame(dict(notional=pd.Series(notional, dtype=float)))


class HedgeRatioTests(unittest.TestCase):
    def test_the_same_series_is_one_to_one(self):
        px, _ = _walk()
        ratio, warn = hedge_ratio(_frame(px)["Close"], _frame(px * 50)["Close"])
        self.assertAlmostEqual(ratio, 1.0, places=9)
        self.assertIsNone(warn)

    def test_a_more_volatile_etf_needs_more_futures(self):
        _, r = _walk()
        etf, fut = 100 * np.cumprod(1 + 1.5 * r), 100 * np.cumprod(1 + r)
        ratio, _ = hedge_ratio(_frame(etf)["Close"], _frame(fut)["Close"])
        self.assertAlmostEqual(ratio, 1.5, delta=0.02)

    def test_a_roll_gap_does_not_move_it(self):
        px, _ = _walk()
        gapped = px.copy()
        gapped[250:] *= 1.08          # the splice of a contract roll
        clean, _ = hedge_ratio(_frame(px)["Close"], _frame(px)["Close"])
        ratio, _ = hedge_ratio(_frame(px)["Close"], _frame(gapped)["Close"])
        self.assertAlmostEqual(ratio, clean, delta=0.05)

    def test_short_history_falls_back_to_one_and_says_so(self):
        px, _ = _walk(n=30)
        ratio, warn = hedge_ratio(_frame(px)["Close"], _frame(px)["Close"])
        self.assertEqual(ratio, 1.0)
        self.assertIn("shared bars", warn)

    def test_an_unrelated_series_is_flagged(self):
        a, _ = _walk(seed=1)
        b, _ = _walk(seed=2)
        _, warn = hedge_ratio(_frame(a)["Close"], _frame(b)["Close"])
        self.assertIn("correlation", warn)


class RoundingTests(unittest.TestCase):
    def test_rounds_to_nearest_both_ways(self):
        self.assertEqual([target_contracts(x, 0) for x in (0.4, 0.5, 1.49, -0.6, -2.5, 0.0)],
                         [0, 1, 1, -1, -3, 0])

    def test_a_held_count_is_kept_inside_the_buffer(self):
        self.assertEqual(target_contracts(2.51, 2), 2)
        self.assertEqual(target_contracts(2.49, 3), 3)     # no flip-flop around x.5
        self.assertEqual(target_contracts(3.7, 2), 4)      # a real change goes through

    def test_flat_and_reversed_targets_ignore_the_buffer(self):
        self.assertEqual(target_contracts(0.0, 2), 0)
        self.assertEqual(target_contracts(-0.3, 2), 0)
        self.assertEqual(target_contracts(-1.2, 2), -1)

    def test_levels_keep_the_percentage_distance_over_the_ratio(self):
        # the ETF stop is 3% below; a future half as volatile sits 2% below
        self.assertAlmostEqual(fut_level(97.0, 100.0, 5000.0, 1.5, 0.25), 4900.0)
        self.assertIsNone(fut_level(None, 100.0, 5000.0, 1.0, 0.25))
        # and lands on the tick grid
        lvl = fut_level(98.7654, 100.0, 5123.0, 1.0, 0.25)
        self.assertAlmostEqual(lvl / 0.25, round(lvl / 0.25), places=9)


class BookTests(unittest.TestCase):
    def setUp(self):
        px, _ = _walk()
        self.etf = {"SPY": _frame(px), "IEF": _frame(px), "XYZ": _frame(px)}
        self.fut = {"SPY": _frame(px / px[-1] * 6000.0),      # one MES = $30,000
                    "IEF": _frame(px / px[-1] * 110.0)}       # one ZN  = $110,000

    def test_notional_becomes_whole_contracts(self):
        out = to_contracts(_book(SPY=70_000, IEF=-150_000), self.etf, self.fut)
        b = out["book"]
        self.assertEqual((b.loc["SPY", "root"], b.loc["SPY", "target"]), ("MES", 2))
        self.assertEqual((b.loc["IEF", "root"], b.loc["IEF", "target"]), ("ZN", -1))
        self.assertEqual(list(b["action"]), ["BUY", "SELL"])
        self.assertAlmostEqual(b.loc["SPY", "raw"], 70 / 30)
        # 2 MES hold $10k less than wanted, 1 ZN short holds $40k less
        self.assertAlmostEqual(out["rounding_error"], 10_000 + 40_000, delta=1)
        self.assertAlmostEqual(out["rounding_error_pct"], 50 / 220, places=3)

    def test_holdings_are_by_root_and_the_buffer_applies(self):
        out = to_contracts(_book(SPY=76_000), self.etf, self.fut, held={"MES": 2, "ZN": 1})
        b = out["book"]
        self.assertEqual((b.loc["SPY", "target"], b.loc["SPY", "action"]), (2, "hold"))   # 2.53 wanted
        # a contract still held in a market the book has left is closed
        self.assertEqual((b.loc["IEF", "target"], b.loc["IEF", "action"], b.loc["IEF", "order_contracts"]),
                         (0, "SELL", 1))

    def test_an_unmapped_asset_is_reported_not_guessed(self):
        out = to_contracts(_book(XYZ=50_000), self.etf, self.fut)
        self.assertTrue(out["book"].empty)
        self.assertTrue(any("XYZ" in n and "no futures contract" in n for n in out["notes"]))

    def test_a_mis_scaled_quote_is_refused(self):
        self.fut["SPY"] = self.fut["SPY"] / 100.0             # $300 a contract: not an MES
        out = to_contracts(_book(SPY=70_000), self.etf, self.fut)
        self.assertNotIn("SPY", out["book"].index)
        self.assertTrue(any("mis-scaled" in n for n in out["notes"]))

    def test_a_missing_price_is_reported(self):
        del self.fut["IEF"]
        out = to_contracts(_book(IEF=150_000), self.etf, self.fut)
        self.assertTrue(out["book"].empty)
        self.assertTrue(any("no price" in n for n in out["notes"]))

    def test_a_flat_book_keeps_its_columns(self):
        out = to_contracts(_book(), self.etf, self.fut)
        self.assertTrue(out["book"].empty)
        self.assertIn("target", out["book"].columns)
        self.assertEqual(out["rounding_error_pct"], 0.0)

    def test_working_levels_are_restated_on_the_future(self):
        conv = to_contracts(_book(SPY=70_000), self.etf, self.fut)["conversions"]
        close = float(self.etf["SPY"]["Close"].iloc[-1])
        states = [
            dict(slot="SPY|a", asset="SPY", position=1, shares=700.0,
                 exit_orders=[dict(kind="stop", side=-1, level=close * 0.97, note="")],
                 entry_orders=[]),
            dict(slot="SPY|b", asset="SPY", position=None, shares=0.0, exit_orders=[],
                 entry_orders=[dict(kind="stop", side=1, level=close * 1.02, shares=300.0, note=""),
                               dict(kind="market_on_open", side=1, level=None, shares=300.0, note="")]),
            dict(slot="XYZ|a", asset="XYZ", position=None, shares=0.0, exit_orders=[], entry_orders=[]),
        ]
        o = translate_orders(states, conv)
        self.assertEqual(list(o["what"]), ["exit", "entry", "entry"])
        self.assertEqual(list(o["side"]), [-1, 1, 1])
        self.assertAlmostEqual(o["fut_level"][0], 6000 * 0.97, delta=0.25)
        self.assertAlmostEqual(o["fut_level"][1], 6000 * 1.02, delta=0.25)
        self.assertTrue(pd.isna(o["fut_level"][2]))
        self.assertAlmostEqual(o["contracts_raw"][0], 700 * close * 0.97 / 30_000, places=6)

    def test_the_table_is_consistent(self):
        roots = [c.root for c in CONTRACTS.values()]
        self.assertEqual(len(roots), len(set(roots)))
        for etf, c in CONTRACTS.items():
            self.assertEqual(etf, c.etf)
            self.assertGreater(c.multiplier, 0)
            self.assertGreater(c.tick, 0)
            self.assertLess(c.notional_range[0], c.notional_range[1])


if __name__ == "__main__":
    unittest.main()
