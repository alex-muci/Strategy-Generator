"""
Exit ordering inside a bar.

The engine knows a bar's open, high, low and close, not the path between
them, so when a stop and a target are both reached intrabar it takes the
stop (the conservative reading). Two cases are NOT ambiguous, and the engine
must not resolve them conservatively:

* the OPEN already trades through the target: the open is the bar's first
  price, so a resting target (a limit) fills there, before anything else;
* on the entry bar, every price beyond a fill at the open, or beyond a stop
  entry's fill in the trade's direction, came after the fill: a target
  reached there was reached by the position.

A limit entry's bar stays ambiguous (its high may precede the fill): its
target waits for the next bar.
"""
from __future__ import annotations
import os
import sys
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategy import StrategyTemplate, backtest  # noqa: E402

K = 30          # the breakout bar


def _series(bars: dict, n: int = 60) -> pd.DataFrame:
    """Quiet bars (open = close = 99.9, high 100.9, low 98.9: ATR 2) under a
    20-bar Donchian channel of 98.8 / 101 set by one wider bar (bar 12), so
    nothing breaks out until a bar in `bars` ({i: (o, h, l, c)}) does."""
    o = np.full(n, 99.9); h = o + 1.0; l = o - 1.0; c = o.copy()
    h[12], l[12] = 101.0, 98.8
    for i, (oo, hh, ll, cc) in bars.items():
        o[i], h[i], l[i], c[i] = oo, hh, ll, cc
    return pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c},
                        index=pd.bdate_range("2025-01-01", periods=n))


def _tpl(**kw):
    # a breakout long at the 20-bar high (101), stop 3 ATR below (95), target
    # 2 ATR above (105); 1,000 shares on 100,000 (risk 6,000 = 6 % at the stop)
    base = dict(direction_logic="trend", channel_type="donchian", entry_style="stop", exit_style="target_stop",
                n_entry=20, atr_n=5, atr_mult_stop=3.0, atr_mult_target=2.0, risk_pct=0.06, max_leverage=10.0,
                cost_bps=0.0, sides="long_only")
    return StrategyTemplate("t", **(base | kw))


class TargetAtTheOpenTests(unittest.TestCase):
    def test_a_target_the_open_gapped_through_fills_at_the_open(self):
        """The reported case: long at 101, target 105, the next bar opens at
        110 and then trades down through the stop. The target was marketable
        at the open: +9,000, not a 6,000 stop loss."""
        df = _series({K: (100.8, 101.6, 100.2, 101.4), K + 1: (110.0, 111.0, 90.0, 92.0)})
        res = backtest(df, _tpl(), first_trade_bar=25)
        t = res["trades"][0]
        self.assertEqual((t["side"], t["entry_date"]), (1, df.index[K]))
        self.assertAlmostEqual(t["entry_price"], 101.0)
        self.assertAlmostEqual(t["shares"], 1000.0)
        self.assertEqual((t["reason"], t["exit_date"]), ("target", df.index[K + 1]))
        self.assertAlmostEqual(t["exit_price"], 110.0)
        self.assertAlmostEqual(t["pnl"], 9_000.0)

    def test_a_stop_the_open_gapped_through_still_fills_at_the_open(self):
        df = _series({K: (100.8, 101.6, 100.2, 101.4), K + 1: (90.0, 112.0, 89.0, 111.0)})
        t = backtest(df, _tpl(), first_trade_bar=25)["trades"][0]
        self.assertEqual((t["reason"], t["exit_price"]), ("stop", 90.0))

    def test_both_reached_inside_the_bar_the_stop_goes_first(self):
        """Opened between the two: the path is unknown, the stop is taken."""
        df = _series({K: (100.8, 101.6, 100.2, 101.4), K + 1: (101.0, 106.0, 94.0, 100.0)})
        t = backtest(df, _tpl(), first_trade_bar=25)["trades"][0]
        self.assertEqual((t["reason"], t["exit_price"]), ("stop", 95.0))

    def test_only_the_target_reached_inside_the_bar(self):
        df = _series({K: (100.8, 101.6, 100.2, 101.4), K + 1: (101.0, 106.0, 99.0, 100.0)})
        t = backtest(df, _tpl(), first_trade_bar=25)["trades"][0]
        self.assertEqual((t["reason"], t["exit_price"]), ("target", 105.0))

    def test_a_countertrend_midline_the_open_gapped_through(self):
        """A fade of the low (long at 98.8) with the channel exit: its target
        is the 20-bar midline (99.8); an open at 103 is through it, and the
        bar then trades far below the hard stop (92.8)."""
        df = _series({K: (99.5, 99.6, 98.6, 99.2), K + 1: (103.0, 104.0, 80.0, 81.0)})
        tpl = _tpl(direction_logic="countertrend", exit_style="channel", n_exit=20)
        t = backtest(df, tpl, first_trade_bar=25)["trades"][0]
        self.assertEqual((t["side"], t["entry_date"], t["entry_price"]), (1, df.index[K], 98.8))
        self.assertEqual((t["reason"], t["exit_date"], t["exit_price"]), ("midline", df.index[K + 1], 103.0))


class TargetOnTheEntryBarTests(unittest.TestCase):
    def test_a_stop_entry_takes_the_target_on_its_own_bar(self):
        """Price ran up THROUGH the buy stop at 101, so the 106 high came
        after the fill: the target at 105 was reached by the position."""
        df = _series({K: (100.8, 106.0, 100.2, 104.0), K + 1: (104.0, 104.5, 90.0, 91.0)})
        t = backtest(df, _tpl(), first_trade_bar=25)["trades"][0]
        self.assertEqual((t["entry_date"], t["exit_date"]), (df.index[K], df.index[K]))
        self.assertEqual((t["reason"], t["exit_price"]), ("target", 105.0))
        self.assertAlmostEqual(t["pnl"], 4_000.0)

    def test_a_fill_at_the_open_takes_the_target_on_its_own_bar(self):
        """A buy stop the open gapped through fills at the open (102): the
        whole bar follows the fill; the target is 102 + 4 = 106."""
        df = _series({K: (102.0, 107.0, 101.5, 106.5)})
        t = backtest(df, _tpl(), first_trade_bar=25)["trades"][0]
        self.assertEqual((t["entry_price"], t["exit_date"], t["reason"], t["exit_price"]),
                         (102.0, df.index[K], "target", 106.0))

    def test_stop_and_target_both_reached_on_the_entry_bar_the_stop_wins(self):
        df = _series({K: (100.8, 106.0, 94.0, 100.0)})
        t = backtest(df, _tpl(), first_trade_bar=25)["trades"][0]
        self.assertEqual((t["exit_date"], t["reason"], t["exit_price"]), (df.index[K], "stop_same_bar", 95.0))

    def test_a_limit_entry_leaves_its_target_for_the_next_bar(self):
        """A fade short at the 101 high (a sell limit): the bar's low at 96 may
        have come BEFORE price rose to the fill, so the 97 target is not
        taken on this bar."""
        df = _series({K: (98.0, 101.5, 96.0, 100.5)})
        tpl = _tpl(direction_logic="countertrend", sides="short_only")
        res = backtest(df, tpl, first_trade_bar=25)
        trades = res["trades"]
        first = trades[0] if trades else res["open_position"]
        self.assertEqual((first["side"], first["entry_date"], first["entry_price"]), (-1, df.index[K], 101.0))
        if trades:
            self.assertGreater(trades[0]["exit_date"], df.index[K])

    def test_the_order_log_shows_the_target_working_from_the_fill(self):
        df = _series({K: (100.8, 106.0, 100.2, 104.0)})
        o = backtest(df, _tpl(), first_trade_bar=25, log_orders=True)["orders"]
        bar = o[o["date"] == df.index[K]]
        self.assertEqual(list(bar["kind"]), ["entry_working", "entry_fill", "stop_working", "target_working",
                                             "exit_fill"])
        self.assertAlmostEqual(float(bar[bar["kind"] == "target_working"]["level"].iloc[0]), 105.0)


if __name__ == "__main__":
    unittest.main()


class TargetOnTheWrongSideTests(unittest.TestCase):
    def test_a_fill_already_through_its_target_exits_at_the_fill_on_the_entry_bar(self):
        """A countertrend long whose exit midline (a 40-bar channel) sits
        BELOW the fill at the open: the target is marketable at the fill, as
        it would be at the open of any later bar, and goes out there as a
        midline exit, before the intrabar stop."""
        B = 45
        o = np.full(70, 100.0); h = o + 1.0; l = o - 1.0; c = o.copy()
        for i in range(0, 26):                     # an older, lower range: the 40-bar midline sits near 91
            o[i] = c[i] = 82.0; h[i] = 83.0; l[i] = 81.0
        for i in range(26, 70):                    # rising lows: no bar touches the 5-bar low before the gap
            l[i] = 99.0 + 0.01 * (i - 26)
        o[B], h[B], l[B], c[B] = 95.0, 95.5, 94.0, 94.5     # gap below the 5-bar low (99): long at the open
        df = pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c}, index=pd.bdate_range("2025-01-01", periods=70))
        tpl = _tpl(direction_logic="countertrend", exit_style="channel", n_entry=5, n_exit=40, atr_mult_stop=0.3)
        t = backtest(df, tpl, first_trade_bar=B - 3)["trades"][0]
        self.assertEqual((t["side"], t["entry_price"], t["exit_date"]), (1, 95.0, df.index[B]))
        self.assertEqual((t["reason"], t["exit_price"]), ("midline", 95.0))


class EntryPrecedenceTests(unittest.TestCase):
    """A bar that reaches both channel levels (98.8 / 101): the order whose
    level the OPEN trades through went first; an open inside the channel
    leaves the order unknowable, and the long side is taken."""

    def _first(self, bar, **kw):
        t = backtest(_series({K: bar}), _tpl(sides="both", exit_style="time_stop", max_hold_bars=5, **kw),
                     first_trade_bar=25)["trades"][0]
        return t["side"], t["entry_price"]

    def test_a_sell_stop_the_open_gapped_through_goes_before_the_buy_stop(self):
        self.assertEqual(self._first((98.0, 102.0, 97.5, 100.0)), (-1, 98.0))

    def test_a_sell_limit_the_open_gapped_through_goes_before_the_buy_limit(self):
        self.assertEqual(self._first((102.0, 102.5, 98.0, 100.0), direction_logic="countertrend"), (-1, 102.0))

    def test_an_open_inside_the_channel_takes_the_long_side(self):
        self.assertEqual(self._first((100.0, 102.0, 97.5, 100.0)), (1, 101.0))
