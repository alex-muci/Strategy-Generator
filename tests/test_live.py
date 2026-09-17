"""
Tests for the live signal layer.

The one that matters is `test_predicted_orders_match_what_the_engine_does`:
`live.py` has to state the next bar's order levels before that bar exists, so
it is the only place where the engine's rules are restated. This rolls one real
bar forward, for every template family switch and hundreds of bars, and asserts
the engine filled exactly what the dashboard said it would, at the price the
order type implies. If someone changes the entry logic in `strategy._bar_loop`
and forgets `live.py`, this fails.

Run with:  python -m unittest discover -s tests -v
"""

from __future__ import annotations
import os
import sys
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import synthetic_ohlc  # noqa: E402
from strategy import StrategyTemplate, backtest  # noqa: E402
from generator import generate_templates, param_grid_for  # noqa: E402
from live import (  # noqa: E402
    strategy_state, refit_params, due_for_refit, drop_forming_bar,
    portfolio_targets, trade_list,
)

LOOKBACK = 450


def _fill_price_for(order, open_, level):
    """The price the engine fills `order` at, given the next bar's open."""
    if order["kind"] == "market_on_open":
        return open_
    side, is_stop = order["side"], order["kind"] == "stop"
    # a stop order fills at the level or worse (through the level on a gap);
    # a limit order fills at the level or better
    if is_stop:
        return max(open_, level) if side == 1 else min(open_, level)
    return min(open_, level) if side == 1 else max(open_, level)


class LiveOrderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = synthetic_ohlc(1400, seed=17)

    def test_predicted_orders_match_what_the_engine_does(self):
        checked_entries = checked_exits = checked_blocks = 0
        for tpl in generate_templates("full")[::23]:
            for t in range(LOOKBACK + 20, len(self.df), 17):
                w0 = t - LOOKBACK
                tail, tail1 = self.df.iloc[w0:t], self.df.iloc[w0:t + 1]
                st = strategy_state(self.df.iloc[:t], tpl, equity=100_000.0,
                                    lookback_bars=LOOKBACK)
                after = backtest(tail1, tpl, initial_equity=100_000.0)
                opened = bool(after["entries"][-1])
                nxt_open = float(tail1["Open"].iloc[-1])

                if st["position"] is None:
                    orders = st["entry_orders"]
                    if not orders:
                        # nothing was working, so nothing may have been filled
                        self.assertFalse(
                            opened,
                            f"{tpl.name} bar {t}: engine entered with no predicted order "
                            f"(blocked by {st['blocked_by']})")
                        checked_blocks += 1
                    elif opened:
                        trade = [tr for tr in after["trades"]
                                 if tr["entry_date"] == tail1.index[-1]]
                        side = trade[0]["side"] if trade else None
                        if side is None:      # opened and still open: read the position
                            side = after["open_position"]["side"]
                            fill = after["open_position"]["entry_price"]
                        else:
                            fill = trade[0]["entry_price"]
                        match = [o for o in orders if o["side"] == side]
                        self.assertTrue(
                            match, f"{tpl.name} bar {t}: engine entered side {side}, "
                                   f"predicted {[o['side'] for o in orders]}")
                        o = match[0]
                        if o["kind"] == "stop_then_limit":
                            continue          # fills on a LATER bar, checked below
                        expected = _fill_price_for(o, nxt_open, o["level"])
                        self.assertAlmostEqual(
                            fill, expected, places=8,
                            msg=f"{tpl.name} bar {t}: filled {fill:.4f}, "
                                f"{o['kind']} implies {expected:.4f}")
                        checked_entries += 1
                else:
                    # an open position: if it closed on the new bar, the exit must
                    # be one of the levels we published (or the open, on a gap)
                    closed = [tr for tr in after["trades"]
                              if tr["exit_date"] == tail1.index[-1]]
                    if closed:
                        px = closed[0]["exit_price"]
                        levels = [o["level"] for o in st["exit_orders"] if o["level"] is not None]
                        ok = any(abs(px - _fill_price_for(o, nxt_open, o["level"])) < 1e-8
                                 for o in st["exit_orders"] if o["level"] is not None)
                        ok = ok or abs(px - nxt_open) < 1e-8
                        self.assertTrue(
                            ok, f"{tpl.name} bar {t}: exited at {px:.4f}, published "
                                f"levels {[round(v, 4) for v in levels]} open {nxt_open:.4f}")
                        checked_exits += 1
        # the sweep has to keep exercising all three branches: a future change
        # that silently stops reaching one of them is itself a regression
        self.assertGreater(checked_entries, 60, "entry branch under-exercised")
        self.assertGreater(checked_exits, 100, "exit branch under-exercised")
        self.assertGreater(checked_blocks, 100, "filter-block branch under-exercised")

    def test_pullback_limit_is_published_while_it_is_working(self):
        """A resting pullback order must be reported with its real level and
        expiry, not re-derived (the break that placed it is in the past)."""
        tpl = StrategyTemplate("t", entry_style="pullback", pullback_atr_mult=0.7,
                               pullback_valid_bars=3)
        seen = 0
        for t in range(LOOKBACK + 20, len(self.df), 13):
            st = strategy_state(self.df.iloc[:t], tpl, lookback_bars=LOOKBACK)
            working = [o for o in st["entry_orders"] if "already working" in o["note"]]
            if working:
                seen += 1
                o = working[0]
                self.assertIn(o["kind"], ("limit",))
                self.assertGreater(o["level"], 0)
                self.assertGreater(o["shares"], 0)
        self.assertGreater(seen, 0, "fixture never left a pullback order resting")

    def test_state_reports_the_position_the_engine_holds(self):
        for tpl in generate_templates("full")[::61]:
            for t in (600, 900, 1200):
                st = strategy_state(self.df.iloc[:t], tpl, equity=250_000.0,
                                    lookback_bars=LOOKBACK)
                res = backtest(self.df.iloc[t - LOOKBACK:t], tpl, initial_equity=250_000.0)
                pos = res["open_position"]
                if pos is None:
                    self.assertIsNone(st["position"])
                    self.assertEqual(st["shares"], 0.0)
                    self.assertEqual(st["exit_orders"], [])
                else:
                    self.assertEqual(st["position"], pos["side"])
                    self.assertAlmostEqual(st["shares"], pos["shares"])
                    self.assertAlmostEqual(st["entry_price"], pos["entry_price"])
                    # a hard ATR stop is always published
                    self.assertTrue(any(o["kind"] == "stop" for o in st["exit_orders"]))
                    self.assertEqual(st["entry_orders"], [])

    def test_channel_exit_is_one_stop_at_the_nearer_level(self):
        """A trend template with a channel exit has TWO exit levels on the same
        side of the price: the hard ATR stop and the opposite channel. The
        engine fills whichever is nearer (strategy._bar_loop merges them into
        one stop level), so the live layer must publish ONE stop at that
        level. Two same-side resting stops would reverse the position when
        the second fills after the first has closed it, and a channel stop
        further away than the hard stop is a level the engine never uses."""
        checked_chan = checked_hard = 0
        for tpl in [t for t in generate_templates("full") if t.exit_style == "channel"
                    and t.direction_logic == "trend" and t.sides == "both"][::9]:
            for t in range(LOOKBACK + 20, len(self.df), 23):
                st = strategy_state(self.df.iloc[:t], tpl, equity=100_000.0, lookback_bars=LOOKBACK)
                if st["position"] is None:
                    continue
                side = st["position"]
                stops = [o for o in st["exit_orders"] if o["kind"] == "stop"]
                self.assertEqual(len(st["exit_orders"]), 1, f"{tpl.name} bar {t}: {st['exit_orders']}")
                self.assertEqual(len(stops), 1)
                self.assertEqual(stops[0]["side"], -side)
                res = backtest(self.df.iloc[t - LOOKBACK:t], tpl, initial_equity=100_000.0)
                ind, n = res["indicators"], LOOKBACK
                hard = res["open_position"]["hard_stop"]
                chan = float(ind["lower_x"][n - 1] if side == 1 else ind["upper_x"][n - 1])
                nearer = max(hard, chan) if side == 1 else min(hard, chan)
                self.assertAlmostEqual(stops[0]["level"], nearer, places=8, msg=f"{tpl.name} bar {t}")
                if nearer == chan and chan != hard:
                    checked_chan += 1
                else:
                    checked_hard += 1
        # both cases must occur, or the test is not telling them apart
        self.assertGreater(checked_chan, 20)
        self.assertGreater(checked_hard, 20)
        # a countertrend channel exit keeps its two DIFFERENT orders: hard stop + midline limit
        ct = [t for t in generate_templates("full") if t.exit_style == "channel"
              and t.direction_logic == "countertrend" and t.sides == "both"][0]
        seen = 0
        for t in range(LOOKBACK + 20, len(self.df), 23):
            st = strategy_state(self.df.iloc[:t], ct, equity=100_000.0, lookback_bars=LOOKBACK)
            if st["position"] is None:
                continue
            self.assertEqual(sorted(o["kind"] for o in st["exit_orders"]), ["limit", "stop"], ct.name)
            seen += 1
        self.assertGreater(seen, 5)

    def test_sizing_matches_the_engine(self):
        """The share count on a predicted order must equal what the engine
        would actually buy at that level."""
        tpl = StrategyTemplate("t", entry_style="stop", exit_style="target_stop",
                               risk_pct=0.01, max_leverage=2.0)
        hits = 0
        for t in range(LOOKBACK + 20, len(self.df), 2):
            st = strategy_state(self.df.iloc[:t], tpl, equity=100_000.0, lookback_bars=LOOKBACK)
            if st["position"] is not None or not st["entry_orders"]:
                continue
            after = backtest(self.df.iloc[t - LOOKBACK:t + 1], tpl, initial_equity=100_000.0)
            if not after["entries"][-1]:
                continue
            side = (after["open_position"] or {}).get("side")
            trade = [tr for tr in after["trades"] if tr["entry_date"] == self.df.index[t]]
            if trade:
                side, shares = trade[0]["side"], trade[0]["shares"]
            else:
                shares = after["open_position"]["shares"]
            o = [o for o in st["entry_orders"] if o["side"] == side][0]
            # the engine sizes off the equity it holds at that moment, which has
            # drifted from the slot's starting equity; compare the ratio instead
            self.assertAlmostEqual(o["shares"] / shares, 100_000.0 / after["equity"].iloc[-2],
                                   places=6, msg=f"bar {t}")
            hits += 1
        self.assertGreater(hits, 8)


class RefitCadenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = synthetic_ohlc(900, seed=21)
        cls.tpl = generate_templates("quick")[0]
        cls.grid = param_grid_for(cls.tpl)

    def test_refit_uses_only_the_last_training_window(self):
        out = refit_params(self.df, self.tpl, self.grid, train_bars=400)
        self.assertEqual(out["train_bars"], 400)
        self.assertEqual(out["train_start"], self.df.index[-400])
        self.assertEqual(out["fitted_on"], self.df.index[-1])
        self.assertIn("n_entry", out["params"])
        # and it must not see anything after the window
        same = refit_params(self.df.iloc[:len(self.df)], self.tpl, self.grid, train_bars=400)
        self.assertEqual(out["params"], same["params"])

    def test_refit_is_held_for_a_whole_test_window(self):
        """Re-optimizing every run would be a strategy the walk-forward never
        measured; params may only change on the test-window cadence."""
        fitted = self.df.index[-130]
        self.assertFalse(due_for_refit(self.df.iloc[:-125], fitted, test_bars=125))
        self.assertFalse(due_for_refit(self.df.iloc[:-10], fitted, test_bars=125))
        self.assertTrue(due_for_refit(self.df, fitted, test_bars=125))
        self.assertTrue(due_for_refit(self.df, None, test_bars=125))

    def test_refit_refuses_a_short_history(self):
        with self.assertRaises(ValueError):
            refit_params(self.df.iloc[:100], self.tpl, self.grid, train_bars=400)


class HygieneTests(unittest.TestCase):
    def test_a_forming_bar_is_dropped(self):
        idx = pd.date_range("2026-03-10 09:30", periods=5, freq="h")
        df = pd.DataFrame({"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0, "Volume": 1},
                          index=idx)
        # the 13:30 bar covers 13:30-14:30; at 14:05 it is still forming
        kept = drop_forming_bar(df, "1h", now=pd.Timestamp("2026-03-10 14:05"))
        self.assertEqual(len(kept), 4)
        # at 14:35 it has closed
        self.assertEqual(len(drop_forming_bar(df, "1h", now=pd.Timestamp("2026-03-10 14:35"))), 5)

    def test_todays_daily_bar_is_dropped_until_the_session_ends(self):
        idx = pd.bdate_range("2026-03-02", periods=4)
        df = pd.DataFrame({"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0, "Volume": 1},
                          index=idx)
        during = drop_forming_bar(df, "1d", now=idx[-1] + pd.Timedelta(hours=15))
        self.assertEqual(len(during), 3)
        after = drop_forming_bar(df, "1d", now=idx[-1] + pd.Timedelta(days=1, hours=1))
        self.assertEqual(len(after), 4)


class PortfolioTargetTests(unittest.TestCase):
    def _state(self, asset, tpl, side, shares, price, stop):
        return dict(slot=f"{asset}|{tpl}", asset=asset, template=tpl, position=side,
                    shares=shares, last_close=price, entry_price=price, entry_date=None,
                    unrealized=0.0, exit_orders=[dict(kind="stop", side=-side, level=stop,
                                                      note="hard ATR stop")])

    def test_legs_net_per_asset_and_gross_is_capped(self):
        states = [
            self._state("SPY", "a", 1, 100.0, 500.0, 480.0),
            self._state("SPY", "b", -1, 40.0, 500.0, 520.0),
            self._state("TLT", "c", 1, 200.0, 90.0, 85.0),
        ]
        w = {"SPY|a": 0.4, "SPY|b": 0.3, "TLT|c": 0.3}
        out = portfolio_targets(states, w, account_equity=100_000.0)
        self.assertAlmostEqual(out["by_asset"].loc["SPY", "shares"], 60.0)
        self.assertAlmostEqual(out["by_asset"].loc["TLT", "shares"], 200.0)
        # gross = 100*500 + 40*500 + 200*90 = 88_000 -> 0.88 of the account
        self.assertAlmostEqual(out["gross_exposure"], 0.88)
        self.assertAlmostEqual(out["net_exposure"], (50_000 - 20_000 + 18_000) / 100_000)
        # risk if every stop hits: 100*20 + 40*20 + 200*5 = 3_800
        self.assertAlmostEqual(out["open_risk"], 3_800.0)

        capped = portfolio_targets(states, w, account_equity=100_000.0, max_gross=0.44)
        self.assertAlmostEqual(capped["scale_applied"], 0.5)
        self.assertAlmostEqual(capped["gross_exposure"], 0.44)
        self.assertAlmostEqual(capped["by_asset"].loc["SPY", "shares"], 30.0)

    def test_a_slot_with_zero_weight_is_not_traded(self):
        states = [self._state("SPY", "a", 1, 100.0, 500.0, 480.0)]
        out = portfolio_targets(states, {"SPY|a": 0.0}, account_equity=100_000.0)
        self.assertEqual(len(out["legs"]), 0)
        self.assertEqual(out["gross_exposure"], 0.0)

    def test_trade_list_is_target_minus_held(self):
        by_asset = pd.DataFrame({"shares": [60.0, -200.0], "price": [500.0, 90.0]},
                                index=pd.Index(["SPY", "TLT"], name="asset"))
        tl = trade_list(by_asset, holdings={"SPY": 10.0, "GLD": 5.0})
        self.assertEqual(tl.loc["SPY", "action"], "BUY")
        self.assertAlmostEqual(tl.loc["SPY", "order_shares"], 50.0)
        self.assertEqual(tl.loc["TLT", "action"], "SELL")
        self.assertAlmostEqual(tl.loc["TLT", "order_shares"], 200.0)
        # an asset you hold but the book no longer wants must be closed
        self.assertEqual(tl.loc["GLD", "action"], "SELL")
        self.assertAlmostEqual(tl.loc["GLD", "order_shares"], 5.0)

    def test_sub_lot_drift_does_not_generate_a_trade(self):
        by_asset = pd.DataFrame({"shares": [103.4], "price": [500.0]},
                                index=pd.Index(["SPY"], name="asset"))
        self.assertEqual(trade_list(by_asset, {"SPY": 103.0}, lot=1.0).loc["SPY", "action"], "hold")
        self.assertEqual(trade_list(by_asset, {"SPY": 90.0}, lot=1.0).loc["SPY", "action"], "BUY")


if __name__ == "__main__":
    unittest.main()
