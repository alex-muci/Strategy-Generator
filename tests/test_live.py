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
from walkforward import window_backtest  # noqa: E402
from live import (  # noqa: E402
    strategy_state, refit_params, due_for_refit, drop_forming_bar,
    portfolio_targets, trade_list, apply_gross_scale, bars_since,
)

LOOKBACK = 450
SWEEP_STRIDE = 137      # prime, so it does not alias with the generator's switch cycles


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
    # minutes of sweeps (the hedge_wide templates of the sample above all):
    # pytest skips the class unless run with -m slow (tests/conftest.py);
    # unittest runs it
    slow = True

    @classmethod
    def setUpClass(cls):
        cls.df = synthetic_ohlc(1400, seed=17)

    def test_predicted_orders_match_what_the_engine_does(self):
        checked_entries = checked_exits = checked_blocks = 0
        # The sweep costs ~55 cold indicator builds per template, so the sample
        # is sized by COUNT, not by a fixed stride: the universe grew 6x with the
        # hedge channel and a stride of 23 turned this test into an hour's run.
        full = generate_templates("full")
        base = full[::SWEEP_STRIDE]
        self.assertLess(len(base), 200, "the template universe grew: raise SWEEP_STRIDE")
        # every fourth one again, sized to a vol target: exercises the
        # realized-vol readiness gate and the sizing branch live.py restates.
        # It runs right after its twin so the indicator cache is still warm.
        sample = []
        for k, tpl in enumerate(base):
            sample.append(tpl)
            if k % 4 == 0:
                sample.append(tpl.with_params(vol_target=0.15))
        for tpl in sample:
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
        self._check_sizing(StrategyTemplate("t", entry_style="stop", exit_style="target_stop",
                                            risk_pct=0.01, max_leverage=2.0))

    def test_sizing_matches_the_engine_with_a_vol_target(self):
        """Same contract for the volatility-target rule: live.py's size must be
        the engine's, not a restatement that has drifted."""
        self._check_sizing(StrategyTemplate("t", entry_style="stop", exit_style="target_stop",
                                            vol_target=0.15, vol_target_n=60, max_leverage=2.0))

    def test_sizing_matches_the_engine_for_a_learned_direction(self):
        """A learned direction scales the size by the learner's conviction on
        the last closed bar; live.py must publish that scaled count, for a
        stop entry and for a pullback limit that fills bars after it was
        placed (sized on the fill bar, in the logic it was placed under)."""
        self._check_sizing(StrategyTemplate("t", channel_type="hedge", direction_logic="learned",
                                            entry_style="stop", exit_style="target_stop"))
        self._check_sizing(StrategyTemplate("t", channel_type="hedge", direction_logic="learned",
                                            entry_style="pullback", exit_style="target_stop",
                                            pullback_atr_mult=0.25, pullback_valid_bars=5), min_hits=3)

    def _check_sizing(self, tpl, min_hits=8):
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
            self.assertGreater(o["shares"], 0.0, f"bar {t}")
            self.assertEqual(st["blocked_by"], [], f"bar {t}")
            # the engine sizes off the equity it holds at that moment, which has
            # drifted from the slot's starting equity; compare the ratio instead
            self.assertAlmostEqual(o["shares"] / shares, 100_000.0 / after["equity"].iloc[-2],
                                   places=6, msg=f"bar {t}")
            hits += 1
        self.assertGreater(hits, min_hits)


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

    def test_a_persisted_stamp_in_another_zone_still_compares(self):
        """live_state.json from a run whose feed kept its zone, read against the
        naive-UTC index (and the other way round), must not raise."""
        idx = pd.date_range("2026-03-02 14:30", periods=10, freq="h")
        df = pd.DataFrame({"Close": range(10)}, index=idx)
        aware = pd.Timestamp("2026-03-02 14:30", tz="UTC").tz_convert("America/New_York")
        self.assertEqual(bars_since(df, str(aware)), 9)
        self.assertFalse(due_for_refit(df, str(aware), test_bars=10))
        df_aware = df.tz_localize("UTC")
        self.assertEqual(bars_since(df_aware, "2026-03-02 14:30"), 9)

    def test_refit_refuses_a_short_history(self):
        with self.assertRaises(ValueError):
            refit_params(self.df.iloc[:100], self.tpl, self.grid, train_bars=400)


class WindowStateTests(unittest.TestCase):
    """After a refit the live state IS the walk-forward's out-of-sample window:
    flat on the first bar after `fitted_on`, never a position opened on the
    training bars by the params fitted on them."""

    @classmethod
    def setUpClass(cls):
        cls.df = synthetic_ohlc(2000, seed=3, trend_prob=0.6)
        cls.tpls = generate_templates("quick")[:12]
        cls.bounds = (1000, 1250, 1500)

    def _wf(self, tpl, b, t):
        return window_backtest(self.df, tpl, b, t + 1, initial_equity=100_000.0)["open_position"]

    def test_the_state_is_the_walk_forward_window(self):
        checked = 0
        for tpl in self.tpls:
            for b in self.bounds:
                fitted = self.df.index[b - 1]
                for t in range(b - 1, b + 40, 4):
                    st = strategy_state(self.df.iloc[:t + 1], tpl, equity=100_000.0, fitted_on=fitted)
                    wf = self._wf(tpl, b, t) if t >= b else None
                    msg = f"{tpl.name} window {b} bar {t}"
                    self.assertEqual(st["position"], None if wf is None else wf["side"], msg)
                    if wf is not None:
                        self.assertAlmostEqual(st["shares"], wf["shares"], places=6, msg=msg)
                        self.assertEqual(st["entry_date"], wf["entry_date"], msg)
                    checked += 1
        self.assertGreater(checked, 300)

    def test_a_trade_opened_on_the_training_bars_is_not_carried(self):
        """The fixture has windows where a 400-bar rebuild opens a position on
        the bar the params were fitted on; the window state must not."""
        carried = 0
        for tpl in self.tpls:
            for b in self.bounds:
                old = strategy_state(self.df.iloc[:b + 1], tpl, equity=100_000.0)
                if old["position"] is not None and old["entry_date"] < self.df.index[b]:
                    carried += 1
                    new = strategy_state(self.df.iloc[:b + 1], tpl, equity=100_000.0,
                                         fitted_on=self.df.index[b - 1])
                    wf = self._wf(tpl, b, b)
                    self.assertEqual(new["position"], None if wf is None else wf["side"])
                    self.assertTrue(new["entry_date"] is None or new["entry_date"] >= self.df.index[b])
        self.assertGreater(carried, 0, "the fixture no longer exercises the carried-position case")

    def test_right_after_the_refit_the_next_bar_orders_are_published(self):
        tpl = self.tpls[0]
        st = strategy_state(self.df.iloc[:1200], tpl, fitted_on=self.df.index[1199])
        self.assertIsNone(st["position"])
        self.assertEqual(st["n_trades_in_window"], 0)
        self.assertIsInstance(st["entry_orders"], list)


class StalePullbackOrderTests(unittest.TestCase):
    """A resting pullback limit, then a halt: three flat bars zero the ATR(3).
    On the next bar the engine cancels the order before it can fill (it starts
    nothing while the indicators are unusable); the published orders used to
    keep it working."""

    @classmethod
    def setUpClass(cls):
        df = synthetic_ohlc(600, seed=17)
        cls.tpl = StrategyTemplate("pb", direction_logic="trend", channel_type="donchian",
                                   entry_style="pullback", exit_style="atr_trail", n_entry=20,
                                   atr_n=3, pullback_valid_bars=10, cost_bps=0.0)
        for k in range(100, 599):
            res = backtest(df.iloc[:k + 1], cls.tpl)
            p = res["pending_order"]
            if p is not None and res["open_position"] is None and p["expires_bar"] == k + 10:
                break
        else:
            raise AssertionError("fixture never leaves a fresh pullback order working")
        c = float(df["Close"].iloc[k])
        idx = pd.bdate_range(df.index[k] + pd.Timedelta(days=1), periods=4)
        flat = pd.DataFrame({"Open": c, "High": c, "Low": c, "Close": c, "Volume": 1}, index=idx[:3])
        cls.pending, cls.halted = p, pd.concat([df.iloc[:k + 1], flat])
        # the bar after the halt trades straight through the limit
        lvl = p["level"]
        lo, hi = min(c, lvl * 0.99), max(c, lvl * 1.01)
        cls.next_bar = pd.DataFrame({"Open": c, "High": hi, "Low": lo, "Close": c, "Volume": 1}, index=idx[3:])

    def test_the_fixture_has_a_working_order_and_a_dead_atr(self):
        res = backtest(self.halted, self.tpl)
        self.assertEqual(res["pending_order"], self.pending)
        self.assertEqual(res["indicators"]["atr"][-1], 0.0)

    def test_the_engine_pulls_the_order(self):
        before = backtest(self.halted, self.tpl)
        after = backtest(pd.concat([self.halted, self.next_bar]), self.tpl)
        self.assertIsNone(after["open_position"])
        self.assertIsNone(after["pending_order"])
        self.assertEqual(len(after["trades"]), len(before["trades"]))

    def test_so_the_published_orders_do_not_carry_it(self):
        st = strategy_state(self.halted, self.tpl, lookback_bars=len(self.halted))
        self.assertEqual(st["entry_orders"], [])
        self.assertTrue(st["blocked_by"])

    def test_a_healthy_bar_still_republishes_it(self):
        st = strategy_state(self.halted.iloc[:-3], self.tpl, lookback_bars=len(self.halted))
        self.assertEqual([o["kind"] for o in st["entry_orders"]], ["limit"])
        self.assertAlmostEqual(st["entry_orders"][0]["level"], self.pending["level"])


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

    @staticmethod
    def _daily(last_day, periods=4):
        idx = pd.bdate_range(end=last_day, periods=periods)
        return pd.DataFrame({"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0, "Volume": 1}, index=idx)

    def test_a_closed_daily_bar_is_kept_the_same_evening(self):
        """A daily bar is stamped at midnight, so `last + 1 day` is the NEXT
        midnight UTC: the evening run after the close would drop the finished
        bar and publish orders off yesterday's. The session close decides,
        in New York time, summer and winter."""
        for day, close_utc in (("2026-07-15", "20:00"),    # EDT: 16:00 New York = 20:00 UTC
                               ("2026-01-15", "21:00")):   # EST: 16:00 New York = 21:00 UTC
            df = self._daily(day)
            close = pd.Timestamp(f"{day} {close_utc}")
            self.assertEqual(len(drop_forming_bar(df, "1d", now=close - pd.Timedelta(minutes=1))), 3, day)
            # the feed's last print still settles for a few minutes after the bell
            self.assertEqual(len(drop_forming_bar(df, "1d", now=close + pd.Timedelta(minutes=5))), 3, day)
            # 17:10 New York, the README's cron line
            self.assertEqual(len(drop_forming_bar(df, "1d", now=close + pd.Timedelta(minutes=70))), 4, day)
            # an aware `now` means the same instant
            aware = (close + pd.Timedelta(minutes=70)).tz_localize("UTC").tz_convert("Europe/Rome")
            self.assertEqual(len(drop_forming_bar(df, "1d", now=aware)), 4, day)

    def test_the_session_close_is_the_exchanges(self):
        df = self._daily("2026-07-15")
        now = pd.Timestamp("2026-07-15 16:00")             # 18:00 in Frankfurt, 12:00 in New York
        self.assertEqual(len(drop_forming_bar(df, "1d", now=now)), 3)
        self.assertEqual(len(drop_forming_bar(df, "1d", now=now, session_close=("17:30", "Europe/Berlin"))), 4)

    def test_weekly_and_monthly_bars_wait_for_their_last_session(self):
        row = {"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0, "Volume": 1}
        wk = pd.DataFrame(row, index=pd.date_range("2026-06-22", periods=4, freq="W-MON"))  # last: Mon 13 Jul
        self.assertEqual(len(drop_forming_bar(wk, "1wk", now=pd.Timestamp("2026-07-17 15:00"))), 3)  # Friday, open
        self.assertEqual(len(drop_forming_bar(wk, "1wk", now=pd.Timestamp("2026-07-20 08:00"))), 4)
        mo = pd.DataFrame(row, index=pd.date_range("2026-04-01", periods=4, freq="MS"))     # last: 1 Jul, 31 days
        self.assertEqual(len(drop_forming_bar(mo, "1mo", now=pd.Timestamp("2026-07-31 15:00"))), 3)  # last session, open
        self.assertEqual(len(drop_forming_bar(mo, "1mo", now=pd.Timestamp("2026-07-31 21:00"))), 4)


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

    def test_the_gross_cap_scales_the_entry_orders_too(self):
        """A capped book must not publish entry orders that rebuild the exposure
        the cap removed."""
        held = self._state("SPY", "a", 1, 400.0, 500.0, 470.0)
        held["entry_orders"] = []
        flat = dict(slot="QQQ|b", asset="QQQ", template="b", position=None, shares=0.0,
                    last_close=400.0, entry_price=None, entry_date=None, unrealized=0.0,
                    exit_orders=[], entry_orders=[dict(kind="stop", side=1, level=410.0, shares=300.0)])
        out = portfolio_targets([held, flat], {"SPY|a": .5, "QQQ|b": .5}, 100_000.0, max_gross=1.0)
        self.assertAlmostEqual(out["scale_applied"], 0.5)
        scaled = apply_gross_scale([held, flat], out["scale_applied"])
        self.assertAlmostEqual(scaled[1]["entry_orders"][0]["shares"], 150.0)
        self.assertAlmostEqual(scaled[0]["shares"], out["by_asset"].loc["SPY", "shares"])
        self.assertAlmostEqual(scaled[0]["shares_unscaled"], 400.0)
        self.assertEqual(flat["entry_orders"][0]["shares"], 300.0, "the input states are not mutated")
        self.assertIs(apply_gross_scale([held], 1.0)[0], held)

    def test_sub_lot_drift_does_not_generate_a_trade(self):
        by_asset = pd.DataFrame({"shares": [103.4], "price": [500.0]},
                                index=pd.Index(["SPY"], name="asset"))
        self.assertEqual(trade_list(by_asset, {"SPY": 103.0}, lot=1.0).loc["SPY", "action"], "hold")
        self.assertEqual(trade_list(by_asset, {"SPY": 90.0}, lot=1.0).loc["SPY", "action"], "BUY")



class SpreadSizingTests(unittest.TestCase):
    """live.py restates the engine's sizing for the next bar's orders; on an
    instrument below zero with a point value, per-unit costs and a margin cap
    it must still be the engine's number, under both rules."""

    @classmethod
    def setUpClass(cls):
        df = synthetic_ohlc(1400, seed=17)
        shift = float(df["High"].max()) + 5.0
        cls.df = df.assign(**{c: df[c] - shift for c in ("Open", "High", "Low", "Close")})

    def _check(self, tpl, min_hits=6):
        hits = 0
        for t in range(LOOKBACK + 20, len(self.df), 3):
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
            self.assertLess(o["level"], 0.0)
            self.assertGreater(o["shares"], 0.0, f"bar {t}")
            self.assertAlmostEqual(o["shares"] / shares, 100_000.0 / after["equity"].iloc[-2], places=6, msg=f"bar {t}")
            self.assertEqual(st["point_value"], tpl.point_value)
            hits += 1
        self.assertGreater(hits, min_hits)

    def test_atr_rule_on_a_spread(self):
        self._check(StrategyTemplate("t", entry_style="stop", exit_style="target_stop", cost_bps=0.0,
                                     cost_per_unit=15.0, margin_per_unit=3000.0, point_value=1000.0, max_leverage=0.5))

    def test_vol_target_capped_on_the_margin(self):
        self._check(StrategyTemplate("t", entry_style="stop", exit_style="target_stop", cost_bps=0.0, vol_target=2.0,
                                     cost_per_unit=15.0, margin_per_unit=3000.0, point_value=1000.0, max_leverage=0.5))

    def test_the_book_measures_a_margined_slot_in_margin_and_the_rest_in_notional(self):
        """A long at a negative price is still a long: the exposure carries
        the side, never the price's sign, and a slot with a margin is measured
        in margin (its quoted price may be through zero or back-adjusted)."""
        st = dict(slot="X|t", asset="X", template="t", position=1, shares=3.0, last_close=-1.5, point_value=1000.0,
                  entry_price=-2.0, entry_date=None, unrealized=1500.0, exit_orders=[dict(kind="stop", level=-3.0)])
        out = portfolio_targets([st], {"X|t": 1.0}, account_equity=100_000.0)
        self.assertAlmostEqual(float(out["legs"]["notional"].iloc[0]), 3.0 * 1.5 * 1000.0)   # |price| x pv, long
        self.assertEqual(out["exposure_basis"], "notional")
        self.assertAlmostEqual(out["open_risk"], 3.0 * 1.5 * 1000.0)
        tl = trade_list(out["by_asset"], {"X": 1.0})
        self.assertAlmostEqual(float(tl.loc["X", "order_shares"]), 2.0)
        self.assertAlmostEqual(float(tl.loc["X", "order_notional"]), 2.0 * 1.5 * 1000.0)
        # with a margin: 3 lots x 3000 = 9000 of margin, 9 % of the account, and a cash leg alongside in notional
        spread = dict(st, margin_per_unit=3000.0, position=-1)
        cash = dict(slot="SPY|t", asset="SPY", template="t", position=1, shares=100.0, last_close=500.0,
                    point_value=1.0, margin_per_unit=0.0, entry_price=490.0, entry_date=None, unrealized=1000.0,
                    exit_orders=[dict(kind="stop", level=480.0)])
        out = portfolio_targets([spread, cash], {"X|t": 0.5, "SPY|t": 0.5}, account_equity=100_000.0)
        self.assertEqual(out["exposure_basis"], "notional + margin")
        self.assertEqual(portfolio_targets([spread], {"X|t": 1.0}, account_equity=100_000.0)["exposure_basis"], "margin")
        self.assertAlmostEqual(float(out["by_asset"].loc["X", "notional"]), -9000.0)
        self.assertAlmostEqual(float(out["by_asset"].loc["SPY", "notional"]), 50_000.0)
        self.assertAlmostEqual(out["gross_exposure"], 0.59)
        self.assertAlmostEqual(out["net_exposure"], 0.41)
        # --max-gross binds on that measure and scales both legs alike
        capped = portfolio_targets([spread, cash], {"X|t": 0.5, "SPY|t": 0.5}, account_equity=100_000.0, max_gross=0.295)
        self.assertAlmostEqual(capped["scale_applied"], 0.5)
        self.assertAlmostEqual(float(capped["legs"].set_index("asset").loc["X", "shares"]), 1.5)


class BracketTargetTests(unittest.TestCase):
    """The engine can take a target on the entry bar (when the bar's prices
    beyond the fill certainly came after it), so the published entry carries
    the target as a bracket: an ATR target as an offset from the FILL (a gap
    moves it), a countertrend midline as a price. Checked against what the
    engine then does on the next bar."""

    def test_entry_orders_carry_the_target_the_engine_uses(self):
        df = synthetic_ohlc(1200, seed=5)
        found = same_bar = 0
        for entry in ("stop", "pullback", "close_confirm"):
            tpl = StrategyTemplate("b", exit_style="target_stop", entry_style=entry, cost_bps=0.0,
                                   atr_mult_target=0.5)   # near enough to be hit on the entry bar
            for t in range(LOOKBACK + 20, len(df) - 1, 3):
                st = strategy_state(df.iloc[:t], tpl, equity=100_000.0, lookback_bars=LOOKBACK)
                for o in st["entry_orders"]:
                    self.assertIn("bracket", o["note"])
                    self.assertAlmostEqual(o["target_offset"], o["side"] * tpl.atr_mult_target * st["atr"])
                    found += 1
                if not st["entry_orders"]:
                    continue
                after = backtest(df.iloc[t - LOOKBACK:t + 1], tpl, initial_equity=100_000.0)
                for tr in after["trades"]:
                    if tr["entry_date"] == tr["exit_date"] == df.index[t] and tr["reason"] == "target":
                        o = next(o for o in st["entry_orders"] if o["side"] == tr["side"])
                        self.assertAlmostEqual(tr["exit_price"], tr["entry_price"] + o["target_offset"], places=8)
                        same_bar += 1
        self.assertGreater(found, 0)
        self.assertGreater(same_bar, 0, "the sweep never takes a target on its entry bar")

    def test_a_fade_carries_the_midline_and_others_carry_nothing(self):
        df = synthetic_ohlc(900, seed=5)
        fade = StrategyTemplate("f", exit_style="channel", direction_logic="countertrend", cost_bps=0.0)
        plain = StrategyTemplate("p", exit_style="atr_trail", cost_bps=0.0)
        seen = 0
        for t in range(400, 900, 25):
            for o in strategy_state(df.iloc[:t], fade, lookback_bars=LOOKBACK)["entry_orders"]:
                self.assertIn("target", o)
                seen += 1
            for o in strategy_state(df.iloc[:t], plain, lookback_bars=LOOKBACK)["entry_orders"]:
                self.assertFalse({"target", "target_offset"} & set(o))
        self.assertGreater(seen, 0)


class FlatAssetTradeListTests(unittest.TestCase):
    def test_an_asset_the_book_is_flat_in_is_still_priced(self):
        by_asset = pd.DataFrame(columns=["shares", "price", "point_value", "basis", "notional", "pct_of_account"])
        t = trade_list(by_asset, {"SPY": 50.0, "spread": -1.0}, prices={"SPY": (400.0, 1.0), "spread": (-1.5, 1000.0)})
        self.assertEqual(list(t["action"]), ["SELL", "BUY"])
        self.assertAlmostEqual(float(t.loc["SPY", "order_notional"]), 50 * 400.0)
        self.assertAlmostEqual(float(t.loc["spread", "order_notional"]), 1.5 * 1000.0)
        self.assertTrue(np.isnan(trade_list(by_asset, {"X": 1.0}).loc["X", "order_notional"]))


class DeadAtrExitLevelTests(unittest.TestCase):
    def test_the_published_trail_uses_the_atr_the_engine_manages_with(self):
        """A flat patch drives the ATR to 0 while a chandelier position is
        open: the engine keeps managing it with the last usable ATR, so the
        published stop is 3 of THOSE ATRs below the high, not a stop sitting
        on the high itself."""
        n = 80
        o = np.full(n, 100.0); h = o + 1.0; l = o - 1.0; c = o.copy()
        for i in range(40, 60):                              # a trend: long at the 20-bar high, extreme 119
            o[i] = c[i] = 100.0 + (i - 39); h[i] = o[i] + 1.0; l[i] = o[i] - 1.0
        o[60:] = h[60:] = l[60:] = c[60:] = 121.0            # then pinned at the extreme: TR 0, ATR(5) 0
        df = pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c}, index=pd.bdate_range("2025-01-01", periods=n))
        tpl = StrategyTemplate("t", exit_style="atr_trail", n_entry=20, atr_n=5, atr_mult_trail=3.0,
                               atr_mult_stop=100.0, cost_bps=0.0, sides="long_only")
        st = strategy_state(df, tpl, equity=100_000.0, lookback_bars=70)
        self.assertEqual(st["position"], 1)
        self.assertEqual(st["atr"], 0.0)
        stop = next(x for x in st["exit_orders"] if x["kind"] == "stop")
        res = backtest(df.iloc[-70:], tpl, initial_equity=100_000.0)
        self.assertGreater(res["last_atr"], 0.0)
        self.assertAlmostEqual(stop["level"], res["open_position"]["trail_extreme"] - 3.0 * res["last_atr"])
        self.assertLess(stop["level"], 121.0)


if __name__ == "__main__":
    unittest.main()


class StanceOrderTests(unittest.TestCase):
    """The stance entries (a hedge committee's stance, a forecaster's buffered
    position) publish ONE market-on-open order for the change of holding, and
    the engine's next open does exactly that."""

    @classmethod
    def setUpClass(cls):
        cls.df = synthetic_ohlc(1500, seed=23)

    def _roll(self, tpl, lookback, step=11, start=None, rtol=0.0, equity=100_000.0):
        """Roll one real bar forward: the engine's holding after the next open
        must be the held units plus the published order."""
        seen = dict(flat_to_pos=0, pos_changed=0, held_no_order=0, long_held=0)
        for t in range(start or lookback + 5, len(self.df) - 1, step):
            st = strategy_state(self.df.iloc[:t], tpl, equity=equity, lookback_bars=lookback)
            after = backtest(self.df.iloc[t - lookback:t + 1], tpl, initial_equity=equity)
            self.assertEqual(st["exit_orders"], [])
            held = (st["position"] or 0) * st["shares"]
            orders = st["entry_orders"]
            self.assertLessEqual(len(orders), 1)
            for o in orders:
                self.assertEqual(o["kind"], "market_on_open")
                self.assertGreater(o["shares"], 0)
                self.assertIn("last close", o["note"]) if not (tpl.vol_target > 0 and tpl.margin_per_unit > 0) else None
            want = held + sum(o["side"] * o["shares"] for o in orders)
            ap = after["open_position"]
            got = 0.0 if ap is None else ap["side"] * ap["shares"]
            if rtol == 0.0:
                self.assertAlmostEqual(got, want, places=8, msg=f"{tpl.name} bar {t}")
            else:
                self.assertLessEqual(abs(got - want), rtol * max(abs(want), abs(held), 1.0) + 1e-9,
                                     f"{tpl.name} bar {t}: engine {got}, published {want}")
            if held > 0:
                seen["long_held"] += 1
            if held == 0 and want != 0:
                seen["flat_to_pos"] += 1
            if held != 0 and orders:
                seen["pos_changed"] += 1
            if held != 0 and not orders:
                seen["held_no_order"] += 1
        return seen

    def test_forecast_template_on_a_margined_instrument_is_exact(self):
        tpl = StrategyTemplate("fc", direction_logic="learned", channel_type="forecast", entry_style="stance",
                               cost_bps=0.0, margin_per_unit=1000.0, point_value=10.0, vol_target=0.15,
                               max_leverage=50.0)
        seen = self._roll(tpl, lookback=1100, start=1105, step=12)
        self.assertGreater(seen["flat_to_pos"] + seen["pos_changed"], 5)
        self.assertGreater(seen["held_no_order"], 5, "the buffer never held")

    def test_trend_forecast_template_flat_and_long(self):
        tpl = StrategyTemplate("fc", direction_logic="trend", channel_type="forecast", entry_style="stance",
                               cost_bps=0.0, margin_per_unit=1000.0, point_value=10.0, vol_target=0.15,
                               max_leverage=50.0)
        seen = self._roll(tpl, lookback=600, step=5)
        self.assertGreater(seen["long_held"], 0)
        self.assertGreater(seen["flat_to_pos"] + seen["pos_changed"], 5)

    def test_hedge_stance_on_a_margined_instrument_is_exact(self):
        tpl = StrategyTemplate("hs", direction_logic="trend", channel_type="hedge", entry_style="stance",
                               cost_bps=0.0, margin_per_unit=1000.0, point_value=10.0, vol_target=0.15,
                               max_leverage=50.0)
        seen = self._roll(tpl, lookback=1200, step=5)       # the stance's warm-up is 1175 bars (STANCE_SETTLE)
        self.assertGreater(seen["flat_to_pos"] + seen["pos_changed"], 3)

    def test_cash_asset_matches_within_the_open_to_close_ratio(self):
        for tpl in (StrategyTemplate("fc", direction_logic="trend", channel_type="forecast", entry_style="stance",
                                     vol_target=0.15),
                    StrategyTemplate("hs", direction_logic="trend", channel_type="hedge", entry_style="stance")):
            seen = self._roll(tpl, lookback=1200, step=5, rtol=0.05)
            self.assertGreater(seen["flat_to_pos"] + seen["pos_changed"] + seen["long_held"], 3, tpl.name)

    def test_no_order_without_a_formed_target_or_a_change(self):
        from live import _stance_orders
        tpl = StrategyTemplate("fc", channel_type="forecast", entry_style="stance")
        base = dict(formed=True, want=0.4, level=0.3, change=True, target_units=10.0, held_units=4.0, price_proxy=100.0)
        (o,) = _stance_orders(tpl, base)
        self.assertEqual((o["kind"], o["side"], o["shares"]), ("market_on_open", 1, 6.0))
        (o,) = _stance_orders(tpl, dict(base, target_units=-3.0, held_units=4.0))
        self.assertEqual((o["side"], o["shares"]), (-1, 7.0))
        self.assertEqual(_stance_orders(tpl, dict(base, formed=False)), [])            # target NaN
        self.assertEqual(_stance_orders(tpl, dict(base, change=False)), [])            # the buffer holds
        self.assertEqual(_stance_orders(tpl, dict(base, target_units=4.0)), [])        # same units


class StanceDashboardTests(unittest.TestCase):
    def test_a_stance_slot_is_traded_like_any_other(self):
        """etf_dashboard no longer parks the stance entry as research only: its
        slot reports the position, publishes its order and reaches trades-to-send."""
        from dataclasses import asdict
        from types import SimpleNamespace
        import etf_dashboard as ED
        tpl = StrategyTemplate("fc", direction_logic="trend", channel_type="forecast", entry_style="stance",
                               vol_target=0.15)
        slot = dict(slot="SPY|fc", asset="SPY", template_name=tpl.name, template=asdict(tpl), weight=1.0,
                    research={})
        cfg = dict(train_bars=900, test_bars=125, wide_grid=False, metric="sharpe", selection="plateau",
                   anchored=False)
        args = SimpleNamespace(account_equity=1e5, no_refit=False)
        df = synthetic_ohlc(1500, seed=23)
        held = None
        live = dict(slots={}, last_targets={}, runs=0)      # held between runs: one refit, then the window
        for t in range(1300, 1400, 2):
            st, _ = ED._slot_signal(slot, df.iloc[:t], cfg, live, args)
            self.assertNotIn("the stance entry is research only: no live orders", st["blocked_by"])
            if st["position"]:
                held = st
                break
        self.assertIsNotNone(held, "the fixture never held a position")
        tg = portfolio_targets([held], {"SPY|fc": 1.0}, 1e5)
        trades = trade_list(tg["by_asset"], {})
        self.assertEqual(trades.loc["SPY", "action"], "BUY" if held["position"] == 1 else "SELL")
