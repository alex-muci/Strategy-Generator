"""
Tests for the order log (strategy.backtest(..., log_orders=True)) and the
replay of a finished run (replay.py).

The order log is read out of the engine's own loop, so the checks here are
about its faithfulness: every fill is a trade the engine reports, every fill
price is explained by an order the log says was working on that bar, a
pullback limit lives a submit -> working -> one ending, and switching the log
on changes nothing about the backtest. The replay checks are about exactness:
the best template and both portfolios of a `main.py` run come back to the
last digit from the saved manifest, serially or in a pool.

Run with: python -m unittest discover -s tests -v
"""

from __future__ import annotations
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from dataclasses import asdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import strategy as S  # noqa: E402
from strategy import backtest, StrategyTemplate, annualized_sharpe  # noqa: E402
from data import synthetic_ohlc  # noqa: E402
import main as M  # noqa: E402
import replay as R  # noqa: E402

BASE = ["--family", "quick", "--max-templates", "4", "--bars", "1100", "--train", "300",
        "--test", "100", "--n-boot", "100", "--min-sharpe", "-5", "--jobs", "1", "--no-matrix"]

TEMPLATES = [
    StrategyTemplate("stop-chan"),
    StrategyTemplate("stop-trail", exit_style="atr_trail"),
    StrategyTemplate("cc-target", entry_style="close_confirm", exit_style="target_stop"),
    StrategyTemplate("pb-time", entry_style="pullback", exit_style="time_stop", max_hold_bars=12, pullback_valid_bars=4),
    StrategyTemplate("fade-chan", direction_logic="countertrend", exit_style="channel"),
    StrategyTemplate("fade-pb", direction_logic="countertrend", entry_style="pullback", exit_style="target_stop"),
    StrategyTemplate("long-stop", sides="long_only", exit_style="target_stop"),
    StrategyTemplate("learned", channel_type="hedge", direction_logic="learned", exit_style="atr_trail"),
    StrategyTemplate("vt", vol_target=0.15, exit_style="atr_trail"),
]


def _quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def _same_template(a: StrategyTemplate, b: StrategyTemplate) -> bool:
    da, db = asdict(a), asdict(b)
    for k in da:
        x, y = da[k], db[k]
        if isinstance(x, float) and isinstance(y, float) and np.isnan(x) and np.isnan(y):
            continue
        if x != y:
            return False
    return True


class OrderLogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = synthetic_ohlc(900, seed=11)
        cls.first = 120
        cls.runs = {t.name: (backtest(cls.df, t, first_trade_bar=cls.first),
                             backtest(cls.df, t, first_trade_bar=cls.first, log_orders=True)) for t in TEMPLATES}
        cls.open_ = cls.df["Open"]

    def _rows(self, name, kind):
        o = self.runs[name][1]["orders"]
        return o[o["kind"] == kind]

    def test_off_by_default_and_a_no_op_when_on(self):
        for name, (off, on) in self.runs.items():
            self.assertIsNone(off["orders"], name)
            np.testing.assert_array_equal(off["equity"].to_numpy(), on["equity"].to_numpy(), err_msg=name)
            self.assertEqual(off["trades"], on["trades"], name)
            self.assertEqual(off["open_position"], on["open_position"], name)
            self.assertIsInstance(on["orders"], pd.DataFrame)
            self.assertGreater(len(on["orders"]), 0, name)
            self.assertLessEqual(len(on["orders"]), 8 * len(self.df) + 8, name)
            # nothing rests or fills on the warm-up bars
            self.assertTrue((on["orders"]["date"] >= self.df.index[self.first]).all(), name)
            self.assertTrue(set(on["orders"]["kind"]) <= set(S.ORDER_KINDS), name)

    def test_every_fill_is_a_trade_and_every_trade_has_its_fills(self):
        for name, (_, on) in self.runs.items():
            trades = on["trades"]
            ef, xf = self._rows(name, "entry_fill"), self._rows(name, "exit_fill")
            n_open = 1 if on["open_position"] is not None else 0
            self.assertEqual(len(ef), len(trades) + n_open, name)   # the open position entered too
            self.assertEqual(len(xf), len(trades), name)
            for t, (_, e), (_, x) in zip(trades, ef.iterrows(), xf.iterrows()):
                self.assertEqual(e["date"], t["entry_date"], name)
                self.assertEqual(e["side"], t["side"], name)
                self.assertAlmostEqual(e["level"], t["entry_price"], msg=name)
                self.assertAlmostEqual(e["qty"], t["shares"], msg=name)
                self.assertEqual(x["date"], t["exit_date"], name)
                self.assertAlmostEqual(x["level"], t["exit_price"], msg=name)
                self.assertAlmostEqual(x["qty"], t["shares"], msg=name)
                self.assertEqual(x["detail_text"], t["reason"], name)
                self.assertAlmostEqual(x["aux"], t["pnl"], msg=name)
            if n_open:
                pos = on["open_position"]
                last = ef.iloc[-1]
                self.assertEqual(last["date"], pos["entry_date"], name)
                self.assertAlmostEqual(last["aux"], pos["hard_stop"], msg=name)   # aux = the initial hard stop

    def test_stop_exits_fill_at_the_stop_that_was_working(self):
        """A stop fill is explained by the stop_working row of its bar: at the
        level, or at the open when the bar gapped through it; a same-bar stop
        goes at the fill itself when the fill was already through it."""
        seen = 0
        for name, (_, on) in self.runs.items():
            o = on["orders"]
            for _, x in o[o["kind"] == "exit_fill"].iterrows():
                if x["detail_text"] not in ("stop", "channel", "stop_same_bar"):
                    continue
                sw = o[(o["kind"] == "stop_working") & (o["date"] == x["date"])]
                self.assertEqual(len(sw), 1, name)
                level = float(sw["level"].iloc[0])
                if x["detail_text"] == "stop_same_bar" or (x["detail_text"] == "channel" and
                                                          len(o[(o["kind"] == "entry_fill") & (o["date"] == x["date"])])):
                    ref = float(o[(o["kind"] == "entry_fill") & (o["date"] == x["date"])]["level"].iloc[0])
                else:
                    ref = float(self.open_.loc[x["date"]])
                expect = min(ref, level) if x["side"] == 1 else max(ref, level)
                self.assertAlmostEqual(float(x["level"]), expect, msg=f"{name} {x['date']}")
                seen += 1
        self.assertGreater(seen, 20)

    def test_target_and_time_exits_are_explained_too(self):
        seen = 0
        for name, (_, on) in self.runs.items():
            o = on["orders"]
            for _, x in o[o["kind"] == "exit_fill"].iterrows():
                if x["detail_text"] in ("target", "midline"):
                    tw = o[(o["kind"] == "target_working") & (o["date"] == x["date"])]
                    self.assertEqual(len(tw), 1, name)
                    op = float(self.open_.loc[x["date"]])
                    expect = max(op, float(tw["level"].iloc[0])) if x["side"] == 1 else min(op, float(tw["level"].iloc[0]))
                    self.assertAlmostEqual(float(x["level"]), expect, msg=name)
                    seen += 1
                elif x["detail_text"] == "time":
                    te = o[(o["kind"] == "time_exit_submit") & (o["date"] == x["date"])]
                    self.assertEqual(len(te), 1, name)
                    self.assertAlmostEqual(float(x["level"]), float(self.open_.loc[x["date"]]), msg=name)
                    seen += 1
        self.assertGreater(seen, 10)

    def test_entries_fill_at_the_order_that_was_working(self):
        """Trend entries are stops (fill at the level or the gapped open),
        fade entries are limits (the mirror), close_confirm is a market order
        at the open, a pullback fills at its resting limit."""
        kinds = {}
        for name, (_, on) in self.runs.items():
            o = on["orders"]
            for _, e in o[o["kind"] == "entry_fill"].iterrows():
                kinds.setdefault(name, set()).add(e["detail_text"])
                op = float(self.open_.loc[e["date"]])
                if e["detail_text"] == "market":
                    sub = o[(o["kind"] == "entry_submit_market") & (o["date"] == e["date"])]
                    self.assertEqual(len(sub), 1, name)
                    self.assertAlmostEqual(float(e["level"]), op, msg=name)
                    continue
                if e["detail_text"] == "pullback":
                    w = o[(o["kind"] == "pullback_working") & (o["date"] == e["date"])]
                else:
                    w = o[(o["kind"] == "entry_working") & (o["date"] == e["date"]) & (o["side"] == e["side"])]
                self.assertEqual(len(w), 1, f"{name} {e['date']}")
                level = float(w["level"].iloc[0])
                if e["detail_text"] == "stop":
                    expect = max(op, level) if e["side"] == 1 else min(op, level)
                else:   # limit, pullback
                    expect = min(op, level) if e["side"] == 1 else max(op, level)
                self.assertAlmostEqual(float(e["level"]), expect, msg=f"{name} {e['date']}")
        self.assertEqual(kinds["stop-chan"], {"stop"})
        self.assertEqual(kinds["fade-chan"], {"limit"})
        self.assertEqual(kinds["cc-target"], {"market"})
        self.assertEqual(kinds["pb-time"], {"pullback"})
        self.assertEqual(kinds["long-stop"], {"stop"})
        self.assertTrue((self._rows("long-stop", "entry_working")["side"] == 1).all())

    def test_pullback_lifecycle(self):
        """submit -> working on the following bars -> exactly one of fill,
        expiry or cancellation, before the next submit; the expiry date is
        `pullback_valid_bars` bars after the submit."""
        for name in ("pb-time", "fade-pb"):
            tpl = next(t for t in TEMPLATES if t.name == name)
            o = self.runs[name][1]["orders"]
            lifecycle = o[o["kind"].str.startswith("pullback") | ((o["kind"] == "entry_fill") & (o["detail_text"] == "pullback"))]
            active, endings, submits = False, 0, 0
            for _, r in lifecycle.iterrows():
                if r["kind"] == "pullback_submit":
                    self.assertFalse(active, f"{name}: submit while a limit rests")
                    active, submits = True, submits + 1
                    i = self.df.index.get_loc(r["date"])
                    self.assertEqual(r["expires"], self.df.index[i + tpl.pullback_valid_bars], name)
                elif r["kind"] == "pullback_working":
                    self.assertTrue(active, name)
                else:   # fill, expire, cancel
                    self.assertTrue(active, f"{name}: {r['kind']} without a resting limit")
                    active, endings = False, endings + 1
            self.assertGreater(submits, 5, name)
            self.assertEqual(endings, submits - (1 if active else 0), name)
            self.assertGreater(len(o[o["kind"] == "pullback_expire"]), 0, name)

    def test_working_stop_kinds_follow_the_exit_style(self):
        self.assertEqual(set(self._rows("stop-chan", "stop_working")["detail_text"]), {"hard", "channel"})
        self.assertEqual(set(self._rows("stop-trail", "stop_working")["detail_text"]), {"hard", "trail"})
        self.assertEqual(set(self._rows("cc-target", "stop_working")["detail_text"]), {"hard"})
        # the target works from the fill like the stop: one target_working row
        # for every stop_working row, the entry bar's included
        for name in ("cc-target", "fade-chan"):
            self.assertEqual(len(self._rows(name, "target_working")),
                             len(self._rows(name, "stop_working")), name)
        self.assertEqual(len(self._rows("stop-chan", "target_working")), 0)

    def test_jit_kernel_matches_pure_python_with_the_log(self):
        if not S.HAVE_NUMBA:
            self.skipTest("numba not installed")
        fast = S._bar_loop_fast
        try:
            for tpl in TEMPLATES[::2]:
                S._bar_loop_fast = S._bar_loop
                slow = backtest(self.df, tpl, first_trade_bar=self.first, log_orders=True)
                S._bar_loop_fast = fast
                quick = backtest(self.df, tpl, first_trade_bar=self.first, log_orders=True)
                pd.testing.assert_frame_equal(slow["orders"], quick["orders"], check_exact=False, rtol=0, atol=1e-9)
        finally:
            S._bar_loop_fast = fast


class RunReplayTests(unittest.TestCase):
    """One small `main.py` run, then the replay of every target from what it saved."""

    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp(prefix="replay-")
        cls.out = _quiet(M.main, BASE + ["--out", cls.dir, "--replay", "best"])
        cls.loaded = R.load_run(cls.dir)
        cls.rep = _quiet(R.replay_run, cls.dir, "all")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, ignore_errors=True)

    def test_run_files_are_written(self):
        for f in (R.MANIFEST, R.DATA_FILE, R.PORTFOLIO_RETURNS, R.SELECTED_RETURNS):
            self.assertTrue(os.path.exists(os.path.join(self.dir, f)), f)
        with open(os.path.join(self.dir, R.MANIFEST), encoding="utf-8") as f:
            m = json.load(f)   # strict JSON: no NaN tokens
        self.assertEqual(m["schema_version"], R.MANIFEST_VERSION)
        self.assertEqual(m["data"]["source"], "synthetic")
        self.assertEqual(set(m["templates"]), set(self.out["results"]))
        self.assertEqual(m["best_template"], self.out["family"]["best_template"])
        self.assertEqual(m["static"]["selected"], self.out["portfolio"]["selected"])

    def test_bars_round_trip_exactly(self):
        pd.testing.assert_frame_equal(self.loaded["df"], synthetic_ohlc(1100, seed=7), check_exact=True, check_freq=False)

    def test_templates_and_windows_round_trip(self):
        for name, res in self.out["results"].items():
            self.assertTrue(_same_template(self.loaded["templates"][name], res["template"]), name)
            for w, stored in zip(res["windows"], self.loaded["manifest"]["templates"][name]["windows"]):
                self.assertEqual(stored["test_start"], w["test_start"])
                self.assertEqual(stored["params"], w["params"])
                self.assertEqual(stored["skipped"], bool(w["skipped"]))
        t = StrategyTemplate("x", regime_filter="trend_only")
        self.assertTrue(_same_template(R.template_from_dict(json.loads(json.dumps(R._jsonable(asdict(t))))), t))

    def test_every_target_reproduces_the_run(self):
        out = self.out
        best = out["family"]["best_template"]
        expect = dict(best=out["results"][best]["summary"]["oos_sharpe"],
                      static=annualized_sharpe(out["portfolio"]["portfolio_returns"]),
                      nested=out["nested"]["sharpe"])
        for which, r in self.rep.items():
            c = r["check"]
            self.assertTrue(c["ok"], (which, c))
            self.assertEqual(c["max_abs_return_diff"], 0.0, which)
            self.assertAlmostEqual(c["sharpe_replayed"], expect[which], places=9, msg=which)
            self.assertAlmostEqual(c["sharpe_stored"], expect[which], places=9, msg=which)
            self.assertEqual(c["windows_mismatched"], [], which)
            folder = os.path.join(self.dir, R.REPLAY_DIR, which)
            for f in ("trades.csv", "orders.csv", "order_events.csv", "returns.csv", "check.json"):
                self.assertTrue(os.path.exists(os.path.join(folder, f)), (which, f))
            with open(os.path.join(folder, "check.json"), encoding="utf-8") as f:
                self.assertTrue(json.load(f)["ok"])
        # per-bar: the replayed series IS the run's series
        pd.testing.assert_series_equal(self.rep["best"]["returns"], out["results"][best]["oos_returns"],
                                       check_names=False, check_freq=False)
        pd.testing.assert_series_equal(self.rep["nested"]["returns"], out["nested"]["portfolio_returns"],
                                       check_names=False, check_freq=False)
        sel = out["portfolio"]["selected"]
        self.assertEqual(self.rep["static"]["check"]["n_trades_stored"],
                         sum(out["results"][n]["summary"]["n_trades_oos"] for n in sel))
        self.assertGreater(len(self.rep["best"]["trades"]), 0)
        self.assertEqual(set(self.rep["nested"]["trades"]["template"]) <= set(self.rep["nested"]["names"]), True)

    def test_the_main_flag_replays_too(self):
        self.assertIn("replay", self.out)
        self.assertTrue(self.out["replay"]["best"]["check"]["ok"])

    def test_nested_weights_are_stored_unrounded(self):
        for s in self.loaded["manifest"]["nested"]["selections"]:
            if s["selected"]:
                self.assertAlmostEqual(sum(s["weights"].values()), 1.0, places=12)

    def test_trades_carry_their_window_and_open_positions_are_listed(self):
        t = self.rep["best"]["trades"]
        for c in ("template", "window", "entry_date", "exit_date", "side", "shares", "entry_price", "exit_price",
                  "reason", "pnl", "closed", "params", "weight"):
            self.assertIn(c, t.columns)
        closed = t[t["closed"]]
        self.assertEqual(len(closed), self.rep["best"]["check"]["n_trades_replayed"])
        self.assertTrue(closed["exit_date"].notna().all())
        self.assertTrue((t[~t["closed"]]["reason"] == "open_at_window_end").all())
        for _, row in t.iterrows():   # a trade lies inside its window's test period
            w = self.loaded["manifest"]["templates"][row["template"]]["windows"][row["window"]]
            self.assertGreaterEqual(row["entry_date"], w["test_start"])
            self.assertLessEqual(row["entry_date"], w["test_end"])

    def test_order_lifecycle_view_keeps_every_event(self):
        ev = self.rep["best"]["orders"]
        col = R.collapse_orders(ev)
        working = ev["kind"].isin(R.WORKING_KINDS)
        self.assertEqual(int(col[col["kind"].isin(R.WORKING_KINDS)]["n_bars"].sum()), int(working.sum()))
        self.assertEqual(len(col[~col["kind"].isin(R.WORKING_KINDS)]), int((~working).sum()))
        self.assertTrue((col["last_date"] >= col["first_date"]).all())
        self.assertTrue(col["first_date"].is_monotonic_increasing)
        self.assertTrue((col[col["n_bars"] > 1]["kind"].isin(R.WORKING_KINDS)).all())

    def test_parallel_replay_equals_serial(self):
        rep2 = _quiet(R.replay_run, self.dir, "all", jobs=2)
        for which in R.TARGETS:
            pd.testing.assert_series_equal(rep2[which]["returns"], self.rep[which]["returns"])
            pd.testing.assert_frame_equal(rep2[which]["trades"], self.rep[which]["trades"])
            pd.testing.assert_frame_equal(rep2[which]["orders"], self.rep[which]["orders"])

    def test_any_template_can_be_replayed(self):
        m = self.loaded["manifest"]
        other = next(n for n in m["templates"] if n not in R._needed_templates(m))
        r = _quiet(R.replay_run, self.dir, "best", template=other)["best"]
        c = r["check"]
        self.assertTrue(c["ok"], c)
        self.assertFalse(c["stored_series"])     # no series kept: Sharpe and trade counts decide
        self.assertAlmostEqual(c["sharpe_replayed"], self.out["results"][other]["summary"]["oos_sharpe"], places=9)
        self.assertTrue(os.path.exists(os.path.join(self.dir, R.REPLAY_DIR, "template_" + R._safe_filename(other), "trades.csv")))
        self.assertTrue(os.path.exists(os.path.join(self.dir, R.REPLAY_DIR, "best", "orders.csv")))   # untouched
        with self.assertRaises(KeyError):
            _quiet(R.replay_run, self.dir, "best", template="no such template")

    def test_no_orders_leaves_no_stale_order_files(self):
        _quiet(R.replay_run, self.dir, "static", orders=False)
        folder = os.path.join(self.dir, R.REPLAY_DIR, "static")
        self.assertTrue(os.path.exists(os.path.join(folder, "trades.csv")))
        self.assertFalse(os.path.exists(os.path.join(folder, "orders.csv")))
        _quiet(R.replay_run, self.dir, "static")
        self.assertTrue(os.path.exists(os.path.join(folder, "orders.csv")))

    def test_synthetic_bars_are_regenerated_when_the_file_is_missing(self):
        copy = tempfile.mkdtemp(prefix="replay-copy-")
        try:
            for f in (R.MANIFEST, R.PORTFOLIO_RETURNS, R.SELECTED_RETURNS):
                shutil.copy(os.path.join(self.dir, f), copy)
            self.assertTrue(_quiet(R.replay_run, copy, "nested")["nested"]["check"]["ok"])
        finally:
            shutil.rmtree(copy, ignore_errors=True)

    def test_naive_runs_stay_naive(self):
        self.assertIsNone(self.loaded["df"].index.tz)
        self.assertIsNone(self.loaded["manifest"]["boundaries"][0].tz)
        self.assertIsNone(self.loaded["portfolio_returns"].index.tz)
        self.assertIsNone(self.loaded["selected_returns"].index.tz)

    def test_tz_aware_runs_are_read_in_utc(self):
        """A run whose bars were stamped in New York (offsets -05:00 and
        -04:00 across DST, which parse to plain objects unless read as UTC)
        loads with UTC stamps everywhere and replays exactly."""
        zone = "America/New_York"
        copy = tempfile.mkdtemp(prefix="replay-tz-")
        try:
            for f in (R.DATA_FILE, R.PORTFOLIO_RETURNS, R.SELECTED_RETURNS):
                d = pd.read_csv(os.path.join(self.dir, f), index_col=0, parse_dates=True,
                                float_precision="round_trip")
                d.index = d.index.tz_localize(zone)
                d.to_csv(os.path.join(copy, f), float_format="%.17g",
                         index_label=None if f == R.DATA_FILE else "date")
            with open(os.path.join(self.dir, R.MANIFEST), encoding="utf-8") as f:
                m = json.load(f)
            aware = lambda x: pd.Timestamp(x).tz_localize(zone).isoformat()
            for spec in m["templates"].values():
                for w in spec["windows"]:
                    for k in ("train_start", "train_end", "test_start", "test_end"):
                        w[k] = aware(w[k])
            for sel in m["nested"]["selections"]:
                sel["period_start"] = aware(sel["period_start"])
            m["boundaries"] = [aware(b) for b in m["boundaries"]]
            m["data"]["first"], m["data"]["last"] = aware(m["data"]["first"]), aware(m["data"]["last"])
            with open(os.path.join(copy, R.MANIFEST), "w", encoding="utf-8") as f:
                json.dump(m, f)
            run = R.load_run(copy)
            want = self.loaded["df"].index.tz_localize(zone).tz_convert("UTC")
            self.assertEqual(str(run["df"].index.tz), "UTC")
            self.assertTrue(run["df"].index.equals(want))
            pd.testing.assert_frame_equal(run["df"].reset_index(drop=True),
                                          self.loaded["df"].reset_index(drop=True), check_exact=True)
            self.assertEqual(str(run["manifest"]["boundaries"][0].tz), "UTC")
            self.assertEqual(str(run["portfolio_returns"].index.tz), "UTC")
            self.assertEqual(str(run["selected_returns"].index.tz), "UTC")
            for which, r in _quiet(R.replay_run, copy, "all").items():
                self.assertTrue(r["check"]["ok"], (which, r["check"]))
                self.assertEqual(r["check"]["max_abs_return_diff"], 0.0, which)
        finally:
            shutil.rmtree(copy, ignore_errors=True)


class EmptyPortfolioTests(unittest.TestCase):
    """Nothing qualifies: the static portfolio is empty and the nested one
    sits in cash. Both replay as consistent, empty targets, not as mismatches."""

    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp(prefix="replay-empty-")
        cls.out = _quiet(M.main, [a if a != "-5" else "5" for a in BASE] + ["--out", cls.dir])
        cls.rep = _quiet(R.replay_run, cls.dir, "all")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, ignore_errors=True)

    def test_empty_targets_are_consistent(self):
        self.assertEqual(self.out["portfolio"]["selected"], [])
        static, nested = self.rep["static"]["check"], self.rep["nested"]["check"]
        self.assertTrue(static["ok"] and static["empty"], static)
        self.assertEqual(static["n_trades_replayed"], 0)
        self.assertTrue(nested["ok"], nested)
        self.assertEqual(nested["n_trades_replayed"], 0)
        self.assertTrue((self.rep["nested"]["returns"] == 0).all())
        self.assertTrue(self.rep["best"]["check"]["ok"])
        for which in R.TARGETS:
            self.assertTrue(os.path.exists(os.path.join(self.dir, R.REPLAY_DIR, which, "check.json")))


class NestedPeriodTests(unittest.TestCase):
    """A nested period is the walk-forward window that starts on its date,
    found by date: periods short of history leave no selection at all."""

    def _run(self, starts, periods):
        windows = [dict(test_start=pd.Timestamp(s), test_end=pd.Timestamp(s) + pd.Timedelta(days=5), skipped=False,
                        params={}, n_trades=0) for s in starts]
        return dict(manifest=dict(templates={"a": dict(windows=windows), "b": dict(windows=list(windows))},
                                  nested=dict(selections=[dict(period_start=pd.Timestamp(p), selected=["a"], weights={"a": 1.0})
                                                          for p in periods])))

    def test_periods_map_to_windows_by_date(self):
        starts = ["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01", "2020-05-01", "2020-06-01"]
        run = self._run(starts, ["2020-05-01", "2020-06-01"])     # the first four periods had too little history
        self.assertEqual([k for _, k in R._nested_periods(run)], [4, 5])
        self.assertEqual(R._window_of(run, "b", pd.Timestamp("2020-03-01")), 2)
        self.assertIsNone(R._window_of(run, "b", pd.Timestamp("2020-03-02")))
        with self.assertRaises(ValueError):
            R._nested_periods(self._run(starts, ["2020-07-01"]))


if __name__ == "__main__":
    unittest.main()
