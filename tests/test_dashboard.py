"""
End-to-end tests for the two-phase ETF dashboard.

These run the real `research` and `signals` commands over synthetic data in a
temporary state directory: the point is that the handover between the phases
works (the JSON spec round-trips, parameters are held for a whole test window
rather than refitted every run) and that the page that comes out says what the
research actually concluded.

Run with:  python -m unittest discover -s tests -v
"""

from __future__ import annotations
import json
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import etf_dashboard as ED  # noqa: E402
from dashboard_html import render_dashboard, _room_to_stop  # noqa: E402
from live import portfolio_targets, trade_list  # noqa: E402

RESEARCH = ["research", "--synthetic", "--assets", "AAA", "BBB", "--family", "quick",
            "--max-templates", "4", "--bars", "1100", "--train", "300", "--test", "100",
            "--jobs", "1", "--n-boot", "100", "--min-sharpe", "-5", "--state-dir"]
SIGNALS = ["signals", "--synthetic", "--bars", "1100", "--jobs", "1",
           "--account-equity", "200000", "--state-dir"]


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp(prefix="etfdash-")
        ED.main(RESEARCH + [cls.dir])
        cls.out = ED.main(SIGNALS + [cls.dir])
        with open(os.path.join(cls.dir, "portfolio.json")) as f:
            cls.spec = json.load(f)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, ignore_errors=True)

    def test_research_writes_a_complete_spec(self):
        s = self.spec
        self.assertEqual(s["version"], ED.SPEC_VERSION)
        self.assertEqual(s["assets"], ["AAA", "BBB"])
        self.assertTrue(s["slots"], "no slot was selected at min_sharpe=-5")
        for slot in s["slots"]:
            self.assertIn(slot["asset"], s["assets"])
            self.assertGreaterEqual(slot["weight"], 0.0)
            # the template must round-trip back into a live StrategyTemplate
            from strategy import StrategyTemplate
            tpl = StrategyTemplate(**slot["template"])
            tpl.validate()
            self.assertAlmostEqual(tpl.cost_bps, s["config"]["cost_bps"])
            self.assertEqual(tpl.vol_target, s["config"]["vol_target"])
            self.assertEqual(tpl.vol_target_n, s["config"]["vol_target_n"])
            # a spec written before the vol target existed rehydrates with it off,
            # which is exactly the rule that spec was researched with
            old = {k: v for k, v in slot["template"].items() if not k.startswith("vol_target")}
            self.assertEqual((StrategyTemplate(**old).vol_target, StrategyTemplate(**old).vol_target_n), (0.0, 60))
        self.assertAlmostEqual(sum(x["weight"] for x in s["slots"]), 1.0, places=6)
        for k in ("pbo_trials", "reality_check_p", "nested_sharpe", "dsr_best", "n_eff"):
            self.assertIn(k, s["diagnostics"])
        self.assertIn(s["verdict"]["level"], ("good", "warning", "critical"))
        self.assertTrue(os.path.exists(os.path.join(self.dir, "research_report.md")))

    def test_signals_writes_the_dashboard_and_the_csvs(self):
        for name in ("dashboard.html", "signals.csv", "signal_log.csv", "live_state.json"):
            p = os.path.join(self.dir, name)
            self.assertTrue(os.path.exists(p), f"{name} missing")
            self.assertGreater(os.path.getsize(p), 0, f"{name} empty")
        html = open(os.path.join(self.dir, "dashboard.html"), encoding="utf-8").read()
        self.assertTrue(html.startswith("<!doctype html>"))
        self.assertIn("</html>", html)
        for section in ("Trades to send", "Orders to work on the next bar",
                        "What you should be holding", "Out-of-sample track record",
                        "Why you should not trust this too much"):
            self.assertIn(section, html)
        # the page is self-contained: nothing is fetched when it is opened
        for bad in ("<script src=", "<link rel=\"stylesheet\"", "http://", "cdn."):
            self.assertNotIn(bad, html, f"page reaches out for {bad}")

    def test_the_page_states_the_research_verdict(self):
        html = open(os.path.join(self.dir, "dashboard.html"), encoding="utf-8").read()
        v = self.spec["verdict"]
        self.assertIn(v["headline"], html)
        self.assertIn(f'class="verdict {v["level"]}"', html)
        import html as _h
        for reason in v["reasons"]:
            self.assertIn(_h.escape(reason)[:40], html)

    def test_every_slot_reports_a_state(self):
        states = self.out["states"]
        self.assertEqual(len(states), len(self.spec["slots"]))
        for st in states:
            self.assertIn(st["asset"], self.spec["assets"])
            self.assertIn(st["position"], (None, 1, -1))
            if st["position"]:
                # an open position always publishes a hard stop
                self.assertTrue(any(o["kind"] == "stop" for o in st["exit_orders"]))
                self.assertGreater(st["shares"], 0)
            else:
                self.assertEqual(st["shares"], 0.0)

    def test_parameters_are_held_between_runs(self):
        """The whole point of due_for_refit: a second run on the same bars must
        reuse the fitted parameters, not re-optimize."""
        with open(os.path.join(self.dir, "live_state.json")) as f:
            before = json.load(f)
        self.assertTrue(before["slots"])
        for slot in before["slots"].values():
            self.assertIsNotNone(slot["fitted_on"])
        out2 = ED.main(SIGNALS + [self.dir])
        with open(os.path.join(self.dir, "live_state.json")) as f:
            after = json.load(f)
        for key, slot in before["slots"].items():
            self.assertEqual(slot["params"], after["slots"][key]["params"], key)
            self.assertEqual(slot["fitted_on"], after["slots"][key]["fitted_on"], key)
        self.assertFalse([n for n in out2["run"]["notes"] if "re-optimized" in n],
                         "a same-bar re-run must not refit")
        self.assertEqual(after["runs"], before["runs"] + 1)

    def test_a_refit_happens_once_a_test_window_has_passed(self):
        with open(os.path.join(self.dir, "live_state.json")) as f:
            st = json.load(f)
        key = next(iter(st["slots"]))
        # pretend this slot was last fitted a full test window ago
        st["slots"][key]["fitted_on"] = "2015-01-01 00:00:00"
        with open(os.path.join(self.dir, "live_state.json"), "w") as f:
            json.dump(st, f, default=str)
        out = ED.main(SIGNALS + [self.dir])
        self.assertTrue([n for n in out["run"]["notes"] if n.startswith(key) and "re-optimized" in n],
                        "a stale slot was not refitted")

    def test_assets_are_put_on_one_shared_bar_index(self):
        """Window boundaries are positions in the index, so the assets have to
        agree about which bars exist."""
        data = ED.load_assets(["AAA", "BBB", "CCC"], interval="1d", start="2015-01-01",
                              synthetic=True, bars=600, quiet=True)
        idx = [df.index for df in data.values()]
        for other in idx[1:]:
            self.assertTrue(idx[0].equals(other))

    def test_a_spec_from_another_version_is_refused(self):
        d = tempfile.mkdtemp(prefix="etfdash-old-")
        try:
            shutil.copy(os.path.join(self.dir, "portfolio.json"), os.path.join(d, "portfolio.json"))
            with open(os.path.join(d, "portfolio.json")) as f:
                spec = json.load(f)
            spec["version"] = ED.SPEC_VERSION - 1
            with open(os.path.join(d, "portfolio.json"), "w") as f:
                json.dump(spec, f)
            with self.assertRaises(SystemExit):
                ED.main(SIGNALS + [d])
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_signals_without_research_says_so(self):
        d = tempfile.mkdtemp(prefix="etfdash-empty-")
        try:
            with self.assertRaises(SystemExit) as cm:
                ED.main(SIGNALS + [d])
            self.assertIn("research", str(cm.exception))
        finally:
            shutil.rmtree(d, ignore_errors=True)


class VerdictTests(unittest.TestCase):
    BASE = dict(nested_sharpe=1.2, nested_max_drawdown=-0.1, pbo_trials=0.1,
                reality_check_p=0.01, dsr_best=0.99, n_eff=5, n_slots=40,
                years_available=12.0, min_btl_years=3.0)
    SLOT = dict(slot="SPY|x", research=dict(cpcv_prob_negative=0.05))

    def test_a_clean_result_passes(self):
        v = ED._verdict(dict(self.BASE), [dict(self.SLOT)])
        self.assertEqual(v["level"], "good")
        self.assertEqual(v["reasons"], [])

    def test_a_losing_nested_curve_is_critical(self):
        v = ED._verdict(dict(self.BASE, nested_sharpe=-0.4), [dict(self.SLOT)])
        self.assertEqual(v["level"], "critical")
        self.assertIn("did not make money out-of-sample", " ".join(v["reasons"]))

    def test_a_coin_toss_pbo_is_critical(self):
        v = ED._verdict(dict(self.BASE, pbo_trials=0.55), [dict(self.SLOT)])
        self.assertEqual(v["level"], "critical")

    def test_snooping_and_weak_dsr_are_flagged(self):
        v = ED._verdict(dict(self.BASE, reality_check_p=0.4, dsr_best=0.5), [dict(self.SLOT)])
        self.assertEqual(v["level"], "critical")
        joined = " ".join(v["reasons"])
        self.assertIn("Reality Check", joined)
        self.assertIn("Deflated Sharpe", joined)

    def test_nothing_selected_is_critical(self):
        self.assertEqual(ED._verdict(dict(self.BASE), [])["level"], "critical")

    def test_a_short_history_only_warns(self):
        v = ED._verdict(dict(self.BASE, years_available=2.0), [dict(self.SLOT)])
        self.assertEqual(v["level"], "warning")
        self.assertIn("Minimum backtest length", " ".join(v["reasons"]))


class RenderTests(unittest.TestCase):
    def _fixture(self, level="warning"):
        spec = dict(
            version=ED.SPEC_VERSION, created="2026-01-01T00:00:00", assets=["SPY"],
            config=dict(interval="1d", start="2010-01-01", family="quick", train_bars=500,
                        test_bars=125, anchored=False, selection="plateau", cost_bps=5.0,
                        risk_pct=0.01, weighting="hrp"),
            slots=[], verdict=dict(level=level, headline="Careful.", reasons=["a reason"]),
            diagnostics=dict(nested_sharpe=0.4, static_sharpe=1.1, pbo_trials=0.3,
                             degradation_slope=-0.5, prob_oos_loss=0.4, pbo_slots=0.35,
                             reality_check_best="SPY|x", reality_check_p=0.08, n_eff=4,
                             n_slots=40, n_trials=900, best_slot="SPY|x", best_oos_sharpe=0.9,
                             sr_star_annual=0.5, dsr_best=0.8, min_btl_years=3.0,
                             years_available=9.0, nested_max_drawdown=-0.12, n_reselections=7),
            curves=dict(dates=["2020-01-0%d" % (i + 1) for i in range(9)],
                        strategy=[1.0, 1.01, 1.03, 1.02, 1.05, 1.04, 1.06, 1.07, 1.08],
                        buy_hold=[1.0, 0.99, 1.02, 1.04, 1.03, 1.06, 1.05, 1.08, 1.09]),
            universe=[dict(slot="SPY|x", asset="SPY", oos_sharpe=0.9, cpcv_mean=0.5,
                           cpcv_prob_negative=0.1, pardo_pass=True, selected=True)],
        )
        state = dict(slot="SPY|x", asset="SPY", template="TR-don-stop-chan-noreg-noV-noB",
                     as_of=pd.Timestamp("2026-01-05"), last_close=500.0, atr=5.0,
                     atr_mult_stop=3.0, exit_style="channel", equity_slot=100_000.0,
                     position=1, shares=66.0, entry_price=490.0,
                     entry_date=pd.Timestamp("2026-01-02"), bars_held=3, unrealized=660.0,
                     n_trades_in_window=4, weight=1.0, params=dict(n_entry=40, n_exit=20),
                     fitted_on="2026-01-05", refit_due=False, bars_since_refit=3,
                     research=dict(oos_sharpe=0.9, cpcv_mean=0.5, cpcv_prob_negative=0.1,
                                   bootstrap_p=0.04),
                     exit_orders=[dict(kind="stop", side=-1, level=485.0, note="hard ATR stop")],
                     entry_orders=[], blocked_by=[])
        targets = portfolio_targets([state], {"SPY|x": 1.0}, 100_000.0)
        trades = trade_list(targets["by_asset"], {"SPY": 10.0})
        run = dict(as_of=pd.Timestamp("2026-01-05"), generated=pd.Timestamp("2026-01-05 18:00"),
                   account_equity=100_000.0, max_gross=1.0,
                   holdings_source="holdings.json", notes=["a note"], bars_available=900)
        return spec, [state], targets, trades, run

    def test_full_and_embedded_documents(self):
        spec, states, targets, trades, run = self._fixture()
        full = render_dashboard(spec, states, targets, trades, run)
        inner = render_dashboard(spec, states, targets, trades, run, full_document=False)
        self.assertTrue(full.startswith("<!doctype html>"))
        self.assertIn("</html>", full)
        self.assertFalse(inner.startswith("<!doctype"))
        self.assertNotIn("<body>", inner)
        self.assertIn("<title>", inner)
        for doc in (full, inner):
            self.assertIn("Careful.", doc)
            self.assertIn("BUY", doc)          # target 66 vs 10 held
            self.assertIn("485.00", doc)       # the published stop

    def test_hostile_names_are_escaped(self):
        spec, states, targets, trades, run = self._fixture()
        states[0]["template"] = '<img src=x onerror="alert(1)">'
        spec["verdict"]["reasons"] = ["</style><script>alert(2)</script>"]
        doc = render_dashboard(spec, states, targets, trades, run)
        self.assertNotIn("<img src=x", doc)
        self.assertNotIn("<script>alert(2)", doc)
        self.assertIn("&lt;img src=x", doc)

    def test_a_missing_curve_does_not_break_the_page(self):
        spec, states, targets, trades, run = self._fixture()
        spec["curves"] = dict(dates=[], strategy=[], buy_hold=[])
        doc = render_dashboard(spec, states, targets, trades, run)
        self.assertIn("No out-of-sample curve yet", doc)

    def test_an_empty_trade_list_keeps_its_columns(self):
        """A flat book with nothing held is a normal quiet day, not an error."""
        empty = trade_list(pd.DataFrame(columns=["shares", "price"],
                                        index=pd.Index([], name="asset")), {})
        self.assertEqual(len(empty), 0)
        for col in ("held", "target", "action", "order_shares", "order_notional"):
            self.assertIn(col, empty.columns)
        self.assertEqual(int((empty["action"] != "hold").sum()), 0)

    def test_a_flat_book_renders(self):
        spec, states, targets, trades, run = self._fixture()
        states[0].update(position=None, shares=0.0, entry_price=None, unrealized=0.0,
                         exit_orders=[], blocked_by=["regime er=0.20 < 0.35 (needs trending)"])
        targets = portfolio_targets(states, {"SPY|x": 1.0}, 100_000.0)
        trades = trade_list(targets["by_asset"], {})
        doc = render_dashboard(spec, states, targets, trades, run)
        self.assertIn("no slot holds a position", doc)
        self.assertIn("needs trending", doc)

    def test_room_to_stop_reads_the_slot_s_own_stop_multiple(self):
        """The meter is a fraction of THIS slot's stop distance, not a default."""
        st = dict(position=1, last_close=500.0, atr=5.0, atr_mult_stop=2.0,
                  exit_orders=[dict(kind="stop", side=-1, level=495.0, note="")])
        frac, level, note = _room_to_stop(st)
        self.assertAlmostEqual(frac, 5.0 / (5.0 * 2.0))       # half the initial distance
        self.assertEqual(level, "")
        st["atr_mult_stop"] = 5.0                              # 20 % of the way left
        frac2, level2, note2 = _room_to_stop(st)
        self.assertAlmostEqual(frac2, 5.0 / (5.0 * 5.0))
        self.assertEqual(level2, "warning")
        self.assertIn("close", note2)
        st["atr_mult_stop"] = 12.0                             # 8 % left
        self.assertEqual(_room_to_stop(st)[1], "critical")
        self.assertIsNone(_room_to_stop(dict(position=None, exit_orders=[]))[0])


if __name__ == "__main__":
    unittest.main()
