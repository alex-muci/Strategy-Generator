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

    def test_the_gross_cap_reaches_positions_and_entry_orders(self):
        """Whatever scale the cap applies to the targets, signals() applies to
        every share count it publishes (page, CSV log, futures levels)."""
        free = ED.main(SIGNALS + [self.dir, "--no-refit"])
        real = ED.portfolio_targets
        ED.portfolio_targets = lambda *a, **k: dict(real(*a, **k), scale_applied=0.5)
        try:
            capped = ED.main(SIGNALS + [self.dir, "--no-refit"])
        finally:
            ED.portfolio_targets = real
        n = 0
        for f, c in zip(free["states"], capped["states"]):
            self.assertAlmostEqual(c["shares"], f["shares"] * 0.5, places=6)
            self.assertEqual(len(f["entry_orders"]), len(c["entry_orders"]))
            for of, oc in zip(f["entry_orders"], c["entry_orders"]):
                self.assertAlmostEqual(oc["shares"], of["shares"] * 0.5, places=6)
                n += 1
        if not n and not any(f["position"] for f in free["states"]):
            self.skipTest("the fixture has neither a position nor an entry order")

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


class _FakeHourlyYF:
    """yfinance for 1h bars: stamps in the exchange's zone, as the real one does."""
    def download(self, ticker, **kw):
        from data import synthetic_ohlc
        df = synthetic_ohlc(n_bars=1500, seed=sum(map(ord, ticker)))
        df.index = pd.date_range("2025-01-02 09:30", periods=len(df), freq="h", tz="America/New_York")
        return df


class HourlySignalsTests(unittest.TestCase):
    """The hourly workflow end to end, with tz-aware intraday stamps from the feed."""

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="etfdash-1h-")
        ED.main(RESEARCH + [self.dir])
        p = os.path.join(self.dir, "portfolio.json")
        with open(p) as f:
            spec = json.load(f)
        spec["assets"] = ["SPY", "TLT"]
        for s in spec["slots"]:
            s["asset"] = {"AAA": "SPY", "BBB": "TLT"}[s["asset"]]
            s["slot"] = s["asset"] + "|" + s["template_name"]
        spec["config"].update(interval="1h", periods_per_year=1764)
        with open(p, "w") as f:
            json.dump(spec, f)
        self.real = sys.modules.get("yfinance")
        sys.modules["yfinance"] = _FakeHourlyYF()

    def tearDown(self):
        if self.real is None:
            sys.modules.pop("yfinance", None)
        else:
            sys.modules["yfinance"] = self.real
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_hourly_signals_render_on_the_utc_clock(self):
        out = ED.main(["signals", "--interval", "1h", "--jobs", "1", "--now", "2026-01-01",
                       "--state-dir", self.dir])
        self.assertIsNone(pd.Timestamp(out["run"]["as_of"]).tz)
        with open(os.path.join(self.dir, "dashboard.html"), encoding="utf-8") as f:
            doc = f.read()
        self.assertIn(" UTC", doc)
        # a second run reads its own persisted fitted_on back without a refit
        out2 = ED.main(["signals", "--interval", "1h", "--jobs", "1", "--now", "2026-01-01",
                        "--state-dir", self.dir])
        self.assertFalse([n for n in out2["run"]["notes"] if "re-optimized" in n])


class SignalsWiringTests(unittest.TestCase):
    """Fixes to how `signals` feeds the live layer, without a research run."""

    def test_an_anchored_spec_loads_from_the_research_start(self):
        class Loaded(Exception):
            pass
        seen = {}

        def fake_load_assets(assets, **kw):
            seen.update(kw)
            raise Loaded

        d = tempfile.mkdtemp(prefix="etfdash-anch-")
        real = ED.load_assets
        try:
            for anchored, want in ((True, "2005-01-01"), (False, None)):
                spec = dict(version=ED.SPEC_VERSION, assets=["SPY"], slots=[],
                            config=dict(interval="1d", periods_per_year=252, train_bars=500,
                                        test_bars=125, anchored=anchored, start="2005-01-01"))
                with open(os.path.join(d, "portfolio.json"), "w") as f:
                    json.dump(spec, f)
                ED.load_assets = fake_load_assets
                with self.assertRaises(Loaded):
                    ED.main(["signals", "--state-dir", d])
                if want:
                    self.assertEqual(seen["start"], want)
                else:
                    self.assertGreater(seen["start"], "2005-01-01")
        finally:
            ED.load_assets = real
            shutil.rmtree(d, ignore_errors=True)

    def test_a_slot_that_could_not_be_fitted_waits_for_its_window(self):
        from dataclasses import asdict
        from types import SimpleNamespace
        from data import synthetic_ohlc
        from generator import generate_templates
        calls = []

        def fake_refit(df, tpl, grid, **kw):
            calls.append(df.index[-1])
            return dict(params=None, is_stats=None, is_score=None, fitted_on=df.index[-1],
                        train_bars=len(df))

        tpl = generate_templates("quick")[0]
        slot = dict(slot="SPY|x", asset="SPY", template_name=tpl.name, template=asdict(tpl),
                    weight=1.0, research={})
        cfg = dict(train_bars=500, test_bars=125, wide_grid=False, metric="sharpe",
                   selection="plateau", anchored=False)
        live = dict(slots={}, last_targets={}, runs=0)
        args = SimpleNamespace(account_equity=1e5, no_refit=False)
        df = synthetic_ohlc(1100, seed=1)
        real = ED.refit_params
        ED.refit_params = fake_refit
        try:
            for t in range(900, 905):                  # five runs inside one test window
                st, _ = ED._slot_signal(slot, df.iloc[:t], cfg, live, args)
                self.assertIsNone(st["position"])
            self.assertEqual(len(calls), 1)
            ED._slot_signal(slot, df.iloc[:900 + 125], cfg, live, args)   # a window later
            self.assertEqual(len(calls), 2)
        finally:
            ED.refit_params = real


class FuturesBookTests(unittest.TestCase):
    """Per-asset sides, the portfolio vol scale and the futures restatement,
    on synthetic series named after ETFs that have a contract."""

    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp(prefix="etfdash-fut-")
        research = [a for a in RESEARCH if a not in ("AAA", "BBB")]
        i = research.index("--assets") + 1
        research[i:i] = ["SPY", "IEF"]
        cls.research = research
        cls.spec = ED.main(research + [cls.dir, "--vol-target", "0.15", "--max-leverage", "6",
                                       "--portfolio-vol", "0.15", "--sides-map", "SPY=long_only",
                                       "--max-strategies", "6", "--corr-ceiling", "1.0"])
        cls.out = ED.main(SIGNALS + [cls.dir, "--futures", "--max-gross", "10"])

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, ignore_errors=True)

    def _copy(self) -> str:
        """The state dir, copied: a second signals run rewrites the page and the
        live state the other tests read."""
        d = tempfile.mkdtemp(prefix="etfdash-fut-copy-")
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        shutil.copytree(self.dir, d, dirs_exist_ok=True)
        return d

    def test_the_sides_map_reaches_its_asset_only(self):
        by_asset = {}
        for row in self.spec["universe"]:
            by_asset.setdefault(row["asset"], []).append(row["slot"])
        self.assertTrue(all(s.endswith("-L") for s in by_asset["SPY"]))
        self.assertFalse(any(s.endswith(("-L", "-S")) for s in by_asset["IEF"]))
        self.assertEqual(len(by_asset["SPY"]), len(by_asset["IEF"]))
        for slot in self.spec["slots"]:
            self.assertEqual(slot["template"]["sides"], "long_only" if slot["asset"] == "SPY" else "both")
        self.assertEqual(self.spec["config"]["sides_map"], {"SPY": ["long_only"]})

    def test_a_bad_sides_map_is_an_error(self):
        for bad in ("QQQ=long_only", "SPY=sideways"):
            with self.assertRaises(SystemExit):
                ED.main(self.research + [self.dir, "--sides-map", bad])

    def test_the_risk_scale_takes_the_nested_curve_to_the_target(self):
        vol, scale = self.spec["diagnostics"]["nested_vol"], self.spec["risk_scale"]
        self.assertGreater(vol, 0)
        lo, hi = ED.RISK_SCALE_BOUNDS
        self.assertAlmostEqual(scale, float(np.clip(0.15 / vol, lo, hi)), places=9)
        self.assertEqual(self.out["run"]["risk_scale"], scale)

    def test_empty_selection_windows_do_not_dilute_the_volatility(self):
        idx = pd.bdate_range("2020-01-01", periods=200)
        r = pd.Series(np.where(np.arange(200) % 2, 0.01, -0.01), index=idx)
        r.iloc[:100] = 0.0
        nested = dict(portfolio_returns=r, selections=[
            dict(period_start=idx[0], selected=[]), dict(period_start=idx[100], selected=["x"])])
        scale, vol = ED._risk_scale(nested, 0.0)
        self.assertEqual(scale, 1.0)
        self.assertAlmostEqual(vol, float(r.iloc[100:].std() * np.sqrt(252)), places=9)

    def test_sizes_are_linear_in_the_risk_scale(self):
        d = self._copy()
        one = ED.main(SIGNALS + [d, "--risk-scale", "1", "--max-gross", "100", "--no-refit"])
        two = ED.main(SIGNALS + [d, "--risk-scale", "2", "--max-gross", "100", "--no-refit"])
        for a, b in zip(one["states"], two["states"]):
            self.assertAlmostEqual(b["shares"], 2 * a["shares"], places=6)
            self.assertAlmostEqual(b["equity_slot"], 2 * a["equity_slot"], places=6)

    def test_a_spec_without_a_risk_scale_is_sized_unscaled(self):
        d = tempfile.mkdtemp(prefix="etfdash-noscale-")
        try:
            spec = {k: v for k, v in self.spec.items() if k != "risk_scale"}
            with open(os.path.join(d, "portfolio.json"), "w") as f:
                json.dump(spec, f, default=str)
            self.assertEqual(ED.main(SIGNALS + [d])["run"]["risk_scale"], 1.0)
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_the_futures_book_is_written_and_shown(self):
        fut = self.out["run"]["futures"]
        for name in ("futures_orders.csv", "futures_levels.csv"):
            self.assertTrue(os.path.exists(os.path.join(self.dir, name)), f"{name} missing")
        html = open(os.path.join(self.dir, "dashboard.html"), encoding="utf-8").read()
        self.assertIn("Futures orders", html)
        self.assertTrue(set(fut["book"]["root"]) <= {"MES", "ZN"})
        for a, r in fut["book"].iterrows():
            want = self.out["targets"]["by_asset"]["notional"].get(a, 0.0) * r["hedge_ratio"] / r["contract_value"]
            self.assertAlmostEqual(r["raw"], want, places=6)
            self.assertLessEqual(abs(r["target"] - r["raw"]), 0.5 + 1e-9)
        with open(os.path.join(self.dir, "live_state.json")) as f:
            self.assertIn("last_futures_targets", json.load(f))

    def test_a_hand_priced_euro_contract_goes_through_the_quotes_file(self):
        # the same spec on a Bund ETF and VIXY: FGBL has no feed even live, so
        # its price must come from futures_quotes.json and its ratio from the
        # contract default; VXM has a (synthetic) feed
        d = self._copy()
        for name in ("portfolio.json", "live_state.json"):
            path = os.path.join(d, name)
            if os.path.exists(path):
                txt = open(path, encoding="utf-8").read().replace('"SPY', '"EXHD.DE').replace('"IEF', '"VIXY')
                open(path, "w", encoding="utf-8").write(txt)
        out = ED.main(SIGNALS + [d, "--futures", "--max-gross", "100"])
        fut = out["run"]["futures"]
        self.assertTrue(any("FGBL" in n and "futures_quotes.json" in n for n in fut["notes"]))
        self.assertNotIn("EXHD.DE", fut["book"].index)
        with open(os.path.join(d, "futures_quotes.json"), "w") as f:
            json.dump({"FGBL": 130.0}, f)
        out = ED.main(SIGNALS + [d, "--futures", "--max-gross", "100", "--no-refit"])
        fut = out["run"]["futures"]
        b = fut["book"]
        if "EXHD.DE" in b.index:
            self.assertEqual(b.loc["EXHD.DE", "price_source"], "futures_quotes.json")
            self.assertEqual(b.loc["EXHD.DE", "currency"], "EUR")
            self.assertEqual(b.loc["EXHD.DE", "hedge_ratio"], 0.9)
            self.assertAlmostEqual(b.loc["EXHD.DE", "contract_value"], 130_000.0)   # synthetic FX is 1
        if "VIXY" in b.index:
            self.assertEqual(b.loc["VIXY", "root"], "VXM")
        self.assertFalse(any("stale" in n for n in fut["notes"]))
        os.utime(os.path.join(d, "futures_quotes.json"), (0, 0))
        fut = ED.main(SIGNALS + [d, "--futures", "--max-gross", "100", "--no-refit"])["run"]["futures"]
        self.assertTrue(any("days old" in n for n in fut["notes"]))
        html = open(os.path.join(d, "dashboard.html"), encoding="utf-8").read()
        self.assertIn("futures_quotes.json", html)

    def test_without_the_flag_the_page_has_no_futures_section(self):
        d = self._copy()
        out = ED.main(SIGNALS + [d, "--no-refit"])
        self.assertIsNone(out["run"]["futures"])
        self.assertNotIn("Futures orders", open(os.path.join(d, "dashboard.html"), encoding="utf-8").read())


class FxRateTests(unittest.TestCase):
    """The euro rate comes through the same loader as every other series, and
    that loader refuses a history shorter than 200 bars."""

    def setUp(self):
        self.orig = ED.load_real
        ED._FX_CACHE.clear()
        self.seen = []

        def fake(ticker, start, interval="1d", now=None, session_close=None):
            self.seen.append(dict(ticker=ticker, start=start, interval=interval))
            idx = pd.bdate_range(start, ED.utcnow().normalize())
            if len(idx) < 200:   # what data.load_yfinance(min_bars=200) does
                raise ValueError(f"{ticker!r}: only {len(idx)} usable bars, need at least 200")
            return pd.DataFrame({c: np.linspace(1.05, 1.10, len(idx)) for c in ("Open", "High", "Low", "Close")},
                                index=idx)

        ED.load_real = fake

    def tearDown(self):
        ED.load_real = self.orig
        ED._FX_CACHE.clear()

    def test_the_rate_window_clears_the_loaders_floor(self):
        rate = ED.fx_rate("EUR")
        self.assertAlmostEqual(rate, 1.10)
        self.assertEqual(self.seen[0]["ticker"], "EURUSD=X")

    def test_the_rate_is_fetched_once_per_run(self):
        ED.fx_rate("EUR")
        ED.fx_rate("EUR")
        self.assertEqual(len(self.seen), 1)
        self.assertEqual(ED.fx_rate("USD"), 1.0)
        self.assertEqual(len(self.seen), 1)

    def test_an_unmapped_currency_is_an_error(self):
        with self.assertRaises(ValueError):
            ED.fx_rate("BRL")


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

    def test_an_aware_bar_stamp_does_not_break_the_page(self):
        spec, states, targets, trades, run = self._fixture()
        run["as_of"] = pd.Timestamp("2026-01-05 15:30", tz="America/New_York")
        doc = render_dashboard(spec, states, targets, trades, run)
        self.assertIn("2026-01-05 20:30 UTC", doc)

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
