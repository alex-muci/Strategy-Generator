"""
Tests for the single-asset entry point (main.py) and the orchestration it
shares with etf_dashboard.py (pipeline.py).

The end-to-end classes run the real `main.main(argv)` over synthetic data in a
temporary output directory. The rest pins the bugs that were found in the
entry point -- each of them silent, none of them visible in the engine tests:

  * --interval applied an intraday annualization factor to the (daily)
    synthetic series, inflating every Sharpe;
  * a history too short for any cell of the walk-forward matrix raised a
    KeyError out of an empty DataFrame;
  * an error while the pool was busy waited for every queued task before it
    surfaced;
  * --real researched on the bar that was still forming.

Run with:  python -m unittest discover -s tests -v
"""

from __future__ import annotations
import contextlib
import io
import os
import shutil
import sys
import tempfile
import time
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main as M  # noqa: E402
import pipeline as P  # noqa: E402
import strategy as S  # noqa: E402
import etf_dashboard as ED  # noqa: E402
from data import synthetic_ohlc  # noqa: E402
from generator import generate_templates  # noqa: E402

BASE = ["--family", "quick", "--max-templates", "4", "--bars", "1100", "--train", "300",
        "--test", "100", "--n-boot", "100", "--min-sharpe", "-5"]


def _main(argv):
    with contextlib.redirect_stdout(io.StringIO()):     # the report is printed too
        return M.main(argv)


def _dash(argv):
    with contextlib.redirect_stdout(io.StringIO()):
        return ED.main(argv)


def _cfg(**over):
    args = M.parse_args([])
    for k, v in over.items():
        setattr(args, k, v)
    return P.eval_config(args, "1d")


class _RestoresAnnualization(unittest.TestCase):
    def setUp(self):
        self._ppy = S.periods_per_year()

    def tearDown(self):
        S.set_periods_per_year(self._ppy)


class EndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp(prefix="main-")
        cls.out = _main(BASE + ["--jobs", "1", "--out", cls.dir])

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, ignore_errors=True)

    def test_every_output_is_written(self):
        for name in ("report.md", "template_ranking.csv", "template_ranking.png",
                     "equity_curves.png", "pbo.png"):
            p = os.path.join(self.dir, name)
            self.assertTrue(os.path.exists(p), f"{name} missing")
            self.assertGreater(os.path.getsize(p), 0, f"{name} empty")
        selected = self.out["portfolio"]["selected"]
        self.assertTrue(selected, "nothing selected at min_sharpe=-5")
        for name in ("selected_windows.csv", "cpcv_distribution.png", "correlation_heatmap.png"):
            self.assertTrue(os.path.exists(os.path.join(self.dir, name)), f"{name} missing")
        self.assertTrue([f for f in os.listdir(self.dir) if f.startswith("wfa_matrix_")])

    def test_report_has_every_section_and_names_the_selection(self):
        text = open(os.path.join(self.dir, "report.md"), encoding="utf-8").read()
        for heading in ("## Family-level overfitting diagnostics", "## Selected strategies",
                        "## Portfolio", "## Benchmark: buy and hold", "## How to read this"):
            self.assertIn(heading, text)
        for name in self.out["portfolio"]["selected"]:
            self.assertIn(f"| {name} |", text)

    def test_ranking_csv_matches_what_was_computed(self):
        table = pd.read_csv(os.path.join(self.dir, "template_ranking.csv"), index_col=0)
        self.assertEqual(sorted(table.index), sorted(self.out["results"]))
        self.assertEqual(len(table), 4)
        self.assertEqual(set(table.index[table["selected"]]), set(self.out["portfolio"]["selected"]))
        # ranked best first, and the Sharpe is the one of the aligned OOS returns
        self.assertTrue(table["oos_sharpe"].is_monotonic_decreasing)
        for name, sr in table["oos_sharpe"].items():
            self.assertAlmostEqual(sr, S.annualized_sharpe(self.out["returns"][name]), places=3)

    def test_portfolio_weights_and_curves_are_consistent(self):
        port, nested = self.out["portfolio"], self.out["nested"]
        self.assertAlmostEqual(float(port["weights"].sum()), 1.0, places=9)
        rets = self.out["returns"]
        expect = (rets[port["selected"]] * port["weights"]).sum(axis=1)
        pd.testing.assert_series_equal(port["portfolio_returns"], expect, check_names=False)
        # the nested portfolio only exists after the minimum history, and only on OOS bars
        self.assertTrue(nested["portfolio_returns"].index.isin(rets.index).all())
        self.assertLess(len(nested["portfolio_returns"]), len(rets))

    def test_templates_carry_the_run_costs(self):
        for res in self.out["results"].values():
            self.assertEqual(res["template"].cost_bps, 5.0)
            self.assertEqual(res["asset"], "synthetic")

    def test_finalists_have_every_diagnostic(self):
        fin = self.out["finalists"]
        self.assertEqual(list(fin), self.out["portfolio"]["selected"])
        for name, d in fin.items():
            for k in ("bootstrap_p", "dsr", "psr0", "cpcv", "summary"):
                self.assertIn(k, d)
            self.assertTrue(0.0 <= d["bootstrap_p"] <= 1.0)
            self.assertTrue(0.0 <= d["dsr"] <= 1.0)
        first = next(iter(fin.values()))
        m = first["wfa_matrix"]
        self.assertEqual(m.index.names, ["train_bars", "test_bars"])
        self.assertTrue(all(tr + te < 1100 for tr, te in m.index))

    def test_family_diagnostics_are_sane(self):
        fam = self.out["family"]
        self.assertEqual(fam["n_templates"], 4)
        self.assertEqual(fam["n_trials"], sum(r["n_trials"] for r in self.out["results"].values()))
        self.assertTrue(0.0 <= fam["pbo_trials"]["pbo"] <= 1.0)
        self.assertTrue(1 <= fam["n_eff"] <= 4)
        self.assertIn(fam["best_template"], self.out["results"])
        # the raw DSR deflates by MORE trials than the clustered one
        self.assertLessEqual(fam["dsr_raw"]["dsr"], fam["dsr_best"]["dsr"] + 1e-9)
        self.assertAlmostEqual(fam["years_available"], len(self.out["returns"]) / 252)

    def test_benchmark_is_the_asset_over_the_oos_bars(self):
        b = self.out["benchmark"]
        rets = self.out["returns"]
        self.assertTrue(b["returns"].index.equals(rets.index))
        df = synthetic_ohlc(n_bars=1100, seed=7)
        expect = df["Close"].pct_change().reindex(rets.index)
        np.testing.assert_allclose(b["returns"].to_numpy(), expect.to_numpy())
        self.assertEqual(b["buy_hold"]["n_bars"], len(rets))


class ParallelTests(unittest.TestCase):
    def test_a_pool_run_gives_the_same_numbers(self):
        """--jobs 2 must reproduce --jobs 1: the results come back in any order
        and the workers are separate processes with their own globals."""
        argv = ["--family", "quick", "--max-templates", "3", "--bars", "900", "--train", "300",
                "--test", "100", "--n-boot", "50", "--min-sharpe", "-5", "--no-matrix"]
        d1, d2 = tempfile.mkdtemp(prefix="main-j1-"), tempfile.mkdtemp(prefix="main-j2-")
        try:
            a = _main(argv + ["--jobs", "1", "--out", d1])
            b = _main(argv + ["--jobs", "2", "--out", d2])
            self.assertEqual(list(a["results"]), list(b["results"]))
            pd.testing.assert_frame_equal(a["returns"], b["returns"])
            self.assertEqual(a["portfolio"]["selected"], b["portfolio"]["selected"])
            self.assertAlmostEqual(a["family"]["pbo_trials"]["pbo"], b["family"]["pbo_trials"]["pbo"])
            t1 = open(os.path.join(d1, "template_ranking.csv")).read()
            t2 = open(os.path.join(d2, "template_ranking.csv")).read()
            self.assertEqual(t1, t2)
        finally:
            shutil.rmtree(d1, ignore_errors=True)
            shutil.rmtree(d2, ignore_errors=True)

    def test_an_error_terminates_the_pool_instead_of_draining_it(self):
        """close()+join() would sit through all the queued sleeps (~18 s)."""
        t0 = time.time()
        with self.assertRaises(RuntimeError):
            with P.worker_pool(2, {}, _cfg()) as pool:
                it = pool.imap_unordered(time.sleep, [3.0] * 12)
                next(it)
                raise RuntimeError("something downstream failed")
        self.assertLess(time.time() - t0, 12.0)

    def test_no_pool_for_a_single_job(self):
        with P.worker_pool(1, {"x": None}, _cfg()) as pool:
            self.assertIsNone(pool)
            self.assertEqual(sorted(P.pool_map(pool, abs, [-1, 2])), [1, 2])


class EntryPointRegressionTests(_RestoresAnnualization):
    def test_synthetic_data_is_annualized_daily_whatever_interval_says(self):
        d = tempfile.mkdtemp(prefix="main-1h-")
        try:
            out = _main(["--max-templates", "2", "--bars", "800", "--train", "300", "--test", "100",
                          "--n-boot", "20", "--min-sharpe", "-5", "--no-matrix", "--jobs", "1",
                          "--interval", "1h", "--out", d])
        finally:
            shutil.rmtree(d, ignore_errors=True)
        self.assertEqual(S.periods_per_year(), 252)
        self.assertAlmostEqual(out["family"]["years_available"], len(out["returns"]) / 252)

    def test_resolve_interval(self):
        self.assertEqual(P.resolve_interval("1h", synthetic=True), "1d")
        self.assertEqual(P.resolve_interval("1h", synthetic=False), "1h")
        self.assertEqual(P.eval_config(M.parse_args([]), "1h")["periods_per_year"], 252 * 7)

    def test_the_dashboard_research_annualizes_synthetic_data_daily_too(self):
        d = tempfile.mkdtemp(prefix="dash-1h-")
        try:
            spec = _dash(["research", "--synthetic", "--assets", "AAA", "--max-templates", "2",
                            "--bars", "800", "--train", "300", "--test", "100", "--jobs", "1",
                            "--n-boot", "20", "--min-sharpe", "-5", "--interval", "1h",
                            "--state-dir", d])
        finally:
            shutil.rmtree(d, ignore_errors=True)
        self.assertEqual(spec["config"]["interval"], "1d")
        self.assertEqual(spec["config"]["periods_per_year"], 252)

    def test_history_too_short_for_any_matrix_cell(self):
        """250 + 63 bars is the smallest cell; with fewer there is nothing to
        run, which used to be a KeyError from grouping an empty frame."""
        df = synthetic_ohlc(n_bars=300, seed=3)
        tpl = generate_templates("quick", max_templates=1)[0]
        results = {tpl.name: dict(asset="a", template=tpl)}
        with P.worker_pool(1, {"a": df}, _cfg()) as pool:
            self.assertEqual(P.walk_forward_matrices(results, [tpl.name], pool), {})
            self.assertEqual(P.walk_forward_matrices(results, [], pool), {})

    def test_too_little_history_exits_with_a_message(self):
        d = tempfile.mkdtemp(prefix="main-short-")
        try:
            with self.assertRaises(SystemExit) as cm:
                _main(["--max-templates", "2", "--bars", "400", "--train", "500", "--jobs", "1", "--out", d])
        finally:
            shutil.rmtree(d, ignore_errors=True)
        self.assertIn("too short", str(cm.exception))

    def test_nothing_selected_still_writes_a_report(self):
        d = tempfile.mkdtemp(prefix="main-none-")
        try:
            out = _main(["--max-templates", "2", "--bars", "800", "--train", "300", "--test", "100",
                          "--n-boot", "20", "--min-sharpe", "50", "--jobs", "1", "--out", d])
            self.assertEqual(out["portfolio"]["selected"], [])
            self.assertEqual(out["finalists"], {})
            self.assertIsNone(out["benchmark"]["static"])
            self.assertTrue(os.path.exists(os.path.join(d, "report.md")))
            self.assertFalse(os.path.exists(os.path.join(d, "cpcv_distribution.png")))
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_real_data_loses_its_forming_bar(self):
        idx = pd.bdate_range("2024-01-01", periods=300)
        full = pd.DataFrame({c: np.linspace(100, 110, 300) for c in ("Open", "High", "Low", "Close")}, index=idx)
        full["Volume"] = 1.0
        seen = {}

        def fake(ticker, start, interval):
            seen.update(ticker=ticker, start=start, interval=interval)
            return full

        orig = P.load_yfinance
        P.load_yfinance = fake
        try:
            during = P.load_real("SPY", "2024-01-01", "1d", now=idx[-1] + pd.Timedelta(hours=15))
            after = P.load_real("SPY", "2024-01-01", "1d", now=idx[-1] + pd.Timedelta(days=1, hours=1))
        finally:
            P.load_yfinance = orig
        self.assertEqual(seen, dict(ticker="SPY", start="2024-01-01", interval="1d"))
        self.assertEqual(len(during), 299)
        self.assertEqual(during.index[-1], idx[-2])
        self.assertEqual(len(after), 300)

    def test_output_file_names_are_portable(self):
        for name in ("TR-don-stop-chan-er:range-V-noB", 'GC=F|a/b\\c*?"<>'):
            safe = M._safe_filename(name)
            self.assertRegex(safe, r"^[A-Za-z0-9._-]+$")
        self.assertEqual(M._safe_filename("TR-don-stop-chan-er:range-V-noB"), "TR-don-stop-chan-er_range-V-noB")


class ParseArgsTests(unittest.TestCase):
    def test_defaults(self):
        a = M.parse_args([])
        self.assertEqual((a.family, a.train, a.test, a.interval), ("quick", 500, 125, "1d"))
        self.assertIsNone(a.real)
        self.assertFalse(a.anchored or a.no_matrix or a.require_pardo or a.wide_grid)
        # the sizing the run uses is the engine's own default unless asked otherwise
        tpl = S.StrategyTemplate(name="x")
        self.assertEqual((a.cost_bps, a.risk_pct, a.max_leverage), (tpl.cost_bps, tpl.risk_pct, tpl.max_leverage))

    def test_bad_choices_are_rejected(self):
        for argv in (["--family", "nope"], ["--interval", "2h"], ["--metric", "sortino"], ["--sides", "up"]):
            with self.assertRaises(SystemExit):
                M.parse_args(argv)

    def test_sizing_and_sides_reach_the_templates(self):
        cfg = _cfg(risk_pct=0.02, max_leverage=1.0, cost_bps=9.0)
        tpl = P._costed(generate_templates("quick", max_templates=1, sides=["long_only"])[0], cfg)
        self.assertEqual((tpl.risk_pct, tpl.max_leverage, tpl.cost_bps, tpl.sides), (0.02, 1.0, 9.0, "long_only"))

    def test_the_two_entry_points_evaluate_with_the_same_settings(self):
        """Every evaluation setting of one CLI exists, with the same default, in
        the other (portfolio weighting is a deliberate difference, not one of them)."""
        m = P.eval_config(M.parse_args([]), "1d")
        e = P.eval_config(ED.parse_args(["research"]), "1d")
        self.assertEqual(m, e)


class BenchmarkMathTests(_RestoresAnnualization):
    def setUp(self):
        super().setUp()
        S.set_periods_per_year(252)
        self.idx = pd.bdate_range("2020-01-01", periods=504)
        self.rng = np.random.default_rng(5)

    def test_curve_stats_of_a_known_curve(self):
        r = pd.Series(0.0, index=self.idx)
        r.iloc[100] = 0.10
        r.iloc[200] = -0.50
        r.iloc[300] = 0.20
        s = P.curve_stats(r)
        self.assertEqual(s["n_bars"], 504)
        self.assertAlmostEqual(s["max_dd"], -0.5)
        self.assertAlmostEqual(s["cagr"], (1.1 * 0.5 * 1.2) ** 0.5 - 1)
        self.assertAlmostEqual(s["sharpe"], S.annualized_sharpe(r))

    def test_curve_stats_edge_cases(self):
        self.assertEqual(P.curve_stats(pd.Series([0.01])), dict(sharpe=0.0, cagr=0.0, max_dd=0.0, n_bars=1))
        self.assertEqual(P.curve_stats(pd.Series(dtype=float))["n_bars"], 0)
        wiped = P.curve_stats(pd.Series([0.1, -1.0, 0.1]))
        self.assertEqual(wiped["cagr"], -1.0)
        self.assertEqual(wiped["max_dd"], -1.0)

    def test_max_drawdown(self):
        self.assertEqual(S.max_drawdown([]), 0.0)
        self.assertEqual(S.max_drawdown([1.0, 2.0, 3.0]), 0.0)
        self.assertAlmostEqual(S.max_drawdown(pd.Series([100.0, 120.0, 90.0, 130.0, 117.0])), -0.25)

    def test_beta_corr_and_information_ratio(self):
        x = pd.Series(self.rng.normal(0.0003, 0.01, 504), index=self.idx)
        noise = pd.Series(self.rng.normal(0.0, 0.0005, 504), index=self.idx)
        a = P.against_benchmark(2 * x + noise, x)
        self.assertAlmostEqual(a["beta"], 2.0, delta=0.05)
        self.assertGreater(a["corr"], 0.99)
        # a pure multiple of the benchmark has nothing left once beta is removed
        pure = P.against_benchmark(2 * x, x)
        self.assertAlmostEqual(pure["beta"], 2.0)
        self.assertAlmostEqual(pure["corr"], 1.0)
        # ... and a constant edge on top of it is all information ratio
        alpha = pd.Series(np.where(np.arange(504) % 2, 0.0010, 0.0006), index=self.idx)
        edge = P.against_benchmark(0.5 * x + alpha, x)
        self.assertAlmostEqual(edge["beta"], 0.5, delta=0.01)
        self.assertGreater(edge["info_ratio"], 10)

    def test_undefined_against_a_flat_stream(self):
        x = pd.Series(self.rng.normal(0, 0.01, 50), index=self.idx[:50])
        flat = pd.Series(0.0, index=self.idx[:50])
        for a in (P.against_benchmark(flat, x), P.against_benchmark(x, flat), P.against_benchmark(x.iloc[:2], x)):
            self.assertTrue(all(np.isnan(v) for v in a.values()))

    def test_benchmark_is_aligned_on_the_portfolio_dates(self):
        """The benchmark series covers the whole history; the strategies only
        the OOS part, the nested portfolio less still."""
        bh = pd.Series(self.rng.normal(0.0004, 0.01, 504), index=self.idx)
        bh.iloc[0] = np.nan                               # pct_change's first bar
        oos = self.idx[100:]
        rets = pd.DataFrame({"good": bh[oos] * 0.2 + 0.001, "bad": -bh[oos] * 0.2 - 0.001})
        port = dict(portfolio_returns=rets["good"])
        nested = dict(portfolio_returns=rets["good"].iloc[200:])
        b = P.benchmark_stats(bh, rets, port, nested)
        self.assertTrue(b["returns"].index.equals(rets.index))
        self.assertEqual(b["n_templates"], 2)
        self.assertEqual(b["n_templates_beat_bh"], 1)
        self.assertEqual(b["buy_hold"]["n_bars"], 404)
        self.assertEqual(b["buy_hold_nested_period"]["n_bars"], 204)
        self.assertAlmostEqual(b["static"]["beta"], 0.2, places=6)
        self.assertAlmostEqual(b["nested"]["corr"], 1.0, places=6)
        self.assertAlmostEqual(b["buy_hold_nested_period"]["sharpe"], S.annualized_sharpe(bh[oos].iloc[200:]))

    def test_no_portfolio_no_comparison(self):
        bh = pd.Series(self.rng.normal(0, 0.01, 504), index=self.idx)
        rets = pd.DataFrame({"a": bh * 0.1})
        empty = dict(portfolio_returns=pd.Series(dtype=float))
        b = P.benchmark_stats(bh, rets, empty, empty)
        self.assertIsNone(b["static"])
        self.assertIsNone(b["nested"])
        self.assertIsNone(b["buy_hold_nested_period"])

    def test_cscv_partitions(self):
        self.assertEqual([P.cscv_partitions_for(t) for t in (100, 1599, 1600, 5000)], [8, 8, 16, 16])


class SharedPipelineTests(unittest.TestCase):
    """main.py on one asset and the dashboard's research on the same asset are
    the same research: same slots, same numbers."""

    @classmethod
    def setUpClass(cls):
        cls.d1, cls.d2 = tempfile.mkdtemp(prefix="parity-main-"), tempfile.mkdtemp(prefix="parity-dash-")
        common = ["--family", "quick", "--max-templates", "3", "--train", "300", "--test", "100",
                  "--n-boot", "50", "--min-sharpe", "-5", "--jobs", "1", "--weighting", "equal"]
        # the dashboard's first synthetic asset is synthetic_ohlc(seed=100, trend_prob=0.45)
        cls.main = _main(common + ["--bars", "1000", "--seed", "100", "--trend-prob", "0.45",
                                    "--no-matrix", "--out", cls.d1])
        cls.spec = _dash(["research", "--synthetic", "--assets", "AAA", "--bars", "1000",
                            "--state-dir", cls.d2] + common)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.d1, ignore_errors=True)
        shutil.rmtree(cls.d2, ignore_errors=True)

    def test_same_universe_same_sharpes(self):
        ranked = {row["slot"].split("|", 1)[1]: row for row in self.spec["universe"]}
        self.assertEqual(sorted(ranked), sorted(self.main["results"]))
        for name, row in ranked.items():
            # the spec rounds to 3 decimals
            self.assertAlmostEqual(row["oos_sharpe"], S.annualized_sharpe(self.main["returns"][name]), delta=6e-4)

    def test_same_family_diagnostics(self):
        fam, diag = self.main["family"], self.spec["diagnostics"]
        self.assertEqual(diag["n_trials"], fam["n_trials"])
        self.assertEqual(diag["n_eff"], fam["n_eff"])
        self.assertAlmostEqual(diag["pbo_trials"], fam["pbo_trials"]["pbo"])
        self.assertAlmostEqual(diag["reality_check_p"], fam["reality_check"]["p_value"])
        self.assertAlmostEqual(diag["dsr_best"], fam["dsr_best"]["dsr"])
        self.assertAlmostEqual(diag["dsr_raw"], fam["dsr_raw"]["dsr"])
        self.assertEqual(diag["best_slot"], "AAA|" + fam["best_template"])

    def test_same_selection_and_same_honest_sharpe(self):
        self.assertEqual([s["template_name"] for s in self.spec["slots"]], self.main["portfolio"]["selected"])
        self.assertAlmostEqual(self.spec["diagnostics"]["nested_sharpe"], self.main["nested"]["sharpe"])
        for s in self.spec["slots"]:
            f = self.main["finalists"][s["template_name"]]
            self.assertAlmostEqual(s["research"]["bootstrap_p"], f["bootstrap_p"])
            self.assertAlmostEqual(s["research"]["dsr"], f["dsr"])


if __name__ == "__main__":
    unittest.main()
