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
import json
import os
import pickle
import shutil
import sys
import tempfile
import time
import unittest
from multiprocessing import get_context
from unittest import mock

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


def _seen_by_worker(asset):
    """What a pool worker holds for `asset`: the frame, and the annualization
    the config asked for, as its own strategy module reads it."""
    return asset, os.getpid(), P._DATA[asset], P._CFG["periods_per_year"], S.periods_per_year()


class _RestoresAnnualization(unittest.TestCase):
    def setUp(self):
        self._ppy = S.periods_per_year()
        self._share = S.HEDGE_SHARE

    def tearDown(self):
        S.set_periods_per_year(self._ppy)
        S.set_hedge_share(self._share)


def _share_seen_by_worker(_):
    return P._CFG["hedge_share"], S.HEDGE_SHARE


class HedgeShareConfigTests(_RestoresAnnualization):
    """--hedge-share picks how the hedge learners forget; it travels in the
    run's config to every worker, and a config written before the option
    existed (no key) is replayed the way it ran: discounted."""

    def test_the_flag_reaches_the_config_and_the_workers(self):
        self.assertEqual(M.parse_args([]).hedge_share, "fixed_share")
        self.assertEqual(_cfg()["hedge_share"], "fixed_share")
        cfg = _cfg(hedge_share="discount")
        self.assertEqual(cfg["hedge_share"], "discount")
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            M.parse_args(["--hedge-share", "decay"])
        with P.worker_pool(2, {"a": None}, cfg, context=get_context("spawn")) as pool:
            self.assertEqual(S.HEDGE_SHARE, "discount")            # this process too
            seen = set(P.pool_map(pool, _share_seen_by_worker, range(4)))
        self.assertEqual(seen, {("discount", "discount")})

    def test_an_old_config_runs_discounted(self):
        self.assertEqual(P.hedge_share_of({"periods_per_year": 252}), "discount")
        self.assertEqual(P.hedge_share_of(_cfg()), "fixed_share")
        P.init_worker({}, {"periods_per_year": 252})
        self.assertEqual(S.HEDGE_SHARE, "discount")


class VolTargetRunTests(unittest.TestCase):
    """A run sized to a vol target: the flag reaches every template, the report
    says so, and the chart is drawn (buy & hold on the strategies' axis)."""

    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp(prefix="main-vt-")
        cls.out = _main(BASE + ["--jobs", "1", "--no-matrix", "--vol-target", "0.15", "--out", cls.dir])

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, ignore_errors=True)

    def test_every_template_is_sized_to_the_target(self):
        for res in self.out["results"].values():
            self.assertEqual(res["template"].vol_target, 0.15)
            self.assertEqual(res["template"].vol_target_n, 60)

    def test_report_and_chart_state_the_rule(self):
        text = open(os.path.join(self.dir, "report.md"), encoding="utf-8").read()
        self.assertIn("15% annualized volatility", text)
        self.assertNotIn("smaller scale than an unlevered holding", text)
        p = os.path.join(self.dir, "equity_curves.png")
        self.assertTrue(os.path.exists(p) and os.path.getsize(p) > 0)


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
                     "equity_curves.png", "pbo.png", "run.json", "data.csv", "portfolio_returns.csv"):
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
        """close()+join() would sit through the ten sleeps still queued (~15 s).
        The clock starts at the error, not at the pool: spawning two workers
        (each imports numba and pandas) takes longer than that on its own when
        the suite runs in parallel and every core is busy."""
        with self.assertRaises(RuntimeError):
            with P.worker_pool(2, {}, _cfg()) as pool:
                it = pool.imap_unordered(time.sleep, [3.0] * 12)
                next(it)
                t0 = time.time()
                raise RuntimeError("something downstream failed")
        self.assertLess(time.time() - t0, 8.0)

    def test_no_pool_for_a_single_job(self):
        with P.worker_pool(1, {"x": None}, _cfg()) as pool:
            self.assertIsNone(pool)
            self.assertEqual(sorted(P.pool_map(pool, abs, [-1, 2])), [1, 2])

    # spawn is what Windows (and macOS) run: the workers inherit nothing from
    # this process, so whatever they see of the data went through worker_pool.
    def test_spawned_workers_see_the_full_frames_and_the_config(self):
        data = {"a": synthetic_ohlc(n_bars=500, seed=1), "b": synthetic_ohlc(n_bars=700, seed=2)}
        cfg = P.eval_config(M.parse_args([]), "1h")    # intraday: a worker left at the daily default shows
        ppy0 = S.periods_per_year()
        try:
            with P.worker_pool(2, data, cfg, context=get_context("spawn")) as pool:
                self.assertIs(P._DATA["a"], data["a"])  # this process: the frames themselves, not a copy
                seen = list(P.pool_map(pool, _seen_by_worker, ["a", "b"] * 4))
        finally:
            S.set_periods_per_year(ppy0)
        self.assertEqual(len(seen), 8)
        for asset, pid, df, cfg_ppy, ppy in seen:
            self.assertNotEqual(pid, os.getpid())
            pd.testing.assert_frame_equal(df, data[asset])
            self.assertEqual((cfg_ppy, ppy), (252 * 7, 252 * 7))

    def test_the_frames_reach_the_workers_by_file_not_through_initargs(self):
        """Under spawn Process.start() writes initargs into a pipe the child
        drains only after re-importing __main__, so with the frames in
        initargs Pool() took n_jobs imports, one after the other. initargs
        must stay small; the pickle is removed with the pool."""
        data = {"a": synthetic_ohlc(n_bars=2000, seed=1)}
        ctx = get_context("spawn")
        with mock.patch.object(ctx, "Pool", wraps=ctx.Pool) as spy:
            with P.worker_pool(2, data, _cfg(), context=ctx) as pool:
                path, cfg = spy.call_args.kwargs["initargs"]
                self.assertTrue(os.path.exists(path))
                self.assertEqual(cfg, _cfg())
                self.assertLess(len(pickle.dumps((path, cfg))), 4096)
                self.assertGreater(len(pickle.dumps(data)), 4096 * 10)
                with open(path, "rb") as f:
                    pd.testing.assert_frame_equal(pickle.load(f)["a"], data["a"])
                self.assertEqual(sorted(P.pool_map(pool, abs, [-1, 2])), [1, 2])
        self.assertFalse(os.path.exists(path))

    def test_the_data_file_is_removed_when_the_pool_is_terminated_too(self):
        ctx = get_context("spawn")
        with mock.patch.object(ctx, "Pool", wraps=ctx.Pool) as spy:
            with self.assertRaises(RuntimeError):
                with P.worker_pool(2, {"a": synthetic_ohlc(n_bars=300, seed=1)}, _cfg(), context=ctx):
                    path = spy.call_args.kwargs["initargs"][0]
                    raise RuntimeError("something downstream failed")
        self.assertFalse(os.path.exists(path))


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

        def fake(ticker, start, interval, drop_nonpositive=True):
            seen.update(ticker=ticker, start=start, interval=interval, drop_nonpositive=drop_nonpositive)
            return full

        orig = P.load_yfinance
        P.load_yfinance = fake
        try:
            during = P.load_real("SPY", "2024-01-01", "1d", now=idx[-1] + pd.Timedelta(hours=15))
            after = P.load_real("SPY", "2024-01-01", "1d", now=idx[-1] + pd.Timedelta(days=1, hours=1))
        finally:
            P.load_yfinance = orig
        self.assertEqual(seen, dict(ticker="SPY", start="2024-01-01", interval="1d", drop_nonpositive=True))
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
        self.assertEqual((a.vol_target, a.vol_target_n), (tpl.vol_target, tpl.vol_target_n))
        self.assertEqual(a.vol_target, 0.0, "the vol target is opt-in")

    def test_bad_choices_are_rejected(self):
        for argv in (["--family", "nope"], ["--interval", "2h"], ["--metric", "sortino"], ["--sides", "up"]):
            with self.assertRaises(SystemExit):
                M.parse_args(argv)

    def test_sizing_and_sides_reach_the_templates(self):
        cfg = _cfg(risk_pct=0.02, max_leverage=1.0, cost_bps=9.0, vol_target=0.12, vol_target_n=40)
        tpl = P._costed(generate_templates("quick", max_templates=1, sides=["long_only"])[0], cfg)
        self.assertEqual((tpl.risk_pct, tpl.max_leverage, tpl.cost_bps, tpl.sides), (0.02, 1.0, 9.0, "long_only"))
        self.assertEqual((tpl.vol_target, tpl.vol_target_n), (0.12, 40))

    def test_sizing_text_describes_the_rule_in_force(self):
        self.assertIn("1.0% of equity", P.sizing_text(dict(risk_pct=0.01)))              # a pre-feature spec
        self.assertIn("1.0% of equity", P.sizing_text(dict(risk_pct=0.01, vol_target=0.0)))
        self.assertIn("15% annualized vol target", P.sizing_text(dict(risk_pct=0.01, vol_target=0.15, vol_target_n=60)))

    def test_the_two_entry_points_evaluate_with_the_same_settings(self):
        """Every evaluation setting of one CLI exists, with the same default, in
        the other (portfolio weighting is a deliberate difference, not one of them)."""
        m = P.eval_config(M.parse_args([]), "1d")
        e = P.eval_config(ED.parse_args(["research"]), "1d")
        self.assertEqual(m, e)


class SharedCommandLineTests(unittest.TestCase):
    """main.py and `etf_dashboard.py research` build their research flags with
    the same function (pipeline.add_research_args), so a flag cannot mean one
    thing in one entry point and another in the other."""

    @staticmethod
    def _options(parser):
        return {a.dest: (a.default, tuple(a.choices) if a.choices else None)
                for a in parser._actions if a.option_strings and a.dest != "help"}

    def test_every_research_flag_is_shared_with_the_same_default(self):
        import argparse
        shared = self._options(P.add_research_args(argparse.ArgumentParser(add_help=False), start="x"))
        main = self._options(_parser_of(M.parse_args))
        research = self._options(_research_parser())
        for dest, spec in shared.items():
            if dest == "start":       # the one default the entry points choose differently
                continue
            self.assertEqual(main.get(dest), spec, dest)
            self.assertEqual(research.get(dest), spec, dest)

    def test_the_instrument_map_names_main_s_asset(self):
        a = M.parse_args(["--csv", "data/brent_z25z26.csv", "--instrument-map", "brent_z25z26=1000,15,3000,30"])
        with contextlib.redirect_stdout(io.StringIO()):
            imap = M.resolve_instrument(a, M.asset_label(a))
        self.assertEqual((a.point_value, a.cost_per_unit, a.margin_per_unit, a.roll_cost_per_unit), (1000.0, 15.0, 3000.0, 30.0))
        self.assertEqual(a.cost_bps, 0.0)                         # a margin: costed per unit only
        self.assertEqual(imap["brent_z25z26"]["margin_per_unit"], 3000.0)
        # a name that is not the run's asset is an error, as in the dashboard
        b = M.parse_args(["--real", "SPY", "--instrument-map", "QQQ=1"])
        with self.assertRaises(SystemExit):
            M.resolve_instrument(b, M.asset_label(b))
        # a Yahoo futures ticker carries its own '=': the map splits on the last one
        f = M.parse_args(["--real", "CL=F", "--instrument-map", "CL=F=1000,2.5,6000"])
        with contextlib.redirect_stdout(io.StringIO()):
            M.resolve_instrument(f, M.asset_label(f))
        self.assertEqual((f.point_value, f.margin_per_unit), (1000.0, 6000.0))
        for bad in (["SPY=nan"], ["SPY=inf"], ["SPY=1", "SPY=2"]):
            g = M.parse_args(["--real", "SPY", "--instrument-map", *bad])
            with self.assertRaises(SystemExit):
                M.resolve_instrument(g, "SPY")
        # mapped as a plain share: the run-wide futures flags do not leak in
        c = M.parse_args(["--real", "SPY", "--margin-per-unit", "5000", "--instrument-map", "SPY=1"])
        M.resolve_instrument(c, "SPY")
        self.assertEqual((c.point_value, c.margin_per_unit, c.cost_bps), (1.0, 0.0, 5.0))

    def test_a_margin_drops_the_bps_cost_in_both_entry_points(self):
        """The rule the dashboard applied to a mapped future, now the rule:
        a margined instrument is costed per unit, whatever --cost-bps says."""
        cfg = _cfg(point_value=1000.0, cost_per_unit=15.0, margin_per_unit=3000.0, cost_bps=5.0)
        tpl = P._costed(generate_templates("quick", max_templates=1)[0], cfg)
        self.assertEqual((tpl.cost_bps, tpl.cost_per_unit), (0.0, 15.0))
        a = M.parse_args(["--margin-per-unit", "3000", "--cost-per-unit", "15"])
        with contextlib.redirect_stdout(io.StringIO()) as out:
            M.resolve_instrument(a, "synthetic")
        self.assertEqual(a.cost_bps, 0.0)
        self.assertIn("--cost-bps 5 not charged", out.getvalue())
        share = P._costed(generate_templates("quick", max_templates=1)[0], _cfg(cost_bps=5.0))
        self.assertEqual(share.cost_bps, 5.0)                      # a cash asset keeps it

    def test_bars_per_day_sets_the_annualization(self):
        self.assertEqual(P.eval_config(M.parse_args(["--bars-per-day", "23"]), "1h")["periods_per_year"], 252 * 23)
        self.assertEqual(P.eval_config(ED.parse_args(["research", "--bars-per-day", "23"]), "1h")["periods_per_year"],
                         252 * 23)
        self.assertEqual(P.eval_config(M.parse_args([]), "1h")["periods_per_year"], 252 * 7)   # the default session
        for interval, bpd in (("1d", 1), ("1wk", 1), ("1h", 25), ("30m", 49), ("1h", 0)):
            with self.assertRaises(ValueError):
                S.periods_per_year_for_interval(interval, bpd)
        with self.assertRaises(SystemExit):
            _main(["--csv", "x.csv", "--interval", "1d", "--bars-per-day", "23"])


def _parser_of(parse_args):
    """The ArgumentParser behind an entry point's parse_args."""
    import argparse
    captured = {}
    real = argparse.ArgumentParser.parse_args

    def grab(self, args=None, namespace=None):
        captured["p"] = self
        return real(self, args, namespace)
    with mock.patch.object(argparse.ArgumentParser, "parse_args", grab):
        parse_args([])
    return captured["p"]


def _research_parser():
    import argparse
    p = _parser_of(lambda argv: ED.parse_args(["research"]))
    sub = next(a for a in p._actions if isinstance(a, argparse._SubParsersAction))
    return sub.choices["research"]


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


class InstrumentSettingsTests(unittest.TestCase):
    def test_defaults_are_the_engine_defaults(self):
        a = M.parse_args([])
        tpl = S.StrategyTemplate(name="x")
        self.assertEqual((a.point_value, a.cost_per_unit, a.margin_per_unit),
                         (tpl.point_value, tpl.cost_per_unit, tpl.margin_per_unit))
        self.assertIsNone(a.csv)

    def test_real_and_csv_are_exclusive(self):
        with self.assertRaises(SystemExit):
            M.parse_args(["--real", "SPY", "--csv", "x.csv"])

    def test_instrument_reaches_the_templates_and_old_configs_still_do(self):
        cfg = _cfg(point_value=1000.0, cost_per_unit=15.0, margin_per_unit=3000.0, cost_bps=0.0)
        tpl = P._costed(generate_templates("quick", max_templates=1)[0], cfg)
        self.assertEqual((tpl.point_value, tpl.cost_per_unit, tpl.margin_per_unit, tpl.cost_bps), (1000.0, 15.0, 3000.0, 0.0))
        old = dict(cost_bps=5.0, risk_pct=0.01, max_leverage=2.0, vol_target=0.0, vol_target_n=60)   # a pre-feature run.json
        tpl = P._costed(generate_templates("quick", max_templates=1)[0], old)
        self.assertEqual((tpl.point_value, tpl.cost_per_unit, tpl.margin_per_unit), (1.0, 0.0, 0.0))

    def test_sizing_text_names_a_non_default_instrument_only(self):
        self.assertNotIn("instrument", P.sizing_text(dict(risk_pct=0.01)))
        text = P.sizing_text(dict(risk_pct=0.01, point_value=1000.0, cost_per_unit=15.0, margin_per_unit=3000.0))
        self.assertIn("point value 1000", text)
        self.assertIn("margin 3000", text)

    def test_benchmark_kind_follows_the_prices(self):
        df = synthetic_ohlc(300, seed=2)
        kind, r = P.benchmark_returns(df)
        self.assertEqual(kind, P.BENCH_BUY_HOLD)
        pd.testing.assert_series_equal(r, df["Close"].pct_change())
        neg = df.assign(**{c: df[c] - 500.0 for c in ("Open", "High", "Low", "Close")})
        kind, r = P.benchmark_returns(neg, point_value=1000.0, initial_equity=100_000.0)
        self.assertEqual(kind, P.BENCH_ONE_UNIT)
        pd.testing.assert_series_equal(r, 1000.0 * neg["Close"].diff() / 100_000.0)
        self.assertTrue(np.isfinite(r.iloc[1:]).all())
        # a series that merely touches zero has no buy-and-hold return either
        touch = df.assign(**{c: df[c] - float(df["Low"].min()) for c in ("Open", "High", "Low", "Close")})
        self.assertEqual(P.benchmark_returns(touch)[0], P.BENCH_ONE_UNIT)


class SpreadRunTests(unittest.TestCase):
    """main.py end to end on a spread: a CSV of negative prices, a point
    value, per-unit costs and a margin, replayed and verified."""

    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp()
        df = synthetic_ohlc(1100, seed=7)
        shift = float(df["High"].max()) + 5.0
        cls.df = df.assign(**{c: df[c] - shift for c in ("Open", "High", "Low", "Close")})
        cls.csv = os.path.join(cls.dir, "brent_z25z26.csv")
        cls.df.to_csv(cls.csv, float_format="%.17g")
        cls.argv = ["--csv", cls.csv, "--point-value", "1000", "--margin-per-unit", "3000", "--cost-per-unit", "15",
                    "--cost-bps", "0", "--max-leverage", "0.5", "--family", "quick", "--max-templates", "4",
                    "--train", "300", "--test", "100", "--n-boot", "100", "--min-sharpe", "-5", "--jobs", "1",
                    "--no-matrix", "--replay", "best", "--out", cls.dir]
        cls.out = _main(cls.argv)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, ignore_errors=True)

    def test_the_run_completes_and_replays_exactly(self):
        check = self.out["replay"]["best"]["check"]
        self.assertTrue(check["ok"], check)
        trades = pd.read_csv(os.path.join(self.dir, "replay", "best", "trades.csv"))
        self.assertGreater(len(trades), 0)
        self.assertTrue((trades["entry_price"] < 0).all())
        self.assertTrue((trades["cost"] > 0).all())
        for name, res in self.out["results"].items():
            self.assertEqual((res["template"].point_value, res["template"].margin_per_unit,
                              res["template"].cost_per_unit, res["template"].cost_bps), (1000.0, 3000.0, 15.0, 0.0))

    def test_the_same_instrument_through_the_map_is_the_same_run(self):
        """--instrument-map on main.py: the CSV's stem names the asset, and the
        default --cost-bps 5 is not charged on a margined instrument -- the
        same numbers as the run given the flags with --cost-bps 0."""
        d = tempfile.mkdtemp()
        try:
            argv = ["--csv", self.csv, "--instrument-map", "brent_z25z26=1000,15,3000", "--max-leverage", "0.5",
                    "--family", "quick", "--max-templates", "4", "--train", "300", "--test", "100",
                    "--n-boot", "100", "--min-sharpe", "-5", "--jobs", "1", "--no-matrix", "--out", d]
            other = _main(argv)
            for name, res in self.out["results"].items():
                pd.testing.assert_series_equal(other["results"][name]["oos_returns"], res["oos_returns"])
                self.assertEqual(other["results"][name]["template"], res["template"])
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_the_manifest_records_the_instrument_and_the_file(self):
        with open(os.path.join(self.dir, "run.json"), encoding="utf-8") as f:
            m = json.load(f)
        self.assertEqual((m["config"]["point_value"], m["config"]["margin_per_unit"], m["config"]["cost_per_unit"]),
                         (1000.0, 3000.0, 15.0))
        self.assertEqual((m["data"]["source"], m["data"]["ticker"]), ("csv", "brent_z25z26"))
        self.assertEqual(m["data"]["path"], self.csv)

    def test_the_report_names_the_instrument_and_the_one_unit_benchmark(self):
        text = open(os.path.join(self.dir, "report.md"), encoding="utf-8").read()
        self.assertIn("## Benchmark: hold 1 unit brent_z25z26", text)
        self.assertIn("point value 1000", text)
        self.assertIn("csv " + self.csv, text)
        self.assertEqual(self.out["benchmark"]["kind"], P.BENCH_ONE_UNIT)
        self.assertTrue(np.isfinite(self.out["benchmark"]["buy_hold"]["sharpe"]))

    def test_a_spread_without_a_margin_fails_before_the_pool(self):
        argv = [a for a in self.argv if a not in ("--margin-per-unit", "3000")]
        with self.assertRaises(ValueError) as cm:
            _main(argv + ["--out", tempfile.mkdtemp()])
        self.assertIn("margin_per_unit", str(cm.exception))


class AdditiveBenchmarkTests(unittest.TestCase):
    def test_additive_returns_are_summed_not_compounded(self):
        r = pd.Series([0.0, 0.5, -0.5, 0.2], index=pd.bdate_range("2024-01-01", periods=4))
        pd.testing.assert_series_equal(P.benchmark_curve(r, additive=True), 1 + r.cumsum())
        pd.testing.assert_series_equal(P.benchmark_curve(r, additive=False), (1 + r).cumprod())
        a, m = P.curve_stats(r, additive=True), P.curve_stats(r, additive=False)
        self.assertEqual(a["sharpe"], m["sharpe"])                # scale-free either way
        # 1.5 -> 1.0 on the summed curve: half the initial equity, not a third of the peak
        self.assertAlmostEqual(a["max_dd"], -0.5)
        self.assertAlmostEqual(a["cagr"], 0.2 * P.periods_per_year() / 4)   # simple annual P&L
        self.assertNotEqual(a["cagr"], m["cagr"])
        # a one-unit stream with a bar losing more than the initial equity stays a curve,
        # and its drawdown is that loss over the initial equity, not a ratio to a peak
        # the curve has crossed zero from
        r = pd.Series([0.0, -1.5, 0.2, 0.2], index=r.index)
        s = P.curve_stats(r, additive=True)
        self.assertAlmostEqual(s["max_dd"], -1.5)
        self.assertAlmostEqual(s["cagr"], -1.1 * P.periods_per_year() / 4)
        self.assertAlmostEqual(P.benchmark_curve(r, additive=True).iloc[-1], -0.1)

    def test_benchmark_stats_carry_the_additive_flag(self):
        idx = pd.bdate_range("2020-01-01", periods=300)
        rng = np.random.default_rng(1)
        bh = pd.Series(rng.normal(0, 0.01, 300), index=idx)
        rets = pd.DataFrame({"a": bh * 0.5}, index=idx)
        port = dict(portfolio_returns=rets["a"])
        b = P.benchmark_stats(bh, rets, port, port, kind=P.BENCH_ONE_UNIT)
        self.assertTrue(b["additive"])
        self.assertEqual(b["buy_hold"], P.curve_stats(bh, additive=True))
        self.assertFalse(P.benchmark_stats(bh, rets, port, port)["additive"])

    def test_whole_units_and_the_instrument_map_reach_the_templates(self):
        cfg = _cfg(whole_units=True, point_value=1.0)
        cfg["instrument_map"] = {"brent": dict(point_value=1000.0, cost_per_unit=15.0, margin_per_unit=3000.0)}
        base = generate_templates("quick", max_templates=1)[0]
        spy, brent = P._costed(base, cfg, "SPY"), P._costed(base, cfg, "brent")
        self.assertEqual((spy.point_value, spy.cost_bps, spy.whole_units), (1.0, 5.0, True))
        self.assertEqual((brent.point_value, brent.cost_per_unit, brent.margin_per_unit, brent.cost_bps, brent.whole_units),
                         (1000.0, 15.0, 3000.0, 0.0, True))
        self.assertEqual(P._costed(base, cfg).point_value, 1.0)      # no asset: the run-wide settings
        self.assertIn("whole units", P.sizing_text(cfg))
        self.assertFalse(M.parse_args([]).whole_units)

    def test_a_mapped_asset_does_not_inherit_the_run_wide_future(self):
        """SPY=1 in a Brent book is a plain share: no Brent margin, no per-lot cost,
        the run's bps costs (a margin only is what zeroes them)."""
        cfg = _cfg(point_value=1000.0, cost_per_unit=10.0, margin_per_unit=5000.0, cost_bps=5.0)
        cfg["instrument_map"] = ED._parse_instrument_map(["SPY=1"], ["SPY", "brent"])
        base = generate_templates("quick", max_templates=1)[0]
        spy, brent = P._costed(base, cfg, "SPY"), P._costed(base, cfg, "brent")
        self.assertEqual((spy.point_value, spy.cost_per_unit, spy.margin_per_unit, spy.cost_bps), (1.0, 0.0, 0.0, 5.0))
        self.assertEqual((brent.point_value, brent.cost_per_unit, brent.margin_per_unit), (1000.0, 10.0, 5000.0))
        # a spec written before the map was filled out (point value only) reads the same way
        cfg["instrument_map"] = {"SPY": dict(point_value=1.0)}
        self.assertEqual(P._costed(base, cfg, "SPY").margin_per_unit, 0.0)

    def test_a_margined_future_is_benchmarked_by_one_unit(self):
        """A back-adjusted Brent series is positive, but its percentage change is
        not the contract's return: with a margin the benchmark is one lot's P&L."""
        df = synthetic_ohlc(300, seed=3)
        self.assertEqual(P.benchmark_returns(df)[0], P.BENCH_BUY_HOLD)
        kind, r = P.benchmark_returns(df, point_value=1000.0, margin_per_unit=6000.0)
        self.assertEqual(kind, P.BENCH_ONE_UNIT)
        pd.testing.assert_series_equal(r, 1000.0 * df["Close"].diff() / 100_000.0)

    def test_flat_templates_do_not_beat_a_losing_benchmark(self):
        idx = pd.bdate_range("2020-01-01", periods=300)
        bh = pd.Series(np.random.default_rng(2).normal(-0.001, 0.01, 300), index=idx)
        rets = pd.DataFrame({"flat": 0.0, "good": bh * -0.5}, index=idx)
        b = P.benchmark_stats(bh, rets, dict(portfolio_returns=rets["good"]), dict(portfolio_returns=rets["good"]))
        self.assertLess(b["buy_hold"]["sharpe"], 0.0)
        self.assertEqual(b["n_templates_beat_bh"], 1)

    def test_the_dashboard_holding_curve_compounds_shares_in_a_mixed_book(self):
        """One additive asset must not turn the share's compounding into a sum."""
        idx = pd.bdate_range("2020-01-01", periods=400)
        up = pd.DataFrame({c: 100.0 * 1.01 ** np.arange(400) for c in ("Open", "High", "Low", "Close")}, index=idx)
        spr = pd.DataFrame({c: np.linspace(-1.0, 1.0, 400) for c in ("Open", "High", "Low", "Close")}, index=idx)
        cfg = dict(instrument_map={"spr": dict(point_value=1000.0, cost_per_unit=10.0, margin_per_unit=3000.0)})
        nested = dict(portfolio_equity=pd.Series(1.0, index=idx))
        c = ED._curves({"SPY": up, "spr": spr}, None, nested, None, cfg)
        share = 1.01 ** 399
        spread = 1.0 + 1000.0 * 2.0 / 100_000.0
        self.assertAlmostEqual(c["buy_hold"][-1], round((share + spread) / 2, 5), places=4)


class TypicalUnitsTests(unittest.TestCase):
    def test_the_size_a_median_bar_gives(self):
        df = synthetic_ohlc(600, seed=5)
        tpl = S.StrategyTemplate("t")
        a = float(np.nanmedian(S.atr(df, tpl.atr_n)))
        self.assertAlmostEqual(S.typical_units(df, tpl, 100_000.0), 100_000 * tpl.risk_pct / (tpl.atr_mult_stop * a))
        # a Brent lot on the same moves: a thousandth of it, and the margin cap binds on top
        lot = tpl.with_params(point_value=1000.0, margin_per_unit=6000.0, max_leverage=0.5)
        self.assertLess(S.typical_units(df, lot, 100_000.0), 1.0)
        self.assertAlmostEqual(S.typical_units(df, tpl.with_params(max_leverage=0.01), 100_000.0),
                               0.01 * 100_000 / float(np.nanmedian(df["Close"])))


class PortfolioRuinReplayTests(unittest.TestCase):
    """A slot selected at full weight goes to a deficit: the nested book is
    closed at that bar (portfolio.close_after_ruin), the selection log stops
    picking, and the replay reproduces the closed series exactly."""

    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp()
        df = synthetic_ohlc(1500, seed=7)
        cols = [df.columns.get_loc(c) for c in ("Open", "High", "Low", "Close")]
        df.iloc[1030:, cols] *= 0.45                           # a 55 % gap, beyond any margin
        cls.csv = os.path.join(cls.dir, "crash.csv")
        df.to_csv(cls.csv, float_format="%.17g")
        cls.out = _main(["--csv", cls.csv, "--family", "quick", "--max-templates", "12", "--sides", "long_only",
                         "--train", "400", "--test", "100", "--n-boot", "50", "--jobs", "1", "--no-matrix",
                         "--max-leverage", "10", "--risk-pct", "0.3", "--min-sharpe", "-2", "--replay", "all",
                         "--out", cls.dir])

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, ignore_errors=True)

    def test_the_book_is_closed_and_the_replay_matches(self):
        nested = self.out["nested"]
        eq = nested["portfolio_equity"].to_numpy()
        dead = np.flatnonzero(eq <= 0.0)
        self.assertGreater(len(dead), 0, "the fixture never ruins the nested book")
        self.assertTrue((nested["portfolio_returns"].to_numpy()[dead[0] + 1:] == 0.0).all())
        later = [s for s in nested["selections"] if s["period_start"] > nested["portfolio_equity"].index[dead[0]]]
        self.assertGreater(len(later), 0)
        self.assertTrue(all(s["selected"] == [] and s.get("ruined") for s in later))
        for which in ("best", "static", "nested"):
            check = self.out["replay"][which]["check"]
            self.assertTrue(check["ok"], (which, check))
