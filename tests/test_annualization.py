"""
The annualization read off the data's timestamps (strategy.infer_annualization,
pipeline.annualization_for).

An interval string and a bars-per-day count cannot describe bars sampled on
activity (volume, tick bars: a different count every day), nor a session the
table does not know (a ~23 h future, 24/7 crypto). These tests pin that

  * the bars a year and the sessions a year come out of the stamps, for
    daily, weekly, monthly, crypto, equity-hourly, futures-hourly (with a
    daylight-saving break) and volume bars;
  * on intraday bars the Sharpe ratio is the one of the returns compounded
    per session: the same daily returns cut into 3 or 50 bars a day, or
    into a different number every day, give the same Sharpe;
  * daily data is measured exactly as before (per bar), and so is a config
    written before sessions existed;
  * the in-sample Sharpe of a backtest (the optimizer's metric) is measured
    like the out-of-sample one, and the vol target sizes on the data's bars
    a year;
  * main.py runs end to end on a CSV of volume bars with no --interval.

Run with:  python -m pytest tests/test_annualization.py
"""

from __future__ import annotations
import contextlib
import io
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import strategy as S  # noqa: E402
import pipeline as P  # noqa: E402
import main as M  # noqa: E402
from data import synthetic_ohlc  # noqa: E402


# --------------------------------------------------------------------------
# fixtures: stamps of the kinds of bars the project meets (naive UTC intraday,
# as data._naive_index leaves them)
# --------------------------------------------------------------------------

def _us_dst(t: pd.Timestamp) -> bool:
    """US daylight saving: second Sunday of March to first Sunday of November."""
    mar = pd.Timestamp(year=t.year, month=3, day=1)
    nov = pd.Timestamp(year=t.year, month=11, day=1)
    start = mar + pd.Timedelta(days=(6 - mar.weekday()) % 7 + 7)
    end = nov + pd.Timedelta(days=(6 - nov.weekday()) % 7)
    return start <= t < end


def equity_hourly(start="2021-01-01", end="2023-12-31") -> pd.DatetimeIndex:
    """US-equity '1h' bars on a UTC clock: 09:30-16:00 New York, seven a day."""
    out = []
    for d in pd.bdate_range(start, end):
        open_utc = d + pd.Timedelta(hours=13 if _us_dst(d) else 14, minutes=30)
        out += [open_utc + pd.Timedelta(hours=h) for h in range(7)]
    return pd.DatetimeIndex(out)


def futures_hourly(start="2021-01-03", end="2023-12-29") -> pd.DatetimeIndex:
    """'1h' bars of a CME-style future on a UTC clock: Sunday 18:00 to Friday
    17:00 New York, with a daily 17:00-18:00 break that moves on UTC with
    daylight saving."""
    out = []
    for t in pd.date_range(start, end, freq="h"):
        et = t - pd.Timedelta(hours=4 if _us_dst(t) else 5)
        wd = et.weekday()
        if wd == 5 or (wd == 4 and et.hour >= 17) or (wd == 6 and et.hour < 18) or et.hour == 17:
            continue
        out.append(t)
    return pd.DatetimeIndex(out)


def volume_bars(start="2021-01-01", end="2023-12-31", seed=0, lo=3, hi=60):
    """Bars at irregular stamps inside the equity session, a random number of
    them each day (what a volume or tick sampler leaves). Returns the index
    and the number of bars on each day."""
    rng = np.random.default_rng(seed)
    out, counts = [], []
    for d in pd.bdate_range(start, end):
        k = int(rng.integers(lo, hi))
        open_utc = d + pd.Timedelta(hours=13 if _us_dst(d) else 14, minutes=30)
        mins = np.sort(rng.choice(390 * 60, size=k, replace=False)) / 60.0
        out += list(open_utc + pd.to_timedelta(mins, unit="min"))
        counts.append(k)
    return pd.DatetimeIndex(out), np.array(counts)


def split_daily(daily: np.ndarray, counts: np.ndarray, rng) -> np.ndarray:
    """Per-bar simple returns that compound to `daily` on each day, cut into
    `counts[d]` bars of random sizes."""
    out = []
    for r, k in zip(daily, counts):
        w = rng.dirichlet(np.ones(k))
        out.append(np.expm1(np.log1p(r) * w))
    return np.concatenate(out)


class _KeepAnnualization(unittest.TestCase):
    """Restores the process-wide annualization each test changes."""

    def setUp(self):
        self._saved = dict(periods_per_year=S.periods_per_year(), sessions_per_year=S.sessions_per_year(),
                           session_cut_minutes=S.session_cut_minutes())

    def tearDown(self):
        S.set_annualization(self._saved)


# --------------------------------------------------------------------------
# inference
# --------------------------------------------------------------------------

class InferenceTests(unittest.TestCase):

    def test_daily_and_coarser_bars_are_counted_and_stay_per_bar(self):
        cases = {
            "spy": (pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "data_dump", "spy.csv"),
                                index_col=0, parse_dates=True).index, 251, 253),
            "crypto daily": (pd.date_range("2018-01-01", "2021-06-30", freq="D"), 365, 365),
            "business days": (pd.bdate_range("2015-01-01", "2020-12-31"), 261, 261),
            "weekly": (pd.date_range("2010-01-01", "2020-01-01", freq="W-FRI"), 52, 53),
            "monthly": (pd.date_range("2000-01-01", "2020-01-01", freq="ME"), 12, 12),
        }
        for name, (idx, lo, hi) in cases.items():
            with self.subTest(name):
                a = S.infer_annualization(idx)
                self.assertTrue(lo <= a["periods_per_year"] <= hi, a)
                self.assertIsNone(a["sessions_per_year"], "daily bars are their own sessions")

    def test_equity_hourly_bars(self):
        a = S.infer_annualization(equity_hourly())
        self.assertAlmostEqual(a["periods_per_year"] / 7, 261, delta=2)
        self.assertEqual(a["sessions_per_year"], 261)
        # the cut falls in the overnight gap (20:00-13:30 UTC), never inside a session
        self.assertTrue(a["session_cut_minutes"] >= 21 * 60 or a["session_cut_minutes"] <= 13 * 60, a)

    def test_a_23h_future_with_a_moving_break_keeps_one_session_a_day(self):
        idx = futures_hourly()
        a = S.infer_annualization(idx)
        self.assertAlmostEqual(a["periods_per_year"] / 23, 261, delta=3)    # ~23 bars on each weekday
        self.assertEqual(a["sessions_per_year"], 261, "Sunday evening belongs to Monday's session")
        labels = S.session_labels(idx, a["session_cut_minutes"])
        per = pd.Series(1, index=labels).groupby(level=0).size()
        # every whole session holds its 23 bars (the fixture's last day is cut short)
        self.assertTrue(set(per.iloc[1:-1].unique()) <= {23}, per.value_counts())

    def test_24_7_bars_are_cut_at_midnight(self):
        a = S.infer_annualization(pd.date_range("2021-01-01", "2023-01-01", freq="h"))
        self.assertEqual(a["periods_per_year"], 8766)
        self.assertEqual(a["sessions_per_year"], 365)
        self.assertEqual(a["session_cut_minutes"], 0)

    def test_volume_bars(self):
        idx, counts = volume_bars()
        a = S.infer_annualization(idx)
        self.assertEqual(a["sessions_per_year"], 261)
        self.assertAlmostEqual(a["periods_per_year"], counts.mean() * 261, delta=0.02 * counts.mean() * 261)

    def test_a_hole_in_the_data_does_not_shrink_the_count(self):
        idx = pd.bdate_range("2010-01-01", "2020-12-31")
        holed = idx[(idx < "2014-01-01") | (idx >= "2015-06-01")]           # 17 months missing
        self.assertAlmostEqual(S.infer_annualization(holed)["periods_per_year"],
                               S.infer_annualization(idx)["periods_per_year"], delta=1)
        span_years = (holed[-1] - holed[0]).days / 365.25
        self.assertLess(len(holed) / span_years, 230, "the bars over the span would be fooled by the hole")

    def test_too_little_to_measure(self):
        self.assertIsNone(S.infer_annualization(pd.DatetimeIndex(["2020-01-01", "2020-01-02"])))
        self.assertIsNone(S.infer_annualization(pd.RangeIndex(100)))


# --------------------------------------------------------------------------
# the Sharpe ratio no longer depends on how a day was cut
# --------------------------------------------------------------------------

class SessionSharpeTests(_KeepAnnualization):

    def test_volume_bars_give_the_daily_sharpe(self):
        rng = np.random.default_rng(1)
        idx, counts = volume_bars(seed=1)
        daily = rng.normal(0.0004, 0.01, len(counts))
        r = pd.Series(split_daily(daily, counts, rng), index=idx)
        S.set_annualization(S.infer_annualization(idx))
        want = daily.mean() / daily.std(ddof=1) * np.sqrt(261)
        self.assertAlmostEqual(S.annualized_sharpe(r), want, places=10)
        self.assertAlmostEqual(S.annualized_vol(r), daily.std(ddof=1) * np.sqrt(261), places=10)
        # the per-bar formula would be off: bars of different sizes are no i.i.d. sample
        self.assertNotAlmostEqual(S.annualized_sharpe(r.to_numpy()), want, places=2)

    def test_the_same_days_cut_finer_or_coarser_give_the_same_sharpe(self):
        rng = np.random.default_rng(2)
        daily = rng.normal(0.0003, 0.012, 780)
        sharpes = []
        for lo, hi in [(3, 4), (20, 21), (5, 80)]:
            idx, counts = volume_bars(seed=lo, lo=lo, hi=hi)
            idx, counts = idx[:counts[:780].sum()], counts[:780]
            r = pd.Series(split_daily(daily, counts, rng), index=idx)
            S.set_annualization(S.infer_annualization(idx))
            sharpes.append(S.annualized_sharpe(r))
        self.assertAlmostEqual(max(sharpes), min(sharpes), places=10)

    def test_daily_data_is_measured_per_bar_as_before(self):
        idx = pd.bdate_range("2015-01-01", periods=1500)
        r = pd.Series(np.random.default_rng(3).normal(0.0005, 0.01, len(idx)), index=idx)
        S.set_annualization(dict(periods_per_year=252))      # a config from before sessions existed
        self.assertIsNone(S.sessions_per_year())
        x = r.to_numpy()
        self.assertEqual(S.annualized_sharpe(r), float(x.mean() / x.std(ddof=1) * np.sqrt(252)))
        self.assertEqual(S.annualized_sharpe(r), S.annualized_sharpe(x))

    def test_nan_bars_and_an_explicit_ppy(self):
        idx, counts = volume_bars(seed=4)
        r = pd.Series(np.random.default_rng(4).normal(0, 0.002, len(idx)), index=idx)
        S.set_annualization(S.infer_annualization(idx))
        with_nan = r.copy()
        with_nan.iloc[:counts[0]] = np.nan                     # the first day not traded yet
        self.assertAlmostEqual(S.annualized_sharpe(with_nan), S.annualized_sharpe(r.iloc[counts[0]:]), places=12)
        x = r.to_numpy()
        self.assertAlmostEqual(S.annualized_sharpe(r, ppy=1000), x.mean() / x.std(ddof=1) * np.sqrt(1000))

    def test_in_sample_and_out_of_sample_sharpe_are_measured_alike(self):
        """performance_stats (the optimizer's in-sample metric) must be the
        estimator annualized_sharpe applies to an OOS series, or the OOS/IS
        Sharpe ratio compares two different things."""
        idx = equity_hourly("2021-01-01", "2022-06-30")
        base = synthetic_ohlc(n_bars=len(idx), seed=5)
        df = base.set_axis(idx)
        S.set_annualization(S.infer_annualization(idx))
        tpl = S.StrategyTemplate("t", n_entry=40)
        res = S.backtest(df, tpl)
        self.assertGreater(len(res["trades"]), 0)
        self.assertAlmostEqual(res["stats"]["sharpe"], S.annualized_sharpe(res["returns"].iloc[1:]), places=12)
        per_bar = S.annualized_sharpe(res["returns"].iloc[1:].to_numpy())
        self.assertNotAlmostEqual(res["stats"]["sharpe"], per_bar, places=6)


# --------------------------------------------------------------------------
# the run's configuration
# --------------------------------------------------------------------------

class ConfigTests(_KeepAnnualization):

    def test_eval_config_reads_the_data(self):
        idx = futures_hourly()
        args = M.parse_args([])
        c = P.eval_config(args, "1h", index=idx)
        self.assertEqual(c["annualization"], "data")
        self.assertEqual(c["sessions_per_year"], 261)
        self.assertAlmostEqual(c["periods_per_year"] / 23, 261, delta=3)   # not the equity table's 7 a day
        # --bars-per-day still sets the bars a year; the session clock stays the data's
        c = P.eval_config(M.parse_args(["--bars-per-day", "23"]), "1h", index=idx)
        self.assertEqual(c["periods_per_year"], 252 * 23)
        self.assertEqual(c["sessions_per_year"], 261)
        # the table, per bar, on request, without an index, and for an index too short
        for c in (P.eval_config(M.parse_args(["--annualization", "interval"]), "1h", index=idx),
                  P.eval_config(args, "1h"), P.eval_config(args, "1h", index=idx[:2])):
            self.assertEqual((c["periods_per_year"], c["sessions_per_year"]), (252 * 7, None))
        self.assertIn("session-compounded", P.annualization_text(P.eval_config(args, "1h", index=idx)))

    def test_workers_get_the_session_clock(self):
        c = P.eval_config(M.parse_args([]), "1h", index=equity_hourly())
        S.set_annualization(dict(periods_per_year=252))
        P.init_worker({}, c)
        self.assertEqual((S.periods_per_year(), S.sessions_per_year(), S.session_cut_minutes()),
                         (c["periods_per_year"], c["sessions_per_year"], c["session_cut_minutes"]))

    def test_the_vol_target_sizes_on_the_datas_bars_a_year(self):
        """A per-bar vol target is the annual one over sqrt(bars a year): on
        23 h futures bars the table's 7 a day would size every entry
        sqrt(23/7) ~ 1.8 times too big."""
        idx = futures_hourly("2021-01-03", "2021-12-31")
        df = synthetic_ohlc(n_bars=len(idx), seed=6, trend_vol=0.002, range_vol=0.0015).set_axis(idx)
        tpl = S.StrategyTemplate("t", n_entry=40, vol_target=0.15, max_leverage=1000.0)
        sizes = {}
        for name, ppy in (("table", 252 * 7), ("data", S.infer_annualization(idx)["periods_per_year"])):
            S.set_annualization(dict(periods_per_year=ppy))
            t = S.backtest(df, tpl, initial_equity=1e8)["trades"][0]
            sizes[name] = t["shares"] * t["entry_price"]
        self.assertAlmostEqual(sizes["table"] / sizes["data"], np.sqrt(S.infer_annualization(idx)
                                                                        ["periods_per_year"] / (252 * 7)), delta=0.02)


# --------------------------------------------------------------------------
# end to end: a CSV of volume bars, no --interval
# --------------------------------------------------------------------------

class VolumeBarsEndToEndTests(_KeepAnnualization):

    @classmethod
    def setUpClass(cls):
        # main.main sets the process-wide annualization: put it back after
        cls._saved_cls = dict(periods_per_year=S.periods_per_year(), sessions_per_year=S.sessions_per_year(),
                              session_cut_minutes=S.session_cut_minutes())
        cls.dir = tempfile.mkdtemp()
        idx, _ = volume_bars("2023-01-01", "2023-06-30", seed=7, lo=10, hi=30)
        df = synthetic_ohlc(n_bars=len(idx), seed=7, trend_vol=0.002, range_vol=0.0012).set_axis(idx)
        df.index.name = "Datetime"
        cls.csv = os.path.join(cls.dir, "vbars.csv")
        df.to_csv(cls.csv)
        cls.expected = S.infer_annualization(idx)
        cls.log = io.StringIO()
        with contextlib.redirect_stdout(cls.log):
            cls.out = M.main(["--csv", cls.csv, "--family", "quick", "--max-templates", "3", "--train", "600",
                              "--test", "200", "--n-boot", "50", "--jobs", "1", "--no-matrix",
                              "--min-sharpe", "-5", "--out", cls.dir])

    @classmethod
    def tearDownClass(cls):
        S.set_annualization(cls._saved_cls)
        shutil.rmtree(cls.dir, ignore_errors=True)

    def test_the_run_annualizes_on_the_data(self):
        import json
        with open(os.path.join(self.dir, "run.json")) as f:
            cfg = json.load(f)["config"]
        for k in ("periods_per_year", "sessions_per_year", "session_cut_minutes"):
            self.assertEqual(cfg[k], self.expected[k], k)
        self.assertEqual(cfg["annualization"], "data")
        self.assertIn("session-compounded", self.log.getvalue())
        # the reported OOS Sharpe of every template is its session-compounded one
        S.set_annualization(cfg)
        for name, res in self.out["results"].items():
            r = res["oos_returns"]
            if len(r) > 2:
                self.assertAlmostEqual(res["summary"]["oos_sharpe"], S.annualized_sharpe(r), places=10)
                self.assertAlmostEqual(res["summary"]["oos_sharpe"],
                                       S.annualized_sharpe(S.session_returns(r), ppy=cfg["sessions_per_year"]),
                                       places=10)


if __name__ == "__main__":
    unittest.main()
