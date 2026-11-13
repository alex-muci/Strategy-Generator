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
        self._saved = S.current_annualization()

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

    def test_24_7_bars_are_cut_at_midnight_whatever_the_ends_and_holes(self):
        h = pd.date_range
        outage = h("2022-01-01 10:00", "2023-12-30 15:00", freq="h")
        for a, b in (("2022-03-10 05:00", "2022-03-12 18:00"), ("2022-09-01 20:00", "2022-09-05 02:00")):
            outage = outage[(outage < a) | (outage > b)]
        cases = {
            "starts 10:00, ends 15:00": h("2022-01-01 10:00", "2023-12-30 15:00", freq="h"),
            "multi-day outages": outage,
            "a month of 15m bars": h("2022-01-01 07:30", "2022-01-31 12:00", freq="15min"),
            "ten days of 1m bars": h("2022-01-01 13:17", "2022-01-11 09:00", freq="min"),
            "tz-aware 4h bars": h("2022-01-01 02:00", "2023-06-30", freq="4h", tz="UTC"),
        }
        for name, idx in cases.items():
            with self.subTest(name):
                self.assertEqual(S._session_cut(idx), 0)

    def test_24_7_volume_bars_are_cut_at_midnight(self):
        """Activity-sampled bars of a 24/7 market: irregular steps, quieter
        hours, busier days. No hour is a break, so the cut stays at midnight
        rather than wandering with the noise of the steps."""
        rng = np.random.default_rng(3)
        for season in (0.0, 0.5, 0.8):
            stamps = []
            for d in pd.date_range("2022-01-01", periods=730, freq="D"):
                busy = np.exp(rng.normal(0, 0.5))
                for h in range(24):
                    k = rng.poisson(60 / 24 * busy * (1 + season * np.cos(2 * np.pi * (h - 15) / 24)))
                    stamps += list(d + pd.Timedelta(hours=h) + pd.to_timedelta(np.sort(rng.uniform(0, 60, k)), unit="min"))
            with self.subTest(season=season):
                self.assertEqual(S._session_cut(pd.DatetimeIndex(stamps)), 0)

    def test_a_daily_break_on_seven_day_data_is_found(self):
        idx = pd.date_range("2022-01-01", "2023-12-31", freq="15min")
        idx = idx[idx.hour != 3]                                  # an hour's maintenance at 03:00
        cut = S._session_cut(idx)
        self.assertTrue(3 * 60 <= cut < 4 * 60, cut)

    def test_fx_keeps_the_sunday_evening_open(self):
        def fx(start, end):
            s = pd.date_range(start, end, freq="h").to_series()
            shut = (((s.dt.dayofweek == 4) & (s.dt.hour >= 22)) | (s.dt.dayofweek == 5)
                    | ((s.dt.dayofweek == 6) & (s.dt.hour < 22)))
            return pd.DatetimeIndex(s[~shut].index)
        for name, (a, b) in {"two years": ("2022-01-02 22:00", "2023-12-29 21:00"),
                             "two weeks": ("2022-01-02 22:00", "2022-01-14 21:00")}.items():
            with self.subTest(name):
                self.assertEqual(S._session_cut(fx(a, b)), 22 * 60, "Sunday evening opens Monday's session")
        # a single week cannot tell its one Sunday evening from the ends of the data
        self.assertEqual(S._session_cut(fx("2022-01-02 22:00", "2022-01-07 21:00")), 0)

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

    def test_activity_sampled_bars_are_told_from_clock_bars(self):
        idx, _ = volume_bars("2023-01-01", "2023-06-30")
        clock = pd.date_range("2022-01-01", "2022-03-01", freq="min")
        clock = clock.delete(np.random.default_rng(2).choice(len(clock), 2000, replace=False))   # missing bars
        for name, i, irregular in (("volume bars", idx, True), ("equity hourly", equity_hourly(), False),
                                   ("futures hourly", futures_hourly(), False),
                                   ("24/7 hourly", pd.date_range("2021-01-01", "2022-01-01", freq="h"), False),
                                   ("1m with holes", clock, False),
                                   ("business days", pd.bdate_range("2015-01-01", "2020-12-31"), False),
                                   ("too short", idx[:2], False)):
            with self.subTest(name):
                self.assertIs(S.irregular_bars(i), irregular)

    def test_activity_sampled_bars_take_no_bars_per_day_nor_the_interval_table(self):
        idx, _ = volume_bars("2023-01-01", "2023-06-30")
        with self.assertRaisesRegex(ValueError, "--bars-per-day"):
            P.annualization_for("1h", 20, idx)
        with self.assertRaisesRegex(ValueError, "--annualization data"):
            P.annualization_for("1h", None, idx, mode="interval")
        self.assertEqual(P.annualization_for("1d", None, idx)["sessions_per_year"],
                         S.infer_annualization(idx)["sessions_per_year"])
        # clock bars keep both
        self.assertEqual(P.annualization_for("1h", 23, futures_hourly())["periods_per_year"], 252 * 23)
        self.assertEqual(P.annualization_for("1h", None, equity_hourly(), mode="interval")["periods_per_year"], 252 * 7)

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
        cls._saved_cls = S.current_annualization()
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

    def test_no_table_mismatch_note_for_activity_sampled_bars(self):
        self.assertNotIn("bars a year, not the", self.log.getvalue())

    def test_a_tz_aware_csv_of_volume_bars_is_put_on_the_utc_clock(self):
        from data import load_csv
        idx, _ = volume_bars("2023-01-02", "2023-03-01", seed=3, lo=10, hi=30)   # one UTC offset: no DST change
        idx = idx.floor("s").unique()                          # one stamp format in the file
        df = synthetic_ohlc(n_bars=len(idx), seed=3).set_axis(idx)
        df.index = df.index.tz_localize("UTC").tz_convert("America/New_York")
        df.index.name = "Datetime"
        path = os.path.join(self.dir, "vbars_ny.csv")
        df.to_csv(path)
        out = load_csv(path)                                   # the default interval, '1d'
        self.assertIsNone(out.index.tz)
        self.assertTrue(out.index.equals(idx), "stamps on naive UTC, not New York wall time")

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

    def test_the_robustness_statistics_count_sessions_and_calendar_years(self):
        from data import load_csv
        idx = load_csv(self.csv).index
        S.set_annualization(S.infer_annualization(idx))
        n_sessions = len(S.session_starts(idx))
        for name, res in self.out["results"].items():
            blocks = res.get("trial_blocks")
            if blocks is not None:
                self.assertEqual(int(blocks["counts"].sum()), n_sessions, name)    # CSCV rows are sessions
                self.assertEqual(blocks["per_year"], self.expected["sessions_per_year"])
        rets = self.out["returns"]
        span = (rets.index[-1] - rets.index[0]).total_seconds() / (86400 * 365.25)
        self.assertAlmostEqual(self.out["family"]["years_available"], span * len(rets) / (len(rets) - 1), places=9)
        self.assertEqual(self.out["family"]["dsr_best"]["T"], len(S.session_starts(rets.index)))


# --------------------------------------------------------------------------
# the walk-forward's in-sample windows are measured like its OOS series
# --------------------------------------------------------------------------

class WindowInSampleTests(_KeepAnnualization):
    """walkforward.window_backtest is how the optimizer scores every grid
    point of every window: its Sharpe must be the session-compounded one
    (as the OOS series'), or the IS ranking and the OOS/IS ratio would still
    depend on how a day was cut into bars."""

    def test_window_sharpe_is_session_compounded(self):
        from walkforward import window_backtest, warmup_bars
        idx = equity_hourly("2021-01-01", "2022-06-30")
        df = synthetic_ohlc(n_bars=len(idx), seed=5).set_axis(idx)
        S.set_annualization(S.infer_annualization(idx))
        tpl = S.StrategyTemplate("t", n_entry=40)
        start = warmup_bars(tpl) + 50
        for st, end in ((start, start + 900), (0, 900)):          # a warmed-up window, and one from bar 0
            res = window_backtest(df, tpl, st, end)
            r = res["returns"] if st > 0 else res["returns"].iloc[1:]
            self.assertGreater(len(res["trades"]), 0)
            self.assertAlmostEqual(res["stats"]["sharpe"], S.annualized_sharpe(r), places=12)
            self.assertNotAlmostEqual(res["stats"]["sharpe"], S.annualized_sharpe(r.to_numpy()), places=6)


# --------------------------------------------------------------------------
# calendar years, not bars / bars a year
# --------------------------------------------------------------------------

def _dense_then_sparse(seed=8):
    """Volume-bar stamps over 2021-2022: ~40 bars a day in 2021, ~10 in 2022."""
    a, _ = volume_bars("2021-01-01", "2021-12-31", seed=seed, lo=38, hi=43)
    b, _ = volume_bars("2022-01-01", "2022-12-31", seed=seed + 1, lo=8, hi=13)
    return a.append(b)


class CalendarYearsTests(_KeepAnnualization):

    def test_elapsed_years_reads_the_stamps_only_when_annualizing_on_the_data(self):
        idx = _dense_then_sparse()
        dense = idx[idx < "2022-01-01"]
        S.set_annualization(S.infer_annualization(idx))
        self.assertTrue(S.CALENDAR_YEARS)
        span = (dense[-1] - dense[0]).total_seconds() / (86400 * 365.25)
        self.assertAlmostEqual(S.elapsed_years(dense), span * len(dense) / (len(dense) - 1), places=12)
        self.assertAlmostEqual(S.elapsed_years(dense), 1.0, delta=0.01)
        # bars / bars a year would read the dense year as ~1.6 years
        self.assertGreater(len(dense) / S.periods_per_year(), 1.4)
        # the table, the synthetic series, a config from before: bars / bars a year
        for c in (dict(periods_per_year=252), dict(periods_per_year=252, annualization="interval")):
            S.set_annualization(c)
            self.assertFalse(S.CALENDAR_YEARS)
            self.assertEqual(S.elapsed_years(dense), len(dense) / 252)
        self.assertEqual(S.elapsed_years(None, 504), 2.0)
        self.assertEqual(S.elapsed_years(pd.RangeIndex(252)), 1.0)

    def test_cagr_and_the_walk_forward_efficiency_follow_the_calendar(self):
        """The same +10 % year, once in dense and once in sparse bars: one
        annualized return. Bars / bars a year would read it as ~6 % and
        ~25 %, and Pardo's WFE between the two would be ~0.25, not 1."""
        from walkforward import _annualized_return
        idx = _dense_then_sparse()
        S.set_annualization(S.infer_annualization(idx))
        parts = (("dense", idx[idx < "2022-01-01"]), ("sparse", idx[idx >= "2022-01-01"]))

        def stats():
            out = {}
            for name, part in parts:
                eq = 100.0 * np.exp(np.linspace(0.0, np.log(1.10), len(part)))      # +10 % over the year
                out[name] = S.performance_stats(pd.Series(eq, index=part), [], initial_equity=100.0)
            return out
        out = stats()
        for name, st in out.items():
            self.assertAlmostEqual(st["cagr"], 0.10, delta=0.002, msg=name)
            self.assertAlmostEqual(st["years"], 1.0, delta=0.02, msg=name)    # the bars cover Jan 3 - Dec 30
        self.assertAlmostEqual(_annualized_return(out["sparse"]) / _annualized_return(out["dense"]), 1.0, delta=0.02)
        # the same bars counted as bars / bars a year
        S.set_annualization(dict(S.infer_annualization(idx), annualization="interval"))
        out = stats()
        self.assertGreater(_annualized_return(out["sparse"]) / _annualized_return(out["dense"]), 3.0,
                           "bars / bars a year: the sparse year would look 4x as profitable")

    def test_curve_stats_and_the_oos_summary_use_calendar_years(self):
        from walkforward import summarize_walk_forward
        idx = _dense_then_sparse()
        S.set_annualization(S.infer_annualization(idx))
        r = pd.Series(np.full(len(idx), 1.21 ** (1 / (len(idx) - 1)) - 1), index=idx)
        r.iloc[0] = 0.0                                         # +21 % over two calendar years
        self.assertAlmostEqual(P.curve_stats(r)["cagr"], 0.10, delta=0.002)
        w = dict(skipped=False, is_stats=dict(sharpe=1.0, total_return=0.1, n_bars=10, years=1.0),
                 oos_stats=dict(total_return=0.21, n_trades=5, n_bars=len(idx), years=2.0), params_changed=False)
        self.assertAlmostEqual(summarize_walk_forward([w, w, w], r)["oos_cagr"], 0.10, delta=0.002)


# --------------------------------------------------------------------------
# DSR, bootstrap, Reality Check, effective N: one observation per session
# --------------------------------------------------------------------------

def _family(n_days=780, k=5, seed=9):
    """Daily returns of k correlated strategies (T x k)."""
    rng = np.random.default_rng(seed)
    common = rng.normal(0.0, 0.008, n_days)
    own = rng.normal(0.0003, 0.006, (n_days, k))
    return own + common[:, None] * np.linspace(0.2, 1.0, k)


def _cut(daily: np.ndarray, seed: int, lo: int, hi: int, additive: bool = False):
    """The days of `daily` (T or T x k) cut into volume bars: a frame of
    per-bar returns that compound (or, `additive`, sum) to them each day."""
    daily = np.atleast_2d(daily.T).T
    idx, counts = volume_bars(seed=seed, lo=lo, hi=hi)
    counts = counts[:len(daily)]
    idx = idx[:counts.sum()]
    rng = np.random.default_rng(seed)
    cols = []
    for j in range(daily.shape[1]):
        if additive:
            cols.append(np.concatenate([d * rng.dirichlet(np.ones(c)) for d, c in zip(daily[:, j], counts)]))
        else:
            cols.append(split_daily(daily[:, j], counts, rng))
    return pd.DataFrame(np.column_stack(cols), index=idx)


class SessionObservationTests(_KeepAnnualization):

    def setUp(self):
        super().setUp()
        self.daily = _family()
        self.cuts = [_cut(self.daily, seed=s, lo=lo, hi=hi) for s, lo, hi in ((11, 3, 6), (12, 20, 60))]

    def _clock(self, frame):
        S.set_annualization(S.infer_annualization(frame.index))

    def test_dsr_and_bootstrap_are_the_same_however_the_days_are_cut(self):
        import robustness as R
        out = []
        for f in self.cuts:
            self._clock(f)
            r = f[0]
            dsr = R.deflated_sharpe_ratio(r, n_trials=10, var_sr_trials=1e-4)
            boot = R.bootstrap_sharpe_pvalue(r, n_boot=300)
            out.append((dsr, boot))
            self.assertEqual(dsr["T"], len(self.daily))            # one observation per session
            self.assertAlmostEqual(dsr["sharpe_annual"], S.annualized_sharpe(r), places=10)
            ref = R.deflated_sharpe_ratio(S.session_returns(r), 10, 1e-4, ppy=S.sessions_per_year())
            for k in ("dsr", "psr0", "skew", "kurtosis"):
                self.assertAlmostEqual(dsr[k], ref[k], places=12)
        for k in ("dsr", "psr0", "sharpe_annual", "skew", "kurtosis"):
            self.assertAlmostEqual(out[0][0][k], out[1][0][k], places=8, msg=k)
        self.assertEqual(out[0][1]["p_value"], out[1][1]["p_value"])
        self.assertAlmostEqual(out[0][1]["sharpe"], out[1][1]["sharpe"], places=8)

    def test_reality_check_and_effective_trials_work_on_sessions(self):
        import robustness as R
        res = []
        for f in self.cuts:
            self._clock(f)
            res.append((R.reality_check(f, n_boot=200), R.effective_n_trials(f)))
        self.assertEqual(res[0][0]["p_value"], res[1][0]["p_value"])
        self.assertAlmostEqual(res[0][0]["statistic"], res[1][0]["statistic"], places=8)
        self.assertEqual(res[0][1]["n_eff"], res[1][1]["n_eff"])
        self.assertAlmostEqual(res[0][1]["var_sr_period"], res[1][1]["var_sr_period"], places=10)

    def test_daily_data_is_unchanged(self):
        import robustness as R
        idx = pd.bdate_range("2015-01-01", periods=len(self.daily))
        f = pd.DataFrame(self.daily, index=idx)
        S.set_annualization(dict(periods_per_year=252))
        self.assertIsNone(S.session_starts(idx))
        self.assertIs(S.session_frame(f), f)
        a = R.deflated_sharpe_ratio(f[0], 10, 1e-4)
        b = R.deflated_sharpe_ratio(f[0].to_numpy(), 10, 1e-4, ppy=252)
        self.assertEqual(a, b)
        self.assertEqual(R.bootstrap_sharpe_pvalue(f[0], n_boot=200), R.bootstrap_sharpe_pvalue(f[0].to_numpy(),
                                                                                                n_boot=200))


# --------------------------------------------------------------------------
# CPCV and PBO: groups and blocks of whole sessions, session Sharpes
# --------------------------------------------------------------------------

class SessionCpcvPboTests(_KeepAnnualization):

    def setUp(self):
        super().setUp()
        self.daily = _family(n_days=600, k=12, seed=13)          # 12 "trials", fixed-capital returns

    def _bars(self, lo, hi, seed):
        f = _cut(self.daily, seed=seed, lo=lo, hi=hi, additive=True)
        S.set_annualization(S.infer_annualization(f.index))
        return f.to_numpy(), S.session_starts(f.index)

    def test_cpcv_groups_hold_whole_sessions(self):
        import robustness as R
        R_, starts = self._bars(5, 40, seed=21)
        b = R._snap_bounds(np.linspace(0, len(R_), 7).astype(int), starts)
        self.assertTrue(np.isin(b[:-1], starts).all())
        self.assertEqual((b[0], b[-1]), (0, len(R_)))
        self.assertIs(R._snap_bounds(b, None), b)

    def test_cpcv_is_the_same_however_the_days_are_cut(self):
        """A constant number of bars a day puts the group bounds on the same
        sessions, so two cuts must select the same trials and give the same
        path Sharpes -- those of the daily matrix itself."""
        import robustness as R
        outs = []
        for k, seed in ((3, 31), (24, 32)):
            R_, starts = self._bars(k, k + 1, seed)
            outs.append(R.cpcv(R_, n_groups=6, k_test=2, embargo_bars=0, selection="best", additive=True,
                               sessions=starts))
        np.testing.assert_array_equal(outs[0]["chosen_trials"], outs[1]["chosen_trials"])
        np.testing.assert_allclose(outs[0]["path_sharpes"], outs[1]["path_sharpes"], rtol=1e-9)
        spy = S.sessions_per_year()
        S.set_annualization(dict(periods_per_year=spy))       # the daily matrix, per row at the same count
        daily = R.cpcv(self.daily, n_groups=6, k_test=2, embargo_bars=0, selection="best", additive=True)
        np.testing.assert_array_equal(daily["chosen_trials"], outs[0]["chosen_trials"])
        np.testing.assert_allclose(daily["path_sharpes"], outs[0]["path_sharpes"], rtol=1e-9)

    def test_pbo_ranks_session_sharpes(self):
        import robustness as R
        res = []
        for lo, hi, seed in ((3, 6, 41), (10, 80, 42)):          # any cut: the blocks are session rows
            R_, starts = self._bars(lo, hi, seed)
            blocks = R.cscv_block_stats(R_, 8, sessions=starts, additive=True)
            self.assertEqual(int(blocks["counts"].sum()), len(self.daily))
            self.assertEqual(blocks["per_year"], S.sessions_per_year())
            merged = R.merge_block_stats([blocks, blocks])
            self.assertEqual(merged["per_year"], blocks["per_year"])
            res.append(R.cscv_pbo(blocks=blocks))
        self.assertEqual(res[0]["pbo"], res[1]["pbo"])
        np.testing.assert_allclose(res[0]["is_sharpe"], res[1]["is_sharpe"], rtol=1e-9)
        np.testing.assert_allclose(res[0]["oos_sharpe"], res[1]["oos_sharpe"], rtol=1e-9)


# --------------------------------------------------------------------------
# review fixes: windows of too few sessions, additive streams
# --------------------------------------------------------------------------

def _minute_bars(n_days=40, per_day=390):
    """US-equity '1m' bars on a UTC clock."""
    days = pd.bdate_range("2024-01-02", periods=n_days)
    return pd.DatetimeIndex([d + pd.Timedelta(hours=14, minutes=30) + pd.Timedelta(minutes=m * 390 // per_day)
                             for d in days for m in range(per_day)])


class WindowSessionsNoteTests(_KeepAnnualization):
    """--train / --test count bars, every Sharpe counts sessions: on fine
    bars the default windows hold one or two sessions, and the optimizer
    would rank parameters on noise without a word."""

    def test_fine_bars_with_the_default_windows_get_a_note(self):
        idx = _minute_bars()
        S.set_annualization(S.infer_annualization(idx))
        note = P.window_sessions_note(idx, 500, 125)
        self.assertIsNotNone(note)
        self.assertIn("~390 bars a session", note)
        self.assertIn(f"--train {20 * 390}", note)            # ~20 sessions of 1m bars
        self.assertIn(f"--test {5 * 390}", note)
        self.assertIsNone(P.window_sessions_note(idx, 20 * 390, 5 * 390))
        self.assertIsNotNone(P.window_sessions_note(idx, 20 * 390, 125), "a too short test window alone")

    def test_hourly_bars_daily_bars_and_per_bar_statistics_get_none(self):
        idx = equity_hourly()
        S.set_annualization(S.infer_annualization(idx))
        self.assertIsNone(P.window_sessions_note(idx, 500, 125))          # ~71 and ~18 sessions
        self.assertIsNotNone(P.window_sessions_note(idx, 70, 125))        # 10 sessions of training
        daily = pd.bdate_range("2015-01-01", periods=2000)
        S.set_annualization(S.infer_annualization(daily))
        self.assertIsNone(P.window_sessions_note(daily, 50, 10))         # per bar: a bar is a session
        S.set_annualization(dict(periods_per_year=252 * 390))            # --annualization interval
        self.assertIsNone(P.window_sessions_note(_minute_bars(), 500, 125))

    def test_main_prints_it(self):
        d = tempfile.mkdtemp()
        try:
            idx = _minute_bars(n_days=12)
            df = synthetic_ohlc(n_bars=len(idx), seed=3, trend_vol=0.0008, range_vol=0.0005).set_axis(idx)
            df.index.name = "Datetime"
            csv = os.path.join(d, "m1.csv")
            df.to_csv(csv)
            log = io.StringIO()
            with contextlib.redirect_stdout(log):
                M.main(["--csv", csv, "--family", "quick", "--max-templates", "2", "--train", "1500",
                        "--test", "500", "--n-boot", "20", "--jobs", "1", "--no-matrix", "--min-sharpe", "-5",
                        "--out", d])
            self.assertIn("NOTE: the data holds ~390 bars a session", log.getvalue())
        finally:
            shutil.rmtree(d, ignore_errors=True)


class AdditiveSessionTests(_KeepAnnualization):
    """P&L over a fixed capital (the one-unit benchmark of a future, a
    fixed-capital backtest) and a residual of two return streams add up:
    per session they are summed, as robustness.cpcv sums the trial returns,
    never compounded."""

    def setUp(self):
        super().setUp()
        idx = equity_hourly("2023-01-01", "2023-12-31")
        S.set_annualization(S.infer_annualization(idx))
        rng = np.random.default_rng(17)
        self.r = pd.Series(rng.normal(0.002, 0.04, len(idx)), index=idx)   # large: compounding would show
        starts = S.session_starts(idx)
        self.sums = np.add.reduceat(self.r.to_numpy(), starts)
        self.prods = np.multiply.reduceat(1 + self.r.to_numpy(), starts) - 1

    def test_sharpe_and_vol_sum_additive_returns(self):
        sp = S.sessions_per_year()
        self.assertAlmostEqual(S.annualized_sharpe(self.r, additive=True),
                               self.sums.mean() / self.sums.std(ddof=1) * np.sqrt(sp), places=12)
        self.assertAlmostEqual(S.annualized_vol(self.r, additive=True), self.sums.std(ddof=1) * np.sqrt(sp),
                               places=12)
        self.assertAlmostEqual(S.annualized_sharpe(self.r),
                               self.prods.mean() / self.prods.std(ddof=1) * np.sqrt(sp), places=12)
        self.assertNotAlmostEqual(S.annualized_sharpe(self.r, additive=True), S.annualized_sharpe(self.r), places=3)
        np.testing.assert_allclose(S.session_returns(self.r, additive=True), self.sums, rtol=1e-12)

    def test_the_one_unit_benchmark_and_the_information_ratio_are_summed(self):
        a = P.curve_stats(self.r, additive=True)
        self.assertAlmostEqual(a["sharpe"], S.annualized_sharpe(self.r, additive=True), places=12)
        self.assertAlmostEqual(P.curve_stats(self.r)["sharpe"], S.annualized_sharpe(self.r), places=12)
        x = self.r.shift(1).fillna(0.0)
        y = 0.5 * x + self.r * 0.3
        ir = P.against_benchmark(y, x)
        beta = ir["beta"]
        self.assertAlmostEqual(ir["info_ratio"], S.annualized_sharpe(y - beta * x, additive=True), places=12)

    def test_a_fixed_capital_backtest_sums_its_returns(self):
        idx = self.r.index
        df = synthetic_ohlc(n_bars=len(idx), seed=5).set_axis(idx)
        res = S.backtest(df, S.StrategyTemplate("t", n_entry=40), fixed_capital=True)
        self.assertGreater(len(res["trades"]), 0)
        self.assertAlmostEqual(res["stats"]["sharpe"],
                               S.annualized_sharpe(res["returns"].iloc[1:], additive=True), places=12)

    def test_templates_beat_the_one_unit_benchmark_on_its_reported_sharpe(self):
        bh = self.r
        summed, compounded = S.annualized_sharpe(bh, additive=True), S.annualized_sharpe(bh)
        target = 0.5 * (summed + compounded)
        # a template whose Sharpe falls between the benchmark's summed and compounded ones
        lo, hi = -0.01, 0.01
        for _ in range(100):
            d = 0.5 * (lo + hi)
            lo, hi = (d, hi) if S.annualized_sharpe(0.01 * bh + d) < target else (lo, d)
        tpl = 0.01 * bh + d
        sharpe = S.annualized_sharpe(tpl)
        self.assertLess(min(summed, compounded), sharpe)
        self.assertLess(sharpe, max(summed, compounded))
        rets = pd.DataFrame({"t": tpl}, index=bh.index)
        port = dict(portfolio_returns=tpl)
        b = P.benchmark_stats(bh, rets, port, port, kind=P.BENCH_ONE_UNIT)
        self.assertAlmostEqual(b["buy_hold"]["sharpe"], summed, places=12)
        self.assertEqual(b["n_templates_beat_bh"], int(sharpe > summed))
        b = P.benchmark_stats(bh, rets, port, port)          # a return benchmark: compounded
        self.assertEqual(b["n_templates_beat_bh"], int(sharpe > compounded))

    def test_daily_data_is_unchanged(self):
        idx = pd.bdate_range("2015-01-01", periods=len(self.r))
        r = pd.Series(self.r.to_numpy(), index=idx)
        S.set_annualization(dict(periods_per_year=252))
        self.assertEqual(S.annualized_sharpe(r, additive=True), S.annualized_sharpe(r))
        self.assertEqual(S.annualized_vol(r, additive=True), S.annualized_vol(r))


if __name__ == "__main__":
    unittest.main()
