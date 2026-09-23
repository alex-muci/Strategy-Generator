"""
Tests for bars-per-year by market session (strategy.SESSIONS).

The annualization factor used to be one table of US cash-session bars, so
hourly bars of a market that trades 22-23 hours a day (ICE Brent, CME
Globex) were counted as seven a day: every Sharpe understated by
sqrt(7/22), every CAGR spread over three times the years, and every
vol-targeted entry sized about 1.8x too big. These pin the session tables,
the flags that choose them, the daily close each session implies, and the
warning when the data's own bar rate disagrees with the factor.

Run with:  python -m unittest discover -s tests -v
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

import main as M  # noqa: E402
import pipeline as P  # noqa: E402
import strategy as S  # noqa: E402
import etf_dashboard as ED  # noqa: E402
from data import synthetic_ohlc  # noqa: E402
from live import drop_forming_bar, utcnow  # noqa: E402
from walkforward import _annualized_return  # noqa: E402


def _hourly(n_bars: int, hours=range(1, 23), seed: int = 11) -> pd.DataFrame:
    """A synthetic series restamped as hourly bars at `hours` of every
    business day: 22 a day by default, ICE Brent's 01:00-23:00."""
    df = synthetic_ohlc(n_bars=n_bars, seed=seed)
    days = pd.bdate_range("2025-01-06", periods=n_bars // len(hours) + 2)
    stamps = [d + pd.Timedelta(hours=h) for d in days for h in hours][:n_bars]
    return df.set_axis(pd.DatetimeIndex(stamps, name="Date"))


class _RestoresAnnualization(unittest.TestCase):
    def setUp(self):
        self._ppy = S.periods_per_year()

    def tearDown(self):
        S.set_periods_per_year(self._ppy)


class SessionTableTests(unittest.TestCase):
    def test_the_us_cash_table_is_unchanged(self):
        self.assertEqual(S.BARS_PER_YEAR, {
            "1mo": 12, "1wk": 52, "1d": 252,
            "1h": 252 * 7, "60m": 252 * 7, "90m": 252 * 5,
            "30m": 252 * 13, "15m": 252 * 26, "5m": 252 * 78, "1m": 252 * 390,
        })
        for i in S.INTERVALS:
            self.assertEqual(S.periods_per_year_for_interval(i, "us_cash"), S.BARS_PER_YEAR[i])

    def test_ice_brent_trades_22_hours_a_day(self):
        f = lambda i: S.periods_per_year_for_interval(i, "ice_europe")  # noqa: E731
        self.assertEqual((f("1h"), f("60m"), f("30m"), f("1m")), (252 * 22, 252 * 22, 252 * 44, 252 * 1320))
        self.assertEqual(f("90m"), 252 * 15)          # 1320 / 90 = 14.7: the stub is a bar

    def test_cme_globex_trades_23_hours_a_day(self):
        self.assertEqual(S.periods_per_year_for_interval("1h", "cme_globex"), 252 * 23)
        self.assertEqual(S.periods_per_year_for_interval("5m", "cme_globex"), 252 * 276)

    def test_daily_and_longer_bars_do_not_depend_on_the_hours(self):
        for s in S.SESSIONS:
            self.assertEqual(S.periods_per_year_for_interval("1d", s), 252)
            self.assertEqual(S.periods_per_year_for_interval("1wk", s), 52)
            self.assertEqual(S.periods_per_year_for_interval("1mo", s), 12)

    def test_unknown_session_or_interval_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown session"):
            S.periods_per_year_for_interval("1h", "lme_ring")
        with self.assertRaisesRegex(ValueError, "unknown interval"):
            S.periods_per_year_for_interval("4h", "ice_europe")


class AnnualizedStatisticsTests(_RestoresAnnualization):
    """The same hourly returns, annualized as the market they came from."""

    def setUp(self):
        super().setUp()
        self.r = np.random.default_rng(3).normal(0.0002, 0.002, 5544)
        self.us = S.periods_per_year_for_interval("1h", "us_cash")
        self.ice = S.periods_per_year_for_interval("1h", "ice_europe")

    def test_sharpe(self):
        S.set_periods_per_year(self.us)
        us = S.annualized_sharpe(self.r)
        S.set_periods_per_year(self.ice)
        self.assertAlmostEqual(S.annualized_sharpe(self.r) / us, np.sqrt(22 / 7), places=9)

    def test_cagr_of_one_year_of_brent_hours(self):
        """One year of 22-hour days is one year, not three."""
        eq = 100_000 * np.cumprod(np.full(self.ice, 1.0 + 0.10 / self.ice))
        S.set_periods_per_year(self.ice)
        stats = S.performance_stats(eq, [], 100_000.0)
        self.assertAlmostEqual(stats["cagr"], eq[-1] / 100_000 - 1, places=9)
        S.set_periods_per_year(self.us)
        self.assertLess(S.performance_stats(eq, [], 100_000.0)["cagr"], 0.04)

    def test_walk_forward_efficiency_inputs(self):
        stats = dict(total_return=0.02, n_bars=500)
        S.set_periods_per_year(self.ice)
        self.assertAlmostEqual(_annualized_return(stats), 0.02 * self.ice / 500)

    def test_deflated_sharpe(self):
        import robustness as R
        S.set_periods_per_year(self.ice)
        d = R.deflated_sharpe_ratio(self.r, n_trials=10, var_sr_trials=1e-5)
        self.assertAlmostEqual(d["sharpe_annual"], S.annualized_sharpe(self.r), places=6)

    def test_vol_target_sizes_on_the_sessions_hours(self):
        """A 15% vol target is 15% a year of 22-hour days: sizing on seven
        a day would put on sqrt(22/7) times the intended position."""
        df = _hourly(400)
        tpl = S.StrategyTemplate(name="vt", vol_target=0.15, vol_target_n=40, max_leverage=100.0)
        notional = {}
        for name, ppy in (("us", self.us), ("ice", self.ice)):
            S.set_periods_per_year(ppy)
            res = S.backtest(df, tpl)
            first = res["trades"][0]
            notional[name] = first["shares"] * first["entry_price"]
        self.assertAlmostEqual(notional["us"] / notional["ice"], np.sqrt(22 / 7), places=6)


class ConfigTests(unittest.TestCase):
    def test_session_sets_the_factor(self):
        args = M.parse_args(["--session", "ice_europe"])
        cfg = P.eval_config(args, "1h")
        self.assertEqual((cfg["session"], cfg["periods_per_year"]), ("ice_europe", 252 * 22))
        self.assertEqual(P.eval_config(M.parse_args([]), "1h")["session"], "us_cash")

    def test_bars_per_year_overrides_the_session(self):
        args = M.parse_args(["--session", "ice_europe", "--bars-per-year", "6000"])
        self.assertEqual(P.eval_config(args, "1h")["periods_per_year"], 6000)

    def test_the_two_entry_points_take_the_same_flags(self):
        argv = ["--session", "cme_globex", "--bars-per-year", "5000"]
        self.assertEqual(P.eval_config(M.parse_args(argv), "1h"),
                         P.eval_config(ED.parse_args(["research", *argv]), "1h"))

    def test_unknown_session_is_a_usage_error(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            M.parse_args(["--session", "lme_ring"])

    def test_synthetic_data_ignores_the_overrides(self):
        args = M.parse_args(["--interval", "1h", "--bars-per-year", "6000"])
        notes = P.resolve_bar_clock(args, synthetic=True)
        self.assertEqual((args.interval, args.bars_per_year, len(notes)), ("1d", None, 2))
        args = M.parse_args(["--interval", "1h", "--bars-per-year", "6000"])
        self.assertEqual(P.resolve_bar_clock(args, synthetic=False), [])
        self.assertEqual((args.interval, args.bars_per_year), ("1h", 6000))


class ObservedRateTests(unittest.TestCase):
    def test_brent_hours_measured_off_the_index(self):
        idx = _hourly(22 * 120).index
        seen = P.observed_bars_per_year(idx)
        self.assertAlmostEqual(seen / (22 * 261), 1.0, delta=0.03)   # business days, holidays aside
        self.assertIsNone(P.bars_per_year_warning(idx, 252 * 22))
        warning = P.bars_per_year_warning(idx, 252 * 7)
        self.assertIn("scales every Sharpe by 0.55x", warning)
        self.assertIn("--session", warning)

    def test_daily_series_is_fine_at_252(self):
        self.assertIsNone(P.bars_per_year_warning(synthetic_ohlc(n_bars=600).index, 252))

    def test_too_short_to_tell(self):
        idx = _hourly(22 * 10).index
        self.assertIsNone(P.observed_bars_per_year(idx))
        self.assertIsNone(P.bars_per_year_warning(idx, 1))


class DailyCloseTests(unittest.TestCase):
    """A daily Brent bar is still trading hours after the New York close."""

    def setUp(self):
        idx = pd.bdate_range("2026-02-02", "2026-03-10")
        self.df = pd.DataFrame({c: 1.0 for c in ("Open", "High", "Low", "Close", "Volume")}, index=idx)

    def test_brent_bar_kept_only_after_the_london_close(self):
        at_2100_utc = pd.Timestamp("2026-03-10 21:00")        # 17:00 New York, 21:00 London
        kept = drop_forming_bar(self.df, "1d", now=at_2100_utc, session_close=P.session_close("us_cash"))
        dropped = drop_forming_bar(self.df, "1d", now=at_2100_utc, session_close=P.session_close("ice_europe"))
        self.assertEqual(len(kept), len(self.df))
        self.assertEqual(len(dropped), len(self.df) - 1)
        after = drop_forming_bar(self.df, "1d", now=pd.Timestamp("2026-03-10 23:20"),
                                 session_close=P.session_close("ice_europe"))
        self.assertEqual(len(after), len(self.df))

    def test_dashboard_loads_unlisted_assets_on_the_sessions_close(self):
        seen = {}

        def fake(ticker, start, interval="1d", now=None, session_close=None):
            seen[ticker] = session_close
            return self.df

        orig = ED.load_real
        ED.load_real = fake
        try:
            ED.load_assets(["BZ=F"], interval="1d", start="2026-01-01", synthetic=False, bars=0,
                           quiet=True, default_close=P.session_close("cme_globex"))
        finally:
            ED.load_real = orig
        self.assertEqual(seen, {"BZ=F": ("16:00", "America/Chicago")})


class HistoryStartTests(unittest.TestCase):
    def test_more_bars_a_day_need_less_calendar_history(self):
        us = pd.Timestamp(ED._start_for_bars(1000, "1h"))
        ice = pd.Timestamp(ED._start_for_bars(1000, "1h", 252 * 22))
        self.assertGreater(ice, us)

    def test_never_older_than_yahoo_keeps(self):
        for interval, days in (("1h", 730), ("30m", 60), ("1m", 7)):
            start = pd.Timestamp(ED._start_for_bars(100_000, interval))
            self.assertLess((utcnow().normalize() - start).days, days, interval)
        daily = pd.Timestamp(ED._start_for_bars(500, "1d"))           # daily bars are not clamped
        self.assertAlmostEqual((utcnow() - daily).days, 365.25 * (500 / 252 * 1.6 + 0.5), delta=2)


class EntryPointTests(_RestoresAnnualization):
    ARGV = ["--real", "BZ", "--interval", "1h", "--max-templates", "2", "--train", "300", "--test", "100",
            "--n-boot", "20", "--min-sharpe", "-5", "--no-matrix", "--jobs", "1"]

    def _run(self, *extra):
        df = _hourly(22 * 40)
        seen = {}

        def fake(ticker, start, interval="1d", now=None, session_close=None):
            seen.update(interval=interval, session_close=session_close)
            return df

        d = tempfile.mkdtemp(prefix="main-ice-")
        orig = M.load_real
        M.load_real = fake
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                out = M.main([*self.ARGV, *extra, "--out", d])
            with open(os.path.join(d, "report.md")) as f:
                report = f.read()
        finally:
            M.load_real = orig
            shutil.rmtree(d, ignore_errors=True)
        return out, report, seen

    def test_main_annualizes_brent_hours(self):
        out, report, seen = self._run("--session", "ice_europe")
        self.assertEqual(S.periods_per_year(), 252 * 22)
        self.assertEqual(seen, dict(interval="1h", session_close=("23:00", "Europe/London")))
        self.assertAlmostEqual(out["family"]["years_available"], len(out["returns"]) / (252 * 22))
        self.assertIn("ice_europe session", report)
        self.assertIn(f"annualized at {252 * 22} bars/year", report)
        self.assertNotIn("**Warning:**", report)

    def test_the_report_flags_a_session_that_does_not_fit(self):
        _, report, _ = self._run()                            # us_cash, 7 bars a day
        self.assertIn("**Warning:** the data has about", report)


if __name__ == "__main__":
    unittest.main()
