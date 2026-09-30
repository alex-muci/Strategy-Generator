"""
Tests for data.py.

`synthetic_ohlc` is the fixture under almost every other test in the suite, so
a defect in it (bars whose High does not contain the Close, a series that
changes between runs) would quietly weaken all of them. `load_yfinance` is
exercised against a fake yfinance module: no network is touched.

Run with:  python -m unittest discover -s tests -v
"""

from __future__ import annotations
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data as D  # noqa: E402


class SyntheticDataTests(unittest.TestCase):
    def test_shape_and_index(self):
        for n in (1, 50, 777):
            df = D.synthetic_ohlc(n_bars=n, seed=1)
            self.assertEqual(len(df), n)
            self.assertEqual(list(df.columns), ["Open", "High", "Low", "Close", "Volume"])
        self.assertTrue(df.index.is_monotonic_increasing and df.index.is_unique)
        self.assertTrue((df.index.dayofweek < 5).all(), "business days only")
        self.assertEqual(df.index.name, "Date")

    def test_bars_are_well_formed(self):
        df = D.synthetic_ohlc(n_bars=2000, seed=3)
        self.assertFalse(df.isna().any().any())
        self.assertTrue((df[["Open", "High", "Low", "Close"]] > 0).all().all())
        self.assertTrue((df["High"] >= df[["Open", "Close", "Low"]].max(axis=1)).all())
        self.assertTrue((df["Low"] <= df[["Open", "Close", "High"]].min(axis=1)).all())
        # a bar opens where the previous one closed
        np.testing.assert_allclose(df["Open"].to_numpy()[1:], df["Close"].to_numpy()[:-1])

    def test_a_seed_is_a_series(self):
        a, b = D.synthetic_ohlc(500, seed=9), D.synthetic_ohlc(500, seed=9)
        pd.testing.assert_frame_equal(a, b)
        self.assertFalse(np.allclose(a["Close"], D.synthetic_ohlc(500, seed=10)["Close"]))
        # a longer series extends a shorter one's closes (truncation tests rely on it)
        np.testing.assert_allclose(D.synthetic_ohlc(300, seed=9)["Close"], a["Close"].iloc[:300])

    def test_the_regime_knobs_do_something(self):
        def trendiness(**kw):
            c = D.synthetic_ohlc(4000, seed=5, **kw)["Close"]
            r = np.log(c).diff().dropna()
            return abs(r.rolling(60).sum()).mean()
        self.assertGreater(trendiness(trend_prob=1.0, trend_drift=0.003),
                           2 * trendiness(trend_prob=0.0))


class _FakeYF:
    """Stands in for the yfinance module; returns whatever frame it was given."""
    def __init__(self, frame):
        self.frame, self.calls = frame, []

    def download(self, ticker, **kw):
        self.calls.append((ticker, kw))
        return self.frame


def _frame(n=300, cols=("Open", "High", "Low", "Close", "Volume")):
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame({c: np.linspace(100.0, 120.0, n) for c in cols}, index=idx)


class LoaderTests(unittest.TestCase):
    def _load(self, frame, **kw):
        fake = _FakeYF(frame)
        real = sys.modules.get("yfinance")
        sys.modules["yfinance"] = fake
        try:
            return D.load_yfinance("TEST", start="2020-01-01", **kw), fake
        finally:
            if real is None:
                del sys.modules["yfinance"]
            else:
                sys.modules["yfinance"] = real

    def test_a_clean_frame_passes_through(self):
        df, fake = self._load(_frame(), interval="1h")
        self.assertEqual(list(df.columns), ["Open", "High", "Low", "Close", "Volume"])
        self.assertEqual((len(df), df.index.name), (300, "Date"))
        ticker, kw = fake.calls[0]
        self.assertEqual((ticker, kw["interval"], kw["start"], kw["auto_adjust"]), ("TEST", "1h", "2020-01-01", True))

    def test_multiindex_columns_are_flattened(self):
        f = _frame()
        f.columns = pd.MultiIndex.from_product([f.columns, ["TEST"]])
        df, _ = self._load(f)
        self.assertEqual(list(df.columns), ["Open", "High", "Low", "Close", "Volume"])

    def test_extra_columns_are_dropped_and_volume_is_optional(self):
        f = _frame().assign(Dividends=0.0).drop(columns="Volume")
        df, _ = self._load(f)
        self.assertEqual(list(df.columns), ["Open", "High", "Low", "Close", "Volume"])
        self.assertTrue(df["Volume"].isna().all())

    def test_rows_with_a_missing_price_are_dropped(self):
        f = _frame()
        f.iloc[10, f.columns.get_loc("Close")] = np.nan
        f.iloc[20, f.columns.get_loc("Volume")] = np.nan       # a missing volume is not a missing bar
        df, _ = self._load(f)
        self.assertEqual(len(df), 299)
        self.assertNotIn(f.index[10], df.index)
        self.assertIn(f.index[20], df.index)

    def test_failures_raise_instead_of_returning_rubbish(self):
        for frame, msg in ((pd.DataFrame(), "no data"),
                           (_frame(cols=("Open", "High", "Close")), "missing columns"),
                           (_frame(n=50), "only 50 usable bars")):
            with self.assertRaises(ValueError) as cm:
                self._load(frame)
            self.assertIn(msg, str(cm.exception))
        df, _ = self._load(_frame(n=50), min_bars=50)
        self.assertEqual(len(df), 50)

    def test_nothing_at_all_is_an_error_not_an_attribute_error(self):
        with self.assertRaises(ValueError) as cm:
            self._load(None)
        self.assertIn("no data", str(cm.exception))

    def test_a_bar_at_or_below_zero_is_a_bad_print_and_is_dropped(self):
        """A share cannot trade at 0: Yahoo's occasional zero print would
        otherwise make the engine refuse the whole series as a spread."""
        frame = _frame()
        frame.iloc[10, frame.columns.get_loc("Low")] = 0.0
        frame.iloc[20, frame.columns.get_loc("Close")] = -3.0
        df, _ = self._load(frame)
        self.assertEqual(len(df), 298)
        self.assertNotIn(frame.index[10], df.index)
        self.assertNotIn(frame.index[20], df.index)
        self.assertTrue((df[["Open", "High", "Low", "Close"]] > 0).all().all())

    def test_repeated_and_unordered_bars_are_cleaned(self):
        f = _frame()
        last = f.iloc[[-1]].assign(Close=999.0)             # the later print of the same bar
        f = pd.concat([f.iloc[::-1], last])
        df, _ = self._load(f)
        self.assertEqual(len(df), 300)
        self.assertTrue(df.index.is_unique and df.index.is_monotonic_increasing)
        self.assertEqual(df["Close"].iloc[-1], 999.0)

    def test_intraday_stamps_become_naive_utc(self):
        """yfinance stamps intraday bars in the exchange's zone; the project's
        clock (live.utcnow, drop_forming_bar) is naive UTC."""
        f = _frame()
        f.index = pd.date_range("2026-03-02 09:30", periods=300, freq="h", tz="America/New_York")
        df, _ = self._load(f, interval="1h")
        self.assertIsNone(df.index.tz)
        self.assertEqual(df.index[0], pd.Timestamp("2026-03-02 14:30"))
        self.assertEqual(df.index.name, "Date")

    def test_an_aware_daily_index_keeps_its_local_date(self):
        f = _frame()
        f.index = f.index.tz_localize("Asia/Tokyo")          # midnight Tokyo is the previous day in UTC
        df, _ = self._load(f, interval="1d")
        self.assertIsNone(df.index.tz)
        self.assertEqual(df.index[0], pd.Timestamp("2020-01-01"))


if __name__ == "__main__":
    unittest.main()


class CsvLoaderTests(unittest.TestCase):
    """`load_csv`: a local file for what Yahoo does not serve, e.g. a spread."""

    def setUp(self):
        import tempfile
        self.dir = tempfile.mkdtemp()
        df = D.synthetic_ohlc(400, seed=9)
        shift = float(df["High"].max()) + 3.0
        for c in ("Open", "High", "Low", "Close"):
            df[c] = df[c] - shift            # every price below zero
        self.df = df
        self.path = os.path.join(self.dir, "spread.csv")
        df.to_csv(self.path, float_format="%.17g")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_round_trip_keeps_negative_prices(self):
        out = D.load_csv(self.path)
        pd.testing.assert_frame_equal(out, self.df, check_exact=False, rtol=1e-12, check_names=False, check_freq=False)
        self.assertEqual(out.index.name, "Date")
        self.assertIsNone(out.index.tz)
        self.assertLess(float(out["High"].max()), 0.0)

    def test_no_trade_rows_are_dropped_and_zero_closes_are_kept(self):
        df = self.df.copy()
        df.iloc[10, :4] = 0.0                            # a no-trade day: all-zero prices
        df.iloc[20, [0, 3]] = 0.0                        # a real zero print: kept
        df.iloc[30, 1] = np.nan                          # a missing price: dropped
        df = pd.concat([df, df.iloc[[5]]]).sample(frac=1.0, random_state=1)   # a duplicate stamp, shuffled
        df.to_csv(os.path.join(self.dir, "messy.csv"))
        out = D.load_csv(os.path.join(self.dir, "messy.csv"))
        self.assertEqual(len(out), 398)
        self.assertTrue(out.index.is_monotonic_increasing and out.index.is_unique)
        self.assertNotIn(self.df.index[10], out.index)
        self.assertNotIn(self.df.index[30], out.index)
        self.assertEqual(float(out.loc[self.df.index[20], "Close"]), 0.0)

    def test_columns_match_any_case_and_volume_is_optional(self):
        df = self.df.rename(columns=str.lower).drop(columns="volume")
        df.index.name = "timestamp"
        df.to_csv(os.path.join(self.dir, "lower.csv"))
        out = D.load_csv(os.path.join(self.dir, "lower.csv"))
        self.assertEqual(list(out.columns), ["Open", "High", "Low", "Close", "Volume"])
        self.assertTrue(out["Volume"].isna().all())
        np.testing.assert_allclose(out["Close"].to_numpy(), self.df["Close"].to_numpy())

    def test_missing_price_columns_and_too_few_bars_raise(self):
        self.df.drop(columns="Low").to_csv(os.path.join(self.dir, "nolow.csv"))
        with self.assertRaises(ValueError) as cm:
            D.load_csv(os.path.join(self.dir, "nolow.csv"))
        self.assertIn("Low", str(cm.exception))
        with self.assertRaises(ValueError):
            D.load_csv(self.path, min_bars=1000)

    def test_mixed_utc_offsets_and_the_no_trade_switch(self):
        """Stamps across a DST change carry two offsets; a genuine 0/0/0/0
        session is kept when asked."""
        idx = pd.date_range("2024-03-01 15:00", periods=400, freq="h", tz="America/New_York")
        df = self.df.iloc[:400].set_axis(idx)
        df.iloc[7, :4] = 0.0
        p = os.path.join(self.dir, "dst.csv")
        df.to_csv(p)
        out = D.load_csv(p, interval="1h")
        self.assertIsInstance(out.index, pd.DatetimeIndex)
        self.assertIsNone(out.index.tz)
        self.assertEqual(out.index[0], pd.Timestamp("2024-03-01 20:00"))     # naive UTC, like yfinance's
        self.assertEqual(len(out), 399)
        kept = D.load_csv(p, interval="1h", drop_no_trade_rows=False)
        self.assertEqual(len(kept), 400)
        self.assertEqual(float(kept["Close"].iloc[7]), 0.0)

