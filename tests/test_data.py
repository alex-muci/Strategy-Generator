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
