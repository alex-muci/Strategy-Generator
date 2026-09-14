"""
data.py
-------
Data loading for the Ranger-style breakout system generator.

Two sources are provided:

1. load_yfinance(ticker, start, end)  -> real daily OHLC data.
   Requires `pip install yfinance` and an internet connection.
   Use this in your own environment to run the pipeline on real markets.

2. synthetic_ohlc(...)  -> a regime-switching random walk used for
   offline testing/demo purposes (no internet needed). It alternates
   between trending and range-bound regimes so that the different
   strategy templates (trend / counter-trend / sideways) actually
   have something structurally different to react to.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def load_yfinance(ticker: str, start: str, end: str | None = None) -> pd.DataFrame:
    """Fetch daily OHLC data for `ticker` between `start` and `end` (YYYY-MM-DD).

    Returns a DataFrame indexed by date with columns:
    Open, High, Low, Close, Volume
    """
    import yfinance as yf

    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.index.name = "Date"
    return df


def synthetic_ohlc(
    n_bars: int = 2500,
    seed: int = 42,
    start_price: float = 100.0,
    regime_len_range: tuple[int, int] = (60, 180),
    trend_vol: float = 0.010,
    range_vol: float = 0.006,
    trend_drift: float = 0.0009,
    trend_prob: float = 0.45,
    start_date: str = "2015-01-01",
) -> pd.DataFrame:
    """Generate a synthetic daily OHLC series that alternates between
    trending regimes (steady drift, higher vol) and range-bound regimes
    (near-zero drift, mean-reverting, lower vol).

    This is only meant to exercise the pipeline end-to-end when no
    internet/data feed is available. Swap this out for load_yfinance
    (or your own data source) for real analysis.
    """
    rng = np.random.default_rng(seed)
    closes = [start_price]
    regimes = []

    bars_left = 0
    regime = "trend"
    direction = 1
    mean_level = start_price

    while len(closes) < n_bars:
        if bars_left <= 0:
            regime = rng.choice(["trend", "range"], p=[trend_prob, 1 - trend_prob])
            bars_left = rng.integers(*regime_len_range)
            direction = rng.choice([-1, 1])
            mean_level = closes[-1]

        last = closes[-1]
        if regime == "trend":
            ret = direction * trend_drift + rng.normal(0, trend_vol)
        else:
            # mean-reverting pull back toward mean_level of this regime
            pull = 0.02 * (mean_level - last) / last
            ret = pull + rng.normal(0, range_vol)

        closes.append(last * (1 + ret))
        regimes.append(regime)
        bars_left -= 1

    closes = np.array(closes[: n_bars + 1])
    dates = pd.bdate_range(start=start_date, periods=len(closes))

    # Build OHLC around the close path with plausible intrabar range
    intrabar = np.abs(rng.normal(0, 0.004, size=len(closes))) + 0.0015
    high = closes * (1 + intrabar)
    low = closes * (1 - intrabar)
    open_ = np.roll(closes, 1)
    open_[0] = closes[0]
    volume = rng.integers(1_000_000, 5_000_000, size=len(closes))

    df = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": closes, "Volume": volume},
        index=dates,
    )
    df.index.name = "Date"
    # make sure High/Low actually bracket Open/Close
    df["High"] = df[["High", "Open", "Close"]].max(axis=1)
    df["Low"] = df[["Low", "Open", "Close"]].min(axis=1)
    return df.iloc[:n_bars]
