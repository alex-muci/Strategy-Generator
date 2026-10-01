"""
data.py
-------
Data loading for the Ranger-style breakout system generator.

Three sources are provided:

1. load_yfinance(ticker, start, end)  -> real daily OHLC data.
   Requires `pip install yfinance` and an internet connection.
   Use this in your own environment to run the pipeline on real markets.

2. load_csv(path)  -> OHLC(V) from a local file: anything Yahoo does not
   serve, such as a futures calendar spread, whose prices may be zero or
   negative (the engine handles them; see strategy.validate_instrument).

3. synthetic_ohlc(...)  -> a regime-switching random walk used for
   offline testing/demo purposes (no internet needed). It alternates
   between trending and range-bound regimes so that the different
   strategy templates (trend / counter-trend / sideways) actually
   have something structurally different to react to.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def load_yfinance(
    ticker: str,
    start: str,
    end: str | None = None,
    interval: str = "1d",
    min_bars: int = 200,
    drop_nonpositive: bool = True,
) -> pd.DataFrame:
    """Fetch OHLC data for `ticker` between `start` and `end` (YYYY-MM-DD).

    `interval` is any yfinance interval ('1d', '1h', '30m', ...). Note that
    Yahoo only serves intraday history for a limited window (about 2 years for
    '1h', 60 days for finer bars).

    Returns a DataFrame indexed by timestamp with columns
    Open, High, Low, Close, Volume: sorted, one row per stamp, and tz-naive
    (intraday bars in UTC, daily and longer bars on their exchange-local date;
    see `_naive_index`).

    `drop_nonpositive` drops bars with a price at or below zero: on a share
    they are bad prints. Turn it off for a future that genuinely traded
    there (WTI on 2020-04-20): dropping that bar would erase the gap every
    stop and P&L must go through.

    Raises ValueError rather than returning an empty frame: a wrong ticker, a
    rate limit or no network all make yfinance return an empty DataFrame, and a
    silent empty frame turns into a confusing IndexError deep in the pipeline.
    """
    import yfinance as yf

    df = yf.download(ticker, start=start, end=end, interval=interval,
                     auto_adjust=True, progress=False)
    if df is None or df.empty:
        raise ValueError(
            f"yfinance returned no data for {ticker!r} (interval={interval}, start={start}, "
            f"end={end}). Check the symbol, the date range (intraday history is short), "
            f"your network, and whether you are being rate-limited."
        )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    missing = [c for c in ("Open", "High", "Low", "Close") if c not in df.columns]
    if missing:
        raise ValueError(f"{ticker!r}: yfinance response is missing columns {missing}")
    if "Volume" not in df.columns:
        df = df.assign(Volume=np.nan)
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df = df[df[["Open", "High", "Low", "Close"]].notna().all(axis=1)]
    # a share cannot trade at or below zero: such a bar is a bad print, and the
    # engine would refuse the whole series over it (strategy.validate_instrument)
    bad = (df[["Open", "High", "Low", "Close"]] <= 0).any(axis=1)
    if drop_nonpositive and bad.any():
        print(f"  {ticker}: dropped {int(bad.sum())} bar(s) with a price at or below zero (bad prints)")
        df = df[~bad]
    df = df.set_axis(_naive_index(df.index, interval))
    # a repeated stamp (Yahoo sometimes serves the last bar twice) would be an
    # extra bar to every indicator and survive `drop_forming_bar`; keep the
    # latest print of each bar, in time order
    df = df[~df.index.duplicated(keep="last")].sort_index()
    if len(df) < min_bars:
        raise ValueError(
            f"{ticker!r}: only {len(df)} usable bars (interval={interval}), need at least "
            f"{min_bars}. Widen the date range or use a coarser interval."
        )
    df.index.name = "Date"
    return df


def load_csv(path: str, interval: str = "1d", min_bars: int = 200, drop_no_trade_rows: bool = True) -> pd.DataFrame:
    """OHLC(V) bars from a local CSV: the first column is the bar's timestamp,
    the others hold Open, High, Low, Close and optionally Volume and Roll
    (matched by name, any case; other columns are dropped). Roll is 1 (or
    true) on a bar at whose close the series rolled from one contract into
    the next, as `extra_utils/ETF_trick_spreads.py` writes it: the engine
    charges `--roll-cost-per-unit` there to whatever is held.

    Rows with a missing price are dropped, and, with `drop_no_trade_rows`,
    rows where every price is exactly 0 (the no-trade days a spread vendor
    prints; turn it off for a spread that can genuinely close a whole session
    at 0.00). Nothing else is: a negative or zero price is a price, on an
    instrument that has them, and a share's CSV should not carry one. The
    file is not stripped of a forming bar: export it after the close.
    The index comes back tz-naive (`_naive_index`), deduplicated (last print
    of a repeated stamp) and sorted, like `load_yfinance`'s.
    """
    raw = pd.read_csv(path, index_col=0, parse_dates=True)
    if not isinstance(raw.index, pd.DatetimeIndex):
        # stamps with mixed UTC offsets (a DST change) parse to objects: read
        # them all as UTC, which `_naive_index` then strips like yfinance's
        raw.index = pd.to_datetime(raw.index, utc=True)
    by_name = {str(c).strip().lower(): c for c in raw.columns}
    missing = [c for c in ("open", "high", "low", "close") if c not in by_name]
    if missing:
        raise ValueError(f"{path!r}: missing columns {[m.capitalize() for m in missing]} (have {list(raw.columns)})")
    df = pd.DataFrame({name: pd.to_numeric(raw[by_name[name.lower()]], errors="coerce")
                       for name in ("Open", "High", "Low", "Close")}, index=raw.index)
    df["Volume"] = pd.to_numeric(raw[by_name["volume"]], errors="coerce") if "volume" in by_name else np.nan
    if "roll" in by_name:
        df["Roll"] = _roll_flags(raw[by_name["roll"]], path)
    # in time order first (a file can be newest-first), then one print per
    # stamp (the last), a roll on any of the duplicates kept with it
    df = df.set_axis(_naive_index(df.index, interval)).sort_index(kind="stable")
    if "Roll" in df.columns:
        df["Roll"] = df["Roll"].groupby(level=0).transform("max")
    df = df[~df.index.duplicated(keep="last")]
    prices = df[["Open", "High", "Low", "Close"]]
    keep = prices.notna().all(axis=1)
    if drop_no_trade_rows:
        keep &= ~(prices == 0).all(axis=1)
    if "Roll" in df.columns:
        # a roll on a dropped row (a no-trade day) happened all the same: it
        # moves to the last kept bar before it, whose close the position held
        df["Roll"] = align_rolls(df["Roll"], df.index[keep])
    df = df[keep]
    if len(df) < min_bars:
        raise ValueError(f"{path!r}: only {len(df)} usable bars, need at least {min_bars}.")
    df.index.name = "Date"
    return df


# how far back Yahoo serves intraday bars (a request that starts earlier comes
# back empty): about 730 days of '1h', 60 days of anything finer. One day of
# margin each, since Yahoo counts from its own clock.
YAHOO_INTRADAY_DAYS = {"1h": 729, "60m": 729}
YAHOO_FINE_DAYS = 59


_ROLL_TOKENS = {"true": 1.0, "yes": 1.0, "y": 1.0, "t": 1.0, "false": 0.0, "no": 0.0, "n": 0.0, "f": 0.0,
                "": 0.0, "nan": 0.0, "none": 0.0}


def _roll_flags(col: pd.Series, path: str = "") -> pd.Series:
    """A CSV's Roll column as 0/1 floats: numbers (non-zero = a roll), booleans,
    or text (true/false, yes/no, blank). Anything else is an error: a roll flag
    silently read as 0 would charge no roll cost at all."""
    if pd.api.types.is_numeric_dtype(col) or pd.api.types.is_bool_dtype(col):
        return (pd.to_numeric(col, errors="coerce").fillna(0.0) != 0).astype(float)
    text = col.astype(str).str.strip().str.lower()
    num = pd.to_numeric(text, errors="coerce")
    tok = text.map(_ROLL_TOKENS)
    bad = num.isna() & tok.isna() & col.notna()
    if bad.any():
        raise ValueError(f"{path!r}: Roll column has values that are not a flag: "
                         f"{sorted(set(col[bad].astype(str)))[:5]} (use 1/0 or true/false)")
    return (num.fillna(tok).fillna(0.0) != 0).astype(float)


def align_rolls(roll: pd.Series, keep: pd.Index) -> pd.Series:
    """`roll` (1 on the bars at whose close a contract rolled) on the bars
    `keep`, a subset of its index: a roll on a bar that is not kept moves to
    the last kept bar before it (a position held at that close was still
    held when the roll happened, unless it exited in between, which a bar
    that is not there cannot show); one before the first kept bar is
    dropped (nothing could be held through it). Several rolls landing on one
    bar add up (each is paid). The result is indexed like `roll` (0 off
    `keep`), so it can be assigned back before the rows are dropped."""
    r = roll.fillna(0.0).astype(float)
    kept = r.index.isin(keep)
    if kept.all() or not kept.any():
        return r.where(kept, 0.0)
    # the kept bar each row's roll lands on: the last kept bar at or before it
    pos = np.arange(len(r))
    last_kept = pd.Series(np.where(kept, pos, np.nan)).ffill().to_numpy()
    ok = ~np.isnan(last_kept)
    out = np.zeros(len(r))
    np.add.at(out, last_kept[ok].astype(int), r.to_numpy()[ok])
    return pd.Series(out, index=r.index)


def yahoo_earliest_start(interval: str, now: pd.Timestamp | None = None) -> pd.Timestamp | None:
    """The earliest start date Yahoo still serves `interval` bars from, or None
    for daily and longer bars (decades of history)."""
    if not _is_intraday(interval):
        return None
    now = pd.Timestamp.now("UTC").tz_convert(None) if now is None else pd.Timestamp(now)
    if now.tz is not None:
        now = now.tz_convert("UTC").tz_localize(None)
    days = YAHOO_INTRADAY_DAYS.get(interval, YAHOO_FINE_DAYS)
    return (now - pd.Timedelta(days=days)).normalize()


def _is_intraday(interval: str) -> bool:
    return interval.endswith(("m", "h")) and not interval.endswith("mo")


def _naive_index(index: pd.Index, interval: str) -> pd.Index:
    """The project's one timestamp convention: every index is tz-NAIVE.

    * intraday bars: naive UTC (yfinance stamps them in the exchange's zone).
      `live.drop_forming_bar` and `live.utcnow` read naive stamps as UTC, and
      bars from different exchanges line up on the same clock;
    * daily and longer bars: the exchange-local DATE (Yahoo's own convention
      for them), which `drop_forming_bar` resolves against the exchange's
      closing time.
    """
    if not isinstance(index, pd.DatetimeIndex) or index.tz is None:
        return index
    if _is_intraday(interval):
        return index.tz_convert("UTC").tz_localize(None)
    return index.tz_localize(None)


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
