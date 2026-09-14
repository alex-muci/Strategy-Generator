"""
strategy.py
-----------
A Ranger-style "breakout strategy generator" building block.

Instead of one fixed system, a strategy is defined by a set of
structural SWITCHES (categorical choices that change the logic
entirely) plus a small set of NUMERIC PARAMS (tuned by walk-forward
analysis in walkforward.py).

Switches (define a "template" -- a structurally distinct strategy):

  direction_logic : 'trend'        -> trade WITH the break (buy new highs,
                                      sell new lows)
                    'countertrend' -> FADE the break (sell new highs, buy
                                      new lows), i.e. range / reversion

  channel_type    : 'donchian'  -> highest high / lowest low of n bars
                    'keltner'   -> EMA(n) +/- channel_k * ATR(atr_n)
                    'bollinger' -> SMA(n) +/- channel_k * stdev(n)

  entry_style     : 'stop'          -> enter the moment the channel trades
                                       (stop order for trend, limit for fade)
                    'close_confirm' -> require a CLOSE beyond the channel,
                                       enter at the next open
                    'pullback'      -> after the break, wait for price to
                                       pull back `pullback_atr_mult` * ATR
                                       from the channel (limit order, valid
                                       `pullback_valid_bars` bars)

  exit_style      : 'channel'     -> trend: opposite n_exit channel
                                     (Turtle "channel-in / channel-out");
                                     countertrend: revert to the n_exit
                                     channel midline (mean-reversion target)
                    'atr_trail'   -> ATR trailing stop (chandelier)
                    'target_stop' -> fixed ATR-multiple stop and target
                    'time_stop'   -> exit after `max_hold_bars` bars
                    (a hard ATR stop is ALWAYS active, whatever the style)

  regime_indicator: which "trendiness" measure the regime filter uses:
                    'er'   Kaufman Efficiency Ratio          (0..1)
                    'adx'  Wilder's ADX                      (0..100)
                    'cti'  |Ehlers' Correlation Trend Index| (0..1)
                    'chop' 100 - Choppiness Index            (0..100)
                    'vr'   Variance ratio (Lo-MacKinlay)     (~1 = random walk)
                    All are oriented so HIGHER = MORE TRENDING.

  regime_filter   : 'none'       -> trade in any regime
                    'trend_only' -> only when indicator >= regime_threshold
                    'range_only' -> only when indicator <  regime_threshold
                                    (Ranger's "sideways" mode)

  vol_filter      : True/False -> skip entries when ATR is in an extreme
                                  percentile of its own history

  bias_filter     : 'none' -> longs and shorts always allowed
                    'sma'  -> longs only above SMA(bias_n), shorts only
                              below it ("market direction" filter from
                              financial-hacker's Market Regime Filter)

Numeric params (walk-forward optimized, see generator.param_grid_for):
  n_entry, n_exit, atr_n, channel_k, atr_mult_stop, atr_mult_target,
  atr_mult_trail, pullback_atr_mult, pullback_valid_bars, max_hold_bars,
  regime_n, regime_threshold, vol_lookback, vol_low_pct, vol_high_pct,
  bias_n, risk_pct, max_leverage, cost_bps.

Execution model (no look-ahead):
  * every decision on bar i uses indicator values fully formed on bar i-1
  * stops/limits are filled intrabar at the level, or at the open if the
    open gapped through the level
  * costs: `cost_bps` (commission + slippage) charged per side on notional
  * equity is marked to market at the CLOSE of each bar, after all fills
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd

PERIODS_PER_YEAR = 252


# --------------------------------------------------------------------------
# Indicators
# --------------------------------------------------------------------------

def true_range(df: pd.DataFrame) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    return pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)


def atr(df: pd.DataFrame, n: int) -> pd.Series:
    return true_range(df).rolling(n).mean()


def sma(x: pd.Series, n: int) -> pd.Series:
    return x.rolling(n).mean()


def ema(x: pd.Series, n: int) -> pd.Series:
    # min_periods=n so the first n-1 values are NaN (no partially warmed-up EMA)
    return x.ewm(span=n, adjust=False, min_periods=n).mean()


def efficiency_ratio(close: pd.Series, n: int) -> pd.Series:
    """Kaufman's Efficiency Ratio: net move / sum of absolute moves over n bars.
    ~1.0 = strongly trending (efficient), ~0.0 = choppy/sideways."""
    change = (close - close.shift(n)).abs()
    volatility = close.diff().abs().rolling(n).sum()
    er = change / volatility.replace(0, np.nan)
    return er.clip(0, 1)


def adx(df: pd.DataFrame, n: int) -> pd.Series:
    """Wilder's Average Directional Index (0..100). >25 is usually read as trending."""
    high, low = df["High"], df["Low"]
    up = high.diff()
    down = -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    tr = true_range(df)
    alpha = 1.0 / n
    tr_s = tr.ewm(alpha=alpha, adjust=False, min_periods=n).mean()
    plus_di = 100 * plus_dm.ewm(alpha=alpha, adjust=False, min_periods=n).mean() / tr_s.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=alpha, adjust=False, min_periods=n).mean() / tr_s.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=alpha, adjust=False, min_periods=n).mean()


def cti(close: pd.Series, n: int) -> pd.Series:
    """Ehlers' Correlation Trend Indicator (TASC 5/2020): Pearson correlation
    between the last n closes and a straight line. +1 = perfect up-trend,
    -1 = perfect down-trend, ~0 = no trend. Computed with rolling sums so it
    is O(T) rather than O(T*n)."""
    y = close.to_numpy(dtype=float)
    T = len(y)
    out = np.full(T, np.nan)
    if T < n:
        return pd.Series(out, index=close.index)
    x = np.arange(n, dtype=float)
    sx, sxx = x.sum(), (x * x).sum()
    sy = np.convolve(y, np.ones(n), "valid")
    syy = np.convolve(y * y, np.ones(n), "valid")
    # sum_k x_k * y_{t-n+1+k}: correlate with reversed weights
    sxy = np.convolve(y, x[::-1], "valid")
    num = n * sxy - sx * sy
    den = np.sqrt(np.maximum(n * sxx - sx * sx, 0) * np.maximum(n * syy - sy * sy, 0))
    with np.errstate(invalid="ignore", divide="ignore"):
        val = np.where(den > 0, num / den, 0.0)
    out[n - 1 :] = val
    return pd.Series(out, index=close.index).clip(-1, 1)


def choppiness(df: pd.DataFrame, n: int) -> pd.Series:
    """Choppiness Index (Dreiss), 0..100. High (>61.8) = choppy, low (<38.2) = trending."""
    tr_sum = true_range(df).rolling(n).sum()
    rng = df["High"].rolling(n).max() - df["Low"].rolling(n).min()
    with np.errstate(invalid="ignore", divide="ignore"):
        ci = 100 * np.log10(tr_sum / rng.replace(0, np.nan)) / np.log10(n)
    return ci.clip(0, 100)


def variance_ratio(close: pd.Series, n: int, q: int = 5) -> pd.Series:
    """Lo-MacKinlay style variance ratio over a rolling n-bar window:
    Var(q-bar log return) / (q * Var(1-bar log return)).
    ~1 random walk, >1 trending (positive autocorrelation), <1 mean-reverting."""
    r = np.log(close).diff()
    rq = r.rolling(q).sum()
    v1 = r.rolling(n).var()
    vq = rq.rolling(n).var()
    return (vq / (q * v1.replace(0, np.nan))).clip(0, 5)


def donchian(df: pd.DataFrame, n: int):
    upper = df["High"].rolling(n).max()
    lower = df["Low"].rolling(n).min()
    return upper, lower, (upper + lower) / 2


def keltner(df: pd.DataFrame, n: int, k: float, atr_n: int):
    mid = ema(df["Close"], n)
    width = k * atr(df, atr_n)
    return mid + width, mid - width, mid


def bollinger(df: pd.DataFrame, n: int, k: float):
    mid = sma(df["Close"], n)
    width = k * df["Close"].rolling(n).std(ddof=0)
    return mid + width, mid - width, mid


def channel(df: pd.DataFrame, kind: str, n: int, k: float, atr_n: int):
    if kind == "donchian":
        return donchian(df, n)
    if kind == "keltner":
        return keltner(df, n, k, atr_n)
    if kind == "bollinger":
        return bollinger(df, n, k)
    raise ValueError(f"unknown channel_type {kind}")


# Regime "trendiness" registry. Every entry is oriented so that a HIGHER
# value means MORE trending; `threshold` is the default split between
# trend and range, `thresholds` the walk-forward search grid.
REGIME_INDICATORS = {
    "er":   dict(fn=lambda df, n: efficiency_ratio(df["Close"], n), n=20, threshold=0.35, thresholds=[0.25, 0.35, 0.45]),
    "adx":  dict(fn=lambda df, n: adx(df, n), n=14, threshold=25.0, thresholds=[20.0, 25.0, 30.0]),
    "cti":  dict(fn=lambda df, n: cti(df["Close"], n).abs(), n=20, threshold=0.5, thresholds=[0.4, 0.5, 0.6]),
    "chop": dict(fn=lambda df, n: 100.0 - choppiness(df, n), n=14, threshold=50.0, thresholds=[38.2, 50.0, 61.8]),
    "vr":   dict(fn=lambda df, n: variance_ratio(df["Close"], n), n=60, threshold=1.0, thresholds=[0.9, 1.0, 1.1]),
}


# --------------------------------------------------------------------------
# Template / config
# --------------------------------------------------------------------------

DIRECTION_LOGICS = ["trend", "countertrend"]
CHANNEL_TYPES = ["donchian", "keltner", "bollinger"]
ENTRY_STYLES = ["stop", "close_confirm", "pullback"]
EXIT_STYLES = ["channel", "atr_trail", "target_stop", "time_stop"]
REGIME_INDICATOR_NAMES = list(REGIME_INDICATORS)
REGIME_FILTERS = ["none", "trend_only", "range_only"]
VOL_FILTERS = [False, True]
BIAS_FILTERS = ["none", "sma"]


@dataclass
class StrategyTemplate:
    """A structurally distinct strategy: the categorical switches, plus
    default numeric params (these defaults get overridden per-window by
    walk-forward optimization -- see walkforward.py)."""

    name: str
    direction_logic: str = "trend"
    channel_type: str = "donchian"
    entry_style: str = "stop"
    exit_style: str = "channel"
    regime_indicator: str = "er"
    regime_filter: str = "none"
    vol_filter: bool = False
    bias_filter: str = "none"

    # numeric params / defaults (subject to WFA tuning)
    n_entry: int = 40
    n_exit: int = 20
    atr_n: int = 20
    channel_k: float = 2.0
    atr_mult_stop: float = 3.0
    atr_mult_target: float = 4.0
    atr_mult_trail: float = 3.0
    pullback_atr_mult: float = 0.5
    pullback_valid_bars: int = 3
    max_hold_bars: int = 20
    regime_n: int = 0            # 0 -> use the indicator's default lookback
    regime_threshold: float = np.nan   # NaN -> indicator default
    vol_lookback: int = 100
    vol_low_pct: float = 0.10
    vol_high_pct: float = 0.90
    bias_n: int = 200
    risk_pct: float = 0.01
    max_leverage: float = 2.0
    cost_bps: float = 5.0        # per side, commission + slippage, in basis points of notional

    def with_params(self, **kwargs) -> "StrategyTemplate":
        d = asdict(self)
        d.update(kwargs)
        return StrategyTemplate(**d)

    def switches(self) -> dict:
        return dict(
            direction_logic=self.direction_logic,
            channel_type=self.channel_type,
            entry_style=self.entry_style,
            exit_style=self.exit_style,
            regime_indicator=self.regime_indicator if self.regime_filter != "none" else "-",
            regime_filter=self.regime_filter,
            vol_filter=self.vol_filter,
            bias_filter=self.bias_filter,
        )

    def validate(self):
        assert self.direction_logic in DIRECTION_LOGICS
        assert self.channel_type in CHANNEL_TYPES
        assert self.entry_style in ENTRY_STYLES
        assert self.exit_style in EXIT_STYLES
        assert self.regime_indicator in REGIME_INDICATORS
        assert self.regime_filter in REGIME_FILTERS
        assert self.bias_filter in BIAS_FILTERS


# --------------------------------------------------------------------------
# Indicator cache
# --------------------------------------------------------------------------
# Walk-forward re-runs the same template on the same slice for every grid
# point, so most indicator arrays are recomputed dozens of times. Keyed on
# the slice's shape, date range, a few sampled closes and the indicator
# spec, this cache removes that redundancy (each worker process has its own).

_IND_CACHE: dict = {}
_IND_CACHE_MAX = 4096


def _df_key(df: pd.DataFrame, close=None) -> tuple:
    c = df["Close"].to_numpy() if close is None else close
    n = len(c)
    return (n, df.index[0], df.index[-1], float(c[0]), float(c[n // 3]), float(c[(2 * n) // 3]), float(c[-1]))


def _cached(df: pd.DataFrame, spec: tuple, fn, dfkey=None):
    key = (_df_key(df) if dfkey is None else dfkey, spec)
    arr = _IND_CACHE.get(key)
    if arr is None:
        if len(_IND_CACHE) >= _IND_CACHE_MAX:
            _IND_CACHE.clear()
        arr = fn()
        _IND_CACHE[key] = arr
    return arr


def _to_arr(x) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(x, dtype=np.float64))


def _channel_arrays(df, kind, n, k, atr_n, dfkey=None):
    def build():
        up, lo, mid = channel(df, kind, n, k, atr_n)
        return (_to_arr(up), _to_arr(lo), _to_arr(mid))
    return _cached(df, ("channel", kind, n, k if kind != "donchian" else 0.0, atr_n if kind == "keltner" else 0), build, dfkey)


def _compute_indicators(df: pd.DataFrame, tpl: StrategyTemplate, dfkey=None) -> dict:
    """All indicator arrays needed by `tpl`, plus a boolean `ready` array:
    ready[i] is True when every indicator used by this template is fully
    formed on bar i. Only indicators the template actually uses count
    toward the warm-up, so a template without a vol filter does not waste
    100 bars of every training window waiting for one."""
    n = len(df)
    ind = {}
    ready = np.ones(n, dtype=np.bool_)
    dfkey = _df_key(df) if dfkey is None else dfkey

    def use(name, arr):
        ind[name] = arr
        nonlocal ready
        ready = ready & ~np.isnan(arr)

    up, lo, _ = _channel_arrays(df, tpl.channel_type, tpl.n_entry, tpl.channel_k, tpl.atr_n, dfkey)
    use("upper", up)
    use("lower", lo)
    use("atr", _cached(df, ("atr", tpl.atr_n), lambda: _to_arr(atr(df, tpl.atr_n)), dfkey))

    if tpl.exit_style == "channel":
        upx, lox, midx = _channel_arrays(df, tpl.channel_type, tpl.n_exit, tpl.channel_k, tpl.atr_n, dfkey)
        use("upper_x", upx)
        use("lower_x", lox)
        use("mid_x", midx)

    if tpl.regime_filter != "none":
        spec = REGIME_INDICATORS[tpl.regime_indicator]
        rn = tpl.regime_n if tpl.regime_n > 0 else spec["n"]
        use("regime", _cached(df, ("regime", tpl.regime_indicator, rn), lambda: _to_arr(spec["fn"](df, rn)), dfkey))
        ind["regime_threshold"] = (
            tpl.regime_threshold if not np.isnan(tpl.regime_threshold) else spec["threshold"]
        )

    if tpl.vol_filter:
        use("vol_rank", _cached(df, ("vol_rank", tpl.atr_n, tpl.vol_lookback),
                                lambda: _to_arr(atr(df, tpl.atr_n).rolling(tpl.vol_lookback).rank(pct=True)), dfkey))

    if tpl.bias_filter == "sma":
        use("bias", _cached(df, ("sma", tpl.bias_n), lambda: _to_arr(sma(df["Close"], tpl.bias_n)), dfkey))

    ind["ready"] = ready
    return ind


# --------------------------------------------------------------------------
# Backtest engine (single asset, bar-by-bar for correct stateful exits)
# --------------------------------------------------------------------------

ENTRY_CODES = {"stop": 0, "close_confirm": 1, "pullback": 2}
EXIT_CODES = {"channel": 0, "atr_trail": 1, "target_stop": 2, "time_stop": 3}
REGIME_CODES = {"none": 0, "trend_only": 1, "range_only": 2}
REASONS = ["stop", "channel", "midline", "target", "time", "stop_same_bar"]


def _bar_loop(open_, high, low, close, ready, upper, lower, atr_v,
              upper_x, lower_x, mid_x, regime, vol_rank, bias,
              is_trend, entry_style, exit_style, regime_mode, regime_thr,
              has_vol, vol_low, vol_high, has_bias,
              atr_mult_stop, atr_mult_target, atr_mult_trail, pullback_atr_mult,
              pullback_valid_bars, max_hold_bars, risk_pct, max_leverage, cost_rate,
              initial_equity):
    """The bar loop. Plain numpy code so numba can compile it unchanged;
    the pure-Python version is used when numba is not installed.

    Returns equity, entries and the closed-trade columns
    (entry_bar, exit_bar, side, entry_px, exit_px, shares, pnl, cost, reason, count)."""
    n = len(close)
    equity = np.empty(n)
    entries = np.zeros(n, dtype=np.int8)
    t_entry = np.empty(n, dtype=np.int64)
    t_exit = np.empty(n, dtype=np.int64)
    t_side = np.empty(n, dtype=np.int64)
    t_reason = np.empty(n, dtype=np.int64)
    t_entry_px = np.empty(n)
    t_exit_px = np.empty(n)
    t_shares = np.empty(n)
    t_pnl = np.empty(n)
    t_cost = np.empty(n)
    n_trades = 0

    cash = initial_equity
    position = 0
    shares = 0.0
    entry_price = 0.0
    stop_price = 0.0
    target_price = 0.0
    trail_extreme = 0.0
    entry_bar = -1
    entry_cost = 0.0
    pend_active = False
    pend_side = 0
    pend_level = 0.0
    pend_expires = 0
    last_a = 0.0

    equity[0] = initial_equity
    for i in range(1, n):
        # `a_ok`: every indicator this template uses is fully formed on bar i-1
        # and the ATR is usable. New business (filters, orders, entries) needs
        # that; an OPEN POSITION is managed on every bar regardless, otherwise a
        # flat patch that drives the ATR to zero would suspend its stop just when
        # the gap through it arrives. `last_a` is the most recent usable ATR and
        # is always > 0 while a position is open (entering required a_ok).
        a_ok = ready[i - 1] and atr_v[i - 1] > 0.0
        if a_ok:
            a = atr_v[i - 1]
            last_a = a
        else:
            a = last_a

        # ---- manage open position: exits (checked intrabar on bar i) ----
        if position != 0:
            exit_price = 0.0
            reason = -1
            side = position
            stop_level = stop_price  # hard stop is always active

            if exit_style == 1:
                # trailing level from the extreme up to bar i-1 (no intrabar look-ahead)
                trail_stop = trail_extreme - side * atr_mult_trail * a
                if side == 1:
                    stop_level = max(stop_level, trail_stop)
                else:
                    stop_level = min(stop_level, trail_stop)

            if side == 1 and low[i] <= stop_level:
                exit_price = min(open_[i], stop_level)
                reason = 0
            elif side == -1 and high[i] >= stop_level:
                exit_price = max(open_[i], stop_level)
                reason = 0

            if reason < 0:
                if exit_style == 0:
                    if is_trend:
                        if side == 1 and low[i] <= lower_x[i - 1]:
                            exit_price = min(open_[i], lower_x[i - 1])
                            reason = 1
                        elif side == -1 and high[i] >= upper_x[i - 1]:
                            exit_price = max(open_[i], upper_x[i - 1])
                            reason = 1
                    else:
                        if side == 1 and high[i] >= mid_x[i - 1]:
                            exit_price = max(open_[i], mid_x[i - 1])
                            reason = 2
                        elif side == -1 and low[i] <= mid_x[i - 1]:
                            exit_price = min(open_[i], mid_x[i - 1])
                            reason = 2
                elif exit_style == 2:
                    if side == 1 and high[i] >= target_price:
                        exit_price = max(open_[i], target_price)
                        reason = 3
                    elif side == -1 and low[i] <= target_price:
                        exit_price = min(open_[i], target_price)
                        reason = 3
                elif exit_style == 3:
                    if i - entry_bar >= max_hold_bars:
                        exit_price = open_[i]
                        reason = 4

            if reason >= 0:
                gross = position * shares * (exit_price - entry_price)
                xcost = cost_rate * shares * exit_price
                cash += gross - xcost
                t_entry[n_trades] = entry_bar
                t_exit[n_trades] = i
                t_side[n_trades] = position
                t_entry_px[n_trades] = entry_price
                t_exit_px[n_trades] = exit_price
                t_shares[n_trades] = shares
                t_pnl[n_trades] = gross - xcost - entry_cost
                t_cost[n_trades] = entry_cost + xcost
                t_reason[n_trades] = reason
                n_trades += 1
                position = 0
                shares = 0.0
                equity[i] = cash
                continue  # no re-entry on the exit bar

            if exit_style == 1:
                if side == 1:
                    trail_extreme = max(trail_extreme, high[i])
                else:
                    trail_extreme = min(trail_extreme, low[i])

        # ---- no usable indicators: manage what is open, start nothing new ----
        if not a_ok:
            pend_active = False  # don't leave a stale resting order behind
            equity[i] = cash + (position * shares * (close[i] - entry_price) if position != 0 else 0.0)
            continue

        # ---- filters (previous bar's fully-formed values) ----
        can_enter = True
        if regime_mode == 1:
            can_enter = regime[i - 1] >= regime_thr
        elif regime_mode == 2:
            can_enter = regime[i - 1] < regime_thr
        if can_enter and has_vol:
            can_enter = vol_low <= vol_rank[i - 1] <= vol_high

        long_ok = True
        short_ok = True
        if has_bias:
            long_ok = close[i - 1] > bias[i - 1]
            short_ok = close[i - 1] < bias[i - 1]

        fill_side = 0
        fill_px = 0.0

        # ---- pending pullback order ----
        if position == 0 and pend_active:
            if pend_side == 1 and low[i] <= pend_level:
                fill_side = 1
                fill_px = min(open_[i], pend_level)
                pend_active = False
            elif pend_side == -1 and high[i] >= pend_level:
                fill_side = -1
                fill_px = max(open_[i], pend_level)
                pend_active = False
            elif i >= pend_expires:
                pend_active = False

        # ---- new entry signals (based on previous bar's channel) ----
        if position == 0 and fill_side == 0 and not pend_active and can_enter:
            if entry_style == 1:
                ok = i >= 2 and ready[i - 2]
                broke_up = ok and close[i - 1] > upper[i - 2]
                broke_down = ok and close[i - 1] < lower[i - 2]
                level_up = open_[i]
                level_down = open_[i]
            else:
                broke_up = high[i] >= upper[i - 1]
                broke_down = low[i] <= lower[i - 1]
                level_up = upper[i - 1]
                level_down = lower[i - 1]

            side = 0
            level = 0.0
            if is_trend:
                if broke_up and long_ok:
                    side = 1
                    level = level_up
                elif broke_down and short_ok:
                    side = -1
                    level = level_down
            else:
                if broke_down and long_ok:
                    side = 1
                    level = level_down
                elif broke_up and short_ok:
                    side = -1
                    level = level_up

            if side != 0:
                if entry_style == 1:
                    fill_side = side
                    fill_px = open_[i]
                elif entry_style == 0:
                    if is_trend:
                        fill_px = max(open_[i], level) if side == 1 else min(open_[i], level)
                    else:
                        fill_px = min(open_[i], level) if side == 1 else max(open_[i], level)
                    fill_side = side
                else:
                    pend_active = True
                    pend_side = side
                    pend_level = level - side * pullback_atr_mult * a
                    pend_expires = i + pullback_valid_bars

        # ---- open the position ----
        entered = False
        if fill_side != 0:
            stop_dist = atr_mult_stop * a
            qty = cash * risk_pct / stop_dist
            qty = min(qty, max_leverage * cash / fill_px)
            if qty > 0:
                entry_cost = cost_rate * qty * fill_px
                cash -= entry_cost
                position = fill_side
                shares = qty
                entry_price = fill_px
                stop_price = fill_px - fill_side * stop_dist
                target_price = fill_px + fill_side * atr_mult_target * a
                trail_extreme = fill_px
                entry_bar = i
                entries[i] = 1
                entered = True

        # ---- conservative same-bar stop check on the entry bar ----
        if entered:
            hit = (position == 1 and low[i] <= stop_price) or (position == -1 and high[i] >= stop_price)
            if hit:
                gross = position * shares * (stop_price - entry_price)
                xcost = cost_rate * shares * stop_price
                cash += gross - xcost
                t_entry[n_trades] = entry_bar
                t_exit[n_trades] = i
                t_side[n_trades] = position
                t_entry_px[n_trades] = entry_price
                t_exit_px[n_trades] = stop_price
                t_shares[n_trades] = shares
                t_pnl[n_trades] = gross - xcost - entry_cost
                t_cost[n_trades] = entry_cost + xcost
                t_reason[n_trades] = 5
                n_trades += 1
                position = 0
                shares = 0.0

        # ---- the entry bar counts toward the chandelier's anchor ----
        # (bar i is complete when bar i+1 is traded, so this is not look-ahead;
        # done after the same-bar stop so the stop still wins on the entry bar)
        if entered and position != 0 and exit_style == 1:
            if position == 1:
                trail_extreme = max(trail_extreme, high[i])
            else:
                trail_extreme = min(trail_extreme, low[i])

        # ---- mark to market at the close of bar i ----
        equity[i] = cash + (position * shares * (close[i] - entry_price) if position != 0 else 0.0)

    return (equity, entries, t_entry, t_exit, t_side, t_entry_px, t_exit_px, t_shares, t_pnl, t_cost, t_reason, n_trades)


try:  # compile the loop once per process; falls back to plain Python without numba
    from numba import njit as _njit
    _bar_loop_fast = _njit(cache=True, nogil=True)(_bar_loop)
    HAVE_NUMBA = True
except Exception:  # pragma: no cover
    _bar_loop_fast = _bar_loop
    HAVE_NUMBA = False


def backtest(df: pd.DataFrame, tpl: StrategyTemplate, initial_equity: float = 100_000.0) -> dict:
    """Run `tpl` over `df` (must have Open/High/Low/Close). Returns a dict:
        equity   : pd.Series of end-of-bar equity, indexed like df
        returns  : pd.Series of per-bar simple returns of equity
        entries  : np.ndarray (1 on bars where a new trade was opened)
        trades   : list of trade dicts
        stats    : summary performance stats
    """
    n = len(df)
    close = _to_arr(df["Close"])
    open_ = _to_arr(df["Open"])
    high = _to_arr(df["High"])
    low = _to_arr(df["Low"])

    ind = _compute_indicators(df, tpl, _df_key(df, close))
    zeros = np.zeros(n)
    out = _bar_loop_fast(
        open_, high, low, close, ind["ready"], ind["upper"], ind["lower"], ind["atr"],
        ind.get("upper_x", zeros), ind.get("lower_x", zeros), ind.get("mid_x", zeros),
        ind.get("regime", zeros), ind.get("vol_rank", zeros), ind.get("bias", zeros),
        tpl.direction_logic == "trend", ENTRY_CODES[tpl.entry_style], EXIT_CODES[tpl.exit_style],
        REGIME_CODES[tpl.regime_filter], float(ind.get("regime_threshold", 0.0)),
        bool(tpl.vol_filter), float(tpl.vol_low_pct), float(tpl.vol_high_pct), tpl.bias_filter == "sma",
        float(tpl.atr_mult_stop), float(tpl.atr_mult_target), float(tpl.atr_mult_trail), float(tpl.pullback_atr_mult),
        int(tpl.pullback_valid_bars), int(tpl.max_hold_bars), float(tpl.risk_pct), float(tpl.max_leverage),
        tpl.cost_bps / 1e4, float(initial_equity),
    )
    (equity, entries, t_entry, t_exit, t_side, t_entry_px, t_exit_px, t_shares, t_pnl, t_cost, t_reason, n_trades) = out

    idx = df.index
    idx_arr = idx.to_numpy()
    trades = [
        dict(entry_date=pd.Timestamp(idx_arr[t_entry[k]]), side=int(t_side[k]), entry_price=float(t_entry_px[k]),
             shares=float(t_shares[k]), cost=float(t_cost[k]), exit_date=pd.Timestamp(idx_arr[t_exit[k]]),
             exit_price=float(t_exit_px[k]), reason=REASONS[t_reason[k]], pnl=float(t_pnl[k]),
             bars_held=int(t_exit[k] - t_entry[k]))
        for k in range(n_trades)
    ]
    rets = np.zeros(n)
    rets[1:] = equity[1:] / equity[:-1] - 1.0
    equity_s = pd.Series(equity, index=idx)
    returns = pd.Series(rets, index=idx)
    stats = _performance_stats(equity, trades, initial_equity, rets, t_pnl[:n_trades],
                               (t_exit[:n_trades] - t_entry[:n_trades]).astype(float))
    return {"equity": equity_s, "returns": returns, "entries": entries, "trades": trades, "stats": stats}


def annualized_sharpe(rets: pd.Series | np.ndarray, periods_per_year: int = PERIODS_PER_YEAR) -> float:
    r = np.asarray(rets, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) < 2:
        return 0.0
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(periods_per_year)) if sd > 0 else 0.0


def _performance_stats(equity, trades: list, initial_equity: float, rets=None, pnls=None, bars_held=None) -> dict:
    """Summary stats from an equity path. Accepts a numpy array or a Series;
    `rets`, `pnls` and `bars_held` may be passed to skip recomputation."""
    eq = np.asarray(equity, dtype=float)
    n_bars = len(eq)
    if rets is None:
        rets = np.zeros(n_bars)
        rets[1:] = eq[1:] / eq[:-1] - 1.0
    if pnls is None:
        pnls = np.array([t.get("pnl", 0.0) for t in trades], dtype=float)
    if bars_held is None:
        bars_held = np.array([t.get("bars_held", 0) for t in trades], dtype=float)
    n_years = max(n_bars / PERIODS_PER_YEAR, 1e-6)
    final = float(eq[-1])
    total_return = final / initial_equity - 1
    cagr = (final / initial_equity) ** (1 / n_years) - 1 if final > 0 else -1.0
    sharpe = annualized_sharpe(rets[1:])
    running_max = np.maximum.accumulate(eq)
    max_dd = float((eq / running_max - 1).min())
    wins = pnls[pnls > 0].sum()
    losses = -pnls[pnls < 0].sum()
    profit_factor = wins / losses if losses > 0 else (np.inf if wins > 0 else 0.0)
    n_tr = len(pnls)
    return dict(
        total_return=float(total_return),
        cagr=float(cagr),
        sharpe=float(sharpe),
        max_drawdown=max_dd,
        n_trades=n_tr,
        win_rate=float((pnls > 0).mean()) if n_tr else 0.0,
        profit_factor=float(profit_factor),
        avg_bars_held=float(bars_held.mean()) if n_tr else 0.0,
        exposure=float(bars_held.sum() / n_bars) if n_bars else 0.0,
        n_bars=n_bars,
    )
