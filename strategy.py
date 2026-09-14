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
# Backtest engine (single asset, bar-by-bar for correct stateful exits)
# --------------------------------------------------------------------------

def _compute_indicators(df: pd.DataFrame, tpl: StrategyTemplate) -> dict:
    """All indicator arrays needed by `tpl`, plus a boolean `ready` array:
    ready[i] is True when every indicator used by this template is fully
    formed on bar i. Only indicators the template actually uses count
    toward the warm-up, so a template without a vol filter does not waste
    100 bars of every training window waiting for one."""
    ind = {}
    ready = np.ones(len(df), dtype=bool)

    def use(name, series):
        arr = np.asarray(series, dtype=float)
        ind[name] = arr
        nonlocal ready
        ready = ready & ~np.isnan(arr)

    up, lo, mid = channel(df, tpl.channel_type, tpl.n_entry, tpl.channel_k, tpl.atr_n)
    use("upper", up)
    use("lower", lo)
    use("atr", atr(df, tpl.atr_n))

    if tpl.exit_style == "channel":
        upx, lox, midx = channel(df, tpl.channel_type, tpl.n_exit, tpl.channel_k, tpl.atr_n)
        use("upper_x", upx)
        use("lower_x", lox)
        use("mid_x", midx)

    if tpl.regime_filter != "none":
        spec = REGIME_INDICATORS[tpl.regime_indicator]
        n = tpl.regime_n if tpl.regime_n > 0 else spec["n"]
        use("regime", spec["fn"](df, n))
        ind["regime_threshold"] = (
            tpl.regime_threshold if not np.isnan(tpl.regime_threshold) else spec["threshold"]
        )

    if tpl.vol_filter:
        a = atr(df, tpl.atr_n)
        use("vol_rank", a.rolling(tpl.vol_lookback).rank(pct=True))

    if tpl.bias_filter == "sma":
        use("bias", sma(df["Close"], tpl.bias_n))

    ind["ready"] = ready
    return ind


def backtest(df: pd.DataFrame, tpl: StrategyTemplate, initial_equity: float = 100_000.0) -> dict:
    """Run `tpl` over `df` (must have Open/High/Low/Close). Returns a dict:
        equity   : pd.Series of end-of-bar equity, indexed like df
        returns  : pd.Series of per-bar simple returns of equity
        entries  : np.ndarray (1 on bars where a new trade was opened)
        trades   : list of trade dicts
        stats    : summary performance stats
    """
    n = len(df)
    close = df["Close"].to_numpy(dtype=float)
    open_ = df["Open"].to_numpy(dtype=float)
    high = df["High"].to_numpy(dtype=float)
    low = df["Low"].to_numpy(dtype=float)

    ind = _compute_indicators(df, tpl)
    ready = ind["ready"]
    upper_v, lower_v, atr_v = ind["upper"], ind["lower"], ind["atr"]
    upper_xv = ind.get("upper_x")
    lower_xv = ind.get("lower_x")
    mid_xv = ind.get("mid_x")
    regime_v = ind.get("regime")
    regime_thr = ind.get("regime_threshold")
    vol_v = ind.get("vol_rank")
    bias_v = ind.get("bias")

    cost_rate = tpl.cost_bps / 1e4
    is_trend = tpl.direction_logic == "trend"

    equity = np.full(n, np.nan)
    entries = np.zeros(n, dtype=np.int8)
    cash = initial_equity

    position = 0        # -1, 0, +1
    shares = 0.0
    entry_price = np.nan
    stop_price = np.nan
    target_price = np.nan
    trail_extreme = np.nan
    entry_bar = -1
    pending_order = None  # {'side', 'level', 'expires'} for pullback entries

    trades = []
    open_trade = None

    def open_position(i, side, fill):
        nonlocal position, shares, entry_price, stop_price, target_price, trail_extreme, entry_bar, open_trade, cash
        a = atr_v[i - 1]
        stop_dist = tpl.atr_mult_stop * a
        risk_amt = cash * tpl.risk_pct
        qty = risk_amt / stop_dist
        qty = min(qty, tpl.max_leverage * cash / fill)   # notional cap
        if qty <= 0:
            return False
        cost = cost_rate * qty * fill
        cash -= cost
        position = side
        shares = qty
        entry_price = fill
        stop_price = fill - side * stop_dist
        target_price = fill + side * tpl.atr_mult_target * a
        trail_extreme = fill
        entry_bar = i
        entries[i] = 1
        open_trade = dict(entry_date=df.index[i], side=side, entry_price=fill, shares=qty, cost=cost)
        return True

    def close_position(i, exit_price, reason):
        nonlocal position, shares, entry_price, stop_price, target_price, trail_extreme, open_trade, cash
        gross = position * shares * (exit_price - entry_price)
        cost = cost_rate * shares * exit_price
        cash += gross - cost
        open_trade.update(
            exit_date=df.index[i], exit_price=exit_price, reason=reason,
            pnl=gross - cost - open_trade["cost"], bars_held=i - entry_bar,
        )
        open_trade["cost"] += cost
        trades.append(open_trade)
        open_trade = None
        position = 0
        shares = 0.0
        entry_price = stop_price = target_price = trail_extreme = np.nan

    equity[0] = initial_equity
    for i in range(1, n):
        if not ready[i - 1]:
            equity[i] = cash + (position * shares * (close[i] - entry_price) if position else 0.0)
            continue

        a = atr_v[i - 1]
        if not (a > 0):
            equity[i] = cash + (position * shares * (close[i] - entry_price) if position else 0.0)
            continue

        # ---- manage open position: exits (checked intrabar on bar i) ----
        if position != 0:
            exit_price, reason = None, None
            side = position
            # hard stop is always active (risk control)
            stop_level = stop_price

            if tpl.exit_style == "atr_trail":
                # trailing level is based on the extreme up to bar i-1 (no intrabar look-ahead)
                trail_stop = trail_extreme - side * tpl.atr_mult_trail * a
                stop_level = max(stop_level, trail_stop) if side == 1 else min(stop_level, trail_stop)

            if side == 1 and low[i] <= stop_level:
                exit_price, reason = min(open_[i], stop_level), "stop"
            elif side == -1 and high[i] >= stop_level:
                exit_price, reason = max(open_[i], stop_level), "stop"

            if exit_price is None:
                if tpl.exit_style == "channel":
                    if is_trend:   # exit on the opposite side of the exit channel
                        if side == 1 and low[i] <= lower_xv[i - 1]:
                            exit_price, reason = min(open_[i], lower_xv[i - 1]), "channel"
                        elif side == -1 and high[i] >= upper_xv[i - 1]:
                            exit_price, reason = max(open_[i], upper_xv[i - 1]), "channel"
                    else:          # countertrend: take profit at the channel midline
                        if side == 1 and high[i] >= mid_xv[i - 1]:
                            exit_price, reason = max(open_[i], mid_xv[i - 1]), "midline"
                        elif side == -1 and low[i] <= mid_xv[i - 1]:
                            exit_price, reason = min(open_[i], mid_xv[i - 1]), "midline"
                elif tpl.exit_style == "target_stop":
                    if side == 1 and high[i] >= target_price:
                        exit_price, reason = max(open_[i], target_price), "target"
                    elif side == -1 and low[i] <= target_price:
                        exit_price, reason = min(open_[i], target_price), "target"
                elif tpl.exit_style == "time_stop":
                    if i - entry_bar >= tpl.max_hold_bars:
                        exit_price, reason = open_[i], "time"

            if exit_price is not None:
                close_position(i, exit_price, reason)
                equity[i] = cash
                continue  # no re-entry on the exit bar

            # update trailing extreme AFTER the exit check, with this bar's data
            if tpl.exit_style == "atr_trail":
                trail_extreme = max(trail_extreme, high[i]) if side == 1 else min(trail_extreme, low[i])

        # ---- filters (previous bar's fully-formed values) ----
        can_enter = True
        if regime_v is not None:
            if tpl.regime_filter == "trend_only":
                can_enter = regime_v[i - 1] >= regime_thr
            else:
                can_enter = regime_v[i - 1] < regime_thr
        if can_enter and vol_v is not None:
            can_enter = tpl.vol_low_pct <= vol_v[i - 1] <= tpl.vol_high_pct

        long_ok = short_ok = True
        if bias_v is not None:
            long_ok = close[i - 1] > bias_v[i - 1]
            short_ok = close[i - 1] < bias_v[i - 1]

        entered = False

        # ---- pending pullback order ----
        if position == 0 and pending_order is not None:
            side, level = pending_order["side"], pending_order["level"]
            filled = None
            if side == 1 and low[i] <= level:
                filled = min(open_[i], level)
            elif side == -1 and high[i] >= level:
                filled = max(open_[i], level)
            if filled is not None:
                entered = open_position(i, side, filled)
                pending_order = None
            elif i >= pending_order["expires"]:
                pending_order = None

        # ---- new entry signals (based on previous bar's channel) ----
        if position == 0 and pending_order is None and can_enter:
            if tpl.entry_style == "close_confirm":
                # a CLOSE beyond the channel that existed before that close
                ok = i >= 2 and ready[i - 2]
                broke_up = ok and close[i - 1] > upper_v[i - 2]
                broke_down = ok and close[i - 1] < lower_v[i - 2]
                level_up, level_down = open_[i], open_[i]
            else:
                broke_up = high[i] >= upper_v[i - 1]
                broke_down = low[i] <= lower_v[i - 1]
                level_up, level_down = upper_v[i - 1], lower_v[i - 1]

            # The channel that was actually broken determines the fill level.
            side, level = 0, None
            if is_trend:
                if broke_up and long_ok:
                    side, level = 1, level_up
                elif broke_down and short_ok:
                    side, level = -1, level_down
            else:
                if broke_down and long_ok:
                    side, level = 1, level_down
                elif broke_up and short_ok:
                    side, level = -1, level_up

            if side != 0:
                if tpl.entry_style == "close_confirm":
                    entered = open_position(i, side, open_[i])
                elif tpl.entry_style == "stop":
                    # trend: stop orders (fill at open if gapped through);
                    # countertrend: limit orders (fill at open if better)
                    if is_trend:
                        fill = max(open_[i], level) if side == 1 else min(open_[i], level)
                    else:
                        fill = min(open_[i], level) if side == 1 else max(open_[i], level)
                    entered = open_position(i, side, fill)
                else:  # pullback: arm a limit order for the following bars
                    pb_level = level - side * tpl.pullback_atr_mult * a
                    pending_order = {"side": side, "level": pb_level, "expires": i + tpl.pullback_valid_bars}

        # ---- conservative same-bar stop check on the entry bar ----
        if entered and position != 0:
            if position == 1 and low[i] <= stop_price:
                close_position(i, stop_price, "stop_same_bar")
            elif position == -1 and high[i] >= stop_price:
                close_position(i, stop_price, "stop_same_bar")

        # ---- mark to market at the close of bar i ----
        equity[i] = cash + (position * shares * (close[i] - entry_price) if position else 0.0)

    equity_s = pd.Series(equity, index=df.index).ffill()
    returns = equity_s.pct_change().fillna(0.0)
    stats = _performance_stats(equity_s, trades, initial_equity, entries)
    return {"equity": equity_s, "returns": returns, "entries": entries, "trades": trades, "stats": stats}


def annualized_sharpe(rets: pd.Series | np.ndarray, periods_per_year: int = PERIODS_PER_YEAR) -> float:
    r = np.asarray(rets, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) < 2:
        return 0.0
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(periods_per_year)) if sd > 0 else 0.0


def _performance_stats(equity: pd.Series, trades: list, initial_equity: float, entries=None) -> dict:
    rets = equity.pct_change().dropna()
    n_bars = len(equity)
    n_years = max(n_bars / PERIODS_PER_YEAR, 1e-6)
    final = float(equity.iloc[-1])
    total_return = final / initial_equity - 1
    cagr = (final / initial_equity) ** (1 / n_years) - 1 if final > 0 else -1.0
    sharpe = annualized_sharpe(rets)
    running_max = equity.cummax()
    max_dd = float((equity / running_max - 1).min())
    pnls = np.array([t.get("pnl", 0.0) for t in trades], dtype=float)
    wins = pnls[pnls > 0].sum()
    losses = -pnls[pnls < 0].sum()
    profit_factor = wins / losses if losses > 0 else (np.inf if wins > 0 else 0.0)
    bars_held = [t.get("bars_held", 0) for t in trades]
    return dict(
        total_return=float(total_return),
        cagr=float(cagr),
        sharpe=float(sharpe),
        max_drawdown=max_dd,
        n_trades=len(trades),
        win_rate=float((pnls > 0).mean()) if len(pnls) else 0.0,
        profit_factor=float(profit_factor),
        avg_bars_held=float(np.mean(bars_held)) if bars_held else 0.0,
        exposure=float(sum(bars_held) / n_bars) if n_bars else 0.0,
        n_bars=n_bars,
    )
