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
                    'learned'      -> the online learner (see 'hedge' below)
                                      scores a FOLLOW and a FADE expert per
                                      lookback and the side with more weight
                                      decides, bar by bar, whether the
                                      template follows or fades the break,
                                      and by how much it leads, how big the
                                      position is; a trade keeps the logic
                                      it was opened under. Works with any
                                      channel_type.

  channel_type    : 'donchian'  -> highest high / lowest low of n bars
                    'keltner'   -> EMA(n) +/- channel_k * ATR(atr_n)
                    'bollinger' -> SMA(n) +/- channel_k * stdev(n)
                    'hedge'     -> ONLINE-LEARNED channel: no lookback to
                                   optimize (n_entry / n_exit are unused). A
                                   fixed ladder of Donchian "experts"
                                   (HEDGE_LADDER) is weighted bar by bar by
                                   AdaHedge, a parameter-free exponential-
                                   weights (Hedge / follow-the-leader)
                                   learner, discounted over a ladder of
                                   lifetimes (HEDGE_HORIZONS) it also learns
                                   to pick from, scoring the last
                                   HEDGE_MEMORY bars net of cost_bps; the
                                   channel is the weight-averaged expert
                                   channel. Rewards are signed by
                                   direction_logic, so a countertrend
                                   template learns which lookback pays to
                                   FADE (anti-correlation).
                    'hedge_wide' -> the same learner over a WIDER fixed
                                   ladder (HEDGE_LADDERS['hedge_wide']): the
                                   Donchian rungs and a Keltner band (SMA
                                   +/- 2 ATR) at every rung. Still nothing
                                   to fit: the ladder is fixed in advance.
                                   On this ladder the experts are scored on
                                   the legs the template's `sides` lets it
                                   trade (a long-only fade expert is "buy
                                   new lows", not "buy new lows and short
                                   new highs") and every entry is sized by
                                   the learner's POSITION (hedge_position):
                                   the stance its weighted committee of
                                   experts holds, times the weight a
                                   committee-or-cash aggregation puts on
                                   trading at all.

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

  sides           : 'both' / 'long_only' / 'short_only' -> which side of the
                    market the template may take at all. On an asset with a
                    persistent drift the two sides are different strategies,
                    not mirror images: on SPY every short leg of every
                    symmetric template lost money over 2005-2026.

Numeric params (walk-forward optimized, see generator.param_grid_for):
  n_entry, n_exit, atr_n, channel_k, atr_mult_stop, atr_mult_target,
  atr_mult_trail, pullback_atr_mult, pullback_valid_bars, max_hold_bars,
  regime_n, regime_threshold, vol_lookback, vol_low_pct, vol_high_pct,
  bias_n, risk_pct, max_leverage, cost_bps.
  vol_target, vol_target_n are sizing settings like risk_pct: set once per
  run from the CLI, never tuned by the walk-forward. So are the instrument
  settings point_value (currency per 1.0 of price per unit: 1 for a share,
  1000 for a Brent lot), cost_per_unit (currency per unit per side, on top
  of cost_bps) and margin_per_unit (initial margin per unit; 0 = none).

Execution model (no look-ahead):
  * every decision on bar i uses indicator values fully formed on bar i-1
  * position size (in UNITS: shares or lots), fixed at entry and held to
    the exit: `risk_pct` of equity lost at the `atr_mult_stop` ATR stop,
    units = equity x risk_pct / (atr_mult_stop x ATR x point_value); or,
    when `vol_target` > 0, the units whose dollar volatility is the target:
    units = equity x (vol_target / sqrt(bars per year)) / (sigma x
    point_value), sigma the std of close-to-close DIFFERENCES (price
    points) over `vol_target_n` bars. Neither rule divides by a price, so
    both work unchanged on an instrument that trades at or below zero (a
    calendar spread), and adding a constant to every price changes
    nothing. A 'learned' direction scales either by the learner's
    conviction (0..1, see hedge_direction): on the plain ladder its net
    side weight in favour of the side taken, on a position-sized ladder
    ('hedge_wide') the magnitude of its committee's position, which also
    sizes a fixed direction there (hedge_active). Either way capped at
    `max_leverage` x equity of margin (units x margin_per_unit) when a
    margin is given, else of notional (units x point_value x |price|); the
    ATR stop is unchanged
  * stops/limits are filled intrabar at the level, or at the open if the
    open gapped through the level; a time exit is an order at the open, so
    it goes before any intrabar stop on its bar
  * every stop the exit rules run (hard ATR stop, chandelier, opposite
    channel) is already working on the entry bar: a bar that trades through
    it after the fill is a same-bar exit at its level
  * a 'pullback' limit rests `pullback_atr_mult` ATRs from the broken level
    on the side of the trade: back inside the channel when following the
    break, deeper beyond it when fading
  * an exit order the OPEN already trades through (a stop, or a target)
    fills at the open before anything intrabar; inside the bar the stop
    goes first. On the entry bar a target is taken when the bar's prices
    beyond the fill on its side certainly came after the fill (see _bar_loop)
  * costs per side: `cost_bps` of the notional (units x point_value x
    |price|) plus `cost_per_unit` x units; a spread has no notional, so it
    is costed per unit with cost_bps = 0. A contract roll (the data's `Roll`
    column) costs `roll_cost_per_unit` x units to whatever is held through it
  * P&L: side x units x point_value x (price change); equity is marked to
    market at the CLOSE of each bar, after all fills
  * ruin: an equity at or below zero closes the account at that close; a
    deficit (a loss beyond the cash) is kept, as a futures account owes it
"""

from __future__ import annotations
from dataclasses import dataclass, asdict, replace
import numpy as np
import pandas as pd

# Bars per year, used for every annualization (Sharpe, CAGR, WFE, DSR...).
# Do NOT import this by value: `from strategy import PERIODS_PER_YEAR` freezes it
# at import time and set_periods_per_year() can then no longer be honoured.
# Read it through periods_per_year() instead.
PERIODS_PER_YEAR = 252

# Regular-session bars per year for the intervals yfinance serves. US equity
# ETFs trade 6.5h a day, which Yahoo cuts into seven '1h' bars (the last one is
# a 30-minute stub), thirteen '30m' bars, and so on. A future on a ~23h
# session has more: pass --bars-per-day (periods_per_year_for_interval).
BARS_PER_YEAR = {
    "1mo": 12, "1wk": 52, "1d": 252,
    "1h": 252 * 7, "60m": 252 * 7, "90m": 252 * 5,
    "30m": 252 * 13, "15m": 252 * 26, "5m": 252 * 78, "1m": 252 * 390,
}


def periods_per_year() -> int:
    """Bars per year currently used for annualization."""
    return PERIODS_PER_YEAR


def set_periods_per_year(n: int) -> None:
    """Set the bar frequency for every annualized statistic in the project.

    Call this ONCE, before running anything, and in every worker process (the
    pool initializer in main.py does it). Annualizing hourly bars at 252 would
    understate every Sharpe ratio by about sqrt(7).
    """
    global PERIODS_PER_YEAR
    n = int(n)
    if n < 1:
        raise ValueError(f"periods_per_year must be >= 1, got {n}")
    PERIODS_PER_YEAR = n


INTRADAY_MINUTES = {"1h": 60, "60m": 60, "90m": 90, "30m": 30, "15m": 15, "5m": 5, "1m": 1}
TRADING_DAYS_PER_YEAR = 252


def periods_per_year_for_interval(interval: str, bars_per_day: float | None = None) -> int:
    """Bars per year for a yfinance interval string ('1d', '1h', '30m', ...).

    The intraday entries of BARS_PER_YEAR count a US equity session (6.5 h:
    seven '1h' bars a day). A future trading ~23 h a day has 23 '1h' bars a
    day, and annualizing them at 7 understates every Sharpe by sqrt(23/7):
    `bars_per_day` replaces the session's count (252 x bars_per_day a year).
    It is for intraday intervals only, and at most a whole day of them."""
    if interval not in BARS_PER_YEAR:
        raise ValueError(f"unknown interval {interval!r}; known: {', '.join(BARS_PER_YEAR)}")
    if bars_per_day is None:
        return BARS_PER_YEAR[interval]
    if interval not in INTRADAY_MINUTES:
        raise ValueError(f"bars_per_day applies to intraday intervals ({', '.join(INTRADAY_MINUTES)}), "
                         f"not {interval!r}: those are one bar per day, week or month by definition")
    most = 24 * 60 / INTRADAY_MINUTES[interval]
    if not 0 < bars_per_day <= most:
        raise ValueError(f"bars_per_day {bars_per_day:g} for {interval} bars: must be in (0, {most:g}] "
                         "(a day has 24 hours)")
    return max(1, int(round(TRADING_DAYS_PER_YEAR * bars_per_day)))


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


def variance_ratio(close: pd.Series, n: int, q: int = 5, log_returns: bool = False) -> pd.Series:
    """Lo-MacKinlay style variance ratio over a rolling n-bar window:
    Var(q-bar change) / (q * Var(1-bar change)).
    ~1 random walk, >1 trending (positive autocorrelation), <1 mean-reverting.
    On price DIFFERENCES by default: defined for a price at or below zero and
    unmoved when a constant is added to every price (a future or spread).
    `log_returns` takes the changes of log(price) instead, the classic form
    for a cash asset (`is_cash_asset`), whose price is always positive."""
    r = np.log(close).diff() if log_returns else close.diff()
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


def channel(df: pd.DataFrame, kind: str, n: int, k: float, atr_n: int,
            mode: str = "trend", role: str = "entry", cost_bps: float = 0.0, sides: str = "both",
            cost_pts: float = 0.0):
    if kind == "donchian":
        return donchian(df, n)
    if kind == "keltner":
        return keltner(df, n, k, atr_n)
    if kind == "bollinger":
        return bollinger(df, n, k)
    if kind in HEDGE_CHANNELS:
        return hedge_channel(df, atr_n, mode=mode, scale=HEDGE_EXIT_SCALE if role == "exit" else 1.0,
                             cost_bps=cost_bps, ladder=kind, sides=sides, cost_pts=cost_pts)
    raise ValueError(f"unknown channel_type {kind}")


# --------------------------------------------------------------------------
# Online-learned ("hedge") channel and learned direction
# --------------------------------------------------------------------------
# The Donchian / Bollinger / Keltner channels all need a lookback (and a
# width) that the walk-forward has to re-fit every window. The hedge
# channel replaces that fit with an ONLINE LEARNER from the prediction-
# with-expert-advice literature:
#
#   * experts  : Donchian channels with the fixed lookbacks HEDGE_LADDER
#                (a log-spaced ladder, not a tuned parameter). A trend
#                template scores FOLLOW experts, a countertrend one FADE
#                experts, a 'learned' direction both. The 'hedge_wide'
#                channel runs the same learner over a wider fixed ladder
#                (HEDGE_LADDERS, HedgeExpert): the Donchian rungs and a
#                Keltner band at every rung (a break of a band is a move of
#                k ATRs from the mean rather than a new n-bar extreme, so
#                the ladder spans shape as well as period). Anything fixed
#                in advance can be an expert; the mixture stays causal and
#                free of fitted parameters. On the wide ladder the experts
#                are scored on the legs the template's `sides` switch lets
#                it trade (_expert_stances); on the plain ladder both legs,
#                as a two-sided template trades them.
#   * loss     : each bar, expert e is scored on the ATR-normalised return
#                of the stance it implied on the previous bar (long after
#                a new n-bar high, short after a new n-bar low, for up to
#                n bars after that break, flat when there was none; a fade
#                expert takes the opposite side), net of the cost of the
#                sides it traded to reach its new stance (cost_bps per
#                side, in ATRs, the engine's own charge), so the learner
#                rewards the lookback whose edge survives its turnover.
#                Losses are in [0, 1].
#   * learner  : AdaHedge (de Rooij, van Erven, Grunwald, Koolen, JMLR
#                2014): exponential weights w_e ~ exp(-eta * deficit_e)
#                (deficit = cumulative loss behind the leader) whose
#                learning rate eta = ln(N) / (accumulated mixability gap)
#                starts at infinity, i.e. follow-the-leader, and shrinks
#                only as much as the data forces it to. Two changes to the
#                plain algorithm, both parameter-free:
#                - DISCOUNTING: losses and the gap decay with a lifetime H
#                  (gamma = 1 - 1/H). Plain AdaHedge's eta only ever
#                  falls; a discounted gap lets it rise again after a calm
#                  stretch, so eta reflects RECENT surprise. A discounted
#                  deficit is bounded by H times the expert's shortfall
#                  per bar, so no expert is ever written off for good: one
#                  that starts winning is back in front after about
#                  H ln 2 bars, whatever it lost before.
#                - A LADDER OF LIFETIMES HEDGE_HORIZONS instead of one
#                  memory: one learner per lifetime, and a Bayesian
#                  mixture on top (Vovk's aggregating algorithm, unit
#                  rate, see _hedge_window) scores each learner on its
#                  own hedge loss. When the long-memory learner is surprised
#                  by a regime break the short-memory one has already
#                  moved and its lower hedge loss wins the meta learner
#                  over, so a new leader is in front within tens of bars;
#                  in a stable regime the long-memory learner is the more
#                  concentrated and takes the weight back. The played
#                  weights stay a convex combination of the experts.
#                (A fixed-share floor on the weights was tried and
#                dropped: it bought no revival the ladder does not already
#                give, and it flipped the learned direction on every
#                pullback inside a trend.)
#                Everything is still restarted every bar over the last
#                HEDGE_MEMORY bars and every loss in them is a function
#                of the last 2 n bars (the stance looks back n bars for a
#                break, see _expert_stances), so the learner state is an
#                exact function of a fixed number of past bars
#                (hedge_warmup), which the walk-
#                forward's warm-up buffer relies on (walkforward.
#                window_backtest: a window warmed on `warmup_bars` of
#                history must match a full-history run exactly). The
#                window's edge now carries gamma^HEDGE_MEMORY of a bar's
#                weight (e^-1.6 at H = 160, e^-3 at H = 80, e^-6 at H =
#                40) instead of all of it: a horizon, not a cliff.
#   * channel  : upper/lower/mid = the weight-averaged expert channels. The
#                exit channel reuses the weights over the ladder scaled by
#                HEDGE_EXIT_SCALE (Turtle 20/10, 55/20 style).
#   * direction: for direction_logic 'learned', +1 (follow) or -1 (fade)
#                from the sign of the net side weight.
#   * position : on the wide ladder, the size of every entry is the
#                committee's own position (hedge_position): the stance the
#                weighted experts hold, times the trade weight, one more
#                aggregation of the played mixture against cash. The plain
#                ladder sizes a learned direction by the net side weight and
#                a fixed one at full size, as it always did.

HEDGE_LADDER = (10, 20, 40, 80)
HEDGE_EXIT_SCALE = 0.5
HEDGE_MEMORY = 250            # bars of losses the learner scores (about a year of daily bars)
HEDGE_HORIZONS = (20, 40, 80, 160)   # lifetimes H of the discounted learners (gamma = 1 - 1/H): the lookback
                                     # ladder doubled, from a month to most of the memory. Shorter lifetimes
                                     # revive a written-off expert sooner (16: ~16 bars, 20: ~20, 32: ~25) but
                                     # flip the learned direction more inside a trend (16 fades 11 % of a
                                     # trend's bars on the regime test series, 20 7 %, 32 7 %, 64 1 %)
HEDGE_MODES = ("trend", "countertrend", "learned")

# The wide ladder ('hedge_wide'). Every number here is a rung, fixed in
# advance like HEDGE_LADDER, never fitted.
HEDGE_WIDE_K = 2.0              # half-width of the Keltner experts, in ATRs: the template default channel_k,
                                # one band per Donchian rung (SMA(n) +/- 2 ATR for n in HEDGE_LADDER)


@dataclass(frozen=True)
class HedgeExpert:
    """One expert of a hedge ladder: a channel and the side it takes on a
    break of it (+1 follow, -1 fade). An expert is a rule fixed in advance,
    not a parameter, and its stance on a bar is an exact function of the
    `lead` bars behind it (the warm-up contract, see `hedge_warmup`)."""
    kind: str = "donchian"   # 'donchian'  highest high / lowest low of n bars
                             # 'keltner'   SMA(n) +/- k ATR(atr_n): Keltner's original SMA midline, not the
                             #             template's EMA, because a rolling window is exact once full
                             #             and an EMA never is (see walkforward._ema_settle_bars); the SMA
                             #             is reduced window by window (_window_mean) so a slice
                             #             reproduces it bit for bit
    n: int = 20
    k: float = 0.0           # band half-width (Keltner only)
    side: int = 1            # +1 follow the break, -1 fade it (stamped by `hedge_ladder`)

    @property
    def span(self) -> int:
        """Bars a break stays the expert's stance: its own lookback, when the
        break bar leaves the Donchian channel or the SMA window behind a band."""
        return int(self.n)

    @property
    def label(self) -> str:
        base = str(int(self.n)) if self.kind == "donchian" else f"kel{int(self.n)}x{self.k:g}"
        return f"{'follow' if self.side > 0 else 'fade'}_{base}"

    def formed(self, atr_n: int) -> int:
        """Index of the first bar whose bands are fully formed."""
        if self.kind == "keltner":
            return max(int(self.n) - 1, int(atr_n))   # atr_n true ranges, each with the close before it
        return int(self.n) - 1

    def lead(self, atr_n: int) -> int:
        """Bars before the stance on bar r is exact: the oldest break it can
        hold is span - 1 bars back and was tested against the bands of the
        bar before that, which must be formed (2 n for a Donchian rung)."""
        return self.span + self.formed(atr_n) + 1

    def bands(self, df: pd.DataFrame, atr_n: int, scale: float = 1.0):
        """(upper, lower) arrays of the expert's channel at `scale` times its
        lookback (the exit channel is the entry one at HEDGE_EXIT_SCALE)."""
        m = max(2, int(round(int(self.n) * scale)))
        if self.kind == "donchian":
            up, lo, _ = donchian(df, m)
            return _to_arr(up), _to_arr(lo)
        if self.kind != "keltner":
            raise ValueError(f"unknown expert kind {self.kind}")
        mid = _window_mean(df["Close"].to_numpy(), m)   # the SMA, window by window
        width = self.k * _to_arr(atr(df, atr_n))
        return mid + width, mid - width


def _window_mean_loop(x, n):
    """Rolling mean of `x` over `n` bars, every window summed on its own, in
    order (see `_window_mean`). NaN before the first full window."""
    T = x.shape[0]
    mean = np.full(T, np.nan)
    for t in range(n - 1, T):
        s = 0.0
        for j in range(t - n + 1, t + 1):
            s += x[j]
        mean[t] = s / n
    return mean


def _window_mean(x: np.ndarray, n: int) -> np.ndarray:
    """Rolling mean of `x` over `n` bars, each window summed on its own
    rather than by a running sum. pandas' rolling mean carries a compensated
    running sum whose last bits depend on where the series started (3e-14
    on a mean of prices), so a slice would not reproduce the full series'
    bands and the ladder's warm-up contract (`hedge_warmup`: a warmed window
    matches a full-history run to the test's 1e-12) would hold less
    tightly. An explicit sequential loop rather than a numpy reduction over
    a sliding view, so the result does not depend on numpy's (unspecified,
    pairwise) summation order either. O(T n), and n is at most a ladder
    rung."""
    return _window_mean_fast(np.ascontiguousarray(x, dtype=np.float64), int(n))


# channel_type -> the side-less experts of its ladder (see hedge_ladder for the sided ones)
HEDGE_LADDERS = {
    "hedge": tuple(HedgeExpert("donchian", n) for n in HEDGE_LADDER),
    "hedge_wide": tuple(HedgeExpert("donchian", n) for n in HEDGE_LADDER)
                  + tuple(HedgeExpert("keltner", n, HEDGE_WIDE_K) for n in HEDGE_LADDER),
}
HEDGE_CHANNELS = tuple(HEDGE_LADDERS)   # the channel types the learner builds

# The ladders whose templates are sized by the learner's POSITION (see
# hedge_position): the experts are scored on the legs the template's `sides`
# switch lets it trade, and an entry is scaled by the position the weighted
# committee of experts itself holds, times the weight a committee-or-cash
# aggregation puts on trading at all. The plain ladder keeps the original
# rule: both legs scored, a learned direction sized by its net side weight, a
# fixed one at full size.
HEDGE_POSITION_SIZED = ("hedge_wide",)


def hedge_position_sized(ladder: str) -> bool:
    _ladder_spec(ladder)
    return ladder in HEDGE_POSITION_SIZED


def hedge_scored_sides(ladder: str, sides: str) -> str:
    """The sides the learner scores its experts on for a template with the
    `sides` switch: the template's own on a position-sized ladder, both on
    the plain one. A learner sized by its net side weight compares follow
    with fade, which only means something when both legs are scored (on
    one leg they are two trades rather than opposite ones); a learner sized
    by its position must hold that position on the legs the template can
    take, or a committee net short would size a long-only entry."""
    if sides not in SIDES:
        raise ValueError(f"unknown sides {sides!r}; known: {', '.join(SIDES)}")
    return sides if hedge_position_sized(ladder) else "both"


def hedge_ladder_for(channel_type: str) -> str:
    """The ladder a template's learner runs on: its own for a hedge channel,
    the plain Donchian one for a learned direction on a fitted channel."""
    return channel_type if channel_type in HEDGE_LADDERS else "hedge"


def hedge_warmup(atr_n: int, ladder: str = "hedge") -> int:
    """Bars before the learner's first fully formed output, which is then
    exactly what a full-history run computes (the walk-forward's warm-up
    buffer relies on it). Row t of the weights replays the losses of rows
    t - HEDGE_MEMORY + 1 .. t, and loss row r is exact once
      * the expert stances on r and r-1 are: each expert's `lead` (a stance
        looks back `span` bars for a break and each break test needs formed
        bands behind it, so r >= 2 n for a Donchian rung, see
        `_expert_stances`), and
      * the ATR on r-1 is: atr_n true ranges, each needing the close before
        it, so r >= atr_n + 1."""
    experts = _ladder_spec(ladder)
    return int(HEDGE_MEMORY + max(max(e.lead(atr_n) for e in experts), int(atr_n) + 1))


def _ladder_spec(ladder: str):
    try:
        return HEDGE_LADDERS[ladder]
    except KeyError:
        raise ValueError(f"unknown hedge ladder {ladder!r}; known: {', '.join(HEDGE_LADDERS)}") from None


def hedge_ladder(mode: str, ladder: str = "hedge") -> list:
    """The sided experts the learner scores under a direction mode: the
    ladder on the follow side (trend), on the fade side (countertrend) or
    both (learned, so the learner can switch between following and fading
    as well as between channels)."""
    spec = _ladder_spec(ladder)
    if mode == "trend":
        return [replace(e, side=1) for e in spec]
    if mode == "countertrend":
        return [replace(e, side=-1) for e in spec]
    if mode == "learned":
        return [replace(e, side=1) for e in spec] + [replace(e, side=-1) for e in spec]
    raise ValueError(f"unknown hedge mode {mode}")


def hedge_experts(mode: str, ladder: str = "hedge"):
    """(lookbacks, sides) arrays of `hedge_ladder(mode, ladder)`."""
    experts = hedge_ladder(mode, ladder)
    return (np.array([int(e.n) for e in experts], dtype=np.int64),
            np.array([float(e.side) for e in experts]))


def _hedge_round(l, d, w, z, delta, eta, gamma):
    """One AdaHedge round of one learner over N experts, in place.

    `l` are this round's losses (in [0, 1]), `w` the weights the learner
    PLAYS on them (overwritten with the next round's), `d` its deficits
    (discounted cumulative loss behind the leader, so min d = 0), `z` the
    normaliser of `w` (sum_e exp(-eta d[e])), `delta` and `eta` its
    accumulated mixability gap and learning rate and `gamma` the discount
    (1 for none). Returns (z, delta, eta, gap, h): the new state, the
    round's gap and the Hedge loss h = w . l the played weights suffered."""
    N = l.shape[0]
    h = 0.0
    for e in range(N):
        h += w[e] * l[e]
    if delta > 0.0:
        # The mix loss -log(sum_e w[e] exp(-eta l[e])) / eta, from the
        # deficits the weights were made of (w[e] = exp(-eta d[e]) / z), not
        # from w itself. While delta is tiny eta is huge and a trailing
        # expert's weight underflows to exactly 0; on the round it beats the
        # leaders by a wide margin every term of the sum over w would be 0,
        # the mix loss +inf, the round's gap discarded and eta left huge --
        # on the very round that should end follow-the-leader. Here the
        # smallest exponent of the sum is 0, so it lies in [1, N], as z does.
        m1 = np.inf                            # min over e of d[e] + l[e]
        for e in range(N):
            if d[e] + l[e] < m1:
                m1 = d[e] + l[e]
        s1 = 0.0
        for e in range(N):
            s1 += np.exp(-eta * (d[e] + l[e] - m1))
        mix = m1 - np.log(s1 / z) / eta
    else:                                      # eta = inf and w uniform: the best expert's loss
        mix = np.inf
        for e in range(N):
            if l[e] < mix:
                mix = l[e]
    gap = h - mix
    if gap < 0.0:                              # rounding only: the mix loss never beats the Hedge loss
        gap = 0.0
    delta = gamma * delta + gap                # discounted cumulative mixability gap
    # discounted deficits, re-centred on the new leader
    m = np.inf
    for e in range(N):
        d[e] = gamma * d[e] + l[e]
        if d[e] < m:
            m = d[e]
    for e in range(N):
        d[e] -= m
    # new learning rate and weights
    if delta > 0.0:
        eta = np.log(N) / delta
        z = 0.0
        for e in range(N):
            w[e] = np.exp(-eta * d[e])
            z += w[e]
        for e in range(N):
            w[e] /= z
    else:
        # only while every round so far scored all experts alike, so the
        # deficits are all 0 and follow-the-leader is a tie
        eta = np.inf
        z = float(N)
        for e in range(N):
            w[e] = 1.0 / N
    return z, delta, eta, gap, h


def _hedge_window(loss, t0, t1, gammas, d, w, z, delta, eta, h, dm, v):
    """The K discounted learners (one per lifetime, gammas[k]) and the meta
    learner over them, from a cold start over loss[t0:t1]. The work arrays
    hold the state: d, w (K x N) the deficits and played weights per
    learner, z / delta / eta (K), h (K) the learners' Hedge losses of the
    round, dm / v (K) the meta learner's deficits and weights. Returns
    (meta Hedge loss, best expert loss, trade weight) of the last round;
    the final weights are left in w and v.

    The meta learner is scored on the loss each learner's played weights
    suffered (sum_k v[k] h[k] is then the loss the played mixture suffered)
    and is the aggregating algorithm (Vovk 1990) with a unit learning rate,
    i.e. Bayesian averaging with likelihood exp(-loss), discounted at the
    longest lifetime. AdaHedge's own rate is wrong at this level: the
    learners' losses are near-identical most of the time, so its mixability
    gap stays tiny, eta explodes and the meta weights become follow-the-
    leader on rounding noise (a mirror-image series moved 0.1 of the weight
    on a 1e-14 loss difference). With a unit rate a tie stays a tie and a
    regime break, 20 bars of 0.1 lower loss at the short lifetime, is 7:1
    odds in its favour; the losses being in [0, 1] is what makes 1 the
    natural rate.

    The trade weight is one more aggregation of the same kind, one level
    up: the played mixture (its Hedge loss hm each round) against cash (the
    neutral loss 0.5), unit rate, discounted at the longest lifetime. It is
    the weight the aggregate puts on trading at all, and unlike a cash
    expert on the ladder it is free of the winner's curse: the mixture's
    realised loss is one causal sequence, not the best of N noisy ones, so
    on noise it averages a half (with the spread of a sigmoid of a
    discounted P&L, sd about 0.25) and it drops only when what the learner
    actually played lost to standing aside. The unit rate and the longest
    lifetime are the two conventions of the rule, taken from the meta
    learner; here they are a choice, not a mixability argument."""
    K = gammas.shape[0]
    N = loss.shape[1]
    gm = 0.0
    for k in range(K):
        if gammas[k] > gm:
            gm = gammas[k]
        z[k] = float(N)
        delta[k] = 0.0
        eta[k] = np.inf
        dm[k] = 0.0
        v[k] = 1.0 / K
        for e in range(N):
            d[k, e] = 0.0
            w[k, e] = 1.0 / N
    hm = 0.5
    best = 0.5
    dt = 0.0                    # discounted loss of the played mixture ...
    dc = 0.0                    # ... and of cash, for the trade weight
    for t in range(t0, t1):
        l = loss[t]
        hm = 0.0
        for k in range(K):
            z[k], delta[k], eta[k], _, h[k] = _hedge_round(l, d[k], w[k], z[k], delta[k], eta[k], gammas[k])
            hm += v[k] * h[k]
        m = np.inf
        for k in range(K):
            dm[k] = gm * dm[k] + h[k]
            if dm[k] < m:
                m = dm[k]
        s = 0.0
        for k in range(K):
            dm[k] -= m
            v[k] = np.exp(-dm[k])
            s += v[k]
        for k in range(K):
            v[k] /= s
        best = np.inf
        for e in range(N):
            if l[e] < best:
                best = l[e]
        dt = gm * dt + hm
        dc = gm * dc + 0.5
    u = 1.0 / (1.0 + np.exp(dt - dc))     # exp(-dt) / (exp(-dt) + exp(-dc)), never overflowing
    return hm, best, u


def _hedge_core(loss, memory, gammas):
    """Windowed learner over a T x N loss matrix: row t of the returned
    T x N weights is the mixture played after the last `memory` rounds up to
    and including t, from a cold start. Also returns the learners' eta
    (T x K), the meta learner's weights over them (T x K), the surprise
    of each round: the Hedge loss the played mixture suffered minus the
    best expert's loss (T), and the trade weight (T): the weight of the
    played mixture against cash (see `_hedge_window`). Plain numpy so numba
    can compile it."""
    T, N = loss.shape
    K = gammas.shape[0]
    W = np.empty((T, N))
    ETA = np.empty((T, K))
    V = np.empty((T, K))
    SURPRISE = np.empty(T)
    TRADE = np.empty(T)
    d = np.empty((K, N)); w = np.empty((K, N))
    z = np.empty(K); delta = np.empty(K); eta = np.empty(K); h = np.empty(K); dm = np.empty(K); v = np.empty(K)
    for t in range(T):
        t0 = t + 1 - memory
        if t0 < 0:
            t0 = 0
        hm, best, u = _hedge_window(loss, t0, t + 1, gammas, d, w, z, delta, eta, h, dm, v)
        for e in range(N):
            s = 0.0
            for k in range(K):
                s += v[k] * w[k, e]
            W[t, e] = s
        for k in range(K):
            ETA[t, k] = eta[k]
            V[t, k] = v[k]
        SURPRISE[t] = hm - best if hm > best else 0.0   # a convex mix can round a few ulps under the best
        TRADE[t] = u
    return W, ETA, V, SURPRISE, TRADE


def _adahedge_loop(loss, memory=HEDGE_MEMORY, horizons=HEDGE_HORIZONS, with_trade=False):
    """Windowed, discounted AdaHedge over a ladder of lifetimes: W (T x N)
    the played weights, ETA (T x K) the learners' learning rates, V (T x K)
    the meta weights over the lifetimes, SURPRISE (T) the played mixture's
    Hedge loss minus the best expert's, each round, and with `with_trade`
    also TRADE (T), the weight on trading at all (see `_hedge_window`)."""
    loss = np.ascontiguousarray(loss, dtype=np.float64)
    gammas = 1.0 - 1.0 / np.asarray(horizons, dtype=np.float64)
    out = _hedge_fast(loss, int(memory), np.ascontiguousarray(gammas))
    return out if with_trade else out[:4]


def _expert_stances(high, low, close, uppers, lowers, sides, spans, allow_long, allow_short):
    """Stance (+1/-1/0) each expert holds at the CLOSE of bar t: a break of
    its upper band (high[t] >= upper[t-1]) puts a follow expert
    (sides[e] = +1) long, a break of its lower band puts it short; a fade
    expert (sides[e] = -1) takes the opposite side. A bar that breaks both
    edges is no signal.

    The stance is that of the most recent one-sided break within the last
    `spans[e]` bars (t - spans[e] < break bar <= t), and 0 (flat) when there
    was none. The span is the expert's own lookback: that is when the break
    bar leaves the expert's n-bar channel, i.e. when the expert itself no
    longer "sees" it. Bounding it is what makes the stance a function of a
    FIXED number of past bars (2 n: the span plus the channel behind its
    oldest break test) rather than of the whole history -- a stance simply
    held until the next break could depend on bars arbitrarily far back
    (nothing forces an n-bar channel to break within any fixed time), and a
    walk-forward window warmed on `hedge_warmup` bars then learned different
    weights than a full-history run did.

    A stance on a side the template may not take (`allow_long` /
    `allow_short` False, the template's `sides` switch) is flat instead: the
    expert is scored on the leg the template would trade, not on the one it
    never will. The break is still the most recent one, so the other leg's
    breaks end the stance as they would have."""
    T, N = uppers.shape
    S = np.zeros((T, N))
    for e in range(N):
        sign = sides[e]
        span = spans[e]
        last_t = -1                 # bar of the most recent one-sided break
        last_s = 0.0                # ... and the stance it implied
        for t in range(1, T):
            u = uppers[t - 1, e]
            lo = lowers[t - 1, e]
            if not np.isnan(u):
                up = high[t] >= u
                dn = low[t] <= lo
                if up and not dn:
                    last_t = t
                    last_s = sign
                elif dn and not up:
                    last_t = t
                    last_s = -sign
            if last_t >= 0 and t - last_t < span:
                if (last_s > 0.0 and allow_long) or (last_s < 0.0 and allow_short):
                    S[t, e] = last_s
    return S


try:
    from numba import njit as _njit_h
    _hedge_round = _njit_h(cache=True, nogil=True)(_hedge_round)
    _hedge_window = _njit_h(cache=True, nogil=True)(_hedge_window)
    _hedge_fast = _njit_h(cache=True, nogil=True)(_hedge_core)
    _stances_fast = _njit_h(cache=True, nogil=True)(_expert_stances)
    _window_mean_fast = _njit_h(cache=True, nogil=True)(_window_mean_loop)
except Exception:  # pragma: no cover
    _hedge_fast = _hedge_core
    _stances_fast = _expert_stances
    _window_mean_fast = _window_mean_loop


def _allowed(sides: str):
    """(long allowed, short allowed) under a template's `sides` switch."""
    return sides != "short_only", sides != "long_only"


def _hedge_stances(df: pd.DataFrame, atr_n: int, mode: str, ladder: str = "hedge", sides: str = "both"):
    """(S, formed, experts): the T x N stance (+1/-1/0) each expert of
    `hedge_ladder(mode, ladder)` holds at the close of every bar (see
    `_expert_stances`; only on the sides a `sides` template may take), the
    T x N mask of the bars its bands were formed on, and the experts."""
    experts = hedge_ladder(mode, ladder)
    high = _to_arr(df["High"]); low = _to_arr(df["Low"]); close = _to_arr(df["Close"])
    T = len(close)
    N = len(experts)
    uppers = np.empty((T, N)); lowers = np.empty((T, N))
    cache = {}
    for j, e in enumerate(experts):
        key = (e.kind, int(e.n), float(e.k))
        if key not in cache:
            cache[key] = e.bands(df, atr_n)
        uppers[:, j], lowers[:, j] = cache[key]
    esides = np.array([float(e.side) for e in experts])
    spans = np.array([e.span for e in experts], dtype=np.int64)
    allow_long, allow_short = _allowed(sides)
    S = _stances_fast(high, low, close, uppers, lowers, esides, spans, allow_long, allow_short)
    return S, ~np.isnan(uppers), experts


def _stance_loss(df: pd.DataFrame, atr_n: int, S: np.ndarray, formed: np.ndarray, cost_bps: float,
                 cost_pts: float = 0.0) -> np.ndarray:
    """The T x N loss matrix the learner scores (in [0, 1]) from the T x N
    stances: expert e's loss on bar t is 0.5 * (1 - payoff) where the payoff
    is the ATR-normalised move of bar t in the direction of the stance e
    held at the previous close, minus the cost (per side, in ATRs, as the
    engine charges it: `cost_bps` of |price| plus `cost_pts` price points,
    the per-unit cost over the point value) of the sides e traded at the
    close of t to reach its new stance, halved and clipped to [-1, 1]."""
    close = _to_arr(df["Close"])
    T, N = S.shape
    a = _to_arr(atr(df, atr_n))
    a_prev = np.where(a[:-1] > 0, a[:-1], np.nan)
    z = (close[1:] - close[:-1]) / a_prev                    # next-bar move, in ATRs
    c = (np.abs(close[1:]) * (float(cost_bps) / 1e4) + float(cost_pts)) / a_prev   # one side's cost, in ATRs
    flips = np.abs(S[1:] - S[:-1])                            # sides traded to reach the new stance (a reversal is two)
    payoff = np.clip((S[:-1] * z[:, None] - c[:, None] * flips) / 2.0, -1.0, 1.0)
    loss = np.full((T, N), 0.5)
    loss[1:] = 0.5 * (1.0 - np.nan_to_num(payoff, nan=0.0))
    loss = np.where(formed, loss, 0.5)   # an unformed expert has no stance: neutral loss
    return np.ascontiguousarray(loss)


def _hedge_loss(df: pd.DataFrame, atr_n: int, mode: str, cost_bps: float = 0.0, ladder: str = "hedge",
                sides: str = "both", cost_pts: float = 0.0):
    """The loss matrix the learner scores (see `_stance_loss`) plus the
    ladder's (lookbacks, sides)."""
    S, formed, _ = _hedge_stances(df, atr_n, mode, ladder, sides)
    lookbacks, esides = hedge_experts(mode, ladder)
    return _stance_loss(df, atr_n, S, formed, cost_bps, cost_pts), lookbacks, esides


def _hedge_run(df: pd.DataFrame, atr_n: int, mode: str, cost_bps: float, ladder: str, sides: str,
               cost_pts: float = 0.0):
    S, formed, _ = _hedge_stances(df, atr_n, mode, ladder, sides)
    loss = _stance_loss(df, atr_n, S, formed, cost_bps, cost_pts)
    W, ETA, V, SURPRISE, TRADE = _adahedge_loop(loss, with_trade=True)
    # the committee's position: the stance the weighted experts hold at the
    # close of t, in [-1, 1], times the weight on trading at all
    POSITION = TRADE * np.clip((W * S).sum(axis=1), -1.0, 1.0)
    return W, ETA, V, SURPRISE, TRADE, POSITION, loss


def _hedge_cached(df: pd.DataFrame, atr_n: int, mode: str, cost_bps: float, ladder: str = "hedge",
                  sides: str = "both", dfkey=None, cost_pts: float = 0.0):
    """One learner run per (bars, ladder, mode, ATR length, costs, scored
    sides): the entry channel, the exit channel and the learned direction
    all read the same weights."""
    sides = hedge_scored_sides(ladder, sides)
    return _cached(df, ("hedge", ladder, mode, int(atr_n), float(cost_bps), sides, float(cost_pts)),
                   lambda: _hedge_run(df, atr_n, mode, cost_bps, ladder, sides, cost_pts), dfkey)


def hedge_weights(df: pd.DataFrame, atr_n: int, mode: str = "trend", cost_bps: float = 0.0,
                  ladder: str = "hedge", sides: str = "both", cost_pts: float = 0.0) -> np.ndarray:
    """T x n_experts learner weights over `hedge_ladder(mode, ladder)`, row t
    computed from bars <= t (the last HEDGE_MEMORY of them). `sides` is the
    template's: on a position-sized ladder the experts are scored on the
    legs it can trade (see `hedge_scored_sides`). `cost_pts` is the per-unit
    cost in price points (see `_stance_loss`)."""
    return _hedge_cached(df, atr_n, mode, cost_bps, ladder, sides, cost_pts=cost_pts)[0]


def hedge_diagnostics(df: pd.DataFrame, atr_n: int, mode: str = "trend", cost_bps: float = 0.0,
                      ladder: str = "hedge", sides: str = "both", cost_pts: float = 0.0) -> dict:
    """What the learner did, bar by bar, for notebooks and dashboards:
    `weights` (expert weights, columns follow_10 / fade_20 / ... and, on the
    wide ladder, follow_kel20x2 / fade_kel40x2 ...), `loss` (the expert
    losses it scored), `eta` (each lifetime's learning rate, columns =
    HEDGE_HORIZONS), `horizon_weights` (the meta learner's weights over the
    lifetimes: shorter ones gaining is the learner shortening its memory),
    `surprise` (the loss the played mixture suffered minus the best
    expert's, in [0, 1]), `trade_weight` (the weight of the played mixture
    against cash, in [0, 1]) and `position` (the committee's position, see
    `hedge_position`; both informational on the plain ladder, which does
    not size by them). Nothing here changes the strategy."""
    W, ETA, V, SURPRISE, TRADE, POSITION, loss = _hedge_cached(df, atr_n, mode, cost_bps, ladder, sides,
                                                               cost_pts=cost_pts)
    experts = [e.label for e in hedge_ladder(mode, ladder)]
    idx = df.index
    return {
        "weights": pd.DataFrame(W, index=idx, columns=experts),
        "loss": pd.DataFrame(loss, index=idx, columns=experts),
        "eta": pd.DataFrame(ETA, index=idx, columns=list(HEDGE_HORIZONS)),
        "horizon_weights": pd.DataFrame(V, index=idx, columns=list(HEDGE_HORIZONS)),
        "surprise": pd.Series(SURPRISE, index=idx),
        "trade_weight": pd.Series(TRADE, index=idx),
        "position": pd.Series(POSITION, index=idx),
    }


def hedge_position(df: pd.DataFrame, atr_n: int, mode: str = "trend", cost_bps: float = 0.0,
                   ladder: str = "hedge", sides: str = "both", cost_pts: float = 0.0) -> np.ndarray:
    """Per-bar position of the learner's committee, in [-1, 1]: the stance
    the weighted experts hold at the close of the bar (sum of weight times
    stance, +1 every expert long, -1 every expert short, 0 flat or split),
    scaled by the trade weight (the played mixture against cash, see
    `_hedge_window`). A template on a position-sized ladder scales its
    entries by its MAGNITUDE, a measure of how positioned the committee is
    and whether that has paid lately: near 1 when the experts agree and
    have been paying, near 0 when they are split, flat, or losing to cash.
    The sign is not used: the template trades the break of its channel,
    which the committee, by construction, is not yet in. NaN until the
    learner is formed."""
    p = _hedge_cached(df, atr_n, mode, cost_bps, ladder, sides, cost_pts=cost_pts)[5].copy()
    p[:min(len(p), hedge_warmup(atr_n, ladder))] = np.nan
    return p


def hedge_direction(df: pd.DataFrame, atr_n: int, cost_bps: float = 0.0, ladder: str = "hedge",
                    sides: str = "both", cost_pts: float = 0.0) -> np.ndarray:
    """Per-bar signed conviction of the learner over the follow and fade
    experts, in [-1, 1], for direction_logic 'learned': its sign is the
    direction (> 0 follow the next break, else fade it, from the net side
    weight, follow minus fade) and its magnitude the conviction that scales
    the position (see `_bar_loop`). NaN until the learner is formed.

    On the plain ladder the conviction is the net side weight itself: +1 is
    every expert on the follow side, 0 a dead heat. On a position-sized
    ladder it is the magnitude of the committee's position
    (`hedge_position`), so a near tie between follow and fade experts that
    all hold the same stance (a long-only ladder in a bull market, where
    buying new highs and buying dips both pay) is a full position, and a
    committee that is split, flat or losing to cash is a small one."""
    experts = hedge_ladder("learned", ladder)
    W = hedge_weights(df, atr_n, "learned", cost_bps, ladder, sides, cost_pts)
    esides = np.array([float(e.side) for e in experts])
    net = np.clip(W @ esides, -1.0, 1.0)      # the rows of W sum to 1 up to rounding
    if hedge_position_sized(ladder):
        d = np.where(net < 0.0, -1.0, 1.0) * np.abs(hedge_position(df, atr_n, "learned", cost_bps, ladder, sides,
                                                                   cost_pts))
    else:
        d = net
    d[:min(len(d), hedge_warmup(atr_n, ladder))] = np.nan
    return d


def hedge_active(df: pd.DataFrame, atr_n: int, mode: str, cost_bps: float = 0.0,
                 ladder: str = "hedge", sides: str = "both", cost_pts: float = 0.0) -> np.ndarray:
    """Per-bar size, in [0, 1], of a `trend` or `countertrend` template on a
    position-sized ladder: the magnitude of its committee's position (the
    direction array is +/- this, see `_compute_indicators`). Following
    breaks in a range, or fading them in a trend, leaves the committee
    split and losing to cash, and the template stands mostly aside instead
    of trading at full size off a channel the learner has given up on. 1
    everywhere on the plain ladder. NaN until the learner is formed."""
    if hedge_position_sized(ladder):
        return np.abs(hedge_position(df, atr_n, mode, cost_bps, ladder, sides, cost_pts))
    a = np.ones(len(df))      # exactly 1, not the rows of W summed to within an ulp
    a[:min(len(a), hedge_warmup(atr_n, ladder))] = np.nan
    return a


def hedge_channel(df: pd.DataFrame, atr_n: int, mode: str = "trend", scale: float = 1.0,
                  cost_bps: float = 0.0, ladder: str = "hedge", sides: str = "both", cost_pts: float = 0.0):
    """Weight-averaged channel over the expert ladder (each expert's bands at
    `scale` times its lookback), NaN until the learner is formed."""
    W = hedge_weights(df, atr_n, mode, cost_bps, ladder, sides, cost_pts)
    experts = hedge_ladder(mode, ladder)
    T = len(df)
    up = np.zeros(T); lo = np.zeros(T)
    cache = {}
    for j, e in enumerate(experts):
        key = (e.kind, int(e.n), float(e.k))
        if key not in cache:
            cache[key] = e.bands(df, atr_n, scale)
        up += W[:, j] * cache[key][0]
        lo += W[:, j] * cache[key][1]
    warm = min(T, hedge_warmup(atr_n, ladder))
    up[:warm] = np.nan; lo[:warm] = np.nan
    idx = df.index
    upper = pd.Series(up, index=idx); lower = pd.Series(lo, index=idx)
    return upper, lower, (upper + lower) / 2


# Regime "trendiness" registry. Every entry is oriented so that a HIGHER
# value means MORE trending; `threshold` is the default split between
# trend and range, `thresholds` the walk-forward search grid.
REGIME_INDICATORS = {
    "er":   dict(fn=lambda df, n: efficiency_ratio(df["Close"], n), n=20, threshold=0.35, thresholds=[0.25, 0.35, 0.45]),
    "adx":  dict(fn=lambda df, n: adx(df, n), n=14, threshold=25.0, thresholds=[20.0, 25.0, 30.0]),
    "cti":  dict(fn=lambda df, n: cti(df["Close"], n).abs(), n=20, threshold=0.5, thresholds=[0.4, 0.5, 0.6]),
    "chop": dict(fn=lambda df, n: 100.0 - choppiness(df, n), n=14, threshold=50.0, thresholds=[38.2, 50.0, 61.8]),
    # `cash_fn`: the form a cash asset uses instead of `fn` (see is_cash_asset)
    "vr":   dict(fn=lambda df, n: variance_ratio(df["Close"], n), n=60, threshold=1.0, thresholds=[0.9, 1.0, 1.1],
                 cash_fn=lambda df, n: variance_ratio(df["Close"], n, log_returns=True)),
}


# --------------------------------------------------------------------------
# Template / config
# --------------------------------------------------------------------------

DIRECTION_LOGICS = ["trend", "countertrend", "learned"]
CHANNEL_TYPES = ["donchian", "keltner", "bollinger", "hedge", "hedge_wide"]
ENTRY_STYLES = ["stop", "close_confirm", "pullback"]
EXIT_STYLES = ["channel", "atr_trail", "target_stop", "time_stop"]
REGIME_INDICATOR_NAMES = list(REGIME_INDICATORS)
REGIME_FILTERS = ["none", "trend_only", "range_only"]
VOL_FILTERS = [False, True]
BIAS_FILTERS = ["none", "sma"]
SIDES = ["both", "long_only", "short_only"]


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
    sides: str = "both"

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
    vol_target: float = 0.0      # annualized vol the entry is sized to; 0 = risk_pct on the ATR stop
    vol_target_n: int = 60       # bars of close-to-close returns in the realized-vol estimate
    cost_bps: float = 5.0        # per side, commission + slippage, in basis points of notional
    # the instrument: run settings like cost_bps, never tuned by the walk-forward
    point_value: float = 1.0     # currency per 1.0 of price per unit (share 1, Brent lot 1000, ES 50)
    cost_per_unit: float = 0.0   # currency per unit per side (commission + slippage in ticks), on top of cost_bps
    margin_per_unit: float = 0.0  # initial margin per unit, the leverage cap's basis; 0 = none (cap on notional)
    whole_units: bool = False    # floor every size to whole units (contracts); a size below 1 opens nothing
    roll_cost_per_unit: float = 0.0  # currency per unit held through a contract roll (the data's `Roll` bars)

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
            sides=self.sides,
        )

    def validate(self):
        assert self.direction_logic in DIRECTION_LOGICS
        assert self.channel_type in CHANNEL_TYPES
        assert self.entry_style in ENTRY_STYLES
        assert self.exit_style in EXIT_STYLES
        assert self.regime_indicator in REGIME_INDICATORS
        assert self.regime_filter in REGIME_FILTERS
        assert self.bias_filter in BIAS_FILTERS
        assert self.sides in SIDES
        assert self.point_value > 0, "point_value must be positive"
        assert self.cost_per_unit >= 0 and self.margin_per_unit >= 0 and self.roll_cost_per_unit >= 0


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
    # High/Low feed the ATR and every channel, so they are part of the key too:
    # two frames with identical closes but different wicks must not share arrays
    h = df["High"].to_numpy()
    lo = df["Low"].to_numpy()
    return (n, df.index[0], df.index[-1], float(c[0]), float(c[n // 3]), float(c[(2 * n) // 3]), float(c[-1]),
            float(h.sum()), float(lo.sum()), float(h[n // 2]), float(lo[n // 2]))


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


def _channel_arrays(df, kind, n, k, atr_n, dfkey=None, mode="trend", role="entry", cost_bps=0.0, sides="both",
                    cost_pts=0.0):
    def build():
        up, lo, mid = channel(df, kind, n, k, atr_n, mode=mode, role=role, cost_bps=cost_bps, sides=sides,
                              cost_pts=cost_pts)
        return (_to_arr(up), _to_arr(lo), _to_arr(mid))
    if kind in HEDGE_CHANNELS:
        # no lookback / width: keyed on the ladder (the kind), direction mode
        # (signed rewards), ATR length, entry/exit role, the costs the experts
        # are charged and the sides they are scored on
        spec = ("channel", kind, role, mode, atr_n, float(cost_bps), sides, float(cost_pts))
    else:
        spec = ("channel", kind, n, k if kind != "donchian" else 0.0, atr_n if kind == "keltner" else 0)
    return _cached(df, spec, build, dfkey)


def is_cash_asset(tpl: StrategyTemplate) -> bool:
    """A cash asset (no `margin_per_unit`: a share, an ETF) or a future or
    spread (`margin_per_unit` given). Two rules read the price differently:

    * the volatility target: a cash asset sizes on the std of its percentage
      returns (one share's dollar vol is that times the price, the rule a
      share has always been sized by); a future on the std of its price
      changes in points (its quoted level is back-adjusted or goes through
      zero, so a percentage of it means nothing, while one lot's dollar vol
      is exactly point_value x that std);
    * the variance-ratio regime filter (`vr`): on log returns for a cash
      asset, on point changes for a future.

    Everything else works on price differences for both."""
    return not tpl.margin_per_unit > 0


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

    mode = tpl.direction_logic
    ladder = hedge_ladder_for(tpl.channel_type)
    # the sides the learner scores on (the template's own on a position-sized
    # ladder, both otherwise): canonical, so the three `sides` of a plain-ladder
    # template share one channel and one direction in the cache as they share one
    # learner
    sides = hedge_scored_sides(ladder, tpl.sides)
    # the per-unit cost in price points, the unit the learner scores in
    cost_pts = float(tpl.cost_per_unit) / float(tpl.point_value)
    up, lo, _ = _channel_arrays(df, tpl.channel_type, tpl.n_entry, tpl.channel_k, tpl.atr_n, dfkey, mode, "entry",
                                tpl.cost_bps, sides, cost_pts)
    use("upper", up)
    use("lower", lo)
    use("atr", _cached(df, ("atr", tpl.atr_n), lambda: _to_arr(atr(df, tpl.atr_n)), dfkey))

    # per-bar direction: +1 follow the break, -1 fade it; learned from the
    # follow/fade expert ladder (NaN while the learner is unformed), else constant
    if mode == "learned":
        use("direction", _cached(df, ("hedge_dir", ladder, tpl.atr_n, float(tpl.cost_bps), sides, cost_pts),
                                 lambda: hedge_direction(df, tpl.atr_n, tpl.cost_bps, ladder, sides, cost_pts),
                                 dfkey))
    elif tpl.channel_type in HEDGE_CHANNELS and hedge_position_sized(ladder):
        # a fixed direction on a position-sized ladder: sized by the committee's
        # position (on a bar where that is exactly 0 the loop reads the +0.0 as the
        # fade logic and sizes the fill to nothing, as a learned tie does)
        sign = 1.0 if mode == "trend" else -1.0
        use("direction", _cached(df, ("hedge_active", ladder, mode, tpl.atr_n, float(tpl.cost_bps), sides, cost_pts),
                                 lambda: sign * hedge_active(df, tpl.atr_n, mode, tpl.cost_bps, ladder, sides,
                                                             cost_pts),
                                 dfkey))
    else:
        ind["direction"] = np.full(n, 1.0 if mode == "trend" else -1.0)

    if tpl.exit_style == "channel":
        upx, lox, midx = _channel_arrays(df, tpl.channel_type, tpl.n_exit, tpl.channel_k, tpl.atr_n, dfkey, mode, "exit",
                                         tpl.cost_bps, sides, cost_pts)
        use("upper_x", upx)
        use("lower_x", lox)
        use("mid_x", midx)

    if tpl.regime_filter != "none":
        spec = REGIME_INDICATORS[tpl.regime_indicator]
        rn = tpl.regime_n if tpl.regime_n > 0 else spec["n"]
        # a cash asset reads the indicator on its returns where that differs
        # (the variance ratio on log returns), a future or spread on point changes
        cash = is_cash_asset(tpl) and "cash_fn" in spec
        fn = spec["cash_fn"] if cash else spec["fn"]
        use("regime", _cached(df, ("regime", tpl.regime_indicator, rn, cash), lambda: _to_arr(fn(df, rn)), dfkey))
        ind["regime_threshold"] = (
            tpl.regime_threshold if not np.isnan(tpl.regime_threshold) else spec["threshold"]
        )

    if tpl.vol_filter:
        use("vol_rank", _cached(df, ("vol_rank", tpl.atr_n, tpl.vol_lookback),
                                lambda: _to_arr(atr(df, tpl.atr_n).rolling(tpl.vol_lookback).rank(pct=True)), dfkey))

    if tpl.vol_target > 0:
        # realized per-bar vol, NOT annualized (the cached array does not
        # depend on periods_per_year(); the target is scaled to per-bar in
        # backtest()), measured the way the instrument is (is_cash_asset):
        if is_cash_asset(tpl):
            # a cash asset: std of pct changes, the share's own return vol
            use("rvol", _cached(df, ("rvol", tpl.vol_target_n),
                                lambda: _to_arr(df["Close"].pct_change().rolling(tpl.vol_target_n).std()), dfkey))
        else:
            # a future or spread: std of close differences in PRICE POINTS,
            # defined at any price level and the same after a constant is
            # added to every price (a back-adjusted or through-zero series)
            use("rvol", _cached(df, ("rvol_pts", tpl.vol_target_n),
                                lambda: _to_arr(df["Close"].diff().rolling(tpl.vol_target_n).std()), dfkey))

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
REASONS = ["stop", "channel", "midline", "target", "time", "stop_same_bar", "ruin"]

# The order log (backtest(..., log_orders=True)): what the loop had WORKING on
# a bar and what it did on it, read out of the loop itself rather than rebuilt
# from the rules. One row per event; the row's bar is the bar the order was
# live on or executed on. `detail` is per kind: the order type of an entry
# (ORDER_TYPES), the stop kind of a working stop (STOP_KINDS), the expiry bar
# of a pullback limit, the exit reason (REASONS) of an exit fill, and for a
# cancelled pullback 0 = warm-up bar, 1 = indicators not usable.
ORDER_KINDS = [
    "entry_working",        # entry stop (trend) / limit (fade) / pullback trigger watched on this bar
    "entry_submit_market",  # close_confirm: market-on-open order after a confirming close
    "pullback_submit",      # pullback limit placed (rests from the next bar)
    "pullback_working",     # pullback limit resting on this bar
    "pullback_expire",      # ... expired unfilled
    "pullback_cancel",      # ... cancelled (warm-up, or indicators no longer usable)
    "entry_fill",           # position opened: price, shares; aux = initial hard stop
    "entry_reject",         # a fill that sized to zero (learner conviction 0, or no cash)
    "stop_working",         # the stop in force on this bar (hard / opposite channel / trail)
    "target_working",       # the ATR target or channel midline in force on this bar
    "time_exit_submit",     # time exit: market order at the open of this bar
    "exit_fill",            # position closed: price, shares; aux = trade P&L, detail = REASONS index
    "roll",                 # the position was held through a contract roll at this close; aux = the roll cost
]
ORDER_TYPES = ["-", "stop", "limit", "pullback", "market"]
STOP_KINDS = ["hard", "channel", "trail"]
_O_ENTRY_WORKING, _O_SUBMIT_MKT, _O_PB_SUBMIT, _O_PB_WORKING, _O_PB_EXPIRE, _O_PB_CANCEL = 0, 1, 2, 3, 4, 5
_O_ENTRY_FILL, _O_ENTRY_REJECT, _O_STOP_WORKING, _O_TARGET_WORKING, _O_TIME_EXIT, _O_EXIT_FILL = 6, 7, 8, 9, 10, 11
_O_ROLL = 12


def _olog(o_i, o_f, m, bar, kind, side, detail, level, qty, aux):
    """Append one order-log event (numba does not bounds-check, so the write
    is guarded; the count still advances so an overflow is detectable)."""
    if m < o_i.shape[0]:
        o_i[m, 0] = bar
        o_i[m, 1] = kind
        o_i[m, 2] = side
        o_i[m, 3] = detail
        o_f[m, 0] = level
        o_f[m, 1] = qty
        o_f[m, 2] = aux
    return m + 1


def _bar_loop(open_, high, low, close, ready, upper, lower, atr_v,
              upper_x, lower_x, mid_x, regime, vol_rank, bias,
              direction, rvol, entry_style, exit_style, regime_mode, regime_thr,
              has_vol, vol_low, vol_high, has_bias, allow_long, allow_short,
              atr_mult_stop, atr_mult_target, atr_mult_trail, pullback_atr_mult,
              pullback_valid_bars, max_hold_bars, risk_pct, max_leverage, vol_target_bar, cost_rate,
              point_value, cost_per_unit, margin_per_unit, whole_units,
              roll, roll_cost_per_unit, fixed_capital,
              initial_equity, first_trade_bar, log_orders):
    """The bar loop. Plain numpy code so numba can compile it unchanged;
    the pure-Python version is used when numba is not installed.

    Sizing, in units (shares or lots; `point_value` currency per point per
    unit): `risk_pct` of cash lost at the ATR stop, or, when `vol_target_bar`
    > 0, the units whose dollar vol is cash * vol_target_bar, i.e. cash *
    vol_target_bar / (rvol[i-1] * point_value) with rvol in price points
    when a margin is given (a future or spread), cash * vol_target_bar /
    (rvol[i-1] * |fill| * point_value) with rvol the pct vol otherwise (a
    cash asset; see `is_cash_asset`) (both per-bar vols); scaled by the learner's conviction |direction[i-1]|
    when the direction is learned (direction is a constant +/-1 otherwise);
    capped at `max_leverage` * cash of margin (units * margin_per_unit) when
    a margin is given, else of notional (units * point_value * |price|);
    fixed for the life of the trade. Nothing on the margined path divides by
    a price, so a price at or below zero is an ordinary price (the cash path
    is for positive prices only: `validate_instrument`).

    Costs per side: cost_rate * notional + cost_per_unit * units. P&L:
    side * units * point_value * (price change). `whole_units` floors the
    size to an integer after the cap (a contract is indivisible). A roll:
    a position still open at the close of a bar with roll[i] > 0 (the
    contract was rolled at that close) pays roll_cost_per_unit * units,
    whichever its side: the cost of trading out of one contract and into
    the next is charged to what is held, never folded into the price
    series (where a short would collect it as a gain).

    Exits: a stop or a target (an ATR target, a countertrend's channel
    midline) the OPEN already trades through fills at the open, before
    anything intrabar: the open is the bar's first price, so that ordering
    is not ambiguous. Only when both are reached intrabar does the stop go
    first (the conservative reading of an unknown path). On the entry bar the
    stops are checked as on any bar, and the target too when every price
    beyond the fill on the target's side came after it: a fill at the open
    (a market order, or any order the open gapped through) or a stop entry
    (price runs THROUGH a buy stop upwards, so a high above it is later). A
    limit entry's bar is ambiguous (its high may precede the fill), and its
    target waits for the next bar.

    Ruin: a close that leaves the equity at or below zero (a gap through the
    stop beyond the margin) liquidates the position at that close (reason
    "ruin") and closes the account: nothing trades after it and the equity
    stays at what the liquidation left. That is a DEFICIT when the loss went
    beyond the cash: a futures account owes it to the broker, so it is kept
    (a -167 % bar, not a -100 % one). The bar returns after it are 0.

    `fixed_capital` sizes every entry (and the leverage cap) on
    `initial_equity` instead of the cash, and turns ruin off: each bar's P&L
    then depends on the prices alone, never on what the account made or
    lost before, which is what a cross-validation that re-uses the bars
    after a held-out block needs (robustness.trial_returns).

    Bars before `first_trade_bar` are indicator warm-up only: nothing is
    entered on them (and no order rests on them), so the walk-forward can
    hand the loop a slice that starts before its window without any trade
    decided on the earlier bars leaking into the window's result.

    `log_orders` keeps the order log (ORDER_KINDS) in `o_i` (bar, kind, side,
    detail) and `o_f` (level, qty, aux), `m` rows; off, both are one row and
    nothing is written, so the research loop pays a predictable branch per
    bar and no more.

    Returns equity, entries, the closed-trade columns
    (entry_bar, exit_bar, side, entry_px, exit_px, shares, pnl, cost, reason, count),
    the loop's final state (open position, resting order, last usable ATR)
    and the order log (o_i, o_f, m)."""
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
    # order log: at most 6 events on any bar (see the sites below), 8 is the margin
    cap = 8 * n + 8 if log_orders else 1
    o_i = np.zeros((cap, 4), dtype=np.int64)
    o_f = np.zeros((cap, 3))
    m = 0

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
    pos_trend = True      # direction logic the OPEN trade was entered under
    pend_trend = True     # ... and the resting pullback order
    ruined = False

    equity[0] = initial_equity
    for i in range(1, n):
        # ---- ruin: the previous close left nothing ----
        if ruined:
            equity[i] = cash
            continue
        if not fixed_capital and equity[i - 1] <= 0.0:
            # (a roll charged at that close stays charged: the ruin was decided on
            # the equity net of it, and undoing it could leave a closed account
            # with cash; one roll on a dead account is the conservative reading)
            if position != 0:
                # liquidated at that close; a loss beyond the cash is a deficit
                # the account owes, not the broker's to absorb
                gross = position * shares * point_value * (close[i - 1] - entry_price)
                xcost = cost_rate * shares * point_value * abs(close[i - 1]) + cost_per_unit * shares
                cash += gross - xcost
                t_entry[n_trades] = entry_bar
                t_exit[n_trades] = i - 1
                t_side[n_trades] = position
                t_entry_px[n_trades] = entry_price
                t_exit_px[n_trades] = close[i - 1]
                t_shares[n_trades] = shares
                t_pnl[n_trades] = gross - xcost - entry_cost
                t_cost[n_trades] = entry_cost + xcost
                t_reason[n_trades] = 6
                if log_orders:
                    m = _olog(o_i, o_f, m, i - 1, _O_EXIT_FILL, position, 6, close[i - 1], shares, t_pnl[n_trades])
                n_trades += 1
                position = 0
                shares = 0.0
            # the account is closed: it stays at what is left, a deficit included
            equity[i - 1] = cash
            equity[i] = cash
            pend_active = False
            ruined = True
            continue

        # direction in force for NEW signals on bar i (constant unless 'learned')
        is_trend = direction[i - 1] > 0.0
        # `a_ok`: every indicator this template uses is fully formed on bar i-1
        # and the ATR is usable (and the realized vol, when the size targets
        # it). New business (filters, orders, entries) needs that; an OPEN
        # POSITION is managed on every bar regardless, otherwise a flat patch
        # that drives the ATR to zero would suspend its stop just when the gap
        # through it arrives. `last_a` is the most recent usable ATR and is
        # always > 0 while a position is open (entering required a_ok).
        atr_ok = ready[i - 1] and atr_v[i - 1] > 0.0
        if atr_ok:
            a = atr_v[i - 1]
            last_a = a
        else:
            a = last_a
        a_ok = atr_ok and (vol_target_bar <= 0.0 or rvol[i - 1] > 0.0)

        # ---- manage open position: exits (checked intrabar on bar i) ----
        if position != 0:
            exit_price = 0.0
            reason = -1
            side = position
            stop_level = stop_price  # hard stop is always active
            stop_reason = 0

            stop_kind = 0

            if exit_style == 3 and i - entry_bar >= max_hold_bars:
                # the time exit is an order at the OPEN: it is out before
                # anything intrabar can reach a stop
                exit_price = open_[i]
                reason = 4
                if log_orders:
                    m = _olog(o_i, o_f, m, i, _O_TIME_EXIT, side, 0, exit_price, shares, 0.0)
            else:
                if exit_style == 1:
                    # trailing level from the extreme up to bar i-1 (no intrabar look-ahead)
                    trail_stop = trail_extreme - side * atr_mult_trail * a
                    if side == 1 and trail_stop > stop_level:
                        stop_level = trail_stop
                        stop_kind = 2
                    elif side == -1 and trail_stop < stop_level:
                        stop_level = trail_stop
                        stop_kind = 2
                elif exit_style == 0 and pos_trend:
                    # the opposite channel is a stop on the SAME side as the hard
                    # stop; on the way through, whichever sits nearer to the price
                    # is hit first, so it must fill at that level, not at the
                    # further one
                    if side == 1 and lower_x[i - 1] > stop_level:
                        stop_level = lower_x[i - 1]
                        stop_reason = 1
                        stop_kind = 1
                    elif side == -1 and upper_x[i - 1] < stop_level:
                        stop_level = upper_x[i - 1]
                        stop_reason = 1
                        stop_kind = 1

                if log_orders:
                    m = _olog(o_i, o_f, m, i, _O_STOP_WORKING, side, stop_kind, stop_level, shares, 0.0)
                    if exit_style == 2:
                        m = _olog(o_i, o_f, m, i, _O_TARGET_WORKING, side, 0, target_price, shares, 0.0)
                    elif exit_style == 0 and not pos_trend:
                        m = _olog(o_i, o_f, m, i, _O_TARGET_WORKING, side, 0, mid_x[i - 1], shares, 0.0)

                # the target in force: the ATR target, or a countertrend
                # trade's channel midline (a limit order on the far side)
                has_tgt = exit_style == 2 or (exit_style == 0 and not pos_trend)
                tgt = target_price if exit_style == 2 else mid_x[i - 1]
                tgt_reason = 3 if exit_style == 2 else 2

                # the open is the bar's first price: an order it already trades
                # through fills there, before anything intrabar. Stop first
                # (a target and a stop both through the open fill at the same
                # price), then the target, which the open can reach without
                # going anywhere near the stop
                if side == 1 and open_[i] <= stop_level:
                    exit_price = open_[i]
                    reason = stop_reason
                elif side == -1 and open_[i] >= stop_level:
                    exit_price = open_[i]
                    reason = stop_reason
                elif has_tgt and ((side == 1 and open_[i] >= tgt) or (side == -1 and open_[i] <= tgt)):
                    exit_price = open_[i]
                    reason = tgt_reason
                # intrabar, the path is unknown: the stop goes first
                elif side == 1 and low[i] <= stop_level:
                    exit_price = min(open_[i], stop_level)
                    reason = stop_reason
                elif side == -1 and high[i] >= stop_level:
                    exit_price = max(open_[i], stop_level)
                    reason = stop_reason

                if reason < 0:
                    if exit_style == 0:
                        if not pos_trend:
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

            if reason >= 0:
                gross = position * shares * point_value * (exit_price - entry_price)
                xcost = cost_rate * shares * point_value * abs(exit_price) + cost_per_unit * shares
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
                if log_orders:
                    m = _olog(o_i, o_f, m, i, _O_EXIT_FILL, position, reason, exit_price, shares, t_pnl[n_trades])
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

        # ---- no usable indicators (or still warming up): manage what is open,
        # start nothing new ----
        if not a_ok or i < first_trade_bar:
            if log_orders and pend_active:
                m = _olog(o_i, o_f, m, i, _O_PB_CANCEL, pend_side, 0 if i < first_trade_bar else 1,
                          pend_level, 0.0, 0.0)
            pend_active = False  # don't leave a stale resting order behind
            if position != 0 and roll[i] > 0.0 and roll_cost_per_unit > 0.0:
                # held through a roll at this close: the position pays it
                # (roll[i] rolls: more than one when a dropped bar's moved here)
                rc = roll_cost_per_unit * shares * roll[i]
                cash -= rc
                entry_cost += rc
                if log_orders:
                    m = _olog(o_i, o_f, m, i, _O_ROLL, position, 0, close[i], shares, rc)
            equity[i] = cash + (position * shares * point_value * (close[i] - entry_price) if position != 0 else 0.0)
            continue

        # ---- filters (previous bar's fully-formed values) ----
        can_enter = True
        if regime_mode == 1:
            can_enter = regime[i - 1] >= regime_thr
        elif regime_mode == 2:
            can_enter = regime[i - 1] < regime_thr
        if can_enter and has_vol:
            can_enter = vol_low <= vol_rank[i - 1] <= vol_high

        # the `sides` switch first, then the SMA bias narrows it further
        long_ok = allow_long
        short_ok = allow_short
        if has_bias:
            long_ok = long_ok and close[i - 1] > bias[i - 1]
            short_ok = short_ok and close[i - 1] < bias[i - 1]

        fill_side = 0
        fill_px = 0.0
        fill_type = 0   # ORDER_TYPES index of the order that fills (order log only)

        # ---- pending pullback order ----
        fill_trend = is_trend
        if position == 0 and pend_active:
            if log_orders:
                m = _olog(o_i, o_f, m, i, _O_PB_WORKING, pend_side, pend_expires, pend_level, 0.0, 0.0)
            if pend_side == 1 and low[i] <= pend_level:
                fill_side = 1
                fill_px = min(open_[i], pend_level)
                fill_trend = pend_trend
                fill_type = 3
                pend_active = False
            elif pend_side == -1 and high[i] >= pend_level:
                fill_side = -1
                fill_px = max(open_[i], pend_level)
                fill_trend = pend_trend
                fill_type = 3
                pend_active = False
            elif i >= pend_expires:
                pend_active = False
                if log_orders:
                    m = _olog(o_i, o_f, m, i, _O_PB_EXPIRE, pend_side, pend_expires, pend_level, 0.0, 0.0)

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
                if log_orders:
                    # what is watched on this bar: a stop at the channel when
                    # following the break, a limit at it when fading, the
                    # trigger of a pullback limit for the pullback style
                    otype = 3 if entry_style == 2 else (1 if is_trend else 2)
                    if long_ok:
                        m = _olog(o_i, o_f, m, i, _O_ENTRY_WORKING, 1, otype,
                                  level_up if is_trend else level_down, 0.0, 0.0)
                    if short_ok:
                        m = _olog(o_i, o_f, m, i, _O_ENTRY_WORKING, -1, otype,
                                  level_down if is_trend else level_up, 0.0, 0.0)

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
                    fill_type = 4
                    if log_orders:
                        m = _olog(o_i, o_f, m, i, _O_SUBMIT_MKT, side, 4, fill_px, 0.0, 0.0)
                elif entry_style == 0:
                    if is_trend:
                        fill_px = max(open_[i], level) if side == 1 else min(open_[i], level)
                        fill_type = 1
                    else:
                        fill_px = min(open_[i], level) if side == 1 else max(open_[i], level)
                        fill_type = 2
                    fill_side = side
                else:
                    pend_active = True
                    pend_side = side
                    pend_trend = is_trend
                    pend_level = level - side * pullback_atr_mult * a
                    pend_expires = i + pullback_valid_bars
                    if log_orders:
                        m = _olog(o_i, o_f, m, i, _O_PB_SUBMIT, side, pend_expires, pend_level, 0.0, 0.0)

        # ---- open the position ----
        entered = False
        if fill_side != 0:
            stop_dist = atr_mult_stop * a
            # the capital the size is a fraction of: the account's cash, or a
            # constant one (fixed_capital: no compounding, no path dependence)
            base = initial_equity if fixed_capital else cash
            if vol_target_bar > 0.0:
                # constant dollar volatility: the dollar vol wanted over the
                # dollar vol of one unit; the stop still sits atr_mult_stop
                # ATRs away, but the loss there is no longer risk_pct of equity
                # (rvol in price points; for a cash asset it is the pct vol,
                # and one unit's dollar vol is that times the fill price)
                if margin_per_unit > 0.0:
                    qty = base * vol_target_bar / (rvol[i - 1] * point_value)
                else:
                    qty = base * vol_target_bar / (rvol[i - 1] * abs(fill_px) * point_value)
            else:
                qty = base * risk_pct / (stop_dist * point_value)
            # scaled by the learner's conviction in the logic the trade is
            # entered under: the net side weight in its favour on the bar
            # before the fill (1 unless the direction is learned). A pullback
            # limit placed under a logic the learner has since abandoned
            # sizes to 0 and opens nothing, like a zero ATR size would.
            qty *= max(direction[i - 1] * (1.0 if fill_trend else -1.0), 0.0)
            # the leverage cap's basis per unit: the margin when one is given
            # (a spread has no notional), else the notional at the fill
            basis = margin_per_unit if margin_per_unit > 0.0 else point_value * abs(fill_px)
            qty = min(qty, max_leverage * base / basis) if basis > 0.0 else 0.0
            if whole_units:
                qty = np.floor(qty)
            if qty > 0:
                entry_cost = cost_rate * qty * point_value * abs(fill_px) + cost_per_unit * qty
                cash -= entry_cost
                position = fill_side
                pos_trend = fill_trend
                shares = qty
                entry_price = fill_px
                stop_price = fill_px - fill_side * stop_dist
                target_price = fill_px + fill_side * atr_mult_target * a
                trail_extreme = fill_px
                entry_bar = i
                entries[i] = 1
                entered = True
                if log_orders:
                    m = _olog(o_i, o_f, m, i, _O_ENTRY_FILL, fill_side, fill_type, fill_px, qty, stop_price)
            elif log_orders:
                m = _olog(o_i, o_f, m, i, _O_ENTRY_REJECT, fill_side, fill_type, fill_px, qty, 0.0)

        # ---- conservative same-bar stop check on the entry bar ----
        # every stop the exit block would run from the next bar is already
        # working on this one: the hard stop, the chandelier (anchored at the
        # fill) and, for a trend trade with a channel exit, the opposite
        # channel -- whichever sits nearest the fill. A stop the fill is
        # already through (a close_confirm open that gapped past the exit
        # channel) goes out at the fill itself. The target is working from
        # the fill too; it is taken on this bar only when the bar's prices on
        # its side certainly came AFTER the fill (see the docstring) and no
        # stop was reached (both reached: the stop, as on any bar).
        if entered:
            sb_level = stop_price
            sb_reason = 5
            sb_kind = 0
            if exit_style == 1:
                trail_stop = fill_px - position * atr_mult_trail * a
                if position == 1 and trail_stop > sb_level:
                    sb_level = trail_stop
                    sb_kind = 2
                elif position == -1 and trail_stop < sb_level:
                    sb_level = trail_stop
                    sb_kind = 2
            elif exit_style == 0 and pos_trend:
                if position == 1 and lower_x[i - 1] > sb_level:
                    sb_level = lower_x[i - 1]
                    sb_reason = 1
                    sb_kind = 1
                elif position == -1 and upper_x[i - 1] < sb_level:
                    sb_level = upper_x[i - 1]
                    sb_reason = 1
                    sb_kind = 1
            sb_tgt = exit_style == 2 or (exit_style == 0 and not pos_trend)
            tgt = target_price if exit_style == 2 else mid_x[i - 1]
            if log_orders:
                m = _olog(o_i, o_f, m, i, _O_STOP_WORKING, position, sb_kind, sb_level, shares, 0.0)
                if sb_tgt:
                    m = _olog(o_i, o_f, m, i, _O_TARGET_WORKING, position, 0, tgt, shares, 0.0)
            # every price beyond the fill on the target's side is later than
            # the fill: a fill at the open, or a stop entry (fill_type 1),
            # whose price ran through the level in the trade's direction
            after_fill = fill_px == open_[i] or fill_type == 1
            # the same order as on any later bar (the fill is this bar's
            # "open" for the position): an order the fill is already through
            # goes out at the fill, the stop first; then intrabar, the stop
            # first, then the target
            stop_at_fill = (position == 1 and fill_px <= sb_level) or (position == -1 and fill_px >= sb_level)
            tgt_at_fill = (sb_tgt and after_fill
                           and ((position == 1 and fill_px >= tgt) or (position == -1 and fill_px <= tgt)))
            hit = (position == 1 and low[i] <= sb_level) or (position == -1 and high[i] >= sb_level)
            tgt_hit = (sb_tgt and after_fill
                       and ((position == 1 and high[i] >= tgt) or (position == -1 and low[i] <= tgt)))
            if stop_at_fill:
                sb_px = fill_px
            elif tgt_at_fill:
                sb_px = fill_px
                sb_reason = 3 if exit_style == 2 else 2
            elif hit:
                sb_px = sb_level
            elif tgt_hit:
                sb_px = tgt
                sb_reason = 3 if exit_style == 2 else 2
            if stop_at_fill or tgt_at_fill or hit or tgt_hit:
                gross = position * shares * point_value * (sb_px - entry_price)
                xcost = cost_rate * shares * point_value * abs(sb_px) + cost_per_unit * shares
                cash += gross - xcost
                t_entry[n_trades] = entry_bar
                t_exit[n_trades] = i
                t_side[n_trades] = position
                t_entry_px[n_trades] = entry_price
                t_exit_px[n_trades] = sb_px
                t_shares[n_trades] = shares
                t_pnl[n_trades] = gross - xcost - entry_cost
                t_cost[n_trades] = entry_cost + xcost
                t_reason[n_trades] = sb_reason
                if log_orders:
                    m = _olog(o_i, o_f, m, i, _O_EXIT_FILL, position, sb_reason, sb_px, shares, t_pnl[n_trades])
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

        # ---- a roll at this close: the position held through it pays it ----
        if position != 0 and roll[i] > 0.0 and roll_cost_per_unit > 0.0:
            rc = roll_cost_per_unit * shares * roll[i]
            cash -= rc
            entry_cost += rc
            if log_orders:
                m = _olog(o_i, o_f, m, i, _O_ROLL, position, 0, close[i], shares, rc)

        # ---- mark to market at the close of bar i ----
        equity[i] = cash + (position * shares * point_value * (close[i] - entry_price) if position != 0 else 0.0)

    # ---- ruin on the last bar: the loop above would have closed it on the next ----
    if n > 1 and equity[n - 1] <= 0.0 and not ruined and not fixed_capital:
        if position != 0:
            gross = position * shares * point_value * (close[n - 1] - entry_price)
            xcost = cost_rate * shares * point_value * abs(close[n - 1]) + cost_per_unit * shares
            cash += gross - xcost
            t_entry[n_trades] = entry_bar
            t_exit[n_trades] = n - 1
            t_side[n_trades] = position
            t_entry_px[n_trades] = entry_price
            t_exit_px[n_trades] = close[n - 1]
            t_shares[n_trades] = shares
            t_pnl[n_trades] = gross - xcost - entry_cost
            t_cost[n_trades] = entry_cost + xcost
            t_reason[n_trades] = 6
            if log_orders:
                m = _olog(o_i, o_f, m, n - 1, _O_EXIT_FILL, position, 6, close[n - 1], shares, t_pnl[n_trades])
            n_trades += 1
            position = 0
            shares = 0.0
        equity[n - 1] = cash
        pend_active = False

    # the trailing state is returned too, so the live signal layer in live.py can
    # read the CURRENT position and resting order out of the same loop the
    # backtest runs, instead of reimplementing the rules and drifting from them
    return (equity, entries, t_entry, t_exit, t_side, t_entry_px, t_exit_px, t_shares, t_pnl, t_cost,
            t_reason, n_trades, position, shares, entry_price, stop_price, target_price, trail_extreme,
            entry_bar, entry_cost, pend_active, pend_side, pend_level, pend_expires, last_a,
            pos_trend, pend_trend, o_i, o_f, m)


try:  # compile the loop once per process; falls back to plain Python without numba
    from numba import njit as _njit
    _olog = _njit(cache=True, nogil=True)(_olog)   # before the loop is compiled: it is called from it
    _bar_loop_fast = _njit(cache=True, nogil=True)(_bar_loop)
    HAVE_NUMBA = True
except Exception:  # pragma: no cover
    _bar_loop_fast = _bar_loop
    HAVE_NUMBA = False


def _orders_frame(o_i: np.ndarray, o_f: np.ndarray, idx_arr: np.ndarray) -> pd.DataFrame:
    """The order log of `_bar_loop` as a frame with real dates: one row per
    event, `detail` decoded per kind (see ORDER_KINDS) into `detail_text`,
    `expires` the date a pullback limit lapses."""
    n = len(idx_arr)
    kinds = np.asarray(ORDER_KINDS, dtype=object)[o_i[:, 1]]
    detail = o_i[:, 3]
    text = np.full(len(kinds), "", dtype=object)
    for k, table in ((_O_ENTRY_WORKING, ORDER_TYPES), (_O_SUBMIT_MKT, ORDER_TYPES), (_O_ENTRY_FILL, ORDER_TYPES),
                     (_O_ENTRY_REJECT, ORDER_TYPES), (_O_STOP_WORKING, STOP_KINDS), (_O_EXIT_FILL, REASONS)):
        sel = o_i[:, 1] == k
        text[sel] = np.asarray(table, dtype=object)[detail[sel]]
    sel = o_i[:, 1] == _O_PB_CANCEL
    text[sel] = np.where(detail[sel] == 0, "warm-up", "indicators")
    dated = np.issubdtype(idx_arr.dtype, np.datetime64)
    # NaT with an explicit unit: numpy deprecates the unit-less one; it converts
    # to whatever resolution the index has
    expires = (np.full(len(kinds), np.datetime64("NaT", "ns"), dtype=idx_arr.dtype) if dated
               else np.full(len(kinds), None, dtype=object))   # a frame with no dates: raw index labels
    pb = (o_i[:, 1] == _O_PB_SUBMIT) | (o_i[:, 1] == _O_PB_WORKING) | (o_i[:, 1] == _O_PB_EXPIRE)
    sel = pb & (detail < n)
    expires[sel] = idx_arr[detail[sel]]
    return pd.DataFrame(dict(
        date=idx_arr[o_i[:, 0]], kind=kinds, side=o_i[:, 2], level=o_f[:, 0], qty=o_f[:, 1], aux=o_f[:, 2],
        detail=detail, detail_text=text, expires=expires,
    ))


def validate_instrument(df: pd.DataFrame, tpl: StrategyTemplate) -> None:
    """Refuse a run whose sizing or costs would read a price that is not
    there. An instrument that trades at or below zero anywhere (a calendar
    spread) has no notional: the leverage cap needs `margin_per_unit` and the
    costs must be per unit (`cost_per_unit`, with `cost_bps` = 0). On a share
    a price at or below zero is a bad print: `data.load_yfinance` drops such
    bars, a CSV of a cash asset should not carry them. Cheap (one reduction),
    so `backtest` runs it every call."""
    if not tpl.point_value > 0:
        raise ValueError(f"point_value must be positive, got {tpl.point_value}")
    if tpl.roll_cost_per_unit > 0 and "Roll" not in df.columns:
        raise ValueError(
            f"roll_cost_per_unit {tpl.roll_cost_per_unit:g} needs a Roll column in the data (1 on the bars at "
            "whose close the contract rolled; extra_utils/ETF_trick_spreads.py writes one, data.load_csv reads it)")
    low = float(np.nanmin(df[["Open", "High", "Low", "Close"]].to_numpy(dtype=float))) if len(df) else 1.0
    if low <= 0:
        if not tpl.margin_per_unit > 0:
            raise ValueError(
                f"a price at or below zero (lowest {low:g}). On a share that is a bad print: drop the bar. "
                "On a spread there is no notional, so the leverage cap needs margin_per_unit (initial "
                "margin per unit, in currency); size costs with cost_per_unit and set cost_bps to 0")
        if tpl.cost_bps > 0:
            raise ValueError(
                f"a price at or below zero (lowest {low:g}) with cost_bps {tpl.cost_bps:g}: a basis-point cost is "
                "a fraction of a notional this instrument does not have; use cost_per_unit with cost_bps = 0")


def instrument_warnings(tpl: StrategyTemplate) -> list:
    """Settings that run but are probably not what a futures trader means:
    printed by the entry points before the pool, never raised."""
    out = []
    if tpl.margin_per_unit > 0 and tpl.max_leverage > 1:
        out.append(f"with margin_per_unit the leverage cap is margin / equity, and max_leverage "
                   f"{tpl.max_leverage:g} lets the margin exceed the equity; 0.5 or less is a realistic cap")
    if tpl.point_value != 1 and not tpl.margin_per_unit > 0:
        out.append(f"point_value {tpl.point_value:g} without margin_per_unit: the leverage cap is on the "
                   "notional at the quoted price, which on a back-adjusted futures series is an artificial "
                   "level; give the contract's margin")
    if tpl.point_value != 1 and tpl.cost_bps > 0:
        out.append(f"point_value {tpl.point_value:g} with cost_bps {tpl.cost_bps:g}: a future is costed per "
                   "contract (cost_per_unit, cost_bps 0); a basis-point cost on a back-adjusted price is arbitrary")
    return out


def typical_units(df: pd.DataFrame, tpl: StrategyTemplate, initial_equity: float = 100_000.0) -> float:
    """The size, in units, the engine's rule gives an entry of `tpl` on a
    median bar of `df` (median ATR or realized vol, median |close|), before
    the learner's conviction and `whole_units`: what a flat run with
    `whole_units` needs to be told it floors to 0."""
    close = df["Close"].to_numpy(dtype=float)
    if tpl.vol_target > 0:
        if is_cash_asset(tpl):
            pct = float(np.nanmedian(df["Close"].pct_change().rolling(tpl.vol_target_n).std().to_numpy()))
            sigma = pct * float(np.nanmedian(np.abs(close)))
        else:
            sigma = float(np.nanmedian(df["Close"].diff().rolling(tpl.vol_target_n).std().to_numpy()))
        qty = initial_equity * tpl.vol_target / np.sqrt(periods_per_year()) / (sigma * tpl.point_value)
    else:
        a = float(np.nanmedian(atr(df, tpl.atr_n).to_numpy()))
        qty = initial_equity * tpl.risk_pct / (tpl.atr_mult_stop * a * tpl.point_value)
    basis = tpl.margin_per_unit if tpl.margin_per_unit > 0 else tpl.point_value * float(np.nanmedian(np.abs(close)))
    if basis > 0:
        qty = min(qty, tpl.max_leverage * initial_equity / basis)
    return float(qty) if np.isfinite(qty) else 0.0


def backtest(df: pd.DataFrame, tpl: StrategyTemplate, initial_equity: float = 100_000.0,
             first_trade_bar: int = 0, log_orders: bool = False, fixed_capital: bool = False) -> dict:
    """Run `tpl` over `df` (must have Open/High/Low/Close). Returns a dict:
        equity   : pd.Series of end-of-bar equity, indexed like df
        returns  : pd.Series of per-bar simple returns of equity
        entries  : np.ndarray (1 on bars where a new trade was opened)
        trades   : list of trade dicts (`shares` = units: shares or lots,
                   `pnl` and `cost` in currency)
        stats    : summary performance stats
        orders   : the order log as a DataFrame (see ORDER_KINDS) when
                   `log_orders`, else None

    `first_trade_bar` > 0 uses the first bars as indicator warm-up only: the
    equity stays at `initial_equity` and no trade can open before that bar.

    `fixed_capital` sizes every entry on `initial_equity` rather than on the
    account's cash, never declares ruin, and returns each bar's P&L over
    `initial_equity` (additive returns) instead of over the previous equity:
    every return is then a function of the prices only, the same whatever
    the account did on earlier bars (see `_bar_loop`).

    A `Roll` column in `df` (> 0 on the bars at whose close the contract
    rolled) charges `tpl.roll_cost_per_unit` per unit to a position held
    through that close, long or short.

    Prices may be zero or negative (see `validate_instrument`): every rule
    works on price differences, and adding a constant to every price leaves
    the trades, the P&L and the equity unchanged.
    """
    n = len(df)
    first_trade_bar = int(max(first_trade_bar, 0))
    validate_instrument(df, tpl)
    # the roll bars matter only when a roll costs something (a Roll column of
    # any dtype is then left alone)
    roll = (_to_arr(df["Roll"].fillna(0.0)) if "Roll" in df.columns and tpl.roll_cost_per_unit > 0
            else np.zeros(n))
    close = _to_arr(df["Close"])
    open_ = _to_arr(df["Open"])
    high = _to_arr(df["High"])
    low = _to_arr(df["Low"])

    ind = _compute_indicators(df, tpl, _df_key(df, close))
    zeros = np.zeros(n)
    # annualized target -> per-bar, read through periods_per_year() (never the
    # constant) so a worker's set_periods_per_year is honoured and the cached
    # realized-vol array stays frequency-agnostic
    vol_target_bar = float(tpl.vol_target) / np.sqrt(periods_per_year()) if tpl.vol_target > 0 else 0.0
    out = _bar_loop_fast(
        open_, high, low, close, ind["ready"], ind["upper"], ind["lower"], ind["atr"],
        ind.get("upper_x", zeros), ind.get("lower_x", zeros), ind.get("mid_x", zeros),
        ind.get("regime", zeros), ind.get("vol_rank", zeros), ind.get("bias", zeros),
        ind["direction"], ind.get("rvol", zeros), ENTRY_CODES[tpl.entry_style], EXIT_CODES[tpl.exit_style],
        REGIME_CODES[tpl.regime_filter], float(ind.get("regime_threshold", 0.0)),
        bool(tpl.vol_filter), float(tpl.vol_low_pct), float(tpl.vol_high_pct), tpl.bias_filter == "sma",
        tpl.sides != "short_only", tpl.sides != "long_only",
        float(tpl.atr_mult_stop), float(tpl.atr_mult_target), float(tpl.atr_mult_trail), float(tpl.pullback_atr_mult),
        int(tpl.pullback_valid_bars), int(tpl.max_hold_bars), float(tpl.risk_pct), float(tpl.max_leverage),
        float(vol_target_bar), tpl.cost_bps / 1e4,
        float(tpl.point_value), float(tpl.cost_per_unit), float(tpl.margin_per_unit), bool(tpl.whole_units),
        roll, float(tpl.roll_cost_per_unit), bool(fixed_capital),
        float(initial_equity), first_trade_bar, bool(log_orders),
    )
    (equity, entries, t_entry, t_exit, t_side, t_entry_px, t_exit_px, t_shares, t_pnl, t_cost,
     t_reason, n_trades, f_position, f_shares, f_entry_price, f_stop, f_target, f_trail,
     f_entry_bar, f_entry_cost, f_pend_active, f_pend_side, f_pend_level, f_pend_expires, f_last_atr,
     f_pos_trend, f_pend_trend, o_i, o_f, n_orders) = out

    idx = df.index
    idx_arr = idx.to_numpy()
    orders = None
    if log_orders:
        if n_orders > o_i.shape[0]:  # pragma: no cover - the capacity bound is proven in _bar_loop
            raise RuntimeError(f"order log overflow: {n_orders} events, capacity {o_i.shape[0]}")
        orders = _orders_frame(o_i[:n_orders], o_f[:n_orders], idx_arr)
    trades = [
        dict(entry_date=pd.Timestamp(idx_arr[t_entry[k]]), side=int(t_side[k]), entry_price=float(t_entry_px[k]),
             shares=float(t_shares[k]), cost=float(t_cost[k]), exit_date=pd.Timestamp(idx_arr[t_exit[k]]),
             exit_price=float(t_exit_px[k]), reason=REASONS[t_reason[k]], pnl=float(t_pnl[k]),
             bars_held=int(t_exit[k] - t_entry[k]))
        for k in range(n_trades)
    ]
    rets = np.zeros(n)
    if fixed_capital:
        # P&L over the constant capital the sizes are a fraction of: additive
        rets[1:] = np.diff(equity) / float(initial_equity)
    else:
        # a ruined run: one bar down to what the liquidation left (below -100 %
        # when it left a deficit), then flat (never a ratio of non-positive equities)
        with np.errstate(divide="ignore", invalid="ignore"):
            rets[1:] = np.where(equity[:-1] > 0.0, equity[1:] / equity[:-1] - 1.0, 0.0)
    equity_s = pd.Series(equity, index=idx)
    returns = pd.Series(rets, index=idx)
    stats = performance_stats(equity, trades, initial_equity, rets, t_pnl[:n_trades],
                               (t_exit[:n_trades] - t_entry[:n_trades]).astype(float))

    open_position = None
    if f_position != 0:
        open_position = dict(
            side=int(f_position), shares=float(f_shares), entry_price=float(f_entry_price),
            entry_bar=int(f_entry_bar), entry_date=pd.Timestamp(idx_arr[int(f_entry_bar)]),
            entry_cost=float(f_entry_cost), hard_stop=float(f_stop), target=float(f_target),
            trail_extreme=float(f_trail), bars_held=int(n - 1 - int(f_entry_bar)),
            unrealized=float(f_position * f_shares * tpl.point_value * (close[-1] - f_entry_price)),
            is_trend=bool(f_pos_trend),
        )
    pending_order = None
    if f_pend_active:
        pending_order = dict(side=int(f_pend_side), level=float(f_pend_level),
                             expires_bar=int(f_pend_expires), is_trend=bool(f_pend_trend))

    return {"equity": equity_s, "returns": returns, "entries": entries, "trades": trades,
            "stats": stats, "open_position": open_position, "pending_order": pending_order,
            "last_atr": float(f_last_atr), "indicators": ind, "orders": orders}


def annualized_sharpe(rets: pd.Series | np.ndarray, ppy: int | None = None) -> float:
    r = np.asarray(rets, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) < 2:
        return 0.0
    sd = r.std(ddof=1)
    ppy = periods_per_year() if ppy is None else ppy
    return float(r.mean() / sd * np.sqrt(ppy)) if sd > 0 else 0.0


def compound(returns) -> np.ndarray:
    """Growth of 1 from per-bar simple returns (1-D, or 2-D column by column),
    for an account that is closed once it reaches zero or less: from the bar
    the curve first sits at or below 0 it stays at that value. A deficit is
    owed, not invested: compounding it with later returns would flip their
    sign (a gain would deepen it), and a later window or another CPCV group
    trading on is a restart the account could not make."""
    eq = np.cumprod(1.0 + np.asarray(returns, dtype=float), axis=0)
    dead = eq <= 0.0
    if not dead.any():
        return eq
    first = np.argmax(dead, axis=0)
    if eq.ndim == 1:
        eq[first:] = eq[first]
    else:
        for j in np.flatnonzero(dead.any(axis=0)):
            eq[first[j]:, j] = eq[first[j], j]
    return eq


def max_drawdown(equity) -> float:
    """Deepest peak-to-trough fall of an equity path, as a (negative) fraction
    of the running peak; 0.0 for an empty path."""
    eq = np.asarray(equity, dtype=float)
    if len(eq) == 0:
        return 0.0
    return float((eq / np.maximum.accumulate(eq) - 1).min())


def performance_stats(equity, trades: list, initial_equity: float, rets=None, pnls=None, bars_held=None) -> dict:
    """Summary stats from an equity path. Accepts a numpy array or a Series;
    `rets`, `pnls` and `bars_held` may be passed to skip recomputation."""
    eq = np.asarray(equity, dtype=float)
    n_bars = len(eq)
    if rets is None:
        rets = np.zeros(n_bars)
        with np.errstate(divide="ignore", invalid="ignore"):
            rets[1:] = np.where(eq[:-1] > 0.0, eq[1:] / eq[:-1] - 1.0, 0.0)
    if pnls is None:
        pnls = np.array([t.get("pnl", 0.0) for t in trades], dtype=float)
    if bars_held is None:
        bars_held = np.array([t.get("bars_held", 0) for t in trades], dtype=float)
    n_years = max(n_bars / periods_per_year(), 1e-6)
    final = float(eq[-1])
    total_return = final / initial_equity - 1
    cagr = (final / initial_equity) ** (1 / n_years) - 1 if final > 0 else -1.0
    sharpe = annualized_sharpe(rets[1:])
    max_dd = max_drawdown(eq)
    wins =pnls[pnls > 0].sum()
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


_performance_stats = performance_stats  # backwards-compatible name
