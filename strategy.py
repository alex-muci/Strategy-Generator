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
                                      template follows or fades the break;
                                      a trade keeps the logic it was opened
                                      under. Works with any channel_type.

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
  run from the CLI, never tuned by the walk-forward.

Execution model (no look-ahead):
  * every decision on bar i uses indicator values fully formed on bar i-1
  * position size, fixed at entry and held to the exit: `risk_pct` of equity
    lost at the `atr_mult_stop` ATR stop, or, when `vol_target` > 0, a
    notional of equity x (vol_target / sqrt(bars per year)) / realized
    per-bar vol (std of close-to-close returns over `vol_target_n` bars).
    Either way capped at `max_leverage` x equity; the ATR stop is unchanged
  * stops/limits are filled intrabar at the level, or at the open if the
    open gapped through the level
  * costs: `cost_bps` (commission + slippage) charged per side on notional
  * equity is marked to market at the CLOSE of each bar, after all fills
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd

# Bars per year, used for every annualization (Sharpe, CAGR, WFE, DSR...).
# Do NOT import this by value: `from strategy import PERIODS_PER_YEAR` freezes it
# at import time and set_periods_per_year() can then no longer be honoured.
# Read it through periods_per_year() instead.
PERIODS_PER_YEAR = 252

# Regular-session bars per year for the intervals yfinance serves. US equity
# ETFs trade 6.5h a day, which Yahoo cuts into seven '1h' bars (the last one is
# a 30-minute stub), thirteen '30m' bars, and so on.
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


def periods_per_year_for_interval(interval: str) -> int:
    """Bars per year for a yfinance interval string ('1d', '1h', '30m', ...)."""
    try:
        return BARS_PER_YEAR[interval]
    except KeyError:
        raise ValueError(
            f"unknown interval {interval!r}; known: {', '.join(BARS_PER_YEAR)}"
        ) from None


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


def channel(df: pd.DataFrame, kind: str, n: int, k: float, atr_n: int,
            mode: str = "trend", role: str = "entry", cost_bps: float = 0.0):
    if kind == "donchian":
        return donchian(df, n)
    if kind == "keltner":
        return keltner(df, n, k, atr_n)
    if kind == "bollinger":
        return bollinger(df, n, k)
    if kind == "hedge":
        return hedge_channel(df, atr_n, mode=mode, scale=HEDGE_EXIT_SCALE if role == "exit" else 1.0,
                             cost_bps=cost_bps)
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
#                experts, a 'learned' direction both.
#   * loss     : each bar, expert e is scored on the ATR-normalised return
#                of the stance it implied on the previous bar (new n-bar
#                high -> long, new n-bar low -> short, else hold; a fade
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
#                HEDGE_MEMORY bars, so the learner state is an exact
#                function of a fixed number of past bars, which the walk-
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

HEDGE_LADDER = (10, 20, 40, 80)
HEDGE_EXIT_SCALE = 0.5
HEDGE_MEMORY = 250            # bars of losses the learner scores (about a year of daily bars)
HEDGE_HORIZONS = (20, 40, 80, 160)   # lifetimes H of the discounted learners (gamma = 1 - 1/H): the lookback
                                     # ladder doubled, from a month to most of the memory. Shorter lifetimes
                                     # revive a written-off expert sooner (16: ~16 bars, 20: ~20, 32: ~25) but
                                     # flip the learned direction more inside a trend (16 fades 11 % of a
                                     # trend's bars on the regime test series, 20 7 %, 32 7 %, 64 1 %)
HEDGE_MODES = ("trend", "countertrend", "learned")


def hedge_warmup(atr_n: int) -> int:
    """Bars before the learner's first fully formed output: every loss in
    its memory needs formed experts and a formed ATR on the bar before."""
    return int(max(HEDGE_LADDER) + atr_n + HEDGE_MEMORY)


def hedge_experts(mode: str):
    """(lookbacks, sides) of the expert ladder for a direction mode:
    trend -> follow experts only, countertrend -> fade experts only,
    learned -> both, so the learner can switch between following and
    fading as well as between periods."""
    if mode == "trend":
        return np.array(HEDGE_LADDER, dtype=np.int64), np.ones(len(HEDGE_LADDER))
    if mode == "countertrend":
        return np.array(HEDGE_LADDER, dtype=np.int64), -np.ones(len(HEDGE_LADDER))
    if mode == "learned":
        n = np.array(HEDGE_LADDER * 2, dtype=np.int64)
        return n, np.concatenate([np.ones(len(HEDGE_LADDER)), -np.ones(len(HEDGE_LADDER))])
    raise ValueError(f"unknown hedge mode {mode}")


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
    (meta Hedge loss, best expert loss) of the last round; the final
    weights are left in w and v.

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
    natural rate."""
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
    return hm, best


def _hedge_core(loss, memory, gammas):
    """Windowed learner over a T x N loss matrix: row t of the returned
    T x N weights is the mixture played after the last `memory` rounds up to
    and including t, from a cold start. Also returns the learners' eta
    (T x K), the meta learner's weights over them (T x K) and the surprise
    of each round: the Hedge loss the played mixture suffered minus the
    best expert's loss (T). Plain numpy so numba can compile it."""
    T, N = loss.shape
    K = gammas.shape[0]
    W = np.empty((T, N))
    ETA = np.empty((T, K))
    V = np.empty((T, K))
    SURPRISE = np.empty(T)
    d = np.empty((K, N)); w = np.empty((K, N))
    z = np.empty(K); delta = np.empty(K); eta = np.empty(K); h = np.empty(K); dm = np.empty(K); v = np.empty(K)
    for t in range(T):
        t0 = t + 1 - memory
        if t0 < 0:
            t0 = 0
        hm, best = _hedge_window(loss, t0, t + 1, gammas, d, w, z, delta, eta, h, dm, v)
        for e in range(N):
            s = 0.0
            for k in range(K):
                s += v[k] * w[k, e]
            W[t, e] = s
        for k in range(K):
            ETA[t, k] = eta[k]
            V[t, k] = v[k]
        SURPRISE[t] = hm - best if hm > best else 0.0   # a convex mix can round a few ulps under the best
    return W, ETA, V, SURPRISE


def _adahedge_loop(loss, memory=HEDGE_MEMORY, horizons=HEDGE_HORIZONS):
    """Windowed, discounted AdaHedge over a ladder of lifetimes: W (T x N)
    the played weights, ETA (T x K) the learners' learning rates, V (T x K)
    the meta weights over the lifetimes, SURPRISE (T) the played mixture's
    Hedge loss minus the best expert's, each round."""
    loss = np.ascontiguousarray(loss, dtype=np.float64)
    gammas = 1.0 - 1.0 / np.asarray(horizons, dtype=np.float64)
    return _hedge_fast(loss, int(memory), np.ascontiguousarray(gammas))


def _expert_stances(high, low, close, uppers, lowers, sides):
    """Stance (+1/-1/0) each Donchian expert holds at the CLOSE of bar t:
    a new n-bar high (high[t] >= upper[t-1]) puts a follow expert
    (sides[e] = +1) long, a new n-bar low puts it short; a fade expert
    (sides[e] = -1) takes the opposite side. Otherwise, and on a bar that
    breaks both edges, the previous stance is held."""
    T, N = uppers.shape
    S = np.zeros((T, N))
    for e in range(N):
        sign = sides[e]
        s = 0.0
        for t in range(1, T):
            u = uppers[t - 1, e]
            lo = lowers[t - 1, e]
            if not np.isnan(u):
                up = high[t] >= u
                dn = low[t] <= lo
                if up and not dn:
                    s = sign
                elif dn and not up:
                    s = -sign
            S[t, e] = s
    return S


try:
    from numba import njit as _njit_h
    _hedge_round = _njit_h(cache=True, nogil=True)(_hedge_round)
    _hedge_window = _njit_h(cache=True, nogil=True)(_hedge_window)
    _hedge_fast = _njit_h(cache=True, nogil=True)(_hedge_core)
    _stances_fast = _njit_h(cache=True, nogil=True)(_expert_stances)
except Exception:  # pragma: no cover
    _hedge_fast = _hedge_core
    _stances_fast = _expert_stances


def _hedge_loss(df: pd.DataFrame, atr_n: int, mode: str, cost_bps: float = 0.0):
    """The T x N loss matrix the learner scores (in [0, 1]) plus the ladder's
    (lookbacks, sides): expert e's loss on bar t is 0.5 * (1 - payoff) where
    the payoff is the ATR-normalised move of bar t in the direction of the
    stance e held at the previous close, minus the cost (per side, in ATRs,
    as the engine charges it) of the sides e traded at the close of t to
    reach its new stance, halved and clipped to [-1, 1]."""
    lookbacks, sides = hedge_experts(mode)
    high = _to_arr(df["High"]); low = _to_arr(df["Low"]); close = _to_arr(df["Close"])
    T = len(close)
    N = len(lookbacks)
    uppers = np.empty((T, N)); lowers = np.empty((T, N))
    cache = {}
    for e, n in enumerate(lookbacks):
        if int(n) not in cache:
            up, lo, _ = donchian(df, int(n))
            cache[int(n)] = (_to_arr(up), _to_arr(lo))
        uppers[:, e], lowers[:, e] = cache[int(n)]
    S = _stances_fast(high, low, close, uppers, lowers, np.ascontiguousarray(sides))
    a = _to_arr(atr(df, atr_n))
    a_prev = np.where(a[:-1] > 0, a[:-1], np.nan)
    z = (close[1:] - close[:-1]) / a_prev                    # next-bar move, in ATRs
    c = close[1:] * (float(cost_bps) / 1e4) / a_prev         # one side's cost, in ATRs
    flips = np.abs(S[1:] - S[:-1])                            # sides traded to reach the new stance (a reversal is two)
    payoff = np.clip((S[:-1] * z[:, None] - c[:, None] * flips) / 2.0, -1.0, 1.0)
    loss = np.full((T, N), 0.5)
    loss[1:] = 0.5 * (1.0 - np.nan_to_num(payoff, nan=0.0))
    formed = ~np.isnan(uppers)           # an unformed expert has no stance: neutral loss
    loss = np.where(formed, loss, 0.5)
    return np.ascontiguousarray(loss), lookbacks, sides


def _hedge_run(df: pd.DataFrame, atr_n: int, mode: str, cost_bps: float):
    loss, _, _ = _hedge_loss(df, atr_n, mode, cost_bps)
    W, ETA, V, SURPRISE = _adahedge_loop(loss)
    return W, ETA, V, SURPRISE, loss


def _hedge_cached(df: pd.DataFrame, atr_n: int, mode: str, cost_bps: float, dfkey=None):
    """One learner run per (bars, mode, ATR length, cost): the entry channel,
    the exit channel and the learned direction all read the same weights."""
    return _cached(df, ("hedge", mode, int(atr_n), float(cost_bps)),
                   lambda: _hedge_run(df, atr_n, mode, cost_bps), dfkey)


def hedge_weights(df: pd.DataFrame, atr_n: int, mode: str = "trend", cost_bps: float = 0.0) -> np.ndarray:
    """T x n_experts learner weights, row t computed from bars <= t
    (the last HEDGE_MEMORY of them)."""
    return _hedge_cached(df, atr_n, mode, cost_bps)[0]


def hedge_diagnostics(df: pd.DataFrame, atr_n: int, mode: str = "trend", cost_bps: float = 0.0) -> dict:
    """What the learner did, bar by bar, for notebooks and dashboards:
    `weights` (expert weights, columns follow_10 / fade_20 / ...), `loss`
    (the expert losses it scored), `eta` (each lifetime's learning rate,
    columns = HEDGE_HORIZONS), `horizon_weights` (the meta learner's weights
    over the lifetimes: shorter ones gaining is the learner shortening its
    memory) and `surprise` (the loss the played mixture suffered minus the
    best expert's, in [0, 1]). Nothing here changes the strategy."""
    W, ETA, V, SURPRISE, loss = _hedge_cached(df, atr_n, mode, cost_bps)
    lookbacks, sides = hedge_experts(mode)
    experts = [f"{'follow' if s > 0 else 'fade'}_{int(n)}" for n, s in zip(lookbacks, sides)]
    idx = df.index
    return {
        "weights": pd.DataFrame(W, index=idx, columns=experts),
        "loss": pd.DataFrame(loss, index=idx, columns=experts),
        "eta": pd.DataFrame(ETA, index=idx, columns=list(HEDGE_HORIZONS)),
        "horizon_weights": pd.DataFrame(V, index=idx, columns=list(HEDGE_HORIZONS)),
        "surprise": pd.Series(SURPRISE, index=idx),
    }


def hedge_direction(df: pd.DataFrame, atr_n: int, cost_bps: float = 0.0) -> np.ndarray:
    """Per-bar direction for direction_logic 'learned': +1 follow the
    break, -1 fade it, from the net side weight of the learner over
    follow and fade experts. NaN until the learner is formed."""
    W = hedge_weights(df, atr_n, "learned", cost_bps)
    _, sides = hedge_experts("learned")
    d = np.where(W @ sides >= 0.0, 1.0, -1.0)
    d[:min(len(d), hedge_warmup(atr_n))] = np.nan
    return d


def hedge_channel(df: pd.DataFrame, atr_n: int, mode: str = "trend", scale: float = 1.0,
                  cost_bps: float = 0.0):
    """Weight-averaged Donchian channel over the expert ladder (lookbacks
    scaled by `scale`), NaN until the learner is formed."""
    W = hedge_weights(df, atr_n, mode, cost_bps)
    lookbacks, _ = hedge_experts(mode)
    T = len(df)
    up = np.zeros(T); lo = np.zeros(T)
    cache = {}
    for e, n in enumerate(lookbacks):
        m = max(2, int(round(int(n) * scale)))
        if m not in cache:
            u, l, _ = donchian(df, m)
            cache[m] = (_to_arr(u), _to_arr(l))
        up += W[:, e] * cache[m][0]
        lo += W[:, e] * cache[m][1]
    warm = min(T, hedge_warmup(atr_n))
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
    "vr":   dict(fn=lambda df, n: variance_ratio(df["Close"], n), n=60, threshold=1.0, thresholds=[0.9, 1.0, 1.1]),
}


# --------------------------------------------------------------------------
# Template / config
# --------------------------------------------------------------------------

DIRECTION_LOGICS = ["trend", "countertrend", "learned"]
CHANNEL_TYPES = ["donchian", "keltner", "bollinger", "hedge"]
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


def _channel_arrays(df, kind, n, k, atr_n, dfkey=None, mode="trend", role="entry", cost_bps=0.0):
    def build():
        up, lo, mid = channel(df, kind, n, k, atr_n, mode=mode, role=role, cost_bps=cost_bps)
        return (_to_arr(up), _to_arr(lo), _to_arr(mid))
    if kind == "hedge":
        # no lookback / width: keyed on direction mode (signed rewards), ATR
        # length, entry/exit role and the cost the experts are charged
        spec = ("channel", kind, role, mode, atr_n, float(cost_bps))
    else:
        spec = ("channel", kind, n, k if kind != "donchian" else 0.0, atr_n if kind == "keltner" else 0)
    return _cached(df, spec, build, dfkey)


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
    up, lo, _ = _channel_arrays(df, tpl.channel_type, tpl.n_entry, tpl.channel_k, tpl.atr_n, dfkey, mode, "entry",
                                tpl.cost_bps)
    use("upper", up)
    use("lower", lo)
    use("atr", _cached(df, ("atr", tpl.atr_n), lambda: _to_arr(atr(df, tpl.atr_n)), dfkey))

    # per-bar direction: +1 follow the break, -1 fade it; learned from the
    # follow/fade expert ladder (NaN while the learner is unformed), else constant
    if mode == "learned":
        use("direction", _cached(df, ("hedge_dir", tpl.atr_n, float(tpl.cost_bps)),
                                 lambda: hedge_direction(df, tpl.atr_n, tpl.cost_bps), dfkey))
    else:
        ind["direction"] = np.full(n, 1.0 if mode == "trend" else -1.0)

    if tpl.exit_style == "channel":
        upx, lox, midx = _channel_arrays(df, tpl.channel_type, tpl.n_exit, tpl.channel_k, tpl.atr_n, dfkey, mode, "exit",
                                         tpl.cost_bps)
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

    if tpl.vol_target > 0:
        # realized per-bar vol (NOT annualized, so the cached array does not
        # depend on periods_per_year()); the target is scaled to per-bar in backtest()
        use("rvol", _cached(df, ("rvol", tpl.vol_target_n),
                            lambda: _to_arr(df["Close"].pct_change().rolling(tpl.vol_target_n).std()), dfkey))

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
              direction, rvol, entry_style, exit_style, regime_mode, regime_thr,
              has_vol, vol_low, vol_high, has_bias, allow_long, allow_short,
              atr_mult_stop, atr_mult_target, atr_mult_trail, pullback_atr_mult,
              pullback_valid_bars, max_hold_bars, risk_pct, max_leverage, vol_target_bar, cost_rate,
              initial_equity, first_trade_bar):
    """The bar loop. Plain numpy code so numba can compile it unchanged;
    the pure-Python version is used when numba is not installed.

    Sizing: `risk_pct` of cash lost at the ATR stop, or, when `vol_target_bar`
    > 0, a notional of cash * vol_target_bar / rvol[i-1] (both per-bar vols);
    capped at `max_leverage` either way and fixed for the life of the trade.

    Bars before `first_trade_bar` are indicator warm-up only: nothing is
    entered on them (and no order rests on them), so the walk-forward can
    hand the loop a slice that starts before its window without any trade
    decided on the earlier bars leaking into the window's result.

    Returns equity, entries, the closed-trade columns
    (entry_bar, exit_bar, side, entry_px, exit_px, shares, pnl, cost, reason, count)
    and the loop's final state (open position, resting order, last usable ATR)."""
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
    pos_trend = True      # direction logic the OPEN trade was entered under
    pend_trend = True     # ... and the resting pullback order

    equity[0] = initial_equity
    for i in range(1, n):
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

            if exit_style == 1:
                # trailing level from the extreme up to bar i-1 (no intrabar look-ahead)
                trail_stop = trail_extreme - side * atr_mult_trail * a
                if side == 1:
                    stop_level = max(stop_level, trail_stop)
                else:
                    stop_level = min(stop_level, trail_stop)
            elif exit_style == 0 and pos_trend:
                # the opposite channel is a stop on the SAME side as the hard
                # stop; on the way through, whichever sits nearer to the price
                # is hit first, so it must fill at that level, not at the
                # further one
                if side == 1 and lower_x[i - 1] > stop_level:
                    stop_level = lower_x[i - 1]
                    stop_reason = 1
                elif side == -1 and upper_x[i - 1] < stop_level:
                    stop_level = upper_x[i - 1]
                    stop_reason = 1

            if side == 1 and low[i] <= stop_level:
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

        # ---- no usable indicators (or still warming up): manage what is open,
        # start nothing new ----
        if not a_ok or i < first_trade_bar:
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

        # the `sides` switch first, then the SMA bias narrows it further
        long_ok = allow_long
        short_ok = allow_short
        if has_bias:
            long_ok = long_ok and close[i - 1] > bias[i - 1]
            short_ok = short_ok and close[i - 1] < bias[i - 1]

        fill_side = 0
        fill_px = 0.0

        # ---- pending pullback order ----
        fill_trend = is_trend
        if position == 0 and pend_active:
            if pend_side == 1 and low[i] <= pend_level:
                fill_side = 1
                fill_px = min(open_[i], pend_level)
                fill_trend = pend_trend
                pend_active = False
            elif pend_side == -1 and high[i] >= pend_level:
                fill_side = -1
                fill_px = max(open_[i], pend_level)
                fill_trend = pend_trend
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
                    pend_trend = is_trend
                    pend_level = level - side * pullback_atr_mult * a
                    pend_expires = i + pullback_valid_bars

        # ---- open the position ----
        entered = False
        if fill_side != 0:
            stop_dist = atr_mult_stop * a
            if vol_target_bar > 0.0:
                # constant-volatility notional: the stop still sits atr_mult_stop
                # ATRs away, but the loss there is no longer risk_pct of equity
                qty = cash * (vol_target_bar / rvol[i - 1]) / fill_px
            else:
                qty = cash * risk_pct / stop_dist
            qty = min(qty, max_leverage * cash / fill_px)
            if qty > 0:
                entry_cost = cost_rate * qty * fill_px
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

    # the trailing state is returned too, so the live signal layer in live.py can
    # read the CURRENT position and resting order out of the same loop the
    # backtest runs, instead of reimplementing the rules and drifting from them
    return (equity, entries, t_entry, t_exit, t_side, t_entry_px, t_exit_px, t_shares, t_pnl, t_cost,
            t_reason, n_trades, position, shares, entry_price, stop_price, target_price, trail_extreme,
            entry_bar, entry_cost, pend_active, pend_side, pend_level, pend_expires, last_a,
            pos_trend, pend_trend)


try:  # compile the loop once per process; falls back to plain Python without numba
    from numba import njit as _njit
    _bar_loop_fast = _njit(cache=True, nogil=True)(_bar_loop)
    HAVE_NUMBA = True
except Exception:  # pragma: no cover
    _bar_loop_fast = _bar_loop
    HAVE_NUMBA = False


def backtest(df: pd.DataFrame, tpl: StrategyTemplate, initial_equity: float = 100_000.0,
             first_trade_bar: int = 0) -> dict:
    """Run `tpl` over `df` (must have Open/High/Low/Close). Returns a dict:
        equity   : pd.Series of end-of-bar equity, indexed like df
        returns  : pd.Series of per-bar simple returns of equity
        entries  : np.ndarray (1 on bars where a new trade was opened)
        trades   : list of trade dicts
        stats    : summary performance stats

    `first_trade_bar` > 0 uses the first bars as indicator warm-up only: the
    equity stays at `initial_equity` and no trade can open before that bar.
    """
    n = len(df)
    first_trade_bar = int(max(first_trade_bar, 0))
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
        float(vol_target_bar), tpl.cost_bps / 1e4, float(initial_equity), first_trade_bar,
    )
    (equity, entries, t_entry, t_exit, t_side, t_entry_px, t_exit_px, t_shares, t_pnl, t_cost,
     t_reason, n_trades, f_position, f_shares, f_entry_price, f_stop, f_target, f_trail,
     f_entry_bar, f_entry_cost, f_pend_active, f_pend_side, f_pend_level, f_pend_expires, f_last_atr,
     f_pos_trend, f_pend_trend) = out

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
    stats = performance_stats(equity, trades, initial_equity, rets, t_pnl[:n_trades],
                               (t_exit[:n_trades] - t_entry[:n_trades]).astype(float))

    open_position = None
    if f_position != 0:
        open_position = dict(
            side=int(f_position), shares=float(f_shares), entry_price=float(f_entry_price),
            entry_bar=int(f_entry_bar), entry_date=pd.Timestamp(idx_arr[int(f_entry_bar)]),
            entry_cost=float(f_entry_cost), hard_stop=float(f_stop), target=float(f_target),
            trail_extreme=float(f_trail), bars_held=int(n - 1 - int(f_entry_bar)),
            unrealized=float(f_position * f_shares * (close[-1] - f_entry_price)),
            is_trend=bool(f_pos_trend),
        )
    pending_order = None
    if f_pend_active:
        pending_order = dict(side=int(f_pend_side), level=float(f_pend_level),
                             expires_bar=int(f_pend_expires), is_trend=bool(f_pend_trend))

    return {"equity": equity_s, "returns": returns, "entries": entries, "trades": trades,
            "stats": stats, "open_position": open_position, "pending_order": pending_order,
            "last_atr": float(f_last_atr), "indicators": ind}


def annualized_sharpe(rets: pd.Series | np.ndarray, ppy: int | None = None) -> float:
    r = np.asarray(rets, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) < 2:
        return 0.0
    sd = r.std(ddof=1)
    ppy = periods_per_year() if ppy is None else ppy
    return float(r.mean() / sd * np.sqrt(ppy)) if sd > 0 else 0.0


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
        rets[1:] = eq[1:] / eq[:-1] - 1.0
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
