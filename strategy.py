"""
strategy.py
-----------
A Ranger-style "breakout strategy generator" building block.

Instead of one fixed system, a strategy is defined by a set of
structural SWITCHES (categorical choices that change the logic
entirely) plus a small set of NUMERIC PARAMS (tuned by walk-forward
analysis in walkforward.py):

Switches (define a "template" -- a structurally distinct strategy):
  direction_logic : 'trend'        -> classic breakout, trade WITH the
                                       break (buy new highs, sell new lows)
                    'countertrend' -> fade the break (sell new highs,
                                       buy new lows), i.e. range/reversion
  entry_style     : 'stop'          -> enter the moment the breakout level trades
                    'pullback'      -> wait for price to pull back
                                    `pullback_atr_mult` * ATR from the
                                    breakout level before entering (limit
                                    order in the direction of the trade)
  exit_style      : 'opposite'   -> exit on the opposite Donchian channel
                                    (classic channel-in / channel-out)
                     'atr_trail' -> ATR trailing stop
                     'target_stop' -> fixed R-multiple stop and target
  regime_filter   : 'none'       -> trade in any regime
                     'trend_only'-> only trade when Kaufman Efficiency
                                    Ratio (ER) is ABOVE threshold (market
                                    is trending)
                     'range_only'-> only trade when ER is BELOW threshold
                                    (market is choppy/sideways) -- this is
                                    Ranger's "trade only when the market
                                    moves sidewards" mode
  vol_filter      : True/False ->  skip new entries when ATR is in an
                                   extreme percentile (too dead or too wild)

Numeric params (tunable, walk-forward optimized):
  n_entry              : entry channel lookback (bars)
  n_exit               : exit channel lookback, used by 'opposite' exit
  atr_n                : ATR lookback
  atr_mult_stop         : initial stop distance, in ATR multiples
  atr_mult_target       : profit target distance, in ATR multiples ('target_stop')
  atr_mult_trail        : trailing stop distance, in ATR multiples ('atr_trail')
  pullback_atr_mult     : pullback depth required before entry ('pullback' style)
  er_lookback           : Efficiency Ratio lookback (regime filter)
  er_threshold          : ER threshold (0..1) separating trend/range
  risk_pct              : fraction of equity risked per trade (position sizing)
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from itertools import product
import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# Indicators
# --------------------------------------------------------------------------

def atr(df: pd.DataFrame, n: int) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.rolling(n).mean()


def efficiency_ratio(close: pd.Series, n: int) -> pd.Series:
    """Kaufman's Efficiency Ratio: net move / sum of absolute moves over n bars.
    ~1.0 = strongly trending (efficient), ~0.0 = choppy/sideways."""
    change = (close - close.shift(n)).abs()
    volatility = close.diff().abs().rolling(n).sum()
    er = change / volatility.replace(0, np.nan)
    return er.clip(0, 1)


def donchian(df: pd.DataFrame, n: int):
    upper = df["High"].rolling(n).max()
    lower = df["Low"].rolling(n).min()
    return upper, lower


# --------------------------------------------------------------------------
# Template / config
# --------------------------------------------------------------------------

DIRECTION_LOGICS = ["trend", "countertrend"]
ENTRY_STYLES = ["stop", "pullback"]
EXIT_STYLES = ["opposite", "atr_trail", "target_stop"]
REGIME_FILTERS = ["none", "trend_only", "range_only"]
VOL_FILTERS = [False, True]


@dataclass
class StrategyTemplate:
    """A structurally distinct strategy: the categorical switches, plus
    default numeric params (these defaults get overridden per-window by
    walk-forward optimization -- see walkforward.py)."""

    name: str
    direction_logic: str = "trend"
    entry_style: str = "stop"
    exit_style: str = "opposite"
    regime_filter: str = "none"
    vol_filter: bool = False

    # numeric params / defaults (subject to WFA tuning)
    n_entry: int = 40
    n_exit: int = 20
    atr_n: int = 20
    atr_mult_stop: float = 3.0
    atr_mult_target: float = 4.0
    atr_mult_trail: float = 3.0
    pullback_atr_mult: float = 0.5
    er_lookback: int = 20
    er_threshold: float = 0.35
    vol_lookback: int = 100
    vol_low_pct: float = 0.10
    vol_high_pct: float = 0.90
    risk_pct: float = 0.01

    def with_params(self, **kwargs) -> "StrategyTemplate":
        d = asdict(self)
        d.update(kwargs)
        return StrategyTemplate(**d)


# --------------------------------------------------------------------------
# Backtest engine (single asset, bar-by-bar for correct stateful exits)
# --------------------------------------------------------------------------
def backtest(df: pd.DataFrame, tpl: StrategyTemplate, initial_equity: float = 100_000.0):
    """Run `tpl` over `df` (must have Open/High/Low/Close). Returns a dict:
        equity   : pd.Series of daily equity, indexed like df
        trades   : list of trade dicts
        stats    : summary performance stats
    Entries/exits execute on the bar AFTER the signal bar (next Open),
    to avoid look-ahead bias.
    """
    n = len(df)
    close = df["Close"].values
    open_ = df["Open"].values
    high = df["High"].values
    low = df["Low"].values

    upper, lower = donchian(df, tpl.n_entry)
    upper_x, lower_x = donchian(df, tpl.n_exit)
    atr_series = atr(df, tpl.atr_n)
    er_series = efficiency_ratio(df["Close"], tpl.er_lookback)
    vol_rank = atr_series.rolling(tpl.vol_lookback).rank(pct=True)

    upper_v, lower_v = upper.values, lower.values
    upper_xv, lower_xv = upper_x.values, lower_x.values
    atr_v = atr_series.values
    er_v = er_series.values
    vol_v = vol_rank.values

    equity = np.full(n, np.nan)
    equity[0] = initial_equity
    cash_equity = initial_equity

    position = 0        # -1, 0, +1
    shares = 0.0
    entry_price = np.nan
    stop_price = np.nan
    target_price = np.nan
    trail_extreme = np.nan
    pending_order = None  # dict: {'side':..., 'level':...} for pullback entries

    trades = []
    open_trade = None

    warmup = max(tpl.n_entry, tpl.n_exit, tpl.atr_n, tpl.er_lookback, tpl.vol_lookback) + 2

    for i in range(1, n):
        equity[i] = cash_equity + (position * shares * (close[i - 1] - entry_price) if position != 0 else 0.0)

        if i < warmup:
            continue

        a = atr_v[i - 1]
        if np.isnan(a) or a <= 0:
            continue

        # ---- regime / vol filters (based on previous bar's fully-formed values) ----
        allowed_regime = True
        if tpl.regime_filter == "trend_only":
            allowed_regime = (not np.isnan(er_v[i - 1])) and er_v[i - 1] >= tpl.er_threshold
        elif tpl.regime_filter == "range_only":
            allowed_regime = (not np.isnan(er_v[i - 1])) and er_v[i - 1] < tpl.er_threshold

        allowed_vol = True
        if tpl.vol_filter:
            vr = vol_v[i - 1]
            allowed_vol = (not np.isnan(vr)) and (tpl.vol_low_pct <= vr <= tpl.vol_high_pct)

        can_enter = allowed_regime and allowed_vol

        # ---- manage open position: exits (checked intrabar on bar i) ----
        if position != 0:
            exit_price = None
            reason = None

            if tpl.exit_style == "opposite":
                if position == 1 and low[i] <= lower_xv[i - 1]:
                    exit_price = min(open_[i], lower_xv[i - 1])
                    reason = "opposite_channel"
                elif position == -1 and high[i] >= upper_xv[i - 1]:
                    exit_price = max(open_[i], upper_xv[i - 1])
                    reason = "opposite_channel"

            elif tpl.exit_style == "target_stop":
                if position == 1:
                    if low[i] <= stop_price:
                        exit_price = min(open_[i], stop_price)
                        reason = "stop"
                    elif high[i] >= target_price:
                        exit_price = max(open_[i], target_price)
                        reason = "target"
                else:
                    if high[i] >= stop_price:
                        exit_price = max(open_[i], stop_price)
                        reason = "stop"
                    elif low[i] <= target_price:
                        exit_price = min(open_[i], target_price)
                        reason = "target"

            elif tpl.exit_style == "atr_trail":
                if position == 1:
                    trail_extreme = max(trail_extreme, high[i])
                    trail_stop = trail_extreme - tpl.atr_mult_trail * a
                    if low[i] <= trail_stop:
                        exit_price = min(open_[i], trail_stop)
                        reason = "trail"
                    elif low[i] <= stop_price:
                        exit_price = min(open_[i], stop_price)
                        reason = "stop"
                else:
                    trail_extreme = min(trail_extreme, low[i])
                    trail_stop = trail_extreme + tpl.atr_mult_trail * a
                    if high[i] >= trail_stop:
                        exit_price = max(open_[i], trail_stop)
                        reason = "trail"
                    elif high[i] >= stop_price:
                        exit_price = max(open_[i], stop_price)
                        reason = "stop"

            # always also respect hard stop for opposite-channel exit style (risk control)
            if reason is None and tpl.exit_style == "opposite":
                if position == 1 and low[i] <= stop_price:
                    exit_price = min(open_[i], stop_price)
                    reason = "stop"
                elif position == -1 and high[i] >= stop_price:
                    exit_price = max(open_[i], stop_price)
                    reason = "stop"

            if exit_price is not None:
                pnl = position * shares * (exit_price - entry_price)
                cash_equity += pnl
                open_trade.update(exit_date=df.index[i], exit_price=exit_price, reason=reason, pnl=pnl)
                trades.append(open_trade)
                open_trade = None
                position = 0
                shares = 0.0
                entry_price = stop_price = target_price = trail_extreme = np.nan
                continue  # no re-entry on same bar

        # ---- pending pullback order management ----
        if position == 0 and pending_order is not None:
            side, level = pending_order["side"], pending_order["level"]
            filled = None
            if side == 1 and low[i] <= level:
                filled = min(open_[i], level)
            elif side == -1 and high[i] >= level:
                filled = max(open_[i], level)
            if filled is not None:
                position = side
                entry_price = filled
                risk_amt = cash_equity * tpl.risk_pct
                shares = risk_amt / (tpl.atr_mult_stop * a)
                if position == 1:
                    stop_price = entry_price - tpl.atr_mult_stop * a
                    target_price = entry_price + tpl.atr_mult_target * a
                else:
                    stop_price = entry_price + tpl.atr_mult_stop * a
                    target_price = entry_price - tpl.atr_mult_target * a
                trail_extreme = entry_price
                open_trade = dict(entry_date=df.index[i], side=position, entry_price=entry_price)
                pending_order = None
            else:
                pending_order = None  # pullback order valid for one bar only

        # ---- new entry signals (based on previous bar's channel) ----
        if position == 0 and pending_order is None and can_enter:
            broke_up = high[i] >= upper_v[i - 1] if not np.isnan(upper_v[i - 1]) else False
            broke_down = low[i] <= lower_v[i - 1] if not np.isnan(lower_v[i - 1]) else False

            # The channel that was actually broken determines the fill level.
            # 'trend' logic trades WITH the break (breakout up -> long at the
            # upper channel). 'countertrend' logic FADES the break (breakout
            # up -> short at the upper channel; breakdown -> long at the
            # lower channel) -- the level always corresponds to whichever
            # channel triggered the signal, never picked from `side`.
            side, level = 0, None
            if tpl.direction_logic == "trend":
                if broke_up:
                    side, level = 1, upper_v[i - 1]
                elif broke_down:
                    side, level = -1, lower_v[i - 1]
            else:  # countertrend
                if broke_down:
                    side, level = 1, lower_v[i - 1]
                elif broke_up:
                    side, level = -1, upper_v[i - 1]

            if side != 0:
                if tpl.entry_style == "stop":
                    # For trend logic, breakout entries use STOP orders
                    # For countertrend logic, fading entries use LIMIT orders
                    if tpl.direction_logic == "trend":
                        fill = max(open_[i], level) if side == 1 else min(open_[i], level)
                    else:
                        fill = min(open_[i], level) if side == 1 else max(open_[i], level)
                    
                    position = side
                    entry_price = fill
                    risk_amt = cash_equity * tpl.risk_pct
                    shares = risk_amt / (tpl.atr_mult_stop * a)
                    if position == 1:
                        stop_price = entry_price - tpl.atr_mult_stop * a
                        target_price = entry_price + tpl.atr_mult_target * a
                    else:
                        stop_price = entry_price + tpl.atr_mult_stop * a
                        target_price = entry_price - tpl.atr_mult_target * a
                    trail_extreme = entry_price
                    open_trade = dict(entry_date=df.index[i], side=position, entry_price=entry_price)
                else:  # pullback: arm a limit order for the *next* bar
                    pullback_level = (
                        level - tpl.pullback_atr_mult * a
                        if side == 1
                        else level + tpl.pullback_atr_mult * a
                    )
                    pending_order = {"side": side, "level": pullback_level}

    # close any open position at the last close (mark-to-market, not a real trade)
    equity[-1] = cash_equity + (position * shares * (close[-1] - entry_price) if position != 0 else 0.0)

    equity_s = pd.Series(equity, index=df.index).ffill().fillna(initial_equity)
    stats = _performance_stats(equity_s, trades, initial_equity)
    return {"equity": equity_s, "trades": trades, "stats": stats}


def _performance_stats(equity: pd.Series, trades: list, initial_equity: float) -> dict:
    rets = equity.pct_change().dropna()
    n_years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-6)
    total_return = equity.iloc[-1] / initial_equity - 1
    cagr = (equity.iloc[-1] / initial_equity) ** (1 / n_years) - 1 if equity.iloc[-1] > 0 else -1
    sharpe = (rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0.0
    running_max = equity.cummax()
    dd = equity / running_max - 1
    max_dd = dd.min()
    wins = [t for t in trades if t.get("pnl", 0) > 0]
    win_rate = len(wins) / len(trades) if trades else 0.0
    return dict(
        total_return=total_return,
        cagr=cagr,
        sharpe=sharpe,
        max_drawdown=max_dd,
        n_trades=len(trades),
        win_rate=win_rate,
    )
