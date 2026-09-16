"""
live.py
-------
Turn the research pipeline into something you can act on at a broker.

`walkforward.walk_forward` answers "would this template have worked?".
This module answers the two questions you actually need each morning:

  1. WHAT AM I SUPPOSED TO BE HOLDING right now, and where is its stop?
  2. WHAT ORDER DO I PLACE for the next bar, at what price?

Both answers come out of the same `strategy.backtest` loop the research used
-- `backtest()` returns its final open position and resting order -- so the
dashboard cannot drift away from the thing that was validated. The only logic
restated here is the *next* bar's order levels (`next_bar_orders`), because
that bar does not exist yet; `tests/test_live.py` rolls one real bar forward
and asserts the engine did what this module predicted.

Two rules keep the live path honest:

* PARAMETERS ONLY CHANGE AT WINDOW BOUNDARIES. Re-optimizing every morning
  would be a different (and unvalidated) strategy from the walk-forward that
  was tested with `test_bars`-long holds. `due_for_refit` enforces the same
  cadence, and the chosen params are persisted with the bar they were fitted
  on.
* THE LAST BAR MAY STILL BE FORMING. Intraday, the most recent bar from the
  data feed is a partial bar; acting on it is look-ahead against yourself.
  `drop_forming_bar` removes it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategy import (
    StrategyTemplate, backtest, periods_per_year, _compute_indicators,
    ENTRY_CODES, EXIT_CODES,
)
from walkforward import grid_combos, optimize_window, warmup_bars


# --------------------------------------------------------------------------
# data hygiene
# --------------------------------------------------------------------------

def utcnow() -> pd.Timestamp:
    """Naive UTC now. One helper so the whole project agrees on the clock."""
    return pd.Timestamp.now("UTC").tz_convert(None)


def drop_forming_bar(df: pd.DataFrame, interval: str, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Drop the last bar if it has not closed yet.

    A feed queried at 11:15 returns an 11:00 hourly bar built from 15 minutes
    of trading. Its high, low and close will all move, so any decision taken
    on it is a decision you cannot actually have taken. Daily bars are treated
    the same way: today's bar is dropped until the session is over.
    """
    if df.empty:
        return df
    now = utcnow() if now is None else pd.Timestamp(now)
    last = df.index[-1]
    if getattr(last, "tz", None) is not None:
        last = last.tz_convert("UTC").tz_localize(None)
    step = _interval_to_timedelta(interval)
    # a bar stamped at `last` covers [last, last + step); it is only final once
    # that window has passed
    if last + step > now:
        return df.iloc[:-1]
    return df


def _interval_to_timedelta(interval: str) -> pd.Timedelta:
    table = {
        "1m": pd.Timedelta(minutes=1), "5m": pd.Timedelta(minutes=5),
        "15m": pd.Timedelta(minutes=15), "30m": pd.Timedelta(minutes=30),
        "60m": pd.Timedelta(hours=1), "1h": pd.Timedelta(hours=1),
        "90m": pd.Timedelta(minutes=90), "1d": pd.Timedelta(days=1),
        "1wk": pd.Timedelta(weeks=1), "1mo": pd.Timedelta(days=30),
    }
    if interval not in table:
        raise ValueError(f"unknown interval {interval!r}")
    return table[interval]


# --------------------------------------------------------------------------
# re-optimization, on the walk-forward's cadence
# --------------------------------------------------------------------------

def refit_params(
    df: pd.DataFrame,
    tpl: StrategyTemplate,
    param_grid: dict,
    train_bars: int = 500,
    *,
    metric: str = "sharpe",
    selection: str = "plateau",
    min_trades: int = 5,
    anchored: bool = False,
    initial_equity: float = 100_000.0,
) -> dict:
    """Re-optimize `tpl`'s numeric params on the most recent training window.

    This is exactly the in-sample step of `walkforward.walk_forward`, run on
    the last complete window, so the parameters the dashboard trades are
    chosen the same way the walk-forward chose them.
    """
    combos, idx = grid_combos(param_grid)
    start = 0 if anchored else len(df) - train_bars
    if start < 0:
        raise ValueError(f"need {train_bars} bars to refit, have {len(df)}")
    train = df.iloc[start:]

    # the window's indicators warm up on the bars before it (when there are
    # any), exactly as the walk-forward's training windows do
    opt = optimize_window(df, tpl, combos, idx, start, len(df), metric=metric, selection=selection,
                          min_trades=min_trades, initial_equity=initial_equity)
    best, scores, stats_list = opt["best"], opt["scores"], opt["stats"]
    return dict(
        params=None if best is None else combos[best],
        is_stats=None if best is None else stats_list[best],
        is_score=None if best is None else float(scores[best]),
        fitted_on=df.index[-1],
        train_start=train.index[0],
        train_bars=len(train),
        n_candidates=len(combos),
    )


def due_for_refit(df: pd.DataFrame, fitted_on, test_bars: int) -> bool:
    """True when `test_bars` bars have passed since the last refit.

    Holding parameters for exactly one test window is what makes the live run
    the same process the walk-forward measured. Refitting more often is a
    different strategy with no out-of-sample evidence behind it.
    """
    if fitted_on is None:
        return True
    fitted_on = pd.Timestamp(fitted_on)
    after = df.index[df.index > fitted_on]
    return len(after) >= test_bars


# --------------------------------------------------------------------------
# where am I, and what do I place next?
# --------------------------------------------------------------------------

def strategy_state(
    df: pd.DataFrame,
    tpl: StrategyTemplate,
    *,
    equity: float = 100_000.0,
    lookback_bars: int | None = None,
) -> dict:
    """Current position, live stop levels and the orders for the next bar.

    `tpl` must already carry the params chosen by `refit_params`. `equity` is
    the capital allocated to THIS strategy slot, so the share counts come out
    in tradeable units. `lookback_bars` limits how much history the state is
    rebuilt from (default: the training window's worth, plus warm-up).
    """
    need = warmup_bars(tpl) + 10
    if lookback_bars is None:
        lookback_bars = max(need, 400)
    tail = df.iloc[-max(lookback_bars, need):]
    if len(tail) < need:
        raise ValueError(f"need at least {need} bars of history for {tpl.name}, have {len(tail)}")

    res = backtest(tail, tpl, initial_equity=equity)
    ind = res["indicators"]
    n = len(tail)
    pos = res["open_position"]

    state = dict(
        template=tpl.name,
        as_of=tail.index[-1],
        last_close=float(tail["Close"].iloc[-1]),
        atr=float(ind["atr"][n - 1]),
        # the multiple the hard stop sits at, so a reader can express "how close
        # is this to its stop" as a fraction of the distance it started with
        atr_mult_stop=float(tpl.atr_mult_stop),
        exit_style=tpl.exit_style,
        equity_slot=float(equity),
        position=None if pos is None else pos["side"],
        shares=0.0 if pos is None else pos["shares"],
        entry_price=None if pos is None else pos["entry_price"],
        entry_date=None if pos is None else pos["entry_date"],
        bars_held=0 if pos is None else pos["bars_held"],
        unrealized=0.0 if pos is None else pos["unrealized"],
        n_trades_in_window=len(res["trades"]),
    )
    state["exit_orders"] = _exit_orders(tail, tpl, ind, pos) if pos else []
    state["entry_orders"] = [] if pos else _entry_orders(tail, tpl, ind, res["pending_order"], equity)
    state["blocked_by"] = _filter_block(tail, tpl, ind)
    return state


def _last_ready(ind: dict, n: int) -> bool:
    return bool(ind["ready"][n - 1]) and ind["atr"][n - 1] > 0


def _filter_block(df: pd.DataFrame, tpl: StrategyTemplate, ind: dict) -> list:
    """Which entry filters are switched off for the next bar (and why)."""
    n = len(df)
    out = []
    if not _last_ready(ind, n):
        out.append("indicators not fully formed on the last bar")
        return out
    if tpl.regime_filter != "none":
        v = float(ind["regime"][n - 1])
        thr = float(ind["regime_threshold"])
        trending = v >= thr
        if tpl.regime_filter == "trend_only" and not trending:
            out.append(f"regime {tpl.regime_indicator}={v:.2f} < {thr:.2f} (needs trending)")
        if tpl.regime_filter == "range_only" and trending:
            out.append(f"regime {tpl.regime_indicator}={v:.2f} >= {thr:.2f} (needs ranging)")
    if tpl.vol_filter:
        r = float(ind["vol_rank"][n - 1])
        if not (tpl.vol_low_pct <= r <= tpl.vol_high_pct):
            out.append(f"ATR percentile {r:.0%} outside "
                       f"[{tpl.vol_low_pct:.0%}, {tpl.vol_high_pct:.0%}]")
    return out


def _bias(df: pd.DataFrame, tpl: StrategyTemplate, ind: dict) -> tuple[bool, bool]:
    """(long allowed, short allowed) under the `sides` switch and the SMA bias
    filter, in the same order the engine applies them."""
    long_ok, short_ok = tpl.sides != "short_only", tpl.sides != "long_only"
    if tpl.bias_filter != "sma":
        return long_ok, short_ok
    n = len(df)
    c, b = float(df["Close"].iloc[-1]), float(ind["bias"][n - 1])
    return long_ok and c > b, short_ok and c < b


def _entry_orders(df, tpl: StrategyTemplate, ind: dict, pending, equity: float) -> list:
    """The order(s) to have working on the next bar while flat.

    Mirrors the entry branch of `strategy._bar_loop` one bar into the future:
    the loop decides bar i from the channel at i-1, so the levels below are the
    channel on the LAST CLOSED bar.
    """
    n = len(df)
    if pending is not None:
        side = pending["side"]
        return [dict(kind="limit", side=side, level=pending["level"],
                     shares=_size(equity, tpl, ind, n, pending["level"]),
                     note=f"pullback limit already working, expires in "
                          f"{max(pending['expires_bar'] - (n - 1), 0)} bar(s)")]
    if not _last_ready(ind, n) or _filter_block(df, tpl, ind):
        return []

    upper, lower = float(ind["upper"][n - 1]), float(ind["lower"][n - 1])
    long_ok, short_ok = _bias(df, tpl, ind)
    is_trend = tpl.direction_logic == "trend"
    # which channel edge opens a long, and which opens a short
    long_level, short_level = (upper, lower) if is_trend else (lower, upper)
    out = []

    if tpl.entry_style == "close_confirm":
        # the break is already decided by the last close: this is a market
        # order on the next open, or nothing at all
        if n < 3 or not ind["ready"][n - 2]:
            return []
        c = float(df["Close"].iloc[-1])
        broke_up = c > float(ind["upper"][n - 2])
        broke_down = c < float(ind["lower"][n - 2])
        take_long = (broke_up if is_trend else broke_down) and long_ok
        take_short = (broke_down if is_trend else broke_up) and short_ok
        if take_long:
            out.append(dict(kind="market_on_open", side=1, level=None,
                            shares=_size(equity, tpl, ind, n, float(df["Close"].iloc[-1])),
                            note="close confirmed beyond the channel"))
        elif take_short:
            out.append(dict(kind="market_on_open", side=-1, level=None,
                            shares=_size(equity, tpl, ind, n, float(df["Close"].iloc[-1])),
                            note="close confirmed beyond the channel"))
        return out

    # 'stop' fills AT the channel edge; 'pullback' waits for the break, then
    # places a limit that far inside it
    a = float(ind["atr"][n - 1])
    for side, level, ok in ((1, long_level, long_ok), (-1, short_level, short_ok)):
        if not ok:
            continue
        if tpl.entry_style == "pullback":
            out.append(dict(
                kind="stop_then_limit", side=side, level=level,
                limit=level - side * tpl.pullback_atr_mult * a,
                shares=_size(equity, tpl, ind, n, level),
                note=f"on a break of {level:.2f}, work a limit at "
                     f"{level - side * tpl.pullback_atr_mult * a:.2f} for "
                     f"{tpl.pullback_valid_bars} bar(s)"))
        else:
            # a trend break is a stop order (fills as price runs through the
            # level); fading it is a limit order (fills as price reaches it)
            out.append(dict(kind="stop" if is_trend else "limit", side=side, level=level,
                            shares=_size(equity, tpl, ind, n, level),
                            note="breakout" if is_trend else "fade the break"))
    return out


def _size(equity: float, tpl: StrategyTemplate, ind: dict, n: int, price: float) -> float:
    """Shares the engine would buy: fixed fractional risk on the ATR stop,
    capped by the leverage limit. Same formula as `strategy._bar_loop`."""
    a = float(ind["atr"][n - 1])
    if not (a > 0) or not (price > 0):
        return 0.0
    qty = equity * tpl.risk_pct / (tpl.atr_mult_stop * a)
    return float(max(min(qty, tpl.max_leverage * equity / price), 0.0))


def _exit_orders(df, tpl: StrategyTemplate, ind: dict, pos: dict) -> list:
    """Every exit level live on the next bar for an open position.

    The hard ATR stop is always one of them. Levels are quoted as the engine
    would use them on the next bar, i.e. from the last closed bar's indicators.
    """
    n = len(df)
    side = pos["side"]
    a = float(ind["atr"][n - 1])
    out = [dict(kind="stop", side=-side, level=float(pos["hard_stop"]), note="hard ATR stop")]

    if tpl.exit_style == "atr_trail":
        trail = pos["trail_extreme"] - side * tpl.atr_mult_trail * a
        level = max(pos["hard_stop"], trail) if side == 1 else min(pos["hard_stop"], trail)
        out = [dict(kind="stop", side=-side, level=float(level),
                    note=f"chandelier: {tpl.atr_mult_trail:g} ATR from "
                         f"{pos['trail_extreme']:.2f} (hard stop {pos['hard_stop']:.2f})")]
    elif tpl.exit_style == "channel":
        if tpl.direction_logic == "trend":
            lvl = float(ind["lower_x"][n - 1] if side == 1 else ind["upper_x"][n - 1])
            out.append(dict(kind="stop", side=-side, level=lvl,
                            note=f"opposite {tpl.n_exit}-bar channel (Turtle exit)"))
        else:
            out.append(dict(kind="limit", side=-side, level=float(ind["mid_x"][n - 1]),
                            note=f"{tpl.n_exit}-bar channel midline (mean-reversion target)"))
    elif tpl.exit_style == "target_stop":
        out.append(dict(kind="limit", side=-side, level=float(pos["target"]),
                        note=f"{tpl.atr_mult_target:g} ATR target"))
    elif tpl.exit_style == "time_stop":
        left = tpl.max_hold_bars - pos["bars_held"]
        out.append(dict(kind="market_on_open", side=-side, level=None,
                        note=f"time stop: exit at the open in {max(left, 0)} bar(s)"
                             if left > 0 else "time stop: exit at the next open"))
    return out


# --------------------------------------------------------------------------
# from per-slot states to one account-level book
# --------------------------------------------------------------------------

def portfolio_targets(
    states: list,
    weights: dict,
    account_equity: float,
    *,
    max_gross: float | None = None,
) -> dict:
    """Aggregate per-slot states into target positions per asset.

    Each slot (asset, template) is sized on `account_equity * weights[slot]`,
    which is what the portfolio step allocated to it. Several slots can be long
    the same ETF at once, so positions are netted per asset and the gross book
    is reported: if it exceeds `max_gross` (as a fraction of the account) every
    position is scaled down by the same factor, which keeps the relative book
    intact while respecting the account's real limit.
    """
    rows = []
    for st in states:
        key = st["slot"]
        w = float(weights.get(key, 0.0))
        if st["position"] is None or w <= 0:
            continue
        rows.append(dict(
            slot=key, asset=st["asset"], template=st["template"], weight=w,
            side=st["position"], shares=st["shares"], price=st["last_close"],
            notional=st["shares"] * st["last_close"] * st["position"],
            entry_price=st["entry_price"], entry_date=st["entry_date"],
            unrealized=st["unrealized"],
            stop=min((o["level"] for o in st["exit_orders"]
                      if o["kind"] == "stop" and o["level"] is not None), default=None)
            if st["position"] == -1 else
            max((o["level"] for o in st["exit_orders"]
                 if o["kind"] == "stop" and o["level"] is not None), default=None),
        ))
    legs = pd.DataFrame(rows)
    scale = 1.0
    gross = float(legs["notional"].abs().sum()) if len(legs) else 0.0
    if max_gross is not None and gross > max_gross * account_equity > 0:
        scale = max_gross * account_equity / gross
        legs["shares"] *= scale
        legs["notional"] *= scale

    if len(legs):
        by_asset = legs.assign(signed=legs["shares"] * legs["side"]).groupby("asset").agg(
            shares=("signed", "sum"), price=("price", "last"))
        by_asset["notional"] = by_asset["shares"] * by_asset["price"]
        by_asset["pct_of_account"] = by_asset["notional"] / account_equity
    else:
        by_asset = pd.DataFrame(columns=["shares", "price", "notional", "pct_of_account"])

    risk = 0.0
    for r in legs.itertuples() if len(legs) else ():
        if r.stop is not None and not np.isnan(r.stop):
            risk += abs(r.shares) * abs(r.price - r.stop)
    return dict(
        legs=legs, by_asset=by_asset, scale_applied=scale,
        gross_before_scaling=gross,
        gross_exposure=float(legs["notional"].abs().sum()) / account_equity if len(legs) else 0.0,
        net_exposure=float(legs["notional"].sum()) / account_equity if len(legs) else 0.0,
        open_risk=risk, open_risk_pct=risk / account_equity if account_equity else 0.0,
    )


def trade_list(by_asset: pd.DataFrame, holdings: dict, lot: float = 1.0) -> pd.DataFrame:
    """Target minus held, per asset: the orders to send.

    `holdings` is what you actually have at the broker (signed share counts).
    `lot` rounds the order size; shares below one lot are reported as 'hold'
    so a 3-share drift does not generate a trade every morning.
    """
    assets = sorted(set(by_asset.index) | set(holdings))
    if not assets:
        # a flat book with nothing held: an empty frame still has to carry the
        # columns, or every consumer of it blows up on a quiet day
        return pd.DataFrame(columns=["held", "target", "delta", "action", "order_shares",
                                     "price", "order_notional"],
                            index=pd.Index([], name="asset"))
    rows = []
    for a in assets:
        target = float(by_asset["shares"].get(a, 0.0))
        held = float(holdings.get(a, 0.0))
        delta = target - held
        rounded = np.sign(delta) * (abs(delta) // lot) * lot if lot > 0 else delta
        price = float(by_asset["price"].get(a, np.nan))
        rows.append(dict(
            asset=a, held=held, target=round(target, 2), delta=round(delta, 2),
            action="hold" if rounded == 0 else ("BUY" if rounded > 0 else "SELL"),
            order_shares=abs(rounded), price=price,
            order_notional=abs(rounded) * price if np.isfinite(price) else np.nan,
        ))
    return pd.DataFrame(rows).set_index("asset")
