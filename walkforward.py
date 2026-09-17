"""
walkforward.py
---------------
Pardo's core contribution: Walk-Forward Analysis (WFA).

For a given StrategyTemplate (fixed switches), slide a training window
across history, re-optimize the template's numeric params on that
window, apply the chosen params to the following unseen test window,
then roll forward. Stitching all the out-of-sample (OOS) test segments
together gives one continuous equity curve that was NEVER built with
hindsight over that period.

What is here beyond the textbook loop:

* rolling OR anchored (expanding) training windows
* optional embargo gap between the end of training and the start of
  testing (Lopez de Prado, AFML ch. 7) -- not needed for plain forward
  WFA, but handy when the test block is used as training data later
* 'plateau' parameter selection: instead of the single best grid point,
  pick the point whose grid NEIGHBOURHOOD scores best. Pardo calls the
  single-best choice "peak picking" and warns against it; a parameter
  set surrounded by good neighbours is far more likely to survive OOS.
* Pardo's Walk-Forward Efficiency (annualized OOS return / annualized
  IS return) and his acceptance criteria: WFE >= 50 %, a majority of
  profitable OOS windows, OOS profitable overall.
* a walk-forward MATRIX (Pardo) / "WFO profile" (financial-hacker):
  the same template re-run over a grid of train/test lengths -- a robust
  template is profitable across most cells, not only at one setting.
"""

from __future__ import annotations
from itertools import product as iproduct
import numpy as np
import pandas as pd

from strategy import (
    backtest, annualized_sharpe, periods_per_year, performance_stats, REGIME_INDICATORS, hedge_warmup,
)


# --------------------------------------------------------------------------
# grid helpers
# --------------------------------------------------------------------------

def grid_combos(grid: dict):
    """Return (list of param dicts, int array N x D of lattice indices)."""
    keys = list(grid.keys())
    values = [list(v) for v in grid.values()]
    combos, idx = [], []
    for ix in iproduct(*[range(len(v)) for v in values]):
        combos.append({k: values[d][i] for d, (k, i) in enumerate(zip(keys, ix))})
        idx.append(ix)
    return combos, np.array(idx, dtype=int).reshape(len(combos), len(keys))


def _ema_settle_bars(alpha: float, tol: float = 1e-4) -> int:
    """Bars until an EMA started cold has forgotten its seed to within `tol`.

    A rolling window is exact once it is full; an exponential average never
    is -- it carries (1 - alpha)^t of its (arbitrary) first value forever. A
    walk-forward window backtested from a cold start therefore sees a Keltner
    channel or an ADX that differs from the one a trader with full history
    sees, unless the warm-up runs long enough for that residual to vanish.
    """
    return int(np.ceil(np.log(tol) / np.log1p(-alpha)))


def warmup_bars(tpl) -> int:
    """Bars needed before the first bar on which `tpl` can trade AND on which
    every indicator it uses matches what a full-history run would show."""
    need = [tpl.n_entry, tpl.atr_n]
    if tpl.channel_type == "keltner":
        need.append(_ema_settle_bars(2.0 / (tpl.n_entry + 1)))
    if tpl.channel_type == "hedge" or tpl.direction_logic == "learned":
        # the online learner scores a fixed number of past bars, each of
        # which needs formed experts and a formed ATR (a rolling window: exact
        # once it is full)
        need.append(hedge_warmup(tpl.atr_n))
    if tpl.exit_style == "channel":
        need.append(tpl.n_exit)
        if tpl.channel_type == "keltner":
            need.append(_ema_settle_bars(2.0 / (tpl.n_exit + 1)))
    if tpl.regime_filter != "none":
        spec = REGIME_INDICATORS[tpl.regime_indicator]
        rn = tpl.regime_n or spec["n"]
        if tpl.regime_indicator == "adx":
            # Wilder smoothing (alpha = 1/n) applied twice: the DX smoothing
            # only starts once the DI smoothing has settled
            need.append(rn + 2 * _ema_settle_bars(1.0 / rn))
        else:
            need.append(rn + 10)
    if tpl.vol_filter:
        need.append(tpl.vol_lookback + tpl.atr_n)
    if tpl.bias_filter == "sma":
        need.append(tpl.bias_n)
    return int(max(need)) + 5


def score_stats(stats: dict, metric: str, min_trades: int) -> float:
    if stats["n_trades"] < min_trades:
        return -np.inf  # not enough trades to trust this window
    if metric == "sharpe":
        return stats["sharpe"]
    if metric == "return_over_dd":
        dd = abs(stats["max_drawdown"]) or 1e-6
        return stats["total_return"] / dd
    if metric == "profit_factor":
        return min(stats["profit_factor"], 10.0)
    raise ValueError(f"unknown metric {metric}")


def smooth_scores(scores: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Average each grid point's score with its lattice neighbours
    (Chebyshev distance <= 1 in index space). Points that scored -inf
    (too few trades) stay -inf and are excluded from neighbours' means."""
    finite = np.isfinite(scores)
    out = np.full_like(scores, -np.inf, dtype=float)
    for i in range(len(scores)):
        if not finite[i]:
            continue
        nb = np.all(np.abs(idx - idx[i]) <= 1, axis=1) & finite
        out[i] = scores[nb].mean()
    return out


def select_params(scores: np.ndarray, idx: np.ndarray, method: str = "plateau") -> int | None:
    if not np.isfinite(scores).any():
        return None
    if method == "best":
        return int(np.argmax(scores))
    if method == "plateau":
        return int(np.argmax(smooth_scores(scores, idx)))
    raise ValueError(f"unknown selection method {method}")


def _annualized_return(stats: dict) -> float:
    n = max(stats.get("n_bars", 1), 1)
    return stats["total_return"] * periods_per_year() / n


# --------------------------------------------------------------------------
# one window, warmed up but traded only inside itself
# --------------------------------------------------------------------------

def window_backtest(df: pd.DataFrame, tpl, start: int, end: int, *,
                    warmup: int | None = None, initial_equity: float = 100_000.0) -> dict:
    """Backtest `tpl` on the bars df.iloc[start:end], with the indicators
    warmed up on the bars before `start` but NO trade opened before it.

    This is the one way both halves of the walk-forward run a window:

    * a cold start inside the window throws away its first `warmup_bars`
      bars (and a different number of them for every parameter set, which
      makes the in-sample scores of a grid non-comparable);
    * a warm start that is allowed to trade on the buffer opens positions
      BEFORE the window, chosen by parameters that were fitted on exactly
      those bars, and their P&L then lands in the window's "out-of-sample"
      return.

    Returns the `backtest` dict with `returns`, `equity` and `stats` cut to
    the window (the equity starts the window at `initial_equity`; all trades
    lie inside it) plus `window_start`, the window's first timestamp.
    """
    n = len(df)
    start, end = int(start), int(min(end, n))
    warmup = warmup_bars(tpl) if warmup is None else int(warmup)
    buf = max(0, start - warmup)
    res = backtest(df.iloc[buf:end], tpl, initial_equity=initial_equity, first_trade_bar=start - buf)
    off = start - buf
    eq = res["equity"].to_numpy()[off:]
    out = dict(res)
    out["equity"] = res["equity"].iloc[off:]
    out["returns"] = res["returns"].iloc[off:]
    out["entries"] = res["entries"][off:]
    out["stats"] = performance_stats(eq, res["trades"], initial_equity)
    out["window_start"] = df.index[start]
    return out


def optimize_window(df: pd.DataFrame, tpl, combos: list, idx: np.ndarray, start: int, end: int, *,
                    metric: str = "sharpe", selection: str = "plateau", min_trades: int = 5,
                    initial_equity: float = 100_000.0) -> dict:
    """The in-sample step: score every parameter set of `combos` on the
    window df.iloc[start:end] (each warmed up on the bars before it) and pick
    one with `selection`. Shared by the walk-forward loop and the live refit
    so the two cannot choose parameters differently.

    Returns dict(best=index into combos or None, scores, stats)."""
    warm = max(warmup_bars(tpl.with_params(**p)) for p in combos)
    scores = np.empty(len(combos))
    stats_list = []
    for j, params in enumerate(combos):
        res = window_backtest(df, tpl.with_params(**params), start, end, warmup=warm,
                              initial_equity=initial_equity)
        stats_list.append(res["stats"])
        scores[j] = score_stats(res["stats"], metric, min_trades)
    best = select_params(scores, idx, selection)
    return dict(best=best, scores=scores, stats=stats_list, warmup=warm)


# --------------------------------------------------------------------------
# the walk-forward loop
# --------------------------------------------------------------------------

def walk_forward(
    df: pd.DataFrame,
    tpl,
    param_grid: dict,
    train_bars: int = 500,
    test_bars: int = 125,
    *,
    anchored: bool = False,
    embargo_bars: int = 0,
    metric: str = "sharpe",
    selection: str = "plateau",
    min_trades: int = 5,
    min_test_bars: int = 20,
    initial_equity: float = 100_000.0,
) -> dict:
    """Walk-forward optimization of `tpl` over `df`.

    Returns dict with:
      oos_returns : pd.Series of per-bar OOS returns (0 on bars of windows
                    that could not be optimized -- flat, not missing)
      oos_equity  : pd.Series, stitched OOS equity (compounded from returns)
      windows     : list of per-window dicts (dates, chosen params, IS/OOS stats)
      boundaries  : list of test-window start dates (for nested selection)
      summary     : robustness summary incl. Pardo's WFE and criteria
      template    : tpl
    """
    n = len(df)
    combos, idx = grid_combos(param_grid)
    windows = []
    oos_ret_parts = []
    boundaries = []
    prev_params = None

    test_start = train_bars + embargo_bars
    while test_start < n:
        test_end = min(test_start + test_bars, n)
        if test_end - test_start < min_test_bars:
            break
        train_end = test_start - embargo_bars
        train_start = 0 if anchored else max(0, train_end - train_bars)

        # ---- optimize on the training window ----
        opt = optimize_window(df, tpl, combos, idx, train_start, train_end, metric=metric,
                              selection=selection, min_trades=min_trades, initial_equity=initial_equity)
        best, scores, stats_list = opt["best"], opt["scores"], opt["stats"]

        boundaries.append(df.index[test_start])
        oos_slice_index = df.index[test_start:test_end]

        if best is None:
            # nothing traded enough in-sample: stay flat this window
            oos_ret_parts.append(pd.Series(0.0, index=oos_slice_index))
            windows.append(dict(
                train_start=df.index[train_start], train_end=df.index[train_end - 1],
                test_start=df.index[test_start], test_end=df.index[test_end - 1],
                params=None, is_stats=None, oos_stats=None, skipped=True,
            ))
            test_start += test_bars
            continue

        chosen_params = combos[best]
        chosen = tpl.with_params(**chosen_params)

        # ---- apply OOS: indicators warmed up before the window, first trade
        # no earlier than its first bar (see window_backtest) ----
        oos_res = window_backtest(df, chosen, test_start, test_end, initial_equity=initial_equity)
        oos_rets = oos_res["returns"]
        oos_ret_parts.append(oos_rets)

        oos_stats = dict(
            total_return=float((1 + oos_rets).prod() - 1),
            sharpe=annualized_sharpe(oos_rets),
            n_trades=len(oos_res["trades"]),
            n_bars=len(oos_rets),
        )
        is_stats = stats_list[best]
        is_ann, oos_ann = _annualized_return(is_stats), _annualized_return(oos_stats)

        windows.append(dict(
            train_start=df.index[train_start], train_end=df.index[train_end - 1],
            test_start=df.index[test_start], test_end=df.index[test_end - 1],
            params=chosen_params, is_stats=is_stats, oos_stats=oos_stats, skipped=False,
            is_score=float(scores[best]),
            params_changed=(prev_params is not None and chosen_params != prev_params),
            wfe=float(np.clip(oos_ann / is_ann, -10, 10)) if is_ann > 0 else np.nan,
        ))
        prev_params = chosen_params
        test_start += test_bars

    if oos_ret_parts:
        oos_returns = pd.concat(oos_ret_parts)
        oos_returns = oos_returns[~oos_returns.index.duplicated(keep="last")].sort_index()
    else:
        oos_returns = pd.Series(dtype=float)

    oos_equity = initial_equity * (1 + oos_returns).cumprod() if len(oos_returns) else pd.Series(dtype=float)
    summary = summarize_walk_forward(windows, oos_returns)
    return {
        "oos_returns": oos_returns,
        "oos_equity": oos_equity,
        "windows": windows,
        "boundaries": boundaries,
        "summary": summary,
        "template": tpl,
    }


def summarize_walk_forward(windows: list, oos_returns: pd.Series) -> dict:
    live = [w for w in windows if not w.get("skipped")]
    n_win = len(windows)
    if not live or len(oos_returns) < 2:
        return dict(n_windows=n_win, n_live_windows=0, oos_sharpe=0.0, oos_cagr=0.0,
                    oos_max_drawdown=0.0, oos_total_return=0.0, is_sharpe_mean=0.0,
                    wfe=np.nan, pct_profitable_windows=0.0, param_change_rate=np.nan,
                    oos_is_sharpe_ratio=np.nan, n_trades_oos=0, pardo_pass=False)

    eq = (1 + oos_returns).cumprod()
    total = float(eq.iloc[-1] - 1)
    n_bars = len(oos_returns)
    cagr = float(eq.iloc[-1] ** (periods_per_year() / n_bars) - 1) if eq.iloc[-1] > 0 else -1.0
    max_dd = float((eq / eq.cummax() - 1).min())
    oos_sharpe = annualized_sharpe(oos_returns)

    is_sharpes = np.array([w["is_stats"]["sharpe"] for w in live])
    is_ann = np.array([_annualized_return(w["is_stats"]) for w in live])
    oos_ann = np.array([_annualized_return(w["oos_stats"]) for w in live])
    is_ann_mean = float(is_ann.mean())
    oos_ann_mean = float(oos_ann.mean())
    # Pardo: annualized OOS return / annualized IS return, both as the mean of
    # the per-window simple-annualized returns. Annualizing the COMPOUNDED
    # total of the whole OOS history against per-window IS figures would grow
    # with the length of the history alone (a flat 10 %/y over 30 years
    # compounds to a "WFE" of 6). Undefined when the optimizer could not even
    # find a profitable IS fit; clipped because a tiny IS denominator makes
    # the ratio meaningless either way.
    wfe = float(np.clip(oos_ann_mean / is_ann_mean, -10, 10)) if is_ann_mean > 0 else np.nan

    profitable = np.array([w["oos_stats"]["total_return"] > 0 for w in live])
    pct_profitable = float(profitable.mean())
    changes = [w["params_changed"] for w in live[1:]]
    param_change_rate = float(np.mean(changes)) if changes else 0.0
    is_mean = float(is_sharpes.mean())

    pardo_pass = bool(
        len(live) >= 3 and total > 0 and pct_profitable >= 0.5 and (np.isfinite(wfe) and wfe >= 0.5)
    )
    return dict(
        n_windows=n_win,
        n_live_windows=len(live),
        oos_sharpe=oos_sharpe,
        oos_cagr=cagr,
        oos_max_drawdown=max_dd,
        oos_total_return=total,
        is_sharpe_mean=is_mean,
        wfe=float(wfe) if np.isfinite(wfe) else np.nan,
        pct_profitable_windows=pct_profitable,
        param_change_rate=param_change_rate,
        oos_is_sharpe_ratio=(oos_sharpe / is_mean) if is_mean > 0 else np.nan,
        n_trades_oos=int(sum(w["oos_stats"]["n_trades"] for w in live)),
        pardo_pass=pardo_pass,
    )


def walk_forward_matrix(
    df: pd.DataFrame,
    tpl,
    param_grid: dict,
    train_lengths=(250, 375, 500, 750),
    test_lengths=(63, 125, 250),
    **kwargs,
) -> pd.DataFrame:
    """Pardo's walk-forward matrix: re-run the WFA over a grid of
    train/test lengths. Returns a DataFrame indexed by (train, test)
    with OOS Sharpe, WFE, % profitable windows and the Pardo pass flag.
    A robust template shows positive OOS Sharpe in MOST cells."""
    rows = []
    for tr, te in iproduct(train_lengths, test_lengths):
        if tr + te >= len(df):
            continue
        s = walk_forward(df, tpl, param_grid, train_bars=tr, test_bars=te, **kwargs)["summary"]
        rows.append(dict(train_bars=tr, test_bars=te, oos_sharpe=s["oos_sharpe"], wfe=s["wfe"],
                         pct_profitable=s["pct_profitable_windows"], n_windows=s["n_windows"],
                         pardo_pass=s["pardo_pass"]))
    out = pd.DataFrame(rows)
    return out.set_index(["train_bars", "test_bars"]) if len(out) else out
