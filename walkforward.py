"""
walkforward.py
---------------
Pardo's core contribution: Walk-Forward Analysis (WFA).

For a given StrategyTemplate (fixed switches), slide a training window
across history, re-optimize the template's numeric params on that
window, apply the chosen params to the following unseen test window,
then roll forward. Stitching all the out-of-sample (OOS) test segments
together gives one continuous equity curve that was NEVER built with
hindsight over that period -- a much more honest estimate of live
performance than an in-sample backtest.
"""

from __future__ import annotations
from itertools import product as iproduct
import numpy as np
import pandas as pd

from strategy import backtest


def _grid_combos(grid: dict):
    keys = list(grid.keys())
    for values in iproduct(*grid.values()):
        yield dict(zip(keys, values))


def _score(stats: dict, metric: str) -> float:
    if stats["n_trades"] < 5:
        return -np.inf  # not enough trades to trust this window
    if metric == "sharpe":
        return stats["sharpe"]
    if metric == "return_over_dd":
        dd = abs(stats["max_drawdown"]) or 1e-6
        return stats["total_return"] / dd
    raise ValueError(f"unknown metric {metric}")


def walk_forward(
    df: pd.DataFrame,
    tpl,
    param_grid: dict,
    train_bars: int = 500,
    test_bars: int = 125,
    warmup_buffer: int = 260,
    metric: str = "sharpe",
    initial_equity: float = 100_000.0,
):
    """Run a rolling walk-forward optimization of `tpl` over `df`.

    Returns dict with:
      oos_equity   : pd.Series, stitched out-of-sample equity curve
      windows      : list of per-window dicts (dates, chosen params, is/oos stats)
    """
    n = len(df)
    windows = []
    oos_returns = []
    oos_index = []

    start = 0
    while start + train_bars + test_bars <= n:
        train_slice = df.iloc[start : start + train_bars]
        test_start = start + train_bars
        test_end = min(test_start + test_bars, n)

        # optimize on the training window
        best_score, best_params, best_is_stats = -np.inf, None, None
        for params in _grid_combos(param_grid):
            candidate = tpl.with_params(**params)
            result = backtest(train_slice, candidate, initial_equity=initial_equity)
            score = _score(result["stats"], metric)
            if score > best_score:
                best_score, best_params, best_is_stats = score, params, result["stats"]

        if best_params is None:
            # nothing traded enough in-sample; skip window
            start += test_bars
            continue

        # apply chosen params out-of-sample, with a warmup buffer for indicators
        buf_start = max(0, test_start - warmup_buffer)
        test_slice = df.iloc[buf_start:test_end]
        chosen = tpl.with_params(**best_params)
        oos_result = backtest(test_slice, chosen, initial_equity=initial_equity)

        oos_eq = oos_result["equity"]
        # keep only the true OOS portion (drop the warmup buffer)
        oos_eq = oos_eq.loc[df.index[test_start] :]
        if len(oos_eq) > 1:
            rets = oos_eq.pct_change().dropna()
            oos_returns.append(rets)
            oos_index.append(rets.index)

        windows.append(
            dict(
                train_start=df.index[start],
                train_end=df.index[start + train_bars - 1],
                test_start=df.index[test_start],
                test_end=df.index[test_end - 1],
                params=best_params,
                is_stats=best_is_stats,
                oos_stats=oos_result["stats"],
            )
        )

        start += test_bars

    if oos_returns:
        all_rets = pd.concat(oos_returns)
        oos_equity = initial_equity * (1 + all_rets).cumprod()
        
        # We need the equity to start at initial_equity on the day *before* the first return.
        first_ret_idx = df.index.get_loc(all_rets.index[0])
        start_date = df.index[first_ret_idx - 1] if first_ret_idx > 0 else df.index[0]
        
        oos_equity = pd.concat([pd.Series([initial_equity], index=[start_date]), oos_equity])
        oos_equity = oos_equity[~oos_equity.index.duplicated(keep="last")].sort_index()
    else:
        oos_equity = pd.Series([initial_equity], index=[df.index[0]])

    return {"oos_equity": oos_equity, "windows": windows, "template": tpl}
