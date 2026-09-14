"""
portfolio.py
------------
Ranger's "portfolio concept": don't pick the single best strategy,
build a family whose return streams don't all rise and fall together.

Given many templates' out-of-sample (walk-forward) equity curves,
this module:
  1. filters out weak/overfit-looking candidates (min Sharpe, min trades)
  2. computes the correlation matrix of their daily OOS returns
  3. greedily builds a subset that maximizes the equal-weight
     portfolio's Sharpe ratio, only adding candidates that stay below
     a correlation ceiling with everything already selected
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def _returns_frame(wfa_results: dict) -> pd.DataFrame:
    """wfa_results: {name: walk_forward() output dict} -> DataFrame of
    daily returns, one column per template, aligned on the union of dates,
    missing days (before a strategy's first OOS trade window) filled with 0."""
    series = {}
    for name, res in wfa_results.items():
        eq = res["oos_equity"]
        if len(eq) < 3:
            continue
        series[name] = eq.pct_change().fillna(0.0)
    return pd.DataFrame(series).fillna(0.0)


def _sharpe(rets: pd.Series) -> float:
    if rets.std() == 0 or rets.empty:
        return 0.0
    return rets.mean() / rets.std() * np.sqrt(252)


def select_portfolio(
    wfa_results: dict,
    min_sharpe: float = 0.2,
    min_windows: int = 3,
    max_strategies: int = 8,
    corr_ceiling: float = 0.6,
):
    """Returns dict with:
      selected      : list of template names chosen for the portfolio
      corr_matrix   : full correlation matrix (all qualifying candidates)
      portfolio_returns : equal-weight daily returns of the selected set
      portfolio_equity  : cumulative equity of the selected set
      candidate_stats   : per-candidate OOS Sharpe / return / trades used for filtering
    """
    rets = _returns_frame(wfa_results)

    candidate_stats = {}
    qualifying = []
    for name, res in wfa_results.items():
        if name not in rets.columns:
            continue
        s = _sharpe(rets[name])
        n_windows = len(res["windows"])
        candidate_stats[name] = {"sharpe": s, "n_windows": n_windows}
        if s >= min_sharpe and n_windows >= min_windows:
            qualifying.append(name)

    if not qualifying:
        return dict(
            selected=[],
            corr_matrix=pd.DataFrame(),
            portfolio_returns=pd.Series(dtype=float),
            portfolio_equity=pd.Series(dtype=float),
            candidate_stats=candidate_stats,
        )

    corr_matrix = rets[qualifying].corr()

    # greedy build: start with the highest-Sharpe qualifying candidate
    ranked = sorted(qualifying, key=lambda nm: candidate_stats[nm]["sharpe"], reverse=True)
    selected = [ranked[0]]

    for _ in range(max_strategies - 1):
        best_candidate, best_sharpe = None, -np.inf
        for name in ranked:
            if name in selected:
                continue
            # correlation ceiling vs everything already selected
            if any(abs(corr_matrix.loc[name, s]) > corr_ceiling for s in selected):
                continue
            trial = selected + [name]
            combo_rets = rets[trial].mean(axis=1)
            sh = _sharpe(combo_rets)
            if sh > best_sharpe:
                best_sharpe, best_candidate = sh, name
        if best_candidate is None:
            break
        # only keep adding if it actually improves the portfolio Sharpe
        current_sharpe = _sharpe(rets[selected].mean(axis=1))
        if best_sharpe > current_sharpe:
            selected.append(best_candidate)
        else:
            break

    portfolio_returns = rets[selected].mean(axis=1)
    
    portfolio_equity = 100_000.0 * (1 + portfolio_returns).cumprod()
    if not portfolio_returns.empty:
        start_date = rets.index[0] - pd.Timedelta(days=1)
        portfolio_equity = pd.concat([pd.Series([100_000.0], index=[start_date]), portfolio_equity])


    return dict(
        selected=selected,
        corr_matrix=corr_matrix,
        portfolio_returns=portfolio_returns,
        portfolio_equity=portfolio_equity,
        candidate_stats=candidate_stats,
    )
