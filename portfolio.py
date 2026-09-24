"""
portfolio.py
------------
Ranger's "portfolio concept": don't pick the single best strategy,
build a family whose return streams don't all rise and fall together.

Given many templates' out-of-sample (walk-forward) return streams,
this module:
  1. filters out weak/overfit-looking candidates (min OOS Sharpe, min
     windows, optionally Pardo's WFA criteria)
  2. computes the correlation matrix of their daily OOS returns
  3. picks a subset either greedily (best Sharpe first, add only
     candidates below a correlation ceiling while the portfolio Sharpe
     improves) or by hierarchical clustering (best member of each
     correlation cluster)
  4. weights it equally or with Hierarchical Risk Parity (AFML ch. 16)

The catch -- and the reason for `walk_forward_portfolio` -- is that
step 1-3 look at the WHOLE out-of-sample history. Selecting the best of
hundreds of "OOS" curves makes the resulting portfolio curve in-sample
again. The nested walk-forward re-runs the selection -- the same
candidate filter, recomputed from the walk-forward windows that had
ended by then -- at every window boundary using only the OOS history
available up to then and applies it to the next window: a "doubly
out-of-sample" curve, which is the honest number to quote.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from strategy import annualized_sharpe
from robustness import hrp_weights
from walkforward import summarize_walk_forward

# the static filter's evidence thresholds, shared with the nested selection
# (pipeline.build_portfolios passes them to both)
MIN_TRADES = 10
MIN_WINDOWS = 3


def returns_frame(wfa_results: dict) -> pd.DataFrame:
    """{name: walk_forward() output} -> DataFrame of per-bar OOS returns,
    one column per template, on the dates every template has covered."""
    series = {name: res["oos_returns"] for name, res in wfa_results.items() if len(res["oos_returns"]) > 2}
    if not series:
        return pd.DataFrame()
    return pd.concat(series, axis=1, join="inner").fillna(0.0)


def candidate_table(wfa_results: dict, rets: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, res in wfa_results.items():
        s = res["summary"]
        rows.append(dict(
            template=name,
            oos_sharpe=annualized_sharpe(rets[name]) if name in rets.columns else 0.0,
            oos_cagr=s["oos_cagr"], oos_max_dd=s["oos_max_drawdown"],
            wfe=s["wfe"], pct_profitable_windows=s["pct_profitable_windows"],
            n_windows=s["n_windows"],
            # windows the optimizer found a tradeable fit in; `n_windows` is the
            # same for every template on the same bars, so it filters nothing
            n_live_windows=s.get("n_live_windows", s["n_windows"]),
            n_trades_oos=s["n_trades_oos"],
            param_change_rate=s["param_change_rate"], pardo_pass=s["pardo_pass"],
            oos_exposure=s.get("oos_exposure", np.nan),
            oos_notional=s.get("oos_notional", np.nan),
            oos_avg_net_exposure=s.get("oos_avg_net_exposure", np.nan),
        ))
    return pd.DataFrame(rows).set_index("template")


def _qualifying(table: pd.DataFrame, min_sharpe, min_windows, require_pardo, min_wfe, min_trades,
                available=None):
    q = ((table["oos_sharpe"] >= min_sharpe) & (table["n_live_windows"] >= min_windows)
         & (table["n_trades_oos"] >= min_trades))
    if require_pardo:
        q &= table["pardo_pass"]
    if min_wfe is not None:
        q &= table["wfe"].fillna(-np.inf) >= min_wfe
    if available is not None:
        # a template whose walk-forward produced no usable OOS series has no
        # column in the aligned returns frame; it is scored 0.0 in `table`, so
        # with min_sharpe <= 0 it would otherwise qualify and then KeyError.
        q &= table.index.isin(available)
    return list(table.index[q])


def select_subset(rets: pd.DataFrame, candidates: list, method: str = "greedy",
                  max_strategies: int = 8, corr_ceiling: float = 0.6) -> list:
    """Pick a diversified subset of `candidates` (columns of `rets`)."""
    if not candidates:
        return []
    sharpes = {c: annualized_sharpe(rets[c]) for c in candidates}
    ranked = sorted(candidates, key=lambda c: sharpes[c], reverse=True)
    if len(candidates) == 1:
        return ranked

    if method == "cluster":
        corr = rets[candidates].corr().fillna(0.0)
        dist = np.sqrt(0.5 * (1 - corr.clip(-1, 1))).to_numpy().copy()
        np.fill_diagonal(dist, 0.0)
        link = linkage(squareform(dist, checks=False), method="average")
        labels = fcluster(link, t=min(max_strategies, len(candidates)), criterion="maxclust")
        chosen = {}
        for name, lab in zip(candidates, labels):
            if lab not in chosen or sharpes[name] > sharpes[chosen[lab]]:
                chosen[lab] = name
        return sorted(chosen.values(), key=lambda c: sharpes[c], reverse=True)

    # greedy: start with the best, add while the equal-weight Sharpe improves
    corr = rets[candidates].corr()
    selected = [ranked[0]]
    for _ in range(max_strategies - 1):
        current = annualized_sharpe(rets[selected].mean(axis=1))
        best_c, best_s = None, current
        for name in ranked:
            if name in selected:
                continue
            # a flat (zero-variance) stream has undefined correlation: treat it as
            # failing the ceiling rather than as conveniently uncorrelated
            if any(not (abs(corr.loc[name, s]) <= corr_ceiling) for s in selected):
                continue
            sh = annualized_sharpe(rets[selected + [name]].mean(axis=1))
            if sh > best_s:
                best_s, best_c = sh, name
        if best_c is None:
            break
        selected.append(best_c)
    return selected


def portfolio_weights(rets: pd.DataFrame, weighting: str = "equal") -> pd.Series:
    if rets.shape[1] == 0:
        return pd.Series(dtype=float)
    if weighting == "hrp":
        w = hrp_weights(rets)
        return w / w.sum()
    return pd.Series(1.0 / rets.shape[1], index=rets.columns)


def select_portfolio(
    wfa_results: dict,
    min_sharpe: float = 0.2,
    min_windows: int = MIN_WINDOWS,
    min_trades: int = MIN_TRADES,
    max_strategies: int = 8,
    corr_ceiling: float = 0.6,
    require_pardo: bool = False,
    min_wfe: float | None = None,
    method: str = "greedy",
    weighting: str = "equal",
    initial_equity: float = 100_000.0,
) -> dict:
    """Static (full-history) portfolio selection. Returns dict with:
      selected, weights, corr_matrix, portfolio_returns, portfolio_equity,
      candidate_stats (DataFrame, one row per template), qualifying (list)
    """
    rets = returns_frame(wfa_results)
    table = candidate_table(wfa_results, rets)
    qualifying = _qualifying(table, min_sharpe, min_windows, require_pardo, min_wfe, min_trades,
                             available=set(rets.columns))
    selected = select_subset(rets, qualifying, method, max_strategies, corr_ceiling)

    if not selected:
        return dict(selected=[], weights=pd.Series(dtype=float), corr_matrix=pd.DataFrame(),
                    portfolio_returns=pd.Series(dtype=float), portfolio_equity=pd.Series(dtype=float),
                    candidate_stats=table, qualifying=qualifying)

    weights = portfolio_weights(rets[selected], weighting)
    port_rets = (rets[selected] * weights).sum(axis=1)
    return dict(
        selected=selected,
        weights=weights,
        corr_matrix=rets[qualifying].corr(),
        portfolio_returns=port_rets,
        portfolio_equity=initial_equity * (1 + port_rets).cumprod(),
        candidate_stats=table,
        qualifying=qualifying,
    )


def trade_count_proxy(rets: pd.DataFrame) -> pd.Series:
    """Trades per column estimated from returns alone: the number of runs of
    non-zero returns. A strategy that is flat earns exactly 0 on the bar, so
    each holding period shows up as one run (a held bar on which the price
    did not move splits a run, and back-to-back trades merge: an estimate)."""
    active = rets.fillna(0.0).ne(0.0)
    return (active & ~active.shift(1, fill_value=False)).sum()


def _causal_filter(cands: list, hist: pd.DataFrame, windows: dict, start, min_trades: int,
                   min_windows: int, require_pardo: bool, min_wfe) -> list:
    """The static filter (trades, live windows, Pardo, WFE) with every number
    recomputed from the walk-forward windows whose OOS period ENDED before
    `start`, and the OOS returns up to then."""
    keep = []
    for c in cands:
        past = [w for w in windows.get(c, []) if w.get("test_end") is not None and w["test_end"] < start]
        s = summarize_walk_forward(past, hist[c])
        if s["n_trades_oos"] < min_trades or s["n_live_windows"] < min_windows:
            continue
        if require_pardo and not s["pardo_pass"]:
            continue
        if min_wfe is not None and not (np.isfinite(s["wfe"]) and s["wfe"] >= min_wfe):
            continue
        keep.append(c)
    return keep


def walk_forward_portfolio(
    rets: pd.DataFrame,
    boundaries: list,
    min_history_windows: int = 4,
    min_sharpe: float = 0.2,
    min_trades_proxy: int = 0,
    max_strategies: int = 8,
    corr_ceiling: float = 0.6,
    method: str = "greedy",
    weighting: str = "equal",
    initial_equity: float = 100_000.0,
    *,
    windows: dict | None = None,
    min_trades: int = 0,
    min_windows: int = 0,
    require_pardo: bool = False,
    min_wfe: float | None = None,
) -> dict:
    """Nested walk-forward of the PORTFOLIO SELECTION itself.

    At each test-window boundary (after `min_history_windows` windows of
    OOS history exist) the candidate filter, subset selection and weights
    are recomputed from the OOS returns realised SO FAR, then held over
    the next window. Nothing about the future enters the choice, so the
    resulting curve is out-of-sample with respect to both the parameter
    optimization and the template selection.

    The candidate filter:
      min_sharpe        OOS Sharpe of the history so far
      min_trades_proxy  trades so far as estimated from the returns alone
                        (`trade_count_proxy`: runs of non-zero returns); for
                        callers that have no walk-forward windows
      windows           {name: walk_forward()["windows"]}; when given, the
                        static filter's evidence is recomputed from the
                        windows whose OOS period ended before the block:
                        `min_trades` OOS trades and `min_windows` live windows
                        SO FAR, and optionally Pardo's criteria / a min WFE
                        (see select_portfolio). Counts accumulate, so early
                        blocks face a stricter bar than the full-history filter.
    """
    boundaries = sorted(set(boundaries))
    parts, log = [], []
    for j in range(min_history_windows, len(boundaries)):
        start = boundaries[j]
        end = boundaries[j + 1] if j + 1 < len(boundaries) else None
        hist = rets.loc[: start - pd.Timedelta(nanoseconds=1)]
        if len(hist) < 60:
            continue
        sharpes = hist.apply(annualized_sharpe)
        ok = sharpes >= min_sharpe
        if min_trades_proxy > 0:
            ok &= trade_count_proxy(hist) >= min_trades_proxy
        cands = list(sharpes.index[ok])
        if windows is not None:
            cands = _causal_filter(cands, hist, windows, start, min_trades, min_windows, require_pardo, min_wfe)
        sel = select_subset(hist, cands, method, max_strategies, corr_ceiling)
        block = rets.loc[start:end] if end is None else rets.loc[start: end - pd.Timedelta(nanoseconds=1)]
        if not sel or block.empty:
            parts.append(pd.Series(0.0, index=block.index))
            log.append(dict(period_start=start, selected=[], weights={}))
            continue
        w = portfolio_weights(hist[sel], weighting)
        parts.append((block[sel] * w).sum(axis=1))
        log.append(dict(period_start=start, selected=sel, weights=w.round(3).to_dict()))

    port = pd.concat(parts) if parts else pd.Series(dtype=float)
    return dict(
        portfolio_returns=port,
        portfolio_equity=initial_equity * (1 + port).cumprod() if len(port) else pd.Series(dtype=float),
        selections=log,
        sharpe=annualized_sharpe(port) if len(port) > 2 else 0.0,
    )
