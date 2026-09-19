"""
pipeline.py
-----------
The research orchestration shared by the two entry points:

  main.py           one asset, a report and plots
  etf_dashboard.py  several assets, a JSON spec and a daily dashboard

Both do the same thing to get there -- evaluate every (asset, template) slot
in a process pool, run the family-level overfitting diagnostics, select a
static and a nested walk-forward portfolio, stress the finalists -- and they
used to carry a copy each. A copy drifts: one grows a flag, a guard or a
second DSR the other never hears about, and "the research" quietly becomes
two different procedures. Everything that decides a NUMBER lives here, once;
what is left in the entry points is argument parsing, printing and output.
"""

from __future__ import annotations
from contextlib import contextmanager
from multiprocessing import Pool

import numpy as np
import pandas as pd

from data import load_yfinance
from generator import param_grid_for
from live import drop_forming_bar
from portfolio import select_portfolio, walk_forward_portfolio
from robustness import (
    cscv_pbo, deflated_sharpe_ratio, min_backtest_length, bootstrap_sharpe_pvalue,
    reality_check, effective_n_trials, merge_block_stats, evaluate_template,
)
from strategy import (
    annualized_sharpe, max_drawdown, periods_per_year, set_periods_per_year,
    periods_per_year_for_interval,
)
from walkforward import matrix_cells, matrix_row, matrix_frame


SYNTHETIC_INTERVAL = "1d"   # data.synthetic_ohlc is always business-daily


# --------------------------------------------------------------------------
# data and configuration
# --------------------------------------------------------------------------

def resolve_interval(interval: str, synthetic: bool) -> str:
    """The bar interval the DATA actually has. The synthetic series is daily
    whatever --interval says; annualizing it at an intraday factor would
    inflate every Sharpe by the square root of the bars-per-day."""
    return SYNTHETIC_INTERVAL if synthetic else interval


def load_real(ticker: str, start: str, interval: str = "1d", now=None) -> pd.DataFrame:
    """yfinance history WITHOUT the bar that is still forming. Research on a
    half-finished last bar is research on a price nobody could have traded."""
    return drop_forming_bar(load_yfinance(ticker, start=start, interval=interval), interval, now=now)


def cscv_partitions_for(T: int) -> int:
    """CSCV blocks: 16 needs >= 100 bars per block to be meaningful."""
    return 16 if T >= 1600 else 8


def eval_config(args, interval: str) -> dict:
    """The evaluation settings of a research run, as a plain (picklable,
    JSON-able) dict read off an argparse namespace."""
    return dict(
        interval=interval, periods_per_year=periods_per_year_for_interval(interval),
        train_bars=args.train, test_bars=args.test, anchored=args.anchored,
        metric=args.metric, selection=args.selection, wide_grid=args.wide_grid,
        cost_bps=args.cost_bps, risk_pct=args.risk_pct, max_leverage=args.max_leverage,
        cpcv_groups=args.cpcv_groups, cpcv_k=args.cpcv_k,
    )


# --------------------------------------------------------------------------
# workers (run in a process pool)
# --------------------------------------------------------------------------
_DATA = None
_CFG = None


def init_worker(data: dict, cfg: dict) -> None:
    global _DATA, _CFG
    _DATA, _CFG = data, cfg
    # a fresh process re-imports strategy at the daily default; without this every
    # annualized number computed in the pool would be wrong for intraday bars
    set_periods_per_year(cfg["periods_per_year"])


def _costed(tpl, c: dict):
    return tpl.with_params(cost_bps=c["cost_bps"], risk_pct=c["risk_pct"], max_leverage=c["max_leverage"])


def _wfa_kwargs(c: dict) -> dict:
    return dict(anchored=c["anchored"], metric=c["metric"], selection=c["selection"])


def evaluate_slot(job):
    """Walk-forward + CPCV for one (name, asset, template) slot. Only the CSCV
    block statistics travel back to the parent, never the T x N trials matrix."""
    name, asset, tpl = job
    c = _CFG
    df = _DATA[asset]
    tpl = _costed(tpl, c)
    wfa = evaluate_template(
        df, tpl, param_grid_for(tpl, wide=c["wide_grid"]),
        train_bars=c["train_bars"], test_bars=c["test_bars"],
        cpcv_groups=c["cpcv_groups"], cpcv_k=c["cpcv_k"],
        cscv_partitions_n=cscv_partitions_for(len(df)), **_wfa_kwargs(c),
    )
    wfa["asset"] = asset
    return name, wfa


def matrix_cell(job):
    """One cell of Pardo's walk-forward matrix. `tpl` is a template that has
    already been through `evaluate_slot` (it carries the run's costs)."""
    name, asset, tpl, tr, te = job
    c = _CFG
    row = matrix_row(_DATA[asset], tpl, param_grid_for(tpl, wide=c["wide_grid"]), tr, te, **_wfa_kwargs(c))
    return name, row


@contextmanager
def worker_pool(n_jobs: int, data: dict, cfg: dict):
    """Yield a Pool (or None for n_jobs <= 1) whose workers, and this process,
    are initialised with `data` and `cfg`.

    On an error -- a worker exception, Ctrl-C, a SystemExit further down the
    pipeline -- the pool is TERMINATED: close() + join() would first wait for
    every task still queued, i.e. minutes of silence before the traceback."""
    pool = Pool(n_jobs, initializer=init_worker, initargs=(data, cfg)) if n_jobs > 1 else None
    init_worker(data, cfg)
    try:
        yield pool
    except BaseException:
        if pool is not None:
            pool.terminate()
            pool.join()
        raise
    else:
        if pool is not None:
            pool.close()
            pool.join()


def pool_map(pool, fn, jobs):
    """Unordered results of fn over jobs, in the pool if there is one."""
    return pool.imap_unordered(fn, jobs) if pool is not None else map(fn, jobs)


def evaluate_slots(jobs: list, pool, on_result=None) -> dict:
    """{name: evaluation} for every job, in the order of `jobs` whatever order
    the pool finished them in (so the run is reproducible across --jobs)."""
    done = {}
    for i, (name, res) in enumerate(pool_map(pool, evaluate_slot, jobs), 1):
        done[name] = res
        if on_result is not None:
            on_result(i, len(jobs), name, res)
    return {name: done[name] for name, _, _ in jobs}


def walk_forward_matrices(results: dict, names: list, pool) -> dict:
    """{name: Pardo walk-forward matrix} for the slots in `names`, all cells
    in the pool. A slot with no feasible cell (short history) is left out."""
    jobs = []
    for n in names:
        res = results[n]
        n_bars = len(_DATA[res["asset"]])
        jobs += [(n, res["asset"], res["template"], tr, te) for tr, te in matrix_cells(n_bars)]
    rows = {}
    for n, row in pool_map(pool, matrix_cell, jobs):
        rows.setdefault(n, []).append(row)
    return {n: matrix_frame(rows[n]) for n in names if n in rows}


# --------------------------------------------------------------------------
# family-level diagnostics, portfolios, finalists
# --------------------------------------------------------------------------

def family_diagnostics(results: dict, rets: pd.DataFrame, n_boot: int = 1000) -> dict:
    """Overfitting diagnostics for the WHOLE family of trials."""
    # PBO over every (slot, param combo) trial the generator tried
    n_trials = sum(r["n_trials"] for r in results.values())
    pbo = cscv_pbo(blocks=merge_block_stats([r["trial_blocks"] for r in results.values()]))
    # PBO over the slot-level OOS curves (the selection step's trials)
    pbo_tpl = cscv_pbo(rets.to_numpy(), n_partitions=cscv_partitions_for(len(rets))) if rets.shape[1] > 1 else None
    rc = reality_check(rets, n_boot=n_boot)
    oos_sharpes = rets.apply(annualized_sharpe)
    best = oos_sharpes.idxmax()
    # raw DSR: every slot is an independent trial (very conservative)
    var_sr_raw = float(oos_sharpes.var()) / periods_per_year() if rets.shape[1] > 1 else 0.0
    dsr_raw = deflated_sharpe_ratio(rets[best], n_trials=rets.shape[1], var_sr_trials=var_sr_raw)
    # effective DSR: correlated slots collapsed into clusters
    eff = effective_n_trials(rets)
    dsr_best = deflated_sharpe_ratio(rets[best], n_trials=eff["n_eff"], var_sr_trials=eff["var_sr_period"])
    return dict(
        n_templates=rets.shape[1], n_trials=n_trials, pbo_trials=pbo, pbo_templates=pbo_tpl,
        reality_check=rc, oos_sharpes=oos_sharpes, n_eff=eff["n_eff"], var_sr_period=eff["var_sr_period"],
        best_template=best, dsr_best=dsr_best, dsr_raw=dsr_raw,
        min_btl_years=min_backtest_length(eff["n_eff"], max(float(oos_sharpes.max()), 1e-6)),
        years_available=len(rets) / periods_per_year(),
    )


def build_portfolios(results: dict, rets: pd.DataFrame, args) -> tuple[dict, dict]:
    """(static selection on the full OOS history, nested walk-forward selection)."""
    port = select_portfolio(
        results, min_sharpe=args.min_sharpe, max_strategies=args.max_strategies,
        corr_ceiling=args.corr_ceiling, require_pardo=args.require_pardo,
        method=args.select_method, weighting=args.weighting,
    )
    boundaries = next(iter(results.values()))["boundaries"]
    nested = walk_forward_portfolio(
        rets, boundaries, min_sharpe=args.min_sharpe, max_strategies=args.max_strategies,
        corr_ceiling=args.corr_ceiling, method=args.select_method, weighting=args.weighting,
    )
    return port, nested


def finalist_stats(results: dict, selected: list, fam: dict, n_boot: int = 1000) -> dict:
    """Bootstrap p-value and Deflated Sharpe of each selected slot, deflated
    by the family's effective number of trials."""
    out = {}
    for name in selected:
        r = results[name]["oos_returns"]
        boot = bootstrap_sharpe_pvalue(r, n_boot=n_boot)
        dsr = deflated_sharpe_ratio(r, n_trials=fam["n_eff"], var_sr_trials=fam["var_sr_period"])
        out[name] = dict(bootstrap_p=boot["p_value"], dsr=dsr["dsr"], psr0=dsr["psr0"])
    return out


# --------------------------------------------------------------------------
# curves against a benchmark
# --------------------------------------------------------------------------

def curve_stats(r: pd.Series) -> dict:
    eq = (1 + r).cumprod()
    n = len(r)
    if n < 2:
        return dict(sharpe=0.0, cagr=0.0, max_dd=0.0, n_bars=n)
    return dict(
        sharpe=annualized_sharpe(r),
        cagr=float(eq.iloc[-1] ** (periods_per_year() / n) - 1) if eq.iloc[-1] > 0 else -1.0,
        max_dd=max_drawdown(eq),
        n_bars=n,
    )


def against_benchmark(r: pd.Series, bh: pd.Series) -> dict:
    """Beta, correlation and information ratio of a return stream against
    buy-and-hold over the stream's own dates. The IR (annualized Sharpe of the
    residual after removing beta x benchmark) is scale-free, so it compares
    a strategy risking 1 % per trade with an unlevered holding."""
    x = bh.reindex(r.index).fillna(0.0).to_numpy()
    y = r.to_numpy()
    if len(y) < 3 or x.std(ddof=1) == 0 or y.std(ddof=1) == 0:
        return dict(beta=np.nan, corr=np.nan, info_ratio=np.nan)
    beta = float(np.cov(x, y, ddof=1)[0, 1] / x.var(ddof=1))
    return dict(beta=beta, corr=float(np.corrcoef(x, y)[0, 1]), info_ratio=annualized_sharpe(y - beta * x))


def benchmark_stats(bh_returns: pd.Series, rets: pd.DataFrame, port: dict, nested: dict) -> dict:
    """Buy-and-hold over the same out-of-sample bars, and the two portfolios
    measured against it.

    Every template here was walked forward, selected and stress-tested; the
    asset itself was not, so it is the one curve with no selection bias at
    all. Sharpe is the number to compare (the templates risk 1 % of equity
    per trade, so their CAGR is on a different scale); beta says how much of a
    portfolio is just the asset's own drift, and the information ratio how
    much is left once that is removed."""
    bh = bh_returns.reindex(rets.index).fillna(0.0)
    tpl_sharpes = rets.apply(annualized_sharpe)
    out = dict(
        returns=bh,
        buy_hold=curve_stats(bh),
        n_templates=int(rets.shape[1]),
        n_templates_beat_bh=int((tpl_sharpes > annualized_sharpe(bh)).sum()),
        static=None, nested=None, buy_hold_nested_period=None,
    )
    pr = port["portfolio_returns"]
    if len(pr) > 2:
        out["static"] = curve_stats(pr) | against_benchmark(pr, bh)
    nr = nested["portfolio_returns"]
    if len(nr) > 2:
        out["nested"] = curve_stats(nr) | against_benchmark(nr, bh)
        out["buy_hold_nested_period"] = curve_stats(bh.reindex(nr.index).fillna(0.0))
    return out
