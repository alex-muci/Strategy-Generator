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
import argparse
from contextlib import contextmanager
from multiprocessing import get_context
import os
import pickle
import tempfile

import numpy as np
import pandas as pd

from data import load_yfinance
from generator import param_grid_for, FAMILIES
from live import drop_forming_bar
from portfolio import select_portfolio, walk_forward_portfolio, MIN_TRADES, MIN_WINDOWS
from robustness import (
    cscv_pbo, deflated_sharpe_ratio, min_backtest_length, bootstrap_sharpe_pvalue,
    reality_check, effective_n_trials, merge_block_stats, evaluate_template,
)
from strategy import (
    annualized_sharpe, max_drawdown, periods_per_year, set_periods_per_year, compound,
    periods_per_year_for_interval, SIDES, HEDGE_SHARE, HEDGE_SHARES, set_hedge_share,
)
from walkforward import matrix_cells, matrix_row, matrix_frame


SYNTHETIC_INTERVAL = "1d"   # data.synthetic_ohlc is always business-daily


# --------------------------------------------------------------------------
# the command line both entry points share
# --------------------------------------------------------------------------

def add_research_args(p: argparse.ArgumentParser, *, start: str) -> argparse.ArgumentParser:
    """The research flags of `main.py` and `etf_dashboard.py research`, defined
    once: the template family, the walk-forward, sizing, the instrument, the
    portfolio step and the stress tests. A flag that means the same thing in
    both must not be able to drift (a default, a choice list, a help text).
    What is left to each entry point is what only it has: where the data
    comes from (`--real`/`--csv`/synthetic vs `--assets`), `--interval` and
    `--jobs` (which the dashboard's signals phase shares too), the output
    and the per-asset `--sides-map` / `--portfolio-vol` of a multi-asset book.
    `start` is the default of `--start` (the two entry points differ)."""
    p.add_argument("--family", default="quick", choices=list(FAMILIES))
    p.add_argument("--max-templates", type=int, default=None)
    p.add_argument("--sides", nargs="+", default=None, choices=SIDES,
                   help="restrict every template to these sides (default: the family's own list, "
                        "'both' for quick and default). On an asset with a drift, e.g. --sides long_only")
    p.add_argument("--start", default=start)
    p.add_argument("--bars-per-day", type=float, default=None,
                   help="intraday bars per trading day, replacing the US-equity session --interval assumes "
                        "(7 '1h' bars): e.g. 23 for '1h' bars of a future on a ~23 h session. Sets the "
                        "annualization (252 x this bars a year); intraday intervals only")
    p.add_argument("--bars", type=int, default=3000, help="synthetic bars (per asset)")
    p.add_argument("--train", type=int, default=500, help="training window (bars)")
    p.add_argument("--test", type=int, default=125, help="test window (bars)")
    p.add_argument("--anchored", action="store_true", help="expanding instead of rolling training window")
    p.add_argument("--selection", default="plateau", choices=["plateau", "best"])
    p.add_argument("--metric", default="sharpe", choices=["sharpe", "return_over_dd", "profit_factor"])
    p.add_argument("--wide-grid", action="store_true")
    p.add_argument("--cost-bps", type=float, default=5.0,
                   help="commission+slippage per side, bps of notional. A cash asset's cost: an instrument with a "
                        "margin is costed per unit only (--cost-per-unit), its bps cost is 0")
    p.add_argument("--risk-pct", type=float, default=0.01, help="equity risked per trade (per slot)")
    p.add_argument("--max-leverage", type=float, default=2.0)
    p.add_argument("--vol-target", type=float, default=0.0,
                   help="annualized volatility each entry is sized to (e.g. 0.15); 0 = risk --risk-pct on the ATR "
                        "stop. Set it near the asset's own vol to put the strategies on the buy & hold scale; the "
                        "same target gives every asset of a book the same risk")
    p.add_argument("--vol-target-n", type=int, default=60,
                   help="bars of close-to-close changes in the realized-vol estimate")
    p.add_argument("--point-value", type=float, default=1.0,
                   help="currency per 1.0 of price per unit: 1 for a share, 1000 for a Brent lot, 50 for ES")
    p.add_argument("--cost-per-unit", type=float, default=0.0,
                   help="commission+slippage per unit per side in currency, on top of --cost-bps")
    p.add_argument("--margin-per-unit", type=float, default=0.0,
                   help="initial margin per unit in currency. Give it for every future or spread: it makes the "
                        "series a future (--max-leverage caps margin / equity, so set it at or below 1; the vol "
                        "target sizes on price-point changes; costs are per unit only, --cost-bps is 0; the "
                        "benchmark is one unit's P&L). Without it the series is a cash asset (cap on notional, "
                        "vol target on %% returns). Required for prices at or below zero")
    p.add_argument("--roll-cost-per-unit", type=float, default=0.0,
                   help="currency per unit charged to a position held through a contract roll, long or short: "
                        "the data needs a Roll column (1 on the bars at whose close the contract rolled; "
                        "extra_utils/ETF_trick_spreads.py writes it). Never fold roll costs into the price "
                        "series: a short would collect them")
    p.add_argument("--whole-units", action="store_true",
                   help="floor every size to whole units (contracts); a size below one opens nothing. "
                        "Recommended for futures: research then trades what the live orders round to")
    p.add_argument("--instrument-map", nargs="+", default=None, metavar="ASSET=PV[,COST[,MARGIN[,ROLL]]]",
                   help="per-asset instrument: point value, cost per unit per side, margin per unit, roll cost per "
                        "unit, e.g. brent_z25z26=1000,15,3000; other assets keep the run-wide flags. A mapped asset "
                        "takes only what is given here (a cost, margin or roll cost left out is 0, e.g. SPY=1 is a "
                        "plain share). ASSET is a ticker, or a CSV's file stem")
    p.add_argument("--min-sharpe", type=float, default=0.3)
    p.add_argument("--max-strategies", type=int, default=8)
    p.add_argument("--corr-ceiling", type=float, default=0.6)
    p.add_argument("--require-pardo", action="store_true", help="only candidates passing Pardo's WFA criteria")
    p.add_argument("--select-method", default="greedy", choices=["greedy", "cluster"])
    p.add_argument("--weighting", default="equal", choices=["equal", "hrp"])
    p.add_argument("--cpcv-groups", type=int, default=8)
    p.add_argument("--cpcv-k", type=int, default=2)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--hedge-share", default=HEDGE_SHARE, choices=HEDGE_SHARES,
                   help="how the hedge learners forget: fixed share (one rung per expected number of "
                        "switches in the memory, alpha = m / memory) or discounting (one rung per lifetime H)")
    return p


INSTRUMENT_FIELDS = ("point_value", "cost_per_unit", "margin_per_unit", "roll_cost_per_unit")


def parse_instrument_map(items, assets) -> dict:
    """{'brent': {'point_value': 1000.0, 'cost_per_unit': 15.0, 'margin_per_unit': 3000.0,
    'roll_cost_per_unit': 0.0}} from ['brent=1000,15,3000']; everything after the
    point value may be left out (and is then 0). `assets` are the run's asset
    names: a mapped name that is not one of them is an error, not a no-op."""
    usage = "expected ASSET=POINT_VALUE[,COST_PER_UNIT[,MARGIN_PER_UNIT[,ROLL_COST_PER_UNIT]]]"
    out = {}
    for item in items or []:
        # the LAST '=': a Yahoo futures ticker has one of its own (CL=F=1000,2.5,6000)
        asset, _, spec = item.rpartition("=")
        if asset not in assets:
            raise SystemExit(f"--instrument-map {item}: {asset!r} is not an asset of this run ({', '.join(assets)})")
        if asset in out:
            raise SystemExit(f"--instrument-map {item}: {asset!r} is mapped twice")
        try:
            vals = [float(v) for v in spec.split(",")]
        except ValueError:
            raise SystemExit(f"--instrument-map {item}: {usage}")
        if (not 1 <= len(vals) <= len(INSTRUMENT_FIELDS) or not all(np.isfinite(vals))
                or vals[0] <= 0 or any(v < 0 for v in vals)):
            raise SystemExit(f"--instrument-map {item}: {usage}")
        # a full instrument: what is left out is a cash share's (no per-unit cost,
        # no margin, no roll), never the run-wide futures settings
        vals += [0.0] * (len(INSTRUMENT_FIELDS) - len(vals))
        out[asset] = dict(zip(INSTRUMENT_FIELDS, vals))
    return out


# --------------------------------------------------------------------------
# data and configuration
# --------------------------------------------------------------------------

def resolve_interval(interval: str, synthetic: bool) -> str:
    """The bar interval the DATA actually has. The synthetic series is daily
    whatever --interval says; annualizing it at an intraday factor would
    inflate every Sharpe by the square root of the bars-per-day."""
    return SYNTHETIC_INTERVAL if synthetic else interval


def load_real(ticker: str, start: str, interval: str = "1d", now=None,
              session_close: tuple[str, str] | None = None, drop_nonpositive: bool = True) -> pd.DataFrame:
    """yfinance history WITHOUT the bar that is still forming. Research on a
    half-finished last bar is research on a price nobody could have traded.
    `session_close` is the listing exchange's (HH:MM, zone); default New York."""
    kw = {} if session_close is None else dict(session_close=session_close)
    return drop_forming_bar(load_yfinance(ticker, start=start, interval=interval, drop_nonpositive=drop_nonpositive),
                            interval, now=now, **kw)


def cscv_partitions_for(T: int) -> int:
    """CSCV blocks: 16 needs >= 100 bars per block to be meaningful."""
    return 16 if T >= 1600 else 8


def eval_config(args, interval: str) -> dict:
    """The evaluation settings of a research run, as a plain (picklable,
    JSON-able) dict read off an argparse namespace. `cost_bps` is the
    run-wide one as given: `instrument_of` zeroes it for an instrument with a
    margin."""
    bars_per_day = getattr(args, "bars_per_day", None)
    return dict(
        interval=interval, bars_per_day=bars_per_day,
        periods_per_year=periods_per_year_for_interval(interval, bars_per_day),
        train_bars=args.train, test_bars=args.test, anchored=args.anchored,
        metric=args.metric, selection=args.selection, wide_grid=args.wide_grid,
        cost_bps=args.cost_bps, risk_pct=args.risk_pct, max_leverage=args.max_leverage,
        vol_target=args.vol_target, vol_target_n=args.vol_target_n,
        point_value=args.point_value, cost_per_unit=args.cost_per_unit, margin_per_unit=args.margin_per_unit,
        roll_cost_per_unit=float(getattr(args, "roll_cost_per_unit", 0.0) or 0.0),
        whole_units=bool(getattr(args, "whole_units", False)),
        cpcv_groups=args.cpcv_groups, cpcv_k=args.cpcv_k,
        hedge_share=getattr(args, "hedge_share", HEDGE_SHARE),
    )


def hedge_share_of(c: dict) -> str:
    """The hedge learners' forgetting of a research config: a config written
    before the option existed ran discounted."""
    return c.get("hedge_share", "discount")


def instrument_of(c: dict, asset: str | None = None) -> dict:
    """The instrument settings of a research config, with the defaults of a
    cash share for configs written before they existed; `asset` picks up the
    per-asset overrides of `instrument_map` (a book that mixes shares and a
    future, or a single-asset run given its instrument by name).

    An instrument with a margin is costed per unit only: `cost_bps` is 0 for
    it, whether the margin is run-wide or mapped. Its quoted level is a
    back-adjusted or spread price, not a notional, so a basis-point cost of it
    is arbitrary (and through zero undefined); main.py and the dashboard
    apply this one rule."""
    out = dict(point_value=float(c.get("point_value", 1.0) or 1.0),
               cost_per_unit=float(c.get("cost_per_unit", 0.0) or 0.0),
               margin_per_unit=float(c.get("margin_per_unit", 0.0) or 0.0),
               roll_cost_per_unit=float(c.get("roll_cost_per_unit", 0.0) or 0.0),
               whole_units=bool(c.get("whole_units", False)))
    over = (c.get("instrument_map") or {}).get(asset) if asset is not None else None
    if over:
        # a mapped asset is its own instrument: what the map leaves out is a
        # cash share's, not the run-wide futures settings (SPY=1 in a Brent
        # book must not inherit the Brent margin and per-lot cost)
        out.update(point_value=1.0, cost_per_unit=0.0, margin_per_unit=0.0, roll_cost_per_unit=0.0)
        out.update({k: (bool(v) if k == "whole_units" else float(v)) for k, v in over.items() if k in out})
    if out["margin_per_unit"] > 0:
        out["cost_bps"] = 0.0
    return out


def cost_bps_of(c: dict, asset: str | None = None) -> float:
    """The bps cost an asset of the run is actually charged (0 with a margin)."""
    return float(instrument_of(c, asset).get("cost_bps", c.get("cost_bps", 0.0)))


def costs_text(c: dict, assets=None) -> str:
    """The bps costs a run actually charges, for a report: one number when
    every asset pays the same, else per asset (a margined one pays 0)."""
    assets = list(assets) if assets else [None]
    bps = {a: cost_bps_of(c, a) for a in assets}

    def one(v):
        return f"{v:g} bps/side" if v > 0 else "per unit only"
    if len(set(bps.values())) == 1:
        return one(next(iter(bps.values())))
    return ", ".join(f"{a} {one(v)}" for a, v in bps.items())


def instrument_text(c: dict, asset: str | None = None) -> str:
    """One phrase describing a non-default instrument, empty for a cash share."""
    ins = {k: v for k, v in instrument_of(c, asset).items() if k != "cost_bps"}
    if ins == dict(point_value=1.0, cost_per_unit=0.0, margin_per_unit=0.0, roll_cost_per_unit=0.0,
                   whole_units=False):
        return ""
    return (f"point value {ins['point_value']:g} per unit, {ins['cost_per_unit']:g} per unit per side, "
            f"margin {ins['margin_per_unit']:g} per unit"
            + (f", {ins['roll_cost_per_unit']:g} per unit per roll" if ins["roll_cost_per_unit"] else "")
            + (", whole units" if ins["whole_units"] else ""))


def sizing_text(c: dict) -> str:
    """One phrase describing the sizing rule of a research config (old specs
    have no vol_target keys: they were run with the ATR-stop rule)."""
    vt = float(c.get("vol_target", 0.0) or 0.0)
    if vt > 0:
        text = f"{vt:.0%} annualized vol target per entry ({c.get('vol_target_n', 60)}-bar realized vol, in price points)"
    else:
        text = f"{c['risk_pct']:.1%} of equity risked per trade"
    ins = instrument_text(c)
    return f"{text}; instrument: {ins}" if ins else text


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
    set_hedge_share(hedge_share_of(cfg))


def _init_worker_from_file(path: str, cfg: dict) -> None:
    """Pool initializer: the frames come from a pickle on disk, not from
    initargs (see worker_pool)."""
    with open(path, "rb") as f:
        init_worker(pickle.load(f), cfg)


def _costed(tpl, c: dict, asset: str | None = None):
    return tpl.with_params(**(dict(cost_bps=c["cost_bps"], risk_pct=c["risk_pct"], max_leverage=c["max_leverage"],
                                   vol_target=c["vol_target"], vol_target_n=c["vol_target_n"])
                              | instrument_of(c, asset)))


def _wfa_kwargs(c: dict) -> dict:
    return dict(anchored=c["anchored"], metric=c["metric"], selection=c["selection"])


def evaluate_slot(job):
    """Walk-forward + CPCV for one (name, asset, template) slot. Only the CSCV
    block statistics travel back to the parent, never the T x N trials matrix."""
    name, asset, tpl = job
    c = _CFG
    df = _DATA[asset]
    tpl = _costed(tpl, c, asset)
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
def worker_pool(n_jobs: int, data: dict, cfg: dict, *, context=None):
    """Yield a Pool (or None for n_jobs <= 1) whose workers, and this process,
    are initialised with `data` and `cfg`. `context` is a multiprocessing
    context (start method); default: the platform's.

    The frames reach the workers through a pickle on disk, and initargs carry
    only its path. Under the spawn start method (Windows, macOS) initargs are
    written into a pipe by Process.start(), and the child only drains it after
    re-importing __main__ -- numba, pandas, sklearn: seconds -- so with the
    frames in initargs each start() blocked until its child was up, and Pool()
    took n_jobs times the import time. A path fits in the pipe's buffer, so
    start() returns at once and the children import in parallel; every worker
    still gets the full frames, now from the file.

    On an error -- a worker exception, Ctrl-C, a SystemExit further down the
    pipeline -- the pool is TERMINATED: close() + join() would first wait for
    every task still queued, i.e. minutes of silence before the traceback."""
    if n_jobs <= 1:
        init_worker(data, cfg)
        yield None
        return
    ctx = get_context() if context is None else context
    # the file outlives the pool: a worker that dies is replaced by one that
    # runs the initializer again. ignore_cleanup_errors: a directory Windows
    # will not delete yet is not worth failing a finished run for.
    with tempfile.TemporaryDirectory(prefix="strategy-generator-", ignore_cleanup_errors=True) as tmp:
        path = os.path.join(tmp, "data.pkl")
        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        pool = ctx.Pool(n_jobs, initializer=_init_worker_from_file, initargs=(path, cfg))
        init_worker(data, cfg)
        try:
            yield pool
        except BaseException:
            pool.terminate()
            pool.join()
            raise
        else:
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
    """(static selection on the full OOS history, nested walk-forward selection).
    Both apply the same candidate filter; the nested one recomputes its trade,
    window and Pardo evidence from the windows that had ended at each boundary."""
    port = select_portfolio(
        results, min_sharpe=args.min_sharpe, min_trades=MIN_TRADES, min_windows=MIN_WINDOWS,
        max_strategies=args.max_strategies,
        corr_ceiling=args.corr_ceiling, require_pardo=args.require_pardo,
        method=args.select_method, weighting=args.weighting,
    )
    boundaries = next(iter(results.values()))["boundaries"]
    nested = walk_forward_portfolio(
        rets, boundaries, min_sharpe=args.min_sharpe, max_strategies=args.max_strategies,
        corr_ceiling=args.corr_ceiling, method=args.select_method, weighting=args.weighting,
        windows={n: r.get("windows", []) for n, r in results.items()},
        min_trades=MIN_TRADES, min_windows=MIN_WINDOWS, require_pardo=args.require_pardo,
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

def curve_stats(r: pd.Series, additive: bool = False) -> dict:
    """Sharpe, CAGR and max drawdown of a return stream. `additive` returns
    (the one-unit benchmark's, P&L over a fixed initial equity) add up to
    their curve; compounding them would describe neither the P&L nor an
    investment. For them `cagr` is the simple annual P&L over the initial
    equity and `max_dd` the deepest fall from a running peak, also over the
    initial equity (how a futures account measures both): the summed curve
    can cross zero, where a ratio to the peak or a compounded rate means
    nothing."""
    eq = benchmark_curve(r, additive)
    n = len(r)
    if n < 2:
        return dict(sharpe=0.0, cagr=0.0, max_dd=0.0, n_bars=n)
    if additive:
        v = np.concatenate([[1.0], eq.to_numpy(dtype=float)])
        cagr = float((v[-1] - 1.0) * periods_per_year() / n)
        max_dd = float((v - np.maximum.accumulate(v)).min())
    else:
        cagr = float(eq.iloc[-1] ** (periods_per_year() / n) - 1) if eq.iloc[-1] > 0 else -1.0
        max_dd = max_drawdown(eq, start=1.0)      # funded at 1 before the first bar
    return dict(sharpe=annualized_sharpe(r), cagr=cagr, max_dd=max_dd, n_bars=n)


def against_benchmark(r: pd.Series, bh: pd.Series) -> dict:
    """Beta, correlation and information ratio of a return stream against
    buy-and-hold over the stream's own dates. The IR (annualized Sharpe of the
    residual after removing beta x benchmark) and the correlation are
    scale-free, so they compare a strategy sized on its own rule (1 % per
    trade, or a vol target) with an unlevered holding. The beta is in units
    of the benchmark: against the one-unit benchmark of a future it scales
    with 1 / point value (`benchmark_returns`)."""
    x = bh.reindex(r.index).fillna(0.0).to_numpy()
    y = r.to_numpy()
    if len(y) < 3 or x.std(ddof=1) == 0 or y.std(ddof=1) == 0:
        return dict(beta=np.nan, corr=np.nan, info_ratio=np.nan)
    beta = float(np.cov(x, y, ddof=1)[0, 1] / x.var(ddof=1))
    return dict(beta=beta, corr=float(np.corrcoef(x, y)[0, 1]), info_ratio=annualized_sharpe(y - beta * x))


BENCH_BUY_HOLD = "buy and hold"
BENCH_ONE_UNIT = "hold 1 unit"


def benchmark_returns(df: pd.DataFrame, point_value: float = 1.0, initial_equity: float = 100_000.0,
                      margin_per_unit: float = 0.0):
    """(kind, per-bar returns) of the benchmark nobody optimized. Holding the
    asset is a return series only while its price is positive and is the
    price of what is held; an instrument that trades at or below zero (a
    spread), or a margined future (whose quoted level, back-adjusted or
    rolled, is not the price of an investment: its percentage change is not
    the contract's return), is benchmarked by the P&L of holding one unit on
    the initial equity: additive, on an arbitrary scale. Its Sharpe, the
    correlation to it and the information ratio are scale-free; the beta is
    NOT (it scales with initial_equity / point_value: it reads as the number
    of units held on average), nor are CAGR and drawdown."""
    close = df["Close"]
    if (df["Low"] > 0).all() and not margin_per_unit > 0:
        return BENCH_BUY_HOLD, close.pct_change()
    return BENCH_ONE_UNIT, float(point_value) * close.diff() / float(initial_equity)


def benchmark_curve(r: pd.Series, additive: bool = False) -> pd.Series:
    """Growth of 1 from per-bar returns: compounded, or summed for an additive stream."""
    if additive:
        return 1 + r.cumsum()
    # compounded, and closed at a deficit (strategy.compound): a portfolio's
    # returns can go below -100 % once a futures slot owes more than it had
    return pd.Series(compound(r.to_numpy()), index=r.index)


def benchmark_stats(bh_returns: pd.Series, rets: pd.DataFrame, port: dict, nested: dict,
                    kind: str = BENCH_BUY_HOLD) -> dict:
    """Buy-and-hold over the same out-of-sample bars (or, for an instrument
    that trades through zero, holding one unit: `benchmark_returns`), and the
    two portfolios measured against it. `kind` names which.

    Every template here was walked forward, selected and stress-tested; the
    asset itself was not, so it is the one curve with no selection bias at
    all. Sharpe is the number to compare (the templates are sized on their
    own rule, 1 % of equity per trade unless a vol target is set, so their
    CAGR can be on a different scale); beta says how much of a
    portfolio is just the asset's own drift, and the information ratio how
    much is left once that is removed."""
    bh = bh_returns.reindex(rets.index).fillna(0.0)
    additive = kind == BENCH_ONE_UNIT
    tpl_sharpes = rets.apply(annualized_sharpe)
    out = dict(
        kind=kind,
        additive=additive,
        returns=bh,
        buy_hold=curve_stats(bh, additive),
        n_templates=int(rets.shape[1]),
        # a template that never traded has a Sharpe of 0: sitting flat "beats" a losing
        # benchmark without being evidence of anything, so only templates that traded count
        n_templates_beat_bh=int(((tpl_sharpes > annualized_sharpe(bh)) & rets.ne(0.0).any()).sum()),
        static=None, nested=None, buy_hold_nested_period=None,
    )
    pr = port["portfolio_returns"]
    if len(pr) > 2:
        out["static"] = curve_stats(pr) | against_benchmark(pr, bh)
    nr = nested["portfolio_returns"]
    if len(nr) > 2:
        out["nested"] = curve_stats(nr) | against_benchmark(nr, bh)
        out["buy_hold_nested_period"] = curve_stats(bh.reindex(nr.index).fillna(0.0), additive)
    return out
