"""
etf_dashboard.py
----------------
A two-phase dashboard for trading a handful of ETFs by hand.

    # once (slow: minutes to an hour) -- decides WHAT to trade
    python etf_dashboard.py research --assets SPY TLT GLD QQQ --family default \
        --start 2010-01-01 --jobs 8

    # every morning, or every few hours on hourly bars (fast: seconds)
    python etf_dashboard.py signals --account-equity 100000

Why two phases. The research phase is the whole Pardo + Lopez de Prado pipeline
run per (asset, template) pair: walk-forward analysis, CPCV, PBO, Reality Check,
then a correlation-diversified portfolio whose SELECTION is itself walked
forward. It costs minutes and its answer -- which (asset, template) slots to
trade and at what weight -- is only supposed to change when you deliberately
re-examine it. Re-running it every morning would be a fresh data-mining exercise
every morning, which is exactly the failure mode the pipeline exists to measure.

So research writes `portfolio.json` and the signals phase just applies it: pull
fresh bars, re-optimize each slot's numeric params ONLY when a whole test window
has elapsed (the walk-forward's own cadence -- see live.due_for_refit), read each
slot's current position out of the engine, and render the orders. Seconds, so you
can run it on a cron 1-4 times a day.

    crontab, daily bars:   10 17 * * 1-5  cd <repo> && python etf_dashboard.py signals
    crontab, hourly bars:  35 10-16 * * 1-5  ... (after each bar closes)

Outputs land in `--state-dir` (default ./state):

    portfolio.json      what research decided (slots, weights, diagnostics, verdict)
    live_state.json     per-slot chosen params, when they were fitted, last targets
    dashboard.html      the page you actually look at
    signals.csv         the same orders, for a spreadsheet
    signal_log.csv      one row per asset per run, appended (an audit trail)
    holdings.json       YOU maintain this: {"SPY": 120, "TLT": -50} actual broker
                        positions, so the trade list is against reality rather
                        than against yesterday's target
    research_report.md  the full diagnostics write-up

Nothing here places an order. It tells you what to place.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from multiprocessing import Pool

import numpy as np
import pandas as pd

from data import load_yfinance, synthetic_ohlc
from generator import generate_templates, param_grid_for
from live import (
    drop_forming_bar, refit_params, due_for_refit, strategy_state,
    portfolio_targets, trade_list, utcnow,
)
from portfolio import returns_frame, select_portfolio, walk_forward_portfolio
from robustness import (
    cscv_pbo, merge_block_stats, reality_check, effective_n_trials,
    deflated_sharpe_ratio, min_backtest_length, bootstrap_sharpe_pvalue,
    evaluate_template,
)
from strategy import (
    StrategyTemplate, annualized_sharpe, periods_per_year, set_periods_per_year,
    periods_per_year_for_interval, BARS_PER_YEAR, SIDES,
)
from dashboard_html import render_dashboard

SPEC_VERSION = 2


# ==========================================================================
# CLI
# ==========================================================================

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Two-phase ETF strategy dashboard: research once, read signals daily",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Outputs land in")[0].split("Why two phases.")[1].strip(),
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--state-dir", default="state", help="where portfolio.json etc. live")
    common.add_argument("--interval", default="1d", choices=sorted(BARS_PER_YEAR),
                        help="bar interval; also sets the annualization factor")
    common.add_argument("--synthetic", action="store_true",
                        help="use synthetic data instead of yfinance (for trying the "
                             "wiring out without a data feed)")
    common.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))

    r = sub.add_parser("research", parents=[common], help="decide what to trade (slow)")
    r.add_argument("--assets", nargs="+", default=["SPY", "TLT", "GLD", "QQQ"])
    r.add_argument("--start", default="2010-01-01")
    r.add_argument("--family", default="quick", choices=["quick", "default", "full"])
    r.add_argument("--max-templates", type=int, default=None)
    r.add_argument("--sides", nargs="+", default=None, choices=SIDES,
                   help="restrict every template to these sides (default: the family's own list)")
    r.add_argument("--train", type=int, default=500, help="training window (bars)")
    r.add_argument("--test", type=int, default=125, help="test window (bars)")
    r.add_argument("--anchored", action="store_true", help="expanding training window")
    r.add_argument("--metric", default="sharpe", choices=["sharpe", "return_over_dd", "profit_factor"])
    r.add_argument("--selection", default="plateau", choices=["plateau", "best"])
    r.add_argument("--wide-grid", action="store_true")
    r.add_argument("--cost-bps", type=float, default=5.0, help="commission+slippage per side")
    r.add_argument("--risk-pct", type=float, default=0.01, help="equity risked per trade, per slot")
    r.add_argument("--max-leverage", type=float, default=2.0)
    r.add_argument("--min-sharpe", type=float, default=0.3)
    r.add_argument("--max-strategies", type=int, default=8)
    r.add_argument("--corr-ceiling", type=float, default=0.6)
    r.add_argument("--require-pardo", action="store_true")
    r.add_argument("--select-method", default="greedy", choices=["greedy", "cluster"])
    r.add_argument("--weighting", default="hrp", choices=["equal", "hrp"])
    r.add_argument("--cpcv-groups", type=int, default=8)
    r.add_argument("--cpcv-k", type=int, default=2)
    r.add_argument("--n-boot", type=int, default=1000)
    r.add_argument("--bars", type=int, default=3000, help="synthetic bars per asset")

    s = sub.add_parser("signals", parents=[common], help="today's positions and orders (fast)")
    s.add_argument("--account-equity", type=float, default=100_000.0,
                   help="capital the whole book is sized on")
    s.add_argument("--max-gross", type=float, default=1.0,
                   help="cap on gross exposure as a multiple of account equity; the "
                        "whole book is scaled down proportionally if it is exceeded")
    s.add_argument("--lot", type=float, default=1.0, help="round order sizes to this")
    s.add_argument("--no-refit", action="store_true",
                   help="never re-optimize, even when a test window has elapsed")
    s.add_argument("--now", default=None,
                   help="pretend it is this UTC timestamp (for testing the forming-bar rule)")
    s.add_argument("--bars", type=int, default=3000, help="synthetic bars per asset")
    return p.parse_args(argv)


# ==========================================================================
# data
# ==========================================================================

def load_assets(assets, *, interval, start, synthetic, bars, now=None, quiet=False) -> dict:
    """Load every asset and put them on ONE shared bar index.

    The nested walk-forward portfolio needs a single list of window boundaries,
    and those come from positions in the bar index -- so if SPY and GLD disagreed
    about which days exist, their windows would not line up and the selection
    step would be comparing misaligned segments. Intersecting the indices up
    front costs a few holidays and makes every slot directly comparable.
    """
    raw = {}
    for i, a in enumerate(assets):
        if synthetic:
            # a different seed per asset, so the assets are not the same series
            raw[a] = synthetic_ohlc(n_bars=bars, seed=100 + 7 * i, trend_prob=0.45 + 0.05 * i)
        else:
            raw[a] = load_yfinance(a, start=start, interval=interval)
            raw[a] = drop_forming_bar(raw[a], interval, now=now)
        if not quiet:
            print(f"  {a}: {len(raw[a])} bars, {raw[a].index[0].date()} to {raw[a].index[-1].date()}")

    common = None
    for a in assets:
        common = raw[a].index if common is None else common.intersection(raw[a].index)
    if len(common) == 0:
        raise ValueError("the assets share no bars at all -- check the symbols and the interval")
    dropped = {a: len(raw[a]) - len(common) for a in assets}
    if not quiet and any(dropped.values()):
        print(f"  aligned on {len(common)} shared bars (dropped {dropped})")
    return {a: raw[a].loc[common] for a in assets}


def _history_bars_needed(spec: dict) -> int:
    """Bars of history the signals phase needs: one training window, plus the
    longest warm-up any template could ask for, plus slack."""
    return int(spec["config"]["train_bars"] * 1.25) + 400


def _start_for_bars(n_bars: int, interval: str) -> str:
    """A start date comfortably older than `n_bars` bars ago."""
    per_year = BARS_PER_YEAR[interval]
    years = n_bars / per_year * 1.6 + 0.5
    return (utcnow() - pd.Timedelta(days=365.25 * years)).strftime("%Y-%m-%d")


# ==========================================================================
# research phase
# ==========================================================================

_DATA = None
_CFG = None


def _init_worker(data, cfg):
    global _DATA, _CFG
    _DATA, _CFG = data, cfg
    set_periods_per_year(cfg["periods_per_year"])


def _evaluate_slot(job):
    """One (asset, template) slot: walk-forward + CPCV. Runs in the pool."""
    asset, tpl = job
    c = _CFG
    tpl = tpl.with_params(cost_bps=c["cost_bps"], risk_pct=c["risk_pct"],
                          max_leverage=c["max_leverage"])
    df = _DATA[asset]
    grid = param_grid_for(tpl, wide=c["wide_grid"])
    wfa = evaluate_template(
        df, tpl, grid, train_bars=c["train_bars"], test_bars=c["test_bars"],
        anchored=c["anchored"], metric=c["metric"], selection=c["selection"],
        cpcv_groups=c["cpcv_groups"], cpcv_k=c["cpcv_k"],
        cscv_partitions_n=16 if len(df) >= 1600 else 8,
    )
    wfa["asset"] = asset
    return f"{asset}|{tpl.name}", wfa


def research(args) -> dict:
    t0 = time.time()
    os.makedirs(args.state_dir, exist_ok=True)
    ppy = periods_per_year_for_interval(args.interval)
    set_periods_per_year(ppy)

    print(f"Loading {len(args.assets)} assets ({args.interval} bars)"
          f"{' [synthetic]' if args.synthetic else ''}...")
    data = load_assets(args.assets, interval=args.interval, start=args.start,
                       synthetic=args.synthetic, bars=args.bars)
    n_bars = len(next(iter(data.values())))
    if n_bars < args.train + 2 * args.test:
        raise SystemExit(
            f"only {n_bars} shared bars: too few for train={args.train} + test={args.test}. "
            f"Use an earlier --start, a coarser --interval, or smaller windows.")

    overrides = {"sides": args.sides} if args.sides else {}
    templates = generate_templates(args.family, max_templates=args.max_templates, **overrides)
    cfg = dict(
        interval=args.interval, periods_per_year=ppy, start=args.start,
        family=args.family, sides=args.sides, train_bars=args.train, test_bars=args.test,
        anchored=args.anchored, metric=args.metric, selection=args.selection,
        wide_grid=args.wide_grid, cost_bps=args.cost_bps, risk_pct=args.risk_pct,
        max_leverage=args.max_leverage, cpcv_groups=args.cpcv_groups, cpcv_k=args.cpcv_k,
        min_sharpe=args.min_sharpe, corr_ceiling=args.corr_ceiling,
        weighting=args.weighting, select_method=args.select_method,
    )
    jobs = [(a, t) for a in args.assets for t in templates]
    print(f"{len(templates)} templates x {len(args.assets)} assets = {len(jobs)} slots; "
          f"walk-forward train={args.train} test={args.test} "
          f"{'anchored' if args.anchored else 'rolling'}, {args.cost_bps} bps/side")

    pool = Pool(args.jobs, initializer=_init_worker, initargs=(data, cfg)) if args.jobs > 1 else None
    _init_worker(data, cfg)
    try:
        results = {}
        mapper = pool.imap_unordered if pool is not None else map
        for i, (key, res) in enumerate(mapper(_evaluate_slot, jobs), 1):
            results[key] = res
            if i % max(1, len(jobs) // 20) == 0 or i == len(jobs):
                print(f"  [{i}/{len(jobs)}] {key:<52} oos_sharpe="
                      f"{res['summary']['oos_sharpe']:>6.2f} cpcv={res['cpcv']['sharpe_mean']:>6.2f}")
        spec = _assemble_spec(data, results, args, cfg, pool)
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    path = os.path.join(args.state_dir, "portfolio.json")
    with open(path, "w") as f:
        json.dump(spec, f, indent=2, default=str)
    _write_research_report(spec, os.path.join(args.state_dir, "research_report.md"))
    # a re-run invalidates any params fitted for the previous slot list
    _prune_live_state(args.state_dir, {s["slot"] for s in spec["slots"]})

    print(f"\nWrote {path} ({len(spec['slots'])} slots) and research_report.md "
          f"in {time.time() - t0:.0f}s")
    print(f"\nVERDICT [{spec['verdict']['level'].upper()}] {spec['verdict']['headline']}")
    for r in spec["verdict"]["reasons"]:
        print(f"  - {r}")
    return spec


def _assemble_spec(data, results, args, cfg, pool) -> dict:
    rets = returns_frame(results)
    if rets.empty:
        raise SystemExit("no slot produced a usable out-of-sample series -- history too short")

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

    print("\nFamily-level diagnostics over every (asset, template) trial...")
    diag = _diagnostics(results, rets, nested, port, args)
    selected = port["selected"]
    weights = port["weights"] if len(selected) else pd.Series(dtype=float)

    finalists = {}
    for name in selected:
        boot = bootstrap_sharpe_pvalue(results[name]["oos_returns"], n_boot=args.n_boot)
        dsr = deflated_sharpe_ratio(results[name]["oos_returns"], n_trials=diag["n_eff"],
                                    var_sr_trials=diag["var_sr_period"])
        finalists[name] = dict(bootstrap_p=boot["p_value"], dsr=dsr["dsr"], psr0=dsr["psr0"])
        print(f"  {name:<52} boot p={boot['p_value']:.3f}  DSR={dsr['dsr']:.2f}")

    slots = []
    for name in selected:
        res = results[name]
        tpl = res["template"]
        s, cp = res["summary"], res["cpcv"]
        slots.append(dict(
            slot=name, asset=res["asset"], template_name=tpl.name,
            template=asdict(tpl), weight=float(weights.get(name, 0.0)),
            research=dict(
                oos_sharpe=float(annualized_sharpe(rets[name])),
                oos_cagr=float(s["oos_cagr"]), oos_max_drawdown=float(s["oos_max_drawdown"]),
                wfe=None if not np.isfinite(s["wfe"]) else float(s["wfe"]),
                pct_profitable_windows=float(s["pct_profitable_windows"]),
                n_windows=int(s["n_windows"]), n_trades_oos=int(s["n_trades_oos"]),
                pardo_pass=bool(s["pardo_pass"]),
                cpcv_mean=float(cp["sharpe_mean"]), cpcv_std=float(cp["sharpe_std"]),
                cpcv_prob_negative=float(cp["prob_sharpe_negative"]),
                bootstrap_p=float(finalists[name]["bootstrap_p"]),
                dsr=float(finalists[name]["dsr"]),
            ),
        ))

    return dict(
        version=SPEC_VERSION,
        created=utcnow().isoformat(timespec="seconds"),
        assets=list(args.assets), config=cfg, slots=slots,
        diagnostics=diag, verdict=_verdict(diag, slots),
        curves=_curves(data, rets, nested, port),
        universe=_ranking_table(results, rets, selected),
    )


def _diagnostics(results, rets, nested, port, args) -> dict:
    n_trials = sum(r["n_trials"] for r in results.values())
    pbo = cscv_pbo(blocks=merge_block_stats([r["trial_blocks"] for r in results.values()]))
    pbo_slots = (cscv_pbo(rets.to_numpy(), n_partitions=16 if len(rets) >= 1600 else 8)
                 if rets.shape[1] > 1 else None)
    rc = reality_check(rets, n_boot=args.n_boot)
    oos_sharpes = rets.apply(annualized_sharpe)
    best = oos_sharpes.idxmax()
    eff = effective_n_trials(rets)
    dsr_best = deflated_sharpe_ratio(rets[best], n_trials=eff["n_eff"],
                                     var_sr_trials=eff["var_sr_period"])
    pr, ne = port["portfolio_returns"], nested["portfolio_equity"]
    out = dict(
        n_slots=int(rets.shape[1]), n_trials=int(n_trials),
        pbo_trials=float(pbo["pbo"]), degradation_slope=float(pbo["degradation_slope"]),
        prob_oos_loss=float(pbo["prob_oos_loss"]),
        pbo_slots=None if pbo_slots is None else float(pbo_slots["pbo"]),
        reality_check_best=str(rc["best"]), reality_check_p=float(rc["p_value"]),
        n_eff=int(eff["n_eff"]), var_sr_period=float(eff["var_sr_period"]),
        best_slot=str(best), best_oos_sharpe=float(dsr_best["sharpe_annual"]),
        sr_star_annual=float(dsr_best["sr_star_annual"]), dsr_best=float(dsr_best["dsr"]),
        min_btl_years=float(min_backtest_length(eff["n_eff"], max(float(oos_sharpes.max()), 1e-6))),
        years_available=float(len(rets) / periods_per_year()),
        static_sharpe=float(annualized_sharpe(pr)) if len(pr) > 2 else 0.0,
        nested_sharpe=float(nested["sharpe"]),
        nested_max_drawdown=float((ne / ne.cummax() - 1).min()) if len(ne) else 0.0,
        n_reselections=len(nested["selections"]),
    )
    print(f"  PBO over {n_trials} parameter trials: {out['pbo_trials']:.2f}"
          f"   Reality Check p={out['reality_check_p']:.3f}"
          f"   effective trials {out['n_eff']} of {out['n_slots']}")
    print(f"  best slot OOS Sharpe {out['best_oos_sharpe']:.2f} vs E[max of {out['n_eff']} noise "
          f"trials] {out['sr_star_annual']:.2f} -> DSR {out['dsr_best']:.2f}")
    print(f"  portfolio: static Sharpe {out['static_sharpe']:.2f} (biased), "
          f"NESTED walk-forward Sharpe {out['nested_sharpe']:.2f} (honest)")
    return out


def _verdict(diag: dict, slots: list) -> dict:
    """A plain reading of the diagnostics, carried onto the dashboard.

    The pipeline's whole point is that a good-looking backtest usually is not
    evidence. That conclusion has to survive the trip to the thing you look at
    every morning, so it is computed once here and shown at the top of the page.
    """
    reasons, level = [], "good"

    def fail(msg):
        nonlocal level
        reasons.append(msg)
        level = "critical"

    def warn(msg):
        nonlocal level
        reasons.append(msg)
        if level != "critical":
            level = "warning"

    if not slots:
        fail("No slot passed the candidate filter, so there is nothing to trade.")
    if diag["nested_sharpe"] <= 0:
        fail(f"Once the SELECTION is walked forward the portfolio's Sharpe is "
             f"{diag['nested_sharpe']:.2f}. The process did not make money out-of-sample.")
    elif diag["nested_sharpe"] < 0.5:
        warn(f"Nested walk-forward Sharpe is only {diag['nested_sharpe']:.2f} -- thin "
             f"reward for the drawdown risk of {diag['nested_max_drawdown']:.0%}.")
    if diag["pbo_trials"] >= 0.5:
        fail(f"Probability of Backtest Overfitting is {diag['pbo_trials']:.2f}: picking the "
             f"best parameters in-sample is no better than a coin toss out-of-sample.")
    elif diag["pbo_trials"] >= 0.35:
        warn(f"Probability of Backtest Overfitting is {diag['pbo_trials']:.2f} -- elevated.")
    if diag["reality_check_p"] > 0.15:
        fail(f"White's Reality Check p = {diag['reality_check_p']:.2f}: the best result in the "
             f"family is what you would expect from searching this many strategies over noise.")
    elif diag["reality_check_p"] > 0.05:
        warn(f"White's Reality Check p = {diag['reality_check_p']:.2f} -- not significant at 5%.")
    if diag["dsr_best"] < 0.95:
        warn(f"Deflated Sharpe Ratio of the best slot is {diag['dsr_best']:.2f} (<0.95): its "
             f"Sharpe is not distinguishable from the best of {diag['n_eff']} noise trials.")
    if diag["years_available"] < diag["min_btl_years"]:
        warn(f"Minimum backtest length for {diag['n_eff']} effective trials at this Sharpe is "
             f"{diag['min_btl_years']:.1f} years; only {diag['years_available']:.1f} are available.")
    weak = [s["slot"] for s in slots if s["research"]["cpcv_prob_negative"] > 0.3]
    if weak:
        warn(f"{len(weak)} of {len(slots)} slots lose money on more than 30% of their CPCV "
             f"paths, so their single walk-forward path flatters them.")

    headline = {
        "critical": "Do not trade this book. Treat the positions below as paper only.",
        "warning": "Trade small if at all. The evidence is weak in the ways listed.",
        "good": "The book survived every check. Size it as you would any live system.",
    }[level]
    return dict(level=level, headline=headline, reasons=reasons)


def _curves(data, rets, nested, port) -> dict:
    """Downsampled curves for the dashboard chart: the honest (nested
    walk-forward) portfolio against simply holding the assets equally weighted."""
    ne = nested["portfolio_equity"]
    if not len(ne):
        return dict(dates=[], strategy=[], buy_hold=[])
    bh_rets = pd.concat({a: data[a]["Close"].pct_change().fillna(0.0) for a in data},
                        axis=1).mean(axis=1)
    bh = (1 + bh_rets.reindex(ne.index).fillna(0.0)).cumprod()
    strat = ne / ne.iloc[0]
    step = max(1, len(strat) // 400)
    s, b = strat.iloc[::step], bh.iloc[::step]
    return dict(
        dates=[d.isoformat() for d in s.index],
        strategy=[round(float(v), 5) for v in s.values],
        buy_hold=[round(float(v), 5) for v in b.values],
    )


def _ranking_table(results, rets, selected) -> list:
    """Every slot ranked by OOS Sharpe, so the dashboard can show what was
    rejected as well as what was chosen (the denominator of the search)."""
    rows = []
    for name, res in results.items():
        if name not in rets.columns:
            continue
        s, cp = res["summary"], res["cpcv"]
        rows.append(dict(
            slot=name, asset=res["asset"],
            oos_sharpe=round(float(annualized_sharpe(rets[name])), 3),
            cpcv_mean=round(float(cp["sharpe_mean"]), 3),
            cpcv_prob_negative=round(float(cp["prob_sharpe_negative"]), 3),
            pardo_pass=bool(s["pardo_pass"]), selected=name in selected,
        ))
    return sorted(rows, key=lambda r: -r["oos_sharpe"])


# ==========================================================================
# signals phase
# ==========================================================================

def _live_state_path(state_dir):
    return os.path.join(state_dir, "live_state.json")


def _load_live_state(state_dir) -> dict:
    p = _live_state_path(state_dir)
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return dict(slots={}, last_targets={}, runs=0)


def _save_live_state(state_dir, st):
    with open(_live_state_path(state_dir), "w") as f:
        json.dump(st, f, indent=2, default=str)


def _prune_live_state(state_dir, keep: set):
    st = _load_live_state(state_dir)
    st["slots"] = {k: v for k, v in st["slots"].items() if k in keep}
    _save_live_state(state_dir, st)


def _load_holdings(state_dir) -> tuple[dict, bool]:
    """What you actually hold, if you told us. Signed share counts."""
    p = os.path.join(state_dir, "holdings.json")
    if not os.path.exists(p):
        return {}, False
    with open(p) as f:
        raw = json.load(f)
    return {str(k): float(v) for k, v in raw.items()}, True


def signals(args) -> dict:
    t0 = time.time()
    spec_path = os.path.join(args.state_dir, "portfolio.json")
    if not os.path.exists(spec_path):
        raise SystemExit(f"no {spec_path}: run `etf_dashboard.py research ...` first")
    with open(spec_path) as f:
        spec = json.load(f)
    if spec.get("version") != SPEC_VERSION:
        raise SystemExit(f"{spec_path} was written by another version "
                         f"({spec.get('version')} != {SPEC_VERSION}); re-run research")

    cfg = spec["config"]
    set_periods_per_year(cfg["periods_per_year"])
    interval = args.interval if args.interval != "1d" or cfg["interval"] == "1d" else cfg["interval"]
    if interval != cfg["interval"]:
        raise SystemExit(f"--interval {interval} but the research was done on "
                         f"{cfg['interval']} bars; re-run research to change it")
    now = pd.Timestamp(args.now) if args.now else None

    need = _history_bars_needed(spec)
    print(f"Loading {len(spec['assets'])} assets, {need}+ bars of {interval} history"
          f"{' [synthetic]' if args.synthetic else ''}...")
    data = load_assets(spec["assets"], interval=interval,
                       start=_start_for_bars(need, interval), synthetic=args.synthetic,
                       bars=max(args.bars, need + 200), now=now)
    have = len(next(iter(data.values())))
    if have < need:
        print(f"  WARNING: {have} shared bars, wanted {need}; refits use what is there")

    live = _load_live_state(args.state_dir)
    states, notes = [], []
    for slot in spec["slots"]:
        st, note = _slot_signal(slot, data[slot["asset"]], cfg, live, args)
        states.append(st)
        notes.extend(note)

    weights = {s["slot"]: s["weight"] for s in spec["slots"]}
    targets = portfolio_targets(states, weights, args.account_equity, max_gross=args.max_gross)
    holdings, holdings_given = _load_holdings(args.state_dir)
    if not holdings_given:
        holdings = {k: float(v) for k, v in live.get("last_targets", {}).items()}
    trades = trade_list(targets["by_asset"], holdings, lot=args.lot)

    run = dict(
        as_of=max((pd.Timestamp(s["as_of"]) for s in states), default=utcnow()),
        generated=utcnow(),
        account_equity=args.account_equity, max_gross=args.max_gross,
        holdings_source="holdings.json" if holdings_given else "previous run's targets",
        notes=notes, bars_available=have,
    )
    out_html = os.path.join(args.state_dir, "dashboard.html")
    render_dashboard(spec, states, targets, trades, run, out_html)
    _write_signal_csvs(args.state_dir, spec, states, targets, trades, run)

    live["last_targets"] = {a: float(v) for a, v in targets["by_asset"]["shares"].items()}
    live["runs"] = live.get("runs", 0) + 1
    live["last_run"] = run["generated"].isoformat(timespec="seconds")
    _save_live_state(args.state_dir, live)

    _print_signals(spec, states, targets, trades, run, out_html, time.time() - t0)
    return dict(spec=spec, states=states, targets=targets, trades=trades, run=run)


def _slot_signal(slot: dict, df: pd.DataFrame, cfg: dict, live: dict, args) -> tuple[dict, list]:
    """Current state of one slot, re-optimizing only if a test window has passed."""
    notes = []
    key = slot["slot"]
    base = StrategyTemplate(**slot["template"])
    mem = live["slots"].setdefault(key, dict(params=None, fitted_on=None))

    train = min(cfg["train_bars"], len(df))
    stale = due_for_refit(df, mem["fitted_on"], cfg["test_bars"])
    if mem["params"] is None or (stale and not args.no_refit):
        fit = refit_params(df, base, param_grid_for(base, wide=cfg["wide_grid"]),
                           train_bars=train, metric=cfg["metric"],
                           selection=cfg["selection"], anchored=cfg["anchored"])
        if fit["params"] is None:
            notes.append(f"{key}: nothing traded enough in-sample to fit -- slot stays flat")
            mem["params"], mem["fitted_on"] = None, str(df.index[-1])
        else:
            mem["params"] = {k: _jsonable(v) for k, v in fit["params"].items()}
            mem["fitted_on"] = str(fit["fitted_on"])
            mem["is_sharpe"] = float(fit["is_stats"]["sharpe"])
            mem["is_trades"] = int(fit["is_stats"]["n_trades"])
            notes.append(f"{key}: re-optimized on the last {train} bars -> "
                         + ", ".join(f"{k}={v}" for k, v in mem["params"].items()))
    elif stale and args.no_refit:
        notes.append(f"{key}: a refit is due but --no-refit was given")

    if mem["params"] is None:
        st = dict(slot=key, asset=slot["asset"], template=slot["template_name"],
                  as_of=df.index[-1], last_close=float(df["Close"].iloc[-1]), atr=np.nan,
                  equity_slot=args.account_equity * slot["weight"], position=None, shares=0.0,
                  entry_price=None, entry_date=None, bars_held=0, unrealized=0.0,
                  n_trades_in_window=0, exit_orders=[], entry_orders=[],
                  blocked_by=["no parameter set could be fitted"], params={},
                  fitted_on=mem["fitted_on"], refit_due=False, weight=slot["weight"],
                  research=slot["research"])
        return st, notes

    tpl = base.with_params(**mem["params"])
    st = strategy_state(df, tpl, equity=args.account_equity * slot["weight"])
    st.update(slot=key, asset=slot["asset"], template=slot["template_name"],
              params=mem["params"], fitted_on=mem["fitted_on"], weight=slot["weight"],
              research=slot["research"],
              bars_since_refit=int((df.index > pd.Timestamp(mem["fitted_on"])).sum()),
              refit_due=due_for_refit(df, mem["fitted_on"], cfg["test_bars"]))
    return st, notes


def _jsonable(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


def _write_signal_csvs(state_dir, spec, states, targets, trades, run):
    trades.to_csv(os.path.join(state_dir, "signals.csv"))
    rows = []
    for st in states:
        stop = next((o["level"] for o in st["exit_orders"] if o["kind"] == "stop"), None)
        rows.append(dict(
            generated=run["generated"], as_of=st["as_of"], slot=st["slot"], asset=st["asset"],
            weight=st["weight"], position=st["position"] or 0, shares=round(st["shares"], 4),
            price=st["last_close"], stop=stop, unrealized=round(st["unrealized"], 2),
            entry_orders="; ".join(
                f"{'BUY' if o['side'] == 1 else 'SELL'} {o['kind']}"
                + (f" @{o['level']:.4f}" if o.get("level") is not None else "")
                for o in st["entry_orders"]),
        ))
    log = pd.DataFrame(rows)
    p = os.path.join(state_dir, "signal_log.csv")
    log.to_csv(p, mode="a", header=not os.path.exists(p), index=False)


def _print_signals(spec, states, targets, trades, run, out_html, secs):
    v = spec["verdict"]
    print(f"\n[{v['level'].upper()}] {v['headline']}")
    print(f"\nAs of {run['as_of']}  |  account ${run['account_equity']:,.0f}  |  "
          f"gross {targets['gross_exposure']:.0%}  net {targets['net_exposure']:+.0%}  |  "
          f"risk if every stop hits {targets['open_risk_pct']:.1%}")
    if targets["scale_applied"] < 1:
        print(f"  (book scaled to {targets['scale_applied']:.2f} to respect "
              f"--max-gross {run['max_gross']:g})")
    open_pos = [s for s in states if s["position"]]
    print(f"\n{len(open_pos)} of {len(states)} slots hold a position:")
    for s in open_pos:
        stop = next((o["level"] for o in s["exit_orders"] if o["kind"] == "stop"), float("nan"))
        print(f"  {'LONG ' if s['position'] == 1 else 'SHORT'} {s['asset']:<5} "
              f"{s['shares']:>9.1f} sh @ {s['entry_price']:.2f}  stop {stop:.2f}  "
              f"P&L {s['unrealized']:>+9.0f}  [{s['template']}]")
    work = [(s, o) for s in states for o in s["entry_orders"]]
    if work:
        print(f"\n{len(work)} entry order(s) to have working on the next bar:")
        for s, o in work:
            lvl = f"@ {o['level']:.2f}" if o.get("level") is not None else "at the open"
            print(f"  {'BUY ' if o['side'] == 1 else 'SELL'} {o['shares']:>9.1f} {s['asset']:<5} "
                  f"{o['kind']:<16} {lvl:<14} {o['note']}")
    todo = trades[trades["action"] != "hold"]
    print(f"\nTrades to send (vs {run['holdings_source']}):")
    if todo.empty:
        print("  nothing -- the book already matches")
    else:
        for a, r in todo.iterrows():
            print(f"  {r['action']:<4} {r['order_shares']:>9.1f} {a:<5} "
                  f"~${r['order_notional']:>11,.0f}  (held {r['held']:g} -> target {r['target']:g})")
    for n in run["notes"]:
        print(f"  note: {n}")
    print(f"\nDashboard: {out_html}  ({secs:.1f}s)")


# ==========================================================================
# research report
# ==========================================================================

def _write_research_report(spec, path):
    d, v, c = spec["diagnostics"], spec["verdict"], spec["config"]
    L = [f"# ETF strategy research -- {spec['created']}\n\n",
         f"Assets: {', '.join(spec['assets'])} on {c['interval']} bars from {c['start']}\n\n",
         f"Family '{c['family']}': {d['n_slots']} (asset, template) slots, "
         f"{d['n_trials']} parameter trials in total.\n",
         f"Walk-forward train={c['train_bars']} test={c['test_bars']} "
         f"{'anchored' if c['anchored'] else 'rolling'}, selection={c['selection']}, "
         f"costs {c['cost_bps']} bps/side, {c['risk_pct']:.1%} equity risked per trade.\n\n",
         f"## Verdict: {v['level'].upper()}\n\n{v['headline']}\n\n"]
    for r in v["reasons"]:
        L.append(f"- {r}\n")
    L.append("\n## Diagnostics\n\n")
    L.append(f"- Probability of Backtest Overfitting over all {d['n_trials']} trials: "
             f"**{d['pbo_trials']:.2f}** (OOS-vs-IS slope {d['degradation_slope']:.2f}, "
             f"P(OOS loss | IS best) {d['prob_oos_loss']:.2f})\n")
    if d["pbo_slots"] is not None:
        L.append(f"- PBO of the slot-selection step ({d['n_slots']} OOS curves): "
                 f"**{d['pbo_slots']:.2f}**\n")
    L.append(f"- White's Reality Check for {d['reality_check_best']}: p = "
             f"**{d['reality_check_p']:.3f}**\n")
    L.append(f"- Effective independent trials: {d['n_eff']} correlation clusters among "
             f"{d['n_slots']} slots\n")
    L.append(f"- Best slot {d['best_slot']}: OOS Sharpe {d['best_oos_sharpe']:.2f} vs E[max of "
             f"{d['n_eff']} noise trials] {d['sr_star_annual']:.2f} -> DSR "
             f"**{d['dsr_best']:.2f}**\n")
    L.append(f"- Minimum backtest length: {d['min_btl_years']:.1f} years needed, "
             f"{d['years_available']:.1f} available\n")
    L.append(f"- Portfolio Sharpe: {d['static_sharpe']:.2f} with the selection fitted on all "
             f"history (biased), **{d['nested_sharpe']:.2f}** with the selection walked forward "
             f"over {d['n_reselections']} re-selections (honest), max drawdown "
             f"{d['nested_max_drawdown']:.1%}\n\n")
    L.append("## Selected slots\n\n")
    L.append("| slot | weight | OOS Sharpe | WFE | prof. windows | Pardo | CPCV mean+/-sd | "
             "P(CPCV<0) | boot p | DSR |\n|---|---|---|---|---|---|---|---|---|---|\n")
    for s in spec["slots"]:
        r = s["research"]
        L.append(f"| {s['slot']} | {s['weight']:.2f} | {r['oos_sharpe']:.2f} | "
                 f"{'n/a' if r['wfe'] is None else format(r['wfe'], '.2f')} | "
                 f"{r['pct_profitable_windows']:.0%} | {'yes' if r['pardo_pass'] else 'no'} | "
                 f"{r['cpcv_mean']:.2f}+/-{r['cpcv_std']:.2f} | {r['cpcv_prob_negative']:.0%} | "
                 f"{r['bootstrap_p']:.3f} | {r['dsr']:.2f} |\n")
    L.append("\n## How to read this\n\n"
             "- PBO near 0.5 or above: the in-sample best parameter choice is a coin toss.\n"
             "- DSR below 0.95: the best Sharpe is not distinguishable from the best of that "
             "many noise trials.\n"
             "- Reality Check p above 0.05-0.15: consistent with data snooping.\n"
             "- CPCV straddling zero: the single walk-forward path was luck.\n"
             "- The nested walk-forward portfolio Sharpe is the only portfolio number worth "
             "quoting: it is out-of-sample with respect to the parameters AND the selection.\n")
    with open(path, "w") as f:
        f.writelines(L)


def main(argv=None):
    """Returns the phase's result dict, so the pipeline can be driven from Python
    (and tested) as well as from a shell."""
    args = parse_args(argv)
    return research(args) if args.cmd == "research" else signals(args)


if __name__ == "__main__":
    main()
