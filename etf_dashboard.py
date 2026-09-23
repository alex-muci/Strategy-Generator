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
    futures_orders.csv, futures_levels.csv   with `signals --futures`: the book in whole
                        contracts and every working level as a futures price
                        (futures_map.py); held contracts come from
                        holdings_futures.json, {"MES": 2, "ZN": -1}
    futures_quotes.json YOU maintain this too, only for contracts Yahoo has no
                        series for (the Eurex Bund and BTP): {"FGBL": 129.55,
                        "FBTP": {"price": 118.2, "hedge_ratio": 0.85}}

ETFs listed in euros (EXHD.DE, IITB.MI) are loaded in euros and restated in
dollars at today's rate: the returns are the local ones the future pays, the
notional is in dollars. See futures_map.LISTINGS.

Nothing here places an order. It tells you what to place.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict

import numpy as np
import pandas as pd

from data import synthetic_ohlc
from generator import generate_templates, param_grid_for, FAMILIES
from walkforward import warmup_bars
from live import (
    refit_params, due_for_refit, strategy_state,
    portfolio_targets, trade_list, utcnow,
)
from pipeline import (
    resolve_interval, load_real, eval_config, worker_pool, evaluate_slots,
    family_diagnostics, build_portfolios, finalist_stats, sizing_text,
)
from portfolio import returns_frame
from strategy import (
    StrategyTemplate, annualized_sharpe, max_drawdown, set_periods_per_year, periods_per_year,
    periods_per_year_for_interval, SIDES, BARS_PER_YEAR, HEDGE_LEARNERS,
)
from futures_map import (
    CONTRACTS, LISTINGS, FX_SYMBOLS, HEDGE_RATIO_BARS, QUOTES_STALE_DAYS,
    to_contracts, translate_orders, parse_quotes,
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
    r.add_argument("--family", default="quick", choices=list(FAMILIES))
    r.add_argument("--max-templates", type=int, default=None)
    r.add_argument("--sides", nargs="+", default=None, choices=SIDES,
                   help="restrict every template to these sides (default: the family's own list)")
    r.add_argument("--sides-map", nargs="+", default=None, metavar="ASSET=SIDE",
                   help="sides per asset, e.g. SPY=long_only QQQ=long_only; the others keep --sides. "
                        "Whether an asset drifts is a fact about the asset: decide it BEFORE the run")
    r.add_argument("--portfolio-vol", type=float, default=0.0,
                   help="annualized volatility the whole book is scaled to (e.g. 0.15), from the "
                        "realized vol of the nested walk-forward curve; 0 = no scaling. Frozen in "
                        "portfolio.json, never re-estimated by the signals phase")
    r.add_argument("--train", type=int, default=500, help="training window (bars)")
    r.add_argument("--test", type=int, default=125, help="test window (bars)")
    r.add_argument("--anchored", action="store_true", help="expanding training window")
    r.add_argument("--metric", default="sharpe", choices=["sharpe", "return_over_dd", "profit_factor"])
    r.add_argument("--selection", default="plateau", choices=["plateau", "best"])
    r.add_argument("--wide-grid", action="store_true")
    r.add_argument("--cost-bps", type=float, default=5.0, help="commission+slippage per side")
    r.add_argument("--risk-pct", type=float, default=0.01, help="equity risked per trade, per slot")
    r.add_argument("--max-leverage", type=float, default=2.0)
    r.add_argument("--vol-target", type=float, default=0.0,
                   help="annualized volatility each entry is sized to, per slot (e.g. 0.15); "
                        "0 = risk --risk-pct on the ATR stop. The same target gives every asset the same risk")
    r.add_argument("--vol-target-n", type=int, default=60, help="bars of close-to-close returns in the realized-vol estimate")
    r.add_argument("--hedge-learner", default="window", choices=list(HEDGE_LEARNERS),
                   help="memory of the online learner (hedge channel, learned direction): 'discounted' fades old bars with a half-life, 'floored' adds a weight floor to it, 'window' is the original hard 250-bar window")
    r.add_argument("--min-sharpe", type=float, default=0.3)
    r.add_argument("--max-strategies", type=int, default=8)
    r.add_argument("--corr-ceiling", type=float, default=0.6)
    r.add_argument("--require-pardo", action="store_true")
    r.add_argument("--select-method", default="greedy", choices=["greedy", "cluster"])
    r.add_argument("--weighting", default="equal", choices=["equal", "hrp"])
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
    s.add_argument("--risk-scale", type=float, default=None,
                   help="multiply the equity every slot is sized on (default: the research's "
                        "--portfolio-vol scale, 1 if there was none)")
    s.add_argument("--futures", action="store_true",
                   help="also restate the book in whole futures contracts (futures_map.py): "
                        "futures_orders.csv and a section on the page. Held contracts come from "
                        "holdings_futures.json, e.g. {\"MES\": 2, \"ZN\": -1}")
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
        listing = LISTINGS.get(a)
        if synthetic:
            # a different seed per asset, so the assets are not the same series
            raw[a] = synthetic_ohlc(n_bars=bars, seed=100 + 7 * i, trend_prob=0.45 + 0.05 * i)
        else:
            raw[a] = load_real(a, start=start, interval=interval, now=now,
                               session_close=None if listing is None else listing.session_close)
        if not quiet:
            print(f"  {a}: {len(raw[a])} bars, {raw[a].index[0].date()} to {raw[a].index[-1].date()}")
        if listing is not None and not synthetic and listing.currency != "USD":
            # a euro ETF in today's dollars: local returns, dollar notional (what
            # a hedged share class shows; the future's P&L accrues in euros too)
            rate = fx_rate(listing.currency, interval=interval, now=now)
            raw[a] = raw[a].assign(**{c: raw[a][c] * rate for c in ("Open", "High", "Low", "Close")})
            if not quiet:
                print(f"  {a}: {listing.currency} listing restated in USD at {rate:.4f}")

    common = None
    for a in assets:
        common = raw[a].index if common is None else common.intersection(raw[a].index)
    if len(common) == 0:
        raise ValueError("the assets share no bars at all -- check the symbols and the interval")
    dropped = {a: len(raw[a]) - len(common) for a in assets}
    if not quiet and any(dropped.values()):
        print(f"  aligned on {len(common)} shared bars (dropped {dropped})")
    return {a: raw[a].loc[common] for a in assets}


_FX_CACHE: dict = {}


def fx_rate(currency: str, *, interval: str = "1d", now=None) -> float:
    """Dollars per one unit of `currency`, from the last closed bar of the
    Yahoo cross; fetched once per run. A currency without a symbol, or with no
    data, is an error: sizing a euro book on a guessed rate is a silent 10%."""
    if currency == "USD":
        return 1.0
    if currency not in _FX_CACHE:
        sym = FX_SYMBOLS.get(currency)
        if sym is None:
            raise ValueError(f"no Yahoo symbol for {currency}/USD; add it to futures_map.FX_SYMBOLS")
        # only the last close is used, but load_yfinance refuses fewer than
        # 200 bars: ask for the window the futures prices use, not a rate-sized one
        df = load_real(sym, start=_start_for_bars(HEDGE_RATIO_BARS + 100, interval),
                       interval=interval, now=now)
        _FX_CACHE[currency] = float(df["Close"].iloc[-1])
    return _FX_CACHE[currency]


def _slot_template(slot: dict) -> StrategyTemplate:
    """The StrategyTemplate a spec slot was researched with. A spec written
    before hedge_learner existed was researched with the window learner."""
    return StrategyTemplate(**{"hedge_learner": "window", **slot["template"]})


def _history_bars_needed(spec: dict) -> int:
    """Bars of history the signals phase needs: one training window, plus the
    longest warm-up any of the spec's templates asks for, plus slack."""
    warm = max((warmup_bars(_slot_template(s)) for s in spec.get("slots", [])), default=0)
    return int(spec["config"]["train_bars"] * 1.25) + max(400, warm + 100)


def _start_for_bars(n_bars: int, interval: str) -> str:
    """A start date comfortably older than `n_bars` bars ago."""
    per_year = periods_per_year_for_interval(interval)
    years = n_bars / per_year * 1.6 + 0.5
    return (utcnow() - pd.Timedelta(days=365.25 * years)).strftime("%Y-%m-%d")


# ==========================================================================
# research phase
# ==========================================================================

def research(args) -> dict:
    t0 = time.time()
    sides_map = _parse_sides_map(args.sides_map, args.assets)
    os.makedirs(args.state_dir, exist_ok=True)
    interval = resolve_interval(args.interval, args.synthetic)
    if interval != args.interval:
        print(f"NOTE: --interval {args.interval} ignored, the synthetic series is {interval} bars")
    args.interval = interval

    print(f"Loading {len(args.assets)} assets ({args.interval} bars)"
          f"{' [synthetic]' if args.synthetic else ''}...")
    data = load_assets(args.assets, interval=args.interval, start=args.start,
                       synthetic=args.synthetic, bars=args.bars)
    n_bars = len(next(iter(data.values())))
    if n_bars < args.train + 2 * args.test:
        raise SystemExit(
            f"only {n_bars} shared bars: too few for train={args.train} + test={args.test}. "
            f"Use an earlier --start, a coarser --interval, or smaller windows.")

    by_sides = {}

    def templates_for(asset):
        sides = sides_map.get(asset, args.sides)
        key = tuple(sides) if sides else None
        if key not in by_sides:
            overrides = {"sides": list(sides)} if sides else {}
            by_sides[key] = generate_templates(args.family, max_templates=args.max_templates, **overrides)
        return by_sides[key]

    cfg = eval_config(args, args.interval) | dict(
        start=args.start, family=args.family, sides=args.sides,
        sides_map={a: list(s) for a, s in sides_map.items()}, portfolio_vol=args.portfolio_vol,
        min_sharpe=args.min_sharpe, corr_ceiling=args.corr_ceiling,
        weighting=args.weighting, select_method=args.select_method,
    )
    jobs = [(f"{a}|{t.name}", a, t) for a in args.assets for t in templates_for(a)]
    print(f"{len(templates_for(args.assets[0]))} templates x {len(args.assets)} assets = {len(jobs)} slots; "
          f"walk-forward train={args.train} test={args.test} "
          f"{'anchored' if args.anchored else 'rolling'}, {args.cost_bps} bps/side")

    def progress(i, total, key, res):
        if i % max(1, total // 20) == 0 or i == total:
            print(f"  [{i}/{total}] {key:<52} oos_sharpe="
                  f"{res['summary']['oos_sharpe']:>6.2f} cpcv={res['cpcv']['sharpe_mean']:>6.2f}")

    with worker_pool(args.jobs, data, cfg) as pool:
        results = evaluate_slots(jobs, pool, on_result=progress)
        spec = _assemble_spec(data, results, args, cfg)

    path = os.path.join(args.state_dir, "portfolio.json")
    with open(path, "w", encoding="utf-8") as f:
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


def _parse_sides_map(items, assets) -> dict:
    """{'SPY': ['long_only']} from ['SPY=long_only']; an unknown asset or side is
    an error, not a slot that silently trades both ways."""
    out = {}
    for item in items or []:
        asset, _, side = item.partition("=")
        if asset not in assets:
            raise SystemExit(f"--sides-map {item}: {asset!r} is not in --assets")
        if side not in SIDES:
            raise SystemExit(f"--sides-map {item}: side must be one of {SIDES}")
        out.setdefault(asset, [])
        if side not in out[asset]:
            out[asset].append(side)
    return out


RISK_SCALE_BOUNDS = (0.5, 10.0)


def _risk_scale(nested: dict, target: float) -> tuple[float, float]:
    """(scale, realized vol of the nested curve). Slot sizes are linear in the
    equity they are sized on (until --max-leverage binds), so the book reaches
    `target` when that equity is multiplied by target / realized. Windows where
    the selection step picked nothing are left out: they are a flat line, and
    counting them would read a book that is sometimes empty as a calm one."""
    r = nested["portfolio_returns"]
    sels = sorted(nested["selections"], key=lambda s: s["period_start"])
    keep = np.ones(len(r), dtype=bool)
    for s, nxt in zip(sels, sels[1:] + [None]):
        if not s["selected"]:
            keep &= ~((r.index >= pd.Timestamp(s["period_start"]))
                      & (True if nxt is None else r.index < pd.Timestamp(nxt["period_start"])))
    r = r[keep]
    vol = float(r.std() * np.sqrt(periods_per_year())) if len(r) > 2 else 0.0
    if target <= 0 or vol <= 0:
        return 1.0, vol
    return float(np.clip(target / vol, *RISK_SCALE_BOUNDS)), vol


def _assemble_spec(data, results, args, cfg) -> dict:
    rets = returns_frame(results)
    if rets.empty:
        raise SystemExit("no slot produced a usable out-of-sample series -- history too short")

    port, nested = build_portfolios(results, rets, args)

    print("\nFamily-level diagnostics over every (asset, template) trial...")
    diag = _diagnostics(results, rets, nested, port, args)
    selected = port["selected"]
    weights = port["weights"] if len(selected) else pd.Series(dtype=float)

    finalists = finalist_stats(results, selected, diag, n_boot=args.n_boot)
    for name, f in finalists.items():
        print(f"  {name:<52} boot p={f['bootstrap_p']:.3f}  DSR={f['dsr']:.2f}")

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
                oos_exposure=float(s.get("oos_exposure", 0.0)),
                oos_notional=float(s.get("oos_notional", 0.0)),
                oos_avg_net_exposure=float(s.get("oos_avg_net_exposure", 0.0)),
                cpcv_mean=float(cp["sharpe_mean"]), cpcv_std=float(cp["sharpe_std"]),
                cpcv_prob_negative=float(cp["prob_sharpe_negative"]),
                bootstrap_p=float(finalists[name]["bootstrap_p"]),
                dsr=float(finalists[name]["dsr"]),
            ),
        ))

    scale, nested_vol = _risk_scale(nested, args.portfolio_vol)
    diag["nested_vol"] = nested_vol
    print(f"  nested curve realized vol {nested_vol:.1%}"
          + (f" -> risk scale {scale:.2f} for a {args.portfolio_vol:.0%} book" if args.portfolio_vol > 0 else ""))
    if args.portfolio_vol > 0 and scale in RISK_SCALE_BOUNDS:
        print(f"  NOTE: the risk scale hit its bound {scale:g}; the book will not reach the target")

    return dict(
        version=SPEC_VERSION,
        created=utcnow().isoformat(timespec="seconds"),
        assets=list(args.assets), config=cfg, slots=slots, risk_scale=scale,
        diagnostics=diag, verdict=_verdict(diag, slots),
        curves=_curves(data, rets, nested, port),
        universe=_ranking_table(results, rets, selected),
    )


def _diagnostics(results, rets, nested, port, args) -> dict:
    # the numbers are pipeline.family_diagnostics' (the same ones main.py
    # reports); this only flattens them into JSON and adds the portfolio's
    fam = family_diagnostics(results, rets, n_boot=args.n_boot)
    n_trials, pbo, pbo_slots, rc = fam["n_trials"], fam["pbo_trials"], fam["pbo_templates"], fam["reality_check"]
    dsr_best = fam["dsr_best"]
    pr, ne = port["portfolio_returns"], nested["portfolio_equity"]
    out = dict(
        n_slots=int(rets.shape[1]), n_trials=int(n_trials),
        pbo_trials=float(pbo["pbo"]), degradation_slope=float(pbo["degradation_slope"]),
        prob_oos_loss=float(pbo["prob_oos_loss"]),
        pbo_slots=None if pbo_slots is None else float(pbo_slots["pbo"]),
        reality_check_best=str(rc["best"]), reality_check_p=float(rc["p_value"]),
        n_eff=int(fam["n_eff"]), var_sr_period=float(fam["var_sr_period"]),
        best_slot=str(fam["best_template"]), best_oos_sharpe=float(dsr_best["sharpe_annual"]),
        sr_star_annual=float(dsr_best["sr_star_annual"]), dsr_best=float(dsr_best["dsr"]),
        dsr_raw=float(fam["dsr_raw"]["dsr"]),
        min_btl_years=float(fam["min_btl_years"]),
        years_available=float(fam["years_available"]),
        static_sharpe=float(annualized_sharpe(pr)) if len(pr) > 2 else 0.0,
        nested_sharpe=float(nested["sharpe"]),
        nested_max_drawdown=max_drawdown(ne),
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

    # a spec written before --portfolio-vol existed was researched unscaled
    risk_scale = float(spec.get("risk_scale", 1.0) if args.risk_scale is None else args.risk_scale)
    sizing_equity = args.account_equity * risk_scale

    live = _load_live_state(args.state_dir)
    states, notes = [], []
    for slot in spec["slots"]:
        st, note = _slot_signal(slot, data[slot["asset"]], cfg, live, args, sizing_equity)
        states.append(st)
        notes.extend(note)

    # exposure and the gross cap stay against the REAL account
    weights = {s["slot"]: s["weight"] for s in spec["slots"]}
    targets = portfolio_targets(states, weights, args.account_equity, max_gross=args.max_gross)
    holdings, holdings_given = _load_holdings(args.state_dir)
    if not holdings_given:
        holdings = {k: float(v) for k, v in live.get("last_targets", {}).items()}
    trades = trade_list(targets["by_asset"], holdings, lot=args.lot)

    futures = _futures_book(args, spec, data, states, targets, live, interval, now) if args.futures else None
    if futures:
        notes.extend(futures["notes"])

    run = dict(
        as_of=max((pd.Timestamp(s["as_of"]) for s in states), default=utcnow()),
        generated=utcnow(),
        account_equity=args.account_equity, max_gross=args.max_gross, risk_scale=risk_scale,
        holdings_source="holdings.json" if holdings_given else "previous run's targets",
        notes=notes, bars_available=have, futures=futures,
    )
    out_html = os.path.join(args.state_dir, "dashboard.html")
    render_dashboard(spec, states, targets, trades, run, out_html)
    _write_signal_csvs(args.state_dir, spec, states, targets, trades, run)

    live["last_targets"] = {a: float(v) for a, v in targets["by_asset"]["shares"].items()}
    if futures:
        futures["book"].to_csv(os.path.join(args.state_dir, "futures_orders.csv"))
        futures["orders"].to_csv(os.path.join(args.state_dir, "futures_levels.csv"), index=False)
        live["last_futures_targets"] = {r["root"]: int(r["target"]) for _, r in futures["book"].iterrows()}
    live["runs"] = live.get("runs", 0) + 1
    live["last_run"] = run["generated"].isoformat(timespec="seconds")
    _save_live_state(args.state_dir, live)

    _print_signals(spec, states, targets, trades, run, out_html, time.time() - t0)
    return dict(spec=spec, states=states, targets=targets, trades=trades, run=run)


def _futures_book(args, spec, data, states, targets, live, interval, now) -> dict:
    """The ETF book restated in whole contracts, against the contracts held."""
    fut, series, fx, notes = {}, {}, {}, []
    quotes, quote_notes = _load_futures_quotes(args.state_dir)
    notes.extend(quote_notes)
    since = _start_for_bars(HEDGE_RATIO_BARS + 100, interval)
    for a in spec["assets"]:
        c = CONTRACTS.get(a)
        if c is None:
            continue
        if args.synthetic:
            # no feed: the ETF stands in for its own future, rescaled to a
            # price at which one contract is worth a plausible amount. A
            # contract Yahoo has no series for takes the same path as live:
            # its price comes from futures_quotes.json or it is left out.
            fx.setdefault(c.currency, 1.0)
            if c.yahoo is not None:
                lo, hi = c.notional_range
                k = np.sqrt(lo * hi) / c.value(data[a]["Close"].iloc[-1] * c.yahoo_scale)
                fut[a] = data[a][["Open", "High", "Low", "Close"]] * k
            continue
        if c.currency != "USD":
            try:
                fx[c.currency] = fx_rate(c.currency, interval=interval, now=now)
            except ValueError as e:
                notes.append(f"{a}: {e}")
        if c.yahoo is not None:
            try:
                fut[a] = load_real(c.yahoo, start=since, interval=interval, now=now)
            except ValueError as e:
                notes.append(f"{a}: {e}")
        if c.series is not None and c.series != c.yahoo:
            try:
                series[a] = load_real(c.series, start=since, interval=interval, now=now)
            except ValueError as e:
                notes.append(f"{a}: hedge-ratio series {c.series}: {e}")
    p = os.path.join(args.state_dir, "holdings_futures.json")
    given = os.path.exists(p)
    if given:
        with open(p) as f:
            held = {str(k): float(v) for k, v in json.load(f).items()}
    else:
        held = {k: float(v) for k, v in live.get("last_futures_targets", {}).items()}
    out = to_contracts(targets["by_asset"], data, fut, held, fx=fx, quotes=quotes, series=series)
    out["orders"] = translate_orders(states, out["conversions"])
    out["notes"] = notes + out["notes"]
    out["holdings_source"] = "holdings_futures.json" if given else "previous run's targets"
    out["fx"] = fx
    del out["conversions"]      # holds dataclasses; everything downstream reads the frames
    return out


def _load_futures_quotes(state_dir) -> tuple[dict, list]:
    """Hand-kept prices for contracts Yahoo has no series for, and a note when
    the file is old enough that the price probably is too."""
    p = os.path.join(state_dir, "futures_quotes.json")
    if not os.path.exists(p):
        return {}, []
    with open(p) as f:
        quotes = parse_quotes(json.load(f))
    age = (time.time() - os.path.getmtime(p)) / 86400
    notes = []
    if age > QUOTES_STALE_DAYS:
        notes.append(f"futures_quotes.json is {age:.0f} days old: {', '.join(sorted(quotes))} "
                     f"are sized on a stale price")
    return quotes, notes


def _slot_signal(slot: dict, df: pd.DataFrame, cfg: dict, live: dict, args,
                 sizing_equity: float | None = None) -> tuple[dict, list]:
    """Current state of one slot, re-optimizing only if a test window has passed.
    `sizing_equity` is the account equity times the research's risk scale."""
    equity = (args.account_equity if sizing_equity is None else sizing_equity) * slot["weight"]
    notes = []
    key = slot["slot"]
    base = _slot_template(slot)
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
                  equity_slot=equity, position=None, shares=0.0,
                  entry_price=None, entry_date=None, bars_held=0, unrealized=0.0,
                  n_trades_in_window=0, exit_orders=[], entry_orders=[],
                  blocked_by=["no parameter set could be fitted"], params={},
                  fitted_on=mem["fitted_on"], refit_due=False, weight=slot["weight"],
                  research=slot["research"])
        return st, notes

    tpl = base.with_params(**mem["params"])
    st = strategy_state(df, tpl, equity=equity)
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
    if run["risk_scale"] != 1:
        print(f"  (slots sized on ${run['account_equity'] * run['risk_scale']:,.0f}: "
              f"risk scale {run['risk_scale']:.2f})")
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
    fut = run.get("futures")
    if fut:
        print(f"\nFutures book (vs {fut['holdings_source']}), rounding error "
              f"${fut['rounding_error']:,.0f} = {fut['rounding_error_pct']:.0%} of the ETF book:")
        if fut["book"].empty:
            print("  flat -- nothing to hold")
        for a, r in fut["book"].iterrows():
            todo = "hold" if r["action"] == "hold" else f"{r['action']} {r['order_contracts']:g}"
            ccy = "" if r["currency"] == "USD" else f" {r['currency']}"
            print(f"  {r['root']:<4} ({a:<7}) target {r['target']:>+4d}  held {r['held']:>+4d}  "
                  f"{todo:<8} wanted {r['raw']:>+6.2f} @ {r['fut_price']:g}{ccy} ({r['price_source']})  "
                  f"hedge ratio {r['hedge_ratio']:.2f} ({r['ratio_source']})")
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
         f"costs {c['cost_bps']} bps/side, {sizing_text(c)}.\n\n",
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
             f"{d['nested_max_drawdown']:.1%}\n")
    L.append(f"- Realized volatility of the nested curve: {d['nested_vol']:.1%} a year"
             + (f"; slots are sized on equity x **{spec['risk_scale']:.2f}** to run the book at "
                f"{c['portfolio_vol']:.0%} (drawdowns scale with it)" if c.get("portfolio_vol") else "")
             + "\n\n")
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
