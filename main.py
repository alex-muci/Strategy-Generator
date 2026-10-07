"""
main.py
-------
End-to-end Ranger-style pipeline with Pardo + Lopez de Prado robustness:

  1. GENERATE   many structurally distinct breakout strategy templates
                (generator.py)
  2. EVALUATE   each template with walk-forward analysis: optimize
                in-sample (plateau selection), roll forward, test
                out-of-sample only (walkforward.py)
  3. STRESS     the family: Probability of Backtest Overfitting (CSCV),
                White's Reality Check, Deflated Sharpe; the finalists:
                Combinatorial Purged CV paths, bootstrap p-values,
                Pardo's walk-forward matrix (robustness.py)
  4. SELECT     an uncorrelated subset (portfolio.py) -- and re-do that
                selection walk-forward so the quoted portfolio curve is
                out-of-sample with respect to the selection too
  5. REPORT     equity curves, heatmaps, distributions, report.md

Usage:
  python main.py                       # synthetic data, 'quick' family (72 templates)
  python main.py --family default      # 768 templates
  python main.py --real SPY --start 2005-01-01 --family default --jobs 8
  python main.py --csv brent_z25z26.csv --instrument-map brent_z25z26=1000,15,3000,30   # a future, named by its file
  python main.py --help

The research flags are shared with `etf_dashboard.py research`
(pipeline.add_research_args). On a margined future the benchmark is one
unit's P&L: its Sharpe, correlation and information ratio are scale-free, its
beta is not (it scales with 1 / point value: units held on average).
"""

from __future__ import annotations
import argparse
import os
import re
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import synthetic_ohlc, load_csv
from generator import generate_templates
from pipeline import (
    resolve_interval, load_real, eval_config, worker_pool, evaluate_slots, walk_forward_matrices,
    family_diagnostics, build_portfolios, finalist_stats, benchmark_stats, benchmark_returns, instrument_text,
    benchmark_curve, _costed, BENCH_BUY_HOLD, add_research_args, parse_instrument_map, instrument_of,
)
from portfolio import returns_frame
from replay import save_run, replay_run, TARGETS as REPLAY_TARGETS
from strategy import (annualized_sharpe, max_drawdown, periods_per_year, BARS_PER_YEAR, validate_instrument,
                      instrument_warnings, typical_units, set_periods_per_year, set_hedge_share,
                      RESEARCH_EQUITY)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Ranger-style strategy generator with robust walk-forward evaluation")
    # the research flags etf_dashboard.py research takes too (pipeline.add_research_args)
    add_research_args(p, start="2005-01-01")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--real", metavar="TICKER", default=None, help="use yfinance data for TICKER instead of synthetic")
    src.add_argument("--csv", metavar="PATH", default=None,
                     help="use OHLC(V) bars from a local CSV (first column the date, then Open High Low Close "
                          "[Volume] [Roll]); prices may be zero or negative, e.g. a futures calendar spread, which "
                          "then needs --margin-per-unit and --cost-per-unit")
    p.add_argument("--interval", default="1d", choices=sorted(BARS_PER_YEAR),
                   help="bar interval of --real / --csv; also sets the annualization factor (intraday: "
                        "US-equity session bars per day unless --bars-per-day says otherwise). "
                        "Ignored for the synthetic series, which is daily")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--trend-prob", type=float, default=0.45, help="synthetic: probability a regime is trending")
    p.add_argument("--trend-drift", type=float, default=0.0009, help="synthetic: daily drift inside trending regimes")
    p.add_argument("--no-matrix", action="store_true", help="skip Pardo's walk-forward matrix (slow-ish)")
    p.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--out", default="outputs")
    p.add_argument("--replay", default=None, choices=list(REPLAY_TARGETS) + ["all"],
                   help="after the run, replay the best template / static portfolio / nested portfolio from "
                        "the saved run, verify it and write its orders and trades to OUT/replay/ (replay.py)")
    return p.parse_args(argv)


def resolve_instrument(args, asset: str) -> dict:
    """The instrument the run trades, applied to `args` so that every later
    reader of the flags (the data loader, the config, the report) sees the
    same one: `--instrument-map ASSET=...` for this run's asset (a ticker or a
    CSV's stem) replaces the run-wide instrument flags, as it does for a
    dashboard asset, and an instrument with a margin is costed per unit only
    (cost_bps 0, `pipeline.instrument_of`). Returns the parsed map, for the
    run's record."""
    imap = parse_instrument_map(args.instrument_map, [asset])
    ins = instrument_of(dict(point_value=args.point_value, cost_per_unit=args.cost_per_unit,
                             margin_per_unit=args.margin_per_unit, roll_cost_per_unit=args.roll_cost_per_unit,
                             whole_units=args.whole_units, instrument_map=imap), asset)
    if ins.get("cost_bps") == 0.0 and args.cost_bps > 0:
        print(f"NOTE: --cost-bps {args.cost_bps:g} not charged: {asset} has a margin, so it is costed per unit "
              f"only (--cost-per-unit {ins['cost_per_unit']:g})")
        args.cost_bps = 0.0
    args.point_value, args.cost_per_unit = ins["point_value"], ins["cost_per_unit"]
    args.margin_per_unit, args.roll_cost_per_unit = ins["margin_per_unit"], ins["roll_cost_per_unit"]
    return imap


def main(argv=None) -> dict:
    """Run the pipeline; returns what it computed (results, portfolios,
    diagnostics) so it can be driven from a script or a test."""
    args = parse_args(argv)
    t0 = time.time()
    os.makedirs(args.out, exist_ok=True)

    synthetic = not (args.real or args.csv)
    interval = resolve_interval(args.interval, synthetic=synthetic)
    if interval != args.interval:
        print(f"NOTE: --interval {args.interval} ignored, the synthetic series is {interval} bars")
    if synthetic and args.bars_per_day is not None:
        print(f"NOTE: --bars-per-day {args.bars_per_day:g} ignored, the synthetic series is {interval} bars")
        args.bars_per_day = None
    args.interval = interval
    asset = asset_label(args)
    instrument_map = resolve_instrument(args, asset)
    try:
        cfg = eval_config(args, interval) | dict(instrument_map=instrument_map)
    except ValueError as e:
        raise SystemExit(f"--bars-per-day: {e}")
    set_periods_per_year(cfg["periods_per_year"])   # before anything here annualizes (the pool re-sets it)
    set_hedge_share(cfg["hedge_share"])

    if args.real:
        print(f"Loading {args.real} ({interval} bars) from yfinance...")
        # a margined future can genuinely trade at or below zero (WTI, April 2020): keep those bars
        df = load_real(args.real, start=args.start, interval=interval,
                       drop_nonpositive=not args.margin_per_unit > 0)
    elif args.csv:
        print(f"Loading {args.csv} ({interval} bars)...")
        df = load_csv(args.csv, interval=interval)
    else:
        print(f"Loading synthetic regime-switching data (trend_prob={args.trend_prob}, trend_drift={args.trend_drift})...")
        df = synthetic_ohlc(n_bars=args.bars, seed=args.seed, trend_prob=args.trend_prob, trend_drift=args.trend_drift)
    print(f"Data: {len(df)} bars, {df.index[0].date()} to {df.index[-1].date()}, "
          f"annualizing at {cfg['periods_per_year']} bars/year")

    overrides = {"sides": args.sides} if args.sides else {}
    templates = generate_templates(args.family, max_templates=args.max_templates, **overrides)
    print(f"Generated {len(templates)} structurally distinct strategy templates (family={args.family}"
          f"{', sides=' + '/'.join(args.sides) if args.sides else ''})")
    print(f"Walk-forward: train={args.train} test={args.test} {'anchored' if args.anchored else 'rolling'}, "
          f"selection={args.selection}, metric={args.metric}, cost={args.cost_bps}bps/side"
          + (f" + {args.cost_per_unit:g} per unit/side" if args.cost_per_unit else ""))

    # an instrument the sizing cannot handle (prices at or below zero without
    # a margin, or with bps costs) fails here, not in every window of the pool
    validate_instrument(df, _costed(templates[0], cfg, asset))
    for w in instrument_warnings(_costed(templates[0], cfg, asset)):
        print(f"WARNING: {w}")
    if args.whole_units:
        q = typical_units(df, _costed(templates[0], cfg, asset))
        if q < 1.5:
            print(f"WARNING: --whole-units: a typical entry sizes to {q:.2f} units on the {RESEARCH_EQUITY:,.0f} "
                  "the research sizes on, floored to " + ("0: most templates will never trade" if q < 1 else
                                             "1: the size barely varies") +
                  ". Raise --risk-pct / --vol-target (or --max-leverage), or drop --whole-units")
    with worker_pool(args.jobs, {asset: df}, cfg) as pool:
        out = _run(df, asset, templates, pool, args, cfg)
    print(f"\nTotal runtime {time.time() - t0:.1f}s. Outputs in {os.path.join(args.out, '')}")
    if args.replay:
        print()
        out["replay"] = replay_run(args.out, args.replay, jobs=args.jobs)
    return out


def asset_label(args) -> str:
    """What the run traded, for the report: the ticker, the CSV's stem, or 'synthetic'."""
    if args.real:
        return args.real
    if getattr(args, "csv", None):
        return os.path.splitext(os.path.basename(args.csv))[0]
    return "synthetic"


def _source_text(args) -> str:
    if args.real:
        return f"yfinance {args.real} {args.interval}"
    if getattr(args, "csv", None):
        return f"csv {args.csv} {args.interval}"
    return "synthetic"


def _run(df, asset, templates, pool, args, cfg=None) -> dict:
    # ---- 2. walk-forward + CPCV per template, in parallel ----
    t1 = time.time()
    results = evaluate_slots([(t.name, asset, t) for t in templates], pool, on_result=_progress)
    print(f"\nWalk-forward + CPCV of {len(templates)} templates took {time.time() - t1:.1f}s")

    # ---- 3. family-level overfitting diagnostics ----
    rets = returns_frame(results)
    if rets.empty:
        raise SystemExit(
            "No template produced a usable out-of-sample series. The history is probably "
            f"too short for train={args.train} + test={args.test} bars ({len(df)} bars available) "
            "-- lower --train/--test or load more data."
        )
    print("\nFamily-level diagnostics...")
    fam = family_diagnostics(results, rets, n_boot=args.n_boot)
    _print_family(fam)

    # ---- 4. portfolio: static selection, then nested walk-forward selection ----
    port, nested = build_portfolios(results, rets, args)

    # ---- finalists: bootstrap p-value, DSR, walk-forward matrix ----
    finalists = finalist_diagnostics(results, port, fam, args, pool)

    # ---- the benchmark nobody optimized: holding the asset over the same bars ----
    ins = instrument_of(cfg or eval_config(args, args.interval), asset)
    kind, bh = benchmark_returns(df, ins["point_value"], margin_per_unit=ins["margin_per_unit"])
    bench = benchmark_stats(bh, rets, port, nested, kind=kind)
    _print_benchmark(bench, args, asset)

    _report(df, results, port, nested, fam, finalists, bench, args, asset)
    # what a replay needs (replay.py): templates and per-window params, the
    # selections and weights, the bars, the series to check against
    save_run(args.out, df, results, port, nested, fam, args, cfg or eval_config(args, args.interval))
    return dict(results=results, returns=rets, family=fam, portfolio=port, nested=nested,
                finalists=finalists, benchmark=bench)


def _pneg(cp: dict) -> str:
    """P(CPCV Sharpe < 0), with the share of splits that sat flat (nothing
    traded enough in training): 0 % of nothing is no evidence of robustness."""
    txt = f"{cp['prob_sharpe_negative']:.0%}"
    flat = cp.get("frac_flat_splits", 0.0)
    return txt + (f" ({flat:.0%} flat)" if flat > 0 else "")


def _progress(i, total, name, res):
    s = res["summary"]
    cp = res["cpcv"]
    print(f"  [{i:>3}/{total}] {name:<40} oos_sharpe={s['oos_sharpe']:>6.2f} wfe={s['wfe']:>6.2f} "
          f"prof_win={s['pct_profitable_windows']:>4.0%} cpcv_mean={cp['sharpe_mean']:>6.2f} "
          f"P(cpcv<0)={_pneg(cp):>4}")


def _print_family(fam: dict) -> None:
    pbo, pbo_tpl, rc = fam["pbo_trials"], fam["pbo_templates"], fam["reality_check"]
    dsr_best, dsr_raw, n = fam["dsr_best"], fam["dsr_raw"], fam["n_templates"]
    print(f"  trials: {fam['n_trials']}  templates: {n}")
    print(f"  PBO (all param trials, CSCV S={pbo['n_combinations']} splits): {pbo['pbo']:.2f}   "
          f"OOS-vs-IS slope {pbo['degradation_slope']:.2f}   P(OOS loss | IS best) {pbo['prob_oos_loss']:.2f}")
    if pbo_tpl:
        print(f"  PBO (template OOS curves): {pbo_tpl['pbo']:.2f}")
    print(f"  White's Reality Check: best={rc['best']}  p={rc['p_value']:.3f}")
    print(f"  Effective number of independent trials (correlation clusters): {fam['n_eff']} of {n} templates")
    print(f"  Best template OOS Sharpe {dsr_best['sharpe_annual']:.2f} vs E[max of {fam['n_eff']} noise trials] "
          f"{dsr_best['sr_star_annual']:.2f} -> DSR={dsr_best['dsr']:.2f} "
          f"(PSR0={dsr_best['psr0']:.2f}; raw DSR with N={n}: {dsr_raw['dsr']:.2f})")
    print(f"  Min backtest length for {fam['n_eff']} trials at that Sharpe: {fam['min_btl_years']:.1f}y "
          f"(have {fam['years_available']:.1f}y OOS)")


def _print_benchmark(b: dict, args, asset: str | None = None) -> None:
    asset = asset or args.real or "synthetic series"
    print(f"\nBenchmark: {b.get('kind', BENCH_BUY_HOLD)} {asset} over the same OOS bars: Sharpe {b['buy_hold']['sharpe']:.2f}, "
          f"CAGR {b['buy_hold']['cagr']:.1%}, max DD {b['buy_hold']['max_dd']:.1%}; "
          f"{b['n_templates_beat_bh']} of {b['n_templates']} templates traded and have a higher OOS Sharpe")
    if b["nested"]:
        n, bhn = b["nested"], b["buy_hold_nested_period"]
        print(f"  nested walk-forward portfolio Sharpe {n['sharpe']:.2f} vs {b.get('kind', BENCH_BUY_HOLD)} {bhn['sharpe']:.2f} "
              f"over the nested period; beta {n['beta']:.2f}, corr {n['corr']:.2f}, "
              f"information ratio {n['info_ratio']:.2f}")


def finalist_diagnostics(results, port, fam, args, pool) -> dict:
    print("\nFinalist diagnostics (selected templates)...")
    selected = port["selected"]
    # Pardo's walk-forward matrix for the top finalists, all cells in the pool
    matrices = {} if args.no_matrix else walk_forward_matrices(results, selected[:3], pool)
    out = finalist_stats(results, selected, fam, n_boot=args.n_boot)
    for name, d in out.items():
        res = results[name]
        d.update(cpcv=res["cpcv"], summary=res["summary"])
        if name in matrices:
            d["wfa_matrix"] = matrices[name]
        cp = res["cpcv"]
        print(f"  {name:<40} OOS SR {res['summary']['oos_sharpe']:>5.2f}  boot p={d['bootstrap_p']:.3f}  "
              f"DSR={d['dsr']:.2f}  CPCV {cp['sharpe_mean']:.2f}+/-{cp['sharpe_std']:.2f} "
              f"P(<0)={_pneg(cp)}"
              + (f"  WFA-matrix +cells={float((d['wfa_matrix']['oos_sharpe'] > 0).mean()):.0%}" if "wfa_matrix" in d else ""))
    return out


def _safe_filename(name: str) -> str:
    """A template name as a file name that every OS accepts."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def _report(df, results, port, nested, fam, finalists, bench, args, asset: str | None = None):
    asset = asset or args.real or "the asset"
    bench_kind = bench.get("kind", BENCH_BUY_HOLD)
    out = args.out
    selected = port["selected"]
    table = port["candidate_stats"].copy()
    table["cpcv_sharpe_mean"] = [results[n]["cpcv"]["sharpe_mean"] for n in table.index]
    table["cpcv_prob_negative"] = [results[n]["cpcv"]["prob_sharpe_negative"] for n in table.index]
    table["cpcv_frac_flat"] = [results[n]["cpcv"].get("frac_flat_splits", 0.0) for n in table.index]
    table["selected"] = table.index.isin(selected)
    table = table.sort_values("oos_sharpe", ascending=False)
    table.round(4).to_csv(f"{out}/template_ranking.csv")

    # per-window log of selected templates
    win_rows = []
    for name in selected:
        for w in results[name]["windows"]:
            win_rows.append(dict(template=name, **{k: v for k, v in w.items() if k not in ("is_stats", "oos_stats")},
                                 is_sharpe=(w["is_stats"] or {}).get("sharpe"), oos_sharpe=(w["oos_stats"] or {}).get("sharpe"),
                                 oos_return=(w["oos_stats"] or {}).get("total_return")))
    if win_rows:
        pd.DataFrame(win_rows).to_csv(f"{out}/selected_windows.csv", index=False)

    # ---- plot 1: OOS equity curves ----
    fig, ax = plt.subplots(figsize=(12, 6.5))
    for name, res in results.items():
        eq = res["oos_equity"]
        if len(eq) < 3:
            continue
        norm = eq / eq.iloc[0]
        if name in selected:
            ax.plot(norm.index, norm.values, linewidth=1.6, label=name, alpha=0.9)
        else:
            ax.plot(norm.index, norm.values, linewidth=0.5, color="grey", alpha=0.2)
    if len(port["portfolio_equity"]) > 1:
        peq = port["portfolio_equity"] / port["portfolio_equity"].iloc[0]
        ax.plot(peq.index, peq.values, linewidth=3.0, color="black", label="PORTFOLIO (static selection, in-sample w.r.t. selection)")
    if len(nested["portfolio_equity"]) > 1:
        neq = nested["portfolio_equity"] / nested["portfolio_equity"].iloc[0]
        ax.plot(neq.index, neq.values, linewidth=3.0, color="red", linestyle="--", label="PORTFOLIO (nested walk-forward selection)")
    ax.set_title(f"Out-of-sample walk-forward equity: all templates (grey), selected, portfolios, {bench_kind}")
    ax.grid(alpha=0.3)
    # With a vol target the strategies are sized to the asset's scale, so buy &
    # hold shares their axis. Sized 1%/trade on the ATR stop they are on a
    # different scale and the unlevered asset gets a secondary axis; so does
    # the one-unit benchmark of a spread, whose scale is arbitrary.
    same_axis = args.vol_target > 0 and bench_kind == BENCH_BUY_HOLD
    bh_eq = benchmark_curve(bench["returns"], bench.get("additive", False))
    bh_label = (f"{bench_kind.upper()} {asset} (Sharpe {bench['buy_hold']['sharpe']:.2f}; "
                f"unlevered{'' if same_axis else ', right axis'})")
    handles, labels = [], []
    if len(bh_eq) > 1 and same_axis:
        ax.plot(bh_eq.index, bh_eq.values, linewidth=1.8, color="#444444", linestyle=":", label=bh_label)
        ax.set_ylabel(f"Growth of 1.0 (strategies sized to {args.vol_target:.0%} annualized vol)")
    else:
        ax.set_ylabel("Strategies: growth of 1.0")
        if len(bh_eq) > 1:
            ax2 = ax.twinx()
            ax2.plot(bh_eq.index, bh_eq.values, linewidth=1.8, color="#444444", linestyle=":", label=bh_label)
            ax2.set_ylabel(f"{bench_kind} {asset}: growth of 1.0 (right axis)", color="#444444")
            ax2.tick_params(axis="y", colors="#444444")
            handles, labels = ax2.get_legend_handles_labels()
    h1, l1 = ax.get_legend_handles_labels()
    ax.legend(h1 + handles, l1 + labels, fontsize=7, ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{out}/equity_curves.png", dpi=140)
    plt.close(fig)

    # ---- plot 2: correlation heatmap ----
    corr = port["corr_matrix"]
    if not corr.empty:
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=90, fontsize=5)
        ax.set_yticks(range(len(corr.index)))
        ax.set_yticklabels(corr.index, fontsize=5)
        fig.colorbar(im, ax=ax, shrink=0.8, label="correlation of daily OOS returns")
        ax.set_title("Correlation of qualifying strategies' OOS returns")
        fig.tight_layout()
        fig.savefig(f"{out}/correlation_heatmap.png", dpi=140)
        plt.close(fig)

    # ---- plot 3: ranking ----
    top = table.head(60)
    fig, ax = plt.subplots(figsize=(10, max(4, 0.22 * len(top))))
    colors = ["#1f77b4" if s else ("#8fbc8f" if p else "#bbbbbb") for s, p in zip(top["selected"], top["pardo_pass"])]
    ax.barh(top.index, top["oos_sharpe"], color=colors)
    ax.invert_yaxis()
    ax.axvline(fam["dsr_best"]["sr_star_annual"], color="red", linestyle="--", linewidth=1,
               label=f"E[max Sharpe of {fam['n_eff']} independent noise trials] = {fam['dsr_best']['sr_star_annual']:.2f}")
    ax.axvline(bench["buy_hold"]["sharpe"], color="#444444", linestyle=":", linewidth=1.5,
               label=f"{bench_kind} {asset} = {bench['buy_hold']['sharpe']:.2f}")
    ax.set_xlabel("Out-of-sample Sharpe (walk-forward)")
    ax.set_title("Templates ranked (blue = selected, green = passes Pardo WFA criteria)")
    ax.tick_params(axis="y", labelsize=6)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{out}/template_ranking.png", dpi=140)
    plt.close(fig)

    # ---- plot 4: CPCV distributions of finalists ----
    if selected:
        fig, ax = plt.subplots(figsize=(10, 5))
        data = [results[n]["cpcv"]["path_sharpes"] for n in selected]
        ax.boxplot(data, tick_labels=[n[:28] for n in selected], showmeans=True)
        for k, n in enumerate(selected, 1):
            ax.plot(k, results[n]["summary"]["oos_sharpe"], marker="*", color="red", markersize=10)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel("Sharpe per CPCV path")
        ax.set_title(f"Combinatorial purged CV: distribution of OOS Sharpe over {results[selected[0]]['cpcv']['n_paths']} paths "
                     "(red star = single walk-forward path)")
        ax.tick_params(axis="x", labelsize=6, rotation=45)
        fig.tight_layout()
        fig.savefig(f"{out}/cpcv_distribution.png", dpi=140)
        plt.close(fig)

    # ---- plot 5: PBO logits ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    pbo = fam["pbo_trials"]
    axes[0].hist(pbo["logits"][np.isfinite(pbo["logits"])], bins=40, color="#1f77b4", alpha=0.8)
    axes[0].axvline(0, color="red")
    axes[0].set_title(f"CSCV logits, all {pbo['n_trials']} trials: PBO = {pbo['pbo']:.2f}")
    axes[0].set_xlabel("logit of OOS rank of the IS-best trial (<0 = worse than median)")
    axes[1].scatter(pbo["is_sharpe"], pbo["oos_sharpe"], s=6, alpha=0.4)
    xs = np.linspace(pbo["is_sharpe"].min(), pbo["is_sharpe"].max(), 10)
    if np.isfinite(pbo["degradation_slope"]):
        axes[1].plot(xs, pbo["degradation_intercept"] + pbo["degradation_slope"] * xs, color="red",
                     label=f"slope {pbo['degradation_slope']:.2f}")
        axes[1].legend()
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xlabel("IS Sharpe of IS-best trial")
    axes[1].set_ylabel("its OOS Sharpe")
    axes[1].set_title(f"Performance degradation, P(OOS loss) = {pbo['prob_oos_loss']:.2f}")
    fig.tight_layout()
    fig.savefig(f"{out}/pbo.png", dpi=140)
    plt.close(fig)

    # ---- plot 6: walk-forward matrix of the top finalist ----
    for name, d in finalists.items():
        if "wfa_matrix" in d and len(d["wfa_matrix"]):
            m = d["wfa_matrix"]["oos_sharpe"].unstack("test_bars")
            fig, ax = plt.subplots(figsize=(6, 4.5))
            im = ax.imshow(m.values, cmap="RdYlGn", vmin=-1.5, vmax=1.5)
            ax.set_xticks(range(m.shape[1])); ax.set_xticklabels(m.columns)
            ax.set_yticks(range(m.shape[0])); ax.set_yticklabels(m.index)
            ax.set_xlabel("test bars"); ax.set_ylabel("train bars")
            for (r, c), v in np.ndenumerate(m.values):
                ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=8)
            fig.colorbar(im, ax=ax, label="OOS Sharpe")
            ax.set_title(f"Pardo walk-forward matrix: {name[:36]}", fontsize=9)
            fig.tight_layout()
            fig.savefig(f"{out}/wfa_matrix_{_safe_filename(name)}.png", dpi=140)
            plt.close(fig)
            break

    # ---- written summary ----
    L = []
    L.append("# Ranger-style strategy generator -- run report\n\n")
    L.append(f"Data: {len(df)} bars, {df.index[0].date()} to {df.index[-1].date()}"
             f" ({_source_text(args)}),"
             f" annualized at {periods_per_year()} bars/year\n\n")
    L.append(f"Walk-forward: train={args.train} test={args.test} bars, {'anchored' if args.anchored else 'rolling'}, "
             f"parameter selection = {args.selection}, objective = {args.metric}, costs = {args.cost_bps} bps/side"
             f"{' + ' + str(args.cost_per_unit) + ' per unit/side' if args.cost_per_unit else ''}"
             f"{' + ' + str(args.roll_cost_per_unit) + ' per unit per roll' if args.roll_cost_per_unit else ''}\n\n")
    L.append(f"Templates generated: {len(results)} (family '{args.family}'); parameter trials: {fam['n_trials']}\n\n")

    L.append("## Family-level overfitting diagnostics (Lopez de Prado / White)\n\n")
    L.append(f"- Probability of Backtest Overfitting, all {fam['n_trials']} parameter trials (CSCV): **{pbo['pbo']:.2f}** "
             f"(OOS-vs-IS Sharpe slope {pbo['degradation_slope']:.2f}, P(OOS loss | IS best) {pbo['prob_oos_loss']:.2f})\n")
    if fam["pbo_templates"]:
        L.append(f"- PBO of the template-selection step ({fam['n_templates']} OOS curves): **{fam['pbo_templates']['pbo']:.2f}**\n")
    rc = fam["reality_check"]
    L.append(f"- White's Reality Check for the best template ({rc['best']}): p = **{rc['p_value']:.3f}**\n")
    d = fam["dsr_best"]
    L.append(f"- Effective number of independent trials: {fam['n_eff']} correlation clusters among {fam['n_templates']} templates\n")
    L.append(f"- Best template OOS Sharpe {d['sharpe_annual']:.2f}; expected max Sharpe of {fam['n_eff']} noise trials "
             f"{d['sr_star_annual']:.2f}; Deflated Sharpe Ratio **{d['dsr']:.2f}** "
             f"(raw, treating all {fam['n_templates']} templates as independent: {fam['dsr_raw']['dsr']:.2f})\n")
    L.append(f"- Minimum backtest length for {fam['n_eff']} trials at that Sharpe: {fam['min_btl_years']:.1f} years "
             f"(available OOS: {fam['years_available']:.1f} years)\n\n")

    L.append("## Selected strategies (static selection on full OOS history)\n\n")
    L.append("| template | OOS Sharpe | WFE | prof. windows | Pardo | CPCV mean+/-sd | P(CPCV<0) | boot p | DSR | "
             "exposure | notional | net exp. | weight |\n")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    for name in selected:
        s = results[name]["summary"]; cp = results[name]["cpcv"]; f = finalists[name]
        L.append(f"| {name} | {s['oos_sharpe']:.2f} | {s['wfe']:.2f} | {s['pct_profitable_windows']:.0%} | "
                 f"{'yes' if s['pardo_pass'] else 'no'} | {cp['sharpe_mean']:.2f}+/-{cp['sharpe_std']:.2f} | "
                 f"{_pneg(cp)} | {f['bootstrap_p']:.3f} | {f['dsr']:.2f} | "
                 f"{s['oos_exposure']:.0%} | {s['oos_notional']:.2f} | {s['oos_avg_net_exposure']:+.2f} | "
                 f"{port['weights'][name]:.2f} |\n")
    L.append("\nexposure = share of OOS bars with a position; notional = mean |position notional| / equity over all "
             "OOS bars (a leverage, 0 when flat; margin instead of notional when --margin-per-unit is given); "
             "net exp. = the same signed by the side (long > 0, short < 0).\n")
    for name, f in finalists.items():
        if "wfa_matrix" in f and len(f["wfa_matrix"]):
            m = f["wfa_matrix"]
            L.append(f"\nPardo walk-forward matrix for {name}: {float((m['oos_sharpe'] > 0).mean()):.0%} of "
                     f"{len(m)} train/test settings have positive OOS Sharpe, {float(m['pardo_pass'].mean()):.0%} pass Pardo's criteria.\n")

    pr = port["portfolio_returns"]; pe = port["portfolio_equity"]
    L.append(f"\n## Portfolio ({args.weighting} weights)\n\n")
    if len(pe):
        L.append(f"- Static selection: Sharpe {annualized_sharpe(pr):.2f}, max drawdown {max_drawdown(pe):.1%}, "
                 f"final equity ${pe.iloc[-1]:,.0f} -- **biased upward**: the selection saw this whole history.\n")
    if len(nested["portfolio_equity"]):
        ne = nested["portfolio_equity"]
        L.append(f"- Nested walk-forward selection: Sharpe **{nested['sharpe']:.2f}**, max drawdown "
                 f"{max_drawdown(ne):.1%}, final equity ${ne.iloc[-1]:,.0f} "
                 f"({len(nested['selections'])} re-selections). This is the honest number.\n")
        L.append("\nRe-selection log (period start -> templates):\n\n")
        for s in nested["selections"]:
            L.append(f"- {pd.Timestamp(s['period_start']).date()}: {', '.join(s['selected']) or '(cash)'}\n")

    b = bench
    L.append(f"\n## Benchmark: {bench_kind} {asset}\n\n")
    L.append("The asset was not optimized, selected or stress-tested, so it is the one curve with no "
             "selection bias. ")
    if bench_kind != BENCH_BUY_HOLD:
        why = ("It trades at or below zero" if (df["Low"] <= 0).any()
               else "It is a margined future, whose quoted level is not the price of an investment")
        L.append(f"{why}, so it has no buy-and-hold return: the benchmark is the P&L of "
                 f"holding one unit (point value {args.point_value:g}) on the initial equity, an additive "
                 "curve (summed, not compounded). Its Sharpe, the correlation to it and the information ratio "
                 "are scale-free. The beta is NOT: the benchmark's returns are point value / initial equity "
                 "times the price change, so the beta scales with 1 / point value (and with the initial "
                 "equity); read it as the number of units the portfolio behaves like it holds on average "
                 "(a beta of 2 is two units' exposure, not twice the market's). Its CAGR (here the simple "
                 "annual P&L) and drawdown (from the peak) are fractions of the initial equity, on that "
                 "arbitrary scale. ")
    ins = instrument_text(eval_config(args, args.interval))
    if ins:
        L.append(f"Instrument: {ins}. ")
    if args.vol_target > 0:
        cap = (f"margin capped at {args.max_leverage:g}x equity" if args.margin_per_unit > 0
               else f"capped at {args.max_leverage:g}x")
        L.append(f"The templates size every entry to {args.vol_target:.0%} annualized volatility "
                 f"(realized over {args.vol_target_n} bars, {cap}), so "
                 + ("their CAGR and drawdown are on a scale comparable to the unlevered holding; "
                    if bench_kind == BENCH_BUY_HOLD else "")
                 + "time spent flat and the leverage cap keep their realized vol below the target.\n\n")
    else:
        L.append(f"Compare Sharpe: the templates risk {args.risk_pct:.1%} of equity per trade, so their CAGR "
                 "and drawdown are on a " + ("smaller scale than an unlevered holding" if bench_kind == BENCH_BUY_HOLD
                                             else "different scale from the one-unit benchmark") + ".\n\n")
    L.append("| curve | period | Sharpe | CAGR | max DD | beta to B&H | corr | info ratio |\n")
    L.append("|---|---|---|---|---|---|---|---|\n")
    bh = b["buy_hold"]
    L.append(f"| {bench_kind} | all OOS bars ({bh['n_bars']}) | {bh['sharpe']:.2f} | {bh['cagr']:.1%} | {bh['max_dd']:.1%} | 1.00 | 1.00 | - |\n")
    if b["static"]:
        s = b["static"]
        L.append(f"| static portfolio | all OOS bars | {s['sharpe']:.2f} | {s['cagr']:.1%} | {s['max_dd']:.1%} | "
                 f"{s['beta']:.2f} | {s['corr']:.2f} | {s['info_ratio']:.2f} |\n")
    if b["nested"]:
        n, bhn = b["nested"], b["buy_hold_nested_period"]
        L.append(f"| {bench_kind} | nested period ({bhn['n_bars']}) | {bhn['sharpe']:.2f} | {bhn['cagr']:.1%} | {bhn['max_dd']:.1%} | 1.00 | 1.00 | - |\n")
        L.append(f"| **nested portfolio** | nested period | **{n['sharpe']:.2f}** | {n['cagr']:.1%} | {n['max_dd']:.1%} | "
                 f"{n['beta']:.2f} | {n['corr']:.2f} | **{n['info_ratio']:.2f}** |\n")
    hold = "holding" if bench_kind == BENCH_BUY_HOLD else "holding one unit of"
    L.append(f"\n{b['n_templates_beat_bh']} of {b['n_templates']} templates traded and have a higher OOS Sharpe than {hold} {asset}.\n")
    if b["nested"]:
        n, bhn = b["nested"], b["buy_hold_nested_period"]
        if n["sharpe"] < bhn["sharpe"]:
            L.append(f"\nThe honest portfolio Sharpe ({n['sharpe']:.2f}) is BELOW {bench_kind} ({bhn['sharpe']:.2f}) "
                     "over the same bars: the whole search did not beat doing nothing. ")
        else:
            L.append(f"\nThe honest portfolio Sharpe ({n['sharpe']:.2f}) is above {bench_kind} ({bhn['sharpe']:.2f}) "
                     "over the same bars. ")
        L.append(f"Beta {n['beta']:.2f} and correlation {n['corr']:.2f} say how much of it is the asset's own drift; "
                 f"the information ratio {n['info_ratio']:.2f} is what is left once that is removed.\n")

    L.append("\n## How to read this\n\n")
    L.append(f"- {bench_kind.capitalize()}: if the nested portfolio's Sharpe is not above it, the search added nothing; "
             "a high beta with a low information ratio means the templates are a costly way to hold the asset.\n")
    L.append("- PBO near 0.5 or above: picking the best parameter set in-sample is no better than a coin toss out-of-sample.\n")
    L.append("- DSR below ~0.95: the best OOS Sharpe is not distinguishable from the best of that many random trials.\n")
    L.append("- Reality Check p above 0.05-0.15: the family's best result is consistent with data snooping.\n")
    L.append("- CPCV: a template whose path distribution straddles zero owes its single WFA path to luck.\n")
    L.append("- Pardo: WFE >= 0.5 and a majority of profitable OOS windows are the minimum to consider trading.\n")
    L.append("\nFiles: equity_curves.png, correlation_heatmap.png, template_ranking.png/.csv, cpcv_distribution.png, "
             "pbo.png, wfa_matrix_*.png, selected_windows.csv; run.json, data.csv, portfolio_returns.csv and "
             "selected_returns.csv record the run for `python replay.py --out DIR` (orders and trades of the best "
             "template and both portfolios, verified against these numbers)\n")

    with open(f"{out}/report.md", "w", encoding="utf-8") as f:
        f.writelines(L)
    print("\n" + "".join(L))


if __name__ == "__main__":
    main()
