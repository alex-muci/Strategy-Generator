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
  python main.py --family default      # ~300 templates
  python main.py --real SPY --start 2005-01-01 --family default --jobs 8
  python main.py --help
"""

from __future__ import annotations
import argparse
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import synthetic_ohlc, load_yfinance
from generator import generate_templates, param_grid_for
from walkforward import walk_forward, grid_combos
from robustness import (
    cscv_pbo, deflated_sharpe_ratio, min_backtest_length, bootstrap_sharpe_pvalue,
    reality_check, effective_n_trials, merge_block_stats, evaluate_template,
)


from portfolio import select_portfolio, walk_forward_portfolio, returns_frame
from strategy import (
    annualized_sharpe, periods_per_year, set_periods_per_year,
    periods_per_year_for_interval, BARS_PER_YEAR,
)


def _cscv_partitions(T: int) -> int:
    """CSCV blocks: 16 needs >= 100 bars per block to be meaningful."""
    return 16 if T >= 1600 else 8


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Ranger-style strategy generator with robust walk-forward evaluation")
    p.add_argument("--family", default="quick", choices=["quick", "default", "full"])
    p.add_argument("--max-templates", type=int, default=None)
    p.add_argument("--real", metavar="TICKER", default=None, help="use yfinance data for TICKER instead of synthetic")
    p.add_argument("--start", default="2005-01-01")
    p.add_argument("--interval", default="1d", choices=sorted(BARS_PER_YEAR),
                   help="bar interval for --real; also sets the annualization factor")
    p.add_argument("--bars", type=int, default=3000, help="synthetic bars")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--trend-prob", type=float, default=0.45, help="synthetic: probability a regime is trending")
    p.add_argument("--trend-drift", type=float, default=0.0009, help="synthetic: daily drift inside trending regimes")
    p.add_argument("--train", type=int, default=500, help="training window (bars)")
    p.add_argument("--test", type=int, default=125, help="test window (bars)")
    p.add_argument("--anchored", action="store_true", help="expanding instead of rolling training window")
    p.add_argument("--selection", default="plateau", choices=["plateau", "best"])
    p.add_argument("--metric", default="sharpe", choices=["sharpe", "return_over_dd", "profit_factor"])
    p.add_argument("--wide-grid", action="store_true")
    p.add_argument("--cost-bps", type=float, default=5.0, help="commission+slippage per side, bps of notional")
    p.add_argument("--min-sharpe", type=float, default=0.3)
    p.add_argument("--max-strategies", type=int, default=8)
    p.add_argument("--corr-ceiling", type=float, default=0.6)
    p.add_argument("--require-pardo", action="store_true", help="only candidates passing Pardo's WFA criteria")
    p.add_argument("--select-method", default="greedy", choices=["greedy", "cluster"])
    p.add_argument("--weighting", default="equal", choices=["equal", "hrp"])
    p.add_argument("--cpcv-groups", type=int, default=8)
    p.add_argument("--cpcv-k", type=int, default=2)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--no-matrix", action="store_true", help="skip Pardo's walk-forward matrix (slow-ish)")
    p.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--out", default="outputs")
    return p.parse_args(argv)


# --------------------------------------------------------------------------
# per-template worker (runs in a process pool)
# --------------------------------------------------------------------------
_DF = None
_ARGS = None


def _init_worker(df, args):
    global _DF, _ARGS
    _DF, _ARGS = df, args
    # a fresh process re-imports strategy at the daily default; without this every
    # annualized number computed in the pool would be wrong for intraday bars
    set_periods_per_year(periods_per_year_for_interval(args.interval))


def _wfa_cell(job):
    """One cell of Pardo's walk-forward matrix (runs in the pool)."""
    tpl, tr, te = job
    a = _ARGS
    grid = param_grid_for(tpl, wide=a.wide_grid)
    s = walk_forward(_DF, tpl, grid, train_bars=tr, test_bars=te, anchored=a.anchored,
                     metric=a.metric, selection=a.selection)["summary"]
    return dict(template=tpl.name, train_bars=tr, test_bars=te, oos_sharpe=s["oos_sharpe"], wfe=s["wfe"],
                pct_profitable=s["pct_profitable_windows"], n_windows=s["n_windows"], pardo_pass=s["pardo_pass"])


def _evaluate_template(tpl):
    """Walk-forward + CPCV for one template. Only the CSCV block statistics
    travel back to the parent process, never the T x N trials matrix."""
    df, a = _DF, _ARGS
    tpl = tpl.with_params(cost_bps=a.cost_bps)
    grid = param_grid_for(tpl, wide=a.wide_grid)
    wfa = evaluate_template(
        df, tpl, grid, train_bars=a.train, test_bars=a.test, anchored=a.anchored,
        metric=a.metric, selection=a.selection, cpcv_groups=a.cpcv_groups,
        cpcv_k=a.cpcv_k, cscv_partitions_n=_cscv_partitions(len(df)),
    )
    return tpl.name, wfa


def main():
    args = parse_args()
    t0 = time.time()
    os.makedirs(args.out, exist_ok=True)

    set_periods_per_year(periods_per_year_for_interval(args.interval))

    if args.real:
        print(f"Loading {args.real} ({args.interval} bars) from yfinance...")
        df = load_yfinance(args.real, start=args.start, interval=args.interval)
    else:
        print(f"Loading synthetic regime-switching data (trend_prob={args.trend_prob}, trend_drift={args.trend_drift})...")
        df = synthetic_ohlc(n_bars=args.bars, seed=args.seed, trend_prob=args.trend_prob, trend_drift=args.trend_drift)
    print(f"Data: {len(df)} bars, {df.index[0].date()} to {df.index[-1].date()}, "
          f"annualizing at {periods_per_year()} bars/year")

    templates = generate_templates(args.family, max_templates=args.max_templates)
    print(f"Generated {len(templates)} structurally distinct strategy templates (family={args.family})")
    print(f"Walk-forward: train={args.train} test={args.test} {'anchored' if args.anchored else 'rolling'}, "
          f"selection={args.selection}, metric={args.metric}, cost={args.cost_bps}bps/side")

    pool = Pool(args.jobs, initializer=_init_worker, initargs=(df, args)) if args.jobs > 1 else None
    _init_worker(df, args)
    try:
        _run(df, templates, pool, args, t0)
    finally:
        if pool is not None:
            pool.close()
            pool.join()


def _run(df, templates, pool, args, t0):
    # ---- 2. walk-forward + CPCV per template, in parallel ----
    results = {}
    mapper = pool.imap_unordered if pool is not None else map
    for i, (name, res) in enumerate(mapper(_evaluate_template, templates), 1):
        results[name] = res
        _progress(i, len(templates), name, res)
    results = {t.name: results[t.name] for t in templates}
    print(f"\nWalk-forward + CPCV of {len(templates)} templates took {time.time() - t0:.1f}s")

    # ---- 3. family-level overfitting diagnostics ----
    rets = returns_frame(results)
    if rets.empty:
        raise SystemExit(
            "No template produced a usable out-of-sample series. The history is probably "
            f"too short for train={args.train} + test={args.test} bars ({len(df)} bars available) "
            "-- lower --train/--test or load more data."
        )
    fam = family_diagnostics(results, rets, args)

    # ---- 4. portfolio: static selection, then nested walk-forward selection ----
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

    # ---- finalists: bootstrap p-value, DSR, walk-forward matrix ----
    finalists = finalist_diagnostics(results, port, fam, args, pool)

    _report(df, results, port, nested, fam, finalists, args)
    print(f"\nTotal runtime {time.time() - t0:.1f}s. Outputs in ./{args.out}/")


def _progress(i, total, name, res):
    s = res["summary"]
    cp = res["cpcv"]
    print(f"  [{i:>3}/{total}] {name:<40} oos_sharpe={s['oos_sharpe']:>6.2f} wfe={s['wfe']:>6.2f} "
          f"prof_win={s['pct_profitable_windows']:>4.0%} cpcv_mean={cp['sharpe_mean']:>6.2f} "
          f"P(cpcv<0)={cp['prob_sharpe_negative']:>4.0%}")


def family_diagnostics(results: dict, rets: pd.DataFrame, args) -> dict:
    """Overfitting diagnostics for the WHOLE family of trials."""
    print("\nFamily-level diagnostics...")
    # PBO over every (template, param combo) trial the generator tried
    n_trials = sum(r["n_trials"] for r in results.values())
    pbo = cscv_pbo(blocks=merge_block_stats([r["trial_blocks"] for r in results.values()]))
    # PBO over the template-level OOS curves (the selection step's trials)
    pbo_tpl = cscv_pbo(rets.to_numpy(), n_partitions=_cscv_partitions(len(rets))) if rets.shape[1] > 1 else None
    rc = reality_check(rets, n_boot=args.n_boot)
    oos_sharpes = rets.apply(annualized_sharpe)
    best = oos_sharpes.idxmax()
    # raw DSR: every template is an independent trial (very conservative)
    var_sr_raw = float(oos_sharpes.var()) / periods_per_year()
    dsr_raw = deflated_sharpe_ratio(rets[best], n_trials=rets.shape[1], var_sr_trials=var_sr_raw)
    # effective DSR: correlated templates collapsed into clusters
    eff = effective_n_trials(rets)
    dsr_best = deflated_sharpe_ratio(rets[best], n_trials=eff["n_eff"], var_sr_trials=eff["var_sr_period"])
    out = dict(
        n_templates=rets.shape[1], n_trials=n_trials, pbo_trials=pbo, pbo_templates=pbo_tpl,
        reality_check=rc, oos_sharpes=oos_sharpes, n_eff=eff["n_eff"], var_sr_period=eff["var_sr_period"],
        best_template=best, dsr_best=dsr_best, dsr_raw=dsr_raw,
        min_btl_years=min_backtest_length(eff["n_eff"], max(float(oos_sharpes.max()), 1e-6)),
        years_available=len(rets) / periods_per_year(),
    )
    print(f"  trials: {n_trials}  templates: {rets.shape[1]}")
    print(f"  PBO (all param trials, CSCV S={pbo['n_combinations']} splits): {pbo['pbo']:.2f}   "
          f"OOS-vs-IS slope {pbo['degradation_slope']:.2f}   P(OOS loss | IS best) {pbo['prob_oos_loss']:.2f}")
    if pbo_tpl:
        print(f"  PBO (template OOS curves): {pbo_tpl['pbo']:.2f}")
    print(f"  White's Reality Check: best={rc['best']}  p={rc['p_value']:.3f}")
    print(f"  Effective number of independent trials (correlation clusters): {eff['n_eff']} of {rets.shape[1]} templates")
    print(f"  Best template OOS Sharpe {dsr_best['sharpe_annual']:.2f} vs E[max of {eff['n_eff']} noise trials] "
          f"{dsr_best['sr_star_annual']:.2f} -> DSR={dsr_best['dsr']:.2f} "
          f"(PSR0={dsr_best['psr0']:.2f}; raw DSR with N={rets.shape[1]}: {dsr_raw['dsr']:.2f})")
    print(f"  Min backtest length for {eff['n_eff']} trials at that Sharpe: {out['min_btl_years']:.1f}y "
          f"(have {out['years_available']:.1f}y OOS)")
    return out


def finalist_diagnostics(results, port, fam, args, pool) -> dict:
    print("\nFinalist diagnostics (selected templates)...")
    out = {}
    # Pardo's walk-forward matrix for the top finalists, all cells in the pool
    matrices = {}
    if not args.no_matrix and port["selected"]:
        jobs = [(results[n]["template"], tr, te) for n in port["selected"][:3]
                for tr in (250, 375, 500, 750) for te in (63, 125, 250) if tr + te < len(_DF)]
        cells = list(pool.imap_unordered(_wfa_cell, jobs) if pool is not None else map(_wfa_cell, jobs))
        cells = pd.DataFrame(cells)
        for n, sub in cells.groupby("template"):
            matrices[n] = sub.drop(columns="template").set_index(["train_bars", "test_bars"]).sort_index()

    for name in port["selected"]:
        res = results[name]
        boot = bootstrap_sharpe_pvalue(res["oos_returns"], n_boot=args.n_boot)
        dsr = deflated_sharpe_ratio(res["oos_returns"], n_trials=fam["n_eff"], var_sr_trials=fam["var_sr_period"])
        d = dict(bootstrap_p=boot["p_value"], dsr=dsr["dsr"], psr0=dsr["psr0"], cpcv=res["cpcv"], summary=res["summary"])
        if name in matrices:
            d["wfa_matrix"] = matrices[name]
        out[name] = d
        cp = res["cpcv"]
        print(f"  {name:<40} OOS SR {res['summary']['oos_sharpe']:>5.2f}  boot p={boot['p_value']:.3f}  "
              f"DSR={dsr['dsr']:.2f}  CPCV {cp['sharpe_mean']:.2f}+/-{cp['sharpe_std']:.2f} "
              f"P(<0)={cp['prob_sharpe_negative']:.0%}"
              + (f"  WFA-matrix +cells={float((d['wfa_matrix']['oos_sharpe'] > 0).mean()):.0%}" if "wfa_matrix" in d else ""))
    return out


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def _report(df, results, port, nested, fam, finalists, args):
    out = args.out
    selected = port["selected"]
    table = port["candidate_stats"].copy()
    table["cpcv_sharpe_mean"] = [results[n]["cpcv"]["sharpe_mean"] for n in table.index]
    table["cpcv_prob_negative"] = [results[n]["cpcv"]["prob_sharpe_negative"] for n in table.index]
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
    ax.set_title("Out-of-sample walk-forward equity: all templates (grey), selected, and portfolios")
    ax.set_ylabel("Growth of 1.0")
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    ax.grid(alpha=0.3)
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
            fig.savefig(f"{out}/wfa_matrix_{name.replace(':', '_')}.png", dpi=140)
            plt.close(fig)
            break

    # ---- written summary ----
    L = []
    L.append("# Ranger-style strategy generator -- run report\n\n")
    L.append(f"Data: {len(df)} bars, {df.index[0].date()} to {df.index[-1].date()}"
             f" ({'yfinance ' + args.real + ' ' + args.interval if args.real else 'synthetic'}),"
             f" annualized at {periods_per_year()} bars/year\n\n")
    L.append(f"Walk-forward: train={args.train} test={args.test} bars, {'anchored' if args.anchored else 'rolling'}, "
             f"parameter selection = {args.selection}, objective = {args.metric}, costs = {args.cost_bps} bps/side\n\n")
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
    L.append("| template | OOS Sharpe | WFE | prof. windows | Pardo | CPCV mean+/-sd | P(CPCV<0) | boot p | DSR | weight |\n")
    L.append("|---|---|---|---|---|---|---|---|---|---|\n")
    for name in selected:
        s = results[name]["summary"]; cp = results[name]["cpcv"]; f = finalists[name]
        L.append(f"| {name} | {s['oos_sharpe']:.2f} | {s['wfe']:.2f} | {s['pct_profitable_windows']:.0%} | "
                 f"{'yes' if s['pardo_pass'] else 'no'} | {cp['sharpe_mean']:.2f}+/-{cp['sharpe_std']:.2f} | "
                 f"{cp['prob_sharpe_negative']:.0%} | {f['bootstrap_p']:.3f} | {f['dsr']:.2f} | {port['weights'][name]:.2f} |\n")
    for name, f in finalists.items():
        if "wfa_matrix" in f and len(f["wfa_matrix"]):
            m = f["wfa_matrix"]
            L.append(f"\nPardo walk-forward matrix for {name}: {float((m['oos_sharpe'] > 0).mean()):.0%} of "
                     f"{len(m)} train/test settings have positive OOS Sharpe, {float(m['pardo_pass'].mean()):.0%} pass Pardo's criteria.\n")

    pr = port["portfolio_returns"]; pe = port["portfolio_equity"]
    L.append(f"\n## Portfolio ({args.weighting} weights)\n\n")
    if len(pe):
        L.append(f"- Static selection: Sharpe {annualized_sharpe(pr):.2f}, max drawdown {float((pe / pe.cummax() - 1).min()):.1%}, "
                 f"final equity ${pe.iloc[-1]:,.0f} -- **biased upward**: the selection saw this whole history.\n")
    if len(nested["portfolio_equity"]):
        ne = nested["portfolio_equity"]
        L.append(f"- Nested walk-forward selection: Sharpe **{nested['sharpe']:.2f}**, max drawdown "
                 f"{float((ne / ne.cummax() - 1).min()):.1%}, final equity ${ne.iloc[-1]:,.0f} "
                 f"({len(nested['selections'])} re-selections). This is the honest number.\n")
        L.append("\nRe-selection log (period start -> templates):\n\n")
        for s in nested["selections"]:
            L.append(f"- {pd.Timestamp(s['period_start']).date()}: {', '.join(s['selected']) or '(cash)'}\n")

    L.append("\n## How to read this\n\n")
    L.append("- PBO near 0.5 or above: picking the best parameter set in-sample is no better than a coin toss out-of-sample.\n")
    L.append("- DSR below ~0.95: the best OOS Sharpe is not distinguishable from the best of that many random trials.\n")
    L.append("- Reality Check p above 0.05-0.15: the family's best result is consistent with data snooping.\n")
    L.append("- CPCV: a template whose path distribution straddles zero owes its single WFA path to luck.\n")
    L.append("- Pardo: WFE >= 0.5 and a majority of profitable OOS windows are the minimum to consider trading.\n")
    L.append("\nFiles: equity_curves.png, correlation_heatmap.png, template_ranking.png/.csv, cpcv_distribution.png, "
             "pbo.png, wfa_matrix_*.png, selected_windows.csv\n")

    with open(f"{out}/report.md", "w") as f:
        f.writelines(L)
    print("\n" + "".join(L))


if __name__ == "__main__":
    main()
