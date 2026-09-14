"""
main.py
-------
End-to-end demo of the Ranger-style pipeline:

  1. GENERATE   many structurally distinct breakout strategy templates
                (generator.py)
  2. EVALUATE   each template with walk-forward analysis: optimize
                in-sample, roll forward, test out-of-sample only
                (walkforward.py)
  3. SELECT     an uncorrelated subset whose combined equity curve is
                smoother than any single strategy (portfolio.py)
  4. REPORT     equity curves, correlation heatmap, and a written summary

By default this runs on synthetic regime-switching data so it works
with no internet connection. Swap `USE_REAL_DATA = True` and set a
ticker to run it on real market data via yfinance.
"""

from __future__ import annotations
import json
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import synthetic_ohlc, load_yfinance
from generator import generate_templates, param_grid_for
from walkforward import walk_forward
from portfolio import select_portfolio

OUT_DIR = "outputs"

USE_REAL_DATA = False
TICKER = "SPY"
START_DATE = "2016-01-01"

TRAIN_BARS = 500
TEST_BARS = 150
MIN_SHARPE = 0.15
MAX_STRATEGIES = 8
CORR_CEILING = 0.6


def main():
    t0 = time.time()

    if USE_REAL_DATA:
        print(f"Loading {TICKER} from yfinance...")
        df = load_yfinance(TICKER, start=START_DATE)
    else:
        print("Loading synthetic regime-switching data (no internet needed)...")
        df = synthetic_ohlc(n_bars=2500, seed=7)

    print(f"Data: {len(df)} bars, {df.index[0].date()} to {df.index[-1].date()}")

    templates = generate_templates()
    print(f"Generated {len(templates)} structurally distinct strategy templates")

    results = {}
    for i, tpl in enumerate(templates, 1):
        grid = param_grid_for(tpl)
        res = walk_forward(
            df, tpl, grid, train_bars=TRAIN_BARS, test_bars=TEST_BARS
        )
        results[tpl.name] = res
        final_eq = res["oos_equity"].iloc[-1]
        n_win = len(res["windows"])
        print(f"  [{i:>2}/{len(templates)}] {tpl.name:<32} windows={n_win:<3} final_oos_equity={final_eq:,.0f}")

    print(f"\nWalk-forward evaluation of all templates took {time.time()-t0:.1f}s")

    portfolio_result = select_portfolio(
        results,
        min_sharpe=MIN_SHARPE,
        max_strategies=MAX_STRATEGIES,
        corr_ceiling=CORR_CEILING,
    )

    _report(results, portfolio_result, df)


def _annualized_sharpe(rets: pd.Series) -> float:
    if rets.std() == 0 or rets.empty:
        return 0.0
    return rets.mean() / rets.std() * np.sqrt(252)


def _report(results: dict, portfolio_result: dict, df: pd.DataFrame):
    import os

    os.makedirs(OUT_DIR, exist_ok=True)

    selected = portfolio_result["selected"]
    candidate_stats = portfolio_result["candidate_stats"]

    # ---- ranking table ----
    rows = []
    for name, res in results.items():
        stats = candidate_stats.get(name, {})
        tpl = res["template"]
        rows.append(
            dict(
                template=name,
                direction=tpl.direction_logic,
                entry=tpl.entry_style,
                exit=tpl.exit_style,
                regime_filter=tpl.regime_filter,
                vol_filter=tpl.vol_filter,
                oos_sharpe=round(stats.get("sharpe", 0.0), 3),
                n_windows=stats.get("n_windows", 0),
                selected=name in selected,
            )
        )
    table = pd.DataFrame(rows).sort_values("oos_sharpe", ascending=False)
    table.to_csv(f"{OUT_DIR}/template_ranking.csv", index=False)

    # ---- plot 1: all qualifying equity curves + selected + portfolio ----
    fig, ax = plt.subplots(figsize=(11, 6))
    for name, res in results.items():
        eq = res["oos_equity"]
        if len(eq) < 3:
            continue
        norm = eq / eq.iloc[0]
        if name in selected:
            ax.plot(norm.index, norm.values, linewidth=2.0, label=name, alpha=0.9)
        else:
            ax.plot(norm.index, norm.values, linewidth=0.6, color="grey", alpha=0.25)

    if len(portfolio_result["portfolio_equity"]) > 1:
        peq = portfolio_result["portfolio_equity"]
        peq_norm = peq / peq.iloc[0]
        ax.plot(peq_norm.index, peq_norm.values, linewidth=3.0, color="black", label="PORTFOLIO (equal-weight)")

    ax.set_title("Out-of-sample walk-forward equity: all templates (grey) vs selected portfolio")
    ax.set_ylabel("Growth of 1.0")
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/equity_curves.png", dpi=140)
    plt.close(fig)

    # ---- plot 2: correlation heatmap of qualifying candidates ----
    corr = portfolio_result["corr_matrix"]
    if not corr.empty:
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=90, fontsize=6)
        ax.set_yticks(range(len(corr.index)))
        ax.set_yticklabels(corr.index, fontsize=6)
        fig.colorbar(im, ax=ax, shrink=0.8, label="correlation of daily OOS returns")
        ax.set_title("Correlation of qualifying strategies' OOS returns")
        fig.tight_layout()
        fig.savefig(f"{OUT_DIR}/correlation_heatmap.png", dpi=140)
        plt.close(fig)

    # ---- plot 3: ranked Sharpe bar chart ----
    fig, ax = plt.subplots(figsize=(10, max(4, 0.22 * len(table))))
    colors = ["#1f77b4" if s else "#bbbbbb" for s in table["selected"]]
    ax.barh(table["template"], table["oos_sharpe"], color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("Out-of-sample Sharpe (walk-forward)")
    ax.set_title("All generated templates, ranked (blue = selected into portfolio)")
    ax.tick_params(axis="y", labelsize=6)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/template_ranking.png", dpi=140)
    plt.close(fig)

    # ---- written summary ----
    port_rets = portfolio_result["portfolio_returns"]
    port_sharpe = _annualized_sharpe(port_rets)
    port_eq = portfolio_result["portfolio_equity"]
    port_dd = (port_eq / port_eq.cummax() - 1).min() if len(port_eq) else 0.0

    lines = []
    lines.append("# Ranger-style strategy generator -- run report\n")
    lines.append(f"Data: {len(df)} bars, {df.index[0].date()} to {df.index[-1].date()}\n")
    lines.append(f"Templates generated: {len(results)}\n")
    lines.append(f"Templates qualifying (min Sharpe {MIN_SHARPE}, min windows 3): {len(candidate_stats)}\n")
    lines.append(f"Templates selected into final portfolio: {len(selected)}\n\n")
    lines.append("## Selected strategies\n")
    for name in selected:
        s = candidate_stats[name]
        lines.append(f"- **{name}** -- OOS Sharpe {s['sharpe']:.2f} across {s['n_windows']} walk-forward windows\n")
    lines.append(f"\n## Combined portfolio (equal-weight of selected strategies)\n")
    lines.append(f"- Annualized Sharpe: {port_sharpe:.2f}\n")
    lines.append(f"- Max drawdown: {port_dd:.1%}\n")
    lines.append(f"- Final equity (from $100,000): ${port_eq.iloc[-1]:,.0f}\n" if len(port_eq) else "")
    lines.append("\nSee equity_curves.png, correlation_heatmap.png, template_ranking.png and template_ranking.csv for details.\n")

    with open(f"{OUT_DIR}/report.md", "w") as f:
        f.writelines(lines)

    print("\n".join(l.rstrip() for l in lines))


if __name__ == "__main__":
    main()
