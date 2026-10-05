"""
run_study.py
------------
Walk-forward performance AND turnover of template families under a cost sweep.

Evaluation matches the pipeline (main.py): train 500 / test 125 bars, rolling,
metric sharpe, plateau selection, risk_pct 1 %, max_leverage 2, the grid of
`generator.param_grid_for`. Each (template, sides, cost) is one walk-forward,
run in a process pool.

Cost sweep
  * shares / ETFs: `cost_bps` per side in `cost_bps_list` (default 0..20).
  * a futures spread (`instrument` given: point_value, cost_per_unit,
    roll_cost_per_unit, margin_per_unit): cost_bps is 0 and the per-unit costs
    are scaled by `cost_mult` in (0, 0.5, 1, 2, 4); row column `cost_mult`.

Turnover (annual traded notional / equity, one-way: a round trip counts 2)
  * `turnover`: the cost drag, (ann_return(0) - ann_return(c)) / (c / 1e4) at
    the largest swept cost. With a per-unit sweep it is `cost_drag_per_mult`,
    the annual return lost per unit of cost_mult.
  * `turnover_direct`: replay the chosen params of each OOS window
    (`replay_turnover`) and sum, over its trades, shares x point_value x
    (|entry price| + |exit price|) / equity at the entry and exit bars; a
    position still open at the window end is entered but not exited (as in
    the engine, no exit cost). Annualised as sum / n_oos_bars x periods per
    year. The learner scores net of cost_bps and sizing compounds with equity,
    so the two agree only roughly.
  * `breakeven_bps` (`breakeven_mult` for per-unit): the cost at which
    ann_return crosses 0, linear interpolation; inf if it never does, 0 if
    already <= 0 at zero cost.

CLI:  python -m extra_utils.online_study.run_study --csv data_dump/spy.csv \
        --families online online_forecast quick --out <dir>
"""

from __future__ import annotations
import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
             "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_var, "1")   # before numpy/numba import; forked/spawned workers inherit

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import generator
import pipeline
from data import load_csv
from strategy import (annualized_sharpe, compound, max_drawdown, periods_per_year, set_periods_per_year)
from walkforward import position_units, walk_forward, window_backtest

COST_BPS = (0, 2, 5, 10, 20)
COST_MULTS = (0, 0.5, 1, 2, 4)
RISK_DEFAULTS = dict(risk_pct=0.01, max_leverage=2.0)   # main.py defaults
INSTRUMENT_KEYS = ("point_value", "cost_per_unit", "roll_cost_per_unit", "margin_per_unit")


# --------------------------------------------------------------------------
# direct turnover
# --------------------------------------------------------------------------

def replay_turnover(df: pd.DataFrame, res: dict) -> float:
    """Annualised one-way traded notional / equity of a `walk_forward` result,
    from the OOS trades (see the module docstring for the convention). Each
    window is replayed with the params it chose, on the equity it was sized on."""
    tpl = res["template"]
    traded, bars = 0.0, 0
    for w in res["windows"]:
        if w["skipped"]:
            bars += df.index.get_loc(w["test_end"]) - df.index.get_loc(w["test_start"]) + 1
            continue
        s, e = df.index.get_loc(w["test_start"]), df.index.get_loc(w["test_end"]) + 1
        bars += e - s
        r = window_backtest(df, tpl.with_params(**w["params"]), s, e, initial_equity=w["initial_equity"])
        u = position_units(r)
        # change of position at each bar, valued at that bar's close (stance
        # templates resize in place: a trade closes and the next opens at once,
        # only the DIFFERENCE is traded and costed); nothing is held before the window
        d = np.abs(np.diff(u, prepend=0.0))
        eq = r["equity"].to_numpy()
        traded += float((d * np.abs(df["Close"].to_numpy()[s:e]) * tpl.point_value / eq).sum())
    return float(traded / bars * periods_per_year()) if bars else 0.0


# --------------------------------------------------------------------------
# one job
# --------------------------------------------------------------------------

def _init_worker(ppy: int) -> None:
    set_periods_per_year(ppy)


def _costed(tpl, sides, cost_bps, mult, instrument):
    kw = dict(RISK_DEFAULTS, sides=sides, cost_bps=float(cost_bps))
    if instrument:
        kw.update(instrument)
        kw["cost_bps"] = 0.0
        kw["cost_per_unit"] = float(instrument.get("cost_per_unit", 0.0)) * mult
        kw["roll_cost_per_unit"] = float(instrument.get("roll_cost_per_unit", 0.0)) * mult
    return tpl.with_params(**kw)


def _job(args) -> dict:
    df, tpl, family, sides, cost, mult, instrument, train, test = args
    tpl = _costed(tpl, sides, cost, mult, instrument)
    res = walk_forward(df, tpl, generator.param_grid_for(tpl), train_bars=train, test_bars=test)
    r = res["oos_returns"]
    ppy = periods_per_year()
    n = len(r)
    h = n // 2
    live = [w for w in res["windows"] if not w["skipped"]]
    gross = sum(w["oos_stats"]["gross_notional"] for w in live)
    eq = compound(r.to_numpy()) if n else np.array([])
    return dict(
        template=tpl.name, family=family, sides=sides, cost_bps=float(cost) if not instrument else 0.0,
        cost_mult=float(mult) if instrument else np.nan,
        oos_sharpe=annualized_sharpe(r) if n > 1 else 0.0,
        ann_return=float(r.mean() * ppy) if n else 0.0,
        ann_vol=float(r.std(ddof=1) * np.sqrt(ppy)) if n > 1 else 0.0,
        max_dd=max_drawdown(eq) if n else 0.0,
        n_trades=int(sum(w["oos_stats"]["n_trades"] for w in live)),
        avg_gross_exposure=float(gross / n) if n else 0.0,
        sharpe_h1=annualized_sharpe(r.iloc[:h]) if h > 1 else 0.0,
        sharpe_h2=annualized_sharpe(r.iloc[h:]) if n - h > 1 else 0.0,
        n_oos_bars=n,
        turnover_direct=replay_turnover(df, res) if live else 0.0,
    )


# --------------------------------------------------------------------------
# derived columns
# --------------------------------------------------------------------------

def breakeven(xs, ys) -> float:
    """Cost at which the (xs, ys) curve first crosses 0, linearly interpolated;
    0 if ys[0] <= 0, inf if it never gets to <= 0."""
    if ys[0] <= 0:
        return 0.0
    for i in range(1, len(xs)):
        if ys[i] <= 0:
            return float(xs[i - 1] + (xs[i] - xs[i - 1]) * ys[i - 1] / (ys[i - 1] - ys[i]))
    return float("inf")


def add_turnover(rows: pd.DataFrame) -> pd.DataFrame:
    """Add the cost-drag columns per (dataset, template, sides); see the
    module docstring. Rows are kept; the columns repeat over the cost sweep."""
    out = []
    for _, g in rows.groupby(["dataset", "template", "sides"], sort=False):
        g = g.copy()
        per_unit = g["cost_mult"].notna().any()
        key = "cost_mult" if per_unit else "cost_bps"
        g = g.sort_values(key)
        x, y = g[key].to_numpy(float), g["ann_return"].to_numpy(float)
        drag = y[0] - y
        if per_unit:
            g["cost_drag_per_mult"] = drag[-1] / x[-1] if x[-1] > 0 else np.nan
            g["breakeven_mult"] = breakeven(x, y)
        else:
            g["turnover"] = drag[-1] / (x[-1] / 1e4) if x[-1] > 0 else np.nan
            g["cost_drag_5bp"] = (y[0] - np.interp(5.0, x, y)) if x[0] <= 0 <= 5 <= x[-1] else np.nan
            g["breakeven_bps"] = breakeven(x, y)
        out.append(g)
    return pd.concat(out, ignore_index=True)


def buy_hold_row(df: pd.DataFrame, dataset: str, train: int = 500, instrument=None) -> dict:
    """Buy-and-hold (pipeline.benchmark_returns) over the OOS bars: from `train` on."""
    inst = instrument or {}
    _, r = pipeline.benchmark_returns(df, point_value=inst.get("point_value", 1.0),
                                      margin_per_unit=inst.get("margin_per_unit", 0.0))
    r = r.iloc[train:].fillna(0.0)
    n, h, ppy = len(r), len(r) // 2, periods_per_year()
    return dict(dataset=dataset, template="buy_and_hold", family="benchmark", sides="long_only",
                cost_bps=0.0, cost_mult=np.nan, oos_sharpe=annualized_sharpe(r),
                ann_return=float(r.mean() * ppy), ann_vol=float(r.std(ddof=1) * np.sqrt(ppy)),
                max_dd=max_drawdown(compound(r.to_numpy())), n_trades=1, avg_gross_exposure=1.0,
                sharpe_h1=annualized_sharpe(r.iloc[:h]), sharpe_h2=annualized_sharpe(r.iloc[h:]), n_oos_bars=n)


# --------------------------------------------------------------------------
# the study
# --------------------------------------------------------------------------

def evaluate(df, templates, *, cost_bps_list=COST_BPS, sides=("both", "long_only"), train=500, test=125,
             instrument=None, jobs=4, dataset="", family="custom", pool=None) -> pd.DataFrame:
    """Rows of the study for one dataset (see the module docstring).

    `templates` is a list of StrategyTemplate (family label `family`) or a dict
    {family: list}. `instrument` (dict of point_value, cost_per_unit,
    roll_cost_per_unit, margin_per_unit) switches to the per-unit sweep over
    COST_MULTS. `pool` reuses an existing ProcessPoolExecutor."""
    fams = templates if isinstance(templates, dict) else {family: list(templates)}
    sweep = [(0.0, m) for m in COST_MULTS] if instrument else [(float(c), np.nan) for c in cost_bps_list]
    work = [(df, t, f, sd, c, m, instrument, train, test)
            for f, ts in fams.items() for t in ts for sd in sides for c, m in sweep]
    if pool is not None:
        res = list(pool.map(_job, work, chunksize=1))
    elif jobs <= 1:
        res = [_job(w) for w in work]
    else:
        with ProcessPoolExecutor(jobs, initializer=_init_worker, initargs=(periods_per_year(),)) as ex:
            res = list(ex.map(_job, work, chunksize=1))
    rows = pd.DataFrame(res)
    rows.insert(0, "dataset", dataset)
    rows = add_turnover(rows)
    bh = pd.DataFrame([buy_hold_row(df, dataset, train, instrument)])
    return pd.concat([rows, bh], ignore_index=True)


def _md(t: pd.DataFrame) -> str:
    """Markdown pipe table of a frame (no `tabulate` dependency), index included."""
    t = t.round(2).reset_index()
    t.columns = [" / ".join(map(str, c)) if isinstance(c, tuple) else str(c) for c in t.columns]
    rows = [[f"{v:g}" if isinstance(v, float) else str(v) for v in rw] for rw in t.itertuples(index=False)]
    head = ["| " + " | ".join(t.columns) + " |", "|" + "---|" * len(t.columns)]
    return "\n".join(head + ["| " + " | ".join(r) + " |" for r in rows])


def summarize(rows: pd.DataFrame) -> str:
    """Markdown: median OOS Sharpe per family x sides x cost, and per template
    (median over datasets) with turnover and breakeven."""
    r = rows[rows["family"] != "benchmark"]
    key = "cost_mult" if r["cost_mult"].notna().any() else "cost_bps"
    L = ["# Online study summary", "", f"Datasets: {', '.join(sorted(rows['dataset'].unique()))}", ""]
    bh = rows[rows["family"] == "benchmark"].set_index("dataset")["oos_sharpe"]
    L += ["Buy and hold OOS Sharpe: " + ", ".join(f"{d} {v:.2f}" for d, v in bh.items()), ""]
    piv = r.pivot_table(index=["family", "sides"], columns=key, values="oos_sharpe", aggfunc="median")
    L += [f"## Median OOS Sharpe by family x sides x {key}", "", _md(piv), ""]
    at = r[r[key] == (1.0 if key == "cost_mult" else 5.0)]
    if len(at):
        d = at.pivot_table(index=["family", "template", "sides"], columns="dataset", values="oos_sharpe")
        cols = ["turnover_direct"] + (["cost_drag_per_mult", "breakeven_mult"] if key == "cost_mult"
                                      else ["turnover", "cost_drag_5bp", "breakeven_bps"])
        extra = at.groupby(["family", "template", "sides"])[cols].median()
        t = d.join(extra)
        if "cost_drag_5bp" in t:
            t = t.rename(columns={"cost_drag_5bp": "cost_drag_5bp_%"}).assign(**{"cost_drag_5bp_%": t["cost_drag_5bp"] * 100})
        t.insert(0, "median_sharpe", d.median(axis=1))
        L += [f"## Per template at {key} = {'1' if key == 'cost_mult' else '5'} "
              "(Sharpe per dataset; turnover/breakeven median over datasets)", "",
              _md(t), ""]
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--csv", nargs="+", required=True)
    ap.add_argument("--families", nargs="+", default=["online", "online_forecast"])
    ap.add_argument("--sides", nargs="+", default=["both", "long_only"])
    ap.add_argument("--costs", nargs="+", type=float, default=list(COST_BPS))
    ap.add_argument("--quick-costs", nargs="+", type=float, default=None,
                    help="cost list for the (large) quick family only")
    ap.add_argument("--train", type=int, default=500)
    ap.add_argument("--test", type=int, default=125)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    parts, t0 = [], time.time()
    for path in a.csv:
        ds = Path(path).stem
        df = load_csv(path)
        for fam in a.families:
            costs = a.quick_costs if (fam == "quick" and a.quick_costs) else a.costs
            t1 = time.time()
            parts.append(evaluate(df, generator.generate_templates(fam), cost_bps_list=costs, sides=a.sides,
                                  train=a.train, test=a.test, jobs=a.jobs, dataset=ds, family=fam))
            print(f"{ds} {fam}: {time.time() - t1:.0f}s (total {time.time() - t0:.0f}s)", flush=True)
            rows = pd.concat(parts, ignore_index=True)
            rows.to_csv(out / "rows.csv", index=False)   # checkpoint after each cell
    rows = rows.drop_duplicates(["dataset", "template", "sides", "cost_bps", "cost_mult", "family"])
    (out / "summary.md").write_text(summarize(rows))
    print(f"wrote {out}/rows.csv and summary.md in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
