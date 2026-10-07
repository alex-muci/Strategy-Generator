"""Fixed share vs discounting in the hedge learner (docs/hedge_fixed_share_study.md).

Four checks, each run under both ways of forgetting (strategy.HEDGE_SHARES):

  A. planted edge: one expert has a known, constant edge (planted_edge.planted);
  B. switching edge: the planted edge moves between a follow and a fade expert
     every `seg` bars, the case fixed share was designed for;
  C. real series: SPY, TLT, USO (and GLD) in data_dump/;
  D. the switching rate: single-rung learners over a grid of rates against the
     ladder of rungs with its meta learner, on A-C.

The learner is causal and fits nothing, so every bar after the warm-up is out
of sample. The committee is the learner's own position (the weighted
experts' stances, clipped to [-1, 1]), charged 5 bps a side.

    python extra_utils/hedge_study/fixed_share_study.py [A B C D]
"""
import itertools
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import strategy as S  # noqa: E402
from data import load_csv  # noqa: E402
from planted_edge import planted  # noqa: E402

COST = 5.0
ATR_N = 20
SHARES = ("discount", "fixed_share")
REAL = ("spy", "tlt", "uso", "gld")   # the three asked for, and GLD
GRID = (10, 20, 40, 80, 160, 320, 640, 1280)    # single-rung H (alpha = 1/H, or lifetime H)
pd.set_option("display.width", 220)


def sharpe(r):
    r = np.asarray(r)
    return r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0.0


def sharpe_diff_t(a, b):
    """t-statistic of the Sharpe gap a - b on the same bars: the mean of the
    difference of the two P&L streams, each scaled to unit vol (fixed share
    holds smaller positions, so the raw difference measures size, not
    skill). Positive: `a` is better."""
    if a.std() == 0 or b.std() == 0:
        return 0.0
    d = a / a.std() - b / b.std()
    return d.mean() / d.std() * np.sqrt(len(d)) if d.std() > 0 else 0.0


def committee_pnl(df, W, Sx, start):
    """Daily P&L of the committee position (W . stance at the close of t,
    held over t+1), 5 bps a side on its changes, from bar `start`."""
    ret = df["Close"].pct_change().to_numpy()
    pos = np.clip(np.nan_to_num((W * Sx).sum(axis=1)), -1, 1)
    p = np.zeros(len(pos))
    p[1:] = pos[:-1] * ret[1:]
    p[1:] -= COST / 1e4 * np.abs(np.diff(pos))
    return p[start:]


def flips(W, sides, start):
    """Share of bars on which the learned direction (sign of the net side
    weight) flips: the cost the old fixed-share floor was dropped for."""
    net = np.sign((W * sides).sum(axis=1))[start:]
    return float(np.mean(net[1:] != net[:-1]))


def planted_switch(T, seg, mu, seed, vol=0.01, a=(40, 1), b=(20, -1)):
    """Random walk whose next-bar drift is side * stance * mu * vol of expert
    `a` = (n, side) for `seg` bars, then of expert `b`, alternating."""
    rng = np.random.default_rng(seed)
    c, hi, lo = [100.0], [100.2], [99.8]
    state = {a[0]: [-1, 0], b[0]: [-1, 0]}       # n -> [last break bar, stance]
    for t in range(1, T):
        n, side = a if (t // seg) % 2 == 0 else b
        lt, ls = state[n]
        s = ls if (lt >= 0 and (t - 1) - lt < n) else 0
        cl = c[-1] * (1 + side * s * mu * vol + rng.normal(0, vol))
        o = c[-1]
        h = max(o, cl) * (1 + abs(rng.normal(0, .003)))
        l_ = min(o, cl) * (1 - abs(rng.normal(0, .003)))
        c.append(cl); hi.append(h); lo.append(l_)
        for m in state:
            if t >= m:
                up, dn = h >= max(hi[t - m:t]), l_ <= min(lo[t - m:t])
                if up and not dn:
                    state[m] = [t, 1]
                elif dn and not up:
                    state[m] = [t, -1]
    idx = pd.bdate_range("2000-01-03", periods=T)
    return pd.DataFrame({"Open": [c[0]] + c[:-1], "High": hi, "Low": lo, "Close": c}, index=idx)


def learner(loss, ladder, mode, share):
    if ladder in S.HEDGE_SPLIT:
        S.set_hedge_share(share)
        return S._split_learner(loss, mode)[0]
    memory, horizons = S.hedge_learner(ladder)
    return S._adahedge_loop(loss, memory, horizons, share=share)[0]


def run_ladders(df, ladders, start, planted_label=None, templates=True):
    rows = []
    for ladder, mode in ladders:
        Sx, formed, ex = S._hedge_stances(df, ATR_N, mode, ladder, "both")
        loss = S._stance_loss(df, ATR_N, Sx, formed, COST)
        sides = np.array([e.side for e in ex], dtype=float)
        labels = [e.label for e in ex]
        pnl = {}
        for share in SHARES:
            W = learner(loss, ladder, mode, share)
            pnl[share] = committee_pnl(df, W, Sx, start)
            row = dict(ladder=ladder, mode=mode, share=share, committee=sharpe(pnl[share]),
                       flips=flips(W, sides, start) if mode == "learned" else np.nan)
            if planted_label is not None:
                # the planted break, whatever its hold (the split ladder's fade_20_h1 / fade_20_h3)
                cols = [j for j, lab in enumerate(labels) if lab == planted_label or lab.startswith(planted_label + "_")]
                row["weight"] = W[start:, cols].sum(axis=1).mean() if cols else np.nan
            if templates:
                S.set_hedge_share(share)
                tpl = S.StrategyTemplate("x", direction_logic=mode, channel_type=ladder, entry_style="stance",
                                         exit_style="channel", cost_bps=COST)
                row["stance_tpl"] = sharpe(S.backtest(df, tpl, fixed_capital=True)["returns"].to_numpy()[start:])
            rows.append(row)
        for r in rows[-2:]:
            r["diff_t"] = sharpe_diff_t(pnl["fixed_share"], pnl["discount"])
    return rows


LADDERS = [("hedge", "learned"), ("hedge_wide", "learned"), ("hedge_slow", "learned"),
           ("hedge_split", "learned"), ("hedge_split", "trend")]


def part_a(seeds=range(4)):
    rows = []
    for mu, (n, side), seed in itertools.product([0.05, 0.1, 0.2], [(20, -1), (40, 1)], seeds):
        df = planted(3000, n, side, mu, seed)
        lab = ("follow_" if side > 0 else "fade_") + str(n)
        ladders = [l for l in LADDERS if not (side < 0 and l[1] == "trend")]
        for r in run_ladders(df, ladders, 1000, lab, templates=False):
            rows.append(dict(edge=lab, mu=mu, seed=seed, **r))
    R = pd.DataFrame(rows)
    print("\nA. constant planted edge (T=3000, from bar 1000; mean over seeds)")
    print(R.groupby(["edge", "mu", "ladder", "mode", "share"]).mean(numeric_only=True)
          .drop(columns=["seed"]).round(3).to_string())
    print(R.groupby(["ladder", "mode", "share"]).mean(numeric_only=True)
          .drop(columns=["seed", "mu"]).round(3).to_string())
    return R


def part_b(seeds=range(4)):
    rows = []
    for seg, mu, seed in itertools.product([250, 500], [0.1, 0.2], seeds):
        df = planted_switch(3000, seg, mu, seed)
        for r in run_ladders(df, LADDERS[:4], 1000, templates=False):
            rows.append(dict(seg=seg, mu=mu, seed=seed, **r))
    R = pd.DataFrame(rows)
    print("\nB. switching planted edge, follow_40 <-> fade_20 every seg bars (from bar 1000; mean over seeds)")
    print(R.groupby(["seg", "mu", "ladder", "share"]).mean(numeric_only=True)
          .drop(columns=["seed", "flips"]).round(3).to_string())
    return R


def part_c():
    rows = []
    for name in REAL:
        df = load_csv(os.path.join(HERE, "..", "..", "data_dump", f"{name}.csv"))
        for r in run_ladders(df, LADDERS, 800):
            rows.append(dict(series=name.upper(), **r))
    R = pd.DataFrame(rows)
    print("\nC. real series 2016-2026 (from bar 800: 2019-03 on), 5 bps a side")
    print(R.set_index(["series", "ladder", "mode", "share"]).round(3).to_string())
    print(R.groupby(["ladder", "mode", "share"]).mean(numeric_only=True).round(3).to_string())
    return R


def rung_weights(loss, memory, gammas, alphas, gm):
    """Played weights of the windowed learner over explicit rungs."""
    return S._hedge_fast(np.ascontiguousarray(loss), int(memory), np.asarray(gammas, dtype=float),
                         np.asarray(alphas, dtype=float), float(gm))[0]


def part_d(seeds=range(3)):
    """How the switching rate is chosen. Per learner and data set, committee
    Sharpe of: single-rung learners over GRID (fixed share alpha = 1/H, or a
    discount with lifetime H), the default ladder of each mode (fixed share:
    alpha = m / memory, m in HEDGE_SHARE_SWITCHES; discount: the ladder's
    lifetimes), the same fixed-share rungs with an undiscounted meta learner,
    alpha = 1/H over the ladder's lifetimes, and alpha = 0 (plain windowed
    AdaHedge: the window restart is the only forgetting)."""
    sets = []
    for seed in seeds:
        sets.append(("planted fade_20", seed, planted(3000, 20, -1, 0.1, seed), 1000))
        sets.append(("planted follow_40", seed, planted(3000, 40, 1, 0.1, seed), 1000))
        sets.append(("switch 500", seed, planted_switch(3000, 500, 0.2, seed), 1000))
        sets.append(("switch 250", seed, planted_switch(3000, 250, 0.2, seed), 1000))
    for name in REAL:
        sets.append((name.upper(), 0, load_csv(os.path.join(HERE, "..", "..", "data_dump", f"{name}.csv")), 800))
    configs = [("hedge", "learned", S.HEDGE_MEMORY, S.HEDGE_HORIZONS),
               ("hedge_slow", "learned", S.HEDGE_SLOW_MEMORY, S.HEDGE_SLOW_HORIZONS),
               ("split follow", "trend", *S.HEDGE_SPLIT_LEARNERS["follow"])]
    rows = []
    for (lab, seed, df, start), (cname, mode, memory, horizons) in itertools.product(sets, configs):
        ladder = "hedge_split" if cname == "split follow" else cname
        Sx, formed, _ = S._hedge_stances(df, ATR_N, mode, ladder, "both")
        loss = S._stance_loss(df, ATR_N, Sx, formed, COST)
        gm = 1.0 - 1.0 / max(horizons)
        sh = lambda W: sharpe(committee_pnl(df, W, Sx, start))   # noqa: E731
        row = dict(data=lab, seed=seed, learner=cname)
        for H in GRID:
            row[f"fs{H}"] = sh(rung_weights(loss, memory, [1.0], [1.0 / H], gm))
        for H in GRID:
            row[f"d{H}"] = sh(rung_weights(loss, memory, [1.0 - 1.0 / H], [0.0], gm))
        row["fixed_share"] = sh(S._adahedge_loop(loss, memory, horizons, share="fixed_share")[0])
        row["discount"] = sh(S._adahedge_loop(loss, memory, horizons, share="discount")[0])
        g, a, _, _ = S.hedge_rungs(memory, horizons, "fixed_share")
        row["fs_meta_nodisc"] = sh(rung_weights(loss, memory, g, a, 1.0))
        row["fs_1/H"] = sh(rung_weights(loss, memory, np.ones(len(horizons)), 1.0 / np.asarray(horizons, float), gm))
        row["alpha0"] = sh(rung_weights(loss, memory, [1.0], [0.0], gm))
        rows.append(row)
    R = pd.DataFrame(rows)
    G = R.groupby(["learner", "data"], sort=False).mean(numeric_only=True).drop(columns=["seed"])
    print("\nD. committee Sharpe: single fixed-share rungs fsH (alpha = 1/H), single discounted rungs dH "
          "(lifetime H), and the ladders (mean over seeds)")
    print(G.round(2).to_string())
    print(G.groupby("learner", sort=False).mean().round(3).to_string())
    print(G.mean().round(3).to_string())
    return R


if __name__ == "__main__":
    parts = sys.argv[1:] or ["A", "B", "C", "D"]
    out = os.environ.get("STUDY_OUT", os.path.join("outputs", "hedge_fixed_share"))
    os.makedirs(out, exist_ok=True)
    for p in parts:
        R = {"A": part_a, "B": part_b, "C": part_c, "D": part_d}[p]()
        R.to_csv(os.path.join(out, f"fixed_share_{p}.csv"), index=False)
    S.set_hedge_share("fixed_share")
