"""Planted-edge check for the hedge learner (docs/hedge_real_data_study.md).

Builds random-walk series on which ONE expert of the ladder has a known edge
(the next bar drifts mu daily vols in the direction of that expert's stance)
and reports, per ladder, the expert's own Sharpe, the weight the learner puts
on it, the Sharpe of the learner's committee (what it is scored on) and the
Sharpe of templates that trade it: a channel break (close_confirm entry) with
the channel or time exit, and the 'stance' entry.

    python extra_utils/hedge_study/planted_edge.py [fixed_share]

The published numbers (docs/hedge_real_data_study.md) were computed with the
discounted learner, so that is what it runs unless told otherwise.
"""
import itertools
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import strategy as S  # noqa: E402


def sharpe(r):
    r = np.asarray(r)
    return r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0.0


def planted(T, n_edge, side, mu, seed, vol=0.01):
    """Series whose next-bar move is side * stance * mu * vol + noise, the
    stance being the Donchian(n_edge) expert's (+1 for n_edge bars after a new
    high, -1 after a new low); side -1 plants a FADE edge."""
    rng = np.random.default_rng(seed)
    c, hi, lo = [100.0], [100.2], [99.8]
    last_t, last_s = -1, 0
    for t in range(1, T):
        s = last_s if (last_t >= 0 and (t - 1) - last_t < n_edge) else 0
        cl = c[-1] * (1 + side * s * mu * vol + rng.normal(0, vol))
        o = c[-1]
        h = max(o, cl) * (1 + abs(rng.normal(0, .003)))
        l_ = min(o, cl) * (1 - abs(rng.normal(0, .003)))
        c.append(cl); hi.append(h); lo.append(l_)
        if t >= n_edge:
            up, dn = h >= max(hi[t - n_edge:t]), l_ <= min(lo[t - n_edge:t])
            if up and not dn:
                last_t, last_s = t, 1
            elif dn and not up:
                last_t, last_s = t, -1
    idx = pd.bdate_range("2000-01-03", periods=T)
    return pd.DataFrame({"Open": [c[0]] + c[:-1], "High": hi, "Low": lo, "Close": c}, index=idx)


def main():
    S.set_hedge_share(sys.argv[1] if len(sys.argv) > 1 else "discount")
    rows, st = [], 1000
    for mu, (n, side), seed in itertools.product([0.05, 0.1, 0.2], [(20, -1), (40, 1)], range(3)):
        df = planted(3000, n, side, mu, seed)
        ret = df["Close"].pct_change().to_numpy()
        for ladder in ["hedge", "hedge_slow", "hedge_wide", "hedge_wide_slow"]:
            Sx, _, ex = S._hedge_stances(df, 20, "learned", ladder, "both")
            W = S.hedge_weights(df, 20, "learned", 5.0, ladder)
            j = [e.label for e in ex].index(("follow_" if side > 0 else "fade_") + str(n))

            def pnl(pos):
                pos = np.nan_to_num(pos)
                p = np.zeros(len(pos)); p[1:] = pos[:-1] * ret[1:]
                return p[st:]
            row = dict(mu=mu, edge=("follow_" if side > 0 else "fade_") + str(n), seed=seed, ladder=ladder,
                       expert=sharpe(pnl(Sx[:, j])), weight=W[st:, j].mean(), committee=sharpe(pnl((W * Sx).sum(1))))
            for lab, es, exs in [("break+channel", "close_confirm", "channel"), ("break+time", "close_confirm", "time_stop"),
                                 ("stance", "stance", "channel")]:
                tpl = S.StrategyTemplate("x", direction_logic="learned", channel_type=ladder, entry_style=es, exit_style=exs)
                row[lab] = sharpe(S.backtest(df, tpl, fixed_capital=True)["returns"].to_numpy()[st:])
            rows.append(row)
    R = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(R.groupby(["edge", "mu", "ladder"]).mean(numeric_only=True).drop(columns="seed").round(2).to_string())


if __name__ == "__main__":
    main()
