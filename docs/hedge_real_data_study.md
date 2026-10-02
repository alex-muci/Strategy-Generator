# Do the hedge families win on real data?

A study of `quick`, `online`, `online_wide` (and `default`) on the series in
`data_dump/`, why the online-learned families rank as they do, and what was
changed in the learner. Everything below is the pipeline's own walk-forward
(`python main.py --csv data_dump/<x>.csv --family <f> --no-matrix`, train 500,
test 125, rolling, 5 bps a side, 1 % ATR risk) unless it says otherwise.

## The short answer

- **The hedge implementation has no bug, but it was naive in two ways, and one
  of them is a real design flaw.** On series with a planted edge the learner
  finds the right expert. But (1) its memory (lifetimes of 20-160 bars) is far
  too short to tell real-world experts apart, so on SPY, TLT, GLD and USO it
  re-weights on noise; and (2) the templates do not trade what the learner
  learns: a fade edge the learner's committee earns a Sharpe of 3.1 on is worth
  about 1.3 through the countertrend entry and exits.
- Both are fixed on this branch (slow ladders, `stance` entry). Neither makes
  the hedge families beat holding SPY or GLD, and no family does.
- **On the real ETFs no family is reliably better than another.** Ten years of
  one asset gives a Sharpe a standard error of about 0.35-0.4; the gaps
  between families are smaller than that. What is robust: the trend
  direction carries all of the edge, countertrend loses, `long_only` on an
  asset with a drift roughly doubles everything, and **no regime or vol filter
  beats "no filter"** in `quick` or `default`.
- **`--anchored` with `online`: no.** The learner's memory is fixed (250 bars
  of its own), so `--anchored` only changes how the few exit and regime
  parameters are refitted; on `online_wide` with the channel exit there is
  nothing to refit and the runs are identical. Over the four ETFs rolling was
  better on three for the nested portfolio, by amounts well inside the noise.
- **Synthetic data**: `data_dump/synthetic_trend.csv` is not what
  `python main.py --trend-prob 0.8 --trend-drift 0.002` generates (it rises
  from 100 to 2.2e10 with a lag-1 return autocorrelation of 0.74; the
  generator's series has 0.08). On the generator's own series, over five
  seeds, `online_wide` has the best honest (nested) Sharpe, 1.66, against 1.41
  for `online` and 1.34 for `quick`. `online` beats it in your run because one
  seed is one draw, and because the best of 288 templates is compared with
  the best of 8.

## 1. Baseline on the real series

Median OOS Sharpe of a family's templates (the fairest comparison across
families of 4 to 768 templates; "best" grows with the family size), and the
nested walk-forward portfolio (the honest selected number). Rolling windows.

Two-sided (the default):

| family | SPY | TLT | GLD | USO | avg median | avg nested |
|---|---|---|---|---|---|---|
| quick (72) | -0.03 | -0.07 | 0.02 | 0.03 | -0.01 | 0.06 |
| default (768) | -0.03 | -0.10 | 0.11 | 0.01 | 0.00 | -0.26 |
| online (288) | -0.12 | -0.05 | 0.27 | 0.10 | 0.05 | -0.01 |
| online_wide (8) | -0.51 | -0.33 | 0.35 | 0.10 | -0.10 | 0.21 |
| online_slow (288, new) | -0.04 | -0.15 | 0.32 | 0.00 | 0.03 | 0.14 |
| online_wide_slow (8, new) | -0.31 | -0.06 | 0.56 | 0.13 | 0.08 | -0.07 |
| online_stance (4, new) | 0.02 | -0.07 | 0.41 | 0.11 | 0.12 | -0.28 |
| buy and hold | 0.81 | -0.09 | 0.85 | 0.33 | 0.48 | |

`--sides long_only`:

| family | SPY | TLT | GLD | USO | avg median | avg nested |
|---|---|---|---|---|---|---|
| quick | 0.29 | -0.10 | 0.31 | 0.16 | 0.17 | 0.40 |
| default | 0.26 | -0.03 | 0.33 | 0.08 | 0.16 | 0.11 |
| online | 0.03 | -0.11 | 0.35 | 0.00 | 0.07 | 0.23 |
| online_wide | 0.07 | 0.03 | 0.69 | 0.18 | 0.24 | 0.26 |
| online_slow (new) | 0.07 | -0.29 | 0.46 | 0.21 | 0.11 | 0.16 |
| online_wide_slow (new) | 0.09 | -0.19 | 0.67 | 0.41 | 0.25 | 0.21 |
| online_stance (new) | 0.42 | -0.25 | 0.59 | 0.24 | 0.25 | 0.21 |

With `--vol-target 0.15` the picture is the same (long-only medians: quick
0.17, online_wide 0.18, online_wide_slow 0.23, online_stance 0.23).

Why two-sided `online_wide` is last on SPY and TLT: all 8 templates are the
`learned` direction, which on a drifting asset keeps paying for its fade
experts' shorts; the fast learner flips between near-tied follow and fade
experts (chasing the recent winner among strategies whose daily returns mean-
revert); and the wider ladder doubles the number of near-tied experts. The
`online` family has `trend` templates on the hedge channel at full size; those
are its better half (median 0.14 against 0.13 learned and -0.18 countertrend).

## 2. Flawed or naive?

### The learner works

`extra_utils/hedge_study/planted_edge.py` plants an edge on one expert of the
ladder (a random walk whose next move drifts with that expert's stance) and
asks whether the learner finds it. Mean of 3 seeds, 3000 bars, scored after
bar 1000:

| planted edge | expert Sharpe | committee, fast `hedge` | committee, `hedge_slow` | weight on it, fast / slow (uniform 0.12) |
|---|---|---|---|---|
| follow_40, weak | 0.49 | 0.28 | 0.23 | 0.19 / 0.28 |
| follow_40, medium | 1.68 | 1.02 | 1.54 | 0.31 / 0.72 |
| follow_40, strong | 3.44 | 3.21 | 3.36 | 0.41 / 0.67 |
| fade_20, weak | 0.54 | 0.00 | 0.24 | 0.17 / 0.24 |
| fade_20, medium | 1.37 | 0.57 | 1.11 | 0.32 / 0.63 |
| fade_20, strong | 3.16 | 2.76 | 3.14 | 0.67 / 0.96 |

So AdaHedge does what it should. With a strong edge the fast learner gets most
of it; with an edge of real-world size (Sharpe 0.5-1.5) it gets little,
because a lifetime of 20-160 bars cannot separate experts whose Sharpe differs
by tenths. That is the "naive" part, and it is fixed by memory, not by a
cleverer update.

### Real data says the same

The committee's own daily P&L (stance x next return, 5 bps on its turnover),
from bar 750, two-sided `hedge`: SPY -0.50, TLT -0.28, GLD 0.02, USO 0.20 with
the shipped memory; -0.21, -0.11, 0.06, 0.02 with a 750-bar memory and
lifetimes 250-750; the equal-weight follow experts alone: 0.24, 0.07, 0.32,
0.23. Longer memory moved every asset toward the uniform mix, monotonically.
Pooling the losses of the four ETFs (one learner, four times the data) helps
the fast learner as much as the slow memory does, and adds nothing on top of
it.

### The template does not trade what the learner learns (the flaw)

The experts are scored on holding a stance for n bars after a break. A
template trades a break of the weight-averaged channel and then exits by its
own rule with a 3-ATR hard stop. For following that is close enough; for
fading it is not:

| planted edge (slow ladder) | committee | break + channel exit | break + time exit | `stance` entry |
|---|---|---|---|---|
| fade_20, medium | 1.11 | 0.48 | 0.76 | 0.98 |
| fade_20, strong | 3.14 | 1.21 | 1.36 | 3.00 |
| follow_40, medium | 1.54 | 1.42 | 1.20 | 1.49 |
| follow_40, strong | 3.36 | 2.92 | 2.60 | 2.89 |

A `learned` template therefore cannot monetise the fade half of what it
learns, whatever the data.

## 3. What was changed (all alongside the old families, nothing overwritten)

- `hedge_slow`, `hedge_wide_slow` ladders: the same experts, memory 750,
  lifetimes 250/500/750 (`strategy.HEDGE_LEARNERS`). Families `online_slow`,
  `online_wide_slow`. Warm-up 911 bars instead of 410.
- `entry_style="stance"` (`strategy._stance_backtest`, `hedge_stance`): hold
  the committee's signed stance from the next open, in quarters of a full
  size, no stop or exit rule. Family `online_stance` (trend and learned on
  the two slow ladders). Research only: cash assets, `live.py` refuses it.
- `full` excludes both, so it is the family it was.

Variants tried at the template level on the four ETFs (both halves of the
out-of-sample period checked separately) and not shipped: full size instead
of the position sizing (helps long-only, hurts two-sided), sizing by the
trade weight alone, a mixed ladder of short and long lifetimes, no
discounting, slower rungs (20-160) on the fast learner. Slower rungs with the
slow memory were the one lead: the two-sided `trend` template on them scored
0.20 / 0.21 in the two halves against about 0 for the shipped ladders, but
that was picked out of a dozen variants on the same data, so it is a lead to
test on other assets, not a result.

## 4. What to do with it

- **Choose sides per asset.** Long-only on SPY / GLD / USO; this matters more
  than any family choice.
- **Regime filters**: none earned their place. In `default`, long-only,
  median by filter: none 0.22, chop-range 0.19, vr-range 0.21, cti 0.16,
  er 0.15, adx 0.14, vr-trend 0.08. In `quick` the vol filter adds 0.01 two-sided and
  0.02 long-only. A simpler family with filters is not better than the hedge; a
  simpler family *without* them, trend-only, is as good as anything here.
- **Hedge or quick?** Long-only, the hedge families (`online_wide`,
  `online_wide_slow`, `online_stance`) have the highest median template
  (0.24-0.25 against 0.17 for quick) with nothing fitted, while `quick`'s
  nested portfolio is higher (0.40 against 0.21-0.26). Both differences are
  inside the noise of four assets. What the hedge buys is not an edge; it is
  one less fitted parameter.
- **None of this beats holding SPY or GLD** over 2016-2026. That is the bar
  for a single-asset breakout system on these instruments, and these ten years
  were a poor stretch for trend following.
- To learn more than this, the test needs more independent data: more
  assets (futures across sectors) and longer history, with the families run
  as a multi-asset book. Pooling the learner across assets is the natural
  next step for the hedge (section 2).
