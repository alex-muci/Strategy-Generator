# Improving the online templates: real ETFs, synthetic series, futures spreads

A study of the `online` (9 templates, hedge_split follow group) and
`online_forecast` (3 forecast-channel stance templates) families on the four
ETFs in `data_dump/` and on synthetic series with a planted, known edge,
including futures calendar spreads, with turnover penalised by costs. It
follows `docs/hedge_real_data_study.md`. Tools: `extra_utils/online_study/`
(`run_study.py`: walk-forward under a cost sweep with turnover and breakeven;
`synth.py`: the generators).

## The short answer

- **Turnover is not excessive anywhere in the online families.** At 5 bps a
  side no template trades more than ~6.5x its equity a year (the target and
  time exits); the forecast templates trade 1-1.7x, the cost drag is at most
  0.4 % a year, and most templates are still positive at 20 bps. The
  templates whose breakeven is near 5-10 bps on some ETF are the target-stop
  and time exits and (before this change) the hedge stance.
- **One change shipped: the hedge stance is held under a no-trade band**
  instead of being rounded to quarters (`STANCE_BUFFER` = 0.125, one-sided
  templates snap flat on any bar with no stance on their side,
  `STANCE_SETTLE` = 200). Walk-forward on the four ETFs, two-sided
  `TR-hsp-stance`: OOS Sharpe 0.17 -> 0.34 at 5 bps, 0.05 -> 0.25 at 20 bps,
  turnover 3.0 -> 1.6x equity a year. Long-only: 0.39 -> 0.40 (noise).
  Nothing else in either family changed (228 of 288 walk-forward rows are
  bit-identical; the rest are this template and the forecast's one-sided
  stance, which moved by 0.01).
- **Three ideas were tested and NOT shipped**: a cost-aware (Whalley-Wilmott)
  band for the forecaster, an AdaHedge meta-learner over forecasters with
  different priors scored on net-of-cost P&L, and a looser prior on the
  fast (reversion) coefficient for spreads. Each lost on the multi-seed
  synthetic suite; details below.
- **Futures spreads: use `online_forecast`, not `online`.** The `online`
  family follows breakouts; on a stationary calendar spread it loses at
  every exit. The forecast templates make money where the spread has an
  edge they can see (countertrend on a fast-reverting spread, all three on
  a spread whose mean trends) and lose their costs where it has none.
- **None of this changes the bar from the earlier study**: no online
  template beats holding SPY or GLD over 2016-2026, and four ETFs give a
  Sharpe a standard error of ~0.35-0.4 each.

## Sources, and what could not be read

The network policy of this environment blocks financial-hacker.com and
Yahoo Finance, so the "Build Better Strategies" series, the Ranger article
and the links in its comments could not be fetched, and no more data could
be downloaded (the generator docstring asks for 2007-2015 and more tickers).
What the work takes from that series is from memory, not from re-reading
it: model-based edges are either trend or mean reversion, so test both
against a random-walk null; the trendiness filters (Hurst, the Market
Meanness Index, lowpass filters) already exist here as the forecast's
context (`forecast_ctx`) and as regime filters, and the earlier study found
none of them beat "no filter"; the process discipline (walk-forward, an
out-of-sample that was not used to choose, counting the trials behind a
result, White's reality check) is what the harness and the adversarial
review below enforce. Nothing here quotes the Ranger rules or results.

## 1. The harness and the synthetic suite

`run_study.evaluate` walks every template forward (train 500, test 125,
rolling, the pipeline's own `walk_forward` and `param_grid_for`) at each
cost of a sweep (bps, or a per-unit multiplier 0 / 0.5 / 1 / 2 / 4 of the
tick and roll costs for a future), and reports OOS Sharpe (and by half),
turnover measured twice (the cost drag per bp, and the traded notional,
margin basis for a future, from the replayed windows; the two agree to
1.00 on the forecast templates), cost drag at 5 bps and the breakeven cost.

`synth.py` (each frame carries its ground truth in `attrs["truth"]`):

| generator | planted structure |
|---|---|
| `random_walk` | GARCH(1,1) random walk: the null, any Sharpe is false discovery |
| `tsmom` | time-series momentum at a set IC (0.02-0.04: annual Sharpe ~0.3-0.6) |
| `reversal_trend` | a slow trend plus short-horizon reversal |
| `regime_switch` | Markov trend / mean-reverting regimes of 60-250 bars |
| `ar1` | AR(1) returns |
| `calendar_spread` | a spread in points through zero: seasonal mean, OU deviation (half-life 5-20), jumps, monthly rolls with a `Roll` column, $12.5 a side per spread and $25 a roll at $1,000 a point |
| `trending_spread` | the same, with a mean that trends in regimes |

Caveats on the spreads (from the review): the OU half-lives are far shorter
than a real crude calendar spread's term-structure persistence, so the
planted edge is stronger than a real one (an oracle fade with the true mean
earns a Sharpe of 1.4-2.9 before costs); the synthetic account trades
fractional contracts (`whole_units` off), so sliver re-sizes that a futures
account cannot make are counted; and the roll offset is booked as a
zero-mean price move.

## 2. Turnover and costs, baseline (ETFs, walk-forward)

Median OOS Sharpe over SPY / TLT / GLD / USO by cost, before the change:

| family | sides | 0 bps | 2 | 5 | 10 | 20 |
|---|---|---|---|---|---|---|
| online | both | 0.21 | 0.17 | 0.13 | 0.07 | -0.04 |
| online | long_only | 0.33 | 0.31 | 0.32 | 0.27 | 0.20 |
| online_forecast | both | 0.18 | 0.17 | 0.16 | 0.13 | 0.09 |
| online_forecast | long_only | 0.23 | 0.23 | 0.22 | 0.22 | 0.19 |
| quick | both | 0.05 | 0.02 | -0.02 | -0.09 | -0.26 |
| quick | long_only | 0.21 | 0.18 | 0.16 | 0.12 | 0.01 |

Turnover (equity a year, median over the ETFs, 5 bps): forecast templates
0.5-1.7 (the buffer works: the position changes ~100 times a year but by
small amounts); `online` channel exits 0.9-1.0, trail 2.5, time 3.9-4.0,
target 6.3-6.5, the stance 3.2. Two-sided breakevens under ~11 bps: the
target, time and close-confirm trail exits (5-11 bps). 5 bps a side is
already a conservative cost for these ETFs, so the online families are not
over-trading; what costs them is their gross edge, not their costs.

## 3. What changed: the hedge stance band

The `stance` entry on a hedge ladder rounded the committee's stance to
quarters. Rounding is a band of 1/8 around FIXED grid points, without
memory: a stance of 0.12 / 0.13 / 0.12 trades 0 -> 0.25 -> 0 every bar.
It now holds the level under a hysteresis band of the same 1/8 (the
forecast channels' mechanism): the level stays while the stance is within
0.125 of it, else moves to the band's near edge.

Evidence (all at 5 bps unless stated):

- Full-history backtest, Sharpe from bar 1000, two-sided average of the
  four ETFs: quarters 0.21, band 0.39; widths 0.1 / 0.15 / 0.2 / 0.25 / 0.3
  all 0.39-0.41 (a plateau; 0.125 is the quarter rule's own tolerance, not
  the argmax). At 20 bps: 0.11 -> 0.30.
- Walk-forward (`run_study`), same ETFs: two-sided 0.17 -> 0.34; 20 bps
  0.05 -> 0.25; long-only 0.39 -> 0.40.
- Synthetic suite, 8 generators x 6-8 fresh seeds: the paired gain is
  +0.05 to +0.11 Sharpe (+/- 0.02) depending on the seed set; the cost drag
  per unit of cost falls 38 % at every cost multiple (1x-4x).
- How it gains (the review's measurement): most of the two-sided gain exists
  at ZERO cost (0.28 -> 0.43), so it is mainly a different exposure path, not
  the cost saving; the cost drag itself falls from 0.07 to 0.04 Sharpe and
  notional turnover from 3.2 to 1.8x a year. The trade COUNT roughly doubles:
  after a move the level sits at the band's edge, so a trending stance
  re-sizes by a sliver on most bars. That is free under proportional costs;
  under per-ticket commissions or whole contracts it is not (with
  `whole_units` the engine already skips a re-size that rounds to the same
  contracts).

One-sided templates. Banding the raw stance and zeroing the forbidden side
afterwards let a long-only level sit at -0.5 through a short stretch and
re-enter late; clamping the target at 0 instead left a level that could
keep any sliver in [0, 0.125] for as long as the stance sat at 0, and a
walk-forward window then needed 550+ bars of extra warm-up to reproduce a
full-history run (which, through the CPCV embargo, left 9 of 28 CPCV splits
with under 400 training bars on a 2,700-bar series). The shipped rule: on a
one-sided template the level goes flat on any bar whose stance has nothing
on the template's side, and bands from there. Exact warm-up from 100 settle
bars in 108 measured cases (ETFs, spreads with per-unit costs and `Roll`,
tsmom; both / long_only / short_only); `STANCE_SETTLE` = 200, so the shipped
stance template's warm-up is 986 bars (786 with quarters). The same rule now
applies to the forecast channels' one-sided stance: their warm-up was
inexact in 54 of 324 cases at `FC_SETTLE` = 100 and is in 1 (a 1e-5
difference); their one-sided Sharpes moved by at most 0.03 (full history)
and 0.01 (walk-forward), with fewer trades.

What this is not: the band-vs-quarters decision was taken on the same four
ETF histories that give the headline numbers, after ~15 holding rules had
been tried on them; the out-of-sample support is the synthetic suite on
fresh seeds and the plateau. The long-only gain is one ETF (GLD) and noise.

## 4. Tried and not shipped

**A cost-aware band for the forecaster.** The forecast's no-trade buffer is
a fixed 0.1 whatever the cost. The Whalley-Wilmott / Garleanu-Pedersen band
h = (3 kappa s^2 / 2 gamma)^(1/3) (kappa the cost in units of the bar's
volatility, s^2 the per-bar variance of the target, gamma = SR_FULL /
sqrt(252) from the forecaster's own sizing) gives ~0.17 for SPY at 5 bps and
~0.5 for a crude calendar spread at one tick. On the ETFs it looked better
(learned SPY 0.44 -> 0.57); on 8 seeds per generator it was neutral on the
trend series and destroyed the fast-reversion edge on spreads (countertrend
on a half-life-5 spread 0.45 -> -0.09): the formula assumes a target that
diffuses, and a fast mean-reverting forecast has a large one-bar variance
but does not wander, so the band comes out far too wide. Measuring s^2 at
the band's time scale (5 or 20 bars) did not rescue it. On the synthetic
suite no width beat the shipped 0.1 by more than noise (0.2: +0.01, 0.05:
-0.03), so `FC_BUFFER` stays.

**Learning the prior strength (AdaHedge over forecasters).** The forecaster
holds its fast (reversion) coefficient to a prior of 2,500 pseudo-bars; over
~220 effective bars of discounted data that keeps a spread's reversion at
~10 % of what the data says (so the code comment "a spread's reversion
overcomes it within a few hundred bars" is optimistic). An AdaHedge
committee over ridge forecasters with priors 25 / 250 / 2,500, an agnostic
variant with the trend free, the prior alone and flat, each scored on its
buffered position's next-bar P&L net of kappa x |change|, spread its weight
over near-tied experts and did not beat the plain forecaster on any series
class, including the null (the earlier study's finding again: learning from
daily P&L mostly learns noise). Sweeping the prior directly (250 / 500 /
1,000 / 2,500) on 6 seeds: looser priors add trades and lose almost
everywhere; on the fast spread the learned template gains (-0.44 -> 0.03)
but countertrend at the shipped prior is as good (0.27). The shipped prior
is right.

## 5. Futures spreads

Walk-forward on the synthetic spreads (3 seeds, 1x the tick and roll costs),
before the change:

| template | spread, half-life 20 | half-life 5 | trending mean |
|---|---|---|---|
| CT-fc-stance | -0.46 | 0.14 | 0.50 |
| LN-fc-stance | -0.51 | -0.32 | 0.32 |
| TR-fc-stance | -0.46 | -0.47 | 0.44 |
| TR-hsp, best of 9 | -0.05 | 0.03 | 0.05 |
| TR-hsp, worst of 9 | -0.29 | -0.71 | -0.56 |

On the half-life-20 spread the observable features carry no edge (IC of the
fast kernel 0.005, against -0.14 for the true, unobservable deviation; the
seasonal mean, its drift and the rolls hide it) and every template loses its
costs. The learned forecaster loses on stationary spreads because its trend
prior points the wrong way there (the slow kernel's IC is -0.02); pick the
countertrend forecaster for a mean-reverting spread, or all three for one
whose level trends. The `online` family is a breakout follower and the
wrong tool for a stationary spread.

## 6. Reproduce

```
PYTHONPATH=. python -m extra_utils.online_study.run_study \
    --csv data_dump/spy.csv data_dump/tlt.csv data_dump/gld.csv data_dump/uso.csv \
    --families online online_forecast --costs 0 5 20 --out out_etf
PYTHONPATH=. python -m extra_utils.online_study.run_study \
    --synth calendar_spread:half_life=5 --seed 3 --families online online_forecast --out out_spread
```
