# Fixed share or discounting in the hedge learner?

The hedge learner (AdaHedge over a ladder of rungs, a meta learner on top,
restarted every bar over the last `memory` bars) needs a way to forget, so
that an expert that starts winning is not buried by what it lost before. It
used to **discount** (losses and the mixability gap decay with a lifetime H,
one learner per lifetime in `HEDGE_HORIZONS`). It now defaults to **fixed
share** (Herbster & Warmuth 1998): after every update a fraction alpha of the
weight is spread evenly over the experts, nothing is discounted. The switch is
`strategy.HEDGE_SHARE` / `--hedge-share {fixed_share,discount}`; runs saved
before the option existed replay discounted.

Reproduce everything below with

    python extra_utils/hedge_study/fixed_share_study.py        # parts A-D, ~15 min
    python main.py --csv data_dump/spy.csv --family online --no-matrix --hedge-share discount   # and fixed_share

The learner is causal and fits nothing, so every bar after its warm-up is
out of sample. "Committee" is the learner's own position (the weighted
experts' stances, clipped to [-1, 1]) charged 5 bps a side; its Sharpe is
what the learner is really scored on. "Stance tpl" is the `stance` entry
template that trades it through the engine.

## The short answer

- **The literature is right about the case it is about.** When the best
  expert *switches*, fixed share tracks it better: on a planted edge that
  moves between a follow and a fade expert it wins 81 % of the runs, on
  every ladder, and by the most on the slow ladder (committee Sharpe 0.62
  against 0.40), whose long lifetimes cannot let go of the old leader
  while fixed share can switch without throwing the evidence away.
- **On a constant edge it does not lose to discounting,** once its rate is
  chosen well (it won 69 % of the runs, by a little; the slow ladder is a
  tie). Neither is the best there: plain windowed AdaHedge (alpha = 0, the
  window restart the only forgetting) beats both on every constant-edge
  cell of part D (e.g. 1.21 against 1.00 and 0.89 on the plain ladder's
  follow_40 edge), as it should when nothing switches. It puts
  less weight on the planted expert (fixed share always keeps alpha / N on
  every other one) but the committee's Sharpe is as good or better.
- **The pipeline's headline number moved against it.** In `main.py
  --family online` (the shipped family) the nested walk-forward portfolio
  is worse under fixed share on three of the four ETFs (SPY -0.10 -> -0.35,
  TLT -0.09 -> -0.59, USO 0.43 -> -0.09, GLD 0.56 -> 0.55; mean 0.20 ->
  -0.12), and the best and stance templates are a little worse on average
  (0.44 -> 0.40, 0.34 -> 0.31). It is a selection over 9 templates that
  sits in cash in many windows, so it is noisy, but it is what a user of
  the family sees.
- **On the learner's own terms, SPY / TLT / USO (and GLD) are a coin flip.** Fixed share has the
  better committee in 10 of 20 series x ladder cells and the better stance
  template in 11 of 20; no paired Sharpe difference has |t| above 2 (the
  largest is 1.7). The median walk-forward template is a shade better
  under fixed share (0.19 against 0.17).
  **Real data does not show fixed share is better, and the pipeline's
  nested number leans against it.** It is the default because it was asked
  for, it is the better-founded rule and it is the clear winner where there
  is something to track; if you trade the `online` family on these ETFs,
  `--hedge-share discount` is one flag away and restores the old numbers
  exactly (discount mode is bit-identical to the previous learner).
- **The rate is learned, not fitted.** One learner per expected number of
  switches `m` in the learner's memory (`HEDGE_SHARE_SWITCHES` = 1/4, 1/2, 1,
  2, 4, 8; alpha = m / memory), weighed by the same meta learner that used
  to weigh the lifetimes. Averaged over 24 data set x learner cells it gets
  within 0.013 of the best single rate chosen in hindsight (0.543 against
  0.556; per learner the gap is 0.033, 0.011 and 0.027) and beats the best
  single discount lifetime chosen in hindsight (0.516). On the split
  ladder's follow group it does not beat discounting (0.401 against 0.426).
- **The old objection does not apply.** A fixed-share floor *on top of* the
  discounted learners was tried before and dropped for flipping the learned
  direction inside trends. Fixed share *instead of* discounting, at a few
  switches per memory, flips the direction *less* than discounting does (on
  the real series 3.6 % of bars against 5.0 % on the plain ladder, 5.4 %
  against 8.0 % on the split ladder).
- **Side effect: smaller positions.** The committee is less concentrated
  (mean |position| 0.18 against 0.27 on one test series), so position-sized
  templates (the wide and split ladders, the learned direction) trade
  smaller and whole-contract sizing rejects more entries. Sharpe is
  unaffected by that; total return and drawdown scale with it.

## How fixed share is put into AdaHedge

AdaHedge keeps per expert a deficit d (cumulative loss behind the leader)
and plays w = exp(-eta d) / z, with eta = ln N / (accumulated mixability
gap). Fixed share mixes the *weights* after the update, w <- (1 - alpha) w +
alpha / N, which is not of that form, so the deficits are re-read from the
shared weights at the new rate: d = -log(w) / eta, re-centred on the leader
(`strategy._hedge_round`). Then w = exp(-eta d) / z holds on the next
round's mix loss and gap, and with alpha = 0 it is plain AdaHedge exactly
(a unit test checks that; another checks the floor and revival with alpha
> 0). A deficit is at most log(N / alpha) / eta at the eta it was shared
at, which is the "never written off" bound: an expert that
trailed for 100 or for 240 bars is back in front within 7 bars of starting
to win by 0.2 a bar, where the discounted learner takes 20.

One deliberate departure from textbook Fixed Share: the shared deficits are
history like any other, and AdaHedge re-reads all of its history whenever
eta changes. When eta falls (it only falls inside a window, since nothing is
discounted) a trailer's floor of alpha / N rises to (alpha / N) ^ (eta_new /
eta_old), so the effective share rate is above alpha while eta is falling,
most in the first rounds of every window. Strict Fixed Share would mix in
weight space and multiply by exp(-eta_t l_t) with each round's own eta. The
version here keeps AdaHedge's deficit form (and its alpha = 0 identity); the
other was not tried.

Nothing is discounted inside the learner (gamma = 1), so the mixability gap
only grows inside the window: eta recovers when a burn leaves the memory,
not before. The meta learner over the rungs and the trade weight (the
played mixture against cash) are still discounted at the longest lifetime
of the ladder, as before, so the change is confined to the learners. The
window restart, the warm-up contract (`hedge_warmup`) and the walk-forward
equivalence are unchanged.

## A. A constant planted edge

`planted_edge.planted`: one Donchian expert (fade_20 or follow_40) carries
an edge of mu daily vols on the next bar; T = 3000, scored from bar 1000,
mu in {0.05, 0.1, 0.2}, four seeds. Mean over edges, mus and seeds:

| ladder | mode | committee disc. | committee f.s. | weight on the edge disc. | f.s. | flips disc. | f.s. |
|---|---|---|---|---|---|---|---|
| hedge | learned | 1.10 | **1.21** | 0.34 | 0.30 | 3 % | 2 % |
| hedge_wide | learned | 1.04 | **1.13** | 0.21 | 0.18 | 3 % | 2 % |
| hedge_slow | learned | **1.46** | 1.43 | 0.57 | 0.49 | 1 % | 1 % |
| hedge_split | learned | 0.38 | **0.52** | 0.17 | 0.14 | 8 % | 4 % |
| hedge_split | trend | 1.57 | **1.61** | 0.31 | 0.23 | | |

On the split ladder the weight on a fade_20 edge is the sum over its two
holds (fade_20_h1, fade_20_h3), neither of which is the planted 20-bar hold.
Fixed share has the better committee in 74 of 108 runs. With the rate set
naively to alpha = 1/H on the old lifetimes (5 % to 0.6 % a bar on the fast
ladder) it was *worse* than discounting here: a rate that high means a dozen
expected switches inside a 250-bar memory, and the learner never commits.
Tying the rates to the memory is what made the difference (part D).

## B. A switching planted edge

The edge alternates between follow_40 and fade_20 every 250 or 500 bars
(mu 0.1 or 0.2, four seeds), the case fixed share was designed for:

| ladder | committee disc. | committee f.s. | flips disc. | f.s. |
|---|---|---|---|---|
| hedge | 0.90 | **1.03** | 4 % | 3 % |
| hedge_wide | 0.87 | **0.99** | 4 % | 3 % |
| hedge_slow | 0.40 | **0.62** | 2 % | 2 % |
| hedge_split (learned) | 0.58 | **0.65** | 8 % | 5 % |

Fixed share is ahead in every one of the 16 (segment, mu, ladder) cells and
in 52 of 64 runs. The slow ladder gains most: on 500-bar segments at mu 0.2
its committee goes from 0.98 to 1.50, with a bar-level paired t between 2.3
and 3.4 on every one of the four seeds.

## C. SPY, TLT, USO and GLD

Daily bars 2016-2026 from `data_dump/`, scored from bar 800 (2019-03), 5 bps
a side. Committee Sharpe, then the paired t of the Sharpe difference
(fixed share minus discounting, both P&L streams scaled to unit vol):

| series | ladder | mode | committee disc. | f.s. | stance tpl disc. | f.s. | t |
|---|---|---|---|---|---|---|---|
| SPY | hedge | learned | -0.45 | -0.45 | -0.04 | -0.09 | 0.0 |
| SPY | hedge_wide | learned | -0.58 | -0.58 | -0.23 | -0.32 | 0.0 |
| SPY | hedge_slow | learned | -0.09 | -0.19 | 0.10 | 0.02 | -1.1 |
| SPY | hedge_split | learned | -0.42 | -0.24 | 0.02 | 0.03 | 1.1 |
| SPY | hedge_split | trend | 0.12 | 0.11 | 0.28 | 0.33 | -0.1 |
| TLT | hedge | learned | -0.24 | -0.28 | -0.15 | -0.04 | -0.3 |
| TLT | hedge_wide | learned | -0.37 | -0.34 | -0.38 | -0.12 | 0.2 |
| TLT | hedge_slow | learned | -0.10 | -0.22 | -0.07 | -0.13 | -1.1 |
| TLT | hedge_split | learned | 0.10 | 0.02 | 0.16 | 0.26 | -0.6 |
| TLT | hedge_split | trend | 0.05 | 0.08 | 0.18 | 0.18 | 0.7 |
| USO | hedge | learned | 0.26 | 0.27 | -0.04 | 0.19 | 0.1 |
| USO | hedge_wide | learned | 0.16 | 0.25 | -0.21 | 0.03 | 0.9 |
| USO | hedge_slow | learned | 0.08 | 0.06 | -0.05 | 0.01 | -0.2 |
| USO | hedge_split | learned | 0.66 | 0.47 | 0.56 | 0.32 | -1.7 |
| USO | hedge_split | trend | 0.47 | 0.42 | 0.36 | 0.33 | -0.9 |
| GLD | hedge | learned | 0.01 | 0.08 | 0.03 | 0.15 | 0.6 |
| GLD | hedge_wide | learned | 0.15 | 0.14 | 0.29 | 0.17 | -0.1 |
| GLD | hedge_slow | learned | 0.07 | 0.21 | 0.22 | 0.38 | 1.7 |
| GLD | hedge_split | learned | 0.42 | 0.33 | 0.50 | 0.42 | -0.9 |
| GLD | hedge_split | trend | 0.35 | 0.44 | 0.63 | 0.66 | 1.7 |

10 of 20 committees and 11 of 20 stance templates are better under fixed
share; the t's straddle zero and none reaches 2. Seven years of one asset
give a Sharpe a standard error of about 0.4, more than any gap here.

The pipeline's walk-forward (`main.py --family online --no-matrix`, train
500 / test 125, rolling, 5 bps; the family ships the split ladder's follow
group, trend direction, 9 templates):

| asset | median OOS disc. | f.s. | best disc. | f.s. | stance disc. | f.s. | nested disc. | f.s. |
|---|---|---|---|---|---|---|---|---|
| SPY | -0.29 | -0.09 | 0.28 | 0.25 | 0.28 | 0.25 | -0.10 | -0.35 |
| TLT | 0.13 | 0.05 | 0.23 | 0.16 | 0.14 | 0.05 | -0.09 | -0.59 |
| USO | 0.16 | 0.19 | 0.49 | 0.38 | 0.28 | 0.28 | 0.43 | -0.09 |
| GLD | 0.66 | 0.63 | 0.77 | 0.80 | 0.66 | 0.65 | 0.56 | 0.55 |
| mean | 0.17 | **0.19** | **0.44** | 0.40 | **0.34** | 0.31 | **0.20** | -0.12 |

The nested portfolio picks among 9 templates window by window (and sits in
cash in many of them), so it is the most variable number in the report; its
gap is the largest here and the one result that leans against fixed share.
Part D says the same thing from the inside: on the split ladder's follow
group (memory 500, a slow learner for a slow edge) the window restart alone
(alpha = 0) is the best forgetting on these series, and discounting at long
lifetimes is closer to that than fixed share is.

## D. Choosing the switching rate

Fixed share has one hyperparameter, alpha, the expected switches per bar.
Rather than fit it, the learner runs one rung per candidate rate and lets the
meta learner (Vovk's aggregating algorithm over the rungs' own losses, the
rule that already chose among the lifetimes) weigh them online. The question
is the grid. Committee Sharpe for the plain, slow and split-follow
learners, mean over eight data sets weighted alike: four synthetic ones
(constant follow_40 and fade_20 edges, edges switching every 250 and 500
bars, each the mean of three seeds) and the four ETFs:

| learner | best single f.s. rate (hindsight) | best single lifetime (hindsight) | f.s., alpha = m / memory (default) | f.s., alpha = 1/H on the lifetimes | discounting (old default) | alpha = 0 |
|---|---|---|---|---|---|---|
| hedge (memory 250) | 0.658 (1/640) | 0.577 (80) | **0.625** | 0.540 | 0.545 | 0.451 |
| hedge_slow (750) | 0.614 (1/320) | 0.579 (80) | **0.603** | 0.608 | 0.532 | 0.375 |
| split follow (500) | 0.428 (1/1280) | 0.464 (1280) | 0.401 | 0.390 | **0.426** | 0.469 |
| all | 0.556 | 0.516 | **0.543** | 0.512 | 0.501 | 0.432 |

What it says:

- **Fixed share is the better family of learners**: its best single rate
  beats the best single lifetime on the plain and slow ladders, and a
  learner that has to pick its rate online (the default ladder, 0.543)
  still beats one that was handed the best lifetime in hindsight (0.516).
- **The grid matters more than the rule.** alpha = 1/H on the old
  lifetimes (1/20 ... 1/160 on the fast ladder) is too fast: a dozen
  switches expected per memory, and on the plain ladder it is no better
  than discounting (0.540 against 0.545, the memory-scaled grid 0.625).
  Counting switches per *memory* (m = 1/4 to 8) puts the grid in the
  right place on the plain ladder; on the slow ladder the two grids tie
  (0.603 against 0.608), and on the split follow group neither beats
  discounting or alpha = 0. The meta learner gets within 0.013 of the
  best single rate in hindsight on average, 0.011 to 0.033 per learner.
- **Some forgetting is worth having** on the plain and slow ladders (alpha = 0,
  the window restart alone, is the worst column), **not on the slow
  follow group**, whose 500-bar memory already is the forgetting it needs.
- An undiscounted meta learner over the same rungs is indistinguishable
  (0.542), so the meta learner's discount is left as it was.

The meta learner does not concentrate on one rung: with a unit learning
rate and per-bar losses that differ by thousandths between rungs, the odds
move slowly, and the played weights are close to an average over the
rungs. That is a feature of the grid choice as much as of the meta learner:
an average over m = 1/4 ... 8 is a learner that expects "about one to a few
switches per memory", which is what these series reward. A sharper meta
learner (AdaHedge at this level) was rejected earlier for following rounding
noise and was not revisited.

## What changed in the code

- `strategy._hedge_round(..., alpha)`: the fixed-share step, in the
  deficit form above. `_hedge_window` / `_hedge_core` take per-rung
  `gammas`, `alphas` and the meta discount `gm` explicitly.
- `strategy.hedge_rungs(memory, horizons, share)`: a ladder's rungs under
  either rule; `HEDGE_SHARE` (default `"fixed_share"`), `HEDGE_SHARES`,
  `HEDGE_SHARE_SWITCHES`, `set_hedge_share`.
- Every cached indicator is keyed on `HEDGE_SHARE`, so switching it in a
  process never serves a stale channel, direction or stance.
- `--hedge-share` on `main.py` and `etf_dashboard.py research`; the value is
  in the run's config (`run.json`, the dashboard spec), set in every
  worker, and a config without it (written before this change) replays
  discounted (`pipeline.hedge_share_of`).
- `hedge_diagnostics`: the `eta` / `horizon_weights` columns are the rungs
  in bars (lifetimes, or `memory / m` between switches).
- Tests: the discount-specific learner tests run with `share="discount"`;
  new tests for the default, the switch and its plumbing, the weight floor,
  revival, alpha = 0 being plain AdaHedge, alpha < 1. Behavioural
  thresholds pinned on the discounted learner were re-checked under fixed
  share, and each relaxation is a real change of behaviour, not noise:
  - the learned direction on the regime series is compared by Sharpe
    (1.83 against 1.10 for trend only). By total return it no longer beats
    trend-only under fixed share (0.652 against 0.664; discounted 0.696
    against 0.651), because it trades smaller;
  - the trend learner's weight on its two slowest experts is 0.84 (pinned
    at 0.8, was 0.9): fixed share keeps 16 % on the fast ones;
  - the wide ladder's smallest conviction is about 6 % (floor), so "graded"
    is checked at < 0.1 instead of < 0.05;
  - "the plain ladder keeps buying dips the wide one stands aside from" is
    now a ratio (plain > 1.5 x wide; 0.32 against 0.16, discounted 0.53
    against 0.13) instead of a 0.2 gap that fixed share's smaller sizes
    could not keep;
  - the spread accounting test sizes at 2 % risk so it still has whole
    contracts to check;
  - the engine regression pinned before this change runs its hedge case
    discounted, and the same case is pinned to the bit under fixed share.
- `extra_utils/online_study/run_study.py` and `extra_utils/hedge_study/
  planted_edge.py` run discounted by default (`--hedge-share` / an
  argument to switch), so the earlier studies' numbers stay reproducible.
