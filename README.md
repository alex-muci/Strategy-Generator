# Strategy Generator

An original Python implementation of the *idea* behind Robert Pardo's
Ranger / RangerZ system (as described in "Evaluating Robert Pardo's
Ranger System", financial-hacker.com): not one strategy, but a
**generator of structurally distinct strategies**, each
evaluated with **walk-forward analysis**, stress-tested for
**backtest overfitting** (Lopez de Prado, *Advances in Financial
Machine Learning*), and combined into a **portfolio of low-correlation
strategies** whose selection is itself walked forward.

Pardo's actual EasyLanguage code is proprietary and was not used or
referenced here -- this is a from-scratch build of the same well-known
concepts (channel breakouts, walk-forward analysis, correlation-based
portfolio construction) plus the modern overfitting diagnostics.

## Setup

```bash
conda create -p ./env python=3.12 pandas scikit-learn scipy matplotlib yfinance numba
conda activate ./env
# or, with pip:  pip install -r requirements.txt   (the versions the suite was last run against)

python -m unittest discover -s tests -v      # 174 tests (engine, templates, hedge learner, walk-forward, robustness, selection, data, live signals, both entry points)
# faster (about 2.5 min instead of 4.5): pip install -r requirements-dev.txt, then, with ./env active,
python -m pytest -n auto --dist loadscope   # same tests in parallel; loadscope keeps a class (and its one-off setup) on one worker

python main.py                                # synthetic data, 72 templates, ~1 min on 8 cores
```

## Running it

```bash
python main.py --help
python main.py --family quick                       # 72 templates (Donchian, ER filter)
python main.py --family default                     # ~770 templates, all switches sampled
python main.py --family online                      # ~290 templates on the online-learned channel (no lookback to fit)
python main.py --real SPY --start 2005-01-01 --family default --jobs 8
python main.py --real SPY --start 2005-01-01 --family quick --sides long_only   # one-sided family (an asset with a drift)
python main.py --real QQQ --start 2010-01-01 --train 500 --test 125   # rolling window (default): each window re-optimizes on the last 500 bars only
python main.py --real GC=F --train 750 --test 250 --anchored --selection best   # anchored: training (window expands from bar 0)
python main.py --trend-prob 0.8 --trend-drift 0.002  # synthetic data with a KNOWN trend edge
```

Outputs land in `./outputs/` (or `--out DIR`):

| file | what it shows |
|---|---|
| `equity_curves.png` | every template's OOS curve (grey), the selected ones, the static portfolio (black), the **nested walk-forward portfolio** (red dashed) and **buy & hold** of the same asset over the same bars (dotted) |
| `template_ranking.png/.csv` | every template ranked by OOS Sharpe, with WFE, % profitable windows, Pardo pass, CPCV mean and P(CPCV<0); red line = expected max Sharpe of that many noise trials, dotted line = buy & hold Sharpe |
| `cpcv_distribution.png` | for each finalist, the distribution of Sharpe over all combinatorial-CV paths vs. the single walk-forward path |
| `pbo.png` | CSCV logit histogram (Probability of Backtest Overfitting) and OOS-vs-IS degradation for the whole family |
| `wfa_matrix_*.png` | Pardo's walk-forward matrix (train x test lengths) for the top finalist |
| `correlation_heatmap.png` | correlation of qualifying templates' OOS returns |
| `selected_windows.csv` | per-window chosen parameters and IS/OOS stats of the finalists |
| `report.md` | written summary with all the numbers, including the **buy & hold benchmark**: Sharpe, CAGR and drawdown of simply holding the asset over the same OOS bars, the nested portfolio's beta and correlation to it and its information ratio (what is left after the asset's own drift is removed), and how many templates beat holding at all |


## Trading it: the ETF dashboard

`main.py` answers "would this have worked?". `etf_dashboard.py` answers "what do
I place at my broker this morning?" for a handful of ETFs, in two phases.

```bash
# once (minutes to an hour) -- decides WHAT to trade
python etf_dashboard.py research --assets SPY TLT GLD QQQ --family default \
    --start 2010-01-01 --jobs 8

# every morning, or a few times a day on hourly bars (seconds)
python etf_dashboard.py signals --account-equity 100000
open state/dashboard.html
```

The split is the point. Research runs the whole pipeline per **(asset, template)
pair** -- so the candidate pool is structurally *and* market diversified -- and
writes `state/portfolio.json`: the slots to trade, their weights, the
diagnostics and a plain-language verdict. Re-running that every morning would be
a fresh data-mining exercise every morning, which is the failure mode the
robustness toolkit exists to measure. The signals phase only *applies* the spec,
so it is fast enough for cron:

```
daily bars:   10 17 * * 1-5   cd <repo> && python etf_dashboard.py signals
hourly bars:  35 10-16 * * 1-5  ...  (a few minutes after each bar closes)
```

Add `--interval 1h` to both phases for hourly bars; every annualized statistic
follows the bar frequency (`strategy.BARS_PER_YEAR`).

What the dashboard shows, in the order you need it:

| | |
|---|---|
| **verdict** | the research conclusion, restated every run, so a weak result cannot quietly become habit |
| **exposure** | gross, net, and the money at risk if every stop fills at once; `--max-gross` scales the whole book down proportionally |
| **trades to send** | target minus held, per ETF. Put your real broker positions in `state/holdings.json` (`{"SPY": 120, "TLT": -50}`) or it assumes the last run's orders were filled |
| **orders to work** | the level AND the order type for the next bar -- a breakout entry is a stop order, fading one is a limit order -- with share counts from the engine's own sizing rule |
| **positions** | entry, current stop, and how much of the original stop distance is left |
| **track record** | the nested walk-forward curve against simply holding the same ETFs |
| **slots** | current parameters and when they are next re-optimized |
| **diagnostics** | PBO, Reality Check, DSR, and the whole search you picked from |

Two rules keep the live path honest, both in `live.py`:

- **Parameters only change at window boundaries.** Re-optimizing every run would
  be a different strategy from the one the walk-forward measured with
  `test_bars`-long parameter holds. `due_for_refit` enforces the same cadence.
- **The forming bar is dropped.** A feed queried at 11:15 returns an 11:00 bar
  built from 15 minutes of trading; its high, low and close all still move, so
  acting on it is a decision you could not have taken.

Everything the dashboard reports about the *current* position (side, size, stop,
target, trailing anchor, resting order) is read out of the same
`strategy.backtest` loop the research validated, not reimplemented next to it.
The one thing `live.py` must restate is the *next* bar's order levels, since that
bar does not exist yet -- so `tests/test_live.py` rolls one real bar forward
across the template family and asserts the engine filled exactly what the
dashboard published, at the price that order type implies.

Nothing in this repository places an order. It tells you what to place.

## Architecture

```
data.py         Load real data (yfinance) or generate synthetic
                regime-switching OHLC (trend probability / drift tunable).

strategy.py     Indicators (ATR, Donchian, Keltner, Bollinger, the AdaHedge
                online-learned channel and direction, Kaufman ER,
                ADX, Ehlers CTI, Choppiness, variance ratio), the
                StrategyTemplate switches, and a bar-by-bar backtest
                engine (ATR position sizing, leverage cap, costs, next-bar
                fills, same-bar stop, close-of-bar mark-to-market). The
                bar loop is Numba-compiled (pure-Python fallback if numba
                is missing) and indicator arrays are cached per slice.

generator.py    Builds families of templates (quick / default / full)
                and the lattice parameter grid for each one.

walkforward.py  Rolling or anchored walk-forward optimizer with embargo,
                plateau parameter selection, Pardo's Walk-Forward
                Efficiency and acceptance criteria, and the walk-forward
                matrix (train/test length robustness).

robustness.py   Lopez de Prado toolkit: trials matrix, Combinatorial Purged
                CV, CSCV Probability of Backtest Overfitting, Probabilistic
                and Deflated Sharpe, effective number of trials, minimum
                backtest length, stationary-bootstrap p-values, White's
                Reality Check, Hierarchical Risk Parity.

portfolio.py    Candidate filter (Sharpe / windows / Pardo), greedy or
                cluster-based subset selection, equal or HRP weights,
                and the NESTED walk-forward of the selection step.

pipeline.py     The research orchestration main.py and etf_dashboard.py share:
                the pool workers (slot evaluation, walk-forward matrix cells),
                family diagnostics, static + nested portfolios, finalist
                statistics, the buy-and-hold benchmark, and loading real data
                without its forming bar. Every number both entry points
                report is computed here, once.

main.py         Single-asset research run: everything in a process pool,
                then the report. `main(argv)` returns what it computed.

live.py         The live layer: current position and stop levels read out of
                the engine, the next bar's orders and their order types,
                re-optimization on the walk-forward's cadence, dropping the
                forming bar, and per-asset target positions / trade list.
etf_dashboard.py  `research` (multi-asset pipeline -> portfolio.json + verdict)
                and `signals` (apply it -> dashboard.html + CSVs).
dashboard_html.py  Renders that into one self-contained HTML page: no CDN,
                no font file, inline SVG charts, light and dark.

tests/          unittest suite: no look-ahead, costs, stops, CPCV path
                coverage, PBO on noise vs. signal, DSR, bootstrap, HRP,
                the annualization contract, the live order predictions
                against the engine, the dashboard round trip, main.py end
                to end (incl. a --jobs 2 run) and its parity with the
                dashboard's research, the data loader against a fake yfinance.
```

## The strategy templates

A **template** is a fixed combination of categorical switches:

| switch | values |
|---|---|
| `direction_logic` | `trend` (trade with the break) / `countertrend` (fade it) / `learned` (the online learner decides bar by bar, see below) |
| `channel_type` | `donchian` / `keltner` (EMA +/- k ATR) / `bollinger` (SMA +/- k sd) / `hedge` (online-learned, see below) |
| `entry_style` | `stop` (at the level) / `close_confirm` (close beyond, next open) / `pullback` (limit k ATR inside the level) |
| `exit_style` | `channel` (Turtle exit; midline target for countertrend) / `atr_trail` / `target_stop` / `time_stop` -- a hard ATR stop is always on |
| `regime_indicator` | `er` Kaufman Efficiency Ratio / `adx` / `cti` Ehlers Correlation Trend / `chop` Choppiness / `vr` variance ratio |
| `regime_filter` | `none` / `trend_only` / `range_only` (Ranger's "sideways" mode) |
| `vol_filter` | skip entries when ATR is in an extreme percentile |
| `bias_filter` | `sma`: longs only above SMA(200), shorts only below (financial-hacker's market-direction filter) |
| `sides` | `both` / `long_only` / `short_only`. Quick and default families are two-sided; restrict them from the command line (`--sides long_only`) because whether an asset has a drift is a property of the asset, not something to let the selection step data-mine. `full` carries all three. |

Its *numeric* parameters (lookbacks, ATR multiples, thresholds...) are
re-optimized every walk-forward window from a small lattice grid
(`generator.param_grid_for`), so "the strategy" in the final portfolio
is the template plus a time-varying parameter set chosen only from
information available up to that point.

### The `hedge` channel: an online-learned alternative to fitted lookbacks

Donchian, Keltner and Bollinger channels all carry a lookback (and a
width) that the walk-forward has to re-fit every window, and that
choice is the single biggest source of curve-fitting in a breakout
system. The `hedge` channel removes it with an **online learning**
algorithm from the prediction-with-expert-advice literature:

- **Experts**: Donchian channels with a fixed, log-spaced ladder of
  lookbacks (`HEDGE_LADDER` = 10, 20, 40, 80 bars). The ladder spans the
  scales; it is not a tuned parameter.
- **Loss**: every bar each expert is scored on the ATR-normalised next
  move of the stance it implied (new n-bar high -> long, new n-bar
  low -> short, hold otherwise). For a `countertrend` template the
  stance is the *fade*, so the learner rewards the lookback whose
  breakouts are most **anti-correlated** with the following move.
- **Learner**: **AdaHedge** (de Rooij, van Erven, Grunwald, Koolen,
  JMLR 2014), exponential weights whose learning rate is set from the
  accumulated mixability gap. It starts as plain **follow-the-leader**
  and only becomes more conservative when the data forces it to. No
  learning rate, no threshold. It scores the last `HEDGE_MEMORY` (250)
  bars: plain AdaHedge finds the best expert *in hindsight* over all
  history, so after a long trend a fade expert would have to pay back
  the whole history before it could win; the bounded memory is what
  lets the learner *track* a change of regime, and it is also what
  makes the learner state reproducible from the walk-forward's warm-up
  buffer (see `walkforward.window_backtest`).
- **Channel**: the weight-averaged expert channel, i.e. an adaptive
  channel whose effective period is learned causally bar by bar. The
  exit channel uses the same weights over the ladder scaled by
  `HEDGE_EXIT_SCALE` (Turtle 20/10 style).
- **Learned direction** (`direction_logic = "learned"`, any channel):
  the ladder doubles to a *follow* and a *fade* expert per lookback and
  the net side weight decides, bar by bar, whether the template follows
  or fades the break. A trade keeps the exit logic of the side it was
  opened under. On a series that alternates trend and mean-reversion
  regimes the direction flips to fade within the learner's memory of
  the range starting and back to follow in the next trend.

`generator.param_grid_for` drops `n_entry` and `n_exit` for hedge
templates: with a `channel` exit and no regime filter the grid is empty
and the walk-forward has nothing left to fit. The `online` family is
288 such templates; `full` includes them alongside the fitted ones.

Why this and not the other online-learning candidates:

- **Cover's universal portfolio, Anticor (Borodin et al. 2004), OLMAR,
  PAMR** are *multi-asset portfolio* algorithms: they need a cross-
  section to rebalance between. On a single instrument they degenerate.
  Anticor's idea (bet on lagged cross-correlation) survives here only as
  the sign flip that makes the countertrend learner score the fade.
- **Plain Hedge / exponentiated gradient** needs a learning rate, and
  **fixed-share** needs a switching rate: parameters again. AdaHedge is
  the parameter-free variant with the same regret guarantee.
- **Follow-the-leader** on its own is unstable on noisy losses (it flips
  between near-tied experts); AdaHedge *is* FTL until the losses show
  the flipping costs something, then smooths.

What it does and does not buy you. On a synthetic "long bull market
with occasional sharp reversals" series the hedge family has a higher
median OOS Sharpe and more profitable templates than the Donchian
family, and lower drawdowns, but the *best* template and the nested
walk-forward portfolio are no better (worse on some seeds). The learner
removes one degree of curve-fitting; it does not add an edge the experts
do not have.

## How the evaluation is made robust

The naive pipeline (optimize on a training window, apply to the next
one, stitch, pick the best templates, report their Sharpe) has two
holes. The parameter choice inside each window is a single lucky-or-
unlucky path, and the *selection* of templates on their "out-of-sample"
curves makes those curves in-sample again. Every step below targets one
of those.

**Per template (walk-forward level)**

- **Plateau selection** instead of peak picking: the grid point whose
  lattice neighbours score best, not the single best point (Pardo;
  financial-hacker's "smooth heatmap" criterion).
- **Pardo's Walk-Forward Efficiency** (annualized OOS return / annualized
  IS return) and his acceptance rule: WFE >= 0.5, majority of profitable
  OOS windows, OOS profitable overall (`--require-pardo` to enforce).
- **Walk-forward matrix**: the same template over 12 train/test-length
  combinations. Robust means profitable in most cells.
- **Rolling or anchored** windows, optional **embargo** gap.
- **Combinatorial Purged CV** (AFML ch. 12): the history is cut into
  groups, every choice of test groups is a train/test split (with embargo
  after each test group), and the results are stitched into
  C(N,k)·k/N complete backtest paths. You get a *distribution* of OOS
  Sharpe rather than one number. A template whose single walk-forward
  path is a star while its CPCV paths straddle zero was lucky.

**Per family (selection level)**

- **Probability of Backtest Overfitting** via CSCV (AFML 11.6) over
  *every* parameter trial the generator ran, and again over the
  template-level OOS curves. Near 0.5 or above means the best-in-sample
  choice is a coin toss out-of-sample. Also reports the OOS-vs-IS
  degradation slope and P(OOS loss | IS best).
- **White's Reality Check** for the best template: is the best of the
  family better than a zero-return benchmark once you account for having
  searched the family? Stationary bootstrap keeps autocorrelation and
  cross-correlation intact.
- **Deflated Sharpe Ratio** (AFML ch. 14) with the **effective number of
  trials**: templates are clustered on return correlation and the DSR
  benchmark is the expected max Sharpe of that many independent noise
  trials. The raw DSR (all templates independent) is printed alongside.
- **Minimum backtest length** implied by the number of trials.

**Per strategy (finalists)**

- **Stationary-bootstrap p-value** of the OOS Sharpe (financial-hacker's
  "Montecarlo Reality Check": p < 5 % good, > 15 % walk away).

**Portfolio**

- Static selection (greedy under a correlation ceiling, or best-of-each
  correlation cluster), equal weights or **Hierarchical Risk Parity**.
- **Nested walk-forward selection**: at every window boundary the
  candidate filter, subset and weights are recomputed from the OOS
  history realised *so far* and held for the next window. That curve is
  out-of-sample with respect to both the parameters and the selection,
  and it is the only portfolio number in the report worth quoting.

On the default synthetic data (mostly noise) the toolkit says so: PBO
about 0.6, Reality Check p about 0.95, DSR far below 0.95, and the
static portfolio's Sharpe of 0.6 turns into -0.5 once the selection is
walked forward. With `--trend-prob 0.8 --trend-drift 0.002` (a real
edge) the finalist passes all 12 walk-forward matrix cells, its
bootstrap p-value is 0, and the nested portfolio keeps a Sharpe near 1.

## Bugs fixed in the third review pass (walk-forward and the bar loop)

- The out-of-sample window was backtested from a cold start `warmup_bars`
  before it, and that buffer was allowed to TRADE. A position opened on the
  buffer was chosen by parameters fitted on exactly those bars, and its P&L
  inside the window was counted as out-of-sample (5 % of OOS bars and 8 % of
  OOS trades on the synthetic default). `backtest()` now takes
  `first_trade_bar`: the buffer only forms indicators, the first trade is on
  the window's first bar, and `walkforward.window_backtest` is the one way
  both the training and the test window are run. Training windows get the
  same warm-up, so a grid point with a 60-bar channel is no longer scored on
  fewer bars than one with a 20-bar channel, and `live.refit_params` calls
  the same optimizer as the walk-forward loop.
- The warm-up was far too short for exponentially smoothed indicators: an
  EMA carries `(1-alpha)^t` of its arbitrary seed forever, and after the old
  buffer a Keltner channel or a Wilder ADX still differed from the value a
  trader with full history sees (the ADX by 27 % at the first bar of a test
  window). `warmup_bars` now budgets the settle time of the EMA.
- Pardo's WFE annualized the COMPOUNDED total return of the whole OOS
  history and divided it by the mean per-window IS return, so it grew with
  the length of the history alone: a flat 10 %/y in-sample and out-of-sample
  reported a WFE of 1.6 over ten years and 6 over thirty. It is now the mean
  of the per-window annualized OOS returns over the mean of the IS ones.
- With a trend template's channel exit, a bar that traded through both the
  channel and the hard stop always filled at the stop, although the channel
  (the nearer level) is hit first on the way through; 4.6 % of channel-exit
  trades on the synthetic default were filled at the wrong, further level.
- The indicator cache was keyed on sampled closes only, so two frames with
  the same closes and different highs and lows shared one (wrong) ATR and
  channel. High and Low are now part of the key.

## Bugs fixed in the second review pass

- An open position was left completely unmanaged whenever the previous bar's
  indicators were not fully formed or ATR(n) was zero: the bar loop skipped
  the whole exit block, so the hard stop, the target and the trailing stop all
  went quiet. Real data has flat stretches that drive ATR(n) to exactly zero,
  and in the regression fixture a trade sized to risk 1% of equity rode one of
  them straight into a gap and lost 13.5%. Exits are now honoured on every bar;
  only NEW business waits for formed indicators.
- The ATR chandelier ignored the entry bar's own extreme, so a large favourable
  move on the entry bar was never locked in (the fixture gave the whole spike
  back and turned a winner into a loser).
- `portfolio._qualifying` could name a template with no column in the aligned
  returns frame, raising KeyError for any `--min-sharpe <= 0`.
- `hrp_weights` divided by zero on a never-traded (zero-variance) strategy, and
  the resulting NaN cluster variance silently degraded the bisection to a 50/50
  split -- handing a dead strategy a quarter of the book.
- `load_yfinance` returned an empty frame on a wrong symbol, a rate limit or no
  network, which surfaced much later as an obscure IndexError.
- `warmup_bars` budgeted n bars for ADX, which is smoothed twice and needs ~2n.
- `PERIODS_PER_YEAR` was imported by value into four modules and baked into two
  default arguments, so the project could not be run on anything but daily bars.
  `strategy.set_periods_per_year()` is now the single source of truth, read at
  call time and set in the worker processes too.

## Bugs fixed relative to the first version

- Trailing stop was updated with the current bar's high *before* being
  tested against the current bar's low (intrabar look-ahead).
- Equity was marked to market with the *previous* close and realised
  P&L landed one bar late; the last bar merged two days of returns.
- Every template waited for a 100-bar volatility-percentile warm-up
  even when it had no vol filter, wasting a fifth of each training
  window. Warm-up now depends on the indicators actually used.
- Countertrend + opposite-channel exit bought a 40-bar low and exited
  on the next 15-bar low, i.e. almost immediately at a loss. The
  countertrend channel exit is now the mean-reversion midline target.
- No transaction costs at all (README claimed a fixed penalty).
  `cost_bps` per side is charged on notional, default 5 bps.
- No leverage cap on ATR position sizing (`max_leverage`).
- Stops were not checked on the entry bar.
- Windows with too few in-sample trades were silently dropped from the
  stitched OOS curve, shortening it; they are now flat.
- The last partial test window was never evaluated.
- Portfolio candidates' Sharpes were computed on zero-padded returns
  over the union of dates; the frame is now aligned on the common dates.
- Pullback limit orders expired after one bar (now `pullback_valid_bars`).

## Caveats

This is a research framework, not a production trading system. The
cost model is a flat bps charge, position sizing is fixed-fractional on
an ATR stop, and the synthetic data is a toy regime-switching random
walk. Results on synthetic data are a pipeline check. Real conclusions
need real data, realistic costs for the instrument, and -- as the
Reality Check numbers make painfully clear -- a lot more history than
a decade of daily bars for a family of hundreds of trials.

## Extending this

- **Multi-asset portfolios**: done, in `etf_dashboard.py research` -- every
  `(asset, template)` pair is one slot in a single candidate pool, so the
  portfolio step diversifies structurally and across markets at once, closer
  to RangerZ's basic portfolio.
- **More switches**: add an indicator to `REGIME_INDICATORS` or a new
  entry/exit branch in `strategy.backtest`, then list it in
  `generator.FAMILIES`.
- **More experts for the hedge channel**: `HEDGE_LADDER` can hold any
  set of Donchian lookbacks; Keltner / Bollinger widths, or a fade expert
  that only acts inside a range regime, could join the ladder (the
  mixture stays causal and parameter-free as long as the ladder is fixed
  in advance).
- **Meta-labelling** (AFML ch. 3): use the template signals as primary
  models and train a classifier on the triple-barrier outcome to size
  or veto trades.
- **Speed**: the bar loop is already Numba-compiled (a 3000-bar backtest
  takes ~1.5 ms, of which the loop itself is a fraction; the rest is
  indicator lookup and result packaging). The remaining cost centres
  for very large families are the family-level Reality Check and the
  silhouette search in `effective_n_trials`, both O(templates^2) or
  O(templates x bootstraps); reduce `--n-boot` or cluster on a subsample
  if you go beyond a few thousand templates.
