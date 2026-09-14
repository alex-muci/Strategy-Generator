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
conda create -p ./env python=3.12 pandas scikit-learn scipy matplotlib yfinance
conda activate ./env
python -m unittest discover -s tests -v      # 21 sanity tests
python main.py                                # synthetic data, 72 templates, ~2 min on 8 cores
```

## Running it

```bash
python main.py --help
python main.py --family quick                       # 72 templates (Donchian, ER filter)
python main.py --family default                     # ~770 templates, all switches sampled
python main.py --real SPY --start 2005-01-01 --family default --jobs 8
python main.py --real QQQ --start 2010-01-01 --train 500 --test 125   # rolling window (default): each window re-optimizes on the last 500 bars only
python main.py --real GC=F --train 750 --test 250 --anchored --selection best   # anchored: training (window expands from bar 0)
python main.py --trend-prob 0.8 --trend-drift 0.002  # synthetic data with a KNOWN trend edge
```

Outputs land in `./outputs/` (or `--out DIR`):

| file | what it shows |
|---|---|
| `equity_curves.png` | every template's OOS curve (grey), the selected ones, the static portfolio (black) and the **nested walk-forward portfolio** (red dashed) |
| `template_ranking.png/.csv` | every template ranked by OOS Sharpe, with WFE, % profitable windows, Pardo pass, CPCV mean and P(CPCV<0); red line = expected max Sharpe of that many noise trials |
| `cpcv_distribution.png` | for each finalist, the distribution of Sharpe over all combinatorial-CV paths vs. the single walk-forward path |
| `pbo.png` | CSCV logit histogram (Probability of Backtest Overfitting) and OOS-vs-IS degradation for the whole family |
| `wfa_matrix_*.png` | Pardo's walk-forward matrix (train x test lengths) for the top finalist |
| `correlation_heatmap.png` | correlation of qualifying templates' OOS returns |
| `selected_windows.csv` | per-window chosen parameters and IS/OOS stats of the finalists |
| `report.md` | written summary with all the numbers |

## Architecture

```
data.py         Load real data (yfinance) or generate synthetic
                regime-switching OHLC (trend probability / drift tunable).

strategy.py     Indicators (ATR, Donchian, Keltner, Bollinger, Kaufman ER,
                ADX, Ehlers CTI, Choppiness, variance ratio), the
                StrategyTemplate switches, and a bar-by-bar backtest
                engine (ATR position sizing, leverage cap, costs, next-bar
                fills, same-bar stop, close-of-bar mark-to-market).

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

main.py         Runs everything in a process pool and writes the report.
tests/          unittest suite: no look-ahead, costs, stops, CPCV path
                coverage, PBO on noise vs. signal, DSR, bootstrap, HRP...
```

## The strategy templates

A **template** is a fixed combination of categorical switches:

| switch | values |
|---|---|
| `direction_logic` | `trend` (trade with the break) / `countertrend` (fade it) |
| `channel_type` | `donchian` / `keltner` (EMA +/- k ATR) / `bollinger` (SMA +/- k sd) |
| `entry_style` | `stop` (at the level) / `close_confirm` (close beyond, next open) / `pullback` (limit k ATR inside the level) |
| `exit_style` | `channel` (Turtle exit; midline target for countertrend) / `atr_trail` / `target_stop` / `time_stop` -- a hard ATR stop is always on |
| `regime_indicator` | `er` Kaufman Efficiency Ratio / `adx` / `cti` Ehlers Correlation Trend / `chop` Choppiness / `vr` variance ratio |
| `regime_filter` | `none` / `trend_only` / `range_only` (Ranger's "sideways" mode) |
| `vol_filter` | skip entries when ATR is in an extreme percentile |
| `bias_filter` | `sma`: longs only above SMA(200), shorts only below (financial-hacker's market-direction filter) |

Its *numeric* parameters (lookbacks, ATR multiples, thresholds...) are
re-optimized every walk-forward window from a small lattice grid
(`generator.param_grid_for`), so "the strategy" in the final portfolio
is the template plus a time-varying parameter set chosen only from
information available up to that point.

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

- **Multi-asset portfolios**: run the pipeline per asset and merge all
  `(template, asset)` pairs into one candidate pool -- structural and
  market diversification at once, closer to RangerZ's basic portfolio.
- **More switches**: add an indicator to `REGIME_INDICATORS` or a new
  entry/exit branch in `strategy.backtest`, then list it in
  `generator.FAMILIES`.
- **Meta-labelling** (AFML ch. 3): use the template signals as primary
  models and train a classifier on the triple-barrier outcome to size
  or veto trades.
- **Speed**: the bar loop is pure Python (~8 us/bar). Numba-jitting it
  would make the `full` family (3168 templates) a coffee break.
