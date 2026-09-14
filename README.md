# Strategy Generator

An original Python implementation of the *idea* behind Robert Pardo's
Ranger / RangerZ system (as described in "Evaluating Robert Pardo's
Ranger System", financial-hacker.com): not one strategy, but a
**generator of structurally distinct breakout strategies**, each
evaluated with **walk-forward analysis**, combined into a **portfolio
of low-correlation strategies**. Pardo's actual EasyLanguage code is
proprietary and was not used or referenced here -- this is a from-scratch
build of the same well-known concepts (Donchian breakout, walk-forward
analysis, correlation-based portfolio construction).

## Setup Instructions

This project uses Conda to manage dependencies. To set up the project locally, follow the instructions below to create a virtual environment with the correct Python version and libraries.

1. **Create a new conda environment:**
   This command creates a new local environment in the `./env` directory using Python 3.12 and installs the required data science libraries (`pandas`, `scikit-learn`, and `scipy`).

   ```bash
   conda create -p ./env python=3.12 pandas scikit-learn scipy matplotlib yfinance
   ```

2. **Activate the environment:**

   ```bash
   conda activate ./env
   ```

## Architecture

```
data.py         Load real data (yfinance) or generate synthetic
                regime-switching OHLC for offline testing.

strategy.py     StrategyTemplate: the "switches" that make each
                variant structurally different (direction logic,
                entry style, exit style, regime filter, vol filter),
                plus a bar-by-bar backtest engine (ATR position
                sizing, next-bar fills, no look-ahead).

generator.py    Builds the full family of templates (Cartesian
                product of the switches) and the numeric-parameter
                search grid to walk-forward optimize for each one.

walkforward.py  Rolling walk-forward optimizer: train window ->
                pick best params -> apply out-of-sample on the next
                window -> roll forward -> stitch OOS equity.

portfolio.py    Correlation-based greedy selection: filters weak
                candidates, then builds the largest-Sharpe subset
                whose pairwise correlation stays under a ceiling.

main.py         Runs the whole pipeline and writes a report.
```

## Running it

```bash
python main.py
```

By default it runs on **synthetic regime-switching data** (no
internet needed) so you can see the pipeline work end to end
immediately. Outputs land in `/mnt/user-data/outputs/`:

- `equity_curves.png` -- every template's out-of-sample equity
  (grey) vs. the selected portfolio (black) and its components
- `correlation_heatmap.png` -- pairwise correlation of qualifying
  strategies' daily OOS returns
- `template_ranking.png` / `.csv` -- every generated template ranked
  by out-of-sample Sharpe
- `report.md` -- written summary

## Running on real data

```python
# in main.py
USE_REAL_DATA = True
TICKER = "SPY"        # or any yfinance ticker: QQQ, GC=F, BTC-USD, ...
START_DATE = "2010-01-01"
```


## What's a "template" vs a "strategy"?

A **template** is a fixed combination of the categorical switches
(e.g. trend-following, stop entry, ATR trailing exit, no regime
filter). Its *numeric* parameters (breakout lookback, ATR multiples,
etc.) are **not** fixed -- they get re-optimized every walk-forward
window, exactly like Ranger/Pardo's process. So "the strategy" that
ends up in the final portfolio is really the template plus a
time-varying set of parameters chosen only from information available
up to that point.

## Extending this

- **More switches**: add new entry/exit/filter types in
  `strategy.py`'s `StrategyTemplate` and backtest logic, then add
  them to the lists in `generator.py`.
- **Multi-asset portfolios**: run the same pipeline per asset and
  merge all `(template, asset)` pairs into one big candidate pool for
  `portfolio.select_portfolio` -- this gives you both structural
  diversification (different switches) and market diversification
  (different assets) at once, closer to RangerZ's actual basic
  portfolio (DJI, S&P500, DAX, Gold, Bitcoin).
- **Smarter selection**: swap the greedy correlation-ceiling method
  in `portfolio.py` for hierarchical clustering (pick the best
  performer from each cluster) or a proper mean-variance /
  risk-parity weighting instead of equal-weight.
- **Monte Carlo reality check**: shuffle/bootstrap the OOS trade
  sequences to see how much of the portfolio's edge could be luck --
  this is what the article calls the final validation step before
  trusting a strategy family.
- **Speed**: the bar-by-bar loop in `strategy.py` is easy to read but
  not fast. For a much larger template/asset sweep, consider
  Numba-jitting the loop or porting the whole engine to Zorro/Lite-C
  (as Pardo and financial-hacker.com actually did) for the
  optimizing/analysis speed advantage they cite.

## Caveats

This is a demonstration framework, not a production trading system:
no transaction costs/slippage model beyond a fixed penalty is
included, position sizing is a simple fixed-fractional-risk model,
and the synthetic data is a toy regime-switching random walk, not a
real market. Treat results on synthetic data as a pipeline check, not
as evidence the approach is profitable -- that requires real data,
realistic costs, and out-of-sample validation on markets the
templates were never touched on.
