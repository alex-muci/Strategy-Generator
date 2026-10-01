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
# either with conda
conda create -p ./env python=3.12 pandas scikit-learn scipy matplotlib yfinance numba
conda activate ./env
# or, alternatively,
python -m venv env  # assuming 3.12 installed
./env/Script/Activate
pip install -r requirements.txt 

python -m unittest discover -s tests -t . -v # 382 tests (engine, order log, templates, hedge learner and its wide ladder, walk-forward, robustness, selection, data, live signals, replay, both entry points, spreads: shift invariance, point value, per-unit costs, margin cap, ruin, whole units, the ETF trick, a mixed cash + spread book)
# faster (about 1:35 min instead of 4.5): pip install -r requirements-dev.txt, then, with ./env active,
python -m pytest -n auto --dist loadscope   # same tests in parallel; loadscope keeps a class (and its one-off setup) on one worker
```

## Running it

```bash
python main.py --help
python main.py --family quick                       # 72 templates (Donchian, ER filter)
python main.py --family default                     # 768 templates, all switches sampled
python main.py --family online                      # 288 templates on the online-learned channel (no lookback to fit)
python main.py --family online_wide                 # 8 templates: the learned direction on a wider ladder, sized by its own position
python main.py --real SPY --start 2005-01-01 --family default --jobs 8
python main.py --real SPY --start 2005-01-01 --family quick --sides long_only   # one-sided family (an asset with a drift)
python main.py --real SPY --start 2005-01-01 --family quick --vol-target 0.1  # use vol-target rather than ATR-stop (see Position sizing section section)
python main.py --real QQQ --start 2010-01-01 --train 500 --test 125   # rolling window (default): each window re-optimizes on the last 500 bars only
python main.py --real GC=F --train 750 --test 250 --anchored --selection best   # anchored: training (window expands from bar 0)
python main.py --csv brent_z25z26.csv --point-value 1000 --margin-per-unit 3000 --cost-per-unit 15 --cost-bps 0 --max-leverage 0.5   # a futures spread from a file (prices through zero; see Futures and spreads)
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
| `run.json`, `data.csv`, `portfolio_returns.csv`, `selected_returns.csv` | the run, recorded for a replay (see below): every template with its per-window parameters, the two portfolios' selections and weights, the exact bars used, the series to check against |
| `replay/<target>/` | written by `python replay.py` (or `--replay`): the orders and trades behind the best template, the static portfolio and the nested portfolio, each verified against the run |
| `report.md` | written summary with all the numbers, including the **buy & hold benchmark**: Sharpe, CAGR and drawdown of simply holding the asset over the same OOS bars, the nested portfolio's beta and correlation to it and its information ratio (what is left after the asset's own drift is removed), and how many templates beat holding at all |

### Replaying a run: the orders and trades behind the numbers

The report quotes three curves -- the best template, the static portfolio and
the nested walk-forward portfolio -- and `replay.py` shows what they traded:

```bash
python replay.py --out outputs                     # best, static and nested
python replay.py --out outputs --which nested --jobs 4
python replay.py --out outputs --which best --template "TR-don-stop-chan-noreg-noV-noB"   # any template of the run
python main.py --real SPY --replay all             # run, then replay, in one go
```

Nothing is re-optimized: the replay re-runs only the (template, window)
pairs a target is made of, with each window's chosen parameters and the same
`window_backtest` call the walk-forward used, on the bars saved in
`data.csv` (a fresh yfinance download is a different backtest). It is exact
by construction and takes seconds. Every target is **verified first**: the
replayed per-bar returns must equal the stored series to the last digit, the
Sharpe must match, the trade count must match, per window. `check.json`
holds the verdict and the numbers; read the lists only when it says `ok`.

Per target, in `outputs/replay/<target>/`:

| file | what it holds |
|---|---|
| `trades.csv` | every executed trade: template, walk-forward window, entry/exit dates and prices, side, shares, exit reason, P&L, costs; a position still open at a window's end is listed too (`closed = False`, `reason = open_at_window_end`: it is marked to market and dropped there, see "Known approximations"); for the portfolios, the template's weight and (nested) the selection period |
| `orders.csv` | the order lifecycle, top to bottom: what was **working** on which bars and at what level (an entry stop at the channel, a fade limit, a resting pullback limit with its expiry, the stop in force -- hard, opposite channel or trailing -- and the target), collapsed into one row per stretch of bars at the same level, plus every submission, fill, expiry and cancellation on its own date |
| `order_events.csv` | the raw per-bar order log the lifecycle rows were built from |
| `returns.csv` | replayed vs stored per-bar returns and their difference |
| `check.json` | the verdict and the numbers behind it |

The order log is read out of the engine's own bar loop
(`backtest(..., log_orders=True)`, off by default and free when off), not
rebuilt from the rules, so it cannot drift from what the engine did. The
fills follow the engine's conventions: a stop or limit fills at its level, or
at the open when the bar gapped through it; a `close_confirm` entry and a
time exit are orders at the open; every stop is already working on the entry
bar. What the portfolios trade is exactly the trades of their selected
templates in the windows they were selected for, scaled by the weights: a
nested period is one walk-forward window, so its trades are those windows'.


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

Those times are New York's (`CRON_TZ=America/New_York`, or convert them). A
daily bar counts as closed 15 minutes after the 16:00 New York bell
(`live.SESSION_CLOSE`); any run from then until the next open sees the same
bars, so the European morning works as well.

Add `--interval 1h` to both phases for hourly bars; every annualized statistic
follows the bar frequency (`strategy.BARS_PER_YEAR`).

What the dashboard shows, in the order you need it:

| | |
|---|---|
| **verdict** | the research conclusion, restated every run, so a weak result cannot quietly become habit |
| **exposure** | gross, net, and the money at risk if every stop fills at once; `--max-gross` scales the whole book down proportionally -- targets, positions and the next bar's entry orders alike |
| **trades to send** | target minus held, per ETF. Put your real broker positions in `state/holdings.json` (`{"SPY": 120, "TLT": -50}`) or it assumes the last run's orders were filled |
| **orders to work** | the level AND the order type for the next bar -- a breakout entry is a stop order, fading one is a limit order -- with share counts from the engine's own sizing rule |
| **positions** | entry, current stop, and how much of the original stop distance is left |
| **track record** | the nested walk-forward curve against simply holding the same ETFs |
| **slots** | current parameters and when they are next re-optimized |
| **diagnostics** | PBO, Reality Check, DSR, and the whole search you picked from |

Two rules keep the live path honest, both in `live.py`:

- **Parameters only change at window boundaries.** Re-optimizing every run would
  be a different strategy from the one the walk-forward measured with
  `test_bars`-long parameter holds. `due_for_refit` enforces the same cadence,
  also for a slot whose last fit found nothing (it stays flat for the window,
  as in the walk-forward). The state after a refit is the walk-forward's
  out-of-sample window reproduced exactly: flat on the first bar after the
  refit, so a position opened on the training bars is closed at the refit, as
  the walk-forward does. With `--anchored` the signals phase loads the history
  from the research's `--start`, so the refit sees the same expanding window.
- **The forming bar is dropped.** A feed queried at 11:15 returns an 11:00 bar
  built from 15 minutes of trading; its high, low and close all still move, so
  acting on it is a decision you could not have taken. A daily bar is dropped
  until its session has closed, by the exchange's clock and not by the date
  it is stamped with. Every index is tz-naive (`data.load_yfinance`): intraday
  bars in UTC, so the page shows their times in UTC, and daily bars on their
  exchange-local date.

Everything the dashboard reports about the *current* position (side, size, stop,
target, trailing anchor, resting order) is read out of the same
`strategy.backtest` loop the research validated, not reimplemented next to it.
The one thing `live.py` must restate is the *next* bar's order levels, since that
bar does not exist yet -- so `tests/test_live.py` rolls one real bar forward
across the template family and asserts the engine filled exactly what the
dashboard published, at the price that order type implies.

Nothing in this repository places an order. It tells you what to place.

### Trading it with futures

The signals stay on ETFs and the orders go to micro (or small) futures:

```bash
python etf_dashboard.py research \
    --assets SPY QQQ IWM FEZ IEF TLT EXHD.DE IITB.MI VIXY GLD SLV PPLT USO UNG \
             CORN WEAT SOYB CANE FXE FXY FXB FXA --start 2012-06-01 \
    --family online --vol-target 0.15 --max-leverage 6 --portfolio-vol 0.15 \
    --sides-map SPY=long_only QQQ=long_only IWM=long_only FEZ=long_only --max-strategies 10
echo '{"FGBL": 129.55, "FBTP": 118.20}' > state/futures_quotes.json   # keep current
python etf_dashboard.py signals --account-equity 200000 --max-gross 5 --futures
```

The universe (`futures_map.CONTRACTS`), one Yahoo ETF with a long history per
market and the contract that expresses it:

| market | ETF (since) | contract | priced off |
|---|---|---|---|
| S&P 500, Nasdaq-100, Russell 2000, Dow | SPY, QQQ, IWM, DIA | MES, MNQ, M2K, MYM | `=F` |
| EURO STOXX 50 | FEZ (2002) | FSXE micro, EUR 1 x index (Eurex) | `^STOXX50E` cash index |
| US Treasuries 2y, 5y, 10y, bond | SHY, IEI, IEF, TLT | ZT, ZF, ZN, ZB | `=F` |
| German Bund | EXHD.DE (2003, EUR) | FGBL, EUR 100k (Eurex) | `futures_quotes.json` |
| Italian BTP | IITB.MI (2012, EUR) | FBTP, EUR 100k (Eurex) | `futures_quotes.json` |
| VIX | VIXY (2011) | VXM mini, $100 x VIX (Cboe) | `^VFTW1`, the front month's end-of-day TWAP |
| gold, silver, copper | GLD, SLV, CPER | MGC, SIL, MHG | `=F` |
| platinum, palladium | PPLT, PALL (2010) | PL (50 oz), PA (100 oz) | `=F` |
| WTI crude, natural gas | USO, UNG | MCL, MNG | `=F` |
| corn, wheat, soybeans | CORN (2010), WEAT, SOYB (2011) | MZC, MZW, MZS micro, 500 bu (CBOT) | `ZC=F`, `ZW=F`, `ZS=F` |
| sugar | CANE (2011) | SB, 112,000 lb (ICE) | `SB=F` |
| EUR, GBP, AUD, CAD, JPY | FXE, FXB, FXA, FXC, FXY | M6E, M6B, M6A, MCD, MJY | `=F` |

- **Why ETFs for the signals.** Long dividend-adjusted histories (one shared
  bar index), and for the ETFs that hold futures themselves -- USO, UNG, VIXY,
  the Teucrium grains, CANE -- the *rolled* return a futures trader earns: the
  roll schedule, the contango, all of it is in the price you fit on. Yahoo's
  `=F` series are unadjusted splices with a gap at every roll; `futures_map.py`
  uses them only as the price that turns dollars into contracts. Every asset
  added to `--assets` shortens the shared history to its own, so a young ETF
  costs years (IITB.MI starts in 2012, the Teucrium funds in 2010-11).
- **Euro listings.** There is no dollar ETF for the Bund or the BTP, so those
  come from Xetra and Milan (`futures_map.LISTINGS`: currency and closing
  time). They are loaded in euros and restated in dollars at *today's* rate --
  the returns stay the local ones the future pays, the notional is in dollars,
  which is what a hedged share class shows. A euro future is hedged the same
  way: its P&L accrues in euros and only the variation margin is exposed. FEZ
  is the unhedged dollar ETF, so its signal carries EUR/USD on top of the index
  (the vol ratio absorbs part of that); `EXW1.DE` is the euro alternative.
- **Contracts Yahoo has no series for** (Eurex FGBL, FBTP) take their price from
  `state/futures_quotes.json`, which you keep current: `{"FGBL": 129.55}` or
  `{"FGBL": {"price": 129.55, "hedge_ratio": 0.85}}`. A file older than three
  days is flagged on the page. Without a price the contract is left out and the
  note says so. The Bund's hedge ratio is estimated against S&P's rolled
  Euro-Bund futures index (`^SPEUBDP`) when Yahoo serves it, the BTP's starts
  from a duration guess (`default_ratio`); both can be pinned in the file.
- **`--sides-map`** sets the sides per asset. Whether a market drifts is a fact
  about the market, decided before the run; the others keep `--sides`.
- **`--portfolio-vol`** is the book's volatility target. `--vol-target` gives
  every *slot* the same risk while it is in a trade, but slots are flat much of
  the time and diversify each other, so the book runs far below it. Research
  measures the realized volatility of the nested walk-forward curve and stores
  `risk_scale = target / realized` in `portfolio.json`; the signals phase sizes
  every slot on `account equity x risk_scale`. That is exact as long as
  `--max-leverage` does not bind, which is why it is raised here, and drawdowns
  scale with it. It is frozen by research, never re-estimated in the morning.
- **`--futures`** restates the book in whole contracts: ETF notional x hedge
  ratio / contract value, rounded to nearest. The hedge ratio is ETF volatility
  over futures volatility (120 bars, tails winsorized so a roll gap does not move
  it): 1 for SPY and MES, about 1.5 for TLT and the T-bond future. A held count
  is kept while the target stays within 0.6 of a contract, so a target drifting
  around 2.5 does not trade every morning. Stops and entry levels are restated
  as futures prices, and the page reports what the rounding left untracked. Put
  your real positions in `state/holdings_futures.json` (`{"MES": 2, "ZN": -1}`).
  A held contract whose ETF is no longer in the portfolio gets a closing order;
  a root the table does not know is flagged in the notes.
  `--max-gross` is a notional cap: FX and rates legs are large notionals with
  small risk, so a futures book needs it well above 1.
- Judge a contract by the dollars it moves in a year, not by its notional: a
  10-year note future is $105k of notional and about the risk of one micro S&P.
  There is no micro Bund or BTP: at EUR 100k face (~$150k) a $200k account's
  bond legs round to zero or one contract, and the page's rounding line shows
  it. **Check multipliers and ticks against the exchange before trading**; a
  quote that would make a contract worth an implausible amount is refused
  rather than sized on.
- VIXY is a rolled long position in the first two VIX futures, so its dollars
  map about one-to-one onto futures notional (`default_ratio` 1). It is priced
  off `^VFTW1`, Cboe's end-of-day TWAP of the front month -- the price the
  contract trades at, not spot VIX, which is far more volatile than any future.
  The Teucrium grains hold the 2nd, 3rd and a deferred contract, never the
  front; the vol ratio against the front-month splice absorbs most of that.
- The research traded the ETF in its cash session. A futures stop that fills
  overnight is a fill the backtest never had; a price-conditional order on the
  ETF is the faithful implementation.

## Architecture

```
data.py         Load real data (yfinance), a local OHLCV CSV (a spread,
                prices through zero) or generate synthetic
                regime-switching OHLC (trend probability / drift tunable).

strategy.py     Indicators (ATR, Donchian, Keltner, Bollinger, the AdaHedge
                online-learned channel and direction over a fixed ladder
                of experts -- Donchian rungs, or those plus Keltner bands,
                sized by the learner's own position -- Kaufman ER,
                ADX, Ehlers CTI, Choppiness, variance ratio), the
                StrategyTemplate switches, and a bar-by-bar backtest
                engine (ATR-stop or volatility-target position sizing,
                leverage cap, costs, next-bar fills, same-bar stop,
                close-of-bar mark-to-market). The
                bar loop is Numba-compiled (pure-Python fallback if numba
                is missing) and indicator arrays are cached per slice.

generator.py    Builds families of templates (quick / default / online /
                online_wide / full) and the lattice parameter grid for
                each one.

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
futures_map.py  ETF -> futures contract table (US, Eurex, Cboe, ICE), euro
                listings, hand-kept quotes, volatility hedge ratio, whole-
                contract rounding with a no-trade buffer, levels on the future.
dashboard_html.py  Renders that into one self-contained HTML page: no CDN,
                no font file, inline SVG charts, light and dark.

tests/          unittest suite: no look-ahead, costs, stops, CPCV path
                coverage, PBO on noise vs. signal, DSR, bootstrap, HRP,
                the annualization contract, the live order predictions
                against the engine, the dashboard round trip, main.py end
                to end (incl. a --jobs 2 run, and the pool's data hand-off
                under the spawn start method Windows uses) and its parity with
                the dashboard's research, the data loader against a fake yfinance.
```

## The strategy templates

A **template** is a fixed combination of categorical switches:

| switch | values |
|---|---|
| `direction_logic` | `trend` (trade with the break) / `countertrend` (fade it) / `learned` (the online learner decides bar by bar, see below) |
| `channel_type` | `donchian` / `keltner` (EMA +/- k ATR) / `bollinger` (SMA +/- k sd) / `hedge` (online-learned, see below) / `hedge_wide` (the same learner over a wider ladder, sized by its own position, see below) |
| `entry_style` | `stop` (at the level) / `close_confirm` (close beyond, next open) / `pullback` (after the break, a limit k ATR from the level: back inside the channel when following, deeper beyond it when fading) |
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

### Position sizing

Size is decided once, at entry, and held to the exit. Two rules, chosen
from the command line (never tuned by the walk-forward):

* **ATR stop, fixed fractional** (default): units such that the loss at
  the `atr_mult_stop` ATR stop is `--risk-pct` of equity (1 %):
  `units = equity x risk_pct / (atr_mult_stop x ATR x point_value)`.
* **Volatility target** (`--vol-target 0.15`): the units whose dollar
  volatility is the target: `units = equity x target_bar / (sigma x
  point_value)`, where `target_bar` is the annualized target scaled to one
  bar and `sigma` the standard deviation of close-to-close **changes in
  price points** over `--vol-target-n` bars (60). The dollar vol wanted
  (target x equity) over the dollar vol of one unit (point value x sigma).
  On a share this is the familiar notional = equity x target / pct vol to
  within a few percent (sigma ~ pct vol x price); on a spread, whose pct
  vol does not exist, it is the only form there is. The ATR stop stays
  where it was, so the loss at the stop is now `atr_mult_stop x ATR x
  point_value x units` rather than `risk_pct` of equity.

A unit is a share (`--point-value 1`, the default) or a lot (`--point-value
1000` for Brent, 50 for ES). Neither rule divides by a price.

For a `learned` direction either rule is then scaled by the learner's
**conviction**: its net side weight in favour of the side the trade
takes, 0 to 1. A bar on which the follow and fade experts are near-tied
opens a small position, a one-sided book a full one, and a pullback
limit placed under a logic the learner has abandoned by the time it
fills opens nothing. On the `hedge_wide` ladder the conviction, and the
size of a `trend` or `countertrend` template too, is the magnitude of
the learner's own **position** (see the wide ladder below). The
dashboard's share counts carry the same scale.

Both are capped at `--max-leverage` x equity of **notional** (units x point
value x |price|), or, when `--margin-per-unit` is given, of **margin**
(units x margin per unit; then set `--max-leverage` at or below 1). Set the
target near the asset's own volatility and the strategies trade at about 1x notional
while in position, so `equity_curves.png` puts buy & hold on the same
axis as the strategies (the ATR rule leaves them on different scales and
the asset gets a secondary axis). Across assets the same target assigns
the same risk to every slot, which is what the multi-asset dashboard
wants. Two things keep a strategy's realized vol below the target: time
spent flat, and the leverage cap, which binds all the time on a quiet
asset (a 15 % target on a 5 % vol asset asks for 3x).

### Futures and spreads (prices at or below zero)

A futures calendar spread (Brent Dec25-Dec26, say) is quoted front minus
back and trades through zero. Nothing needs to be added to its prices to
run it here: the engine is **shift-invariant**. Every rule works on price
differences (channels, ATR, stops, targets, every regime indicator, the
learner's stances and losses, the P&L), so adding any constant to every
price, including one that makes the whole series negative, leaves the
trades, the P&L and the equity unchanged, and `tests/test_instrument.py`
proves it for every switch and both sizing rules. The three things that
used to read the price level -- the leverage cap, the volatility-target
size and the costs -- are described in currency instead:

| flag | meaning | Brent spread example |
|---|---|---|
| `--point-value` | currency per 1.0 of price per unit (lot) | 1000 (1000 bbl) |
| `--cost-per-unit` | commission + slippage per unit per side, in currency | 15 (a tick plus commission), with `--cost-bps 0` |
| `--margin-per-unit` | initial margin per unit, the leverage cap's basis | 3000 (check the exchange) |
| `--whole-units` | floor every size to whole contracts; below one, nothing opens | on, for any future |

Give all four for **any future**, not only a spread: a back-adjusted
continuous contract has positive prices at an artificial level, so a cost in
basis points of that price and a cap on its notional are both arbitrary
(both entry points warn when a point value comes without a margin or with
bps costs). `--cost-bps` is a fraction of a notional a spread does not have,
so a series with a price at or below zero is refused unless
`--margin-per-unit` is given and `--cost-bps` is 0
(`strategy.validate_instrument`, run by both entry points before any window,
and by every `backtest` call). A share cannot trade at or below zero:
`load_yfinance` drops such a bar as a bad print, except on a `--real`
run with `--margin-per-unit` (a future such as WTI on 2020-04-20, whose
negative settlement is a real gap). P&L is
`side x units x point_value x (price change)` on every bar, for a share
(point value 1) and a lot alike; `shares` in the trade lists are units.
Research trades fractional units unless `--whole-units` is set; the live
trade list always rounds to whole units, so a futures research run without
the flag can book P&L on 0.3 contracts that the live book never holds.
With the flag, `main.py` warns before the run when a typical entry sizes
below one unit on the 100,000 the research sizes on (every template would
sit flat); the live book sizes each slot on its share of the account, so a
small slot can floor to 0 where the research traded one lot.

**Ruin.** A close that leaves the equity at or below zero (a gap through the
stop beyond the margin) liquidates the position at that close (trade reason
`ruin`), floors the cash at zero and ends the run flat: one -100 % bar, then
0, never a negative equity whose returns flip sign. A margined future makes
this reachable, which is why `--max-leverage` should be 0.5 or less in
margin terms.

The data comes from a file: `--csv PATH`, the first column the date, then
`Open High Low Close [Volume]` in any case. Rows whose four prices are all
exactly 0 (a vendor's no-trade day) are dropped; a real 0.00 close is kept.
One listed spread has a year or two of liquid history, not enough for a
walk-forward, so stitch successive spreads (Z24-Z25, Z25-Z26, ...) with the
ETF trick in `extra_utils/ETF_trick_spreads.py`, run **in points**:
`point_value=1, contracts=1, side=+1, k0=0`, `roll_cost` in points. The
output is then the listed spread itself between rolls, shifted by a
constant (which does not matter), with each roll's gap and cost folded in;
give the multiplier to the engine once, as `--point-value`, and let the
engine take the short side itself. Use settlements as the close when the
vendor has them, and confirm the sign convention (front minus back) before
reading a `long` as a bet on backwardation.

The benchmark of such a run is not buy and hold (a spread has no return,
and the percentage change of a back-adjusted future's level is not the
contract's return): for a series that touches zero, or any run with
`--margin-per-unit`, it is the P&L of **holding one unit** on the initial
equity, an additive curve (summed, never compounded) on an arbitrary scale.
Its Sharpe, beta, correlation and information ratio are scale-free and read
as before; its CAGR column is the simple annual P&L and its drawdown the
deepest fall from a peak, both as fractions of the initial equity. A
calendar spread that stays positive over the whole sample and is run
without a margin still gets a percentage buy and hold: give the margin.
The walk-forward's exposure statistics measure a unit in margin when one is
given, else in notional at |price|, and sign it by the side.

**The dashboard.** `etf_dashboard.py research` takes a CSV file among its
`--assets` (named by its file stem, reloaded from the same path by
`signals`) and a per-asset instrument with `--instrument-map`, so a book can
mix shares and a spread:

```bash
python etf_dashboard.py research --assets SPY TLT brent_z25z26.csv \
    --instrument-map brent_z25z26=1000,15,3000 --whole-units --max-leverage 0.5 --family quick
python etf_dashboard.py signals --account-equity 200000
```

A mapped asset is its own instrument: a cost or margin left out of its
entry is 0 (`SPY=1` is a plain share, not a share with the run-wide Brent
margin). An asset with a margin in the map is costed per unit only (its
`cost_bps` is 0), and the book measures its exposure in **margin** (units x
margin per unit) while a share's stays notional; on a book with both the
tiles read `notional + margin`, the capital committed rather than a
notional exposure (a lot of Brent counts its margin, not its 80,000), and
`--max-gross` caps that measure. The chart's holding curve compounds each
share and sums each one-unit asset before averaging them. The ETF dashboard's futures mapping
(`--futures`) is the separate ETF-signalled-contract path and is unrelated.

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
  low -> short, for up to n bars after that break, flat when there was
  none -- bounded so the learner is an exact function of a fixed number
  of past bars and a warmed-up walk-forward window matches a
  full-history run), **net of the sides it traded** to get
  there, charged at the template's `cost_bps` in ATR units exactly as
  the engine charges them: a fast lookback that flips every week has to
  earn its turnover back before the learner trusts it. For a
  `countertrend` template the stance is the *fade*, so the learner
  rewards the lookback whose breakouts are most **anti-correlated**
  with the following move.
- **Learner**: **AdaHedge** (de Rooij, van Erven, Grunwald, Koolen,
  JMLR 2014), exponential weights whose learning rate is set from the
  accumulated mixability gap. It starts as plain **follow-the-leader**
  and only becomes more conservative when the data forces it to. No
  learning rate, no threshold. Two additions, both parameter-free:
  - **Discounting.** Losses and the gap decay with a *lifetime* `H`
    (`gamma = 1 - 1/H`) instead of counting in full and then vanishing.
    Plain AdaHedge's learning rate only ever falls; a discounted gap
    lets it climb back after a calm stretch, so it reflects *recent*
    surprise. A discounted deficit is bounded by `H` times the
    expert's shortfall per bar, whatever it lost before, so **no
    expert is ever written off for good**: one that starts winning is
    back in front after about `H ln 2` bars.
  - **A ladder of lifetimes** (`HEDGE_HORIZONS` = 20, 40, 80, 160
    bars, the lookback ladder doubled), one learner each, mixed on top
    by Vovk's aggregating algorithm with a unit learning rate (Bayesian
    averaging with likelihood `exp(-loss)`, discounted at the longest
    lifetime) that scores every learner on its own hedge loss, i.e.
    the memory is learned the same way the lookback is. When a regime
    breaks the short-memory learner has already moved and its lower
    loss tilts the mixture its way, so a new leader is in front within
    about twenty bars; in a stable regime the learners agree. The
    played weights stay a convex combination of the experts. (AdaHedge
    is the wrong rule at this level: the learners' losses are near-
    identical most of the time, its learning rate explodes and the
    mixture becomes follow-the-leader on rounding noise.)
  Everything is still restarted every bar over the last
  `HEDGE_MEMORY` (250) bars, which keeps the learner state an exact
  function of a fixed number of past bars, reproducible from the
  walk-forward's warm-up buffer (see `walkforward.window_backtest`).
  The window's edge now carries `gamma^250` of a bar's weight (e^-1.6
  at H = 160, e^-3 at H = 80) instead of all of it: a horizon, not a
  cliff. `strategy.hedge_diagnostics` returns, bar by bar, the expert
  weights, the losses they were scored on, each lifetime's learning
  rate, the meta learner's weights over the lifetimes (the short ones
  gaining is the learner shortening its memory) and the *surprise*
  (the played mixture's loss minus the best expert's).
- **Channel**: the weight-averaged expert channel, i.e. an adaptive
  channel whose effective period is learned causally bar by bar. The
  exit channel uses the same weights over the ladder scaled by
  `HEDGE_EXIT_SCALE` (Turtle 20/10 style).
- **Learned direction** (`direction_logic = "learned"`, any channel):
  the ladder doubles to a *follow* and a *fade* expert per lookback and
  the net side weight decides, bar by bar, whether the template follows
  or fades the break, and by how much it leads, how large the position
  is (see Position sizing). A trade keeps the exit logic of the side it
  was opened under. On a series that alternates trend and mean-reversion
  regimes the direction flips to fade within the learner's memory of
  the range starting and back to follow in the next trend.

`generator.param_grid_for` drops `n_entry` and `n_exit` for hedge
templates: with a `channel` exit and no regime filter the grid is empty
and the walk-forward has nothing left to fit. The `online` family is
288 such templates; `full` includes them alongside the fitted ones.

#### The wide ladder: `hedge_wide` and the `online_wide` family

The ladder is a list of rules fixed in advance, so anything that is a
rule can be an expert. `hedge_wide` runs the same learner, loss, memory
and lifetimes over `HEDGE_LADDERS["hedge_wide"]` (`strategy.HedgeExpert`
describes one expert):

- **The four Donchian rungs** of `HEDGE_LADDER`, unchanged.
- **A Keltner band at every rung**: SMA(n) +/- 2 ATR for the same four
  lookbacks (`HEDGE_WIDE_K`, the template default `channel_k`). A break
  of a band is a move of two volatility units from the mean rather than
  a new n-bar extreme, so the ladder spans *shape* as well as period:
  the 10-bar band fades are the fast "two ATRs below the two-week mean"
  dip buys, the 80-bar band follows are slow thrusts from a quarterly
  mean. The band uses Keltner's original SMA midline rather than the
  template's EMA: a rolling window is exact once full, so the warm-up
  contract below still holds, and an EMA never is. (The SMA is reduced
  window by window, `strategy._window_mean`, not with pandas' running
  sum, whose last bits depend on where the series started.)

Everything else carries over: no learning rate, no threshold, no
lookback or width in the grid (`param_grid_for` is as empty as for
`hedge`), the same 410-bar warm-up at the default 20-bar ATR (the 80-bar
rung still leads it: a band's stance is exact after its window, the
ATR's and its span), and a window warmed on `hedge_warmup(atr_n,
"hedge_wide")` bars matches a full-history run to 1e-12 (the test covers
both ladders, two-sided and one-sided; the ATR itself is pandas' running
mean, so "to the bit" would overstate it). The learner is O(experts) per
bar: a 16-expert `learned` run costs about twice the 8-expert one.

What is different on this ladder is not the experts but **how a template
is sized**, and it is different because the plain ladder's rule has two
failures on a real series that only show once a ladder is wide enough to
hold both kinds of expert:

- **Scoring the legs the template trades.** The experts are scored on
  the sides the template's `sides` switch lets it take
  (`strategy.hedge_scored_sides`). A two-sided template scores both legs,
  as before. A `long_only` one scores the follow experts as "buy new
  highs" and the fade experts as "buy new lows": the short legs are flat,
  so a fade expert is no longer dragged by the shorts of new highs the
  template never sells, and on an asset with a drift that drag was most of
  its score.
- **Sized by the committee's own position.** The plain ladder sizes a
  learned direction by its *net side weight*, follow minus fade. On one
  leg that is the wrong question: buying new highs and buying dips are two
  compatible trades, and in a bull market where both pay the net is a
  near tie and the template opens almost nothing. On the wide ladder every
  entry, learned or fixed direction, is scaled by the magnitude of the
  learner's **position** (`strategy.hedge_position`): the stance its
  weighted committee of experts holds at the close (weight times stance,
  +1 every expert long, 0 flat or split), times a **trade weight**. The
  trade weight is one more aggregation of the kind the learner already
  runs over its lifetimes: the mixture it actually played, scored on its
  own realised loss, against the neutral loss of cash, unit rate,
  discounted at the longest lifetime (`strategy._hedge_window`). It is the
  weight the aggregate puts on trading at all. It is not a cash expert on
  the ladder: a cash expert with a constant loss loses to the luckiest of
  sixteen noisy experts most of the time (a winner's curse; on a driftless
  random walk it held 1-4 % of the weight, and it was tried and dropped),
  whereas the played mixture's loss is one causal sequence, so on noise
  the trade weight averages a half (it is a sigmoid of a discounted P&L,
  so it wanders, sd about 0.25, and at 50 bps of cost it averages 0.35)
  and it falls only when what the learner played lost to standing aside.
  The docstring's "0.1 lower loss for 20 bars is 7:1" is the arithmetic,
  not the typical response: 0.1 of loss is 0.4 ATR a bar, a Sharpe of 6;
  a realistic edge moves the odds by about one unit over the window
  against noise of the same size. The unit rate and the longest lifetime
  are the rule's two conventions, borrowed from the meta learner. What is
  sized by is the *magnitude* of the position, a measure of how
  positioned the committee is and whether that has paid lately: near 1
  when the experts agree and have been paying, near 0 when they are
  split, flat, or losing to cash. The sign is not used: the template
  trades a break of its channel, which the committee, by construction, is
  not yet in (at the entries the committee's sign agrees with the trade's
  about two thirds of the time on a real series, and no better than a
  coin on the regime series two-sided), so the regret bound is about the
  committee's position, not about the trade this sizes. The direction of
  a `learned` template is still the sign of the net side weight (which
  break to trade); the position only sets the size, and two-sided that
  size can be near full on a bar where follow and fade are near-tied (a
  follow rung in a break and a fade rung flat), so the wide ladder trades
  a close call bigger than the plain one would. A `trend` or
  `countertrend` template on this ladder (the `full` family composes
  them) is sized by the same number: following breaks through a range
  leaves the committee split and losing to cash, and the template stands
  mostly aside instead of trading at full size off a channel the learner
  has given up on, which is what the regime filter is meant to do,
  learned bar by bar rather than fitted per window. (The previous version
  of the ladder did that with a copy of every rung gated on the
  efficiency ratio; the position does it with no indicator, threshold or
  gate.) The price is paid in the right regime: a fresh break is a bar
  the committee is not yet in, so entries are sized below the bar
  average, and on the regime series the wide trend template takes about
  half of the plain one's gain through the trends for a tenth of its loss
  through the range.

The plain `hedge` ladder is untouched by all of this: it scores both
legs whatever the template's `sides`, sizes a learned direction by the
net side weight and a fixed one at full size, and is bit for bit what it
was, so the `online` and `online_wide` families compare on the sizing
rule as much as on the ladder.

**What the evidence is, and what it is not.** Real SPY was not reachable
from the build box; the real daily OHLC that was are four single stocks
(AAPL 1984-2008, ORCL and YHOO 1995-2014, GOOG 2004-2013), all NASDAQ
technology names with overlapping years and pairwise daily correlations
of 0.3-0.4, i.e. not four independent tests. On them, the `learned`
template with the `channel` and `atr_trail` exits (8 cells), paired
against the same template on the plain ladder, mean Sharpe difference
and cells improved:

| sizing rule on `hedge_wide` | `long_only`: Sharpe vs plain, cells, drawdown, total return, mean size | `both`: Sharpe vs plain, cells, drawdown |
|---|---|---|
| position, trade weight x committee stance (shipped) | +0.15, 7/8, 4.3 %, 13 %, 0.33 | +0.01, 6/8, 6.1 % |
| trade weight alone | +0.15, 7/8, 6.3 %, 18 %, 0.58 | -0.08, 2/8, 9.4 % |
| committee stance alone | +0.13, 7/8, 7.3 %, 17 %, 0.53 | +0.01, 5/8, 12 % |
| constant full size | +0.12, 7/8, 11 %, 24 %, 1.00 | -0.08, 1/8, 20 % |
| net side weight (the plain rule) on this ladder | -0.03, 4/8, 5.4 %, 8 %, 0.62 | -0.12, 1/8, 13 % |
| the plain `hedge` ladder itself | 0, drawdown 5.2 %, return 9 %, size 0.59 | 0, drawdown 8.8 % |

So, long-only, most of the *Sharpe* lift is "stop shrinking the size in
a bull market": a constant full size, or either factor of the position
alone, gets the same +0.12 to +0.15. What the product adds is the
drawdown and the notional: the same Sharpe at a third of the size, at
40 % of the constant rule's drawdown, and for half of its total return
(sizing down costs return; the position keeps the risk-adjusted part).
Two-sided, where the plain ladder's net side weight already asks the
right question, the +0.01 over eight correlated cells (the two exits
share their entries, so nearer four) is noise: the wider ladder is a
wash on Sharpe and better on drawdown, and the same Keltner rungs under
the plain rule *cost* 0.12, which is the sizing rule, not the bands. On
a synthetic long-only bear market the position runs at a quarter of full
size through the decline and loses 2 % where the plain ladder loses
4-5 % and a constant size 10 %. Seven sizing rules and five ladders were
tried on these series before this one, so the numbers above are
in-sample for the *design* even though no parameter was fitted; the
pipeline's walk-forward, CPCV and Deflated Sharpe Ratio, run on a series
the design never saw, are the test that counts, and `--family
online_wide --sides long_only` on real SPY is the run to make. Two-sided
single stocks are where this family, like every symmetric template on a
drifting asset, earns nothing.

**Why the family is eight templates.** The `online_wide` family is the
`learned` direction over `hedge_wide`, two entries (`stop`,
`close_confirm`) by four exits, and nothing else: no `trend` or
`countertrend` direction, no regime filter, no SMA bias filter. The
learner already carries follow and fade experts and sizes by its own
position against cash, so a fixed direction on top of it throws away half
the ladder, a regime gate second-guesses per window what the position
learns bar by bar, and an SMA bias adds a fitted 200-bar rule to a
channel that has none. `full` still composes `hedge_wide` with every
switch, so the fixed-direction and filtered variants are there for
whoever wants to test the claim. As with every family, `sides` is a
decision about the asset and is taken on the command line: on an index
with a drift run it `--sides long_only`. `hedge_diagnostics(...,
ladder="hedge_wide")` names the experts `follow_10`, `fade_kel20x2`, and
so on, and adds `trade_weight` and `position`.

**A caveat on conviction, for both ladders.** The learner's weights are
follow-the-leader until the losses prove that flipping costs something,
so on pure noise they still concentrate on whichever expert has been
lucky: on a driftless random walk at zero cost the plain ladder's
two-sided conviction averages 0.63, not zero. The trade weight is the
part of the wide ladder's position that does not have this problem (it
averages a half on noise, though it wanders); the committee's stance
still does. Conviction measures
how clearly the committee is positioned and whether that has paid
lately, not how much evidence there is of an edge; the walk-forward, the
Deflated Sharpe Ratio and the Reality Check are still what stands between
a lucky leader and a live position.

Why this and not the other online-learning candidates:

- **Cover's universal portfolio, Anticor (Borodin et al. 2004), OLMAR,
  PAMR** are *multi-asset portfolio* algorithms: they need a cross-
  section to rebalance between. On a single instrument they degenerate.
  Anticor's idea (bet on lagged cross-correlation) survives here only as
  the sign flip that makes the countertrend learner score the fade.
- **Plain Hedge / exponentiated gradient** needs a learning rate, and
  **fixed-share** needs a switching rate: parameters again. AdaHedge is
  the parameter-free variant with the same regret guarantee, and the
  ladder of lifetimes stands in for the switching rate: the learner
  picks the memory the way it picks the lookback. (A fixed-share floor
  on the weights, `1/(N H)` of the leader's, was tried on top of the
  ladder and dropped: it bought no revival the short lifetime does not
  already give, and it flipped the learned direction to *fade* on every
  pullback inside a trend.)
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
  OOS windows (a skipped window, flat because no parameter set traded
  enough in-sample, counts as not profitable), OOS profitable overall
  (`--require-pardo` to enforce).
- **Walk-forward matrix**: the same template over 12 train/test-length
  combinations. Robust means profitable in most cells.
- **Rolling or anchored** windows, optional **embargo** gap.
- **Combinatorial Purged CV** (AFML ch. 12): the history is cut into
  groups, every choice of test groups is a train/test split (with an
  embargo after each test group in two parts: a fixed one as long as the
  longest warm-up any grid point needs -- channels, EMA settling, filters,
  the whole ~410-bar memory of the online learner, and a pullback order's
  lifetime -- so no later entry decision reads a test-group bar; and, per
  parameter set, on until the trade still open at its end has closed,
  since that trade was opened or steered by the test group's prices and
  can run for hundreds of bars. Tested directly: re-pricing only the test
  groups leaves every training return unchanged, to 1e-12 (Keltner/ADX:
  within the EMA settling tolerance, ~1e-6), apart from a different
  clean trade taken because the book stayed busy longer), the trial is chosen by the same
  `--metric`, `--selection` and min-trades rule as the walk-forward (a
  profit factor counts only trades held entirely inside training; a split
  where nothing trades enough stays flat, and the share of such splits is
  printed next to P(CPCV<0) so an all-flat 0 % is not read as robust), and
  the results are stitched into
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
  history realised *so far* and held for the next window. The filter is
  the static one (min Sharpe, at least 10 OOS trades and 3 live windows,
  `--require-pardo`), with trades, windows and Pardo's criteria counted
  only over the walk-forward windows that had ended by then. That curve is
  out-of-sample with respect to both the parameters and the selection,
  and it is the only portfolio number in the report worth quoting.

On the default synthetic data (mostly noise) the toolkit says so: PBO
about 0.6, Reality Check p about 0.95, DSR far below 0.95, and the
static portfolio's Sharpe of 0.6 turns into -0.5 once the selection is
walked forward. With `--trend-prob 0.8 --trend-drift 0.002` (a real
edge) the finalist passes all 12 walk-forward matrix cells, its
bootstrap p-value is 0, and the nested portfolio keeps a Sharpe near 1.


## Caveats

This is a research framework, not a production trading system. The
cost model is a flat bps charge plus a flat charge per unit, position
sizing is fixed-fractional on an ATR stop (or, with `--vol-target`, a
constant dollar volatility fixed at entry and never rebalanced), and the
synthetic data is a toy
regime-switching random walk. Results on synthetic data are a pipeline check. Real conclusions
need real data, realistic costs for the instrument, and -- as the
Reality Check numbers make painfully clear -- a lot more history than
a decade of daily bars for a family of hundreds of trials.

### Known approximations

Each point below was measured on the synthetic data (2500 bars, 
the 72 templates of the `quick` family, once as noise and once with 
`--trend-prob 0.8 --trend-drift 0.002`) and is deliberately NOT changed. 
They are listed with the number that would justify reopening each one.

- **Every walk-forward window starts flat.** A position still open at the
  end of test window N is marked to market on its last bar and is gone in
  window N+1, which waits for a new signal (about 40 % of windows end
  with a position open). This is the price of the `first_trade_bar` rule
  above: a position carried into a window was opened on bars the window's
  parameters were fitted on. The bias is CONSERVATIVE. Letting each
  window inherit the state of its parameters' continuous path instead
  raised the mean OOS Sharpe by 0.04 with an edge and by 0.01 on noise,
  and left the ranking of the templates unchanged (Spearman 0.99 and
  0.90), so no selection decision depends on it.
- **The position open at the end of a window never pays its exit cost.**
  This one flatters the result, by `cost_bps` x notional / equity per
  window that ends in a position: 1.3 bp of CAGR per year and 0.005 of
  Sharpe at the defaults (5 bps, ATR sizing at 1 % risk), far below
  anything the pipeline decides on. Charging it means editing the
  window's equity, returns, trade list and exposure consistently, for a
  bias an order of magnitude smaller than the conservative one above.
  It scales linearly with cost and leverage: reopen it if
  `cost_bps * max_leverage` goes up, say, tenfold.
- **CPCV and CSCV slice a full-history trials matrix**, so a test block
  can begin in the middle of a trade that was opened before it. This is
  how CSCV is defined (Bailey et al. 2015) and it is not look-ahead: the
  trade was opened by fixed rules on a fixed parameter set, and nothing
  from the test block reaches the selection. What is left is serial
  dependence across the train/test boundary, which is what purging is
  for. Purging `2 * n_entry` bars before every test group (`cpcv(...,
  purge_bars=)`) moved the mean path Sharpe by -0.006 with an edge and
  +0.006 on noise, no systematic sign, against a spread of 0.3 to 1.1
  across templates; it mostly changed results by shrinking the training
  set. The default stays at 0. CSCV cannot be purged by construction;
  its blocks (T/16 bars) are long next to a trade and PBO is a rank
  statistic with every trial treated alike.
- **`adx()` seeds Wilder's smoothing with the first observation**, not
  with the SMA of the first n as charting platforms do. The two differ
  only while the seed is remembered, and `warmup_bars` already keeps the
  strategy from trading until the seed has decayed. `atr()` is a plain rolling mean and has no seed.

## Extending this

- **Multi-asset portfolios**: done, in `etf_dashboard.py research` -- every
  `(asset, template)` pair is one slot in a single candidate pool, so the
  portfolio step diversifies structurally and across markets at once, closer
  to RangerZ's basic portfolio.
- **More switches**: add an indicator to `REGIME_INDICATORS` or a new
  entry/exit branch in `strategy.backtest`, then list it in
  `generator.FAMILIES`.
- **More experts for the hedge channel**: done, as `hedge_wide` and the
  `online_wide` family (a Keltner band per rung, see above). To go
  further, add a `HedgeExpert` to a ladder in `HEDGE_LADDERS`, or a new
  ladder under a new channel type (list it in `HEDGE_POSITION_SIZED` to
  size by the position): an expert is a pair of bands plus a side, and
  needs its `formed` / `lead` right so that `hedge_warmup` keeps the
  warm-up contract (rolling windows only: an EMA-based expert would break
  the match of a warmed window with a full-history run). The mixture
  stays causal and parameter-free as long as the ladder is fixed in
  advance; the `test_warm_window_matches_the_full_run` and
  `WideLadderTests` tests are the checklist. Three things worth trying on
  an index: an *always-long* expert, so the ladder is measured against
  the benchmark that matters there and the position can be "hold"; a
  *1-5 bar fade rung*, the span of an index's short-term reversion, which
  the 10-80 bar ladder does not reach; and a *volatility-gated* pair
  (fade only while the ATR sits in the top of its own history, follow
  only in the bottom: on equity indices reversion lives in high-vol
  stretches and drift in calm ones). A bigger change for a long-only
  index is to let a `learned` template rest both orders at once, a stop
  above and a limit below, each sized by its side's weight, instead of
  picking one trigger per bar.
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
