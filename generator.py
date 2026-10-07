"""
generator.py
------------
The "Ranger" idea: don't hand-pick one strategy, generate the whole
family of structurally distinct ones by combining switches, and let
evaluation (walk-forward analysis + robustness tests + portfolio
selection) decide which ones earn a place in the final portfolio.

Six families are predefined:

  quick           72 templates  (Donchian only, ER regime filter) -- smoke test
  default        768 templates  (two channel types, five regime indicators)
  online          27 templates  (the follow group of the split hedge ladder
                                  -- 20-80 bar breaks held as long as their
                                  lookback under a slow learner -- by the
                                  stop / close-confirm entries and every exit,
                                  and the committee traded directly (the
                                  'stance' entry, whose exit is the learner:
                                  one template per direction); no lookback or
                                  width in the grid, no regime / bias filter,
                                  trend / countertrend / learned. On SPY /
                                  TLT / GLD / USO 2016-2026 the trend (follow)
                                  direction carried all of the hedge families'
                                  edge, countertrend and the split ladder's
                                  fade group lost on every series, no regime
                                  or bias filter beat none, and the slow
                                  memory beat the fast one (docs/
                                  hedge_real_data_study.md); countertrend and
                                  learned are kept all the same, for futures
                                  and data with a short-horizon reversal
                                  (spreads, intraday). Narrow it with
                                  generate_templates("online",
                                  direction_logics=["trend"]); the other hedge
                                  ladders are a switch too
                                  (channel_types=[...]))
  online_long     27 templates  (the 'online' family on the long ladder: the
                                  plain Donchian experts one rung slower,
                                  20-160 bars, under the slow learner -- the
                                  2-8 month horizons on daily bars that no
                                  other family trades; nothing in the grid
                                  but the exits)
  online_forecast  3 templates  (the forecast channel -- slow trend prior plus a
                                  learned fast-scale term, held as a position
                                  under a no-trade buffer -- by trend /
                                  countertrend / learned, nothing in the grid)
  full          32400 templates (every combination of every switch; overnight
                                  run)

The registered online-vs-quick comparison must be re-run on data the redesign
never saw: 2007-2015 of SPY / TLT / GLD / USO, and all of QQQ IWM IEF SLV UNG
FXE FXY FXB FXA.

The bigger the family, the more the *selection* step becomes a data
mining exercise -- which is exactly why main.py reports the Probability
of Backtest Overfitting, the Deflated Sharpe Ratio and White's Reality
Check for the whole family, not just the winners.
"""

from __future__ import annotations
from itertools import product
from strategy import (
    StrategyTemplate,
    DIRECTION_LOGICS,
    CHANNEL_TYPES,
    HEDGE_CHANNELS,
    FORECAST_CHANNELS,
    ENTRY_STYLES,
    EXIT_STYLES,
    REGIME_INDICATORS,
    REGIME_FILTERS,
    VOL_FILTERS,
    BIAS_FILTERS,
    SIDES,
)

# (indicator, filter) pairs; the indicator is irrelevant when filter == 'none'
_ALL_REGIMES = [("er", "none")] + [
    (ind, mode) for ind in REGIME_INDICATORS for mode in REGIME_FILTERS if mode != "none"
]

FAMILIES = {
    "quick": dict(
        direction_logics=["trend", "countertrend"],
        channel_types=["donchian"],
        entry_styles=["stop", "pullback"],
        exit_styles=["channel", "atr_trail", "target_stop"],
        regimes=[("er", "none"), ("er", "trend_only"), ("er", "range_only")],
        vol_filters=VOL_FILTERS,
        bias_filters=["none"],
        sides=["both"],
    ),
    "default": dict(
        direction_logics=["trend", "countertrend"],
        channel_types=["donchian", "keltner"],
        entry_styles=["stop", "close_confirm", "pullback"],
        exit_styles=["channel", "atr_trail", "target_stop", "time_stop"],
        regimes=[("er", "none"), ("er", "trend_only"), ("er", "range_only"),
                 ("adx", "trend_only"), ("cti", "trend_only"),
                 ("chop", "range_only"), ("vr", "trend_only"), ("vr", "range_only")],
        vol_filters=[False],
        bias_filters=["none", "sma"],
        sides=["both"],
    ),
    # the split ladder's follow group (strategy.HEDGE_SPLIT), every direction
    # logic: although countertrend does not seem to work at all on daily SPY /
    # TLT / GLD / USO (see the module docstring), there is no need to exclude it
    # for futures. The stance entry is canonicalised to a single template per
    # direction (its exit is the learner)
    "online": dict(
        direction_logics=DIRECTION_LOGICS,
        channel_types=["hedge_split"],
        entry_styles=["stop", "close_confirm", "stance"],
        exit_styles=EXIT_STYLES,
        regimes=[("er", "none")],
        vol_filters=[False],
        bias_filters=["none"],
        sides=["both"],
    ),
    # the 'online' switches on the long ladder (strategy.HEDGE_LONG_LADDER,
    # Donchian 20-160 under the slow learner): the slow-horizon counterpart of
    # `online`, for the trend horizons beyond its 20-80 bar follow group
    "online_long": dict(
        direction_logics=DIRECTION_LOGICS,
        channel_types=["hedge_long"],
        entry_styles=["stop", "close_confirm", "stance"],
        exit_styles=EXIT_STYLES,
        regimes=[("er", "none")],
        vol_filters=[False],
        bias_filters=["none"],
        sides=["both"],
    ),
    # the forecast channel (strategy.FORECAST_CHANNELS): a slow trend prior and
    # a freely learned fast-scale term, traded as a position under a no-trade
    # buffer; trend (the prior alone), countertrend (the fast learner alone)
    # and learned (both); nothing to fit. `forecast_ctx` (the same with a regime
    # context) stays a switch, generate_templates("online_forecast",
    # channel_types=["forecast_ctx"]), but is not in the family: on the dev ETFs
    # and on every synthetic series it matched `forecast` within noise, so it
    # would only add a clone trial to the PBO / DSR counts.
    "online_forecast": dict(
        direction_logics=DIRECTION_LOGICS,
        channel_types=["forecast"],
        entry_styles=["stance"],
        exit_styles=["channel"],
        regimes=[("er", "none")],
        vol_filters=[False],
        bias_filters=["none"],
        sides=["both"],
    ),
    "full": dict(
        direction_logics=DIRECTION_LOGICS,
        # the slow ladders are the fast ones' experts with a longer memory, so
        # they have their own families rather than doubling the hedge part of
        # this one; the split and long ladders need more warm-up than a short test series has
        channel_types=[c for c in CHANNEL_TYPES if not c.endswith("_slow") and c not in ("hedge_split", "hedge_long")
                       and c not in FORECAST_CHANNELS],
        entry_styles=[e for e in ENTRY_STYLES if e != "stance"],
        exit_styles=EXIT_STYLES,
        regimes=_ALL_REGIMES,
        vol_filters=VOL_FILTERS,
        bias_filters=BIAS_FILTERS,
        sides=SIDES,
    ),
}
# `sides` is deliberately ["both"] in quick and default: restricting a family to
# one side is a decision about the ASSET (does it have a drift?), so it is taken
# on the command line (`--sides long_only`) rather than by tripling the family
# and letting the selection step data-mine it.

_SHORT = {
    "trend": "TR", "countertrend": "CT", "learned": "LN",
    "donchian": "don", "keltner": "kel", "bollinger": "bol", "hedge": "hdg", "hedge_wide": "hdw",
    "hedge_slow": "hds", "hedge_wide_slow": "hws", "hedge_split": "hsp", "hedge_long": "hlg",
    "forecast": "fc", "forecast_ctx": "fcx",
    "stop": "stop", "close_confirm": "cls", "pullback": "pb", "stance": "stance",
    "channel": "chan", "atr_trail": "trail", "target_stop": "tgt", "time_stop": "time",
    "none": "none", "trend_only": "trend", "range_only": "range",
    "sma": "sma",
    "both": "", "long_only": "L", "short_only": "S",
}


def template_name(dl, ch, es, ex, rind, rf, vf, bf, sd="both") -> str:
    regime = "noreg" if rf == "none" else f"{rind}:{_SHORT[rf]}"
    base = f"{_SHORT[dl]}-{_SHORT[ch]}-{_SHORT[es]}-{_SHORT[ex]}-{regime}-{'V' if vf else 'noV'}-{'B' if bf == 'sma' else 'noB'}"
    # two-sided templates keep their historical names; one-sided ones get a suffix
    return base if sd == "both" else f"{base}-{_SHORT[sd]}"


def generate_templates(
    family: str = "quick",
    max_templates: int | None = None,
    **overrides,
) -> list[StrategyTemplate]:
    """One StrategyTemplate per combination of the switch lists of
    `family` (see FAMILIES). Any list can be overridden by keyword, e.g.
    generate_templates("default", channel_types=["bollinger"]).
    Set max_templates to cap the family size (useful while iterating)."""
    spec = dict(FAMILIES[family])
    spec.update(overrides)
    templates = []
    seen = set()
    combos = product(
        spec["direction_logics"], spec["channel_types"], spec["entry_styles"],
        spec["exit_styles"], spec["regimes"], spec["vol_filters"], spec["bias_filters"],
        spec.get("sides", ["both"]),
    )
    for dl, ch, es, ex, (rind, rf), vf, bf, sd in combos:
        if rf == "none":
            rind = "er"  # canonical: indicator is irrelevant without a filter
        if es == "stance":
            ex = "channel"  # canonical: the stance entry has no exit rule, the learner is its exit
        key = (dl, ch, es, ex, rind, rf, vf, bf, sd)
        if key in seen:
            continue
        seen.add(key)
        tpl = StrategyTemplate(
            name=template_name(*key),
            direction_logic=dl, channel_type=ch, entry_style=es, exit_style=ex,
            regime_indicator=rind, regime_filter=rf, vol_filter=vf, bias_filter=bf, sides=sd,
        )
        tpl.validate()
        templates.append(tpl)
        if max_templates and len(templates) >= max_templates:
            break
    return templates


SLOW_HARD_STOP = 6.0   # ATRs: the hard stop of the slow grid (param_grid_for), at the top of its trails


def param_grid_for(tpl: StrategyTemplate, wide: bool = False, slow: bool = False) -> dict:
    """Return the numeric-parameter search grid to walk-forward optimize
    for this template. Only the params relevant to this template's
    switches are varied; everything else stays at the template default.

    The grid is a LATTICE (each param a sorted list) so that the
    walk-forward 'plateau' selection can look at parameter neighbours.
    Kept deliberately small so a full sweep across hundreds of templates
    finishes in minutes; `wide=True` roughly triples it.

    `slow=True` moves every horizon of the grid (the entry and exit
    lookbacks, the time stop) about three times slower, with the ATR exits
    widened to match: on daily bars n_entry 60-250, i.e. the 3-12 month
    breakouts of classic trend following, which the default grid (20-60)
    never trades. The hard stop, always working whatever the exit, is set
    to SLOW_HARD_STOP ATRs (one value, not a search dimension): at the
    default 3 ATR(20) a 250-bar breakout is stopped out by noise long
    before its channel or trail exit is reached. Under --risk-pct sizing
    the wider stop trades proportionally fewer units (same risk per trade);
    under --vol-target it changes no size. It is the same number of values,
    so the same number of trials, on another horizon band, not a finer
    grid: run it as its own research and compare, rather than merging the
    two grids into one search.
    The longest lookback lengthens the warm-up (walkforward.warmup_bars), so
    give it a training window of at least two of them (--train 500 or more on
    daily bars)."""
    grid = {}
    if tpl.channel_type in FORECAST_CHANNELS:
        return grid      # the forecaster learns online and the stance has no exit: nothing to fit
    online = tpl.channel_type in HEDGE_CHANNELS   # lookbacks (and widths) are learned online, not fitted

    def pick(base, wider, slow_base, slow_wider):
        if slow:
            return slow_wider if wide else slow_base
        return wider if wide else base

    if not online:
        grid["n_entry"] = pick([20, 40, 60], [10, 20, 30, 40, 55, 70, 90],
                               [60, 120, 250], [50, 80, 120, 160, 200, 250, 300])

    if tpl.channel_type in ("keltner", "bollinger"):
        grid["channel_k"] = [1.5, 2.5] if not wide else [1.0, 1.5, 2.0, 2.5, 3.0]

    if tpl.exit_style == "channel" and not online:
        grid["n_exit"] = pick([10, 20], [5, 10, 15, 20, 30], [20, 50], [15, 20, 30, 50, 80])
    elif tpl.exit_style == "atr_trail":
        grid["atr_mult_trail"] = pick([2.5, 3.5], [1.5, 2.0, 2.5, 3.0, 3.5, 4.5], [4.0, 6.0], [3.5, 4.0, 5.0, 6.0, 7.0, 8.0])
    elif tpl.exit_style == "target_stop":
        grid["atr_mult_stop"] = pick([2.0, 3.0], [1.5, 2.0, 2.5, 3.0, 4.0], [4.0, 6.0], [3.0, 4.0, 5.0, 6.0, 8.0])
        grid["atr_mult_target"] = pick([3.0, 4.0], [2.0, 3.0, 4.0, 5.0, 6.0], [8.0, 12.0], [6.0, 8.0, 10.0, 12.0, 16.0])
    elif tpl.exit_style == "time_stop":
        grid["max_hold_bars"] = pick([10, 20], [5, 10, 15, 20, 30, 40], [40, 80], [30, 40, 60, 80, 120, 160])

    if slow and "atr_mult_stop" not in grid and tpl.entry_style != "stance":
        grid["atr_mult_stop"] = [SLOW_HARD_STOP]   # the stance entry has no stop

    if tpl.entry_style == "pullback":
        grid["pullback_atr_mult"] = [0.4, 0.7] if not wide else [0.25, 0.5, 0.75, 1.0]

    if tpl.regime_filter != "none":
        grid["regime_threshold"] = list(REGIME_INDICATORS[tpl.regime_indicator]["thresholds"])

    return grid
