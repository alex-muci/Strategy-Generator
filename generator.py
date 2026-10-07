"""
generator.py
------------
The "Ranger" idea: don't hand-pick one strategy, generate the whole
family of structurally distinct ones by combining switches, and let
evaluation (walk-forward analysis + robustness tests + portfolio
selection) decide which ones earn a place in the final portfolio.

Five families are predefined:

  quick           72 templates  (Donchian only, ER regime filter) -- smoke test
  default        768 templates  (two channel types, five regime indicators)
  online           9 templates  (the follow group of the split hedge ladder
                                  -- 20-80 bar breaks held as long as their
                                  lookback under a slow learner -- by the
                                  stop / close-confirm entries and every exit,
                                  and the committee traded directly (the
                                  'stance' entry, whose exit is the learner:
                                  one template); no lookback or width in the
                                  grid, no regime / bias filter, trend
                                  direction only. On SPY / TLT / GLD / USO
                                  2016-2026 the trend (follow) direction
                                  carried all of the hedge families' edge,
                                  countertrend and the split ladder's fade
                                  group lost on every series, no regime or
                                  bias filter beat none, and the slow memory
                                  beat the fast one (docs/
                                  hedge_real_data_study.md). 'learned' and
                                  'countertrend' stay available as switches,
                                  generate_templates("online",
                                  direction_logics=[...]), for data with a
                                  short-horizon reversal (spreads, intraday);
                                  the other hedge ladders likewise
                                  (channel_types=[...]))
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
    # the split ladder's follow group (strategy.HEDGE_SPLIT), trend direction only:
    # see the module docstring for why; the stance entry is canonicalised to a
    # single template (its exit is the learner)
    "online": dict(
        direction_logics=["trend"],
        channel_types=["hedge_split"],
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
        # this one; the split ladder needs more warm-up than a short test series has
        channel_types=[c for c in CHANNEL_TYPES if not c.endswith("_slow") and c != "hedge_split"
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
    "hedge_slow": "hds", "hedge_wide_slow": "hws", "hedge_split": "hsp",
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


def param_grid_for(tpl: StrategyTemplate, wide: bool = False) -> dict:
    """Return the numeric-parameter search grid to walk-forward optimize
    for this template. Only the params relevant to this template's
    switches are varied; everything else stays at the template default.

    The grid is a LATTICE (each param a sorted list) so that the
    walk-forward 'plateau' selection can look at parameter neighbours.
    Kept deliberately small so a full sweep across hundreds of templates
    finishes in minutes; `wide=True` roughly triples it.
    """
    grid = {}
    if tpl.channel_type in FORECAST_CHANNELS:
        return grid      # the forecaster learns online and the stance has no exit: nothing to fit
    online = tpl.channel_type in HEDGE_CHANNELS   # lookbacks (and widths) are learned online, not fitted
    if not online:
        grid["n_entry"] = [20, 40, 60] if not wide else [10, 20, 30, 40, 55, 70, 90]

    if tpl.channel_type in ("keltner", "bollinger"):
        grid["channel_k"] = [1.5, 2.5] if not wide else [1.0, 1.5, 2.0, 2.5, 3.0]

    if tpl.exit_style == "channel" and not online:
        grid["n_exit"] = [10, 20] if not wide else [5, 10, 15, 20, 30]
    elif tpl.exit_style == "atr_trail":
        grid["atr_mult_trail"] = [2.5, 3.5] if not wide else [1.5, 2.0, 2.5, 3.0, 3.5, 4.5]
    elif tpl.exit_style == "target_stop":
        grid["atr_mult_stop"] = [2.0, 3.0] if not wide else [1.5, 2.0, 2.5, 3.0, 4.0]
        grid["atr_mult_target"] = [3.0, 4.0] if not wide else [2.0, 3.0, 4.0, 5.0, 6.0]
    elif tpl.exit_style == "time_stop":
        grid["max_hold_bars"] = [10, 20] if not wide else [5, 10, 15, 20, 30, 40]

    if tpl.entry_style == "pullback":
        grid["pullback_atr_mult"] = [0.4, 0.7] if not wide else [0.25, 0.5, 0.75, 1.0]

    if tpl.regime_filter != "none":
        grid["regime_threshold"] = list(REGIME_INDICATORS[tpl.regime_indicator]["thresholds"])

    return grid
