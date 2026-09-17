"""
generator.py
------------
The "Ranger" idea: don't hand-pick one strategy, generate the whole
family of structurally distinct ones by combining switches, and let
evaluation (walk-forward analysis + robustness tests + portfolio
selection) decide which ones earn a place in the final portfolio.

Three families are predefined:

  quick    ~72 templates  (Donchian only, ER regime filter) -- smoke test
  default  ~300 templates (two channel types, five regime indicators)
  online   ~290 templates (the online-learned 'hedge' channel only, incl.
                           the 'learned' follow-or-fade direction: no
                           lookback or width in the grid, the walk-forward
                           only re-fits exits / regime thresholds)
  full     every combination of every switch (thousands; overnight run)

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
    (ind, mode) for ind in REGIME_INDICATORS for mode in ("trend_only", "range_only")
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
    "online": dict(
        direction_logics=DIRECTION_LOGICS,
        channel_types=["hedge"],
        entry_styles=["stop", "close_confirm"],
        exit_styles=EXIT_STYLES,
        regimes=[("er", "none"), ("er", "trend_only"), ("er", "range_only"),
                 ("vr", "trend_only"), ("vr", "range_only"), ("chop", "range_only")],
        vol_filters=[False],
        bias_filters=["none", "sma"],
        sides=["both"],
    ),
    "full": dict(
        direction_logics=DIRECTION_LOGICS,
        channel_types=CHANNEL_TYPES,
        entry_styles=ENTRY_STYLES,
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
    "donchian": "don", "keltner": "kel", "bollinger": "bol", "hedge": "hdg",
    "stop": "stop", "close_confirm": "cls", "pullback": "pb",
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
    online = tpl.channel_type == "hedge"   # lookbacks are learned online, not fitted
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
