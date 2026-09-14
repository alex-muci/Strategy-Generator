"""
generator.py
------------
The "Ranger" idea: don't hand-pick one strategy, generate the whole
family of structurally distinct ones by combining switches, and let
evaluation (walk-forward analysis + portfolio selection) decide which
ones earn a place in the final portfolio.
"""

from __future__ import annotations
from itertools import product
from strategy import (
    StrategyTemplate,
    DIRECTION_LOGICS,
    ENTRY_STYLES,
    EXIT_STYLES,
    REGIME_FILTERS,
    VOL_FILTERS,
)


def generate_templates(
    direction_logics=DIRECTION_LOGICS,
    entry_styles=ENTRY_STYLES,
    exit_styles=EXIT_STYLES,
    regime_filters=REGIME_FILTERS,
    vol_filters=VOL_FILTERS,
    max_templates: int | None = None,
) -> list[StrategyTemplate]:
    """Full Cartesian product of the switches -> one StrategyTemplate per
    combination. Some combinations are redundant/nonsensical and are
    skipped (e.g. a 'range_only' regime filter combined with a pure
    trend-following opposite-channel exit still works fine, so nothing
    is excluded here beyond trivial duplicates).

    Set max_templates to cap the family size (useful while iterating).
    """
    templates = []
    combos = product(direction_logics, entry_styles, exit_styles, regime_filters, vol_filters)
    for dl, es, ex, rf, vf in combos:
        name = f"{dl[:2].upper()}-{es}-{ex}-{rf}-{'V' if vf else 'noV'}"
        templates.append(
            StrategyTemplate(
                name=name,
                direction_logic=dl,
                entry_style=es,
                exit_style=ex,
                regime_filter=rf,
                vol_filter=vf,
            )
        )
        if max_templates and len(templates) >= max_templates:
            break
    return templates


def param_grid_for(tpl: StrategyTemplate) -> dict:
    """Return the numeric-parameter search grid to walk-forward optimize
    for this template. Only the params relevant to this template's
    switches are varied; everything else stays at the template default.
    """
    # Kept deliberately small so a full sweep across dozens of templates
    # finishes in a reasonable time. Widen these if you have the compute
    # budget (e.g. for an overnight run on real data).
    grid = {
        "n_entry": [20, 40, 60],
    }
    if tpl.exit_style == "opposite":
        grid["n_exit"] = [15, 25]
    elif tpl.exit_style == "atr_trail":
        grid["atr_mult_trail"] = [2.5, 3.5]
        grid["atr_mult_stop"] = [3.0]
    elif tpl.exit_style == "target_stop":
        grid["atr_mult_stop"] = [2.0, 3.0]
        grid["atr_mult_target"] = [3.0, 4.0]

    if tpl.entry_style == "pullback":
        grid["pullback_atr_mult"] = [0.4, 0.7]

    if tpl.regime_filter != "none":
        grid["er_threshold"] = [0.3, 0.4]

    return grid
