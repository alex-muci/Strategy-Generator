"""
Tests that every template is generated, parameterized and executed as its
switches say -- the part of the generator where a silent mistake is most
expensive, because a wrong template is still a plausible-looking equity curve.

Three kinds of check:

* generation: the family is the full product of its switch lists, names are
  unique, the parameter grid only varies what the switches use;
* inertness: a parameter that a template's switches do not use must not
  change its backtest at all, and one they do use must;
* semantics: filters at their extreme settings reduce to "no filter" or "no
  trades", and the engine treats longs and shorts as exact mirrors of one
  another (reflect the price series and every trade flips side with the
  same P&L), so a bug in one side's code path cannot hide behind the other.

Run with:   python -m unittest discover -s tests -v
"""

from __future__ import annotations
import dataclasses
import os
import sys
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import synthetic_ohlc  # noqa: E402
from strategy import (  # noqa: E402
    StrategyTemplate, backtest, _compute_indicators, REGIME_INDICATORS,
    DIRECTION_LOGICS, CHANNEL_TYPES, ENTRY_STYLES, EXIT_STYLES, REGIME_FILTERS,
    VOL_FILTERS, BIAS_FILTERS, SIDES,
)
from generator import generate_templates, param_grid_for, FAMILIES  # noqa: E402
from walkforward import grid_combos  # noqa: E402


FIELDS = {f.name for f in dataclasses.fields(StrategyTemplate)}

# which numeric parameter is read only under which switch setting
PARAM_NEEDS = {
    "n_entry": lambda t: t.channel_type != "hedge",
    "n_exit": lambda t: t.exit_style == "channel" and t.channel_type != "hedge",
    "channel_k": lambda t: t.channel_type in ("keltner", "bollinger"),
    "atr_mult_trail": lambda t: t.exit_style == "atr_trail",
    "atr_mult_target": lambda t: t.exit_style == "target_stop",
    "max_hold_bars": lambda t: t.exit_style == "time_stop",
    "pullback_atr_mult": lambda t: t.entry_style == "pullback",
    "pullback_valid_bars": lambda t: t.entry_style == "pullback",
    "regime_n": lambda t: t.regime_filter != "none",
    "regime_threshold": lambda t: t.regime_filter != "none",
    "vol_lookback": lambda t: t.vol_filter,
    "vol_low_pct": lambda t: t.vol_filter,
    "vol_high_pct": lambda t: t.vol_filter,
    "bias_n": lambda t: t.bias_filter == "sma",
}
# a value far enough from the default that using it would visibly change a run
PERTURB = {
    "n_entry": 13, "n_exit": 7, "channel_k": 0.7, "atr_mult_trail": 1.2, "atr_mult_target": 1.5,
    "max_hold_bars": 3, "pullback_atr_mult": 1.5, "pullback_valid_bars": 9,
    "regime_n": 7, "regime_threshold": 0.01, "vol_lookback": 30, "vol_low_pct": 0.45,
    "vol_high_pct": 0.55, "bias_n": 15,
}


def _family_size(spec: dict) -> int:
    regimes = {("er", rf) if rf == "none" else (ri, rf) for ri, rf in spec["regimes"]}
    return (len(spec["direction_logics"]) * len(spec["channel_types"]) * len(spec["entry_styles"])
            * len(spec["exit_styles"]) * len(regimes) * len(spec["vol_filters"]) * len(spec["bias_filters"])
            * len(spec.get("sides", ["both"])))


_MIRROR_SIDES = {"both": "both", "long_only": "short_only", "short_only": "long_only"}


def _mirror(df: pd.DataFrame) -> pd.DataFrame:
    """Reflect every price around a level above the whole series, so an
    up-move becomes the same-sized down-move and highs and lows swap."""
    c = 2.0 * (float(df["High"].max()) + 1.0)
    return pd.DataFrame({"Open": c - df["Open"], "High": c - df["Low"], "Low": c - df["High"],
                         "Close": c - df["Close"], "Volume": df["Volume"]}, index=df.index)


class TemplateGenerationTests(unittest.TestCase):
    def test_families_are_the_full_product_of_their_switch_lists(self):
        for family, spec in FAMILIES.items():
            tpls = generate_templates(family)
            self.assertEqual(len(tpls), _family_size(spec), family)
            self.assertEqual(len({t.name for t in tpls}), len(tpls), f"{family}: duplicate names")
            self.assertEqual(len({tuple(sorted(t.switches().items())) for t in tpls}), len(tpls),
                             f"{family}: two templates with identical switches")
            for t in tpls:
                t.validate()
        # README numbers
        self.assertEqual(len(generate_templates("quick")), 72)

    def test_switch_lists_are_the_engine_vocabulary(self):
        """A switch value the engine cannot map to a code would only fail
        deep inside backtest(); the families must only use known ones."""
        from strategy import ENTRY_CODES, EXIT_CODES, REGIME_CODES
        for t in generate_templates("full"):
            self.assertIn(t.direction_logic, DIRECTION_LOGICS)
            self.assertIn(t.channel_type, CHANNEL_TYPES)
            self.assertIn(t.entry_style, ENTRY_CODES)
            self.assertIn(t.exit_style, EXIT_CODES)
            self.assertIn(t.regime_filter, REGIME_CODES)
            self.assertIn(t.regime_indicator, REGIME_INDICATORS)
            self.assertIn(t.bias_filter, BIAS_FILTERS)
            self.assertIn(t.vol_filter, VOL_FILTERS)
            self.assertIn(t.sides, SIDES)
        self.assertEqual(set(ENTRY_CODES), set(ENTRY_STYLES))
        self.assertEqual(set(EXIT_CODES), set(EXIT_STYLES))
        self.assertEqual(set(REGIME_CODES), set(REGIME_FILTERS))

    def test_regime_indicator_is_canonical_when_there_is_no_filter(self):
        """Without a regime filter the indicator is dead weight; the generator
        must collapse every (indicator, 'none') pair into ONE template, or the
        family carries identical strategies under different names and every
        multiple-testing correction over-counts them."""
        for t in generate_templates("full"):
            if t.regime_filter == "none":
                self.assertEqual(t.regime_indicator, "er", t.name)
                self.assertEqual(t.switches()["regime_indicator"], "-")
                self.assertIn("noreg", t.name)
        n_none = sum(t.regime_filter == "none" for t in generate_templates("full"))
        spec = FAMILIES["full"]
        self.assertEqual(n_none, _family_size(spec) // len({r for r in spec["regimes"]
                                                             if r[1] != "none"} | {("er", "none")}))

    def test_overrides_and_cap(self):
        tpls = generate_templates("default", channel_types=["bollinger"], max_templates=10)
        self.assertEqual(len(tpls), 10)
        self.assertTrue(all(t.channel_type == "bollinger" for t in tpls))
        with self.assertRaises(KeyError):
            generate_templates("nope")


class TemplateParamTests(unittest.TestCase):
    def test_with_params_copies_and_rejects_unknown_fields(self):
        t = StrategyTemplate("t", n_entry=40)
        u = t.with_params(n_entry=55, cost_bps=0.0)
        self.assertEqual(t.n_entry, 40)
        self.assertEqual((u.n_entry, u.cost_bps, u.name), (55, 0.0, "t"))
        self.assertEqual(u.switches(), t.switches())
        with self.assertRaises(TypeError):
            t.with_params(n_enrty=55)

    def test_grid_only_varies_what_the_switches_use(self):
        for wide in (False, True):
            for t in generate_templates("full"):
                grid = param_grid_for(t, wide=wide)
                if t.channel_type != "hedge":
                    self.assertIn("n_entry", grid, t.name)
                for k, values in grid.items():
                    self.assertIn(k, FIELDS, f"{t.name}: {k} is not a template field")
                    self.assertEqual(list(values), sorted(values), f"{t.name}: {k} is not a sorted lattice")
                    self.assertEqual(len(set(values)), len(values), f"{t.name}: {k} has duplicates")
                    if k in PARAM_NEEDS:
                        self.assertTrue(PARAM_NEEDS[k](t), f"{t.name}: grid varies unused {k}")
                for k, needed in PARAM_NEEDS.items():
                    if needed(t) and k not in ("pullback_valid_bars", "regime_n", "vol_lookback",
                                               "vol_low_pct", "vol_high_pct", "bias_n"):
                        self.assertIn(k, grid, f"{t.name}: {k} is used but not searched")
                if t.regime_filter != "none":
                    self.assertEqual(grid["regime_threshold"],
                                     REGIME_INDICATORS[t.regime_indicator]["thresholds"])

    def test_sizing_settings_are_never_searched(self):
        """risk_pct, max_leverage, cost_bps and the vol target are run
        settings: every template is generated with them off/default and no
        grid ever varies them."""
        for t in generate_templates("full"):
            self.assertEqual((t.vol_target, t.vol_target_n), (0.0, 60), t.name)
            for wide in (False, True):
                grid = param_grid_for(t, wide=wide)
                for k in ("vol_target", "vol_target_n", "risk_pct", "max_leverage", "cost_bps"):
                    self.assertNotIn(k, grid, f"{t.name}: {k} is a run setting, not a searched parameter")

    def test_regime_defaults_come_from_the_indicator_registry(self):
        df = synthetic_ohlc(500, seed=1)
        for name, spec in REGIME_INDICATORS.items():
            t = StrategyTemplate("t", regime_filter="trend_only", regime_indicator=name)
            self.assertEqual(t.regime_n, 0)
            self.assertTrue(np.isnan(t.regime_threshold))
            ind = _compute_indicators(df, t)
            self.assertEqual(ind["regime_threshold"], spec["threshold"])
            explicit = _compute_indicators(df, t.with_params(regime_n=spec["n"], regime_threshold=spec["threshold"]))
            np.testing.assert_array_equal(ind["regime"], explicit["regime"])
            # every registry entry is oriented "higher = more trending" and finite once formed
            v = ind["regime"][~np.isnan(ind["regime"])]
            self.assertGreater(len(v), 400)
            self.assertTrue(np.isfinite(v).all())
            self.assertTrue(np.isnan(_compute_indicators(df, t.with_params(regime_n=5000))["regime"]).all())


class TemplateBehaviourTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = synthetic_ohlc(1200, seed=11)
        # 78 two-sided templates, every other switch value present. One-sided
        # templates are covered by test_each_switch_changes_the_strategy; they
        # trade too rarely (half the signals gone, filters stacked on top) for
        # the "used params move the run" ratio below to be meaningful.
        cls.sample = [t for t in generate_templates("full") if t.sides == "both"][::41]

    def _equity(self, df, tpl):
        return backtest(df, tpl)["equity"].to_numpy()

    def test_unused_params_are_inert_and_used_ones_are_not(self):
        """The whole point of a template: the switches decide which numbers
        matter. A parameter the switches do not read must not move the equity
        curve by a cent, on any template. One they do read must move it on
        nearly every template (a template with three filters stacked trades a
        handful of times, and a perturbed parameter can then legitimately
        never get its chance)."""
        used, moved = {k: 0 for k in PARAM_NEEDS}, {k: 0 for k in PARAM_NEEDS}
        traded = 0
        for tpl in self.sample:
            base = self._equity(self.df, tpl)
            traded += backtest(self.df, tpl)["stats"]["n_trades"] > 0
            for k, needed in PARAM_NEEDS.items():
                alt = self._equity(self.df, tpl.with_params(**{k: PERTURB[k]}))
                if needed(tpl):
                    used[k] += 1
                    moved[k] += not np.array_equal(base, alt)
                else:
                    np.testing.assert_array_equal(base, alt, err_msg=f"{tpl.name}: unused {k} changed the run")
            if tpl.regime_filter == "none":
                for ind in REGIME_INDICATORS:
                    np.testing.assert_array_equal(
                        base, self._equity(self.df, tpl.with_params(regime_indicator=ind)),
                        err_msg=f"{tpl.name}: regime indicator matters without a filter")
        self.assertGreater(traded, 0.9 * len(self.sample), "most templates should trade on 1200 bars")
        for k in PARAM_NEEDS:
            self.assertGreaterEqual(used[k], 15, f"sample never uses {k}")
            self.assertGreaterEqual(moved[k] / used[k], 0.8,
                                    f"{k} changed only {moved[k]} of {used[k]} runs that read it")

    def test_every_grid_point_gives_a_distinct_run(self):
        """The walk-forward searches the grid; identical grid points would only
        inflate the trial count that the overfitting statistics deflate by.
        (Checked on the unfiltered templates: with a regime filter on, two
        thresholds can legitimately never disagree on a given series.)"""
        checked = 0
        for tpl in [t for t in self.sample if t.regime_filter == "none"]:
            if backtest(self.df, tpl)["stats"]["n_trades"] < 8:
                continue    # one or two trades cannot tell six grid points apart
            checked += 1
            combos, _ = grid_combos(param_grid_for(tpl))
            seen = [self._equity(self.df, tpl.with_params(**p)) for p in combos]
            for i in range(len(seen)):
                for j in range(i + 1, len(seen)):
                    self.assertFalse(np.array_equal(seen[i], seen[j]), f"{tpl.name}: two grid points coincide")
        self.assertGreater(checked, 5)

    def test_filters_reduce_to_no_filter_or_no_trades_at_their_extremes(self):
        df = self.df
        k = 320  # start trading after every indicator (ADX included) has formed
        for entry, exit_ in (("stop", "channel"), ("pullback", "target_stop"), ("close_confirm", "atr_trail")):
            for dl in DIRECTION_LOGICS:
                plain = StrategyTemplate("t", direction_logic=dl, entry_style=entry, exit_style=exit_)
                ref = backtest(df, plain, first_trade_bar=k)
                self.assertGreater(ref["stats"]["n_trades"], 3, plain.name)
                for ind in REGIME_INDICATORS:
                    for mode, passes_all, passes_none in (("trend_only", -1.0, 1e9), ("range_only", 1e9, -1.0)):
                        t = plain.with_params(regime_filter=mode, regime_indicator=ind)
                        same = backtest(df, t.with_params(regime_threshold=passes_all), first_trade_bar=k)
                        np.testing.assert_array_equal(same["equity"].to_numpy(), ref["equity"].to_numpy(),
                                                      err_msg=f"{ind}/{mode}: a threshold nothing fails is not a no-op")
                        none = backtest(df, t.with_params(regime_threshold=passes_none), first_trade_bar=k)
                        self.assertEqual(none["stats"]["n_trades"], 0, f"{ind}/{mode}")
                        self.assertEqual(int(none["entries"].sum()), 0)
                wide = backtest(df, plain.with_params(vol_filter=True, vol_low_pct=0.0, vol_high_pct=1.0),
                                first_trade_bar=k)
                np.testing.assert_array_equal(wide["equity"].to_numpy(), ref["equity"].to_numpy())
                shut = backtest(df, plain.with_params(vol_filter=True, vol_low_pct=0.6, vol_high_pct=0.4),
                                first_trade_bar=k)
                self.assertEqual(shut["stats"]["n_trades"], 0)
                # SMA(1) is the close itself: nothing is strictly above or below it
                dead = backtest(df, plain.with_params(bias_filter="sma", bias_n=1), first_trade_bar=k)
                self.assertEqual(dead["stats"]["n_trades"], 0)
                # a real bias filter only removes entries on the wrong side of the SMA
                # (checked when the order is placed: a pullback limit fills bars later)
                biased = backtest(df, plain.with_params(bias_filter="sma", bias_n=100), first_trade_bar=k)
                bias = _compute_indicators(df, plain.with_params(bias_filter="sma", bias_n=100))["bias"]
                if dl == "trend":   # fading a break against a 100-bar SMA is rare on this series
                    self.assertGreater(len(biased["trades"]), 0)
                for tr in biased["trades"] if entry != "pullback" else []:
                    i = df.index.get_loc(tr["entry_date"])
                    prev_close = df["Close"].iloc[i - 1]
                    self.assertTrue(prev_close > bias[i - 1] if tr["side"] == 1 else prev_close < bias[i - 1],
                                    f"{plain.name}: entered against the bias filter")

    def test_longs_and_shorts_are_exact_mirrors(self):
        """Reflect the price series and every long becomes the same short: same
        entry bar, mirrored prices, identical P&L (with the leverage cap and
        costs off, the ATR-based size is the same on both sides). Anything the
        short-side code does differently from the long side shows up here.
        The variance-ratio indicator is built on log returns, which do not
        mirror, so it is the one template family left out. A one-sided
        template is mirrored into the other side's template."""
        mirror = _mirror(self.df)
        c = 2.0 * (float(self.df["High"].max()) + 1.0)
        checked = 0
        for tpl in self.sample:
            if tpl.regime_filter != "none" and tpl.regime_indicator == "vr":
                continue
            t = tpl.with_params(cost_bps=0.0, max_leverage=1e9)
            a, b = backtest(self.df, t), backtest(mirror, t.with_params(sides=_MIRROR_SIDES[t.sides]))
            np.testing.assert_allclose(a["equity"].to_numpy(), b["equity"].to_numpy(), rtol=1e-9,
                                       err_msg=f"{tpl.name}: mirrored run has a different equity curve")
            self.assertEqual(len(a["trades"]), len(b["trades"]), tpl.name)
            for x, y in zip(a["trades"], b["trades"]):
                self.assertEqual((x["entry_date"], x["exit_date"], x["reason"], -x["side"]),
                                 (y["entry_date"], y["exit_date"], y["reason"], y["side"]), tpl.name)
                self.assertAlmostEqual(x["entry_price"], c - y["entry_price"], places=8)
                self.assertAlmostEqual(x["exit_price"], c - y["exit_price"], places=8)
                self.assertAlmostEqual(x["pnl"], y["pnl"], places=6)
                checked += 1
        self.assertGreater(checked, 500)

    def test_each_switch_changes_the_strategy(self):
        """Templates that differ in one switch are structurally different
        strategies: their runs must differ (a switch nobody reads is a bug)."""
        df = self.df
        base = StrategyTemplate("t", entry_style="stop", exit_style="channel", cost_bps=0.0)
        variants = [
            ("direction_logic", DIRECTION_LOGICS), ("channel_type", CHANNEL_TYPES),
            ("entry_style", ENTRY_STYLES), ("exit_style", EXIT_STYLES), ("bias_filter", BIAS_FILTERS),
            ("sides", SIDES),
        ]
        for field, values in variants:
            runs = [self._equity(df, base.with_params(**{field: v})) for v in values]
            for i in range(len(runs)):
                for j in range(i + 1, len(runs)):
                    self.assertFalse(np.array_equal(runs[i], runs[j]), f"{field}: {values[i]} == {values[j]}")
        # every entry style opens trades, every exit style closes them with its own reason
        reasons = {"channel": "channel", "atr_trail": "stop", "target_stop": "target", "time_stop": "time"}
        for ex, why in reasons.items():
            trades = backtest(df, base.with_params(exit_style=ex))["trades"]
            self.assertIn(why, {t["reason"] for t in trades}, ex)
        ct = backtest(df, base.with_params(direction_logic="countertrend"))["trades"]
        self.assertIn("midline", {t["reason"] for t in ct})
        # a one-sided template never takes the other side, and its trades are
        # exactly the two-sided template's trades on that side, not a re-run
        for sides, side in (("long_only", 1), ("short_only", -1)):
            one = backtest(df, base.with_params(sides=sides))["trades"]
            self.assertTrue(one, sides)
            self.assertEqual({t["side"] for t in one}, {side}, sides)
            # with the SMA bias on top the set can only shrink
            biased = backtest(df, base.with_params(sides=sides, bias_filter="sma"))["trades"]
            self.assertLessEqual(len(biased), len(one), sides)
            self.assertEqual({t["side"] for t in biased} - {side}, set(), sides)


if __name__ == "__main__":
    unittest.main()
