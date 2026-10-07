"""extra_utils/online_study/run_study: row schema, cost monotonicity, turnover cross-check."""

import unittest

import numpy as np

import generator
from data import synthetic_ohlc
from extra_utils.online_study.run_study import evaluate, breakeven

COLS = {"dataset", "template", "family", "sides", "cost_bps", "cost_mult", "oos_sharpe", "ann_return", "ann_vol",
        "max_dd", "n_trades", "avg_gross_exposure", "sharpe_h1", "sharpe_h2", "n_oos_bars", "turnover_direct",
        "turnover", "cost_drag_5bp", "breakeven_bps"}


class TestOnlineStudy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        df = synthetic_ohlc(n_bars=1500, seed=3)
        tpl = generator.generate_templates("online_forecast")[:1]
        cls.rows = evaluate(df, tpl, cost_bps_list=(0, 10), sides=("both",), jobs=1, dataset="syn",
                            family="online_forecast")
        cls.t = cls.rows[cls.rows["family"] != "benchmark"].sort_values("cost_bps")

    def test_schema(self):
        self.assertTrue(COLS <= set(self.rows.columns))
        self.assertEqual(len(self.t), 2)
        self.assertEqual((self.rows["family"] == "benchmark").sum(), 1)

    def test_return_not_increasing_in_cost(self):
        r = self.t["ann_return"].to_numpy()
        self.assertLessEqual(r[1], r[0] + 1e-12)

    def test_turnover_agrees_with_cost_drag(self):
        row = self.t.iloc[-1]
        self.assertGreater(row["turnover_direct"], 0)
        ratio = row["turnover_direct"] / row["turnover"]
        self.assertTrue(0.5 <= ratio <= 2.0, ratio)

    def test_spread_attrs_instrument_with_tick_runs_the_per_unit_sweep(self):
        from extra_utils.online_study import synth
        df = synth.calendar_spread(900, 1)
        self.assertIn("tick", df.attrs["instrument"])
        tpl = generator.generate_templates("online_forecast")[:1]
        rows = evaluate(df, tpl, sides=("both",), jobs=1, dataset="cal", instrument=df.attrs["instrument"],
                        train=300, test=100)
        self.assertEqual(int(rows["cost_mult"].notna().sum()), 5)    # the swept multiples (the benchmark row has none)
        self.assertTrue({"cost_drag_per_mult", "breakeven_mult"} <= set(rows.columns))
        self.assertGreater(rows.loc[rows["family"] != "benchmark", "turnover_direct"].max(), 0)

    def test_breakeven(self):
        self.assertEqual(breakeven([0, 5, 10], [-1, -2, -3]), 0.0)
        self.assertEqual(breakeven([0, 5, 10], [1, 2, 3]), np.inf)
        self.assertAlmostEqual(breakeven([0, 5, 10], [2, 1, -1]), 7.5)


if __name__ == "__main__":
    unittest.main()
