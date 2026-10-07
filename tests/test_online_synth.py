import numpy as np
import pandas as pd
import pytest

from extra_utils.online_study import synth


def _check_ohlc(df, n):
    assert len(df) == n and df.index.name == "Date"
    assert df.index[0] == pd.Timestamp("2010-01-04")
    assert not df[["Open", "High", "Low", "Close", "Volume"]].isna().any().any()
    assert (df.High >= df[["Open", "Close"]].max(axis=1) - 1e-12).all()
    assert (df.Low <= df[["Open", "Close"]].min(axis=1) + 1e-12).all()
    assert (df.Volume > 0).all() and np.issubdtype(df.Volume.dtype, np.integer)
    assert df.attrs["truth"]


def _ac1(x):
    return float(np.corrcoef(x[1:], x[:-1])[0, 1])


@pytest.mark.parametrize("name", list(synth.SYNTH))
def test_ohlc_and_truth(name):
    df = synth.SYNTH[name](n_bars=800, seed=1)
    _check_ohlc(df, 800)
    if name.endswith("spread"):
        assert "Roll" in df and df.Roll.sum() > 5 and set(df.Roll.unique()) <= {0.0, 1.0}
        assert set(df.attrs["instrument"]) >= {"point_value", "margin_per_unit", "cost_per_unit",
                                                "roll_cost_per_unit", "tick"}
        assert len(df.attrs["truth"]["roll_bars"]) == int(df.Roll.sum())


def test_spreads_cross_zero():
    for fn in (synth.calendar_spread, synth.trending_spread):
        c = fn(3000, 0).Close
        assert c.min() < 0 < c.max()


def test_ar1_sign():
    for phi in (0.2, -0.2):
        c = synth.ar1(20000, 3, phi=phi).Close
        assert abs(_ac1(np.diff(np.log(c.values))) - phi) < 0.03


def test_calendar_spread_negative_ac():
    c = synth.calendar_spread(20000, 4, half_life=10).Close.values
    assert _ac1(np.diff(c)) < -0.008


def test_tsmom_ic_recovered():
    for ic in (0.02, 0.05):
        df = synth.tsmom(20000, 5, ic=ic)
        tr = df.attrs["truth"]
        got = np.corrcoef(tr["signal"], tr["u"])[0, 1]
        assert abs(got - ic) < 0.012  # SE ~ 1/sqrt(20000) = 0.007
        # and from prices alone
        r = np.diff(np.log(df.Close.values))
        assert np.corrcoef(tr["signal"][1:], r)[0, 1] > 0


def test_random_walk_null():
    r = np.diff(np.log(synth.random_walk(20000, 6).Close.values))
    assert abs(_ac1(r)) < 0.03


def test_reversal_trend_fast_negative():
    tr = synth.reversal_trend(20000, 7).attrs["truth"]
    assert np.corrcoef(tr["fast_signal"], tr["u"])[0, 1] < -0.02
    assert np.corrcoef(tr["signal"], tr["u"])[0, 1] > 0


def test_regime_truth_and_suite():
    suite = synth.make_suite(0, n_bars=600)
    assert len(suite) == 9
    reg = suite["regime_switch"].attrs["truth"]["regime"]
    assert len(reg) == 600 and set(np.unique(reg)) <= {0, 1}
    for df in suite.values():
        _check_ohlc(df, 600)


def test_engine_accepts_spread():
    import generator
    import strategy
    df = synth.calendar_spread(600, 0)
    inst = {k: v for k, v in df.attrs["instrument"].items() if k != "tick"}
    tpl = generator.generate_templates("online_forecast")[2].with_params(**inst, cost_bps=0)
    tpl.validate()
    res = strategy.backtest(df, tpl)
    assert len(res["equity"]) == 600 and np.isfinite(res["equity"].values).all()


def test_suites_of_different_seeds_share_no_stream():
    a, b = synth.make_suite(0, n_bars=300), synth.make_suite(1, n_bars=300)
    assert not np.allclose(a["random_walk_b"]["Close"].to_numpy(), b["random_walk_a"]["Close"].to_numpy())
    again = synth.make_suite(1, n_bars=300)
    np.testing.assert_array_equal(again["tsmom_ic02"]["Close"].to_numpy(), b["tsmom_ic02"]["Close"].to_numpy())
