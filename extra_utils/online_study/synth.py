"""Synthetic daily OHLC series with a KNOWN planted edge, for testing online
learners (Hedge over breakout experts, online ridge forecaster).

Every generator is `fn(n_bars=3000, seed=0, **params) -> DataFrame` (index
`Date`, business days from 2010-01-04; Open/High/Low/Close/Volume) and fills
`df.attrs["truth"]` with the planted structure. Spreads add a `Roll` column
(1.0 on the bar at whose close the contract rolled) and `df.attrs["instrument"]`
(StrategyTemplate field names; run with cost_bps=0).

OHLC: open = previous close + overnight gap carrying `gap_share` of the day's
variance; the intrabar path is a Brownian bridge open -> close (its max/min are
drawn exactly), so High >= max(O, C) and Low <= min(O, C). The close-to-close
process is exactly the planted one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

START = "2010-01-04"
BARS_PER_YEAR = 252


# ----------------------------------------------------------------- helpers
class Truth(dict):
    """The planted-structure dict. pandas compares `attrs` with == when it concatenates frames
    (the engine's true_range does), which raises on a plain dict holding numpy arrays; this
    compares by identity instead."""

    def __eq__(self, other):
        return self is other

    def __ne__(self, other):
        return self is not other

    __hash__ = object.__hash__


def _index(n):
    return pd.bdate_range(START, periods=n, name="Date")


def _unit_shocks(rng, n, t_shocks):
    """Unit-variance shocks: normal, or Student-t(4) rescaled to variance 1."""
    if t_shocks:
        return rng.standard_t(4, n) / np.sqrt(2.0)  # var of t4 = 2
    return rng.standard_normal(n)


class _Garch:
    """GARCH(1,1) conditional sd, one step at a time (const vol if off)."""

    def __init__(self, vol, garch, alpha=0.08, beta=0.90):
        self.v = vol ** 2
        self.on = garch
        self.a, self.b = alpha, beta
        self.omega = self.v * (1.0 - alpha - beta)
        self.h = self.v

    def sd(self):
        return np.sqrt(self.h)

    def update(self, eps):
        if self.on:
            self.h = self.omega + self.a * eps * eps + self.b * self.h


def _bridge_extremes(d, s, rng):
    """Max and min of a Brownian bridge from 0 to d (vol s over the bar)."""
    n = len(d)
    u1 = rng.uniform(1e-12, 1.0, n)
    u2 = rng.uniform(1e-12, 1.0, n)
    s2 = s * s
    mx = 0.5 * (d + np.sqrt(d * d - 2.0 * s2 * np.log(u1)))
    mn = 0.5 * (d - np.sqrt(d * d - 2.0 * s2 * np.log(u2)))
    return mx, mn


def _volume(rng, n, shock):
    base = 1e5 * np.exp(0.3 * rng.standard_normal(n))
    return np.maximum(1, (base * (1.0 + 0.5 * np.abs(shock))).astype(np.int64))


def _frame_from_logret(r, sd, rng, gap_share, p0=100.0, shock=None):
    """OHLC from close-to-close log returns r with per-bar sd (for the gap split)."""
    n = len(r)
    gap = gap_share * r + rng.standard_normal(n) * sd * np.sqrt(gap_share * (1 - gap_share))
    intra = r - gap
    logc = np.log(p0) + np.cumsum(r)
    prev = np.concatenate([[np.log(p0)], logc[:-1]])
    logo = prev + gap
    mx, mn = _bridge_extremes(intra, sd * np.sqrt(1.0 - gap_share), rng)
    close, op = np.exp(logc), np.exp(logo)
    high, low = np.exp(logo + mx), np.exp(logo + mn)
    high = np.maximum(high, np.maximum(op, close))
    low = np.minimum(low, np.minimum(op, close))
    if shock is None:
        shock = r / sd
    return pd.DataFrame({"Open": op, "High": high, "Low": low, "Close": close,
                         "Volume": _volume(rng, n, shock)}, index=_index(n))


def _frame_from_points(close, vol, rng, gap_share, tick, shock):
    """OHLC (additive, in points) for a spread close path."""
    n = len(close)
    d = np.diff(np.concatenate([[close[0]], close]))
    d[0] = rng.standard_normal() * vol
    prev = close - d
    gap = gap_share * d + rng.standard_normal(n) * vol * np.sqrt(gap_share * (1 - gap_share))
    op = prev + gap
    mx, mn = _bridge_extremes(close - op, vol * np.sqrt(1.0 - gap_share), rng)
    high = np.maximum(op + mx, np.maximum(op, close))
    low = np.minimum(op + mn, np.minimum(op, close))
    q = lambda x: np.round(np.round(x / tick) * tick, 10)  # monotone: keeps H/L consistency
    return pd.DataFrame({"Open": q(op), "High": q(high), "Low": q(low), "Close": q(close),
                         "Volume": _volume(rng, n, shock)}, index=_index(n))


def _ewma_signal(u, halflife):
    """Causal EWMA of u (value at t uses u[:t]), scaled to ~unit std for iid unit u."""
    a = 1.0 - 0.5 ** (1.0 / halflife)
    return a, np.sqrt(a / (2.0 - a))


def _signal_sd(a, b):
    """Stationary sd of e_t = EWMA(a) of past u when u = b*e/sd + z (the signal feeds back into the
    returns it is built from, inflating its sd above sqrt(a/(2-a))); solves the fixed point
    sd = a/sqrt(1-rho^2), rho = 1 - a + a*b/sd, by bisection."""
    f = lambda sd: sd - a / np.sqrt(max(1e-12, 1.0 - (1.0 - a + a * b / sd) ** 2))
    lo = b * (1 + 1e-9)  # rho < 1  <=>  sd > b
    hi = 1e3
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


# ------------------------------------------------------- return generators
def _sim_u(n, rng, vol, garch, t_shocks, gap_share, mean_fn, p0=100.0, state=None):
    """Generic loop: u_t = mean_fn(t, u_hist) + z_t ; r_t = sd_t * u_t.
    Returns r, sd, u, z."""
    z = _unit_shocks(rng, n, t_shocks)
    g = _Garch(vol, garch)
    r, sd, u = np.zeros(n), np.zeros(n), np.zeros(n)
    for t in range(n):
        s = g.sd()
        m = mean_fn(t, u)
        u[t] = m + z[t]
        sd[t] = s
        r[t] = s * u[t]
        g.update(s * z[t])
    return r, sd, u, z


def random_walk(n_bars=3000, seed=0, vol=0.01, garch=True, t_shocks=False, gap_share=0.3):
    """Null: zero-drift GBM, no edge. Any learner Sharpe here is false discovery."""
    rng = np.random.default_rng(seed)
    r, sd, u, _ = _sim_u(n_bars, rng, vol, garch, t_shocks, gap_share, lambda t, u: 0.0)
    df = _frame_from_logret(r, sd, rng, gap_share)
    df.attrs["truth"] = Truth(kind="random_walk", edge="none", ic=0.0, vol=vol, garch=garch,
                             t_shocks=t_shocks, sigma=sd)
    return df


def tsmom(n_bars=3000, seed=0, ic=0.03, halflife=60, vol=0.01, garch=True, t_shocks=False,
          gap_share=0.3):
    """Slow trend: u_t = b*s_t + z_t, s_t = unit-scaled EWMA(halflife) of past u.
    b = ic/sqrt(1-ic^2) so corr(s_t, u_t) = ic (annual Sharpe ~ ic*sqrt(252))."""
    rng = np.random.default_rng(seed)
    b = ic / np.sqrt(1.0 - ic * ic)
    a, _ = _ewma_signal(None, halflife)
    nrm = _signal_sd(a, b)
    sig = np.zeros(n_bars)
    state = {"e": 0.0}

    def mean_fn(t, u):
        if t > 0:
            state["e"] += a * (u[t - 1] - state["e"])
        sig[t] = state["e"] / nrm
        return b * sig[t]

    r, sd, u, _ = _sim_u(n_bars, rng, vol, garch, t_shocks, gap_share, mean_fn)
    df = _frame_from_logret(r, sd, rng, gap_share)
    df.attrs["truth"] = Truth(kind="tsmom", edge="slow trend", ic=ic, b=b, halflife=halflife,
                             signal=sig, u=u, sigma=sd, expected_sharpe=ic * np.sqrt(BARS_PER_YEAR))
    return df


def reversal_trend(n_bars=3000, seed=0, ic_slow=0.02, ic_fast=-0.05, halflife=100, fast_lags=5,
                   vol=0.01, garch=True, t_shocks=False, gap_share=0.3):
    """Weak slow trend (follow) plus lag-1..fast_lags reversal (fade).
    u_t = b_s*s_t + b_f*q_t + z_t, q_t = sum_{k<=L} u_{t-k}/sqrt(L) (unit scale),
    so IC(q, u) ~ ic_fast < 0 and IC(s, u) ~ ic_slow (slightly off: s, q correlate)."""
    rng = np.random.default_rng(seed)
    bs = ic_slow / np.sqrt(1.0 - ic_slow ** 2)
    bf = ic_fast / np.sqrt(1.0 - ic_fast ** 2)
    a, _ = _ewma_signal(None, halflife)
    nrm = _signal_sd(a, bs)
    sig, fast = np.zeros(n_bars), np.zeros(n_bars)
    st = {"e": 0.0}

    def mean_fn(t, u):
        if t > 0:
            st["e"] += a * (u[t - 1] - st["e"])
        sig[t] = st["e"] / nrm
        fast[t] = u[max(0, t - fast_lags):t].sum() / np.sqrt(fast_lags)
        return bs * sig[t] + bf * fast[t]

    r, sd, u, _ = _sim_u(n_bars, rng, vol, garch, t_shocks, gap_share, mean_fn)
    df = _frame_from_logret(r, sd, rng, gap_share)
    df.attrs["truth"] = Truth(kind="reversal_trend", edge="follow slow, fade fast", ic_slow=ic_slow,
                             ic_fast=ic_fast, signal=sig, fast_signal=fast, fast_lags=fast_lags,
                             halflife=halflife, u=u, sigma=sd)
    return df


def _markov_regimes(n, rng, dur0, dur1):
    """Two-state chain: state 0 mean duration dur0, state 1 dur1; returns int array."""
    reg = np.zeros(n, dtype=int)
    s = int(rng.integers(2))
    for t in range(n):
        reg[t] = s
        if rng.random() < 1.0 / (dur1 if s == 1 else dur0):
            s = 1 - s
    return reg


def regime_switch(n_bars=3000, seed=0, trend_dur=120, mr_dur=120, trend_drift=0.06, trend_phi=0.08,
                  mr_kappa=0.02, mr_phi=-0.05, vol=0.01, garch=True, t_shocks=False, gap_share=0.3):
    """Markov switching. Regime 1 = trend: persistent drift (+-trend_drift*sigma per bar, random
    sign per episode) and AR(+trend_phi). Regime 0 = mean reversion: OU pull -mr_kappa*(logp - L)
    toward a level L (set at the episode start) and AR(mr_phi). truth['regime'] is the per-bar
    array (1 trend, 0 mean-reverting), truth['direction'] the drift sign in trend episodes."""
    rng = np.random.default_rng(seed)
    reg = _markov_regimes(n_bars, rng, mr_dur, trend_dur)
    direction = np.zeros(n_bars)
    sgn = 1.0
    for t in range(n_bars):
        if t == 0 or reg[t] != reg[t - 1]:
            sgn = rng.choice([-1.0, 1.0])
        direction[t] = sgn if reg[t] == 1 else 0.0
    st = {"logp": 0.0, "L": 0.0}
    sdv = {"sd": vol}
    logp = np.zeros(n_bars)

    def mean_fn(t, u):
        if t > 0 and reg[t] != reg[t - 1]:
            st["L"] = logp[t - 1]
        lp = logp[t - 1] if t > 0 else 0.0
        prev_u = u[t - 1] if t > 0 else 0.0
        if reg[t] == 1:
            m = trend_drift * direction[t] + trend_phi * prev_u
        else:
            m = -mr_kappa * (lp - st["L"]) / vol + mr_phi * prev_u
        return m

    # logp needs the realised returns: run the loop here instead of _sim_u
    z = _unit_shocks(rng, n_bars, t_shocks)
    g = _Garch(vol, garch)
    r, sd, u = np.zeros(n_bars), np.zeros(n_bars), np.zeros(n_bars)
    for t in range(n_bars):
        s = g.sd()
        u[t] = mean_fn(t, u) + z[t]
        sd[t], r[t] = s, s * u[t]
        logp[t] = (logp[t - 1] if t > 0 else 0.0) + r[t]
        g.update(s * z[t])
    df = _frame_from_logret(r, sd, rng, gap_share)
    df.attrs["truth"] = Truth(kind="regime_switch", edge="trend in regime 1, mean-reversion in 0",
                             regime=reg, direction=direction, u=u, sigma=sd, trend_dur=trend_dur,
                             mr_dur=mr_dur)
    return df


def ar1(n_bars=3000, seed=0, phi=0.2, vol=0.01, garch=False, t_shocks=False, gap_share=0.3):
    """Plain AR(1) daily returns, unit-scale u_t = phi*u_{t-1} + sqrt(1-phi^2)*z_t."""
    rng = np.random.default_rng(seed)
    k = np.sqrt(1.0 - phi * phi)
    z = _unit_shocks(rng, n_bars, t_shocks) * k
    g = _Garch(vol, garch)
    r, sd, u = np.zeros(n_bars), np.zeros(n_bars), np.zeros(n_bars)
    for t in range(n_bars):
        s = g.sd()
        u[t] = (phi * u[t - 1] if t else 0.0) + z[t]
        sd[t], r[t] = s, s * u[t]
        g.update(s * u[t])
    df = _frame_from_logret(r, sd, rng, gap_share)
    df.attrs["truth"] = Truth(kind="ar1", edge="return autocorrelation", phi=phi, lag1_ac=phi, u=u, sigma=sd)
    return df


# -------------------------------------------------------- spread generators
def _instrument(cost_per_unit=12.5, roll_cost_per_unit=25.0):
    return dict(point_value=1000.0, margin_per_unit=3000.0, cost_per_unit=cost_per_unit,
                roll_cost_per_unit=roll_cost_per_unit, tick=0.01)


def _spread_core(n, rng, mean_path, half_life, vol, jumps_per_year, jump_mult, roll_every,
                 roll_jump_sd, gap_share, tick, instrument_kw, truth_extra, kind):
    phi = 0.5 ** (1.0 / half_life)
    eps = rng.standard_normal(n) * vol
    jump = np.zeros(n)
    jbars = np.where(rng.random(n) < jumps_per_year / BARS_PER_YEAR)[0]
    jump[jbars] = rng.choice([-1.0, 1.0], len(jbars)) * jump_mult * vol
    dev = np.zeros(n)
    for t in range(n):
        dev[t] = (phi * dev[t - 1] if t else 0.0) + eps[t] + jump[t]
    latent = mean_path + dev
    roll = np.zeros(n)
    roll_bars = np.arange(roll_every - 1, n, roll_every)
    roll[roll_bars] = 1.0
    # listed spread of the held contract = latent + contract offset; at a roll close the listed
    # spread jumps by (new offset - old offset)
    offs = rng.standard_normal(len(roll_bars) + 1) * roll_jump_sd
    offset = np.zeros(n)
    k = 0
    for t in range(n):
        offset[t] = offs[k]
        if roll[t]:
            k += 1
            offset[t] = offs[k]
    close = latent + offset
    df = _frame_from_points(close, vol, rng, gap_share, tick, shock=eps / vol)
    df["Roll"] = roll
    df.attrs["instrument"] = _instrument(**instrument_kw)
    df.attrs["truth"] = Truth(kind=kind, half_life=half_life, phi=phi, vol=vol, mean=mean_path,
                             deviation=dev, latent=latent, roll_bars=roll_bars,
                             roll_gap=np.diff(offs),
                             jump_bars=jbars, expected_lag1_ac_diff=-(1.0 - phi) / 2.0, **truth_extra)
    return df


def calendar_spread(n_bars=3000, seed=0, half_life=10, vol=0.07, base=0.4, amplitude=0.5,
                    mean_drift_sd=0.01, mean_revert_hl=500, jumps_per_year=4, jump_mult=3.0,
                    roll_every=21, roll_jump_sd=0.02, gap_share=0.3, cost_per_unit=12.5,
                    roll_cost_per_unit=25.0, tick=0.01):
    """Front-back spread in points (crosses zero): seasonal sine (annual, `amplitude`) + slowly
    wandering mean + OU deviation (half-life `half_life`, innovation sd `vol` points) + Poisson
    supply jumps (sign random, jump_mult*vol) that decay at the OU rate. Edge = fade deviation
    from the (slow) mean: lag-1 ac of differences ~ -(1-phi)/2."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_bars)
    season = amplitude * np.sin(2 * np.pi * t / BARS_PER_YEAR + rng.uniform(0, 2 * np.pi))
    w = np.zeros(n_bars)
    k = 0.5 ** (1.0 / mean_revert_hl)
    sh = rng.standard_normal(n_bars) * mean_drift_sd
    for i in range(1, n_bars):
        w[i] = k * w[i - 1] + sh[i]
    mean_path = base + season + w
    return _spread_core(n_bars, rng, mean_path, half_life, vol, jumps_per_year, jump_mult,
                        roll_every, roll_jump_sd, gap_share, tick,
                        dict(cost_per_unit=cost_per_unit, roll_cost_per_unit=roll_cost_per_unit),
                        dict(season=season, edge="fade OU deviation from mean"), "calendar_spread")


def trending_spread(n_bars=3000, seed=0, half_life=8, vol=0.07, trend_dur=120, flat_dur=60,
                    trend_drift=0.012, base=0.3, pull_hl=400, jumps_per_year=2, jump_mult=3.0,
                    roll_every=21, roll_jump_sd=0.02, gap_share=0.3, cost_per_unit=12.5,
                    roll_cost_per_unit=25.0, tick=0.01):
    """Spread whose OU mean itself trends: persistent regimes (contango deepening / backwardation:
    mean drifts +-trend_drift points/bar for ~trend_dur bars, random sign per episode; ~flat_dur
    bars of no drift between) with a weak pull to `base`; fast OU reversion (half-life
    `half_life`) around the moving mean. Slow trend + fast reversion, in points."""
    rng = np.random.default_rng(seed)
    reg = _markov_regimes(n_bars, rng, flat_dur, trend_dur)
    drift = np.zeros(n_bars)
    sgn = 1.0
    for t in range(n_bars):
        if t == 0 or reg[t] != reg[t - 1]:
            sgn = rng.choice([-1.0, 1.0])
        drift[t] = sgn * trend_drift if reg[t] == 1 else 0.0
    m = np.zeros(n_bars)
    k = 0.5 ** (1.0 / pull_hl)
    for i in range(1, n_bars):
        m[i] = k * m[i - 1] + drift[i]
    mean_path = base + m
    return _spread_core(n_bars, rng, mean_path, half_life, vol, jumps_per_year, jump_mult,
                        roll_every, roll_jump_sd, gap_share, tick,
                        dict(cost_per_unit=cost_per_unit, roll_cost_per_unit=roll_cost_per_unit),
                        dict(regime=reg, mean_drift=drift, edge="follow slow mean trend, fade fast deviation"),
                        "trending_spread")


SYNTH = {"random_walk": random_walk, "tsmom": tsmom, "reversal_trend": reversal_trend,
         "regime_switch": regime_switch, "calendar_spread": calendar_spread,
         "trending_spread": trending_spread, "ar1": ar1}


def make_suite(seed=0, n_bars=3000):
    """Default suite: name -> DataFrame."""
    s = seed
    return {
        "random_walk_a": random_walk(n_bars, s),
        "random_walk_b": random_walk(n_bars, s + 1),
        "tsmom_ic02": tsmom(n_bars, s + 2, ic=0.02),
        "tsmom_ic04": tsmom(n_bars, s + 3, ic=0.04),
        "reversal_trend": reversal_trend(n_bars, s + 4),
        "regime_switch": regime_switch(n_bars, s + 5),
        "calendar_spread_hl5": calendar_spread(n_bars, s + 6, half_life=5),
        "calendar_spread_hl20": calendar_spread(n_bars, s + 7, half_life=20),
        "trending_spread": trending_spread(n_bars, s + 8),
    }
