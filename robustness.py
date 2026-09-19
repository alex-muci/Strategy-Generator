"""
robustness.py
-------------
Backtest-overfitting diagnostics, mostly from Marcos Lopez de Prado,
"Advances in Financial Machine Learning" (AFML, 2018), plus the two
Monte Carlo checks financial-hacker.com keeps recommending.

A single walk-forward path is ONE realisation. With a strategy
GENERATOR the danger is not a bad path, it is the selection step:
pick the best of hundreds of templates on their "out-of-sample" curves
and the curves are not out-of-sample any more. These tools quantify
how much of what we see can be explained by that selection alone.

  trial_returns            per-bar returns of every grid point of a template
                           over the full history (the T x N "trials matrix")
  cpcv                     Combinatorial Purged Cross-Validation (AFML ch. 12):
                           many backtest PATHS, not one, with purge/embargo
  cscv_pbo                 Probability of Backtest Overfitting via CSCV
                           (Bailey, Borwein, Lopez de Prado, Zhu 2015; AFML 11.6)
  probabilistic_sharpe_ratio / deflated_sharpe_ratio   (AFML ch. 14/15)
  min_backtest_length      how long a track record is needed for N trials
  bootstrap_sharpe_pvalue  stationary-bootstrap Monte Carlo p-value for one
                           strategy (financial-hacker "Montecarlo Reality Check")
  reality_check            White's (2000) Reality Check for the best of a family
  hrp_weights              Hierarchical Risk Parity (AFML ch. 16) for the portfolio
"""

from __future__ import annotations
from itertools import combinations
from math import comb
import numpy as np
import pandas as pd
from scipy.stats import norm, skew as _skew, kurtosis as _kurtosis
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

from strategy import backtest, periods_per_year
from walkforward import smooth_scores, walk_forward, grid_combos

EULER_GAMMA = 0.5772156649015329


# --------------------------------------------------------------------------
# trials matrix
# --------------------------------------------------------------------------

def trial_returns(df: pd.DataFrame, tpl, combos: list, initial_equity: float = 100_000.0):
    """Backtest every param combo over the FULL history.
    Returns (R, E): float array T x N of per-bar returns and int8 array
    T x N flagging bars on which a trade was opened.

    The strategy is path-dependent, so a block cut out of R can start in the
    middle of a trade. That is serial dependence, not look-ahead; see the
    README ("Known approximations") for why cpcv's `purge_bars` stays at 0."""
    T, N = len(df), len(combos)
    R = np.zeros((T, N))
    E = np.zeros((T, N), dtype=np.int8)
    for j, params in enumerate(combos):
        res = backtest(df, tpl.with_params(**params), initial_equity=initial_equity)
        R[:, j] = res["returns"].to_numpy()
        E[:, j] = res["entries"]
    return R, E


def evaluate_template(
    df: pd.DataFrame,
    tpl,
    param_grid: dict,
    *,
    cpcv_groups: int = 8,
    cpcv_k: int = 2,
    cscv_partitions_n: int = 16,
    selection: str = "plateau",
    **wfa_kwargs,
) -> dict:
    """Walk-forward a template AND stress it, in one call.

    Returns the `walk_forward` dict plus:
      cpcv         : the CPCV Sharpe distribution (summary fields only)
      trial_blocks : CSCV block statistics of the trials matrix, so the caller
                     can pool PBO across a whole family without shipping every
                     T x N return matrix back from a worker process
      n_trials     : how many parameter combos were tried

    Shared by main.py (one asset) and etf_dashboard.py (several), so the two
    cannot drift apart on what "evaluated" means.
    """
    wfa = walk_forward(df, tpl, param_grid, selection=selection, **wfa_kwargs)
    combos, idx = grid_combos(param_grid)
    R, E = trial_returns(df, tpl, combos)
    cp = cpcv(R, E, idx, n_groups=cpcv_groups, k_test=cpcv_k,
              embargo_bars=2 * tpl.n_entry, selection=selection)
    wfa["cpcv"] = {k: v for k, v in cp.items()
                   if k in ("path_sharpes", "path_max_dd", "n_paths", "sharpe_mean",
                            "sharpe_std", "sharpe_min", "prob_sharpe_negative")}
    wfa["trial_blocks"] = cscv_block_stats(R, cscv_partitions_n)
    wfa["n_trials"] = R.shape[1]
    return wfa


def _sharpe_cols(R: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Annualized Sharpe of each column of R over the rows selected by mask."""
    X = R[mask]
    if len(X) < 2:
        return np.zeros(R.shape[1])
    sd = X.std(axis=0, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        sr = np.where(sd > 0, X.mean(axis=0) / sd * np.sqrt(periods_per_year()), 0.0)
    return sr


# --------------------------------------------------------------------------
# Combinatorial Purged Cross-Validation  (AFML 12.3 - 12.4)
# --------------------------------------------------------------------------

def cpcv_paths(n_groups: int, k_test: int):
    """Map (combination index, test group) -> path id, following AFML 12.4:
    with N groups and k test groups there are C(N,k)*k/N full backtest paths.
    Returns (list of test-group tuples, dict {(c, g): path})."""
    combos = list(combinations(range(n_groups), k_test))
    n_paths = comb(n_groups, k_test) * k_test // n_groups
    next_path = {g: 0 for g in range(n_groups)}
    assign = {}
    for c, groups in enumerate(combos):
        for g in groups:
            assign[(c, g)] = next_path[g]
            next_path[g] += 1
    assert all(v == n_paths for v in next_path.values())
    return combos, assign, n_paths


def cpcv(
    R: np.ndarray,
    E: np.ndarray | None = None,
    idx: np.ndarray | None = None,
    n_groups: int = 6,
    k_test: int = 2,
    embargo_bars: int = 10,
    purge_bars: int = 0,
    min_trades: int = 5,
    selection: str = "plateau",
) -> dict:
    """Combinatorial purged CV of the parameter selection.

    R : T x N per-bar returns of the N parameter combos (trials)
    E : T x N entry flags (for the min-trades rule); optional
    idx : N x D lattice indices (for 'plateau' selection); optional

    The T bars are cut into n_groups contiguous groups. For every choice
    of k_test groups as the test set, the remaining groups (minus an
    embargo after each test group and an optional purge before it) are
    the training set; the trial that scores best there is applied to
    the test groups. Test-group results are stitched into
    C(n_groups,k_test)*k_test/n_groups complete backtest paths, so the
    result is a DISTRIBUTION of OOS Sharpe ratios rather than one number.
    """
    T, N = R.shape
    bounds = np.linspace(0, T, n_groups + 1).astype(int)
    groups = [np.arange(bounds[g], bounds[g + 1]) for g in range(n_groups)]
    combos, assign, n_paths = cpcv_paths(n_groups, k_test)

    path_returns = np.full((T, n_paths), np.nan)
    chosen = np.zeros((len(combos),), dtype=int)

    for c, test_groups in enumerate(combos):
        train_mask = np.ones(T, dtype=bool)
        for g in test_groups:
            lo, hi = bounds[g], bounds[g + 1]
            train_mask[max(0, lo - purge_bars):min(T, hi + embargo_bars)] = False
        scores = _sharpe_cols(R, train_mask)
        if E is not None:
            n_tr = E[train_mask].sum(axis=0)
            scores = np.where(n_tr >= min_trades, scores, -np.inf)
        if not np.isfinite(scores).any():
            best = int(np.argmax(_sharpe_cols(R, train_mask)))
        elif selection == "plateau" and idx is not None:
            best = int(np.argmax(smooth_scores(scores, idx)))
        else:
            best = int(np.argmax(scores))
        chosen[c] = best
        for g in test_groups:
            p = assign[(c, g)]
            path_returns[groups[g], p] = R[groups[g], best]

    assert not np.isnan(path_returns).any(), "CPCV paths not fully covered"
    sr = np.array([_sharpe_cols(path_returns[:, [p]], np.ones(T, bool))[0] for p in range(n_paths)])
    eq = np.cumprod(1 + path_returns, axis=0)
    max_dd = (eq / np.maximum.accumulate(eq, axis=0) - 1).min(axis=0)
    return dict(
        path_returns=path_returns,
        path_sharpes=sr,
        path_max_dd=max_dd,
        n_paths=n_paths,
        sharpe_mean=float(sr.mean()),
        sharpe_std=float(sr.std(ddof=1)) if n_paths > 1 else 0.0,
        sharpe_min=float(sr.min()),
        prob_sharpe_negative=float((sr < 0).mean()),
        chosen_trials=chosen,
    )


# --------------------------------------------------------------------------
# Probability of Backtest Overfitting  (CSCV; AFML 11.6, Bailey et al. 2015)
# --------------------------------------------------------------------------

def cscv_partitions(T: int, n_partitions: int = 16) -> int:
    """Number of CSCV blocks actually used (always even)."""
    return n_partitions if n_partitions % 2 == 0 else n_partitions + 1


def cscv_block_stats(R: np.ndarray, n_partitions: int = 16) -> dict:
    """Per-block sufficient statistics of a trials matrix: with these the
    Sharpe of ANY union of blocks is O(N), and a family of thousands of
    templates can be pooled without ever holding all their returns."""
    T, N = R.shape
    S = cscv_partitions(T, n_partitions)
    bounds = np.linspace(0, T, S + 1).astype(int)
    sums = np.stack([R[bounds[b]:bounds[b + 1]].sum(axis=0) for b in range(S)])
    sumsq = np.stack([(R[bounds[b]:bounds[b + 1]] ** 2).sum(axis=0) for b in range(S)])
    counts = np.array([bounds[b + 1] - bounds[b] for b in range(S)], dtype=float)
    return dict(sums=sums, sumsq=sumsq, counts=counts)


def merge_block_stats(parts: list) -> dict:
    """Pool block stats of several templates (same T and n_partitions)."""
    return dict(
        sums=np.concatenate([p["sums"] for p in parts], axis=1),
        sumsq=np.concatenate([p["sumsq"] for p in parts], axis=1),
        counts=parts[0]["counts"],
    )


def cscv_pbo(R: np.ndarray | None = None, n_partitions: int = 16, max_combinations: int = 20000,
             seed: int = 0, blocks: dict | None = None) -> dict:
    """Combinatorially Symmetric Cross-Validation.

    R : T x N per-bar returns of N trials (strategy variants), or pass
    `blocks` from cscv_block_stats / merge_block_stats instead.
    The rows are split into n_partitions contiguous blocks; for every
    way of choosing half the blocks as in-sample (IS) the best IS trial
    is located and its RANK among all trials out-of-sample (OOS) is
    recorded as a logit. PBO = share of logits <= 0, i.e. how often the
    IS winner is a below-median OOS performer.

    Also returns the OOS-vs-IS Sharpe degradation regression and the
    probability that the IS winner loses money OOS.
    """
    if blocks is None:
        blocks = cscv_block_stats(R, n_partitions)
    sums, sumsq, counts = blocks["sums"], blocks["sumsq"], blocks["counts"]
    S, N = sums.shape

    def sharpe_of(blk):
        n = counts[list(blk)].sum()
        s = sums[list(blk)].sum(axis=0)
        q = sumsq[list(blk)].sum(axis=0)
        mean = s / n
        var = np.maximum(q / n - mean ** 2, 0) * n / max(n - 1, 1)
        sd = np.sqrt(var)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(sd > 0, mean / sd * np.sqrt(periods_per_year()), 0.0)

    all_combos = list(combinations(range(S), S // 2))
    if len(all_combos) > max_combinations:
        rng = np.random.default_rng(seed)
        pick = rng.choice(len(all_combos), size=max_combinations, replace=False)
        all_combos = [all_combos[i] for i in pick]

    logits, is_sr, oos_sr = [], [], []
    all_blocks = set(range(S))
    for is_blocks in all_combos:
        oos_blocks = tuple(sorted(all_blocks - set(is_blocks)))
        sr_is = sharpe_of(is_blocks)
        sr_oos = sharpe_of(oos_blocks)
        n_star = int(np.argmax(sr_is))
        rank = (sr_oos < sr_oos[n_star]).sum() + 0.5 * (sr_oos == sr_oos[n_star]).sum() + 0.5
        w = rank / (N + 1)  # relative OOS rank of the IS winner, in (0,1)
        logits.append(np.log(w / (1 - w)))
        is_sr.append(sr_is[n_star])
        oos_sr.append(sr_oos[n_star])

    logits, is_sr, oos_sr = map(np.asarray, (logits, is_sr, oos_sr))
    slope, intercept = np.polyfit(is_sr, oos_sr, 1) if len(is_sr) > 1 and is_sr.std() > 0 else (np.nan, np.nan)
    return dict(
        pbo=float((logits <= 0).mean()),
        logits=logits,
        is_sharpe=is_sr,
        oos_sharpe=oos_sr,
        degradation_slope=float(slope),
        degradation_intercept=float(intercept),
        prob_oos_loss=float((oos_sr < 0).mean()),
        n_trials=N,
        n_combinations=len(all_combos),
    )


# --------------------------------------------------------------------------
# Probabilistic / Deflated Sharpe ratio  (AFML ch. 14, Bailey & LdP 2014)
# --------------------------------------------------------------------------

def _moments(returns) -> tuple[float, float, float, int]:
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    T = len(r)
    sd = r.std(ddof=1) if T > 1 else 0.0
    sr = r.mean() / sd if sd > 0 else 0.0                 # per-period Sharpe
    g3 = float(_skew(r)) if T > 2 else 0.0
    g4 = float(_kurtosis(r, fisher=False)) if T > 3 else 3.0  # non-excess kurtosis
    return float(sr), g3, g4, T


def probabilistic_sharpe_ratio(sr: float, sr_benchmark: float, T: int, skew: float = 0.0, kurt: float = 3.0) -> float:
    """PSR = P[true SR > sr_benchmark], with all Sharpes in PER-PERIOD units."""
    denom = np.sqrt(max(1 - skew * sr + (kurt - 1) / 4 * sr ** 2, 1e-12))
    z = (sr - sr_benchmark) * np.sqrt(max(T - 1, 1)) / denom
    return float(norm.cdf(z))


def expected_max_sharpe(n_trials: int, var_sr: float) -> float:
    """E[max SR] of n_trials i.i.d. trials with Sharpe variance var_sr
    (per-period units), Bailey & Lopez de Prado (2014) eq. (6)."""
    if n_trials <= 1 or not var_sr > 0:     # also catches a NaN variance
        return 0.0
    return float(np.sqrt(var_sr) * (
        (1 - EULER_GAMMA) * norm.ppf(1 - 1 / n_trials) + EULER_GAMMA * norm.ppf(1 - 1 / (n_trials * np.e))
    ))


def deflated_sharpe_ratio(returns, n_trials: int, var_sr_trials: float, ppy: int | None = None) -> dict:
    """Deflated Sharpe Ratio of a strategy that was the best of `n_trials`
    attempts whose per-period Sharpes had variance `var_sr_trials`.
    Returns the DSR probability, the deflating benchmark SR* (annualized
    for readability) and the PSR against zero."""
    sr, g3, g4, T = _moments(returns)
    sr_star = expected_max_sharpe(n_trials, var_sr_trials)
    ppy = periods_per_year() if ppy is None else ppy
    return dict(
        sharpe_annual=sr * np.sqrt(ppy),
        sr_star_annual=sr_star * np.sqrt(ppy),
        psr0=probabilistic_sharpe_ratio(sr, 0.0, T, g3, g4),
        dsr=probabilistic_sharpe_ratio(sr, sr_star, T, g3, g4),
        n_trials=n_trials,
        T=T,
        skew=g3,
        kurtosis=g4,
    )


def effective_n_trials(returns_frame: pd.DataFrame, max_clusters: int = 30) -> dict:
    """Effective number of independent trials (Lopez de Prado 2019, "A Data
    Science Solution to the Multiple-Testing Crisis in Financial Research").

    Hundreds of templates are not hundreds of independent bets: a trend
    template with a 20-bar channel and one with a 40-bar channel are
    nearly the same trial. Trials are clustered on the correlation of
    their return streams (hierarchical, average linkage; the number of
    clusters maximizes the silhouette score), each cluster is collapsed
    to its equal-weight average, and the DSR is then computed with
    N = number of clusters and the variance of the CLUSTER Sharpes.
    Returns n_eff, cluster labels and the per-period variance of cluster Sharpes."""
    from sklearn.metrics import silhouette_score

    X = returns_frame.fillna(0.0)
    N = X.shape[1]
    if N < 3:
        sr = X.apply(lambda c: c.mean() / c.std(ddof=1) if c.std(ddof=1) > 0 else 0.0)
        return dict(n_eff=N, labels=np.arange(N), var_sr_period=float(sr.var()) if N > 1 else 0.0)
    corr = X.corr().fillna(0.0).clip(-1, 1)
    dist = np.sqrt(0.5 * (1 - corr)).to_numpy().copy()
    np.fill_diagonal(dist, 0.0)
    link = linkage(squareform(dist, checks=False), method="average")
    from scipy.cluster.hierarchy import fcluster
    best_k, best_s, best_labels = 1, -1.0, np.zeros(N, dtype=int)
    for k in range(2, min(max_clusters, N - 1) + 1):
        labels = fcluster(link, t=k, criterion="maxclust")
        if len(np.unique(labels)) < 2:
            continue
        s = silhouette_score(dist, labels, metric="precomputed")
        if s > best_s:
            best_k, best_s, best_labels = len(np.unique(labels)), s, labels
    cluster_rets = pd.DataFrame({c: X.loc[:, best_labels == c].mean(axis=1) for c in np.unique(best_labels)})
    sr = cluster_rets.apply(lambda c: c.mean() / c.std(ddof=1) if c.std(ddof=1) > 0 else 0.0)
    return dict(n_eff=int(best_k), labels=best_labels, var_sr_period=float(sr.var()) if best_k > 1 else 0.0,
                silhouette=float(best_s))


def min_backtest_length(n_trials: int, target_sharpe_annual: float, ppy: int | None = None) -> float:
    """Minimum Backtest Length (Bailey et al. 2014, AFML 11.5): number of
    YEARS of track record needed so that a Sharpe of `target_sharpe_annual`
    could not be expected from the best of `n_trials` pure-noise trials.
    Uses the closed-form upper bound MinBTL ~ 2 ln N / SR^2 (per period)."""
    if target_sharpe_annual <= 0 or n_trials < 2:
        return np.inf
    ppy = periods_per_year() if ppy is None else ppy
    sr = target_sharpe_annual / np.sqrt(ppy)
    periods = 2 * np.log(n_trials) / sr ** 2
    return float(periods / ppy)


# --------------------------------------------------------------------------
# stationary bootstrap Monte Carlo  (Politis & Romano 1994)
# --------------------------------------------------------------------------

def stationary_bootstrap_indices(T: int, n_boot: int, mean_block: float, rng) -> np.ndarray:
    p = 1.0 / max(mean_block, 1.0)
    out = np.empty((n_boot, T), dtype=np.int64)
    for b in range(n_boot):
        lens = rng.geometric(p, size=int(T * p * 2) + 10)
        while lens.sum() < T:
            lens = np.concatenate([lens, rng.geometric(p, size=int(T * p) + 10)])
        starts = rng.integers(0, T, size=len(lens))
        total = int(lens.sum())
        offsets = np.arange(total) - np.repeat(np.cumsum(lens) - lens, lens)
        out[b] = ((np.repeat(starts, lens) + offsets) % T)[:T]
    return out


def bootstrap_sharpe_pvalue(returns, n_boot: int = 2000, mean_block: float = 20.0, seed: int = 0) -> dict:
    """Monte Carlo p-value of H0 'this strategy has zero expected return'.
    The observed returns are demeaned, resampled with a stationary block
    bootstrap (keeps autocorrelation and volatility clustering), and the
    Sharpe of each resample compared with the observed one.
    financial-hacker's rule of thumb: p < 5 % good, p > 15 % don't trust it."""
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    T = len(r)
    if T < 20 or r.std() == 0:
        return dict(sharpe=0.0, p_value=1.0, n_boot=0)
    rng = np.random.default_rng(seed)
    obs = r.mean() / r.std(ddof=1)
    r0 = r - r.mean()
    idx = stationary_bootstrap_indices(T, n_boot, mean_block, rng)
    samples = r0[idx]
    sd = samples.std(axis=1, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        boot = np.where(sd > 0, samples.mean(axis=1) / sd, 0.0)
    p = float((boot >= obs).mean())
    return dict(sharpe=float(obs * np.sqrt(periods_per_year())), p_value=p, n_boot=n_boot)


def reality_check(returns_frame: pd.DataFrame, n_boot: int = 2000, mean_block: float = 20.0, seed: int = 0) -> dict:
    """White's (2000) Reality Check for data snooping across a FAMILY.
    H0: no strategy in the family beats a zero-return benchmark. The
    statistic is the best mean return in the family; its null
    distribution is obtained by bootstrapping the demeaned family jointly
    (same resampled dates for every strategy, so cross-correlation is kept)."""
    X = returns_frame.fillna(0.0).to_numpy(dtype=float)
    T, K = X.shape
    if T < 20 or K == 0:
        return dict(best=None, statistic=0.0, p_value=1.0)
    rng = np.random.default_rng(seed)
    fbar = X.mean(axis=0)
    V = np.sqrt(T) * fbar.max()
    idx = stationary_bootstrap_indices(T, n_boot, mean_block, rng)
    Vb = np.empty(n_boot)
    for b in range(n_boot):
        Vb[b] = np.sqrt(T) * (X[idx[b]].mean(axis=0) - fbar).max()
    return dict(
        best=returns_frame.columns[int(fbar.argmax())],
        statistic=float(V),
        p_value=float((Vb >= V).mean()),
        n_boot=n_boot,
    )


# --------------------------------------------------------------------------
# Hierarchical Risk Parity  (AFML ch. 16)
# --------------------------------------------------------------------------

def hrp_weights(returns_frame: pd.DataFrame) -> pd.Series:
    cols = list(returns_frame.columns)
    if len(cols) == 1:
        return pd.Series([1.0], index=cols)

    # A column with zero (or non-finite) variance is a strategy that never
    # traded. Inverse-variance weighting would divide by zero for it, and the
    # resulting NaN cluster variance silently degrades the bisection to a 50/50
    # split -- handing a dead strategy a large share of the book. Give those
    # columns no weight and run HRP on the rest.
    var = returns_frame.var()
    live = [c for c in cols if np.isfinite(var[c]) and var[c] > 0]
    if len(live) < len(cols):
        if not live:
            return pd.Series(1.0 / len(cols), index=cols)
        w = hrp_weights(returns_frame[live]) if len(live) > 1 else pd.Series([1.0], index=live)
        return w.reindex(cols).fillna(0.0)

    cov = returns_frame.cov()
    corr = returns_frame.corr().fillna(0.0)
    dist = np.sqrt(0.5 * (1 - corr.clip(-1, 1))).to_numpy().copy()
    np.fill_diagonal(dist, 0.0)
    link = linkage(squareform(dist, checks=False), method="single")
    order = [cols[i] for i in leaves_list(link)]

    def inv_var_w(items):
        ivp = 1.0 / np.diag(cov.loc[items, items].values)
        ivp /= ivp.sum()
        return ivp

    def cluster_var(items):
        w = inv_var_w(items)
        return float(w @ cov.loc[items, items].values @ w)

    w = pd.Series(1.0, index=order)
    clusters = [order]
    while clusters:
        clusters = [c[j:k] for c in clusters if len(c) > 1
                    for j, k in ((0, len(c) // 2), (len(c) // 2, len(c)))]
        for i in range(0, len(clusters), 2):
            left, right = clusters[i], clusters[i + 1]
            vl, vr = cluster_var(left), cluster_var(right)
            alpha = 1 - vl / (vl + vr) if (vl + vr) > 0 else 0.5
            w[left] *= alpha
            w[right] *= 1 - alpha
    return w.reindex(cols)
