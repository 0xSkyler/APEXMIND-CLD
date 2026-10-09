"""Statistics for evaluation under serial dependence and multiple testing.

Trades that are close in time share market conditions, so all uncertainty
estimates here resample or aggregate over *time blocks* rather than treating
trades as independent.

References:
* Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio".
* Bailey, Borwein, Lopez de Prado & Zhu (2017), "The Probability of Backtest
  Overfitting" (CSCV).
* Hansen (2005), "A Test for Superior Predictive Ability"; Politis & Romano
  (1994) stationary bootstrap.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np
from scipy import stats as sps

EULER_GAMMA = 0.5772156649015329


def time_clusters(ts_ns: np.ndarray, block_ns: int) -> np.ndarray:
    return (np.asarray(ts_ns, dtype=np.int64) // max(int(block_ns), 1)).astype(np.int64)


def cluster_mean_se(x: np.ndarray, clusters: np.ndarray) -> tuple[float, float, int]:
    """Mean and cluster-robust standard error of the mean."""
    x = np.asarray(x, float)
    n = len(x)
    if n == 0:
        return math.nan, math.nan, 0
    m = float(x.mean())
    _, inv = np.unique(clusters, return_inverse=True)
    g = int(inv.max()) + 1
    sums = np.bincount(inv, weights=x - m, minlength=g)
    if g < 2:
        return m, math.inf, g
    se = math.sqrt(float((sums**2).sum()) * g / (g - 1)) / n
    return m, se, g


def cluster_bootstrap(values: np.ndarray, clusters: np.ndarray, stat, reps: int = 2000, conf: float = 0.95,
                      seed: int = 0) -> tuple[float, float, float]:
    """Point estimate and percentile CI, resampling whole time blocks."""
    values = np.asarray(values, float)
    if len(values) == 0:
        return math.nan, math.nan, math.nan
    point = float(stat(values))
    _, inv = np.unique(clusters, return_inverse=True)
    groups = [values[inv == k] for k in range(inv.max() + 1)]
    if len(groups) < 2:
        return point, math.nan, math.nan
    rng = np.random.default_rng(seed)
    dist = np.empty(reps)
    for i in range(reps):
        pick = rng.integers(0, len(groups), len(groups))
        dist[i] = stat(np.concatenate([groups[j] for j in pick]))
    a = (1 - conf) / 2
    return point, float(np.nanquantile(dist, a)), float(np.nanquantile(dist, 1 - a))


def profit_factor(pnl: np.ndarray) -> float:
    pnl = np.asarray(pnl, float)
    gains, losses = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
    if losses == 0:
        return math.inf if gains > 0 else math.nan
    return float(gains / losses)


def max_drawdown(equity: np.ndarray) -> tuple[float, float]:
    """(max fractional drawdown, max absolute drawdown) of an equity path."""
    eq = np.asarray(equity, float)
    if len(eq) == 0:
        return 0.0, 0.0
    peak = np.maximum.accumulate(eq)
    dd_abs = peak - eq
    with np.errstate(divide="ignore", invalid="ignore"):
        dd_frac = np.where(peak > 0, dd_abs / peak, 0.0)
    return float(dd_frac.max()), float(dd_abs.max())


def expected_shortfall(x: np.ndarray, alpha: float = 0.05) -> float:
    """Mean of the worst ``alpha`` fraction of outcomes (a negative number
    for losses)."""
    x = np.sort(np.asarray(x, float))
    if len(x) == 0:
        return math.nan
    k = max(1, int(math.ceil(alpha * len(x))))
    return float(x[:k].mean())


def sharpe(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    if len(x) < 2 or x.std(ddof=1) == 0:
        return math.nan
    return float(x.mean() / x.std(ddof=1))


def deflated_sharpe(x: np.ndarray, n_trials: int, sr_trials_var: float) -> dict:
    """Probability that the true Sharpe ratio (per period, of ``x``) exceeds
    the maximum expected from ``n_trials`` unskilled strategies."""
    x = np.asarray(x, float)
    n = len(x)
    if n < 10:
        return {"dsr": math.nan, "sr": math.nan, "sr0": math.nan}
    sr = sharpe(x)
    skew = float(sps.skew(x))
    kurt = float(sps.kurtosis(x, fisher=False))
    n_trials = max(int(n_trials), 1)
    if n_trials > 1 and sr_trials_var > 0:
        z1 = sps.norm.ppf(1 - 1.0 / n_trials)
        z2 = sps.norm.ppf(1 - 1.0 / (n_trials * math.e))
        sr0 = math.sqrt(sr_trials_var) * ((1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2)
    else:
        sr0 = 0.0
    denom = math.sqrt(max(1e-12, 1 - skew * sr + (kurt - 1) / 4.0 * sr**2))
    dsr = float(sps.norm.cdf((sr - sr0) * math.sqrt(n - 1) / denom))
    return {"dsr": dsr, "sr": sr, "sr0": sr0, "skew": skew, "kurtosis": kurt, "n": n, "trials": n_trials}


def pbo_cscv(perf: np.ndarray, n_splits: int = 8) -> dict:
    """Probability of Backtest Overfitting via combinatorially symmetric
    cross-validation. ``perf`` is (periods x strategies), higher is better."""
    perf = np.asarray(perf, float)
    T, N = perf.shape
    if N < 2 or T < n_splits:
        return {"pbo": math.nan, "n_combinations": 0}
    groups = np.array_split(np.arange(T), n_splits)
    logits = []
    for is_idx in itertools.combinations(range(n_splits), n_splits // 2):
        is_rows = np.concatenate([groups[i] for i in is_idx])
        oos_rows = np.concatenate([groups[i] for i in range(n_splits) if i not in is_idx])
        is_perf = np.nanmean(perf[is_rows], axis=0)
        oos_perf = np.nanmean(perf[oos_rows], axis=0)
        best = int(np.nanargmax(is_perf))
        rank = sps.rankdata(oos_perf)[best]  # 1..N, N = best OOS
        w = rank / (N + 1)
        logits.append(math.log(w / (1 - w)))
    logits = np.asarray(logits)
    return {"pbo": float((logits <= 0).mean()), "n_combinations": int(len(logits)),
            "median_logit": float(np.median(logits))}


def stationary_bootstrap_indices(T: int, mean_block: float, rng: np.random.Generator) -> np.ndarray:
    idx = np.empty(T, dtype=np.int64)
    p = 1.0 / max(mean_block, 1.0)
    idx[0] = rng.integers(T)
    for t in range(1, T):
        idx[t] = rng.integers(T) if rng.random() < p else (idx[t - 1] + 1) % T
    return idx


@dataclass
class SPAResult:
    p_value: float
    statistic: float
    best: int
    n_models: int


def spa_test(diff: np.ndarray, reps: int = 2000, mean_block: float | None = None, seed: int = 0) -> SPAResult:
    """Hansen's SPA (consistent version). ``diff`` is (periods x models) of
    model performance minus benchmark performance; H0: no model beats the
    benchmark in expectation."""
    d = np.asarray(diff, float)
    T, K = d.shape
    if T < 5:
        return SPAResult(math.nan, math.nan, -1, K)
    mean_block = mean_block or max(1.0, T ** (1 / 3))
    rng = np.random.default_rng(seed)
    dbar = d.mean(axis=0)
    boots = np.empty((reps, K))
    for b in range(reps):
        boots[b] = d[stationary_bootstrap_indices(T, mean_block, rng)].mean(axis=0)
    omega = np.sqrt(T) * boots.std(axis=0)
    omega[omega == 0] = np.inf
    tstat = np.sqrt(T) * dbar / omega
    stat = max(float(tstat.max()), 0.0)
    # Hansen's g_c: keep a model's mean in the null recentring unless it is
    # very poor; bootstrap means are then recentred as d* - g(dbar).
    thresh = -np.sqrt(2 * np.log(np.log(max(T, 3))))
    g = np.where(tstat >= thresh, dbar, 0.0)
    centered = np.sqrt(T) * (boots - g) / omega
    stat_b = np.maximum(centered.max(axis=1), 0.0)
    return SPAResult(float((stat_b >= stat).mean()), stat, int(np.argmax(tstat)), K)


def paired_difference(a: np.ndarray, b: np.ndarray, reps: int = 2000, conf: float = 0.95, seed: int = 0,
                      mean_block: float | None = None) -> dict:
    """Mean of per-period differences (a - b) with stationary-bootstrap CI
    and one-sided p-value for H0: mean <= 0."""
    d = np.asarray(a, float) - np.asarray(b, float)
    T = len(d)
    if T < 5:
        return {"mean": math.nan, "lo": math.nan, "hi": math.nan, "p_value": math.nan, "periods": T}
    rng = np.random.default_rng(seed)
    mb = mean_block or max(1.0, T ** (1 / 3))
    boots = np.array([d[stationary_bootstrap_indices(T, mb, rng)].mean() for _ in range(reps)])
    alpha = (1 - conf) / 2
    centered = boots - d.mean()
    return {"mean": float(d.mean()), "lo": float(np.quantile(boots, alpha)), "hi": float(np.quantile(boots, 1 - alpha)),
            "p_value": float((centered >= d.mean()).mean()), "periods": T}
