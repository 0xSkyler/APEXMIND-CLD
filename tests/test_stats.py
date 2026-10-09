import numpy as np
import pytest

from apexmind.research.stats import (
    cluster_bootstrap,
    cluster_mean_se,
    deflated_sharpe,
    expected_shortfall,
    max_drawdown,
    paired_difference,
    pbo_cscv,
    profit_factor,
    spa_test,
)


def test_cluster_se_exceeds_naive_under_dependence():
    rng = np.random.default_rng(0)
    shocks = np.repeat(rng.normal(size=50), 20)  # 50 clusters of 20 identical shocks
    x = shocks + rng.normal(scale=0.1, size=1000)
    m, se, g = cluster_mean_se(x, np.repeat(np.arange(50), 20))
    naive = x.std(ddof=1) / np.sqrt(len(x))
    assert g == 50 and se > 3 * naive


def test_cluster_bootstrap_covers_truth():
    rng = np.random.default_rng(1)
    x = rng.normal(0.5, 1.0, 2000)
    point, lo, hi = cluster_bootstrap(x, np.arange(2000) // 20, np.mean, reps=500)
    assert lo < 0.5 < hi and lo < point < hi


def test_basic_metrics():
    assert profit_factor(np.array([2.0, -1.0, 1.0])) == pytest.approx(3.0)
    dd, dd_abs = max_drawdown(np.array([100, 120, 90, 130, 65]))
    assert dd == pytest.approx(0.5) and dd_abs == pytest.approx(65)
    assert expected_shortfall(np.arange(100.0), 0.05) == pytest.approx(2.0)


def test_spa_rejects_real_edge_and_not_noise():
    rng = np.random.default_rng(2)
    T = 200
    edge = np.column_stack([rng.normal(0.3, 1, T), rng.normal(0, 1, T)])
    noise = rng.normal(0, 1, (T, 5))
    assert spa_test(edge, reps=500).p_value < 0.05
    p_noise = [spa_test(rng.normal(0, 1, (T, 5)), reps=300, seed=s).p_value for s in range(10)]
    assert np.mean(np.array(p_noise) < 0.05) <= 0.3
    assert spa_test(noise - 0.2, reps=300).p_value > 0.5


def test_paired_difference_direction():
    rng = np.random.default_rng(3)
    a = rng.normal(0.5, 1, 300)
    b = rng.normal(0.0, 1, 300)
    r = paired_difference(a, b, reps=500)
    assert r["p_value"] < 0.01 and r["lo"] > 0
    r2 = paired_difference(b, a, reps=500)
    assert r2["p_value"] > 0.9


def test_deflated_sharpe_penalizes_trials():
    rng = np.random.default_rng(4)
    x = rng.normal(0.1, 1, 500)
    one = deflated_sharpe(x, 1, 0.0)["dsr"]
    many = deflated_sharpe(x, 1000, 0.01)["dsr"]
    assert many < one


def test_pbo_low_for_persistent_skill_high_for_noise():
    rng = np.random.default_rng(5)
    skill = rng.normal(0, 1, (64, 10))
    skill[:, 3] += 1.0
    assert pbo_cscv(skill, 8)["pbo"] < 0.1
    # on pure noise the best in-sample strategy is a coin flip out of sample
    vals = [pbo_cscv(np.random.default_rng(s).normal(0, 1, (64, 10)), 8)["pbo"] for s in range(12)]
    assert 0.3 < np.mean(vals) < 0.7
