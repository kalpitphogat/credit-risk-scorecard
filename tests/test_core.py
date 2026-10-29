import numpy as np
import pandas as pd

from src import scorecard as sc
from src import validate as v
from src.woe import fit_binning

rng = np.random.default_rng(0)


def _risky_numeric(n=20_000):
    x = pd.Series(rng.uniform(0, 100, n))
    p = 0.05 + 0.25 * x / 100                      # risk rises with x
    y = pd.Series((rng.uniform(size=n) < p).astype(int))
    return x, y


def test_numeric_bins_are_monotonic_and_large_enough():
    x, y = _risky_numeric()
    b = fit_binning(x, y, "x", "numeric", min_bin_share=0.05)
    t = b.table[b.table["bin"] != "MISSING"].sort_values("bin")
    assert t["bad_rate"].is_monotonic_increasing
    assert (t["share"] >= 0.05).all()
    assert b.iv > 0.1


def test_zero_heavy_count_variable_keeps_its_signal():
    """Regression test: a count variable that is mostly 0 must not collapse into one bin."""
    n = 20_000
    x = pd.Series(rng.choice([0, 1, 2, 3, 4], n, p=[0.5, 0.25, 0.13, 0.07, 0.05]))
    y = pd.Series((rng.uniform(size=n) < 0.06 + 0.04 * x).astype(int))
    b = fit_binning(x, y, "inq", "numeric")
    assert len(b.table) >= 3
    assert b.iv > 0.05


def test_missing_values_get_their_own_bin():
    x, y = _risky_numeric()
    x.iloc[:500] = np.nan
    b = fit_binning(x, y, "x", "numeric")
    assert "MISSING" in set(b.table["bin"])


def test_psi_is_zero_for_identical_distributions():
    s = pd.Series(rng.normal(size=5000))
    edges = [-np.inf, -1, 0, 1, np.inf]
    assert v.psi(s, s, edges) < 1e-9


def test_score_scaling_anchor():
    # 600 points <=> 50:1 good:bad odds <=> PD = 1/51
    assert abs(sc.score_to_pd(pd.Series([600.0])).iloc[0] - 1 / 51) < 1e-9
    # +20 points (PDO) doubles the odds
    p1, p2 = sc.score_to_pd(pd.Series([600.0, 620.0]))
    assert abs(((1 - p2) / p2) / ((1 - p1) / p1) - 2) < 1e-9


def test_gini_and_ks_on_perfect_and_random_scores():
    y = np.array([0] * 900 + [1] * 100)
    assert v.gini(y, y) == 1.0 and v.ks(y, y) == 1.0
    assert abs(v.gini(y, rng.uniform(size=1000))) < 0.15
