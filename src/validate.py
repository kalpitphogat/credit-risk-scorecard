"""Validation metrics: discrimination, calibration, stability."""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve


def gini(y, p) -> float:
    return 2 * roc_auc_score(y, p) - 1


def ks(y, p) -> float:
    """Max distance between cumulative bad and good distributions."""
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def psi(expected: pd.Series, actual: pd.Series, edges) -> float:
    """Population Stability Index between two score (or bin) distributions.
    < 0.10 stable, 0.10-0.25 monitor, > 0.25 significant shift."""
    e = pd.cut(expected, edges, include_lowest=True).value_counts(normalize=True, sort=False)
    a = pd.cut(actual, edges, include_lowest=True).value_counts(normalize=True, sort=False)
    e, a = e.clip(lower=1e-6), a.clip(lower=1e-6)
    return float(((a - e) * np.log(a / e)).sum())


def psi_categorical(expected: pd.Series, actual: pd.Series) -> float:
    e = expected.value_counts(normalize=True)
    a = actual.value_counts(normalize=True)
    idx = e.index.union(a.index)
    e, a = e.reindex(idx).fillna(1e-6).clip(lower=1e-6), a.reindex(idx).fillna(1e-6).clip(lower=1e-6)
    return float(((a - e) * np.log(a / e)).sum())


def calibration_table(y: pd.Series, pd_hat: pd.Series, n_groups=10) -> pd.DataFrame:
    """Observed vs predicted bad rate by predicted-PD decile (1 = safest)."""
    g = pd.qcut(pd_hat.rank(method="first"), n_groups, labels=range(1, n_groups + 1))
    t = pd.DataFrame({"g": g, "y": y, "p": pd_hat}).groupby("g", observed=True).agg(
        n=("y", "size"), bads=("y", "sum"), observed=("y", "mean"), predicted=("p", "mean"))
    # Two-sided binomial test: is the observed count consistent with the predicted PD?
    t["binom_p"] = [stats.binomtest(int(b), int(n), p).pvalue
                    for b, n, p in zip(t["bads"], t["n"], t["predicted"])]
    return t.reset_index()


def hosmer_lemeshow(y, pd_hat, n_groups=10) -> tuple[float, float]:
    t = calibration_table(pd.Series(y), pd.Series(pd_hat), n_groups)
    exp_bad = t["predicted"] * t["n"]
    exp_good = t["n"] - exp_bad
    obs_bad = t["bads"]
    obs_good = t["n"] - obs_bad
    chi2 = (((obs_bad - exp_bad) ** 2) / exp_bad + ((obs_good - exp_good) ** 2) / exp_good).sum()
    return float(chi2), float(stats.chi2.sf(chi2, n_groups - 2))


def bootstrap_gini_diff(y, p_a, p_b, n_boot=1000, seed=42) -> dict:
    """Bootstrap 95% intervals for Gini(a), Gini(b) and Gini(a) - Gini(b),
    resampling loans with replacement (same resample for both models)."""
    rng = np.random.default_rng(seed)
    y, p_a, p_b = map(np.asarray, (y, p_a, p_b))
    ga, gb, d = [], [], []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        a, b = gini(y[i], p_a[i]), gini(y[i], p_b[i])
        ga.append(a); gb.append(b); d.append(a - b)
    ci = lambda x: [round(float(np.percentile(x, 2.5)), 4), round(float(np.percentile(x, 97.5)), 4)]
    return {"gini_a_ci": ci(ga), "gini_b_ci": ci(gb), "diff": round(float(np.mean(d)), 4),
            "diff_ci": ci(d), "share_diff_positive": round(float(np.mean(np.array(d) > 0)), 3)}


def bad_rate_by_band(y: pd.Series, score: pd.Series, n_bands=10) -> pd.DataFrame:
    """Rank-ordering check: bad rate should fall as score rises."""
    band = pd.qcut(score.rank(method="first"), n_bands, labels=range(1, n_bands + 1))
    t = pd.DataFrame({"band": band, "y": y, "s": score}).groupby("band", observed=True).agg(
        n=("y", "size"), min_score=("s", "min"), max_score=("s", "max"), bad_rate=("y", "mean"))
    return t.reset_index()
