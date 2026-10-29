"""Characteristic selection, logistic regression on WoE, and points scaling."""
import numpy as np
import pandas as pd
import statsmodels.api as sm

from .woe import Binning

# Score scaling: 600 points = 50:1 good:bad odds, and every 20 points doubles the odds.
BASE_SCORE, BASE_ODDS, PDO = 600, 50, 20
FACTOR = PDO / np.log(2)
OFFSET = BASE_SCORE - FACTOR * np.log(BASE_ODDS)


def select_by_iv(binnings: dict, min_iv=0.02, max_iv=0.5) -> list:
    """Keep characteristics with useful but not suspicious predictive power.
    IV < 0.02 is not predictive; IV > 0.5 usually signals leakage."""
    return [n for n, b in sorted(binnings.items(), key=lambda kv: -kv[1].iv)
            if min_iv <= b.iv <= max_iv]


def drop_correlated(woe: pd.DataFrame, ranked: list, max_corr=0.6) -> list:
    """Walk characteristics in IV order; drop any too correlated with one already kept."""
    kept = []
    corr = woe[ranked].corr().abs()
    for c in ranked:
        if all(corr.loc[c, k] < max_corr for k in kept):
            kept.append(c)
    return kept


def fit_logit(woe: pd.DataFrame, y: pd.Series, features: list, p_max=0.05):
    """Backward elimination. With WoE inputs (higher = safer) every coefficient
    on P(bad) must be negative; a positive one means the characteristic is acting
    against its own univariate evidence (usually collinearity), so it is removed.
    Then drop the least significant until all p-values < p_max."""
    feats = list(features)
    log = []
    while True:
        model = sm.Logit(y, sm.add_constant(woe[feats])).fit(disp=0)
        params, pvals = model.params.drop("const"), model.pvalues.drop("const")
        wrong_sign = params[params > 0]
        if len(wrong_sign):
            f = wrong_sign.index[0]
            log.append((f, "wrong sign"))
        elif pvals.max() > p_max:
            f = pvals.idxmax()
            log.append((f, f"p = {pvals.max():.3f}"))
        else:
            return model, feats, log
        feats.remove(f)


def vif(woe: pd.DataFrame, features: list) -> pd.Series:
    X = woe[features].to_numpy()
    out = {}
    for i, f in enumerate(features):
        others = np.delete(X, i, axis=1)
        others = np.column_stack([np.ones(len(X)), others])
        beta, *_ = np.linalg.lstsq(others, X[:, i], rcond=None)
        resid = X[:, i] - others @ beta
        r2 = 1 - resid.var() / X[:, i].var()
        out[f] = 1 / (1 - r2)
    return pd.Series(out)


def points_table(model, binnings: dict, features: list) -> pd.DataFrame:
    """Points per bin. Score = OFFSET + FACTOR * ln(good/bad odds).
    The intercept is spread evenly across characteristics so every bin
    has an interpretable whole-number contribution."""
    n = len(features)
    a = model.params["const"]
    rows = []
    for f in features:
        beta = model.params[f]
        for _, r in binnings[f].table.iterrows():
            pts = -(beta * r["woe"] + a / n) * FACTOR + OFFSET / n
            rows.append({"characteristic": f, "bin": r["bin"], "woe": r["woe"],
                         "points": int(round(pts))})
    return pd.DataFrame(rows)


def neutral_points(model, n_features: int) -> float:
    """Points for a bin with WoE = 0 (no evidence either way)."""
    return -(model.params["const"] / n_features) * FACTOR + OFFSET / n_features


def score(df: pd.DataFrame, binnings: dict, points: pd.DataFrame, neutral: float) -> pd.Series:
    """Unseen categories at scoring time get neutral points, not zero."""
    total = pd.Series(0, index=df.index, dtype=float)
    for f, grp in points.groupby("characteristic"):
        pts = grp.set_index("bin")["points"]
        total += binnings[f].assign(df[f]).map(pts).fillna(round(neutral))
    return total


def score_to_pd(s: pd.Series) -> pd.Series:
    log_odds_good = (s - OFFSET) / FACTOR
    return 1 / (1 + np.exp(log_odds_good))
