"""Build, score and validate the PD scorecard end to end.

    python scripts/download_data.py
    python run_all.py

Writes: outputs/metrics.json, outputs/scorecard.csv, outputs/iv_table.csv,
outputs/scorecard_validation.xlsx and reports/figures/*.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

from src import data, scorecard as sc, validate as v
from src.woe import describe_bins, fit_binning

ROOT = Path(__file__).resolve().parent
OUT, FIG = ROOT / "outputs", ROOT / "reports" / "figures"
SEED = 42


def main():
    OUT.mkdir(exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)

    # 1. Sample ---------------------------------------------------------------
    df, waterfall = data.build_sample()
    dev = df[df.issue_year <= 2010]
    oot = df[df.issue_year == 2011]
    train, test = train_test_split(dev, test_size=0.3, stratify=dev.bad, random_state=SEED)
    sets = {"train": train, "test": test, "oot": oot}
    print({k: (len(s), round(s.bad.mean(), 4)) for k, s in sets.items()})

    # 2. WoE binning on TRAIN only ---------------------------------------------
    binnings = {c: fit_binning(train[c], train.bad, c,
                               "numeric" if c in data.NUMERIC else "categorical")
                for c in data.CANDIDATES}
    iv_table = (pd.DataFrame({"characteristic": list(binnings),
                              "iv": [b.iv for b in binnings.values()]})
                .sort_values("iv", ascending=False))

    def woe_frame(d):
        return pd.DataFrame({c: binnings[c].transform(d[c]) for c in binnings}, index=d.index)
    woe = {k: woe_frame(s) for k, s in sets.items()}

    # 3. Selection + model ----------------------------------------------------
    ranked = sc.select_by_iv(binnings)
    uncorrelated = sc.drop_correlated(woe["train"], ranked)
    model, features, elim_log = sc.fit_logit(woe["train"], train.bad, uncorrelated)
    vifs = sc.vif(woe["train"], features)
    points = sc.points_table(model, binnings, features)

    # 4. Score every set ------------------------------------------------------
    neutral = sc.neutral_points(model, len(features))
    scores = {k: sc.score(s, binnings, points, neutral) for k, s in sets.items()}
    pds = {k: sc.score_to_pd(scores[k]) for k in sets}

    # 5. Benchmarks -----------------------------------------------------------
    # (a) Lending Club's own sub-grade (A1 safest ... G5 riskiest)
    subgrade_rank = {f"{g}{i}": n for n, (g, i) in enumerate(
        (g, i) for g in "ABCDEFG" for i in range(1, 6))}
    lc_grade = {k: s.sub_grade.map(subgrade_rank) for k, s in sets.items()}
    # (b) Challenger: gradient boosting on the same raw characteristics
    def raw_X(d):
        X = d[data.CANDIDATES].copy()
        for c in data.CATEGORICAL:
            X[c] = X[c].astype("category").cat.set_categories(sorted(train[c].dropna().unique()))
        return X
    gbm = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                         categorical_features="from_dtype", random_state=SEED)
    gbm.fit(raw_X(train), train.bad)
    gbm_p = {k: pd.Series(gbm.predict_proba(raw_X(s))[:, 1], index=s.index) for k, s in sets.items()}

    # 6. Metrics --------------------------------------------------------------
    metrics = {"sample": {}, "discrimination": {}, "calibration": {}, "stability": {},
               "benchmark": {}, "model": {}}
    for k, s in sets.items():
        y = s.bad
        hl_chi2, hl_p = v.hosmer_lemeshow(y, pds[k])
        metrics["sample"][k] = {"n": int(len(s)), "bads": int(y.sum()), "bad_rate": round(float(y.mean()), 4)}
        metrics["discrimination"][k] = {
            "auc": round(roc_auc_score(y, pds[k]), 4),
            "gini": round(v.gini(y, pds[k]), 4),
            "ks": round(v.ks(y, pds[k]), 4)}
        metrics["calibration"][k] = {
            "mean_predicted_pd": round(float(pds[k].mean()), 4),
            "observed_bad_rate": round(float(y.mean()), 4),
            "brier": round(brier_score_loss(y, pds[k]), 4),
            "hosmer_lemeshow_chi2": round(hl_chi2, 2), "hosmer_lemeshow_p": round(hl_p, 4)}
        metrics["benchmark"][k] = {
            "scorecard_gini": round(v.gini(y, pds[k]), 4),
            "lc_subgrade_gini": round(v.gini(y, lc_grade[k]), 4),
            "gbm_challenger_gini": round(v.gini(y, gbm_p[k]), 4)}

    edges = np.unique(np.quantile(scores["train"], np.linspace(0, 1, 11)))
    edges[0], edges[-1] = -np.inf, np.inf
    metrics["stability"]["score_psi_train_vs_test"] = round(v.psi(scores["train"], scores["test"], edges), 4)
    metrics["stability"]["score_psi_train_vs_oot"] = round(v.psi(scores["train"], scores["oot"], edges), 4)
    csi = {f: round(v.psi_categorical(binnings[f].assign(train[f]), binnings[f].assign(oot[f])), 4)
           for f in features}
    metrics["stability"]["characteristic_psi_train_vs_oot"] = csi
    metrics["model"] = {
        "features": features,
        "coefficients": {k: round(float(c), 4) for k, c in model.params.items()},
        "p_values": {k: float(f"{p:.3g}") for k, p in model.pvalues.items()},
        "vif": {k: round(float(x), 2) for k, x in vifs.items()},
        "eliminated": elim_log,
        "candidates": len(data.CANDIDATES),
        "passed_iv": ranked,
        "after_correlation": uncorrelated,
        "score_scaling": {"base_score": sc.BASE_SCORE, "base_odds": sc.BASE_ODDS, "pdo": sc.PDO},
        "score_range": [int(points.groupby("characteristic").points.min().sum()),
                        int(points.groupby("characteristic").points.max().sum())]}

    metrics["benchmark"]["oot_bootstrap_vs_lc_subgrade"] = v.bootstrap_gini_diff(
        oot.bad, pds["oot"], lc_grade["oot"])
    metrics["benchmark"]["oot_bootstrap_gbm_vs_scorecard"] = v.bootstrap_gini_diff(
        oot.bad, gbm_p["oot"], pds["oot"])
    # Robustness: refit on 5 other random development splits (same characteristics)
    import statsmodels.api as sm
    robust = []
    for seed in (1, 2, 3, 4, 5):
        tr2, te2 = train_test_split(dev, test_size=0.3, stratify=dev.bad, random_state=seed)
        bn2 = {c: fit_binning(tr2[c], tr2.bad, c, "numeric" if c in data.NUMERIC else "categorical")
               for c in features}
        W2 = lambda d: sm.add_constant(pd.DataFrame({c: bn2[c].transform(d[c]) for c in features}, index=d.index))
        m2 = sm.Logit(tr2.bad, W2(tr2)).fit(disp=0)
        p_te, p_oot = m2.predict(W2(te2)), m2.predict(W2(oot))
        robust.append({"seed": seed, "test_gini": round(v.gini(te2.bad, p_te), 4),
                       "oot_gini": round(v.gini(oot.bad, p_oot), 4),
                       "test_hl_p": round(v.hosmer_lemeshow(te2.bad, p_te)[1], 4),
                       "oot_hl_p": round(v.hosmer_lemeshow(oot.bad, p_oot)[1], 4)})
    metrics["robustness_other_splits"] = robust

    calib_oot = v.calibration_table(oot.bad, pds["oot"])
    bands_oot = v.bad_rate_by_band(oot.bad, scores["oot"])
    metrics["discrimination"]["oot_rank_ordering_monotonic"] = bool(
        bands_oot.bad_rate.is_monotonic_decreasing)
    metrics["sample"]["waterfall"] = waterfall.to_dict("records")

    # 7. Save -----------------------------------------------------------------
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    points_out = points.merge(
        pd.concat([describe_bins(binnings[f])[["characteristic", "bin", "range", "n", "bad_rate"]]
                   for f in features]), on=["characteristic", "bin"])
    points_out.to_csv(OUT / "scorecard.csv", index=False)
    iv_table.to_csv(OUT / "iv_table.csv", index=False)
    with pd.ExcelWriter(OUT / "scorecard_validation.xlsx") as xl:
        pd.DataFrame([{"set": k, **metrics["sample"][k], **metrics["discrimination"][k],
                       **metrics["calibration"][k]} for k in sets]).to_excel(xl, sheet_name="Summary", index=False)
        points_out.to_excel(xl, sheet_name="Scorecard", index=False)
        iv_table.to_excel(xl, sheet_name="IV", index=False)
        pd.concat([describe_bins(b) for b in binnings.values()]).to_excel(xl, sheet_name="WoE bins", index=False)
        calib_oot.to_excel(xl, sheet_name="OOT calibration", index=False)
        bands_oot.to_excel(xl, sheet_name="OOT rank ordering", index=False)
        pd.DataFrame([{"characteristic": f, "psi": p} for f, p in csi.items()]).to_excel(xl, sheet_name="Stability (CSI)", index=False)
        pd.DataFrame(metrics["benchmark"]).T.to_excel(xl, sheet_name="Benchmark")
        waterfall.to_excel(xl, sheet_name="Sample waterfall", index=False)
        pd.DataFrame(robust).to_excel(xl, sheet_name="Robustness (splits)", index=False)

    # 8. Figures --------------------------------------------------------------
    plt.figure(figsize=(5, 5))
    for k in sets:
        fpr, tpr, _ = roc_curve(sets[k].bad, pds[k])
        plt.plot(fpr, tpr, label=f"{k}  Gini {metrics['discrimination'][k]['gini']:.3f}")
    fpr, tpr, _ = roc_curve(oot.bad, lc_grade["oot"])
    plt.plot(fpr, tpr, "--", label=f"LC sub-grade (OOT)  Gini {metrics['benchmark']['oot']['lc_subgrade_gini']:.3f}")
    plt.plot([0, 1], [0, 1], ":", color="grey")
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.title("ROC")
    plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(FIG / "roc.png", dpi=130); plt.close()

    plt.figure(figsize=(5.5, 4))
    x = calib_oot["g"].astype(int)
    plt.bar(x - 0.2, calib_oot.predicted, 0.4, label="Predicted PD")
    plt.bar(x + 0.2, calib_oot.observed, 0.4, label="Observed bad rate")
    plt.xlabel("PD decile (1 = safest)"); plt.ylabel("Bad rate"); plt.title("Calibration, out-of-time 2011")
    plt.legend(); plt.tight_layout(); plt.savefig(FIG / "calibration_oot.png", dpi=130); plt.close()

    plt.figure(figsize=(5.5, 4))
    bins = np.linspace(min(scores["train"].min(), scores["oot"].min()),
                       max(scores["train"].max(), scores["oot"].max()), 40)
    plt.hist(scores["train"], bins, alpha=0.5, density=True, label="Development (train)")
    plt.hist(scores["oot"], bins, alpha=0.5, density=True, label="Out-of-time 2011")
    plt.xlabel("Score"); plt.title(f"Score distribution  PSI {metrics['stability']['score_psi_train_vs_oot']:.3f}")
    plt.legend(); plt.tight_layout(); plt.savefig(FIG / "score_distribution.png", dpi=130); plt.close()

    plt.figure(figsize=(5.5, 4))
    plt.bar(bands_oot.band.astype(int), bands_oot.bad_rate)
    plt.xlabel("Score decile (1 = lowest score)"); plt.ylabel("Observed bad rate")
    plt.title("Rank ordering, out-of-time 2011"); plt.tight_layout()
    plt.savefig(FIG / "rank_ordering_oot.png", dpi=130); plt.close()

    print(json.dumps({k: metrics[k] for k in ("discrimination", "calibration", "stability", "benchmark")}, indent=1))
    print("features:", features); print("eliminated:", elim_log); print("vif:", metrics["model"]["vif"])


if __name__ == "__main__":
    main()
