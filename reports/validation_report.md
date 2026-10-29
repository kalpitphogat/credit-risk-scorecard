# Model Validation Report — Consumer Loan PD Scorecard

All figures are produced by `python run_all.py` and stored in `outputs/metrics.json` and `outputs/scorecard_validation.xlsx`.

## 1. Summary

| Area | Result | Assessment |
|---|---|---|
| Discrimination (out-of-time 2011) | Gini 0.295, KS 0.221, AUC 0.647 | Comparable to the lender's own grade |
| Discrimination stability | Gini 0.310 train → 0.297 test → 0.295 OOT | Stable |
| Rank ordering (OOT) | Bad rate falls from 20.6% (lowest-score decile) to 4.0% (highest) | Acceptable: 1 minor reversal |
| Calibration (OOT) | Mean predicted PD 10.9% vs observed 10.6%; Hosmer–Lemeshow p = 0.031 | **Finding:** mild over-prediction |
| Population stability | Score PSI 0.013; all characteristic PSIs ≤ 0.023 | Stable (< 0.10) |
| Benchmark | Lending Club sub-grade Gini 0.270; gradient-boosting challenger 0.295 | Scorecard matches both |

**Conclusion:** the scorecard ranks risk consistently out-of-time and its inputs are stable. Calibration has drifted slightly as the 2011 default rate fell. The recommendation is to keep using the model for ranking, and to recalibrate the intercept once newer outcomes are available.

## 2. Model under review

- **Purpose:** estimate the probability that a 36-month unsecured personal loan defaults (charged off) before maturity, using information known at application.
- **Form:** logistic regression on Weight-of-Evidence (WoE) transformed characteristics, scaled to points. A score of 600 corresponds to good:bad odds of 50:1, and every 20 points doubles the odds. The score range is 458–589.
- **Characteristics (5):**

| Characteristic | Description | IV | Coefficient | p-value | VIF |
|---|---|---|---|---|---|
| purpose | Stated loan purpose | 0.097 | −1.019 | < 0.001 | 1.01 |
| revol_util | Revolving credit utilisation (%) | 0.069 | −1.152 | < 0.001 | 1.01 |
| inq_last_6mths | Credit inquiries, last 6 months | 0.066 | −1.053 | < 0.001 | 1.01 |
| pti | Payment-to-income (annual instalments / income) | 0.063 | −0.779 | < 0.001 | 1.14 |
| annual_inc | Annual income | 0.030 | −0.767 | < 0.001 | 1.13 |

All coefficients have the expected sign (higher WoE = safer = lower PD), all are significant, and the VIFs show no collinearity.

## 3. Data

**Source:** Lending Club loans issued June 2007 – December 2011 (`LoanStats3a`), with payment history through December 2015.

| Step | Rule | Loans |
|---|---|---|
| 1 | All loans in file | 42,535 |
| 2 | 36-month term (every loan has reached maturity, so the outcome is final) | 31,534 |
| 3 | Met Lending Club's credit policy | 29,096 |
| 4 | Resolved outcome (Fully Paid / Charged Off / Default) | 29,095 |

**Target:** bad = Charged Off or Default; good = Fully Paid. The one loan still late was excluded as indeterminate.

| Sample | Issue years | Loans | Bads | Bad rate |
|---|---|---|---|---|
| Train (70% of development) | 2007–2010 | 10,495 | 1,210 | 11.5% |
| Test (30% of development) | 2007–2010 | 4,499 | 518 | 11.5% |
| Out-of-time (OOT) | 2011 | 14,101 | 1,499 | 10.6% |

**Leakage controls:**
- Only application-time fields are offered to the model.
- Payment, recovery and post-origination fields are excluded in SQL (`sql/build_sample.sql`).
- Lending Club's grade, sub-grade and interest rate are kept only as a benchmark.
- All binning and model fitting use the train sample only.

## 4. Methodology review

- **Binning:** numeric characteristics start from 10 quantile bins. Neighbouring bins are merged until every bin holds ≥ 5% of loans and the bad rate is monotonic in the characteristic. Missing values get their own bin. Small categories are pooled with categories of similar bad rate, which keeps `small_business` (23.5% bad rate) in a separate high-risk pool rather than averaged with low-risk purposes.
- **Selection:** 17 candidates → 5 with IV between 0.02 and 0.5 → none dropped for pairwise WoE correlation above 0.6 → none dropped by backward elimination (wrong sign or p > 0.05).
- **Excluded for low IV (< 0.02):** credit-history length, open/total accounts, employment length, DTI, months since last delinquency, loan amount, verification status, home ownership, delinquencies, revolving balance, public records.

## 5. Results

### 5.1 Discrimination

| Sample | AUC | Gini | KS |
|---|---|---|---|
| Train | 0.655 | 0.310 | 0.213 |
| Test | 0.649 | 0.297 | 0.233 |
| OOT 2011 | 0.647 | 0.295 | 0.221 |

The drop from train to OOT is 0.015 Gini, so there is no sign of overfitting.

**Robustness:** refitting on five other random development splits gives OOT Gini between 0.287 and 0.302.

### 5.2 Benchmarking (out-of-time)

| Model | Gini | 95% bootstrap CI |
|---|---|---|
| This scorecard | 0.295 | 0.263 – 0.325 |
| Lending Club sub-grade (lender's own risk grade) | 0.270 | 0.239 – 0.298 |
| Gradient-boosting challenger (same raw inputs) | 0.295 | 0.266 – 0.326 |

- **Scorecard vs. Lending Club sub-grade:** +0.025 Gini (95% CI −0.008 to +0.056; scorecard ahead in 93% of 1,000 bootstrap resamples). The difference is not statistically significant at 5%.
- **Challenger vs. scorecard:** +0.001 (95% CI −0.019 to +0.021). There is no meaningful difference.
- **Challenger overfitting:** the challenger reaches Gini 0.568 on train but only 0.296 on test. The transparent scorecard achieves the same out-of-time ranking without that overfitting.

### 5.3 Rank ordering (OOT, score deciles)

The bad rate falls from **20.6%** in the lowest-score decile to **4.0%** in the highest. There is one small reversal: 7.5% → 8.4% between deciles 7 and 8.

Each decile has about 1,410 loans, so the standard error of the difference between two adjacent deciles is about ±1.0 point. The 0.9-point reversal is within sampling noise.

### 5.4 Calibration

| Sample | Mean predicted PD | Observed bad rate | Brier | Hosmer–Lemeshow p |
|---|---|---|---|---|
| Train | 11.5% | 11.5% | 0.098 | 0.23 |
| Test | 11.6% | 11.5% | 0.099 | 0.001 |
| OOT 2011 | 10.9% | 10.6% | 0.093 | 0.03 |

- **Test sample:** the failure is specific to this random split. On the five alternative splits, the test-sample p-values are 0.06–0.25, all passing.
- **OOT sample:** the p-value stays low across splits (0.016–0.139). This is a real, mild over-prediction, consistent with the lower 2011 default rate (10.6% vs 11.5% in development).
- **Per decile:** 1 of 10 OOT deciles fails a two-sided binomial test at 5%, which is roughly what chance alone would produce.

### 5.5 Stability

- **Score PSI (train vs. OOT):** 0.013.
- **Characteristic PSI:**

| Characteristic | PSI |
|---|---|
| purpose | 0.023 |
| inq_last_6mths | 0.014 |
| revol_util | 0.012 |
| pti | 0.005 |
| annual_inc | 0.001 |

All are well below the 0.10 monitoring threshold.

## 6. Findings and recommendations

| # | Finding | Severity | Recommendation |
|---|---|---|---|
| 1 | OOT calibration drift: mean PD over-predicted by 0.3 points; HL p = 0.03 | Medium | Recalibrate the intercept on the most recent outcome window. Keep the binning and the slopes. |
| 2 | Discrimination is moderate (Gini ≈ 0.30) | Low | Expected without bureau scores (e.g. FICO), which this data does not include. Add bureau attributes if available. |
| 3 | One minor rank-ordering reversal (OOT, deciles 7→8) | Low | Within sampling error. Re-check in the next monitoring cycle. |
| 4 | The 5% minimum bin share merges rare events (public records, 2-year delinquencies) into a single bin, so they score IV ≈ 0 | Low | Test a lower minimum share, or hand-crafted flags, for rare derogatory events. |

## 7. Limitations

- Lending Club approved these loans, so declined applicants are absent (no reject inference). The model ranks risk among approved borrowers only.
- The sample covers 2007–2011, which includes the financial crisis. Default behaviour may differ in other credit cycles.
- Only 36-month loans are covered. 60-month loans were excluded because many had not matured at the data snapshot.
