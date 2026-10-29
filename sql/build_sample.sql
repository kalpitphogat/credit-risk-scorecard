-- Modeling sample for the PD scorecard.
-- Input:  raw_loans (Lending Club LoanStats3a, loaded as text columns)
-- Output: model_sample, one row per resolved 36-month loan
--
-- Sample rules
--   * 36-month term only: by the data snapshot (payments through Dec-2015)
--     every 36-month loan issued 2007-2011 has reached maturity, so its
--     outcome is final. Many 60-month loans are still "Current".
--   * Loans that met Lending Club's credit policy only.
--   * Target: bad = 1 for Charged Off / Default, 0 for Fully Paid.
--     Anything else (Late, In Grace Period, Current) is indeterminate and excluded.
--   * Only fields known at application time are kept. Payment, recovery and
--     post-origination fields would leak the outcome.
--   * Lending Club's own grade, sub-grade and interest rate are kept for
--     benchmarking only, never as model inputs.

DROP TABLE IF EXISTS model_sample;

CREATE TABLE model_sample AS
SELECT
    CAST(id AS INTEGER)                                        AS loan_id,
    issue_d,
    CAST(substr(issue_d, -4) AS INTEGER)                       AS issue_year,
    CASE WHEN loan_status IN ('Charged Off', 'Default') THEN 1 ELSE 0 END AS bad,

    -- application-time characteristics
    CAST(loan_amnt AS REAL)                                    AS loan_amnt,
    CAST(annual_inc AS REAL)                                   AS annual_inc,
    CAST(installment AS REAL) * 12.0
        / NULLIF(CAST(annual_inc AS REAL), 0)                  AS pti,      -- payment-to-income
    CAST(dti AS REAL)                                          AS dti,
    emp_length,
    home_ownership,
    verification_status,
    purpose,
    CAST(REPLACE(revol_util, '%', '') AS REAL)                 AS revol_util,
    CAST(revol_bal AS REAL)                                    AS revol_bal,
    CAST(inq_last_6mths AS INTEGER)                            AS inq_last_6mths,
    CAST(delinq_2yrs AS INTEGER)                               AS delinq_2yrs,
    CAST(pub_rec AS INTEGER)                                   AS pub_rec,
    CAST(open_acc AS INTEGER)                                  AS open_acc,
    CAST(total_acc AS INTEGER)                                 AS total_acc,
    CAST(NULLIF(mths_since_last_delinq, '') AS REAL)           AS mths_since_last_delinq,
    earliest_cr_line,

    -- benchmark only (Lending Club's own risk grade)
    grade,
    sub_grade
FROM raw_loans
WHERE TRIM(term) = '36 months'
  AND loan_status IN ('Fully Paid', 'Charged Off', 'Default');


-- Exclusion waterfall, for the validation report
DROP TABLE IF EXISTS sample_waterfall;

CREATE TABLE sample_waterfall AS
SELECT 1 AS step, 'All loans in file' AS rule, COUNT(*) AS n
  FROM raw_loans WHERE id GLOB '[0-9]*'
UNION ALL
SELECT 2, '36-month term', COUNT(*)
  FROM raw_loans WHERE TRIM(term) = '36 months'
UNION ALL
SELECT 3, 'Met credit policy (36-month)', COUNT(*)
  FROM raw_loans WHERE TRIM(term) = '36 months'
   AND loan_status NOT LIKE 'Does not meet%'
UNION ALL
SELECT 4, 'Resolved outcome: Fully Paid / Charged Off / Default', COUNT(*)
  FROM model_sample;
