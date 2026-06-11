# Missing Data Report

*Generated: 2026-06-11 10:46:59*

## Overview

This report diagnoses missingness patterns and recommends a leakage-safe imputation strategy for each column. Mechanism labels are statistical clues from observational data — they do not confirm causal mechanisms.

| Metric | Value |
|--------|-------|
| Total rows | 780 |
| Total columns | 7 |
| Columns with missing values | 3 |
| Overall missing rate | 8.3% |

---

## Missingness Profile

| Column | Missing Rate | Severity | Dtype | Cardinality | Mechanism Clue |
|--------|-------------|----------|-------|-------------|----------------|
| `ed_visit_rate` | 14.5% | low | numeric | 539 | group-dependent missingness |
| `naloxone_admin_rate` | 10.0% | low | numeric | 647 | MCAR-compatible |
| `subprogram_type` | 33.3% | moderate | categorical | 3 | group-dependent missingness |

---

## Mechanism Clues

> **Caution:** These labels are observational clues, not causal claims. MCAR, MAR, and MNAR cannot be confirmed from observational data alone. MAR is the recommended default assumption (van Buuren FIMD). Use these clues to inform — not dictate — imputation choices.

- **group-dependent missingness** (2 column(s)): Missingness concentrated in specific categorical groups. Use groupwise imputation or missing token. This is NOT structural absence — data exists in principle but was not collected for certain groups.
- **MCAR-compatible** (1 column(s)): No significant correlation with other features or target. Median/token imputation is sufficient. MAR assumption likely robust when missing rate < 25% and max correlation < 0.4 (Collins et al., 2001).

---

## Group-Dependent Missingness

The following columns have missingness concentrated in specific categorical groups. This is **not** structural absence — the data exists for some groups but not others (e.g., a measurement collected only in certain jurisdictions). Recommended: groupwise imputation or missing token.

### `ed_visit_rate`
- Correlated group feature: `state`
- Max missingness spread across groups: 46.2%

### `subprogram_type`
- Correlated group feature: `state`
- Max missingness spread across groups: 100.0%
- High-missing groups: `AL` (100%), `CA` (100%), `MI` (100%), `NC` (100%), `OH` (100%)

---

## Structural Absence

The following column pairs show structural absence: when the categorical column is NaN, its numeric companion is 0 or also NaN. This pattern suggests the NaN encodes the **absence of a facility or item**, not a data collection error. Impute with a structural token/zero + indicator.

| Categorical (NA) | Numeric (companion) | Pattern | Explanation |
|-----------------|---------------------|---------|-------------|
| `subprogram_type` | `copresent_subrate` | numeric_near_zero_when_cat_na | companion_numeric_zero_when_categorical_na |

---

## Imputation Plan

All imputation statistics must be fitted on **training data only**.

| Column | Strategy | Indicator | Mechanism | Missing Rate | Reason |
|--------|----------|-----------|-----------|-------------|--------|
| `ed_visit_rate` | `groupwise_numeric_median_plus_indicator` | Yes | group-dependent missingness | 14.5% | group_dependent_missingness_by_feature='state' |
| `naloxone_admin_rate` | `time_series_ffill_bfill_plus_indicator` | Yes | MCAR-compatible | 10.0% | domain='time_series_metric'_temporal_measurement_ffill_preferred_over_median |
| `subprogram_type` | `structural_none_token_plus_indicator` | Yes | structural absence concern | 33.3% | structural_absence_categorical_na_encodes_absence |

### Safety Warnings

- `naloxone_admin_rate`: sort_by_time_column_before_applying_ffill_bfill

### Strategy Summary

- `no_imputation_needed`: 4 column(s)
- `groupwise_numeric_median_plus_indicator`: 1 column(s)
- `time_series_ffill_bfill_plus_indicator`: 1 column(s)
- `structural_none_token_plus_indicator`: 1 column(s)

---

## When to Use Full Multiple Imputation (MICE)

The following columns have conditions where simple median/token imputation may be insufficient for **statistical inference** (confidence intervals, hypothesis tests). For those use cases, upgrade to full MICE. Criteria: missing rate > 25%, or any predictor correlation > 0.40, or missingness correlated with target (Collins et al., 2001 via van Buuren FIMD Ch5).

| Column | Missing Rate | Mechanism | Recommended MICE Method |
|--------|-------------|-----------|------------------------|
| `ed_visit_rate` | 14.5% | group-dependent missingness | `pmm (predictive mean matching)` |
| `subprogram_type` | 33.3% | structural absence concern | `polyreg/polr (multinomial/ordered logit)` |

> **Note:** For ML feature engineering (prediction only), the simple strategies above are acceptable. The goal of full MI is to reflect uncertainty due to missing data, not to predict missing values accurately (Rubin 1987, van Buuren FIMD Ch2).

---

## Per-Column Evidence

Evidence collected for each recommendation:

### `ed_visit_rate`
- **Strategy:** `groupwise_numeric_median_plus_indicator`
- **Missing rate:** 14.5%
- **Mechanism label:** group-dependent missingness
- **MAR robustness:** insufficient_evidence_to_assess
- **Full MI recommended:** Yes (consider MICE for inference tasks)
- **Group dependency:** detected via `state` (spread=0.46)

### `naloxone_admin_rate`
- **Strategy:** `time_series_ffill_bfill_plus_indicator`
- **Missing rate:** 10.0%
- **Mechanism label:** MCAR-compatible
- **MAR robustness:** likely_robust_per_collins2001
- **Safety warning:** sort_by_time_column_before_applying_ffill_bfill

### `subprogram_type`
- **Strategy:** `structural_none_token_plus_indicator`
- **Missing rate:** 33.3%
- **Mechanism label:** structural absence concern
- **MAR robustness:** mechanism_assumption_may_matter_consider_full_mi
- **Full MI recommended:** Yes (consider MICE for inference tasks)
- **Group dependency:** detected via `state` (spread=1.00)
- **Covariate association:** `copresent_subrate` (r=0.726)
- **Structural evidence:** companion columns=['copresent_subrate']

---

## Leakage-Safe Imputation Protocol

> **Rule:** Fit all imputation statistics on training data only. Never use prediction/test data to compute medians, modes, or model imputation parameters.

**sklearn pattern:**
```python
use Pipeline([('imputer', SimpleImputer()), ('model', ...)]); pipeline.fit(X_train, y_train)
```

---

## Multiple Imputation Results (Rubin's Rules)

*Method: MICE (IterativeImputer + BayesianRidge, sample_posterior) pooled with Rubin's rules; m = 5 imputations; estimand = per-column mean.*

Single imputation treats filled values as certain and understates variance. The pooled standard error below propagates the extra uncertainty from missingness via Rubin's rules. **FMI** is the fraction of information about the estimand lost to missing data.

| Column | Missing | Pooled mean | Naive SE (single) | Pooled SE (MI) | 95% CI | FMI |
|--------|---------|------------|-------------------|----------------|--------|-----|
| `ed_visit_rate` | 14.5% | 10.56 | 0.1787 | 0.1795 | [10.2, 10.9] | 0.01 |
| `naloxone_admin_rate` | 10.0% | 23.86 | 0.4562 | 0.4602 | [23, 24.8] | 0.02 |

> The pooled (MI) standard error is ≥ the naive single-imputation SE by construction — that gap is exactly the uncertainty single imputation hides (Rubin 1987; van Buuren FIMD Ch2).

---

## MNAR Sensitivity Analysis (Delta-Adjustment)

*delta-adjustment (pattern-mixture) MNAR sensitivity analysis; tipping-point rule: smallest |delta| (in observed-SD units) at which the full-sample mean leaves the complete-case 95% CI.*

Imputation assumes MAR, which cannot be verified from observed data. Each column's imputed values are shifted by `delta` standard deviations; the **tipping point** is the smallest |delta| at which the mean leaves the complete-case 95% CI. A small tipping point ⇒ the conclusion hinges on the untestable MAR assumption.

| Column | Missing | Tipping point (|δ| SD) | Robustness |
|--------|---------|------------------------|------------|
| `ed_visit_rate` | 14.5% | 0.5 | fragile |

> ⚠️ **Fragile columns** (tip at |δ| ≤ 0.5 SD): `ed_visit_rate`. MAR-based estimates for these are sensitive to plausible MNAR departures — gather domain evidence on why values are missing (van Buuren FIMD Ch9).

---

## Limitations

- Mechanism labels (MCAR-compatible, MAR-like, group-dependent, target-associated) are statistical clues from observational data. They cannot confirm the true causal mechanism (van Buuren FIMD Ch1).
- MAR is the default assumption. MNAR cannot be confirmed or ruled out from observational data alone. Sensitivity analysis (e.g., delta-adjustment) is needed when MNAR is suspected (van Buuren FIMD Ch5).
- Simple median/token imputation (single imputation) underestimates variance and produces confidence intervals that are too narrow. Use full MICE with Rubin's rules when valid statistical inference is required (van Buuren FIMD Ch1, Table 1.1).
- The MAR robustness note uses Collins et al. (2001) thresholds: missing rate < 25% and max correlation < 0.4. These are empirical guidelines, not hard cutoffs.
- Group-dependent missingness may overlap with target-associated missingness when the grouping variable (e.g., jurisdiction) also correlates with the target.
- Structural absence detection relies on categorical NA + numeric zero/absent co-occurrence patterns. False positives are possible for columns where zero is a common legitimate value.
- High-cardinality detection uses heuristic thresholds (>50 unique values or >20% unique ratio). Domain knowledge should override these thresholds.
- Groupwise imputation uses training-set group medians fitted on observed rows only. Groups with fewer than 5 observations fall back to the global median.
