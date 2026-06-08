# Missing Data Report

*Generated: 2026-06-08 19:30:50*

## Overview

This report diagnoses missingness patterns and recommends a leakage-safe imputation strategy for each column. Mechanism labels are statistical clues from observational data — they do not confirm causal mechanisms.

| Metric | Value |
|--------|-------|
| Total rows | 1000 |
| Total columns | 7 |
| Columns with missing values | 4 |
| Overall missing rate | 11.0% |

---

## Missingness Profile

| Column | Missing Rate | Severity | Dtype | Cardinality | Mechanism Clue |
|--------|-------------|----------|-------|-------------|----------------|
| `age` | 10.1% | low | numeric | 887 | MCAR-compatible |
| `income` | 17.7% | low | numeric | 823 | group-dependent missingness |
| `risk_score` | 11.1% | low | numeric | 889 | target-associated missingness |
| `facility_quality` | 37.8% | moderate | categorical | 3 | MAR-like evidence |

---

## Mechanism Clues

> **Caution:** These labels are observational clues, not causal claims. MCAR, MAR, and MNAR cannot be confirmed from observational data alone. MAR is the recommended default assumption (van Buuren FIMD). Use these clues to inform — not dictate — imputation choices.

- **MCAR-compatible** (1 column(s)): No significant correlation with other features or target. Median/token imputation is sufficient. MAR assumption likely robust when missing rate < 25% and max correlation < 0.4 (Collins et al., 2001).
- **group-dependent missingness** (1 column(s)): Missingness concentrated in specific categorical groups. Use groupwise imputation or missing token. This is NOT structural absence — data exists in principle but was not collected for certain groups.
- **target-associated missingness** (1 column(s)): Missingness correlates with the target variable. Missing indicator is important. Consider full MICE for inference tasks (including target in imputation model is correct and reduces bias toward zero).
- **MAR-like evidence** (1 column(s)): Missingness correlates with at least one observed numeric feature. Add missing indicator. Include correlated features as predictors in any downstream imputation model.

---

## Group-Dependent Missingness

The following columns have missingness concentrated in specific categorical groups. This is **not** structural absence — the data exists for some groups but not others (e.g., a measurement collected only in certain jurisdictions). Recommended: groupwise imputation or missing token.

### `income`
- Correlated group feature: `education_level`
- Max missingness spread across groups: 30.0%

---

## Structural Absence

The following column pairs show structural absence: when the categorical column is NaN, its numeric companion is 0 or also NaN. This pattern suggests the NaN encodes the **absence of a facility or item**, not a data collection error. Impute with a structural token/zero + indicator.

| Categorical (NA) | Numeric (companion) | Pattern | Evidence |
|-----------------|---------------------|---------|----------|
| `facility_quality` | `facility_area` | numeric_near_zero_when_cat_na | companion_numeric_zero_when_categorical_na |

---

## Imputation Plan

All imputation statistics must be fitted on **training data only**.

| Column | Strategy | Indicator | Mechanism | Missing Rate | Reason |
|--------|----------|-----------|-----------|-------------|--------|
| `age` | `numeric_median_plus_indicator` | Yes | MCAR-compatible | 10.1% | mechanism='MCAR-compatible'_or_moderate_missing_rate |
| `income` | `groupwise_numeric_median_plus_indicator` | Yes | group-dependent missingness | 17.7% | group_dependent_missingness_by_feature='education_level' |
| `risk_score` | `numeric_median_plus_indicator` | Yes | target-associated missingness | 11.1% | mechanism='target-associated missingness'_or_moderate_missing_rate |
| `facility_quality` | `structural_none_token_plus_indicator` | Yes | structural absence concern | 37.8% | structural_absence_categorical_na_encodes_absence |

### Strategy Summary

- `no_imputation_needed`: 3 column(s)
- `numeric_median_plus_indicator`: 2 column(s)
- `groupwise_numeric_median_plus_indicator`: 1 column(s)
- `structural_none_token_plus_indicator`: 1 column(s)

---

## When to Use Full Multiple Imputation (MICE)

The following columns have conditions where simple median/token imputation may be insufficient for **statistical inference** (confidence intervals, hypothesis tests). For those use cases, upgrade to full MICE. Criteria: missing rate > 25%, or any predictor correlation > 0.40, or missingness correlated with target (Collins et al., 2001 via van Buuren FIMD Ch5).

| Column | Missing Rate | Mechanism | Recommended MICE Method |
|--------|-------------|-----------|------------------------|
| `income` | 17.7% | group-dependent missingness | `pmm (predictive mean matching)` |
| `risk_score` | 11.1% | target-associated missingness | `pmm (predictive mean matching)` |
| `facility_quality` | 37.8% | structural absence concern | `polyreg/polr (multinomial/ordered logit)` |

> **Note:** For ML feature engineering (prediction only), the simple strategies above are acceptable. The goal of full MI is to reflect uncertainty due to missing data, not to predict missing values accurately (Rubin 1987, van Buuren FIMD Ch2).

---

## Per-Column Evidence

Evidence collected for each recommendation:

### `age`
- **Strategy:** `numeric_median_plus_indicator`
- **Missing rate:** 10.1%
- **Mechanism label:** MCAR-compatible
- **MAR robustness:** likely_robust_per_collins2001

### `income`
- **Strategy:** `groupwise_numeric_median_plus_indicator`
- **Missing rate:** 17.7%
- **Mechanism label:** group-dependent missingness
- **MAR robustness:** insufficient_evidence_to_assess
- **Full MI recommended:** Yes (consider MICE for inference tasks)
- **Group dependency:** detected via `education_level` (spread=0.30)

### `risk_score`
- **Strategy:** `numeric_median_plus_indicator`
- **Missing rate:** 11.1%
- **Mechanism label:** target-associated missingness
- **MAR robustness:** mechanism_assumption_may_matter_consider_full_mi
- **Full MI recommended:** Yes (consider MICE for inference tasks)
- **Target association:** correlation=0.427
- **Covariate association:** `income` (r=0.422)

### `facility_quality`
- **Strategy:** `structural_none_token_plus_indicator`
- **Missing rate:** 37.8%
- **Mechanism label:** structural absence concern
- **MAR robustness:** mechanism_assumption_may_matter_consider_full_mi
- **Full MI recommended:** Yes (consider MICE for inference tasks)
- **Covariate association:** `facility_area` (r=0.825)
- **Structural evidence:** companion columns=['facility_area']

---

## Leakage-Safe Imputation Protocol

> **Rule:** Fit all imputation statistics on training data only. Never use prediction/test data to compute medians, modes, or model imputation parameters.

**sklearn pattern:**
```python
use Pipeline([('imputer', SimpleImputer()), ('model', ...)]); pipeline.fit(X_train, y_train)
```

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
