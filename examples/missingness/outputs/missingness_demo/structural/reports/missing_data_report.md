# Missing Data Report

*Generated: 2026-06-08 18:52:09*

## Overview

This report diagnoses missingness patterns and recommends a leakage-safe imputation strategy for each column. Mechanism labels are statistical clues from observational data — they do not confirm causal mechanisms.

| Metric | Value |
|--------|-------|
| Total rows | 200 |
| Total columns | 4 |
| Columns with missing values | 1 |
| Overall missing rate | 10.0% |

---

## Missingness Profile

| Column | Missing Rate | Severity | Dtype | Cardinality | Mechanism Clue |
|--------|-------------|----------|-------|-------------|----------------|
| `facility_type` | 40.0% | moderate | categorical | 2 | target-associated missingness |

---

## Mechanism Clues

> **Caution:** These labels are observational clues, not causal claims. MCAR, MAR, and MNAR cannot be confirmed from observational data alone. MAR is the recommended default assumption (van Buuren FIMD). Use these clues to inform — not dictate — imputation choices.

- **target-associated missingness** (1 column(s)): Missingness correlates with the target variable. Missing indicator is important. Consider full MICE for inference tasks (including target in imputation model is correct and reduces bias toward zero).

---

## Structural Absence

The following column pairs show structural absence: when the categorical column is NaN, its numeric companion is 0 or also NaN. This pattern suggests the NaN encodes the **absence of a facility or item**, not a data collection error. Impute with a structural token/zero + indicator.

| Categorical (NA) | Numeric (companion) | Pattern | Evidence |
|-----------------|---------------------|---------|----------|
| `facility_type` | `facility_capacity` | numeric_near_zero_when_cat_na | companion_numeric_zero_when_categorical_na |

---

## Imputation Plan

All imputation statistics must be fitted on **training data only**.

| Column | Strategy | Indicator | Mechanism | Missing Rate | Reason |
|--------|----------|-----------|-----------|-------------|--------|
| `facility_type` | `structural_none_token_plus_indicator` | Yes | structural absence concern | 40.0% | structural_absence_categorical_na_encodes_absence |

### Strategy Summary

- `no_imputation_needed`: 3 column(s)
- `structural_none_token_plus_indicator`: 1 column(s)

---

## When to Use Full Multiple Imputation (MICE)

The following columns have conditions where simple median/token imputation may be insufficient for **statistical inference** (confidence intervals, hypothesis tests). For those use cases, upgrade to full MICE. Criteria: missing rate > 25%, or any predictor correlation > 0.40, or missingness correlated with target (Collins et al., 2001 via van Buuren FIMD Ch5).

| Column | Missing Rate | Mechanism | Recommended MICE Method |
|--------|-------------|-----------|------------------------|
| `facility_type` | 40.0% | structural absence concern | `logreg (logistic regression)` |

> **Note:** For ML feature engineering (prediction only), the simple strategies above are acceptable. The goal of full MI is to reflect uncertainty due to missing data, not to predict missing values accurately (Rubin 1987, van Buuren FIMD Ch2).

---

## Per-Column Evidence

Evidence collected for each recommendation:

### `facility_type`
- **Strategy:** `structural_none_token_plus_indicator`
- **Missing rate:** 40.0%
- **Mechanism label:** structural absence concern
- **MAR robustness:** mechanism_assumption_may_matter_consider_full_mi
- **Full MI recommended:** Yes (consider MICE for inference tasks)
- **Target association:** correlation=0.101
- **Covariate association:** `facility_capacity` (r=0.811)
- **Structural evidence:** companion columns=['facility_capacity']

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
