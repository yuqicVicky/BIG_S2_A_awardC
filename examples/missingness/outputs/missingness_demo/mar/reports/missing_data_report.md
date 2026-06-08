# Missing Data Report

*Generated: 2026-06-08 18:52:08*

## Overview

This report diagnoses missingness patterns and recommends a leakage-safe imputation strategy for each column. Mechanism labels are statistical clues from observational data — they do not confirm causal mechanisms.

| Metric | Value |
|--------|-------|
| Total rows | 800 |
| Total columns | 4 |
| Columns with missing values | 1 |
| Overall missing rate | 8.0% |

---

## Missingness Profile

| Column | Missing Rate | Severity | Dtype | Cardinality | Mechanism Clue |
|--------|-------------|----------|-------|-------------|----------------|
| `income` | 31.9% | moderate | numeric | 545 | MAR-like evidence |

---

## Mechanism Clues

> **Caution:** These labels are observational clues, not causal claims. MCAR, MAR, and MNAR cannot be confirmed from observational data alone. MAR is the recommended default assumption (van Buuren FIMD). Use these clues to inform — not dictate — imputation choices.

- **MAR-like evidence** (1 column(s)): Missingness correlates with at least one observed numeric feature. Add missing indicator. Include correlated features as predictors in any downstream imputation model.

---

## Imputation Plan

All imputation statistics must be fitted on **training data only**.

| Column | Strategy | Indicator | Mechanism | Missing Rate | Reason |
|--------|----------|-----------|-----------|-------------|--------|
| `income` | `numeric_median_plus_indicator` | Yes | MAR-like evidence | 31.9% | mechanism='MAR-like evidence'_or_moderate_missing_rate |

### Strategy Summary

- `no_imputation_needed`: 3 column(s)
- `numeric_median_plus_indicator`: 1 column(s)

---

## When to Use Full Multiple Imputation (MICE)

The following columns have conditions where simple median/token imputation may be insufficient for **statistical inference** (confidence intervals, hypothesis tests). For those use cases, upgrade to full MICE. Criteria: missing rate > 25%, or any predictor correlation > 0.40, or missingness correlated with target (Collins et al., 2001 via van Buuren FIMD Ch5).

| Column | Missing Rate | Mechanism | Recommended MICE Method |
|--------|-------------|-----------|------------------------|
| `income` | 31.9% | MAR-like evidence | `pmm (predictive mean matching)` |

> **Note:** For ML feature engineering (prediction only), the simple strategies above are acceptable. The goal of full MI is to reflect uncertainty due to missing data, not to predict missing values accurately (Rubin 1987, van Buuren FIMD Ch2).

---

## Per-Column Evidence

Evidence collected for each recommendation:

### `income`
- **Strategy:** `numeric_median_plus_indicator`
- **Missing rate:** 31.9%
- **Mechanism label:** MAR-like evidence
- **MAR robustness:** mechanism_assumption_may_matter_consider_full_mi
- **Full MI recommended:** Yes (consider MICE for inference tasks)
- **Covariate association:** `education_level` (r=0.866)

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
