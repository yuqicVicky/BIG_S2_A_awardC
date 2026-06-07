# Missing Data Report

*Generated: 2026-06-07 21:01:32*

## Overview

| Metric | Value |
|--------|-------|
| Total rows | 600 |
| Total columns | 8 |
| Columns with missing values | 4 |
| Overall missing rate | 13.3% |

## Missingness Profile

| Column | Missing Rate | Severity | Dtype | Mechanism Clue |
|--------|-------------|----------|-------|----------------|
| `age` | 7.3% | low | numeric | MCAR-compatible |
| `income` | 34.5% | moderate | numeric | MAR-like evidence |
| `risk_score` | 29.2% | moderate | numeric | MNAR/structural concern |
| `facility_quality` | 35.2% | moderate | categorical | MAR-like evidence |

## Structural Missingness

Column pairs showing structural absence patterns:

- `facility_quality` (categorical NA) → `facility_area` (numeric_near_zero_when_cat_na)

## Imputation Plan

| Column | Strategy | Add Indicator | Fit On | Reason |
|--------|----------|--------------|--------|--------|
| `age` | `numeric_median` | No | train_only | mcar_compatible_low_missing_rate |
| `income` | `numeric_median_plus_indicator` | Yes | train_only | mechanism='MAR-like evidence'_or_moderate_missing_rate |
| `risk_score` | `numeric_median_plus_indicator` | Yes | train_only | mechanism='MNAR/structural concern'_or_moderate_missing_rate |
| `facility_quality` | `structural_none_or_zero` | Yes | train_only | structural_absence_pattern_detected |

## Strategy Summary

- `no_imputation_needed`: 4 column(s)
- `numeric_median_plus_indicator`: 2 column(s)
- `numeric_median`: 1 column(s)
- `structural_none_or_zero`: 1 column(s)

## Leakage-Safe Imputation Protocol

> **Rule:** Fit all imputation statistics on training data only. Never use prediction/test data to compute medians, modes, or model imputation parameters.

**sklearn pattern:**
```python
use Pipeline([('imputer', SimpleImputer()), ('model', ...)]); pipeline.fit(X_train, y_train)
```
