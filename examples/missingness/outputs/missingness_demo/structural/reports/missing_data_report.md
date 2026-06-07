# Missing Data Report

*Generated: 2026-06-07 20:14:02*

## Overview

| Metric | Value |
|--------|-------|
| Total rows | 200 |
| Total columns | 4 |
| Columns with missing values | 1 |
| Overall missing rate | 10.0% |

## Missingness Profile

| Column | Missing Rate | Severity | Dtype | Mechanism Clue |
|--------|-------------|----------|-------|----------------|
| `facility_type` | 40.0% | moderate | categorical | MNAR/structural concern |

## Structural Missingness

The following column pairs show structural absence patterns (e.g. no facility → both the category and count are absent):

- `facility_type` (categorical NA) → `facility_capacity` (numeric_near_zero_when_cat_na)

## Imputation Plan

| Column | Strategy | Add Indicator | Fit On | Reason |
|--------|----------|--------------|--------|--------|
| `facility_type` | `structural_none_or_zero` | Yes | train_only | structural_absence_pattern_detected |

## Strategy Summary

- `no_imputation_needed`: 3 column(s)
- `structural_none_or_zero`: 1 column(s)

## Leakage-Safe Imputation Protocol

> **Rule:** Fit all imputation statistics on training data only. Never use prediction/test data to compute medians, modes, or model imputation parameters.

**sklearn pattern:**
```python
use Pipeline([('imputer', SimpleImputer()), ('model', ...)]); pipeline.fit(X_train, y_train)
```

**Missing indicator pattern:**
```python
use MissingIndicator(features='missing-only') before imputation in pipeline
```
