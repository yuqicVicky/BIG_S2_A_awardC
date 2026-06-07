# Missing Data Report

*Generated: 2026-06-07 20:14:02*

## Overview

| Metric | Value |
|--------|-------|
| Total rows | 800 |
| Total columns | 4 |
| Columns with missing values | 1 |
| Overall missing rate | 1.7% |

## Missingness Profile

| Column | Missing Rate | Severity | Dtype | Mechanism Clue |
|--------|-------------|----------|-------|----------------|
| `measurement_a` | 6.8% | low | numeric | MCAR-compatible |

## Imputation Plan

| Column | Strategy | Add Indicator | Fit On | Reason |
|--------|----------|--------------|--------|--------|
| `measurement_a` | `numeric_median` | No | train_only | mcar_compatible_low_missing_rate |

## Strategy Summary

- `no_imputation_needed`: 3 column(s)
- `numeric_median`: 1 column(s)

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
