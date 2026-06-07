# Missing Data Report

*Generated: 2026-06-07 20:14:02*

## Overview

| Metric | Value |
|--------|-------|
| Total rows | 800 |
| Total columns | 3 |
| Columns with missing values | 1 |
| Overall missing rate | 10.0% |

## Missingness Profile

| Column | Missing Rate | Severity | Dtype | Mechanism Clue |
|--------|-------------|----------|-------|----------------|
| `credit_score` | 30.1% | moderate | numeric | MNAR/structural concern |

## Imputation Plan

| Column | Strategy | Add Indicator | Fit On | Reason |
|--------|----------|--------------|--------|--------|
| `credit_score` | `numeric_median_plus_indicator` | Yes | train_only | mechanism='MNAR/structural concern'_or_moderate_missing_rate |

## Strategy Summary

- `no_imputation_needed`: 2 column(s)
- `numeric_median_plus_indicator`: 1 column(s)

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
