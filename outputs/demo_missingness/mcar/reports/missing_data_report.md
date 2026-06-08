# Missing Data Report

*Generated: 2026-06-08 11:59:58*

## Overview

This report diagnoses missingness patterns and recommends a leakage-safe imputation strategy for each column. Mechanism labels are statistical clues from observational data — they do not confirm causal mechanisms.

| Metric | Value |
|--------|-------|
| Total rows | 800 |
| Total columns | 4 |
| Columns with missing values | 1 |
| Overall missing rate | 1.7% |

---

## Missingness Profile

| Column | Missing Rate | Severity | Dtype | Cardinality | Mechanism Clue |
|--------|-------------|----------|-------|-------------|----------------|
| `measurement_a` | 6.8% | low | numeric | 746 | MCAR-compatible |

---

## Mechanism Clues

> **Caution:** These labels are observational clues, not causal claims. MCAR, MAR, and MNAR cannot be confirmed from observational data alone. Use these clues to inform — not dictate — imputation choices.

- **MCAR-compatible** (1 column(s)): No significant correlation with other features or target. Median/token imputation is sufficient.

---

## Imputation Plan

All imputation statistics must be fitted on **training data only**.

| Column | Strategy | Indicator | Mechanism | Missing Rate | Reason |
|--------|----------|-----------|-----------|-------------|--------|
| `measurement_a` | `numeric_median` | No | MCAR-compatible | 6.8% | mcar_compatible_low_missing_rate |

### Strategy Summary

- `no_imputation_needed`: 3 column(s)
- `numeric_median`: 1 column(s)

---

## Per-Column Evidence

Evidence collected for each recommendation:

### `measurement_a`
- **Strategy:** `numeric_median`
- **Missing rate:** 6.8%
- **Mechanism label:** MCAR-compatible

---

## Leakage-Safe Imputation Protocol

> **Rule:** Fit all imputation statistics on training data only. Never use prediction/test data to compute medians, modes, or model imputation parameters.

**sklearn pattern:**
```python
use Pipeline([('imputer', SimpleImputer()), ('model', ...)]); pipeline.fit(X_train, y_train)
```

---

## Limitations

- Mechanism labels (MCAR-compatible, MAR-like, group-dependent, target-associated) are statistical clues from observational data. They cannot confirm the true causal mechanism.
- Group-dependent missingness may overlap with target-associated missingness when the grouping variable (e.g., jurisdiction) also correlates with the target.
- Structural absence detection relies on categorical NA + numeric zero/absent co-occurrence patterns. False positives are possible for columns where zero is a common legitimate value.
- High-cardinality detection uses heuristic thresholds (>50 unique values or >20% unique ratio). Domain knowledge should override these thresholds.
- Groupwise imputation uses training-set group medians fitted on observed rows only. Groups with fewer than 5 observations fall back to the global median.
