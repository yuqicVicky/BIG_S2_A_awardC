# Missing Data Report

*Generated: 2026-06-08 11:59:58*

## Overview

This report diagnoses missingness patterns and recommends a leakage-safe imputation strategy for each column. Mechanism labels are statistical clues from observational data — they do not confirm causal mechanisms.

| Metric | Value |
|--------|-------|
| Total rows | 800 |
| Total columns | 3 |
| Columns with missing values | 1 |
| Overall missing rate | 10.0% |

---

## Missingness Profile

| Column | Missing Rate | Severity | Dtype | Cardinality | Mechanism Clue |
|--------|-------------|----------|-------|-------------|----------------|
| `credit_score` | 30.1% | moderate | numeric | 559 | target-associated missingness |

---

## Mechanism Clues

> **Caution:** These labels are observational clues, not causal claims. MCAR, MAR, and MNAR cannot be confirmed from observational data alone. Use these clues to inform — not dictate — imputation choices.

- **target-associated missingness** (1 column(s)): Missingness correlates with the target variable. Missing indicator is important.

---

## Imputation Plan

All imputation statistics must be fitted on **training data only**.

| Column | Strategy | Indicator | Mechanism | Missing Rate | Reason |
|--------|----------|-----------|-----------|-------------|--------|
| `credit_score` | `numeric_median_plus_indicator` | Yes | target-associated missingness | 30.1% | mechanism='target-associated missingness'_or_moderate_missing_rate |

### Strategy Summary

- `no_imputation_needed`: 2 column(s)
- `numeric_median_plus_indicator`: 1 column(s)

---

## Per-Column Evidence

Evidence collected for each recommendation:

### `credit_score`
- **Strategy:** `numeric_median_plus_indicator`
- **Missing rate:** 30.1%
- **Mechanism label:** target-associated missingness
- **Target association:** correlation=1.000

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
