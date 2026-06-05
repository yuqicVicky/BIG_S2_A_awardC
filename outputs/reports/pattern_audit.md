# Data Pattern & Validation Audit Report

_Generated: 2026-06-06 07:25:39_

## 1. Schema Summary

- **Train rows:** 1350
- **Train columns:** 9
- **Predict rows:** 150
- **Predict columns:** 8
- **Train-only columns:** target

## 2. Feature Type Summary

| Type | Count |
|------|-------|
| datetime_like | 3 |
| categorical_low_cardinality | 2 |
| group_entity_id | 1 |
| constant | 1 |
| numeric_continuous | 1 |
| target | 1 |

## 3. Detected Split Pattern

- **Pattern:** `within_period_group`
  - timestamp: predict starts after train ends → time-based split
  - hour: predict overlaps or precedes train max → within-period pattern
  - dayofweek: predict overlaps or precedes train max → within-period pattern

## 4. Distribution Shift

No significant numeric distribution shifts detected.

## 5. Target Pattern Highlights

**Top numeric correlations with target:**

| Column | Pearson r |
|--------|-----------|
| numeric_feature | 0.1264 |

## 6. Leakage & Feature Availability Audit

- **High risk:** 0  **Medium risk:** 0  **Low risk:** 0

## 7. Validation Recommendation

**Recommended Strategy:** `within_period_latest_available_holdout`

- **Confidence:** high
- **Reason:** Prediction period overlaps training AND groups have partial overlap. Use within-period latest-available holdout stratified by group.
- **Fit rule:** Train on all but the latest N observations per group per period.
- **Validation rule:** Validate on the latest per-group observations.
- **Avoid:** time_holdout, kfold

## 8. Feature Engineering Recommendations

| Column | Action | Reason |
|--------|--------|--------|
| timestamp | EXTRACT_DATETIME_FEATURES | Extract hour, dayofweek, month, year, etc. |
| hour | EXTRACT_DATETIME_FEATURES | Extract hour, dayofweek, month, year, etc. |
| dayofweek | EXTRACT_DATETIME_FEATURES | Extract hour, dayofweek, month, year, etc. |
| month | DROP | Near-constant — no signal |
