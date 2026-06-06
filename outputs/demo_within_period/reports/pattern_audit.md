# Train-Test Pattern Audit

_Generated: 2026-06-07 07:26:48_

---

## 1. Detected Train/Prediction Pattern

**Pattern: `Within-Period + Group`**

Evidence from train/prediction comparison:
- timestamp: predict overlaps or precedes train max → within-period pattern

**Dataset shape:** train 855 rows × 6 cols | predict 495 rows × 4 cols

Train-only columns (absent from predict): `demand_so_far`, `target`

Numeric distribution shift: **0/1** columns shifted (KS p < 0.05)

## 2. Feature Availability at Prediction Time

| Category | Count |
|----------|-------|
| Available at prediction | 4 |
| Train-only (must exclude) | 1 |
| High leakage risk (exclude) | 0 |
| Needs verification | 0 |

**Columns to exclude (unavailable or high leakage risk):**
- `demand_so_far` [train_only_feature]

## 3. Why Generic Validation Is Risky Here

Random k-fold leaks future observations and ignores group structure. Time holdout ignores within-period overlap. Group k-fold ignores temporal ordering.

**Do not use:** `time_holdout`, `kfold`

## 4. Recommended Validation Strategy

**Strategy: `within_period_latest_available_holdout`** (Confidence: high)

_Prediction period overlaps training AND groups have partial overlap. Use within-period latest-available holdout stratified by group._

**Fit rule:** Train on all but the latest N observations per group per period.

**Validation rule:** Validate on the latest per-group observations.

## 5. Leakage Audit

| Severity | Count |
|----------|-------|
| High     | 1 |
| Medium   | 0 |
| Low      | 0 |

**High-risk columns — exclude from features:**
- `demand_so_far` (train_only_feature): Column is present in training data but absent in prediction data. If used as a feature, the model cannot generate predictions.
