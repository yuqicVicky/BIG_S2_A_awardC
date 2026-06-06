# Train-Test Pattern Audit

_Generated: 2026-06-06 21:24:42_

---

## 1. Detected Train/Prediction Pattern

**Pattern: `Within-Period + Group`**

Evidence from train/prediction comparison:
- timestamp: predict starts after train ends → time-based split
- hour: predict overlaps or precedes train max → within-period pattern
- dayofweek: predict overlaps or precedes train max → within-period pattern

**Dataset shape:** train 1350 rows × 9 cols | predict 150 rows × 8 cols

Train-only columns (absent from predict): `target`

Numeric distribution shift: **0/1** columns shifted (KS p < 0.05)

## 2. Feature Availability at Prediction Time

| Category | Count |
|----------|-------|
| Available at prediction | 8 |
| Train-only (must exclude) | 0 |
| High leakage risk (exclude) | 0 |
| Needs verification | 0 |

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
| High     | 0 |
| Medium   | 0 |
| Low      | 0 |
