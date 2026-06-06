# Train-Test Pattern Audit

_Generated: 2026-06-07 07:26:59_

---

## 1. Detected Train/Prediction Pattern

**Pattern: `i.i.d. Random`**

Evidence from train/prediction comparison:
- No strong temporal or group signal → assume iid random split

**Dataset shape:** train 800 rows × 4 cols | predict 200 rows × 3 cols

Train-only columns (absent from predict): `target`

Numeric distribution shift: **0/2** columns shifted (KS p < 0.05)

## 2. Feature Availability at Prediction Time

| Category | Count |
|----------|-------|
| Available at prediction | 3 |
| Train-only (must exclude) | 0 |
| High leakage risk (exclude) | 0 |
| Needs verification | 0 |

## 3. Why Generic Validation Is Risky Here

No structural violation was detected. Standard k-fold is appropriate, but verify that rows are truly independent before proceeding.

**Do not use:** `time_holdout`

## 4. Recommended Validation Strategy

**Strategy: `kfold`** (Confidence: medium)

_No strong temporal or group signal detected; iid assumption is reasonable._

**Fit rule:** Use standard K-Fold or stratified K-Fold cross-validation.

**Validation rule:** Randomly assign rows to folds.

## 5. Leakage Audit

| Severity | Count |
|----------|-------|
| High     | 0 |
| Medium   | 0 |
| Low      | 0 |
