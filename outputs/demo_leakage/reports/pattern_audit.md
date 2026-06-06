# Train-Test Pattern Audit

_Generated: 2026-06-07 07:27:14_

---

## 1. Detected Train/Prediction Pattern

**Pattern: `Strict Time Split (Forecasting)`**

Evidence from train/prediction comparison:
- timestamp: predict starts after train ends → time-based split

**Dataset shape:** train 500 rows × 5 cols | predict 100 rows × 2 cols

Train-only columns (absent from predict): `target`, `target_component_1`, `target_component_2`

Numeric distribution shift: **0/1** columns shifted (KS p < 0.05)

## 2. Feature Availability at Prediction Time

| Category | Count |
|----------|-------|
| Available at prediction | 2 |
| Train-only (must exclude) | 2 |
| High leakage risk (exclude) | 0 |
| Needs verification | 0 |

**Columns to exclude (unavailable or high leakage risk):**
- `target_component_1` [train_only_feature]
- `target_component_2` [train_only_feature]

## 3. Why Generic Validation Is Risky Here

Random k-fold or stratified k-fold will train on future data and validate on past data, producing optimistic scores that will not hold in production.

**Do not use:** `kfold`, `random_holdout`, `stratified_kfold`

## 4. Recommended Validation Strategy

**Strategy: `time_holdout`** (Confidence: high)

_Prediction data comes strictly after training data — temporal ordering must be respected._

**Fit rule:** Train on all data up to a cutoff date.

**Validation rule:** Validate on data after the cutoff (simulating the prediction window).

**Alternatives to consider:**
- `rolling_split`: Rolling window validation can better capture temporal model decay.

## 5. Leakage Audit

| Severity | Count |
|----------|-------|
| High     | 2 |
| Medium   | 0 |
| Low      | 0 |

**High-risk columns — exclude from features:**
- `target_component_1` (train_only_feature): Column is present in training data but absent in prediction data. If used as a feature, the model cannot generate predictions.
- `target_component_2` (train_only_feature): Column is present in training data but absent in prediction data. If used as a feature, the model cannot generate predictions.
