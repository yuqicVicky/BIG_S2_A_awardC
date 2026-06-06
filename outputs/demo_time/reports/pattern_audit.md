# Train-Test Pattern Audit

_Generated: 2026-06-07 07:27:13_

---

## 1. Detected Train/Prediction Pattern

**Pattern: `Strict Time Split (Forecasting)`**

Evidence from train/prediction comparison:
- date: predict starts after train ends → time-based split

**Dataset shape:** train 181 rows × 4 cols | predict 184 rows × 3 cols

Train-only columns (absent from predict): `target`

Numeric distribution shift: **0/1** columns shifted (KS p < 0.05)

## 2. Feature Availability at Prediction Time

| Category | Count |
|----------|-------|
| Available at prediction | 3 |
| Train-only (must exclude) | 0 |
| High leakage risk (exclude) | 0 |
| Needs verification | 0 |

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
| High     | 0 |
| Medium   | 0 |
| Low      | 0 |
