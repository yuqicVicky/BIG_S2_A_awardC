# Train-Test Pattern Audit

_Generated: 2026-06-07 07:27:00_

---

## 1. Detected Train/Prediction Pattern

**Pattern: `Group Split (Unseen Groups)`**

Evidence from train/prediction comparison:
- store_id: low group overlap (0.00) → group split

**Dataset shape:** train 480 rows × 4 cols | predict 240 rows × 3 cols

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

Random k-fold will put the same group in both train and validation, leaking group-level signal and inflating scores.

**Do not use:** `kfold`, `random_holdout`

## 4. Recommended Validation Strategy

**Strategy: `group_split`** (Confidence: high)

_Prediction groups are largely unseen at training time — group leakage must be prevented._

**Fit rule:** Train on a subset of groups; hold out a disjoint set of groups for validation.

**Validation rule:** Ensure no group appears in both train and validation splits.

## 5. Leakage Audit

| Severity | Count |
|----------|-------|
| High     | 0 |
| Medium   | 0 |
| Low      | 0 |
