# Data Pattern & Validation Audit

_Generated: 2026-06-06 12:22:29_

---

## 1. Detected Data Pattern

**Pattern: `Within-Period + Group`**

Evidence from train/prediction comparison:
- timestamp: predict starts after train ends → time-based split
- hour: predict overlaps or precedes train max → within-period pattern
- dayofweek: predict overlaps or precedes train max → within-period pattern
- Train-only columns (absent from predict): `target`

**Dataset shape:** train 1350 rows × 9 cols | predict 150 rows × 8 cols

## 2. Why Generic Validation Is Risky Here

Random k-fold leaks future observations and ignores group structure. Time holdout ignores within-period overlap. Group k-fold ignores temporal ordering.

**Do not use:** `time_holdout`, `kfold`

## 3. Recommended Validation Strategy

**Strategy: `within_period_latest_available_holdout`** (Confidence: high)

_Prediction period overlaps training AND groups have partial overlap. Use within-period latest-available holdout stratified by group._

**Fit rule:** Train on all but the latest N observations per group per period.

**Validation rule:** Validate on the latest per-group observations.

## 4. Leakage Audit

| Severity | Count |
|----------|-------|
| High     | 0 |
| Medium   | 0 |
| Low      | 0 |

## 5. Target Pattern Highlights

**Numeric feature correlations with target:**

| Column | Pearson r | Signal |
|--------|-----------|--------|
| `numeric_feature` | +0.1264 | weak |

**Datetime feature target patterns detected:**
- `timestamp`: components=['hour', 'dayofweek', 'month', 'year', 'quarter'], interactions=['hour_x_dayofweek', 'hour_x_workingday', 'hour_x_weekend', 'month_x_year']
- `hour`: components=['hour', 'dayofweek', 'month', 'year', 'quarter'], interactions=['hour_x_dayofweek', 'hour_x_workingday', 'hour_x_weekend', 'month_x_year']
- `dayofweek`: components=['hour', 'dayofweek', 'month', 'year', 'quarter'], interactions=['hour_x_dayofweek', 'hour_x_workingday', 'hour_x_weekend', 'month_x_year']

**Categorical features with target variation:**
- `category` (3 categories)
- `text_note` (3 categories)

**Group/entity features with target variation:**
- `entity_id` (50 groups)

## 6. Feature Recommendations

### Features to Add

| Feature | Source | Type | Reason |
|---------|--------|------|--------|
| `timestamp__hour` | `timestamp` | datetime_derived | Hour of day captures intra-day demand cycles |
| `timestamp__dayofweek` | `timestamp` | datetime_derived | Day of week captures weekly seasonality |
| `timestamp__month` | `timestamp` | datetime_derived | Month captures annual seasonality |
| `timestamp__is_weekend` | `timestamp` | datetime_derived | Binary weekday/weekend flag captures structural behavior change |
| `timestamp__is_workingday` | `timestamp` | datetime_derived | Working-day flag isolates business-hours patterns |
| `hour__hour` | `hour` | datetime_derived | Hour of day captures intra-day demand cycles |
| `hour__dayofweek` | `hour` | datetime_derived | Day of week captures weekly seasonality |
| `hour__month` | `hour` | datetime_derived | Month captures annual seasonality |
| `hour__is_weekend` | `hour` | datetime_derived | Binary weekday/weekend flag captures structural behavior change |
| `hour__is_workingday` | `hour` | datetime_derived | Working-day flag isolates business-hours patterns |
| `dayofweek__hour` | `dayofweek` | datetime_derived | Hour of day captures intra-day demand cycles |
| `dayofweek__dayofweek` | `dayofweek` | datetime_derived | Day of week captures weekly seasonality |
| `dayofweek__month` | `dayofweek` | datetime_derived | Month captures annual seasonality |
| `dayofweek__is_weekend` | `dayofweek` | datetime_derived | Binary weekday/weekend flag captures structural behavior change |
| `dayofweek__is_workingday` | `dayofweek` | datetime_derived | Working-day flag isolates business-hours patterns |

### Features to Exclude

- `month`: constant — carries no generalizable signal

### Interactions to Try

- **hour × dayofweek** (datetime_interaction): Hour-of-day effect varies by day of week — key interaction for temporal models
- **hour × is_workingday** (datetime_interaction): Hour patterns differ significantly between working days and weekends
- **timestamp__hour × entity_id** (datetime_x_group): Temporal demand cycles typically differ by group/entity
- **lag_target_by_entity_id** (lag): Temporal pattern: recent target values per entity are typically the strongest predictor
- **rolling_mean_by_entity_id** (lag): Rolling average per entity captures trend and level shift over time
- **group_target_encode_entity_id** (group_encoding): Group split: target encoding with leave-one-group-out avoids group leakage

## 7. Feature Type Summary

| Type | Count |
|------|-------|
| `datetime_like` | 3 |
| `categorical_low_cardinality` | 2 |
| `group_entity_id` | 1 |
| `constant` | 1 |
| `numeric_continuous` | 1 |
| `target` | 1 |
