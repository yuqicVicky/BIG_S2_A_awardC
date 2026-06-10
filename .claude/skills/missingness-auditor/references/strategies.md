# Imputation Strategy Reference

## Strategy table

### Statistical methods

| Strategy | When used | sklearn / library |
|----------|-----------|-------------------|
| `no_imputation_needed` | Column is fully observed | — |
| `numeric_median` | MCAR-compatible, <10% missing, numeric, few correlated features | `SimpleImputer(strategy="median")` |
| `numeric_median_plus_indicator` | MAR-like / target-associated / ≥10% missing, numeric | `SimpleImputer` + `MissingIndicator` |
| `groupwise_numeric_median_plus_indicator` | Group-dependent numeric; per-group median with global fallback | `df.groupby(group_col)[col].transform("median")` |
| `categorical_missing_token` | Categorical, <10% missing | `SimpleImputer(strategy="constant", fill_value="__MISSING__")` |
| `categorical_missing_token_plus_indicator` | Categorical, ≥10% missing or high-cardinality | constant fill + `MissingIndicator` |
| `structural_zero_plus_indicator` | Structural absence — numeric companion is 0 when categorical NA | `fillna(0)` + indicator |
| `structural_none_token_plus_indicator` | Structural absence — categorical NA encodes real-world absence | `fillna("NONE")` + indicator |
| `drop_column` | >80% missing | drop column |
| `model_based_imputation_optional` | Complex MAR, moderate missingness — falls back to median if ML not requested | `IterativeImputer(BayesianRidge())` |

### ML-based methods

Use ML imputation when statistical methods are too coarse — i.e., when other features carry real predictive signal for the missing values and the missing rate is high enough to matter.

| Strategy | When used | sklearn / library |
|----------|-----------|-------------------|
| `knn_imputation` | MAR-like, 10–30% missing, ≥5 numeric predictors available, moderate dataset size | `KNNImputer(n_neighbors=5)` |
| `random_forest_imputation` | MAR-like or target-associated, ≥25% missing, many predictors, non-linear relationships expected | `IterativeImputer(RandomForestRegressor(n_estimators=100))` for numeric; `IterativeImputer(RandomForestClassifier())` for categorical |
| `gradient_boosting_imputation` | Complex MAR with strong feature interactions; large dataset where RF is slow | `IterativeImputer(HistGradientBoostingRegressor())` — natively handles NaN so can be used as internal estimator |
| `iterative_imputer_ml` | Statistical inference context — need unbiased estimates and valid standard errors | `IterativeImputer(BayesianRidge())` for continuous (PMM semantics); `IterativeImputer(LogisticRegression())` for binary; `IterativeImputer(RandomForestClassifier())` for multi-class |

**Decision upgrade rules — when to prefer ML over statistical:**

| Condition | Upgrade from | Upgrade to |
|-----------|-------------|-----------|
| missing_rate ≥10% **and** MAR-like **and** ≥5 numeric non-missing predictors | `numeric_median_plus_indicator` | `knn_imputation` |
| missing_rate ≥25% **and** (MAR-like or target-associated) | `numeric_median_plus_indicator` | `random_forest_imputation` |
| many features with strong non-linear interactions, large dataset | `random_forest_imputation` | `gradient_boosting_imputation` |
| statistical inference / hypothesis testing context | any statistical method | `iterative_imputer_ml` (BayesianRidge) |

**Leakage note for ML methods:** all ML imputers must be fitted on `df_train` only. For `KNNImputer` and `IterativeImputer`, call `.fit(df_train[feature_cols])` then `.transform(df_predict[feature_cols])`. Never fit on combined train+predict data.

**Deprecated / legacy** (backward-compatible only, do not use in new plans):

| Strategy | Replaced by |
|----------|-------------|
| `structural_none_or_zero` | `structural_zero_plus_indicator` or `structural_none_token_plus_indicator` |
| `categorical_mode_plus_indicator` | `categorical_missing_token_plus_indicator` — mode creates spurious repeated values in high-cardinality columns |

## Mechanism language

Labels are statistical clues, not causal assignments (van Buuren FIMD Ch1). MCAR/MAR/MNAR cannot be confirmed from observational data alone.

| Label | Meaning | Planner response |
|-------|---------|-----------------|
| `MCAR-compatible` | No significant correlation with features or target | `numeric_median` (low rate) or `numeric_median_plus_indicator` |
| `MAR-like evidence` | Missingness correlates with ≥1 observed numeric feature | `numeric_median_plus_indicator`; consider MICE for inference |
| `group-dependent missingness` | Missing rate varies substantially across categorical groups (spread ≥20%) | `groupwise_numeric_median_plus_indicator` if geo-group condition met |
| `target-associated missingness` | Missingness correlates with target | `numeric_median_plus_indicator`; `mi_upgrade_recommended = true` |
| `high-cardinality text/category missingness` | Missing in a high-cardinality categorical column | `categorical_missing_token_plus_indicator` |
| `insufficient evidence` | Too few missing rows (<20 rows) for reliable analysis | `numeric_median_plus_indicator` / `categorical_missing_token` |
| `structural absence concern` | Set by structural detector, not mechanism auditor | `structural_zero_plus_indicator` or `structural_none_token_plus_indicator` |

**MAR robustness note** (`mar_robustness_note` field): when `likely_robust_per_collins2001`, the missing rate is <25% and max covariate correlation <0.4 — omitting a lurking variable has negligible effect on regression estimates (Collins et al., 2001 via van Buuren FIMD Ch5).

## Full MI recommendation thresholds

Upgrade from median to MICE when `mi_upgrade_recommended: true`:
- Missing rate ≥25%, **or**
- Missingness indicator correlates >0.4 with any covariate, **or**
- `target-associated missingness` and the column will be used for statistical inference

Reference: Collins et al. (2001) via van Buuren *Flexible Imputation of Missing Data* (FIMD), Ch5.
