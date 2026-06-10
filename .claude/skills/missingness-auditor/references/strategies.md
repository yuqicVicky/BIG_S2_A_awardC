# Imputation Strategy Reference

## Strategy table

| Strategy | When used | MICE equivalent |
|----------|-----------|-----------------|
| `no_imputation_needed` | Column is fully observed | — |
| `numeric_median` | MCAR-compatible, <10% missing, numeric | PMM |
| `numeric_median_plus_indicator` | MAR-like / target-associated / ≥10% missing, numeric | PMM + indicator |
| `groupwise_numeric_median_plus_indicator` | Group-dependent numeric; per-group median with global fallback | MICE with group predictor |
| `categorical_missing_token` | Categorical, <10% missing | logreg / polyreg |
| `categorical_missing_token_plus_indicator` | Categorical, ≥10% missing or high-cardinality | logreg / polyreg + indicator |
| `structural_zero_plus_indicator` | Structural absence — numeric companion is 0 when categorical NA | 0-fill + indicator |
| `structural_none_token_plus_indicator` | Structural absence — categorical NA encodes real-world absence | NONE token + indicator |
| `drop_column` | >80% missing | drop column |
| `model_based_imputation_optional` | Complex MAR, moderate missingness — falls back to median | MICE (PMM / logreg / polyreg) |

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
