---
name: missingness-audit-planner
description: Diagnose and impute missing data before ML modeling. Triggers on: "nulls", "NaN errors", "clean CSV/parquet", "drop column?", "what to impute", "missing values", "prepare data for training". Handles file-based and in-session DataFrames. Profiles severity, detects MCAR/MAR-like/MNAR clues, finds structural absence, and recommends a leakage-safe per-column strategy. Hard guards: target column is never imputed; imputers are never fit on predict data. Trigger even on vague missing-data questions.
---

# Missingness Audit & Imputation Planner

## Hard Guards

These two rules are non-negotiable and enforced at every step:

1. **Target column is never imputed.** Before executing the plan, remove `target_col` from the column list. Log: `[GUARD] Skipping '{target_col}' (target column)`.
2. **Imputers fit on `df_train` only.** All `.fit()`, `.median()`, `.mode()`, `.groupby()` calls use `df_train`. `df_predict` only receives `.transform()` / `.fillna()` with values derived from `df_train`.

See `scripts/execute_imputation.py` for how both guards are enforced in code.

## Workflow

Follow these steps in order every time this skill triggers.

### Step 1 — Identify input

- **File path (CSV / parquet):** load as `df_train`. If the user also provides a predict/test file, load it as `df_predict` (default: `None`). Ask for `target_col` if not given.
- **Inline data or in-session DataFrame:** use it directly as `df_train`.
- **Code question only** (e.g. "how should I impute X?"): skip Steps 2–3, go straight to the strategy table in `references/strategies.md` and the Execution template. Adapt to their column names.

Variable convention used throughout: `df_train` (always required), `df_predict` (optional, may be `None`), `target_col` (str or `None`).

### Step 1.5 — Column Semantic Analysis

Run before the auditor. Read column names, dtypes, unique counts, and samples to shape the imputation strategy.

```python
col_preview = pd.DataFrame({
    "dtype":       df_train.dtypes,
    "n_unique":    df_train.nunique(),
    "sample":      [df_train[c].dropna().iloc[:3].tolist()
                    if df_train[c].notna().any() else [] for c in df_train.columns],
    "missing_pct": df_train.isna().mean().round(3),
}).to_string()
print(col_preview)
```

#### A — Identify geographic / location columns

Look for columns whose names or values indicate a geographic unit:

| Name patterns | Examples |
|---------------|---------|
| `state`, `state_code`, `fips`, `region` | `"CA"`, `"California"`, `"06"` |
| `county`, `district`, `province`, `prefecture` | `"Los Angeles"` |
| `city`, `municipality`, `town` | `"Denver"` |
| `zip`, `zip_code`, `postal_code` | `"80203"` |
| `lat`/`lon`, `latitude`/`longitude` | `34.05`, `-118.24` |
| `metro_area`, `cbsa`, `msa` | `"LA Metro"` |

If any such column exists, record it as `geo_col` (prefer coarser unit — state > county > city — unless missing column is itself granular).

#### B — Identify domain-sensitive numeric columns

| Domain | Typical column name patterns | Why geography matters |
|--------|-----------------------------|-----------------------|
| **Weather / climate** | `temp*`, `tmax`, `tmin`, `tavg`, `precip*`, `rainfall`, `snowfall`, `humidity`, `wind_speed` | Values cluster strongly by state/region |
| **Air quality** | `pm25`, `aqi`, `ozone`, `no2`, `co2`, `pollution_*` | Regional regulatory and geographic patterns |
| **Agriculture** | `yield_*`, `crop_*`, `soil_*`, `irrigation_*` | Climate zones drive values |
| **Socioeconomic** | `income`, `poverty_rate`, `unemployment`, `gdp_per_capita`, `median_rent` | State/regional variation is large |
| **Health / demographic** | `mortality_*`, `obesity_rate`, `vaccination_rate`, `life_expectancy` | State policy and demographics vary |

#### C — Geo-group imputation (CONDITIONAL — not an automatic override)

For a column to receive `groupwise_numeric_median_plus_indicator`, **all three conditions** must be met:

1. Column belongs to a domain family in Section B.
2. A `geo_col` exists in `df_train`.
3. Between-group variance is meaningful — the std of per-group medians is ≥15% of the column's total std:

```python
group_medians     = df_train.groupby(geo_col)[col].median()
between_group_std = group_medians.std()
total_std         = df_train[col].std()
geo_grouping_useful = (total_std > 0) and (between_group_std / total_std >= 0.15)
```

If condition 3 is **not** met, fall back to `numeric_median_plus_indicator` and record `evidence.geo_grouping_skipped_reason` in the plan.

If `geo_col` is very high-cardinality (e.g. zip code with many unseen values in predict set), use the next coarser column available.

**Always announce the decision** to the user before Step 4:
> *"For `[col]` (weather domain): geo-group condition met (between-group std = X% of total) — using per-`[geo_col]` median."*  
> or  
> *"For `[col]` (weather domain): geo-group condition not met (between-group std = X% of total, threshold 15%) — falling back to global median + indicator."*

#### D — Structural relationships

Look for column pairs where one column's meaning implies a value in another:

- A numeric `_area`, `_sqft`, `_count`, or `_rate` often implies 0 when its categorical parent is absent (e.g. `garage_type = NaN → garage_area = 0`)
- A `_flag`, `_has_*`, or binary column that is 0 for rows where a companion column is NaN suggests structural missingness

Record any such pairs for Step 2.

#### E — Type guard

Before finalising the plan, identify columns that must be excluded from imputation entirely:

```python
import re

TYPE_SKIP: set = set()

# Hard guard: target column
if target_col and target_col in df_train.columns:
    TYPE_SKIP.add(target_col)

for col in df_train.columns:
    if col == target_col:
        continue
    # ID columns: name pattern or all-unique values (surrogate key)
    if (re.search(r"(?i)(^id$|_id$|^id_)", col)
            or df_train[col].nunique() == len(df_train)):
        TYPE_SKIP.add(col)
    # Datetime columns — skip here; handle separately in time-series scenario
    elif pd.api.types.is_datetime64_any_dtype(df_train[col]):
        TYPE_SKIP.add(col)
```

Type notes (not skipped, but need correct strategy):

| Column type | Detection | Correct strategy |
|-------------|-----------|-----------------|
| **Boolean** | `dtype == bool` or unique values ⊆ `{0, 1}` | `categorical_missing_token` (safe for 0/1); never use median |
| **High-cardinality text** | `object` dtype and `nunique / len > 0.5` | `categorical_missing_token_plus_indicator` (mode fill unsafe) |
| **Low-cardinality categorical** | `object` / `category` dtype, `nunique ≤ 20` | `categorical_missing_token` or `categorical_missing_token_plus_indicator` |

Report to user before Step 2:
> **Semantic summary:** `[N]` geographic columns: `[geo_col]`. `[M]` columns will use geo-group imputation: `[list]`. Type-skipped: `[list]`. Structural pairs: `[list]`.

### Step 2 — Run the auditor

```python
import sys
sys.path.insert(0, "src")
from missingness_auditor import MissingnessAuditor

auditor = MissingnessAuditor(df_train, predict_df=df_predict, target_col=target_col)
results = auditor.run()
auditor.save_outputs(results, "outputs/")
```

If no `target_col` is known, omit it — mechanism detection will skip target correlation.

### Step 3 — Read and summarise outputs

Read three JSON files in order:

1. `outputs/logs/missingness_profile.json` → `columns_with_missing`, per-column `severity`, `missing_rate`
2. `outputs/logs/missingness_mechanism_audit.json` → per-column `mechanism_label`, `target_signal`
3. `outputs/logs/imputation_plan.json` → per-column `strategy`, `add_missing_indicator`, `group_col`, `mi_upgrade_recommended`

Schema reference: `references/imputation_plan_schema.md`.

Surface only columns with missing values, sorted by `missing_rate` descending, max 10 rows:

| Column | Missing Rate | Severity | Mechanism | Strategy | Group Col | Add Indicator | MI Upgrade? |
|--------|-------------|----------|-----------|----------|-----------|--------------|-------------|
| ...    | ...         | ...      | ...       | ...      | ...       | ...          | ...         |

`Group Col` is non-empty only when `geo_grouping_applied: true` in `evidence`. Show `geo_col` name so the user can verify the grouping makes sense.

### Step 4 — Present the plan in plain language

For each column with missing values:
> `[col]` — [severity] missingness ([rate]%). Mechanism clue: [mechanism_label]. Recommended: `[strategy]`[, with a missing indicator][, grouped by `[geo_col]`].

If any column has `mi_upgrade_recommended: true`:
> **Full MI recommended for:** `[col1]`, `[col2]` — missing rate or covariate correlation exceeds Collins et al. (2001) thresholds. Median imputation is fine for ML feature engineering, but upgrade to MICE (PMM / logreg / polyreg) if this data will be used for **statistical inference or hypothesis testing** (van Buuren FIMD Ch5).

### Step 5 — Ask for confirmation

> "Here is the imputation plan for [N] columns. Does this look right? If so, I'll execute it and write the cleaned files. Reply **yes** to proceed, or tell me which columns to handle differently."

### Step 6 — Execute and validate

Once the user confirms:
1. Run `scripts/execute_imputation.py` (paste or exec the template, setting `df_train`, `df_predict`, `target_col`).
2. Immediately run `scripts/validate_imputation.py`.
3. Report the validation summary (pass/fail/warnings) to the user.

## Special Scenarios

### Time series data

When the DataFrame has a `DatetimeIndex` or a sorted datetime column, **do not use median fill for numeric columns**. Use forward-fill then backward-fill as the primary strategy:

```python
# Detect time-ordered data
has_datetime_index = isinstance(df_train.index, pd.DatetimeIndex)
datetime_cols = df_train.select_dtypes(include="datetime").columns.tolist()

if has_datetime_index or datetime_cols:
    # Sort by time first
    sort_col = df_train.index if has_datetime_index else datetime_cols[0]
    df_train   = df_train.sort_index() if has_datetime_index else df_train.sort_values(sort_col)
    if df_predict is not None:
        df_predict = df_predict.sort_index() if has_datetime_index else df_predict.sort_values(sort_col)

    for col in numeric_cols_with_missing:
        if col in TYPE_SKIP:
            continue
        if add_ind:
            df_train = _add_indicator(df_train, col)
        df_train[col] = df_train[col].ffill().bfill()
        # For df_predict: if it is a future window, ffill from the end of df_train
        if df_predict is not None:
            last_known = df_train[col].iloc[-1]
            df_predict[col] = df_predict[col].ffill().fillna(last_known)
```

**Note:** carry the same `TYPE_SKIP` and hard guards. Never forward-fill the target column.

### Statistical inference

When the dataset will be used for **statistical inference, hypothesis testing, or generating confidence intervals** (not just ML prediction), median imputation introduces bias. Recommend MICE explicitly:

> "You mentioned this data will be used for statistical inference. Median imputation can bias standard errors and p-values. I recommend upgrading to MICE (`IterativeImputer` with `estimator=BayesianRidge()`) for all columns with >5% missingness, following van Buuren FIMD Ch5 (PMM for continuous, logreg for binary, polyreg for nominal)."

Trigger this recommendation when:
- User mentions "regression", "p-value", "confidence interval", "causal", "significance", OR
- ≥1 column has `mi_upgrade_recommended: true`, OR
- Overall missing rate across any column exceeds 25%

## Outputs

| File | Contents |
|------|----------|
| `outputs/logs/missingness_profile.json` | Per-column missing counts, rates, severity |
| `outputs/logs/missingness_mechanism_audit.json` | MCAR / MAR-like / MNAR clue per column |
| `outputs/logs/structural_missingness_audit.json` | Structural absence pairs and column flags |
| `outputs/logs/imputation_plan.json` | Per-column strategy, indicator flag, fit scope (schema: `references/imputation_plan_schema.md`) |
| `outputs/logs/leakage_safe_imputation_check.json` | Leakage risk per column + global protocol |
| `outputs/reports/missing_data_report.md` | Human-readable narrative summary |
| `outputs/figures/missingness_bar.png` | Bar chart of missing rates by column |
| `outputs/figures/missingness_matrix.png` | Missingness pattern matrix |
| `outputs/figures/missingness_target_signal.png` | Target mean for missing vs present rows |
| `outputs/train_imputed.csv` | Imputed training data |
| `outputs/predict_imputed.csv` | Imputed predict/test data (if `df_predict` was provided) |

## Key questions answered

| Question | Where to look |
|----------|---------------|
| Which columns have missing values? | `missingness_profile.json` → `columns_with_missing` |
| How severe is missingness? | `missingness_profile.json` → `columns[col].severity` |
| Is missingness associated with other features? | `missingness_mechanism_audit.json` → `columns[col].mechanism_label` |
| Is missingness predictive of the target? | `missingness_mechanism_audit.json` → `columns[col].target_signal` |
| Is missingness structural? | `structural_missingness_audit.json` → `column_flags` |
| What imputation strategy should I use? | `imputation_plan.json` → `columns[col].strategy` |
| Was geo-group imputation applied or skipped? | `imputation_plan.json` → `columns[col].evidence.geo_grouping_applied` |
| Should I add a missing indicator? | `imputation_plan.json` → `columns[col].add_missing_indicator` |
| How do I impute without leakage? | `leakage_safe_imputation_check.json` → `global_protocol` |

## Integration in an LLM agent

```python
import json
from missingness_auditor import MissingnessAuditor

def agent_pre_imputation_audit(df_train, target_col):
    results = MissingnessAuditor(df_train, target_col=target_col).run()
    plan    = results["imputation_plan"]
    leakage = results["leakage_safe_check"]

    return json.dumps({
        "columns_to_impute": [
            col for col, e in plan["columns"].items()
            if e["strategy"] != "no_imputation_needed"
        ],
        "strategy_by_column": {
            col: e["strategy"] for col, e in plan["columns"].items()
        },
        "indicator_columns":     plan["summary"]["columns_needing_missing_indicator"],
        "type_skipped_columns":  plan["summary"].get("type_skipped_columns", []),
        "target_col_excluded":   plan["summary"]["target_col_excluded"],
        "leakage_protocol":      leakage["global_protocol"]["rule"],
    }, indent=2)
```

## Key design choices

- **Column semantic pre-scan (Step 1.5)** reads names/samples before the auditor, infers domain families, and checks the between-group variance condition before assigning `groupwise_numeric_median_plus_indicator`. Geographic grouping is never an automatic override.
- **Type guard (Step 1.5 E)** removes target, ID, and datetime columns from the plan before execution.
- **Dataset-agnostic core** — no domain-specific column names hardcoded in `src/`; domain inference happens in the skill layer.
- **Cautious mechanism language** — MCAR / MAR / MNAR are clues, not facts (van Buuren FIMD Ch1).
- **Zero seaborn** — all figures use matplotlib only.
- **Composable** — each sub-auditor can run standalone without `MissingnessAuditor`.

## Scope

This skill covers:
1. Missingness profile (counts, rates, severity, dtype)
2. Mechanism clue detection (MCAR / MAR-like / MNAR)
3. Structural missingness detection
4. Column-specific imputation recommendation
5. Leakage-safe imputation execution
6. Three diagnostic figures

Strategy reference: `references/strategies.md`
Schema reference: `references/imputation_plan_schema.md`
Execution script: `scripts/execute_imputation.py`
Validation script: `scripts/validate_imputation.py`

## After this skill completes, recommend

1. **For `model_based_imputation_optional` columns:** "Shall I implement full MICE for `[col]` using `IterativeImputer`?"
2. **For inference use cases:** "Shall I upgrade to MICE for all columns with `mi_upgrade_recommended: true`?"
3. **For `drop_column` columns:** "I dropped `[col]` — want me to check if it can be recovered using external data or proxy features?"
4. **General next step:** "Ready to move to feature engineering and EDA?"
