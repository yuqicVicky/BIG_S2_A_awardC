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

Run before the auditor. The goal is to understand **what every column means in the real world**, not just its dtype. This understanding shapes which imputation strategy is appropriate and will be published verbatim in the final report.

```python
col_preview = pd.DataFrame({
    "dtype":       df_train.dtypes,
    "n_unique":    df_train.nunique(),
    "sample":      [df_train[c].dropna().iloc[:3].tolist()
                    if df_train[c].notna().any() else [] for c in df_train.columns],
    "min":         df_train.select_dtypes("number").min().reindex(df_train.columns),
    "max":         df_train.select_dtypes("number").max().reindex(df_train.columns),
    "missing_pct": df_train.isna().mean().round(3),
}).to_string()
print(col_preview)
```

#### A — Semantic interpretation of every column

For **each column**, produce a one-line plain-language interpretation. Use column name, dtype, sample values, and value range together — do not match against any fixed list of abbreviations. Reason from what you observe:

- Expand abbreviations using the sample values (e.g. `"06"` or `"CA"` in a column called `st_cd` suggests a state code; `34.05` in a column with `lat` in the name suggests latitude)
- Consider the dataset context the user described, if any
- If a column is ambiguous, say so and list the two most plausible interpretations

Produce a **semantic table** that will be included in the final report:

| Column | Dtype | Sample Values | Inferred Meaning | Domain Tag | Notes / Ambiguity |
|--------|-------|---------------|-----------------|------------|-------------------|
| ...    | ...   | ...           | ...             | ...        | ...               |

**Domain Tag** is a free-form label. Choose from examples like `geographic_identifier`, `administrative_code`, `weather_metric`, `financial_indicator`, `health_outcome`, `demographic_rate`, `binary_flag`, `free_text`, `id_or_key`, `datetime`, `model_score`, or invent a label that fits. Do not hard-code which column names belong to which domain — always reason from the data.

This semantic table is reproduced in the Step 4 report and in `outputs/reports/missing_data_report.md`.

#### B — Identify grouping columns (conditional)

From the semantic table, look for columns tagged `geographic_identifier`, `administrative_code`, or any other natural grouping unit (e.g. a product category, an institution type, a time period). Do **not** assume specific column name patterns — identify them by meaning.

If a candidate grouping column exists, record it as `group_col`. When multiple candidates exist, prefer the coarser grouping (e.g., region over city) unless the missing column itself is granular.

For any numeric column whose real-world meaning suggests strong group-level variation (e.g. a metric that is known to differ substantially across geographic or administrative units), check whether the between-group variance condition is met:

```python
group_medians     = df_train.groupby(group_col)[col].median()
between_group_std = group_medians.std()
total_std         = df_train[col].std()
grouping_useful   = (total_std > 0) and (between_group_std / total_std >= 0.15)
```

For a column to receive `groupwise_numeric_median_plus_indicator`, **all three conditions** must hold:

1. The column's real-world meaning suggests group-level variation (from semantic analysis, not name patterns).
2. A `group_col` exists in `df_train`.
3. Between-group variance is meaningful (between-group std ≥15% of total std, as above).

If condition 3 is **not** met, fall back to `numeric_median_plus_indicator` and record `evidence.geo_grouping_skipped_reason` in the plan.

If `group_col` is very high-cardinality (many unseen values in predict set), use the next coarser available grouping column.

**Always announce the decision** to the user before Step 4:
> *"For `[col]` ([inferred meaning]): group condition met (between-group std = X% of total) — using per-`[group_col]` median."*  
> or  
> *"For `[col]` ([inferred meaning]): group condition not met (between-group std = X% of total, threshold 15%) — falling back to global median + indicator."*

#### C — Structural relationships

Based on the semantic interpretation of each column, look for pairs where one column's real-world meaning implies a value constraint in another:

- A measurement column that can only exist when a parent entity is present (e.g. a numeric area column that must be 0 if no corresponding facility exists)
- A binary presence/absence flag that is 0 in rows where a companion column is NaN, suggesting the NaN encodes real-world absence rather than an error

Record any such pairs for Step 2.

#### D — Type guard

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
    # ID columns: semantic tag is id_or_key, or name pattern matches, or all-unique values
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
> **Semantic summary:** [N] columns interpreted. Grouping column identified: `[group_col]` ([inferred meaning]). [M] columns will use group-wise imputation: [list]. Type-skipped: [list]. Structural pairs: [list].
>
> **Column meanings:** [paste the full semantic table]

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

| Column | Inferred Meaning | Missing Rate | Severity | Mechanism | Strategy | Group Col | Add Indicator | MI Upgrade? |
|--------|-----------------|-------------|----------|-----------|----------|-----------|--------------|-------------|
| ...    | ...             | ...         | ...      | ...       | ...      | ...       | ...          | ...         |

`Inferred Meaning` comes from the semantic table built in Step 1.5 — reproduce it here verbatim so the user can verify the interpretation before acting on the plan. `Group Col` is non-empty only when `grouping_applied: true` in `evidence`; show the column name so the user can verify the grouping makes sense.

### Step 4 — Present the plan in plain language

Begin with the full semantic table from Step 1.5 so the user can verify the column interpretations.

For each column with missing values:
> `[col]` (**[inferred meaning]**) — [severity] missingness ([rate]%). Mechanism clue: [mechanism_label]. Recommended: `[strategy]`[, with a missing indicator][, grouped by `[group_col]`].

Follow the strategy selection rules in `references/strategies.md`. In particular:

- **Statistical methods** (median, mode, missing-token) are appropriate for MCAR-compatible or lightly MAR columns with low-to-moderate missing rates and few predictive features.
- **ML-based methods** are appropriate when missingness is MAR-heavy, the missing rate is moderate-to-high, and other features carry predictive signal for the missing column. Prefer ML imputation over statistical imputation when:
  - Missing rate ≥10% **and** `MAR-like evidence` **and** ≥5 non-missing numeric predictors → consider `knn_imputation`
  - Missing rate ≥25% **and** `MAR-like evidence` or `target-associated missingness` → consider `random_forest_imputation`
  - Many features with non-linear interactions → consider `gradient_boosting_imputation`
  - Statistical inference context → `iterative_imputer_ml` (BayesianRidge estimator)

If any column has `mi_upgrade_recommended: true`:
> **Full MI recommended for:** `[col1]`, `[col2]` — missing rate or covariate correlation exceeds Collins et al. (2001) thresholds. Median imputation is fine for simple ML feature engineering, but upgrade to MICE (`IterativeImputer` with `BayesianRidge` or `RandomForestRegressor`) if this data will be used for **statistical inference or hypothesis testing** (van Buuren FIMD Ch5).

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
| `outputs/logs/mice_pooling.json` | Multiple-imputation Rubin pooling: pooled mean, naive vs MI standard error, FMI per column (inference) |
| `outputs/logs/mnar_sensitivity.json` | MNAR delta-adjustment tipping points + fragile-column list |
| `outputs/reports/missing_data_report.md` | Human-readable narrative summary |
| `outputs/reports/missing_data_report.pdf` | Methods-appendix PDF (attach to a paper) |
| `outputs/figures/missingness_bar.png` | Bar chart of missing rates by column |
| `outputs/figures/pattern_matrix.png` | Missingness pattern matrix |
| `outputs/figures/target_signal.png` | Target mean for missing vs present rows |
| `outputs/figures/decision_flow.png` | CONSORT-style imputation decision flow diagram |
| `outputs/figures/mnar_tipping_point.png` | Delta-adjustment sensitivity trajectories |
| `outputs/reproduce_imputation.py` | Standalone, self-verifying reproduction script |
| `outputs/source_data.csv` | Frozen copy of the input, for the reproduction script |
| `outputs/train_imputed.csv` | Imputed training data |
| `outputs/predict_imputed.csv` | Imputed predict/test data (if `df_predict` was provided) |

**Inference-grade analysis (automatic in `save_outputs`).** Single imputation is fine
for ML feature engineering but understates variance for inference. The auditor therefore
also runs full **MICE** (`IterativeImputer` + `BayesianRidge`, `sample_posterior`) and
pools per-column estimates with **Rubin's rules** (`mice_pooling.json`), and runs an
**MNAR delta-adjustment sensitivity analysis** (`mnar_sensitivity.json`) reporting the
*tipping point* — how large an MNAR departure would overturn a MAR-based conclusion.
Both are leakage-safe (fit on train only) and best-effort (never block the core report).

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

- **Column semantic pre-scan (Step 1.5)** reads names, dtypes, and sample values before the auditor and reasons from observed data — no hardcoded abbreviation lists. Produces a semantic table reproduced in the report. Grouping is a conditional outcome of semantic analysis, not an automatic pattern-match.
- **Type guard (Step 1.5 D)** removes target, ID, and datetime columns from the plan before execution.
- **Statistical + ML strategy ladder** — statistical methods (median, mode) are the baseline; ML-based methods (KNN, Random Forest, Gradient Boosting, IterativeImputer) are recommended when missing rate, feature correlation, or inference requirements justify the added complexity.
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

1. **For `knn_imputation` or `random_forest_imputation` columns:** "Shall I implement the ML imputer for `[col]` using `KNNImputer` / `IterativeImputer(RandomForestRegressor())`?"
2. **For `model_based_imputation_optional` columns:** "Shall I implement full MICE for `[col]` using `IterativeImputer`?"
3. **For inference use cases:** "Shall I upgrade to `iterative_imputer_ml` (BayesianRidge) for all columns with `mi_upgrade_recommended: true`?"
4. **For `drop_column` columns:** "I dropped `[col]` — want me to check if it can be recovered using external data or proxy features?"
5. **General next step:** "Ready to move to feature engineering and EDA?"
