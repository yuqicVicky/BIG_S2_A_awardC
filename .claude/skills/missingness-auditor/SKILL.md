---
name: missingness-audit-planner
description: Use this skill whenever the user mentions missing data, null values, NaN errors, or needs help cleaning a dataset before modeling. It covers phrases like "my data has nulls", "help me clean this CSV", "should I drop this column", "my model crashes on NaN", "what do I impute", "how do I handle missing values", or any request to prepare tabular data for training. It applies to both file-based workflows (CSV/parquet inputs) and Python code guidance (pandas DataFrames in-session). It diagnoses missingness severity, detects mechanism clues (MCAR / MAR-like / MNAR), finds structural absence patterns, recommends a leakage-safe imputation strategy per column, and can execute the imputation. Trigger this skill even when the user's phrasing is vague — if missing data is plausible, consult it.
---

# Missingness Audit & Imputation Planner

## Purpose

A reusable pre-imputation skill for autonomous data-science agents.  It
diagnoses missing data — profiling severity, identifying mechanism clues
(MCAR / MAR-like / MNAR), detecting structural absence patterns, and
recommending a column-specific imputation strategy with a leakage-safe
protocol.

## Workflow

Follow these steps in order every time this skill triggers.

**Step 1 — Identify what the user has**

- If the user provides a file path (CSV / parquet): load it with `pd.read_csv` / `pd.read_parquet`. Ask for the target column name if not given.
- If the user pastes inline data or describes a DataFrame already in session: use that DataFrame directly.
- If the user has a code question only (e.g. "how should I impute X?"): skip Steps 2–3, go straight to presenting the strategy table from the Imputation strategies section and the Execution template. Adapt the template to their column names.

**Step 1.5 — Column Semantic Analysis (run before the auditor)**

Before running the auditor, read all column names and infer their domain meaning. This shapes which imputation strategy is appropriate for each column.

```python
# Quick profile: names, dtypes, first few non-null values
col_preview = pd.DataFrame({
    "dtype": df.dtypes,
    "n_unique": df.nunique(),
    "sample": [df[c].dropna().iloc[:3].tolist() if df[c].notna().any() else [] for c in df.columns],
    "missing_pct": df.isna().mean().round(3),
}).to_string()
print(col_preview)
```

Using the column names, dtypes, unique counts, and sample values, reason through the following checklist before proceeding:

**A — Identify geographic / location columns**

Look for columns whose names or values indicate a geographic unit:

| Name patterns | Examples |
|---------------|---------|
| `state`, `state_code`, `fips`, `region` | `"CA"`, `"California"`, `"06"` |
| `county`, `district`, `province`, `prefecture` | `"Los Angeles"` |
| `city`, `municipality`, `town` | `"Denver"` |
| `zip`, `zip_code`, `postal_code` | `"80203"` |
| `lat` / `lon`, `latitude` / `longitude` | `34.05`, `-118.24` |
| `metro_area`, `cbsa`, `msa` | `"LA Metro"` |

If any such column exists, record it as `geo_col` (prefer coarser units — state > county > city — unless the column with missing values is itself very granular).

**B — Identify domain-sensitive numeric columns**

Cross-reference missing numeric columns against these domain families:

| Domain | Typical column name patterns | Why geography matters |
|--------|-----------------------------|-----------------------|
| **Weather / climate** | `temp*`, `tmax`, `tmin`, `tavg`, `precip*`, `rainfall`, `snowfall`, `humidity`, `wind_speed`, `solar_*`, `evapotranspiration` | Values cluster strongly by state/region; state median outperforms global median |
| **Air quality / environment** | `pm25`, `aqi`, `ozone`, `no2`, `co2`, `pollution_*` | Regional regulatory and geographic patterns |
| **Agriculture** | `yield_*`, `crop_*`, `soil_*`, `irrigation_*` | Climate zones drive values |
| **Socioeconomic** | `income`, `poverty_rate`, `unemployment`, `gdp_per_capita`, `median_rent` | State and regional economic variation is large |
| **Health / demographic** | `mortality_*`, `obesity_rate`, `vaccination_rate`, `life_expectancy` | State policy and demographics vary significantly |

**C — Override strategy for geo-grouped columns**

For every column that is (i) missing values, (ii) belongs to a domain family above, AND (iii) a `geo_col` exists in the dataset:

- Set strategy override: `groupwise_numeric_median_plus_indicator` with `group_col = geo_col`
- Rationale: imputing a state's temperature from a national median discards known geographic signal; per-state median is a strictly better prior
- Note this override in plain language to the user before Step 4

If `geo_col` is very high-cardinality (e.g. zip code with many unseen values in predict set), fall back to county → state → region, or use the next coarser column available.

**D — Infer structural relationships**

Look for column pairs where one column's meaning implies a value in another:

- A numeric `_area`, `_sqft`, `_count`, or `_rate` column often implies 0 when its categorical parent is absent (e.g. `garage_type = NaN → garage_area = 0`)
- A `_flag`, `_has_*`, or binary column that equals 0 for rows where a companion column is NaN suggests structural missingness

Record any such pairs for Step 2.

**Report to user (inline, before the auditor runs):**

> **Column semantic summary:** Found [N] geographic column(s): `[geo_col]`. Found [M] domain-sensitive column(s) with missing values that will use geographic group imputation: `[col_a]` (weather), `[col_b]` (socioeconomic). [Any structural pairs noted.]

**Step 2 — Run the auditor**

```python
import sys
sys.path.insert(0, "src")   # adjust if installed as a package
from missingness_auditor import MissingnessAuditor

auditor = MissingnessAuditor(df, predict_df=predict_df, target_col=target_col)
results = auditor.run()           # pure computation, no disk writes
auditor.save_outputs(results, "outputs/")   # writes JSON, figures, markdown
```

If the user has a separate test/predict file, pass it as `predict_df`. If no target column is known, omit `target_col` — mechanism detection will skip target correlation.

**Step 3 — Read and summarise outputs**

Read the three JSON files in this order:

1. `outputs/logs/missingness_profile.json` — get `columns_with_missing` and per-column `severity` and `missing_rate`
2. `outputs/logs/missingness_mechanism_audit.json` — get per-column `mechanism_label` and `target_signal`
3. `outputs/logs/imputation_plan.json` — get per-column `strategy` and `add_missing_indicator`

Surface to the user **only the columns that have missing values**, sorted by `missing_rate` descending, limited to 10 rows:

| Column | Missing Rate | Severity | Mechanism | Strategy | Group Col | Add Indicator | MI Upgrade? |
|--------|-------------|----------|-----------|----------|-----------|--------------|-------------|
| ...    | ...         | ...      | ...       | ...      | ...       | ...          | ...         |

`Group Col` is populated when Step 1.5 identified a geographic column and overrode the strategy to `groupwise_numeric_median_plus_indicator`. Show the geo column name (e.g. `state`) so the user can verify the grouping makes sense.

The `MI Upgrade?` column comes from `imputation_plan.json` → `columns[col].mi_upgrade_recommended`. If `true`, flag it clearly — this column warrants full MICE for inference tasks.

**Step 4 — Present the plan in plain language**

For each column with missing values, write one sentence:

> `[col]` — [severity] missingness ([rate]%). Mechanism clue: [mechanism_label]. Recommended: `[strategy]`[, with a missing indicator added as a binary feature].

Example:
> `income` — moderate missingness (32%). Mechanism clue: MAR-like evidence (correlated with `education_level`). Recommended: `numeric_median_plus_indicator`, with a missing indicator added as a binary feature.

> `tmax` — low missingness (8%). Mechanism clue: MCAR-compatible. Recommended: `groupwise_numeric_median_plus_indicator` grouped by `state` — temperature varies strongly by state, so per-state median is a better prior than the national median. Missing indicator added.

If any column has `mi_upgrade_recommended: true`, add a separate warning block:

> **Full MI recommended for:** `[col1]`, `[col2]` — missing rate or covariate correlation exceeds Collins et al. (2001) thresholds (>25% missing or correlation >0.4 with missingness indicator). Median imputation is acceptable for ML feature engineering, but upgrade to MICE (PMM for numeric, logreg/polyreg for categorical) if this data will be used for statistical inference or hypothesis testing (van Buuren FIMD Ch5).

**Step 5 — Ask for confirmation**

Before writing any files or executing imputation, say:

> "Here is the imputation plan for [N] columns. Does this look right? If so, I'll execute it now and write the cleaned files. Reply **yes** to proceed, or tell me which columns you'd like to handle differently."

**Step 6 — Execute and validate**

Once the user confirms, run the Execution template below. Then immediately run the Validation checklist. Report the validation summary to the user.

## Execution template

```python
import json
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer

# ── Load plan ────────────────────────────────────────────────────────────────
with open("outputs/logs/imputation_plan.json") as f:
    plan = json.load(f)["columns"]

# ── Helpers ──────────────────────────────────────────────────────────────────
def _add_indicator(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Add binary indicator BEFORE imputing so the flag captures true nulls."""
    df[f"{col}_was_missing"] = df[col].isna().astype(int)
    return df

# ── Apply per-column strategy ─────────────────────────────────────────────────
# All statistics (median, mode, imputer parameters) are fitted on df_train only,
# then applied to df_predict without re-fitting — this prevents leakage.

drop_cols = []

for col, entry in plan.items():
    strategy = entry["strategy"]
    add_ind  = entry["add_missing_indicator"]

    if strategy == "no_imputation_needed":
        continue

    elif strategy == "drop_column":
        drop_cols.append(col)

    elif strategy == "numeric_median":
        median = df_train[col].median()                      # fit on train
        df_train[col] = df_train[col].fillna(median)
        if df_predict is not None:
            df_predict[col] = df_predict[col].fillna(median) # apply same value

    elif strategy == "numeric_median_plus_indicator":
        if add_ind:
            df_train  = _add_indicator(df_train, col)
            if df_predict is not None:
                df_predict = _add_indicator(df_predict, col)
        median = df_train[col].median()                      # fit on train
        df_train[col] = df_train[col].fillna(median)
        if df_predict is not None:
            df_predict[col] = df_predict[col].fillna(median)

    elif strategy in ("categorical_missing_token", "categorical_missing_token_plus_indicator"):
        if add_ind:
            df_train  = _add_indicator(df_train, col)
            if df_predict is not None:
                df_predict = _add_indicator(df_predict, col)
        df_train[col] = df_train[col].fillna("MISSING")
        if df_predict is not None:
            df_predict[col] = df_predict[col].fillna("MISSING")

    elif strategy == "groupwise_numeric_median_plus_indicator":
        # Group-dependent missingness: fill with per-group median, fit on train only.
        # group_col comes from imputation_plan.json → columns[col].group_col
        group_col = entry.get("group_col")
        if add_ind:
            df_train  = _add_indicator(df_train, col)
            if df_predict is not None:
                df_predict = _add_indicator(df_predict, col)
        global_median = df_train[col].median()               # fit on train
        if group_col and group_col in df_train.columns:
            group_map = df_train.groupby(group_col)[col].median().to_dict()
            df_train[col] = df_train[col].fillna(df_train[group_col].map(group_map))
            if df_predict is not None and group_col in df_predict.columns:
                df_predict[col] = df_predict[col].fillna(df_predict[group_col].map(group_map))
        df_train[col] = df_train[col].fillna(global_median)   # fallback for unknown groups
        if df_predict is not None:
            df_predict[col] = df_predict[col].fillna(global_median)

    elif strategy == "structural_zero_plus_indicator":
        # Structural absence (Pattern 2): categorical NA encodes absence → numeric is 0.
        # Example: no garage type → garage_area = 0.
        if add_ind:
            df_train  = _add_indicator(df_train, col)
            if df_predict is not None:
                df_predict = _add_indicator(df_predict, col)
        df_train[col] = df_train[col].fillna(0)
        if df_predict is not None:
            df_predict[col] = df_predict[col].fillna(0)

    elif strategy == "structural_none_token_plus_indicator":
        # Structural absence (categorical side): NA encodes "no facility/item".
        if add_ind:
            df_train  = _add_indicator(df_train, col)
            if df_predict is not None:
                df_predict = _add_indicator(df_predict, col)
        df_train[col] = df_train[col].fillna("NONE")
        if df_predict is not None:
            df_predict[col] = df_predict[col].fillna("NONE")

    elif strategy == "structural_none_or_zero":
        # Legacy alias — dispatches by dtype. Prefer the explicit strategies above.
        if add_ind:
            df_train  = _add_indicator(df_train, col)
            if df_predict is not None:
                df_predict = _add_indicator(df_predict, col)
        fill_val = 0 if pd.api.types.is_numeric_dtype(df_train[col]) else "NONE"
        df_train[col] = df_train[col].fillna(fill_val)
        if df_predict is not None:
            df_predict[col] = df_predict[col].fillna(fill_val)

    elif strategy == "model_based_imputation_optional":
        # IterativeImputer (MICE-like). Fit on train only.
        # For tuning guidance see: references/strategies.md → "model_based_imputation"
        if add_ind:
            df_train  = _add_indicator(df_train, col)
            if df_predict is not None:
                df_predict = _add_indicator(df_predict, col)
        num_cols_for_mice = df_train.select_dtypes(include="number").columns.tolist()
        imp = IterativeImputer(max_iter=10, random_state=0)
        df_train[num_cols_for_mice]  = imp.fit_transform(df_train[num_cols_for_mice])   # fit on train
        if df_predict is not None:
            df_predict[num_cols_for_mice] = imp.transform(df_predict[num_cols_for_mice]) # transform only

# ── Drop columns flagged for removal ─────────────────────────────────────────
if drop_cols:
    df_train  = df_train.drop(columns=drop_cols, errors="ignore")
    if df_predict is not None:
        df_predict = df_predict.drop(columns=drop_cols, errors="ignore")

# ── Write outputs ─────────────────────────────────────────────────────────────
df_train.to_csv("outputs/train_imputed.csv", index=False)
if df_predict is not None:
    df_predict.to_csv("outputs/predict_imputed.csv", index=False)
```

## Validation

Run this checklist immediately after every imputation execution. Report any failures or warnings to the user.

```python
import warnings

print("\n── Validation ──────────────────────────────────────────────────────")

# [ ] 1. All targeted columns now have 0 missing values
imputed_cols = [c for c, e in plan.items() if e["strategy"] not in ("no_imputation_needed", "drop_column")]
for col in imputed_cols:
    if col in df_train.columns:
        n_miss = df_train[col].isna().sum()
        assert n_miss == 0, f"FAIL: {col} still has {n_miss} missing values in train"
print("[ ✓ ] No remaining nulls in imputed columns")

# [ ] 2. Indicator columns exist and are binary
indicator_cols = [f"{c}_was_missing" for c, e in plan.items() if e["add_missing_indicator"]]
for ind_col in indicator_cols:
    if ind_col in df_train.columns:
        unique_vals = set(df_train[ind_col].unique())
        assert unique_vals <= {0, 1}, f"FAIL: {ind_col} is not binary — values: {unique_vals}"
print("[ ✓ ] All missing indicator columns are binary (0/1)")

# [ ] 3. Mean shift check for numeric imputed columns
for col in imputed_cols:
    if col not in df_train_original.columns or col not in df_train.columns:
        continue
    if not pd.api.types.is_numeric_dtype(df_train[col]):
        continue
    old_mean = df_train_original[col].mean()
    new_mean = df_train[col].mean()
    if abs(old_mean) > 1e-9:
        shift = abs(new_mean - old_mean) / abs(old_mean)
        if shift > 0.10:
            warnings.warn(f"WARN: {col} mean shifted by {shift:.1%} after imputation "
                          f"(before={old_mean:.3f}, after={new_mean:.3f})")
print("[ ✓ ] Mean shift check complete (see warnings above if any)")

# [ ] 4. Correlation preservation — top-3 pairwise before/after
num_imputed = [c for c in imputed_cols if c in df_train.columns
               and pd.api.types.is_numeric_dtype(df_train[c])]
if len(num_imputed) >= 2:
    corr_before = df_train_original[num_imputed].corr()
    corr_after  = df_train[num_imputed].corr()
    diff = (corr_after - corr_before).abs()
    # get upper triangle only
    mask = np.triu(np.ones(diff.shape, dtype=bool), k=1)
    diffs_flat = diff.where(mask).stack().sort_values(ascending=False)
    top3 = diffs_flat.head(3)
    flagged = top3[top3 > 0.05]
    if not flagged.empty:
        print("[ ! ] Correlation changes > 0.05 after imputation:")
        for (c1, c2), val in flagged.items():
            print(f"      {c1} ↔ {c2}: Δ={val:.3f}")
    else:
        print("[ ✓ ] Top-3 pairwise correlations stable (all Δ ≤ 0.05)")

# [ ] 5. Leakage guard — imputer fitted on train only
# This is enforced structurally: all .fit() / .median() / .mode() calls above
# use df_train, and df_predict only receives .fillna() or .transform().
print("[ ✓ ] Imputer fitted on train only (structural guarantee in execution template)")

# [ ] 6. Final summary
train_shape   = df_train.shape
predict_shape = df_predict.shape if df_predict is not None else "N/A"
print(f"\nValidation complete. Imputed train shape: {train_shape}. "
      f"Imputed predict shape: {predict_shape}.")
```

> **Note:** `df_train_original` must be a copy taken before imputation: `df_train_original = df_train.copy()`. Insert this line at the top of your execution block.

## Quick start

```python
from missingness_auditor import MissingnessAuditor

auditor = MissingnessAuditor(df, target_col="target")
results = auditor.run()
auditor.save_outputs(results, "outputs/")

# Machine-readable answers for downstream agents
plan    = results["imputation_plan"]["columns"]
profile = results["missingness_profile"]["columns"]

for col, entry in plan.items():
    print(col, entry["strategy"], "indicator:", entry["add_missing_indicator"])
```

## CLI

Single dataset:
```bash
python -m missingness_auditor.cli \
  --data path/to/data.csv \
  --target target \
  --out outputs/
```

Train / predict split:
```bash
python -m missingness_auditor.cli \
  --train train.csv \
  --predict predict.csv \
  --target target \
  --out outputs/
```

## Outputs

| File | Contents |
|------|----------|
| `outputs/logs/missingness_profile.json` | Per-column missing counts, rates, severity |
| `outputs/logs/missingness_mechanism_audit.json` | MCAR / MAR-like / MNAR clue per column |
| `outputs/logs/structural_missingness_audit.json` | Structural absence pairs and column flags |
| `outputs/logs/imputation_plan.json` | Per-column strategy, indicator flag, fit scope |
| `outputs/logs/leakage_safe_imputation_check.json` | Leakage risk per column + global protocol |
| `outputs/reports/missing_data_report.md` | Human-readable narrative summary |
| `outputs/figures/missingness_bar.png` | Bar chart of missing rates by column |
| `outputs/figures/missingness_matrix.png` | Missingness pattern matrix (rows × columns) |
| `outputs/figures/missingness_target_signal.png` | Target mean for missing vs present rows |

## Key questions answered

| Question | Where to look |
|----------|---------------|
| Which columns have missing values? | `missingness_profile.json` → `columns_with_missing` |
| How severe is missingness? | `missingness_profile.json` → `columns[col].severity` |
| Is missingness associated with other features? | `missingness_mechanism_audit.json` → `columns[col].mechanism_label` |
| Is missingness predictive of the target? | `missingness_mechanism_audit.json` → `columns[col].target_signal` |
| Is missingness structural (absence of facility)? | `structural_missingness_audit.json` → `column_flags` |
| What imputation strategy should I use? | `imputation_plan.json` → `columns[col].strategy` |
| Should I add a missing indicator? | `imputation_plan.json` → `columns[col].add_missing_indicator` |
| How do I impute without leakage? | `leakage_safe_imputation_check.json` → `global_protocol` |

## Imputation strategies

| Strategy | When used | MICE equivalent (full MI) |
|----------|-----------|--------------------------|
| `no_imputation_needed` | Column is fully observed | — |
| `numeric_median` | MCAR-compatible, <10% missing, numeric | PMM (predictive mean matching) |
| `numeric_median_plus_indicator` | MAR-like / target-associated / ≥10% missing, numeric | PMM + indicator |
| `groupwise_numeric_median_plus_indicator` | Group-dependent missingness; fill with per-group median, fall back to global median | MICE with group predictor |
| `categorical_missing_token` | Categorical, <10% missing | logreg (binary) / polyreg (nominal) |
| `categorical_missing_token_plus_indicator` | Categorical, ≥10% missing or high-cardinality | logreg / polyreg + indicator |
| `structural_zero_plus_indicator` | Structural absence — numeric companion is 0 when categorical is NA (e.g. no garage → garage_area = 0) | 0-fill + indicator |
| `structural_none_token_plus_indicator` | Structural absence — categorical NA encodes real-world absence | NONE token + indicator |
| `drop_column` | >80% missing | drop column |
| `model_based_imputation_optional` | Complex MAR, moderate missingness — falls back to median; upgrade to MICE recommended | MICE (PMM / logreg / polyreg) |

**Deprecated / legacy strategies** (still handled by the imputer for backward compatibility, do not use in new plans):

| Strategy | Replaced by |
|----------|-------------|
| `structural_none_or_zero` | `structural_zero_plus_indicator` (numeric) or `structural_none_token_plus_indicator` (categorical) |
| `categorical_mode_plus_indicator` | `categorical_missing_token_plus_indicator` — mode fill creates spurious repeated values in high-cardinality columns (van Buuren FIMD Ch5) |

## Missingness mechanism language

Labels are always cautious statistical clues, not causal assignments (van Buuren FIMD Ch1). MCAR, MAR, and MNAR cannot be confirmed from observational data alone.

| Label | Meaning | Planner response |
|-------|---------|-----------------|
| `MCAR-compatible` | No significant correlation with features or target; consistent with MCAR | `numeric_median` (low rate) or `numeric_median_plus_indicator` |
| `MAR-like evidence` | Missingness correlates with ≥1 observed numeric feature | `numeric_median_plus_indicator`; consider MICE for inference |
| `group-dependent missingness` | Missing rate varies substantially across categorical groups (spread ≥ 20%) | `groupwise_numeric_median_plus_indicator` using the top group feature |
| `target-associated missingness` | Missingness correlates with the target variable | `numeric_median_plus_indicator`; `mi_upgrade_recommended = true` |
| `high-cardinality text/category missingness` | Missing values in a high-cardinality categorical column | `categorical_missing_token_plus_indicator` (mode fill unsafe) |
| `insufficient evidence` | Too few missing rows for reliable analysis (<20 rows) | `numeric_median_plus_indicator` / `categorical_missing_token` |
| `structural absence concern` | Set by structural detector, not mechanism auditor | `structural_zero_plus_indicator` or `structural_none_token_plus_indicator` |

**MAR robustness note** (from `mar_robustness_note` field): when `likely_robust_per_collins2001`, the missing rate is <25% and max covariate correlation <0.4 — omitting a lurking variable from the imputation model has negligible effect on regression estimates (Collins et al., 2001 via van Buuren FIMD Ch5).

## Integration in an LLM agent

```python
import json
from missingness_auditor import MissingnessAuditor

def agent_pre_imputation_audit(df, target_col):
    results = MissingnessAuditor(df, target_col=target_col).run()
    plan    = results["imputation_plan"]
    leakage = results["leakage_safe_check"]

    return json.dumps({
        "columns_to_impute": [
            col for col, e in plan["columns"].items()
            if e["strategy"] != "no_imputation_needed"
        ],
        "strategy_by_column": {
            col: e["strategy"]
            for col, e in plan["columns"].items()
        },
        "indicator_columns": plan["summary"]["columns_needing_missing_indicator"],
        "leakage_protocol": leakage["global_protocol"]["rule"],
    }, indent=2)
```

## Key design choices

- **Column semantic pre-scan** — Step 1.5 reads column names and sample values before running the auditor, infers domain families (weather, socioeconomic, agricultural, etc.), and detects geographic grouping columns. This overrides strategy to `groupwise_numeric_median_plus_indicator` for domain-sensitive columns when a geo column exists, because geographic structure is a stronger prior than a global median.
- **Dataset-agnostic core** — no domain-specific column names are hardcoded in `src/`; domain inference happens in the skill layer (Step 1.5) not in the auditor code.
- **Cautious mechanism language** — MCAR / MAR / MNAR are clues, not facts.
- **Zero seaborn** — all figures use matplotlib only.
- **Composable** — each sub-auditor can be used standalone without `MissingnessAuditor`.

## Scope and next steps

This skill covers **only**:
1. Missingness profile (counts, rates, severity, dtype)
2. Mechanism clue detection (MCAR / MAR-like / MNAR)
3. Structural missingness detection (categorical NA → numeric zero)
4. Column-specific imputation recommendation
5. Leakage-safe imputation protocol
6. Three diagnostic figures

## After this skill completes, recommend:

1. **For any column flagged as `model_based_imputation_optional`:** suggest — "Shall I implement MICE imputation for [col] using IterativeImputer?"
2. **To validate imputed data doesn't hurt model performance:** suggest — "Run a quick cross-validation comparison: model trained on original data vs. imputed data?"
3. **If any column was assigned `drop_column`:** suggest — "I dropped [col] — want me to check if it can be recovered using external data or proxy features?"
4. **General next step:** suggest — "Ready to move to feature engineering and EDA?"
