# Missingness Audit & Imputation Planner

A reusable, dataset-agnostic skill for autonomous data-science agents.  It
diagnoses missing data *before* imputation — profiling severity, identifying
mechanism clues, detecting structural absence patterns, and recommending a
column-specific strategy with a leakage-safe protocol.

---

## Why missingness diagnosis matters

Imputing without diagnosis leads to silent errors:

- **Over-imputing** smooth columns where the NaN itself is the signal.
- **Under-counting** missingness indicators that should be features.
- **Leaking test statistics** into training (fitting the median on the full dataset).
- **Misinterpreting structural zeros** as random gaps.

A 30-second audit surfaces all four issues and hands a concrete plan to the
modeling step.

---

## MCAR / MAR / MNAR — in practical terms

| Term | What it means | What to do |
|------|--------------|-----------|
| **MCAR** (Missing Completely At Random) | No pattern; equally likely to be missing for any row. A survey respondent skipped a random question. | Simple median/mode imputation is safe. No indicator needed for low rates. |
| **MAR** (Missing At Random) | Missingness depends on *other observed* variables. Income data missing more often for low-education respondents. | Impute and **add a binary missing indicator** — the pattern carries information. |
| **MNAR** (Missing Not At Random) | Missingness depends on the *unobserved* value itself or the target. A customer's credit score is missing because they defaulted. | The gap IS signal. Always add a missing indicator. Consider keeping NaN as a category. |

This tool uses cautious language:
- **"MCAR-compatible"** — no significant observed correlations; consistent with MCAR.
- **"MAR-like evidence"** — missingness statistically correlated with an observed feature.
- **"MNAR/structural concern"** — missingness correlated with the target variable.

These are observational clues, not causal mechanism assignments.

---

## Quick demo

```python
from missingness_auditor import MissingnessAuditor
import pandas as pd

df = pd.read_csv("my_data.csv")
auditor = MissingnessAuditor(df, target_col="outcome")
results = auditor.run(output_dir="outputs/")

# Machine-readable plan for downstream agents
plan = results["imputation_plan"]["columns"]
for col, entry in plan.items():
    print(f"{col}: {entry['strategy']}  indicator={entry['add_missing_indicator']}")
```

CLI:

```bash
python -m missingness_auditor.cli \
  --data my_data.csv \
  --target outcome \
  --out outputs/
```

---

## Output files

| File | Contents |
|------|----------|
| `outputs/logs/missingness_profile.json` | Per-column missing counts, rates, severity |
| `outputs/logs/missingness_mechanism_audit.json` | MCAR / MAR-like / MNAR clue per column |
| `outputs/logs/structural_missingness_audit.json` | Structural absence pairs |
| `outputs/logs/imputation_plan.json` | Per-column strategy, indicator flag, fit scope |
| `outputs/logs/leakage_safe_imputation_check.json` | Leakage risk per column + global protocol |
| `outputs/reports/missing_data_report.md` | Human-readable narrative |
| `outputs/figures/missingness_bar.png` | Bar chart of missing rates |
| `outputs/figures/missingness_matrix.png` | Row × column missingness matrix |
| `outputs/figures/missingness_target_signal.png` | Target signal by missingness |

---

## Example output — imputation_plan.json

From the four toy demo cases:

**Case 1 — MCAR (7% random missing)**
```json
{ "measurement_a": { "strategy": "numeric_median", "add_missing_indicator": false } }
```

**Case 2 — MAR-like (income missing when education low)**
```json
{ "income": { "strategy": "numeric_median_plus_indicator", "add_missing_indicator": true } }
```

**Case 3 — MNAR/target signal (credit_score missing when defaulted)**
```json
{ "credit_score": { "strategy": "numeric_median_plus_indicator", "add_missing_indicator": true } }
```

**Case 4 — Structural (facility_type=NaN when no facility)**
```json
{ "facility_type": { "strategy": "structural_none_or_zero", "add_missing_indicator": true } }
```

---

## How downstream agents should use imputation_plan.json

```python
import json

plan = json.load(open("outputs/logs/imputation_plan.json"))

for col, entry in plan["columns"].items():
    strategy  = entry["strategy"]
    indicator = entry["add_missing_indicator"]
    fit_on    = entry["fit_on"]     # always "train_only" when imputation is needed

    if strategy == "no_imputation_needed":
        continue
    elif strategy == "numeric_median":
        median = X_train[col].median()
        X_train[col] = X_train[col].fillna(median)
        X_test[col]  = X_test[col].fillna(median)   # use train median, not test!
    elif strategy == "numeric_median_plus_indicator":
        median = X_train[col].median()
        X_train[f"{col}_was_missing"] = X_train[col].isna().astype(int)
        X_test[f"{col}_was_missing"]  = X_test[col].isna().astype(int)
        X_train[col] = X_train[col].fillna(median)
        X_test[col]  = X_test[col].fillna(median)
    elif strategy == "categorical_missing_token":
        X_train[col] = X_train[col].fillna("MISSING")
        X_test[col]  = X_test[col].fillna("MISSING")
    elif strategy == "structural_none_or_zero":
        X_train[f"{col}_was_missing"] = X_train[col].isna().astype(int)
        X_test[f"{col}_was_missing"]  = X_test[col].isna().astype(int)
        X_train[col] = X_train[col].fillna("NONE")
        X_test[col]  = X_test[col].fillna("NONE")
    elif strategy == "drop_column":
        X_train = X_train.drop(columns=[col])
        X_test  = X_test.drop(columns=[col])
```

---

## Available imputation strategies

| Strategy | When applied |
|----------|-------------|
| `no_imputation_needed` | Column is fully observed |
| `numeric_median` | MCAR-compatible + <10% missing |
| `numeric_median_plus_indicator` | MAR-like, MNAR concern, or ≥10% missing |
| `categorical_missing_token` | Categorical + <10% missing |
| `categorical_mode_plus_indicator` | Categorical + ≥10% missing |
| `structural_none_or_zero` | Structural absence pattern detected |
| `drop_column` | >80% missing |
| `model_based_imputation_optional` | Complex MAR, moderate missingness |

---

## Composable sub-auditors

Each component can be used independently:

```python
from missingness_auditor.profiler   import MissingnessProfiler
from missingness_auditor.mechanism  import MechanismAuditor
from missingness_auditor.structural import StructuralMissingnessDetector

profile   = MissingnessProfiler(df, target_col="target").profile()
mechanism = MechanismAuditor(df, target_col="target").audit()
structural = StructuralMissingnessDetector(df).detect()
```

---

## Scope

**In scope:** missingness profile, mechanism clues, structural detection, imputation planning, leakage-safe protocol, 3 diagnostic figures.

**Out of scope:** MICE / iterative imputation implementation, AutoML, model search, EDA, PDF reports.
