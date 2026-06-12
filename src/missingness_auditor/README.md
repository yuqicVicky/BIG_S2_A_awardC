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

This tool uses cautious, significance-test-driven labels (not bare correlation thresholds):
- **"MCAR-compatible"** — no significant association with observed features or target.
- **"MAR-like evidence"** — missingness significantly predicted by observed covariates (logistic-LR test).
- **"group-dependent missingness"** — missingness significantly associated with a categorical group (χ² test).
- **"target-associated missingness"** — missingness significantly associated with the target (point-biserial / χ²).

A global **Little's MCAR test** is also reported. Structural absence is detected separately by
`StructuralMissingnessDetector`, not by the mechanism auditor. These are observational clues, not
causal mechanism assignments.

---

## Quick demo

```python
from missingness_auditor import MissingnessAuditor
import pandas as pd

df = pd.read_csv("my_data.csv")
auditor = MissingnessAuditor(df, target_col="outcome")
results = auditor.run()                      # pure computation, no disk writes
auditor.save_outputs(results, "outputs/")    # writes logs, figures, reports + reproduction artifacts

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
| `outputs/logs/missingness_mechanism_audit.json` | MCAR / MAR-like / group-dependent / target-associated clue per column + Little's MCAR test |
| `outputs/logs/structural_missingness_audit.json` | Structural absence pairs |
| `outputs/logs/imputation_plan.json` | Per-column strategy, indicator flag, fit scope |
| `outputs/logs/leakage_safe_imputation_check.json` | Leakage risk per column + global protocol |
| `outputs/logs/mice_pooling.json` | MICE + Rubin's-rules pooling: pooled mean, MI vs naive SE, FMI (inference) |
| `outputs/logs/mnar_sensitivity.json` | MNAR delta-adjustment tipping points + fragile-column list |
| `outputs/reports/missing_data_report.md` | Human-readable narrative |
| `outputs/reports/missing_data_report.pdf` | Methods-appendix PDF |
| `outputs/figures/missingness_bar.png` | Bar chart of missing rates |
| `outputs/figures/pattern_matrix.png` | Row × column missingness pattern matrix |
| `outputs/figures/target_signal.png` | Target signal by missingness |
| `outputs/figures/missing_correlation.png` | Missingness-indicator correlation heatmap |
| `outputs/figures/decision_flow.png` | CONSORT-style imputation decision flow |
| `outputs/figures/mnar_tipping_point.png` | Delta-adjustment sensitivity trajectories |
| `outputs/reproduce_imputation.py` | Standalone, self-verifying reproduction script |

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
{ "facility_type": { "strategy": "structural_none_token_plus_indicator", "add_missing_indicator": true } }
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
    elif strategy in ("categorical_missing_token", "categorical_missing_token_plus_indicator"):
        if entry["add_missing_indicator"]:
            X_train[f"{col}_was_missing"] = X_train[col].isna().astype(int)
            X_test[f"{col}_was_missing"]  = X_test[col].isna().astype(int)
        X_train[col] = X_train[col].fillna("MISSING")
        X_test[col]  = X_test[col].fillna("MISSING")
    elif strategy == "structural_none_token_plus_indicator":
        X_train[f"{col}_was_missing"] = X_train[col].isna().astype(int)
        X_test[f"{col}_was_missing"]  = X_test[col].isna().astype(int)
        X_train[col] = X_train[col].fillna("NONE")
        X_test[col]  = X_test[col].fillna("NONE")
    elif strategy == "structural_zero_plus_indicator":
        X_train[f"{col}_was_missing"] = X_train[col].isna().astype(int)
        X_test[f"{col}_was_missing"]  = X_test[col].isna().astype(int)
        X_train[col] = X_train[col].fillna(0)
        X_test[col]  = X_test[col].fillna(0)
    elif strategy == "drop_column":
        X_train = X_train.drop(columns=[col])
        X_test  = X_test.drop(columns=[col])
```

---

## Available imputation strategies

| Strategy | When applied |
|----------|-------------|
| `no_imputation_needed` | Column is fully observed or is the target column |
| `numeric_median` | Numeric, MCAR-compatible + <10% missing |
| `numeric_median_plus_indicator` | Numeric, MAR-like / target-associated, or ≥10% missing |
| `groupwise_numeric_median_plus_indicator` | Numeric, group-dependent missingness (per-group median + global fallback) |
| `time_series_ffill_bfill_plus_indicator` | Numeric column with a temporal domain tag (ffill→bfill instead of median) |
| `categorical_missing_token` | Categorical + <10% missing |
| `categorical_missing_token_plus_indicator` | Categorical + ≥10% missing, group-dependent, or high-cardinality text |
| `structural_none_token_plus_indicator` | Categorical structural absence (NaN encodes "none") |
| `structural_zero_plus_indicator` | Numeric structural absence (companion-column zero evidence) |
| `drop_column` | >80% missing |

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

**In scope:** missingness profile; significance-test-driven mechanism clues (Little's MCAR, logistic-LR,
χ², point-biserial); structural detection; column-specific imputation planning; leakage-safe execution;
inference-grade analysis (MICE + Rubin's-rules pooling, MNAR delta-adjustment sensitivity); 5 diagnostic
figures; Markdown + PDF reports; and a standalone self-verifying reproduction script.

**Out of scope:** AutoML, model/hyperparameter search, and general EDA beyond missingness.
