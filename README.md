# Missingness Audit & Imputation Planner

A reusable pre-imputation skill for autonomous data-science agents and interactive
data scientists. It diagnoses missing data — profiling severity, identifying mechanism
clues, detecting structural absence patterns, and recommending a leakage-safe
imputation strategy for every column — before a single imputation is applied.

---

## What it does

Given a CSV or DataFrame, the auditor produces:

| Output | Description |
|--------|-------------|
| **Missingness profile** | Per-column missing rates, severity labels (trace/low/moderate/high), dtype category |
| **Mechanism clues** | MCAR-compatible / MAR-like / MNAR/structural concern — statistical clues, not causal claims |
| **Structural missingness** | Detects `(categorical NA → numeric companion is 0)` patterns — absence of a thing, not data error |
| **Imputation plan** | Column-specific strategy from 8 options, with a leakage-safe fit scope |
| **Leakage-safe protocol** | Confirms all statistics are fit on train only, never on predict |
| **3 diagnostic figures** | Missingness bar chart, pattern matrix, target signal by missingness |
| **Markdown report** | Human-readable narrative summary of all findings |

---

## Why diagnosis matters before imputation

Skipping diagnosis leads to three common errors:

1. **Mean-imputing a structurally absent feature** — e.g., filling `facility_quality=NaN` (facility doesn't exist) with the mean quality creates a meaningless row. The correct fill is `"NONE"` or `0`.
2. **Ignoring MAR missingness** — if `income` is missing for low-education rows and you impute without an indicator column, the model never learns that the patient who refused to report income is different from one who did.
3. **Leakage** — computing the training-set median on the full dataset (including predict rows) inflates apparent model performance on held-out data.

This auditor catches all three.

---

## Missingness mechanisms

| Label | Meaning | Imputation implication |
|-------|---------|----------------------|
| **MCAR-compatible** | No correlation with other features or target | Simple median/mode fill is safe |
| **MAR-like evidence** | Correlated with at least one observed feature | Add a missing indicator so the model can learn the pattern |
| **MNAR/structural concern** | Correlated with the target column | Add indicator; consider whether the column itself is informative |

These are **clues from correlation patterns**, not causal assignments. Use domain knowledge to confirm.

---

## Web demo quickstart

```bash
pip install -r requirements.txt
pip install -e .
python examples/make_demo_missingness_data.py   # creates examples/demo_missingness.csv
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

**What the app does:**
1. Upload a CSV or load the built-in demo dataset (600 rows, 4 missingness types)
2. Select the target column
3. Click **Run Audit** — results appear in 5 tabs: Profile, Mechanisms, Structural, Imputation Plan, Figures
4. Click **Apply Imputation** to see the imputed DataFrame in-browser
5. Download the markdown report, `imputation_plan.json`, and imputed CSVs

---

## CLI quickstart

Single dataset:
```bash
python -m missingness_auditor.cli \
  --data examples/demo_missingness.csv \
  --target target \
  --out outputs/demo_missingness
```

Train / predict split (leakage-safe):
```bash
python -m missingness_auditor.cli \
  --train train.csv \
  --predict predict.csv \
  --target target \
  --out outputs/
```

JSON summary to stdout (for piping):
```bash
python -m missingness_auditor.cli --data data.csv --out /tmp/out --json-summary
```

---

## Worked example — challenge data shape (state × week overdose rates)

A synthetic panel dataset shaped like the STAI-X 2026 problem (suspected nonfatal
overdose ED-visit rates, observed weekly across states) exercises every capability on
the kind of data the challenge targets:

```bash
python examples/overdose/run_overdose_demo.py
```

| Column | Missingness it embeds | Auditor response |
|--------|----------------------|------------------|
| `ed_visit_rate` | a few states under-report some weeks | **groupwise per-state** median + indicator |
| `naloxone_admin_rate` | multi-week reporting outages | **time-series ffill/bfill** + indicator |
| `subprogram_type` | absent where no sub-program exists | **structural `NONE` token** + indicator |
| `ed_visit_rate` | the highest-rate weeks hide themselves | flagged **fragile** by MNAR sensitivity (tips at \|δ\|=0.5 SD) |

The data is **synthetic and illustrative** — generated from random numbers, containing
no real surveillance data and **not** the official competition dataset (the challenge
permits only official data). It exists solely to demonstrate the auditor on the
problem's *shape*. See `examples/overdose/make_overdose_demo.py`.

---

## Python API

```python
from missingness_auditor import MissingnessAuditor

auditor = MissingnessAuditor(df, predict_df=predict_df, target_col="target")
results = auditor.run()               # pure computation, no disk writes
auditor.save_outputs(results, "outputs/")  # writes 9 files

# Apply the plan (leakage-safe: fit on train, apply to predict)
df_train_imputed, df_predict_imputed = auditor.apply_imputation(
    df, results["imputation_plan"], df_predict
)
```

---

## Example output — demo dataset

The demo dataset (`examples/demo_missingness.csv`, 600 rows) has four missingness types:

| Column | Missing Rate | Mechanism | Strategy |
|--------|-------------|-----------|----------|
| `age` | ~7% | MCAR-compatible | `numeric_median` |
| `income` | ~35% | MAR-like (correlated with `education_level`) | `numeric_median_plus_indicator` |
| `risk_score` | ~29% | MNAR/structural concern (missing when `target==1`) | `numeric_median_plus_indicator` |
| `facility_quality` | ~35% | Structural (None when `facility_area==0`) | `structural_none_or_zero` |

After imputation: 0 missing values remain; 3 indicator columns added (`income_was_missing`,
`risk_score_was_missing`, `facility_quality_was_missing`).

---

## How agents use `imputation_plan.json`

The plan is the primary machine-readable output for downstream LLM agents:

```json
{
  "columns": {
    "income": {
      "strategy": "numeric_median_plus_indicator",
      "add_missing_indicator": true,
      "reason": "mechanism='MAR-like evidence'_or_moderate_missing_rate",
      "fit_on": "train_only"
    },
    "risk_score": {
      "strategy": "numeric_median_plus_indicator",
      "add_missing_indicator": true,
      "reason": "mechanism='MNAR/structural concern'_or_moderate_missing_rate",
      "fit_on": "train_only"
    }
  },
  "summary": {
    "strategy_counts": { "numeric_median": 1, "numeric_median_plus_indicator": 2, ... },
    "columns_needing_missing_indicator": ["income", "risk_score", "facility_quality"],
    "columns_to_drop": []
  }
}
```

An agent reads this file and applies the plan without needing to re-run the audit:

```python
import json
from missingness_auditor.imputer import Imputer

with open("outputs/logs/imputation_plan.json") as f:
    plan = json.load(f)

df_train_imputed = Imputer().apply(df_train, plan)
```

---

## Running tests

```bash
pip install -e .
pytest tests/ -v
```

30 tests across 7 test files — all self-contained, no external downloads. The suite
targets the guarantees the skill advertises (not coverage for its own sake):

| Test file | What it proves |
|-----------|----------------|
| `test_hard_guards.py` | Target column is excluded from the plan and its values are never modified — including inside the MICE engine |
| `test_leakage_safe.py` | Fill statistics come from **train only**: injecting extreme values into the predict frame does not change the fill (median and group-median paths) |
| `test_structural.py` | Structural absence is filled with `NONE`/`0` + indicator, never the mean/mode |
| `test_strategies_run.py` | Every strategy (incl. MICE, time-series, groupwise) applies with no error and zero residual nulls |
| `test_mice.py` | Rubin pooling inflates variance over single imputation (`T > Ū`, `FMI ∈ (0,1)`); MICE draws differ |
| `test_sensitivity.py` | MNAR delta-adjustment finds the tipping point for fragile columns and reports robust ones as robust |
| `test_no_hardcoding.py` | No domain-specific column names appear in `src/` code logic (AST-checked, docstrings excluded) |

---

## Streamlit Cloud deployment

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app
3. Set main file to `app.py`
4. Set requirements file to `requirements.txt`
5. Set packages to install: add a `packages.txt` with no content (or leave default)

The app will use the demo CSV bundled in `examples/`. No external data is needed.

---

## Outputs per run

```
outputs/
├── logs/
│   ├── missingness_profile.json          per-column rates, severity, dtype
│   ├── missingness_mechanism_audit.json  MCAR/MAR-like/MNAR clue per column
│   ├── structural_missingness_audit.json structural pair detection results
│   ├── imputation_plan.json              per-column strategy + fit scope
│   ├── leakage_safe_imputation_check.json leakage risk findings + protocol
│   ├── mice_pooling.json                 multiple-imputation Rubin pooling (inference)
│   └── mnar_sensitivity.json             delta-adjustment MNAR tipping points
├── figures/
│   ├── missingness_bar.png               missing rate bar chart by column
│   ├── pattern_matrix.png                row × column presence/absence matrix
│   ├── target_signal.png                 target mean: missing vs present rows
│   ├── decision_flow.png                 CONSORT-style imputation decision flow
│   └── mnar_tipping_point.png            delta-adjustment sensitivity trajectories
├── reports/
│   ├── missing_data_report.md            human-readable narrative summary
│   └── missing_data_report.pdf           methods-appendix PDF (attach to a paper)
├── reproduce_imputation.py               standalone, self-verifying reproduction script
└── source_data.csv                       frozen copy of the input for reproduction
```

### Reproducibility & inference-grade artifacts

- **`reproduce_imputation.py`** is standalone (no dependency on this package): it
  embeds the plan, re-applies it with train-only statistics, and **verifies its own
  output** (no residual nulls, target untouched, indicators binary). Run it months
  later to reproduce the cleaned dataset, or attach it to a methods appendix.
- **`mice_pooling.json`** reports the pooled mean, the naive single-imputation SE,
  the multiple-imputation SE, and the **fraction of missing information** per column —
  the variance single imputation hides (Rubin 1987; van Buuren FIMD Ch2).
- **`mnar_sensitivity.json`** + `mnar_tipping_point.png` report the **tipping point**:
  how large an MNAR departure (in SD units) would overturn a conclusion drawn under
  MAR. Small tipping point ⇒ the result hinges on an untestable assumption (FIMD Ch9).

---

## Imputation strategies

| Strategy | When used |
|----------|-----------|
| `no_imputation_needed` | Column is fully observed |
| `numeric_median` | MCAR-compatible, <10% missing, numeric |
| `numeric_median_plus_indicator` | MAR-like, MNAR concern, or ≥10% missing, numeric |
| `categorical_missing_token` | Categorical, <10% missing |
| `categorical_mode_plus_indicator` | Categorical, ≥10% missing |
| `structural_none_or_zero` | Structural absence pattern detected |
| `time_series_ffill_bfill_plus_indicator` | Temporally-ordered measurement (forward/backward fill) |
| `drop_column` | >80% missing |
| `model_based_imputation_optional` / `mice_multiple_imputation` | Complex MAR — leakage-safe `IterativeImputer` (the MICE engine), not a median fallback |

For **statistical inference** (not just ML features), the auditor additionally runs
full Multiple Imputation by Chained Equations and pools the results with **Rubin's
rules**, and runs an **MNAR delta-adjustment sensitivity analysis** — see below.

---

## Award C — what this demonstrates

This skill shows how a Claude Code agent can:

1. **Diagnose before acting** — audit missingness mechanisms before any imputation, matching what a careful data scientist would do
2. **Produce leakage-safe plans** — fit statistics on train only, apply to predict without re-fitting; the plan explicitly records `"fit_on": "train_only"` for every column
3. **Handle structural missingness** — distinguish "facility doesn't exist" (fill with NONE/0) from "data entry error" (impute the real value)
4. **Generate machine-readable outputs** — `imputation_plan.json` is designed to be read by a downstream agent in the next pipeline step
5. **Work on any dataset** — no domain-specific column names are hardcoded anywhere in `src/`

The web demo makes all of this interactive without requiring any code from the user.
