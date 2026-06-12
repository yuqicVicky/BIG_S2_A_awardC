# Evaluation Guide — AI Missingness Auditor

## What makes this an Award C skill

### 1. Statistical reasoning before imputation
- Profiles missingness by column, dtype, severity, and overall dataset rate.
- Surfaces mechanism clues such as MCAR-compatible, MAR-like, group-dependent, target-associated, and structural absence concerns.
- Treats mechanism labels as observational evidence rather than causal proof.

### 2. Structural missingness detection
- Looks for paired evidence where a missing categorical value aligns with a numeric absence signal, such as `facility_quality` missing when `facility_area` is zero.
- Separates "the thing does not exist" from "the value was not collected."
- Recommends structural tokens or zero fills with missingness indicators when the evidence supports that interpretation.

### 3. Leakage-safe imputation planning
- Produces a per-column `imputation_plan.json` with strategy, reason, indicator flag, mechanism label, and `fit_on: train_only`.
- Preserves missingness signal for target-associated, MAR-like, group-dependent, and structural patterns.
- Prevents held-out or prediction rows from influencing medians, modes, group medians, or other fitted imputation statistics.

### 4. Agent-ready outputs
- Writes structured JSON logs for profile, mechanism audit, structural audit, leakage check, and imputation plan.
- Writes a human-readable Markdown report with evidence and limitations.
- Generates diagnostic figures that make missingness scale, co-missingness, and target signal inspectable.

### 5. LLM-enhanced interactive demo
- The core Python skill can run without an LLM.
- The Streamlit demo can use Claude to narrate findings, recommend useful charts, explain mechanism clues, and suggest strategy alternatives.
- Users can accept or override recommended imputation strategies before exporting artifacts.

## How to evaluate

```bash
cd BIG_S2_A_awardC
pip install -r requirements.txt
pip install -e .

PYTHONPATH=src python -m missingness_auditor.cli \
  --data examples/demo_missingness.csv \
  --target target \
  --out /tmp/missingness-auditor-smoke/ \
  --json-summary
```

To inspect the interactive app:

```bash
streamlit run app.py
```

Then open `http://localhost:8501`, load the bundled demo dataset, select `target`, and run the audit.

## Expected demo evidence

The bundled demo should show 1,000 rows, 7 columns, 4 columns with missing values, and an 11.0% overall missing rate.

| Column | Expected clue | Expected strategy |
|---|---|---|
| `age` | MCAR-compatible | `numeric_median_plus_indicator` |
| `income` | Group-dependent missingness by `education_level` | `groupwise_numeric_median_plus_indicator` |
| `risk_score` | Target-associated missingness | `numeric_median_plus_indicator` |
| `facility_quality` | Structural absence concern with `facility_area` | `structural_none_token_plus_indicator` |

## Expected artifacts

Saved audits should produce:

- `logs/missingness_profile.json`
- `logs/missingness_mechanism_audit.json`
- `logs/structural_missingness_audit.json`
- `logs/imputation_plan.json`
- `logs/leakage_safe_imputation_check.json`
- `reports/missing_data_report.md`
- diagnostic figures such as `missingness_bar.png`, `missingness_matrix.png`, and `missingness_target_signal.png`

## Review checklist

- The report explains missingness mechanisms as evidence, not as definitive causal labels.
- The imputation plan includes missingness indicators when missingness itself may carry information.
- Structural absence is handled differently from ordinary missing data.
- All imputation statistics are fit on training data only.
- The exported JSON and Markdown artifacts are reusable by another data-cleaning or modeling agent.
