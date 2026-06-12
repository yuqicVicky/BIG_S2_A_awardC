# [Award C] AI Missingness Auditor

> **Team info**
> | Legal name | Affiliation | Institutional email | Kaggle username |
> |---|---|---|---|
> | Yuqi Cheng | University of North Carolina at Chapel Hill | yuqi16614994@gmail.com | yuqic1661 |
> | Shucheng Liu | University of North Carolina at Chapel Hill | lsc210204@gmail.com | shuchengliu |
> | Akemi Hara | University of North Carolina at Chapel Hill | akehara1001@gmail.com | akehara |
> | Shan Gao | University of North Carolina at Chapel Hill | ssssgao777@gmail.com | GaoSShan |
>
> **Registered team name:** BIG-S2_A

**GitHub repository:** https://github.com/yuqicVicky/BIG_S2_A_awardC

---

## What it does

AI Missingness Auditor is a reusable statistical skill for diagnosing missing data before imputation. It profiles missingness, detects MCAR-compatible, MAR-like, group-dependent, target-associated, and structural absence clues, then produces a leakage-safe imputation plan that downstream data-science agents can apply. The live demo is a Claude-enhanced Streamlit app that explains the audit, shows diagnostic figures, and lets users accept or override recommended strategies.

## Demo link

Live demo: https://aimissingnessauditor.streamlit.app/

Kaggle discussion index: https://www.kaggle.com/competitions/stai-x-challenge-2026/discussion?sort=hotness

![AI Missingness Auditor Streamlit report](figures/main.png)

## Agent Design and Architecture

| Component | What it does |
|---|---|
| Brain / LLM | Claude supplies the **semantic judgment** a rule table can't: Step 1.5 infers each column's real-world meaning, domain tag, grouping column, and structural pairs. These feed the engine and **change the chosen strategy** (e.g. a temporal tag → forward-fill instead of median). It also narrates findings and picks charts; the statistical core runs without it. |
| Memory | The audit `results` dict carries state across steps (profile → mechanism → structural → plan → leakage check); `imputation_plan.json` persists the plan so a downstream agent can apply it without re-running the audit. |
| Planning | LLM semantic judgment (meaning, grouping, structural pairs) **+** `ImputationPlanner`'s deterministic strategy ladder (mechanism clue, missing rate, cardinality, domain tag), flagging `mi_upgrade_recommended` when single imputation is insufficient. |
| Action | Sub-auditors gather evidence: `MissingnessProfiler` (rates/severity), `MechanismAuditor` (Little's MCAR, logistic-LR, χ², point-biserial tests), `StructuralMissingnessDetector` (absence pairs), plus scikit-learn `IterativeImputer` for model-based and MICE imputation. |
| Execution | `Imputer` applies the plan leakage-safe — every median/mode/group-median/model is fit on `df_train` only and transferred to predict data. Runs via Python API, CLI (`python -m missingness_auditor.cli`), or the Streamlit app. |
| Observation | `LeakageSafeImputationChecker` verifies fit scope; five diagnostic figures, an MNAR delta-adjustment sensitivity analysis, and a standalone self-verifying reproduction script let humans and agents inspect the result. |
| Response | Delivers a Markdown + PDF report, structured JSON logs (`imputation_plan.json`, mechanism/structural/leakage/MICE-pooling/MNAR), diagnostic figures, and optional imputed train/predict CSVs. |

**Design principle:** the LLM judges *semantics*; deterministic statistics own *every label, p-value, and strategy* — reproducible, auditable, offline. The LLM never invents a number.

## Why participants can adopt it

- **Diagnose before imputation:** the module makes the agent inspect missingness mechanisms before filling values.
- **Preserve missingness signal:** target-associated or MAR-like gaps receive missingness indicators instead of being silently smoothed away.
- **Detect structural absence:** the detector separates "the thing does not exist" from "the value was not collected."
- **Prevent leakage:** all medians, modes, group medians, and imputation parameters are fit on training data only.

## Example output

The bundled demo has 1,000 rows, 7 columns, 4 columns with missing values, and an 11.0% overall missing rate.

| Column | Missing rate | Mechanism clue | Recommended strategy |
|---|---:|---|---|
| `age` | 10.1% | MCAR-compatible | `numeric_median_plus_indicator` |
| `income` | 17.7% | Group-dependent missingness by `education_level` | `groupwise_numeric_median_plus_indicator` |
| `risk_score` | 11.1% | Target-associated missingness | `numeric_median_plus_indicator` |
| `facility_quality` | 37.8% | Structural absence concern with `facility_area` | `structural_none_token_plus_indicator` |

![AI Missingness Auditor diagnostics](figures/visual.png)

## Outputs

- `imputation_plan.json`: machine-readable plan for downstream agents.
- `missing_data_report.md` + `missing_data_report.pdf`: human-readable audit report with evidence and limitations.
- JSON logs: mechanism audit, structural audit, leakage check, MICE + Rubin's-rules pooling (`mice_pooling.json`), and MNAR sensitivity (`mnar_sensitivity.json`).
- Five diagnostic figures: missingness bar chart, pattern matrix, target-signal chart, missingness-correlation heatmap, decision-flow diagram, and an MNAR tipping-point plot.
- `reproduce_imputation.py`: standalone, self-verifying reproduction script.
- Imputed CSV outputs: optional train/predict data after applying the selected plan.

## Reproducibility

The full skill, demo, and bundled demo data are in the GitHub repository, with a standalone reproduction path:

```bash
git clone https://github.com/yuqicVicky/BIG_S2_A_awardC
cd BIG_S2_A_awardC
pip install -r requirements.txt
pip install -e .

# Reproduce the bundled demo audit (writes logs, figures, report, reproduction script)
PYTHONPATH=src python -m missingness_auditor.cli \
  --data examples/demo_missingness.csv \
  --target target \
  --out outputs/demo_missingness/ \
  --json-summary

# Or launch the interactive Streamlit demo
streamlit run app.py
```

Every saved audit also emits `outputs/reproduce_imputation.py` — a self-contained script (no dependency on this package) that re-applies the exact plan and verifies its own work (no residual nulls, target untouched, indicators binary). See `EVALUATION.md` for the full review checklist.

## Reuse in another agent

```python
import pandas as pd
from missingness_auditor import MissingnessAuditor

df = pd.read_csv("my_data.csv")
auditor = MissingnessAuditor(df, target_col="target")
results = auditor.run()

plan = results["imputation_plan"]
imputed_df = auditor.apply_imputation(df, plan)
```

This makes the missingness audit a drop-in pre-imputation step for a larger statistical agent, data-cleaning assistant, or modeling pipeline.
