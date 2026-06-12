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
| Brain / LLM | Claude is used in the Streamlit demo to narrate findings, recommend useful charts, explain mechanisms, and suggest alternatives. The core audit can run without an LLM. |
| Memory | JSON logs and Markdown reports store the profile, mechanism audit, structural findings, leakage check, and imputation plan. |
| Planning | Combines missing rates, dtype/cardinality, mechanism clues, structural evidence, and leakage rules into a per-column imputation plan. |
| Action | Runs profilers, mechanism auditors, structural detectors, imputation planners, leakage checks, visualizers, and report writers through Python, CLI, or Streamlit. |
| Execution | Python package plus Streamlit Cloud demo; the bundled demo dataset requires no external data. |
| Observation | Tables, diagnostic charts, JSON evidence, report text, and before/after imputation summaries are inspected before final recommendations. |
| Response | Delivers `imputation_plan.json`, `missing_data_report.md`, diagnostic figures, and optional imputed CSVs. |

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
- `missing_data_report.md`: human-readable audit report with evidence and limitations.
- Diagnostic figures: missingness bar chart, pattern matrix, and target-signal chart.
- Imputed CSV outputs: optional train/predict data after applying the selected plan.

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
