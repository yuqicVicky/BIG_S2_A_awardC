# Train-Test Pattern Auditor

## Purpose

A lightweight pre-modeling skill for autonomous data-science agents. It audits
feature availability, detects the train/prediction split pattern, flags leakage
risks, and recommends a validation strategy — before any model is trained.

## Trigger phrases

Use this skill when you see any of the following:
- "audit the data before modeling"
- "what validation strategy should I use?"
- "check for leakage"
- "understand the train/predict split"
- "which features are available at prediction time?"
- "feature availability check"

## Quick start

```python
from pattern_auditor import PatternAuditor

auditor = PatternAuditor(
    train_df,
    predict_df,
    target_col="target",
    datetime_col="timestamp",   # optional
    group_col="entity_id",      # optional
    row_id_col="row_id",        # optional
)
results = auditor.run(output_dir="outputs/")

# Key machine-readable answers for downstream agents
strategy   = results["validation"]["recommendation"]["strategy"]
to_exclude = results["feature_availability"]["columns_to_exclude"]
pattern    = results["train_prediction_pattern"]["pattern"]
```

## CLI

```bash
python -m pattern_auditor.cli \
  --train path/to/train.csv \
  --predict path/to/predict.csv \
  --target target \
  --datetime timestamp \
  --out outputs/
```

## Outputs

| File | Contents |
|------|----------|
| `outputs/logs/schema_audit.json` | Column dtypes, null rates, unique counts |
| `outputs/logs/feature_availability_audit.json` | Per-column availability, leakage risk, action |
| `outputs/logs/train_prediction_pattern.json` | Split pattern, evidence, distribution shift |
| `outputs/logs/leakage_audit.json` | Leakage risks with severity and reason |
| `outputs/logs/validation_recommendation.json` | Recommended CV strategy |
| `outputs/reports/pattern_audit.md` | Human-readable narrative summary |
| `outputs/figures/train_prediction_coverage.png` | Row/column counts for train vs predict |
| `outputs/figures/feature_availability.png` | Leakage risk breakdown |
| `outputs/figures/distribution_shift_summary.png` | Train vs predict numeric distributions |

## What it answers

| Question | Where to look |
|----------|---------------|
| Which columns are available at prediction time? | `feature_availability_audit.json` → `columns_to_use` |
| Which columns are train-only and must be excluded? | `feature_availability_audit.json` → `columns_to_exclude` |
| What is the train/prediction split pattern? | `train_prediction_pattern.json` → `pattern` |
| What validation strategy best simulates the hidden evaluation? | `validation_recommendation.json` → `recommendation.strategy` |
| Which validation strategies should be avoided? | `validation_recommendation.json` → `recommendation.strategies_to_avoid` |

## Validation strategies

| Detected pattern | Recommended strategy |
|-----------------|---------------------|
| Within-period | `within_period_latest_available_holdout` |
| Time-based | `time_holdout` or `rolling_split` |
| Group-based | `group_split` |
| Group + time | `group_time_split` |
| IID / unknown | `kfold` or `stratified_kfold` |

## Key design choices

- **Dataset-agnostic** — no domain-specific column names are hardcoded in `src/`.
- **Datetime features are not leakage** — timestamps available at prediction time carry valid signal.
- **Zero seaborn** — all figures use matplotlib only.
- **Composable** — each sub-auditor can be used standalone.

## Scope

This skill covers **only**:
1. Feature availability audit (available vs. train-only vs. high-risk)
2. Train/prediction pattern detection (time, group, within-period, i.i.d.)
3. Leakage risk audit (train-only features, target-component suspects, ID columns)
4. Validation strategy recommendation
5. Minimal visualization (3 figures)

Out of scope: model search, residual analysis, target transformation search,
ensemble search, complex text features, EDA, PDF report generation.

## Integration in an LLM agent

```python
import json
from pattern_auditor import PatternAuditor

def agent_pre_model_audit(train_df, predict_df, target_col):
    auditor = PatternAuditor(train_df, predict_df, target_col=target_col)
    results = auditor.run()
    return json.dumps({
        "pattern":      results["train_prediction_pattern"]["pattern"],
        "strategy":     results["validation"]["recommendation"]["strategy"],
        "exclude_cols": results["feature_availability"]["columns_to_exclude"],
        "verify_cols":  results["feature_availability"]["columns_to_verify"],
        "avoid":        results["validation"]["recommendation"].get("strategies_to_avoid", []),
    }, indent=2)
```
