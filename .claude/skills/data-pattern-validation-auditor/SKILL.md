# Data Pattern & Validation Auditor

## Purpose

A statistical pre-modeling skill for autonomous data-science agents. It infers feature types, compares train vs prediction data, discovers target-related patterns, detects leakage/availability risks, and recommends a validation strategy — before any model is trained.

## Trigger phrases

Use this skill when you see any of the following:
- "audit the data before modeling"
- "what validation strategy should I use?"
- "check for leakage"
- "understand the train/predict split"
- "profile the data patterns"
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

strategy = results["validation"]["recommendation"]["strategy"]
# e.g. "within_period_latest_available_holdout"
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

| File | Description |
|------|-------------|
| `outputs/logs/schema_report.json` | Column dtypes, null rates, unique counts |
| `outputs/logs/feature_type_report.json` | Inferred feature type per column |
| `outputs/logs/train_prediction_pattern.json` | Distribution shift and split pattern |
| `outputs/logs/target_pattern_report.json` | Correlations, target means by feature |
| `outputs/logs/leakage_feature_audit.json` | Leakage risks with severity |
| `outputs/logs/validation_recommendation.json` | Recommended CV strategy |
| `outputs/logs/feature_recommendation.json` | Per-column engineering actions |
| `outputs/figures/*.png` | All visualization plots |
| `outputs/reports/pattern_audit.md` | Human-readable summary report |

## Validation strategies

| Pattern | Recommended strategy |
|---------|---------------------|
| Within-period | `within_period_latest_available_holdout` |
| Time-based | `time_holdout` or `rolling_split` |
| Group-based | `group_split` |
| Group + time | `group_time_split` |
| IID / unknown | `kfold` or `stratified_kfold` |

## Key design choices

- **Datetime features are not leakage** — timestamps available at prediction time carry valid signal and are not flagged as leakage risks.
- **Dataset-agnostic** — no domain-specific column names are hardcoded in `src/`.
- **Zero seaborn** — all figures use matplotlib only, no extra dependency.
- **Composable** — each sub-auditor (`FeatureTypeInferrer`, `LeakageAuditor`, etc.) can be used standalone.

## Integration in an LLM agent

```python
import json
from pattern_auditor import PatternAuditor

def agent_pre_model_audit(train_df, predict_df, target_col):
    auditor = PatternAuditor(train_df, predict_df, target_col=target_col)
    results = auditor.run()
    # Pass the compact JSON to the LLM as context
    context = {
        "split_pattern": results["distribution_shift"].get("split_pattern", {}),
        "validation": results["validation"],
        "high_leakage_risks": [
            r for r in results["leakage"].get("risks", [])
            if r["severity"] == "high"
        ],
        "top_features": results["feature_engineering"]["high_signal_numeric"][:5],
    }
    return json.dumps(context, indent=2)
```
