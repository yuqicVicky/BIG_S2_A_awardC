# Data Pattern & Validation Auditor

A reusable Award C skill that helps autonomous data-science agents understand data patterns **before modeling**.

## What it does

| Module | What it infers |
|--------|---------------|
| `FeatureTypeInferrer` | Classifies every column: target, datetime, numeric, categorical, text, group ID, train-only, leakage-risk, etc. |
| `DistributionShiftDetector` | Compares train vs prediction: KS tests, category overlap, datetime coverage, split pattern |
| `TargetPatternAnalyser` | Correlations, target mean by quantile/category/datetime component |
| `LeakageAuditor` | Flags train-only features, target-name suspects, raw IDs — but NOT datetime features that are legitimately available at prediction time |
| `ValidationRecommender` | Recommends one of 8 strategies with reasoning, fit rule, validation rule, and strategies to avoid |
| `Visualizer` | Generates 6–8 matplotlib figures |
| `ReportWriter` | Writes 8 JSON logs + `pattern_audit.md` |

## Supported validation strategies

- `random_holdout`
- `kfold`
- `stratified_kfold`
- `time_holdout`
- `group_split`
- `group_time_split`
- `within_period_latest_available_holdout`
- `rolling_split`

## Quick start

```bash
pip install -r requirements.txt

# Run CLI
python -m pattern_auditor.cli \
  --train examples/data/within_period_train.csv \
  --predict examples/data/within_period_predict.csv \
  --target target \
  --out outputs/

# Or run the demo
python examples/run_demo.py
```

## Python API

```python
from pattern_auditor import PatternAuditor

auditor = PatternAuditor(
    train_df,
    predict_df,
    target_col="target",
    datetime_col="timestamp",
    group_col="entity_id",
)
results = auditor.run(output_dir="outputs/")
print(results["validation"]["recommendation"])
```

## Directory structure

```
src/pattern_auditor/
├── __init__.py           PatternAuditor orchestrator
├── schema.py             Schema inspection
├── feature_types.py      Feature type inference
├── datetime_patterns.py  Datetime component extraction and target aggregations
├── target_patterns.py    Target-aware pattern discovery
├── distribution_shift.py Train/predict comparison and split pattern detection
├── leakage.py            Leakage and feature availability audit
├── validation.py         Validation strategy recommender
├── visualization.py      Matplotlib figures
├── reporting.py          JSON logs and Markdown report
└── cli.py                CLI entry point
```

## Running tests

```bash
cd data-pattern-validation-auditor
pip install -r requirements.txt pytest
pytest tests/ -v
```

## Design principles

1. **Dataset-agnostic** — no domain-specific column names in `src/`
2. **Datetime ≠ leakage** — prediction-time timestamps carry valid signal
3. **Composable** — each sub-auditor works standalone or as part of the orchestrator
4. **LLM-ready** — JSON outputs are designed to be passed as context to an LLM agent
5. **Zero seaborn** — matplotlib only
