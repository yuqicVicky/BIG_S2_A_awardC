# Train-Test Pattern Auditor

A lightweight pre-modeling skill for autonomous data-science agents. It audits
feature availability, detects the train/prediction split pattern, flags leakage
risks, and recommends a validation strategy — before any model is trained.

## What it answers

| Question | Output |
|----------|--------|
| Which columns are available at prediction time? | `feature_availability_audit.json` → `columns_to_use` |
| Which columns are train-only and must be excluded? | `feature_availability_audit.json` → `columns_to_exclude` |
| What is the train/prediction split pattern? | `train_prediction_pattern.json` → `pattern` |
| What validation strategy best simulates the hidden evaluation? | `validation_recommendation.json` → `recommendation.strategy` |
| Which validation strategies should be avoided? | `validation_recommendation.json` → `strategies_to_avoid` |

---

## Quick Test

```bash
pip install -r requirements.txt
pip install -e .
pytest tests/ -v
```

All tests are self-contained — no external data downloads required.

---

## Quick Demo

Each demo case generates synthetic data, runs the full audit, and prints a
2-minute summary to the terminal.

```bash
# Case A — within-period split: random CV would be misleading
python examples/run_demo.py --case within_period --out outputs/demo_within_period

# Case B — i.i.d. split: standard k-fold is appropriate
python examples/run_demo.py --case iid --out outputs/demo_iid

# Case C — unseen-group split: group holdout is required
python examples/run_demo.py --case group --out outputs/demo_group

# Case D — future time split: time holdout required
python examples/run_demo.py --case future_time_split --out outputs/demo_time

# Case E+F — deliberate leakage + valid datetime
python examples/run_demo.py --case leakage --out outputs/demo_leakage
```

---

## What to Look For

### `train_prediction_coverage.png`
Bar charts comparing row counts and column counts between train and predict.
A large difference in column counts suggests train-only columns (leakage risks).

### `feature_availability.png`
Pie chart of leakage risk severity (High / Medium / Low) and a bar chart of
risk types. A large "High" slice signals columns that must be excluded before
modeling.

### `distribution_shift_summary.png`
Overlaid histograms of train vs predict for the most shifted numeric columns
(starred with `*`). Large divergence means the model may encounter out-of-
distribution inputs at prediction time.

### `validation_recommendation.json`
Machine-readable output for downstream agents. Key fields:
```json
{
  "detected_split_pattern": "within_period_group",
  "recommendation": {
    "strategy": "within_period_latest_available_holdout",
    "confidence": "high",
    "strategies_to_avoid": ["time_holdout", "kfold"],
    "fit_rule": "...",
    "validation_rule": "..."
  }
}
```

---

## Python API

```python
from pattern_auditor import PatternAuditor

auditor = PatternAuditor(
    train_df,
    predict_df,
    target_col="target",
    datetime_col="timestamp",  # optional
    group_col="entity_id",     # optional
)
results = auditor.run(output_dir="outputs/")

# Key outputs for a downstream agent
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
  --group entity_id \
  --out outputs/
```

---

## Outputs (9 files per run)

```
outputs/
├── logs/
│   ├── schema_audit.json               column dtypes, null rates, unique counts
│   ├── feature_availability_audit.json per-column: available, leakage risk, action
│   ├── train_prediction_pattern.json   split pattern, evidence, distribution shift
│   ├── leakage_audit.json              leakage risks with severity and reason
│   └── validation_recommendation.json  recommended CV strategy
├── figures/
│   ├── train_prediction_coverage.png
│   ├── feature_availability.png
│   └── distribution_shift_summary.png
└── reports/
    └── pattern_audit.md               human-readable narrative report
```

---

## Supported Validation Strategies

| Detected pattern | Strategy | Confidence |
|-----------------|----------|------------|
| `within_period` | `within_period_latest_available_holdout` | high |
| `within_period_group` | `within_period_latest_available_holdout` | high |
| `time_based_split` | `time_holdout` | high |
| `group_based_split` | `group_split` | high |
| `group_time_split` | `group_time_split` | high |
| `iid_random` | `kfold` | medium |

---

## Directory Structure

```
src/pattern_auditor/
├── __init__.py           PatternAuditor orchestrator
├── schema.py             Schema inspection
├── feature_types.py      Feature type inference
├── datetime_patterns.py  Datetime component extraction
├── distribution_shift.py Train/predict comparison and split pattern detection
├── leakage.py            Leakage and feature availability audit
├── validation.py         Validation strategy recommender
├── visualization.py      Matplotlib figures (3 figures)
├── reporting.py          JSON logs and Markdown report
└── cli.py                CLI entry point

examples/
├── make_within_period_demo.py   Case A generator
├── make_iid_demo.py             Case B generator
├── make_group_demo.py           Case C generator
├── make_time_split_demo.py      Case D generator
├── make_leakage_demo.py         Case E+F generator
└── run_demo.py                  Demo runner (--case / --out)

tests/
├── test_within_period_split.py  Case A assertions
├── test_time_split.py           Case D assertions
├── test_group_split.py          Case C assertions
├── test_leakage_audit.py        Leakage detection unit + Case E
├── test_feature_availability.py Feature availability across cases
├── test_valid_datetime_not_leakage.py  Case F (datetime ≠ leakage)
├── test_validation_recommendation.py  All 5 cases + unit tests
├── test_cli_outputs.py          CLI integration (all 5 cases)
└── test_no_hardcoding.py        No domain-specific terms in src/
```

## Design Principles

1. **Dataset-agnostic** — no domain-specific column names in `src/`
2. **Datetime ≠ leakage** — prediction-time timestamps carry valid signal
3. **Composable** — each sub-auditor works standalone or as part of the orchestrator
4. **LLM-ready** — JSON outputs are designed to be passed as context to an LLM agent
5. **Zero seaborn** — matplotlib only
