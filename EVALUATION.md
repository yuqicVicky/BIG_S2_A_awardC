# Evaluation Guide — Data Pattern & Validation Auditor

## What makes this an Award C skill

### 1. Statistical reasoning beyond profiling
- Infers **feature semantics** (not just dtypes) from name patterns, cardinality, value distributions
- Detects **train/predict split patterns** algorithmically (within-period, time-based, group-based, iid)
- Runs **KS tests** for numeric distribution shift
- Computes **Pearson correlations** and **target mean by bin** for signal discovery

### 2. Leakage detection with correct datetime handling
- Flags train-only columns, target-name suspects, raw IDs — all high or medium severity
- **Correctly does NOT flag datetime-derived features** as leakage: timestamps available at prediction time carry legitimate signal and must not be excluded

### 3. Principled validation recommendation
Eight strategies are supported with structured reasoning:

| Strategy | When recommended |
|----------|-----------------|
| `within_period_latest_available_holdout` | Predict timestamps overlap train period |
| `time_holdout` | Predict strictly after train |
| `rolling_split` | Alternative for temporal splits |
| `group_split` | Disjoint groups train vs predict |
| `group_time_split` | Both group and time structure |
| `kfold` / `stratified_kfold` | No temporal or group signal |
| `random_holdout` | Baseline fallback |

Each recommendation includes: strategy, confidence, reason, fit rule, validation rule, strategies to avoid.

### 4. Composable architecture
Every sub-auditor can be used independently:
```python
from pattern_auditor.leakage import LeakageAuditor
from pattern_auditor.validation import ValidationRecommender
```

### 5. LLM-agent integration
- All outputs are structured JSON
- Compact summary dict designed for LLM context injection
- CLI enables shell-level invocation from agent frameworks

## How to evaluate

```bash
cd data-pattern-validation-auditor
pip install -r requirements.txt pytest
pytest tests/ -v                  # all 8 test modules
python examples/run_demo.py       # end-to-end demo
```

### Expected demo output

```
Detected split pattern : within_period
Recommended strategy   : within_period_latest_available_holdout
Confidence             : high
```

### Expected test results

All tests should pass. Key assertions:
- `test_within_period_split_detected` — detects overlapping timestamp ranges
- `test_datetime_derived_not_flagged_as_leakage` — datetime safety
- `test_train_only_column_flagged` — leakage detection
- `test_within_period_recommends_within_period_holdout` — correct strategy
- `test_no_domain_terms_in_src` — dataset-agnostic
- `test_figures_are_nonempty_png` — visualization integrity
