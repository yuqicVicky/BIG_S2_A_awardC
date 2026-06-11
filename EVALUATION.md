# Evaluation Guide — Missingness Audit & Imputation Planner

This guide tells an evaluator exactly what to run and what each capability proves.
The skill (`missingness-auditor`) diagnoses missing data and produces a leakage-safe,
per-column imputation plan **before** any imputation is applied — plus inference-grade
artifacts (multiple imputation, MNAR sensitivity) and a self-verifying reproduction
script.

---

## What makes this an Award C skill

### 1. Statistical reasoning, not just profiling
Mechanism labels are driven by **hypothesis tests**, not bare correlation thresholds:

| Question | Test | Where |
|----------|------|-------|
| Is the data MCAR globally? | **Little's (1988) χ² MCAR test** (EM-estimated mean/cov) | `mechanism.py::_little_mcar_test`; reported as `little_mcar_test` |
| Does a column's missingness depend on observed covariates (MAR)? | **Logistic-regression likelihood-ratio test** of the missingness indicator on the other numeric columns | `mechanism.py::_logistic_lr_test` → `mar_test` |
| Is missingness concentrated in categorical groups? | **χ² test of independence** | `_chi2_independence` → `group_dependency_evidence.chi2_p_value` |
| Is missingness associated with the target? | **point-biserial** (numeric target) or **χ²** (categorical target) | `target_test` |

Correlations are retained as *effect size*; the **label is decided by the p-value**, and
every test degrades gracefully to the legacy correlation rule if scipy/sklearn are
unavailable or the sample is too small.

### 2. Hard safety guards (enforced, not advisory)
- **The target column is never imputed** — removed from the plan and asserted untouched,
  including inside the MICE engine.
- **Imputers are fit on `df_train` only** — every statistic (median, mode, group median,
  IterativeImputer) is fitted on train and applied to predict via transform/`fillna`.
  Injecting extreme values into the predict frame does not change the fill.

### 3. Structural missingness vs. data error
Detects `(categorical NA ⇔ numeric companion is 0)` patterns and fills them with
`NONE`/`0` + an indicator — never the mean/mode, which would invent a value for a thing
that does not exist (`structural.py`).

### 4. Inference-grade analysis
- **MICE + Rubin's rules** (`mice.py`): pooled estimate, naive vs. multiple-imputation
  standard error, and the **fraction of missing information** per column.
- **MNAR delta-adjustment sensitivity** (`sensitivity.py`): the **tipping point** — how
  large an MNAR departure (in SD units) would overturn a MAR-based conclusion.

### 5. Reproducibility
`codegen.py` emits a **standalone, self-verifying** reproduction script (no dependency on
this package) that re-applies the plan with train-only statistics and asserts its own
output: no residual nulls, target untouched, indicators binary, dropped columns gone.

### 6. Dataset-agnostic core
No domain-specific column names appear in `src/` logic (AST-checked by
`test_no_hardcoding.py`). Domain inference happens only in the skill/LLM layer.

---

## How to evaluate

```bash
pip install -r requirements.txt
pip install -e .
pytest tests/ -v                                  # 36 tests, all self-contained
python -m missingness_auditor.cli \
  --data examples/demo_missingness.csv --target target --out outputs/demo
```

### Expected mechanism output (demo dataset)

Inspect `outputs/demo/logs/missingness_mechanism_audit.json`:

- Top-level `little_mcar_test` → **rejects MCAR** (p ≈ 0 over 5 numeric columns).
- `age` → **MCAR-compatible** (logistic-LR p ≈ 0.3, target p ≈ 0.9)
- `income` → **group-dependent missingness** (χ² vs `education_level`, p ≈ 0)
- `risk_score` → **target-associated missingness** (point-biserial r ≈ 0.43, p ≈ 0)
- `facility_quality` → **MAR-like evidence** (logistic-LR p ≈ 0 via `facility_area`)

### Web demo

```bash
streamlit run app.py
```

The demo runs **with or without** an Anthropic API key: every panel (overview, profile,
mechanisms, structural, plan) renders a deterministic, rule-based narrative built from the
audit JSON, and the LLM *upgrades* the prose when a key is provided.

---

## What each test file proves

| Test file | Guarantee |
|-----------|-----------|
| `test_hard_guards.py` | Target excluded from the plan; its values never change (incl. inside MICE) |
| `test_leakage_safe.py` | Fill statistics come from train only — predict-frame extremes don't move the fill |
| `test_structural.py` | Structural absence is filled with `NONE`/`0` + indicator, never mean/mode |
| `test_strategies_run.py` | Every strategy (MICE, time-series, groupwise, …) applies with zero residual nulls |
| `test_mice.py` | Rubin pooling inflates variance over single imputation (`T > Ū`, `FMI ∈ (0,1)`) |
| `test_sensitivity.py` | MNAR delta-adjustment finds the tipping point for fragile columns; reports robust ones |
| `test_edge_cases.py` | Degenerate inputs (all-null column, single row, no target, constant column, train-null/predict-present) complete with no residual nulls and target untouched |
| `test_no_hardcoding.py` | No domain-specific column names in `src/` logic (AST-checked) |

All tests are self-contained — no external downloads.
