---
name: project-missingness-skill
description: Missingness Audit & Imputation Planner skill built for Award C on the missingness-skill branch
metadata:
  type: project
---

A new `missingness_auditor` Python package was built for the STAI-X Award C submission on branch `missingness-skill`.

**Why:** Award C requires a broadly useful statistical skill for autonomous data-science agents. The original Train-Test Pattern Auditor was too competition-specific; this is a general-purpose missingness diagnosis tool.

**Location:** `src/missingness_auditor/` — 8 source modules, zero hardcoded column names.

**How to apply:** When the user asks about the missingness skill, remember it lives in `src/missingness_auditor/` (not `src/pattern_auditor/`) and tests are in `tests/missingness/`. The demo runs from `examples/missingness/run_demo.py`.

Key design decisions:
- Mechanism labels are cautious: "MCAR-compatible", "MAR-like evidence", "MNAR/structural concern"
- Structural zero pattern: when categorical col is NaN → numeric companion is 0
- `fit_on: "train_only"` is enforced in imputation_plan.json for leakage safety
- Tests run via `pytest tests/missingness/`; CLI tests need `PYTHONPATH=src/` (handled in conftest via `_CLI_ENV`)
