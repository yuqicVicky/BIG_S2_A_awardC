---
name: project-missingness-skill
description: Missingness Audit & Imputation Planner skill built for Award C on the missingness-skill branch
metadata:
  type: project
---

A `missingness_auditor` Python package for the STAI-X Award C submission on branch `missingness-skill`.

**Why:** Award C requires a broadly useful statistical skill for autonomous data-science agents.

**Location:** `src/missingness_auditor/` — 8+ source modules, zero hardcoded column names.

**How to apply:** When the user asks about the missingness skill, remember it lives in `src/missingness_auditor/` (not `src/pattern_auditor/`) and tests are in `tests/missingness/`. The demo runs from `examples/missingness/run_demo.py`.

## Key design decisions (post 2026-06-08 refactor)

**Mechanism labels (cautious, evidence-based):**
- "MCAR-compatible" — no significant correlation found
- "MAR-like evidence" — correlates with observed numeric features
- "group-dependent missingness" — missingness concentrated in categorical groups (e.g. jurisdiction)
- "target-associated missingness" — correlates with target (replaces old "MNAR/structural concern")
- "structural absence concern" — set by planner when structural detector flags the column
- "high-cardinality text/category missingness" — >50 unique values or >20% unique ratio
- "insufficient evidence" — too few missing rows for reliable analysis

**Priority order when multiple signals present:**
1. Structural (from structural detector)
2. Group-dependent missingness (categorical group spread)
3. Target-associated (target correlation)
4. MAR-like evidence (numeric feature correlation)
5. MCAR-compatible

**Strategy names (planner.py):**
- `numeric_median` / `numeric_median_plus_indicator` — MCAR/MAR numeric
- `groupwise_numeric_median_plus_indicator` — group-dependent numeric (leakage-safe per-group median)
- `categorical_missing_token` / `categorical_missing_token_plus_indicator` — categorical (safe for any cardinality)
- `structural_none_token_plus_indicator` — categorical structural absence (fill "NONE")
- `structural_zero_plus_indicator` — numeric structural absence (fill 0, only with zero-companion evidence)
- `drop_column` — >80% missing

**Key rules:**
- Group-dependent missingness (e.g. temp missing by jurisdiction) is NOT structural absence
- Numeric weather-like columns must never receive structural_zero imputation without evidence
- High-cardinality text columns must never receive mode imputation (creates fake repeated values)
- Structural absence requires: categorical NA + companion numeric is 0/NaN co-occurrence

**Evidence fields in imputation_plan.json (per column):**
- missing_rate, dtype, cardinality, mechanism_label, target_association_evidence,
  covariate_association_evidence, group_dependency_evidence, structural_evidence, safety_warning

**Tests:** 233 total; `tests/missingness/` has 74 tests including 6 new ones:
- test_group_dependent_numeric_not_structural.py
- test_weather_like_numeric_not_zero_imputed.py
- test_high_cardinality_text_uses_missing_token.py
- test_structural_absence_detected_with_zero_companion.py
- test_structural_findings_deduplicated.py
- test_recommendation_contains_evidence_fields.py
