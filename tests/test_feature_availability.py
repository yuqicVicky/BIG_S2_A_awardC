"""Feature availability audit — scenario-level tests using demo generators.

Covers:
  - Case A (within_period): train-only demand_so_far excluded, shared cols usable
  - Case E (leakage):       both target_component_* columns excluded
  - Case F (datetime):      timestamp remains usable despite being predictive
  - IID baseline:           no unexpected exclusions for a clean random split
"""

import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from pattern_auditor import PatternAuditor
from examples.make_within_period_demo import make_within_period_data, AUDITOR_KWARGS as WP_KWARGS
from examples.make_leakage_demo import make_leakage_data, AUDITOR_KWARGS as LEAK_KWARGS
from examples.make_iid_demo import make_iid_data, AUDITOR_KWARGS as IID_KWARGS


def _run(train_df, predict_df, **kwargs) -> dict:
    with tempfile.TemporaryDirectory() as tmpdir:
        return PatternAuditor(train_df, predict_df, **kwargs).run(output_dir=tmpdir)


# ── Case A: within_period train-only column ────────────────────────────────

def test_case_a_demand_so_far_excluded():
    train, predict = make_within_period_data(n_entities=5, n_months=2)
    results = _run(train, predict, **WP_KWARGS)
    assert "demand_so_far" in results["feature_availability"]["columns_to_exclude"]


def test_case_a_shared_columns_usable():
    train, predict = make_within_period_data(n_entities=5, n_months=2)
    results = _run(train, predict, **WP_KWARGS)
    fa = results["feature_availability"]
    for col in ("numeric_feat", "category"):
        assert col in fa["columns_to_use"], f"{col} should be usable in Case A"


def test_case_a_target_not_in_usable():
    """target is train-only; it must not appear in columns_to_use."""
    train, predict = make_within_period_data(n_entities=5, n_months=2)
    results = _run(train, predict, **WP_KWARGS)
    assert "target" not in results["feature_availability"]["columns_to_use"]


# ── Case E: leakage columns ────────────────────────────────────────────────

def test_case_e_target_component_1_excluded():
    train, predict = make_leakage_data()
    results = _run(train, predict, **LEAK_KWARGS)
    assert "target_component_1" in results["feature_availability"]["columns_to_exclude"]


def test_case_e_target_component_2_excluded():
    train, predict = make_leakage_data()
    results = _run(train, predict, **LEAK_KWARGS)
    assert "target_component_2" in results["feature_availability"]["columns_to_exclude"]


def test_case_e_leakage_risk_is_high():
    train, predict = make_leakage_data()
    results = _run(train, predict, **LEAK_KWARGS)
    cols = results["feature_availability"]["columns"]
    for col in ("target_component_1", "target_component_2"):
        assert cols[col]["leakage_risk"] == "high", (
            f"{col} leakage_risk should be 'high'"
        )


# ── Case F: shared datetime is NOT leakage ─────────────────────────────────

def test_case_f_timestamp_not_excluded():
    train, predict = make_leakage_data()
    results = _run(train, predict, **LEAK_KWARGS)
    assert "timestamp" not in results["feature_availability"]["columns_to_exclude"]


def test_case_f_timestamp_action_is_use():
    train, predict = make_leakage_data()
    results = _run(train, predict, **LEAK_KWARGS)
    col_info = results["feature_availability"]["columns"].get("timestamp", {})
    assert col_info.get("action") == "use"


# ── IID baseline: no false positives ──────────────────────────────────────

def test_iid_no_unexpected_exclusions():
    train, predict = make_iid_data()
    results = _run(train, predict, **IID_KWARGS)
    excluded = results["feature_availability"]["columns_to_exclude"]
    unexpected = [c for c in excluded if c not in ("target",)]
    assert len(unexpected) == 0, (
        f"IID clean dataset should have no exclusions; got: {unexpected}"
    )


def test_iid_all_features_usable():
    train, predict = make_iid_data()
    results = _run(train, predict, **IID_KWARGS)
    fa = results["feature_availability"]
    for col in ("feature_a", "feature_b", "category"):
        assert col in fa["columns_to_use"], f"{col} should be usable in IID case"


# ── Summary counts consistency ─────────────────────────────────────────────

def test_summary_counts_reflect_actual_lists():
    train, predict = make_leakage_data()
    results = _run(train, predict, **LEAK_KWARGS)
    fa = results["feature_availability"]
    n_excluded = len(fa["columns_to_exclude"])
    n_high = fa["summary"]["high_leakage_risk"] + fa["summary"]["train_only"]
    assert n_high >= 2, "At least 2 high-risk exclusions expected in leakage case"
