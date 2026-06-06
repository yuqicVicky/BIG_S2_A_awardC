"""Case A — within-period split detection and validation recommendation."""

import sys, os, tempfile, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from pattern_auditor import PatternAuditor
from examples.make_within_period_demo import make_within_period_data, AUDITOR_KWARGS


def _run(n_entities=6, n_months=3) -> dict:
    train_df, predict_df = make_within_period_data(n_entities=n_entities, n_months=n_months)
    with tempfile.TemporaryDirectory() as tmpdir:
        auditor = PatternAuditor(train_df, predict_df, **AUDITOR_KWARGS)
        return auditor.run(output_dir=tmpdir)


# ── Pattern detection ──────────────────────────────────────────────────────

def test_within_period_pattern_detected():
    results = _run()
    pattern = results["train_prediction_pattern"]["pattern"]
    assert "within_period" in pattern, (
        f"Expected 'within_period' in pattern, got: {pattern!r}"
    )


def test_evidence_mentions_overlap():
    results = _run()
    evidence = results["train_prediction_pattern"]["evidence"]
    assert len(evidence) >= 1, "Expected at least one evidence entry"
    evidence_text = " ".join(evidence).lower()
    assert "within" in evidence_text or "overlap" in evidence_text or "precedes" in evidence_text


# ── Validation recommendation ──────────────────────────────────────────────

def test_strategy_is_within_period_holdout():
    results = _run()
    strategy = results["validation"]["recommendation"]["strategy"]
    assert strategy == "within_period_latest_available_holdout", (
        f"Expected within_period_latest_available_holdout, got: {strategy!r}"
    )


def test_confidence_is_high():
    results = _run()
    confidence = results["validation"]["recommendation"]["confidence"]
    assert confidence == "high"


def test_avoid_includes_time_holdout():
    """time_holdout is inappropriate for within-period data."""
    results = _run()
    avoid = results["validation"]["recommendation"]["strategies_to_avoid"]
    assert "time_holdout" in avoid, (
        f"time_holdout should be in strategies_to_avoid; got: {avoid}"
    )


def test_why_risky_is_non_empty():
    results = _run()
    why = results["validation"]["why_generic_validation_is_risky"]
    assert len(why) > 20


# ── Feature availability ───────────────────────────────────────────────────

def test_demand_so_far_excluded():
    """Train-only column must be flagged as unavailable and excluded."""
    results = _run()
    fa = results["feature_availability"]
    assert "demand_so_far" in fa["columns_to_exclude"], (
        "demand_so_far (train-only) must be in columns_to_exclude"
    )


def test_demand_so_far_leakage_risk_high():
    results = _run()
    col_info = results["feature_availability"]["columns"]["demand_so_far"]
    assert col_info["leakage_risk"] == "high"
    assert col_info["available_at_prediction"] is False


def test_numeric_feat_is_usable():
    results = _run()
    fa = results["feature_availability"]
    assert "numeric_feat" in fa["columns_to_use"], (
        "numeric_feat (shared column) should be in columns_to_use"
    )


# ── Output files ───────────────────────────────────────────────────────────

def test_all_output_files_written():
    train_df, predict_df = make_within_period_data(n_entities=4, n_months=2)
    with tempfile.TemporaryDirectory() as tmpdir:
        PatternAuditor(train_df, predict_df, **AUDITOR_KWARGS).run(output_dir=tmpdir)
        required = [
            "logs/schema_audit.json",
            "logs/feature_availability_audit.json",
            "logs/train_prediction_pattern.json",
            "logs/leakage_audit.json",
            "logs/validation_recommendation.json",
            "reports/pattern_audit.md",
            "figures/train_prediction_coverage.png",
            "figures/feature_availability.png",
            "figures/distribution_shift_summary.png",
        ]
        for rel in required:
            assert os.path.exists(os.path.join(tmpdir, rel)), f"Missing: {rel}"


def test_validation_json_machine_readable():
    train_df, predict_df = make_within_period_data(n_entities=4, n_months=2)
    with tempfile.TemporaryDirectory() as tmpdir:
        PatternAuditor(train_df, predict_df, **AUDITOR_KWARGS).run(output_dir=tmpdir)
        with open(os.path.join(tmpdir, "logs", "validation_recommendation.json")) as f:
            val = json.load(f)
        assert "recommendation" in val
        assert "strategy" in val["recommendation"]
