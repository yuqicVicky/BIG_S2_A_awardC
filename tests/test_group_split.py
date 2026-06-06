"""Case C — unseen-group split detection and validation recommendation."""

import sys, os, tempfile, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from pattern_auditor import PatternAuditor
from examples.make_group_demo import make_group_data, AUDITOR_KWARGS


def _run() -> dict:
    train_df, predict_df = make_group_data()
    with tempfile.TemporaryDirectory() as tmpdir:
        auditor = PatternAuditor(train_df, predict_df, **AUDITOR_KWARGS)
        return auditor.run(output_dir=tmpdir)


# ── Pattern detection ──────────────────────────────────────────────────────

def test_group_based_split_pattern_detected():
    results = _run()
    pattern = results["train_prediction_pattern"]["pattern"]
    assert pattern == "group_based_split", (
        f"Expected group_based_split, got: {pattern!r}"
    )


def test_evidence_mentions_low_group_overlap():
    results = _run()
    evidence = results["train_prediction_pattern"]["evidence"]
    evidence_text = " ".join(evidence).lower()
    assert "group" in evidence_text or "overlap" in evidence_text


def test_no_time_based_evidence():
    """No datetime column in this case — should not trigger time-based detection."""
    results = _run()
    dt_coverage = results["train_prediction_pattern"]["datetime_coverage"]
    assert len(dt_coverage) == 0


# ── Validation recommendation ──────────────────────────────────────────────

def test_strategy_is_group_split():
    results = _run()
    strategy = results["validation"]["recommendation"]["strategy"]
    assert strategy == "group_split", (
        f"Expected group_split, got: {strategy!r}"
    )


def test_confidence_is_high():
    results = _run()
    assert results["validation"]["recommendation"]["confidence"] == "high"


def test_avoid_includes_kfold_and_random():
    results = _run()
    avoid = results["validation"]["recommendation"]["strategies_to_avoid"]
    assert "kfold" in avoid, f"kfold must be in strategies_to_avoid; got {avoid}"
    assert "random_holdout" in avoid


# ── Feature availability ───────────────────────────────────────────────────

def test_shared_features_are_usable():
    results = _run()
    fa = results["feature_availability"]
    for col in ("sales_volume", "promo_flag", "store_id"):
        assert col in fa["columns_to_use"] or col in fa["columns_to_verify"], (
            f"{col} should be usable or in verify list"
        )


def test_no_high_leakage_in_clean_group_data():
    results = _run()
    high_risks = [
        col for col, info in results["feature_availability"]["columns"].items()
        if info["leakage_risk"] == "high"
    ]
    assert len(high_risks) == 0, f"No high leakage expected; got: {high_risks}"


# ── Output files ───────────────────────────────────────────────────────────

def test_all_output_files_written():
    train_df, predict_df = make_group_data()
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
