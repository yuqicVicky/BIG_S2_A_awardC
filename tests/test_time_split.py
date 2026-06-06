"""Case D — future time-split detection and validation recommendation."""

import sys, os, tempfile, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from pattern_auditor import PatternAuditor
from examples.make_time_split_demo import make_time_split_data, AUDITOR_KWARGS


def _run() -> dict:
    train_df, predict_df = make_time_split_data()
    with tempfile.TemporaryDirectory() as tmpdir:
        auditor = PatternAuditor(train_df, predict_df, **AUDITOR_KWARGS)
        return auditor.run(output_dir=tmpdir)


# ── Pattern detection ──────────────────────────────────────────────────────

def test_time_based_split_pattern_detected():
    results = _run()
    pattern = results["train_prediction_pattern"]["pattern"]
    assert pattern == "time_based_split", (
        f"Expected time_based_split, got: {pattern!r}"
    )


def test_evidence_mentions_predict_after_train():
    results = _run()
    evidence = results["train_prediction_pattern"]["evidence"]
    evidence_text = " ".join(evidence).lower()
    assert "after" in evidence_text or "time" in evidence_text


def test_datetime_coverage_shows_no_overlap():
    results = _run()
    dt_coverage = results["train_prediction_pattern"]["datetime_coverage"]
    assert len(dt_coverage) >= 1
    for cov in dt_coverage:
        if "predict_after_train" in cov:
            assert cov["predict_after_train"] is True


# ── Validation recommendation ──────────────────────────────────────────────

def test_strategy_is_time_holdout():
    results = _run()
    strategy = results["validation"]["recommendation"]["strategy"]
    assert strategy == "time_holdout", (
        f"Expected time_holdout, got: {strategy!r}"
    )


def test_confidence_is_high():
    results = _run()
    assert results["validation"]["recommendation"]["confidence"] == "high"


def test_avoid_includes_kfold_and_random():
    results = _run()
    avoid = results["validation"]["recommendation"]["strategies_to_avoid"]
    assert "kfold" in avoid, f"kfold must be in strategies_to_avoid; got {avoid}"
    assert "random_holdout" in avoid, f"random_holdout must be in strategies_to_avoid"


def test_rolling_split_in_alternatives():
    results = _run()
    alt_strategies = [a["strategy"] for a in results["validation"].get("alternatives", [])]
    assert "rolling_split" in alt_strategies


# ── Leakage ────────────────────────────────────────────────────────────────

def test_no_false_high_leakage():
    """Clean future-time dataset: only target should be missing from predict."""
    results = _run()
    high_risks = [
        col for col, info in results["feature_availability"]["columns"].items()
        if info["leakage_risk"] == "high"
    ]
    assert len(high_risks) == 0, (
        f"No high-leakage columns expected in clean time-split data; got: {high_risks}"
    )


def test_date_col_is_usable():
    results = _run()
    fa = results["feature_availability"]
    assert "date" in fa["columns_to_use"], "date column should be available at prediction"


# ── Output files ───────────────────────────────────────────────────────────

def test_all_output_files_written():
    train_df, predict_df = make_time_split_data()
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
