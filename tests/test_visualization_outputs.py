"""Tests that visualization figures and JSON logs are generated correctly."""

import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import pytest
from pattern_auditor import PatternAuditor


def _minimal_within_period():
    rng = np.random.default_rng(7)
    ts = pd.date_range("2023-01-01", periods=100, freq="h")
    df = pd.DataFrame({
        "timestamp": ts,
        "numeric_feat": rng.normal(size=100),
        "category": rng.choice(["A", "B"], 100),
        "target": rng.normal(size=100),
    })
    train = df.iloc[:80].copy()
    predict = df.iloc[70:].drop(columns=["target"]).copy()
    return train, predict


def test_figures_generated():
    train, predict = _minimal_within_period()
    with tempfile.TemporaryDirectory() as tmpdir:
        PatternAuditor(train, predict, target_col="target", datetime_col="timestamp").run(output_dir=tmpdir)
        files = os.listdir(os.path.join(tmpdir, "figures"))
        assert len(files) >= 3, f"Expected >=3 figures, got {files}"


def test_required_figures_present():
    train, predict = _minimal_within_period()
    with tempfile.TemporaryDirectory() as tmpdir:
        PatternAuditor(train, predict, target_col="target", datetime_col="timestamp").run(output_dir=tmpdir)
        files = set(os.listdir(os.path.join(tmpdir, "figures")))
        required = {
            "train_prediction_coverage.png",
            "feature_availability.png",
            "distribution_shift_summary.png",
        }
        missing = required - files
        assert not missing, f"Missing figures: {missing}"


def test_figures_are_nonempty_png():
    train, predict = _minimal_within_period()
    with tempfile.TemporaryDirectory() as tmpdir:
        PatternAuditor(train, predict, target_col="target", datetime_col="timestamp").run(output_dir=tmpdir)
        fig_dir = os.path.join(tmpdir, "figures")
        for fname in os.listdir(fig_dir):
            path = os.path.join(fig_dir, fname)
            assert os.path.getsize(path) > 1000, f"{fname} is suspiciously small"
            with open(path, "rb") as f:
                header = f.read(8)
            assert header[:4] == b"\x89PNG", f"{fname} is not a valid PNG"


def test_json_logs_written():
    train, predict = _minimal_within_period()
    with tempfile.TemporaryDirectory() as tmpdir:
        PatternAuditor(train, predict, target_col="target", datetime_col="timestamp").run(output_dir=tmpdir)
        logs_dir = os.path.join(tmpdir, "logs")
        expected = [
            "schema_audit.json",
            "feature_availability_audit.json",
            "train_prediction_pattern.json",
            "leakage_audit.json",
            "validation_recommendation.json",
        ]
        for fname in expected:
            assert fname in os.listdir(logs_dir), f"Missing log: {fname}"


def test_markdown_report_written():
    train, predict = _minimal_within_period()
    with tempfile.TemporaryDirectory() as tmpdir:
        PatternAuditor(train, predict, target_col="target", datetime_col="timestamp").run(output_dir=tmpdir)
        report_path = os.path.join(tmpdir, "reports", "pattern_audit.md")
        assert os.path.exists(report_path)
        with open(report_path) as f:
            content = f.read()
        assert "Leakage" in content
        assert "Validation" in content


def test_markdown_narrative_tells_story():
    train, predict = _minimal_within_period()
    with tempfile.TemporaryDirectory() as tmpdir:
        PatternAuditor(train, predict, target_col="target", datetime_col="timestamp").run(output_dir=tmpdir)
        with open(os.path.join(tmpdir, "reports", "pattern_audit.md")) as f:
            content = f.read()
        assert "Train/Prediction Pattern" in content
        assert "Generic Validation Is Risky" in content
        assert "Recommended Validation Strategy" in content
        assert "Feature Availability" in content
