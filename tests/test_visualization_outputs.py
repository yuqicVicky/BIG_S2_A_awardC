"""Tests that visualization figures are generated and are valid PNG files."""

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
        auditor = PatternAuditor(
            train, predict,
            target_col="target",
            datetime_col="timestamp",
        )
        auditor.run(output_dir=tmpdir)
        fig_dir = os.path.join(tmpdir, "figures")
        files = os.listdir(fig_dir)
        assert len(files) >= 5, f"Expected >=5 figures, got {files}"


def test_required_figures_present():
    train, predict = _minimal_within_period()
    with tempfile.TemporaryDirectory() as tmpdir:
        auditor = PatternAuditor(
            train, predict,
            target_col="target",
            datetime_col="timestamp",
        )
        auditor.run(output_dir=tmpdir)
        fig_dir = os.path.join(tmpdir, "figures")
        files = set(os.listdir(fig_dir))
        required = {
            "train_prediction_coverage.png",
            "target_distribution.png",
            "feature_target_signal.png",
            "train_prediction_shift.png",
            "leakage_feature_availability.png",
            "target_by_time_pattern.png",
        }
        missing = required - files
        assert not missing, f"Missing figures: {missing}"


def test_figures_are_nonempty_png():
    train, predict = _minimal_within_period()
    with tempfile.TemporaryDirectory() as tmpdir:
        auditor = PatternAuditor(
            train, predict,
            target_col="target",
            datetime_col="timestamp",
        )
        auditor.run(output_dir=tmpdir)
        fig_dir = os.path.join(tmpdir, "figures")
        for fname in os.listdir(fig_dir):
            path = os.path.join(fig_dir, fname)
            assert os.path.getsize(path) > 1000, f"{fname} is suspiciously small"
            with open(path, "rb") as f:
                header = f.read(8)
            # PNG magic bytes
            assert header[:4] == b"\x89PNG", f"{fname} is not a valid PNG"


def test_hour_dayofweek_heatmap_generated_with_hourly_data():
    train, predict = _minimal_within_period()
    with tempfile.TemporaryDirectory() as tmpdir:
        auditor = PatternAuditor(
            train, predict,
            target_col="target",
            datetime_col="timestamp",
        )
        auditor.run(output_dir=tmpdir)
        fig_dir = os.path.join(tmpdir, "figures")
        assert "hour_by_dayofweek_heatmap.png" in os.listdir(fig_dir)


def test_json_logs_written():
    train, predict = _minimal_within_period()
    with tempfile.TemporaryDirectory() as tmpdir:
        auditor = PatternAuditor(
            train, predict,
            target_col="target",
            datetime_col="timestamp",
        )
        auditor.run(output_dir=tmpdir)
        logs_dir = os.path.join(tmpdir, "logs")
        expected = [
            "schema_report.json",
            "feature_type_report.json",
            "train_prediction_pattern.json",
            "target_pattern_report.json",
            "distribution_shift_report.json",
            "leakage_feature_audit.json",
            "validation_recommendation.json",
            "feature_recommendation.json",
        ]
        for fname in expected:
            assert fname in os.listdir(logs_dir), f"Missing log: {fname}"


def test_markdown_report_written():
    train, predict = _minimal_within_period()
    with tempfile.TemporaryDirectory() as tmpdir:
        auditor = PatternAuditor(
            train, predict,
            target_col="target",
            datetime_col="timestamp",
        )
        auditor.run(output_dir=tmpdir)
        report_path = os.path.join(tmpdir, "reports", "pattern_audit.md")
        assert os.path.exists(report_path)
        with open(report_path) as f:
            content = f.read()
        assert "Validation Recommendation" in content
        assert "Leakage" in content
