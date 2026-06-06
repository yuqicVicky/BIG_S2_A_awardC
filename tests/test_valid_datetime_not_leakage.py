"""Case F — datetime available in both train and predict must NOT be flagged as leakage.

A timestamp column that exists in both train and predict carries valid
prediction-time signal (e.g., hour-of-day demand cycles).  The auditor
must classify it as usable, not as a leakage risk.
"""

import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest
from pattern_auditor import PatternAuditor
from pattern_auditor.leakage import LeakageAuditor
from pattern_auditor.feature_types import FeatureTypeInferrer
from examples.make_leakage_demo import make_leakage_data, AUDITOR_KWARGS


def _run_full() -> dict:
    train_df, predict_df = make_leakage_data()
    with tempfile.TemporaryDirectory() as tmpdir:
        auditor = PatternAuditor(train_df, predict_df, **AUDITOR_KWARGS)
        return auditor.run(output_dir=tmpdir)


# ── Via full orchestrator ──────────────────────────────────────────────────

def test_timestamp_in_columns_to_use():
    """timestamp present in both sets → must not be excluded."""
    results = _run_full()
    fa = results["feature_availability"]
    assert "timestamp" not in fa["columns_to_exclude"], (
        "timestamp should NOT be in columns_to_exclude — it is prediction-time available"
    )


def test_timestamp_action_is_use():
    results = _run_full()
    col_info = results["feature_availability"]["columns"].get("timestamp", {})
    assert col_info.get("action") == "use", (
        f"timestamp action should be 'use', got: {col_info.get('action')!r}"
    )


def test_timestamp_leakage_risk_is_none():
    results = _run_full()
    col_info = results["feature_availability"]["columns"].get("timestamp", {})
    assert col_info.get("leakage_risk") == "none", (
        f"timestamp leakage_risk should be 'none', got: {col_info.get('leakage_risk')!r}"
    )


def test_timestamp_available_at_prediction():
    results = _run_full()
    col_info = results["feature_availability"]["columns"].get("timestamp", {})
    assert col_info.get("available_at_prediction") is True


# ── Via LeakageAuditor directly ────────────────────────────────────────────

def test_leakage_auditor_does_not_flag_shared_datetime():
    rng = np.random.default_rng(0)
    n = 200
    ts = pd.date_range("2023-01-01", periods=n, freq="h")
    train = pd.DataFrame({
        "timestamp": ts[:160],
        "value": rng.normal(size=160),
        "target": rng.normal(size=160),
    })
    predict = pd.DataFrame({
        "timestamp": ts[160:],
        "value": rng.normal(size=40),
    })
    fti = FeatureTypeInferrer(train, predict, target_col="target", datetime_col="timestamp")
    ft = fti.infer()
    la = LeakageAuditor(train, predict, ft, target_col="target")
    report = la.audit()
    risk_cols = {r["column"] for r in report["risks"]}
    assert "timestamp" not in risk_cols, (
        "LeakageAuditor must not flag a shared datetime column"
    )


def test_high_correlation_datetime_still_not_leakage():
    """Even when target strongly depends on hour, timestamp is not leakage."""
    rng = np.random.default_rng(42)
    ts = pd.date_range("2023-01-01", periods=300, freq="h")
    hour = ts.hour.astype(float)
    target = 10 * np.sin(2 * np.pi * hour / 24) + rng.normal(0, 0.5, 300)

    train = pd.DataFrame({
        "timestamp": ts[:240],
        "target": target[:240].round(3),
    })
    predict = pd.DataFrame({"timestamp": ts[240:]})

    fti = FeatureTypeInferrer(train, predict, target_col="target", datetime_col="timestamp")
    ft = fti.infer()
    la = LeakageAuditor(train, predict, ft, target_col="target")
    report = la.audit()
    risk_cols = {r["column"] for r in report["risks"]}
    assert "timestamp" not in risk_cols
