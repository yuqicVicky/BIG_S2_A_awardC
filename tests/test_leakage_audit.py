"""Tests for leakage and feature availability audit."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import pytest
from pattern_auditor.feature_types import FeatureTypeInferrer
from pattern_auditor.leakage import LeakageAuditor


def _ft(train, predict=None, target_col=None, datetime_col=None):
    fti = FeatureTypeInferrer(train, predict, target_col=target_col, datetime_col=datetime_col)
    return fti.infer()


def test_train_only_column_flagged():
    rng = np.random.default_rng(0)
    train = pd.DataFrame({
        "feature_a": rng.normal(size=100),
        "train_only_secret": rng.normal(size=100),  # not in predict
        "target": rng.normal(size=100),
    })
    predict = pd.DataFrame({
        "feature_a": rng.normal(size=20),
    })
    ft = _ft(train, predict, target_col="target")
    la = LeakageAuditor(train, predict, ft, target_col="target")
    report = la.audit()
    risk_cols = {r["column"] for r in report["risks"]}
    assert "train_only_secret" in risk_cols
    train_only_risks = [r for r in report["risks"] if r["column"] == "train_only_secret"]
    assert train_only_risks[0]["severity"] == "high"


def test_target_component_suspect_flagged():
    rng = np.random.default_rng(1)
    n = 100
    target = rng.normal(size=n)
    train = pd.DataFrame({
        "feature_a": rng.normal(size=n),
        "target_shifted": target + 0.01 * rng.normal(size=n),  # suspiciously named
        "target": target,
    })
    ft = _ft(train, target_col="target")
    la = LeakageAuditor(train, None, ft, target_col="target")
    report = la.audit()
    risk_cols = [r["column"] for r in report["risks"] if r["risk_type"] == "target_component_suspect"]
    assert "target_shifted" in risk_cols


def test_datetime_derived_not_flagged_as_leakage():
    """Datetime features available at prediction time must NOT be leakage."""
    rng = np.random.default_rng(2)
    n = 200
    ts = pd.date_range("2023-01-01", periods=n, freq="h")
    train = pd.DataFrame({
        "timestamp": ts,
        "value": rng.normal(size=n),
        "target": rng.normal(size=n),
    })
    predict = pd.DataFrame({
        "timestamp": ts[:50],
        "value": rng.normal(size=50),
    })
    ft = _ft(train, predict, target_col="target", datetime_col="timestamp")
    la = LeakageAuditor(train, predict, ft, target_col="target")
    report = la.audit()
    leakage_cols = {r["column"] for r in report["risks"]}
    assert "timestamp" not in leakage_cols, "timestamp should NOT be flagged as leakage"


def test_no_false_positives_on_normal_numeric():
    rng = np.random.default_rng(3)
    n = 200
    train = pd.DataFrame({
        "x": rng.normal(size=n),
        "y": rng.normal(size=n),
        "target": rng.normal(size=n),
    })
    predict = pd.DataFrame({
        "x": rng.normal(size=50),
        "y": rng.normal(size=50),
    })
    ft = _ft(train, predict, target_col="target")
    la = LeakageAuditor(train, predict, ft, target_col="target")
    report = la.audit()
    high_risks = [r for r in report["risks"] if r["severity"] == "high"]
    assert len(high_risks) == 0


def test_summary_counts_correct():
    rng = np.random.default_rng(4)
    n = 100
    train = pd.DataFrame({
        "feature": rng.normal(size=n),
        "train_secret": rng.normal(size=n),
        "target_proxy": rng.normal(size=n),
        "target": rng.normal(size=n),
    })
    predict = pd.DataFrame({"feature": rng.normal(size=20)})
    ft = _ft(train, predict, target_col="target")
    la = LeakageAuditor(train, predict, ft, target_col="target")
    report = la.audit()
    total = sum(report["summary"].values())
    assert total == len(report["risks"])
