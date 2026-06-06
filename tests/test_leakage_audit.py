"""Tests for leakage and feature availability audit — unit + demo-case scenarios."""

import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest
from pattern_auditor.feature_types import FeatureTypeInferrer
from pattern_auditor.leakage import LeakageAuditor
from pattern_auditor import PatternAuditor


def _ft(train, predict=None, target_col=None, datetime_col=None, group_col=None):
    fti = FeatureTypeInferrer(train, predict, target_col=target_col,
                              datetime_col=datetime_col, group_col=group_col)
    return fti.infer()


# ── Basic leakage detection ────────────────────────────────────────────────

def test_train_only_column_flagged():
    rng = np.random.default_rng(0)
    train = pd.DataFrame({
        "feature_a": rng.normal(size=100),
        "train_only_secret": rng.normal(size=100),
        "target": rng.normal(size=100),
    })
    predict = pd.DataFrame({"feature_a": rng.normal(size=20)})
    ft = _ft(train, predict, target_col="target")
    la = LeakageAuditor(train, predict, ft, target_col="target")
    report = la.audit()
    risk_cols = {r["column"] for r in report["risks"]}
    assert "train_only_secret" in risk_cols
    assert [r for r in report["risks"] if r["column"] == "train_only_secret"][0]["severity"] == "high"


def test_target_component_suspect_flagged():
    rng = np.random.default_rng(1)
    n = 100
    target = rng.normal(size=n)
    train = pd.DataFrame({
        "feature_a": rng.normal(size=n),
        "target_shifted": target + 0.01 * rng.normal(size=n),
        "target": target,
    })
    ft = _ft(train, target_col="target")
    la = LeakageAuditor(train, None, ft, target_col="target")
    report = la.audit()
    risk_cols = [r["column"] for r in report["risks"] if r["risk_type"] == "target_component_suspect"]
    assert "target_shifted" in risk_cols


def test_datetime_derived_not_flagged_as_leakage():
    rng = np.random.default_rng(2)
    n = 200
    ts = pd.date_range("2023-01-01", periods=n, freq="h")
    train = pd.DataFrame({"timestamp": ts, "value": rng.normal(size=n), "target": rng.normal(size=n)})
    predict = pd.DataFrame({"timestamp": ts[:50], "value": rng.normal(size=50)})
    ft = _ft(train, predict, target_col="target", datetime_col="timestamp")
    la = LeakageAuditor(train, predict, ft, target_col="target")
    report = la.audit()
    assert "timestamp" not in {r["column"] for r in report["risks"]}


def test_no_false_positives_on_normal_numeric():
    rng = np.random.default_rng(3)
    n = 200
    train = pd.DataFrame({"x": rng.normal(size=n), "y": rng.normal(size=n), "target": rng.normal(size=n)})
    predict = pd.DataFrame({"x": rng.normal(size=50), "y": rng.normal(size=50)})
    ft = _ft(train, predict, target_col="target")
    la = LeakageAuditor(train, predict, ft, target_col="target")
    report = la.audit()
    assert len([r for r in report["risks"] if r["severity"] == "high"]) == 0


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
    assert sum(report["summary"].values()) == len(report["risks"])


# ── Case E: target_component columns ──────────────────────────────────────

def test_case_e_both_components_flagged():
    """target_component_1 and _2: train-only + name pattern → both high severity."""
    from examples.make_leakage_demo import make_leakage_data
    train_df, predict_df = make_leakage_data()
    ft = _ft(train_df, predict_df, target_col="target", datetime_col="timestamp")
    la = LeakageAuditor(train_df, predict_df, ft, target_col="target")
    report = la.audit()
    risk_cols = {r["column"] for r in report["risks"] if r["severity"] == "high"}
    assert "target_component_1" in risk_cols, "target_component_1 must be high severity"
    assert "target_component_2" in risk_cols, "target_component_2 must be high severity"


def test_case_e_high_risk_count():
    from examples.make_leakage_demo import make_leakage_data
    train_df, predict_df = make_leakage_data()
    with tempfile.TemporaryDirectory() as tmpdir:
        results = PatternAuditor(train_df, predict_df,
                                 target_col="target", datetime_col="timestamp").run(output_dir=tmpdir)
    assert results["leakage"]["summary"]["high"] >= 2


def test_case_e_timestamp_not_in_risks():
    from examples.make_leakage_demo import make_leakage_data
    train_df, predict_df = make_leakage_data()
    ft = _ft(train_df, predict_df, target_col="target", datetime_col="timestamp")
    la = LeakageAuditor(train_df, predict_df, ft, target_col="target")
    report = la.audit()
    assert "timestamp" not in {r["column"] for r in report["risks"]}


# ── Case A: train-only demand_so_far ──────────────────────────────────────

def test_case_a_demand_so_far_is_high_risk():
    from examples.make_within_period_demo import make_within_period_data
    train_df, predict_df = make_within_period_data(n_entities=5, n_months=2)
    ft = _ft(train_df, predict_df, target_col="target",
             datetime_col="timestamp", group_col="entity_id")
    la = LeakageAuditor(train_df, predict_df, ft, target_col="target")
    report = la.audit()
    high_cols = {r["column"] for r in report["risks"] if r["severity"] == "high"}
    assert "demand_so_far" in high_cols
