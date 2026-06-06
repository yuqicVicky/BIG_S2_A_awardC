"""Tests for validation strategy recommendation."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import pytest
from pattern_auditor.feature_types import FeatureTypeInferrer
from pattern_auditor.distribution_shift import DistributionShiftDetector
from pattern_auditor.validation import ValidationRecommender


def _run(train, predict, target_col=None, datetime_col=None, group_col=None):
    fti = FeatureTypeInferrer(train, predict, target_col=target_col,
                              datetime_col=datetime_col, group_col=group_col)
    ft = fti.infer()
    dsd = DistributionShiftDetector(train, predict, ft)
    shift = dsd.detect()
    vr = ValidationRecommender(train, predict, ft, shift, target_col=target_col)
    return vr.recommend()


def test_within_period_recommends_within_period_holdout():
    ts_all = pd.date_range("2023-01-01", periods=100, freq="h")
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"timestamp": ts_all, "val": rng.normal(size=100), "target": rng.normal(size=100)})
    train = df.iloc[:80].copy()
    predict = df.iloc[70:].drop(columns=["target"]).copy()
    rec = _run(train, predict, target_col="target", datetime_col="timestamp")
    assert rec["recommendation"]["strategy"] == "within_period_latest_available_holdout"
    assert "within_period" in rec["detected_split_pattern"]


def test_time_based_recommends_time_holdout():
    ts_train = pd.date_range("2023-01-01", periods=100, freq="D")
    ts_predict = pd.date_range("2023-04-11", periods=20, freq="D")
    rng = np.random.default_rng(1)
    train = pd.DataFrame({"timestamp": ts_train, "val": rng.normal(size=100), "target": rng.normal(size=100)})
    predict = pd.DataFrame({"timestamp": ts_predict, "val": rng.normal(size=20)})
    rec = _run(train, predict, target_col="target", datetime_col="timestamp")
    assert rec["recommendation"]["strategy"] == "time_holdout"
    assert rec["detected_split_pattern"] == "time_based_split"


def test_group_split_recommends_group_split():
    rng = np.random.default_rng(2)
    train_groups = [f"G{i:03d}" for i in range(75)]
    predict_groups = [f"G{i:03d}" for i in range(75, 100)]
    train = pd.DataFrame({
        "group_id": np.repeat(train_groups, 8),
        "val": rng.normal(size=600),
        "target": rng.normal(size=600),
    })
    predict = pd.DataFrame({
        "group_id": np.repeat(predict_groups, 8),
        "val": rng.normal(size=200),
    })
    rec = _run(train, predict, target_col="target", group_col="group_id")
    assert rec["recommendation"]["strategy"] == "group_split"


def test_iid_recommends_kfold():
    rng = np.random.default_rng(3)
    train = pd.DataFrame({"a": rng.normal(size=200), "b": rng.normal(size=200), "target": rng.normal(size=200)})
    predict = pd.DataFrame({"a": rng.normal(size=50), "b": rng.normal(size=50)})
    rec = _run(train, predict, target_col="target")
    assert rec["recommendation"]["strategy"] in ("kfold", "stratified_kfold")


def test_recommendation_has_all_required_fields():
    rng = np.random.default_rng(4)
    train = pd.DataFrame({"a": rng.normal(size=100), "target": rng.normal(size=100)})
    predict = pd.DataFrame({"a": rng.normal(size=20)})
    rec = _run(train, predict, target_col="target")
    r = rec["recommendation"]
    for key in ("strategy", "confidence", "reason", "fit_rule", "validation_rule", "strategies_to_avoid"):
        assert key in r, f"Missing key: {key}"


def test_why_risky_field_present():
    rng = np.random.default_rng(10)
    train = pd.DataFrame({"a": rng.normal(size=100), "target": rng.normal(size=100)})
    predict = pd.DataFrame({"a": rng.normal(size=20)})
    rec = _run(train, predict, target_col="target")
    assert "why_generic_validation_is_risky" in rec
    assert len(rec["why_generic_validation_is_risky"]) > 10


def test_time_based_has_rolling_alternative():
    ts_train = pd.date_range("2023-01-01", periods=100, freq="D")
    ts_predict = pd.date_range("2023-04-11", periods=20, freq="D")
    rng = np.random.default_rng(5)
    train = pd.DataFrame({"timestamp": ts_train, "val": rng.normal(size=100), "target": rng.normal(size=100)})
    predict = pd.DataFrame({"timestamp": ts_predict, "val": rng.normal(size=20)})
    rec = _run(train, predict, target_col="target", datetime_col="timestamp")
    alt_strategies = [a["strategy"] for a in rec.get("alternatives", [])]
    assert "rolling_split" in alt_strategies
