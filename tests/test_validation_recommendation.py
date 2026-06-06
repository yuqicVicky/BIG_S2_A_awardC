"""Validation strategy recommender tests — unit + all five demo cases."""

import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest
from pattern_auditor.feature_types import FeatureTypeInferrer
from pattern_auditor.distribution_shift import DistributionShiftDetector
from pattern_auditor.validation import ValidationRecommender
from pattern_auditor import PatternAuditor


def _recommend(train, predict, target_col=None, datetime_col=None, group_col=None):
    fti = FeatureTypeInferrer(train, predict, target_col=target_col,
                              datetime_col=datetime_col, group_col=group_col)
    ft = fti.infer()
    dsd = DistributionShiftDetector(train, predict, ft)
    shift = dsd.detect()
    vr = ValidationRecommender(train, predict, ft, shift, target_col=target_col)
    return vr.recommend()


def _run_full(train, predict, **kwargs) -> dict:
    with tempfile.TemporaryDirectory() as tmpdir:
        return PatternAuditor(train, predict, **kwargs).run(output_dir=tmpdir)


# ── All required fields present ────────────────────────────────────────────

def test_recommendation_has_all_required_fields():
    rng = np.random.default_rng(0)
    train = pd.DataFrame({"a": rng.normal(size=100), "target": rng.normal(size=100)})
    predict = pd.DataFrame({"a": rng.normal(size=20)})
    rec = _recommend(train, predict, target_col="target")
    for key in ("strategy", "confidence", "reason", "fit_rule",
                "validation_rule", "strategies_to_avoid"):
        assert key in rec["recommendation"], f"Missing field: {key}"


def test_why_risky_field_present_and_non_empty():
    rng = np.random.default_rng(1)
    train = pd.DataFrame({"a": rng.normal(size=100), "target": rng.normal(size=100)})
    predict = pd.DataFrame({"a": rng.normal(size=20)})
    rec = _recommend(train, predict, target_col="target")
    assert "why_generic_validation_is_risky" in rec
    assert len(rec["why_generic_validation_is_risky"]) > 10


# ── Unit-level: each pattern maps to correct strategy ─────────────────────

def test_within_period_recommends_within_period_holdout():
    ts_all = pd.date_range("2023-01-01", periods=100, freq="h")
    rng = np.random.default_rng(10)
    df = pd.DataFrame({"timestamp": ts_all, "val": rng.normal(size=100), "target": rng.normal(size=100)})
    train = df.iloc[:80].copy()
    predict = df.iloc[70:].drop(columns=["target"]).copy()
    rec = _recommend(train, predict, target_col="target", datetime_col="timestamp")
    assert rec["recommendation"]["strategy"] == "within_period_latest_available_holdout"
    assert "within_period" in rec["detected_split_pattern"]


def test_time_based_recommends_time_holdout():
    ts_train = pd.date_range("2023-01-01", periods=100, freq="D")
    ts_predict = pd.date_range("2023-04-11", periods=20, freq="D")
    rng = np.random.default_rng(11)
    train = pd.DataFrame({"timestamp": ts_train, "val": rng.normal(size=100), "target": rng.normal(size=100)})
    predict = pd.DataFrame({"timestamp": ts_predict, "val": rng.normal(size=20)})
    rec = _recommend(train, predict, target_col="target", datetime_col="timestamp")
    assert rec["recommendation"]["strategy"] == "time_holdout"
    assert rec["detected_split_pattern"] == "time_based_split"


def test_group_split_recommends_group_split():
    rng = np.random.default_rng(12)
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
    rec = _recommend(train, predict, target_col="target", group_col="group_id")
    assert rec["recommendation"]["strategy"] == "group_split"


def test_iid_recommends_kfold():
    rng = np.random.default_rng(13)
    train = pd.DataFrame({"a": rng.normal(size=200), "b": rng.normal(size=200), "target": rng.normal(size=200)})
    predict = pd.DataFrame({"a": rng.normal(size=50), "b": rng.normal(size=50)})
    rec = _recommend(train, predict, target_col="target")
    assert rec["recommendation"]["strategy"] in ("kfold", "stratified_kfold")


def test_time_based_has_rolling_alternative():
    ts_train = pd.date_range("2023-01-01", periods=100, freq="D")
    ts_predict = pd.date_range("2023-04-11", periods=20, freq="D")
    rng = np.random.default_rng(14)
    train = pd.DataFrame({"timestamp": ts_train, "val": rng.normal(size=100), "target": rng.normal(size=100)})
    predict = pd.DataFrame({"timestamp": ts_predict, "val": rng.normal(size=20)})
    rec = _recommend(train, predict, target_col="target", datetime_col="timestamp")
    assert "rolling_split" in [a["strategy"] for a in rec.get("alternatives", [])]


# ── Demo-case end-to-end via full orchestrator ─────────────────────────────

def test_case_a_full_recommendation():
    from examples.make_within_period_demo import make_within_period_data, AUDITOR_KWARGS
    train, predict = make_within_period_data(n_entities=5, n_months=2)
    results = _run_full(train, predict, **AUDITOR_KWARGS)
    assert results["validation"]["recommendation"]["strategy"] == "within_period_latest_available_holdout"


def test_case_b_full_recommendation():
    from examples.make_iid_demo import make_iid_data, AUDITOR_KWARGS
    train, predict = make_iid_data()
    results = _run_full(train, predict, **AUDITOR_KWARGS)
    assert results["validation"]["recommendation"]["strategy"] in ("kfold", "stratified_kfold")


def test_case_c_full_recommendation():
    from examples.make_group_demo import make_group_data, AUDITOR_KWARGS
    train, predict = make_group_data()
    results = _run_full(train, predict, **AUDITOR_KWARGS)
    assert results["validation"]["recommendation"]["strategy"] == "group_split"


def test_case_d_full_recommendation():
    from examples.make_time_split_demo import make_time_split_data, AUDITOR_KWARGS
    train, predict = make_time_split_data()
    results = _run_full(train, predict, **AUDITOR_KWARGS)
    assert results["validation"]["recommendation"]["strategy"] == "time_holdout"


# ── Avoid lists correct per case ───────────────────────────────────────────

def test_case_c_avoid_includes_random_holdout():
    from examples.make_group_demo import make_group_data, AUDITOR_KWARGS
    train, predict = make_group_data()
    results = _run_full(train, predict, **AUDITOR_KWARGS)
    avoid = results["validation"]["recommendation"]["strategies_to_avoid"]
    assert "random_holdout" in avoid

def test_case_d_avoid_includes_kfold():
    from examples.make_time_split_demo import make_time_split_data, AUDITOR_KWARGS
    train, predict = make_time_split_data()
    results = _run_full(train, predict, **AUDITOR_KWARGS)
    avoid = results["validation"]["recommendation"]["strategies_to_avoid"]
    assert "kfold" in avoid
