"""Tests for pattern-driven feature recommendation."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import tempfile
import numpy as np
import pandas as pd
import pytest
from pattern_auditor import PatternAuditor


def _run_auditor(train, predict, **kwargs) -> dict:
    with tempfile.TemporaryDirectory() as tmpdir:
        auditor = PatternAuditor(train, predict, **kwargs)
        results = auditor.run(output_dir=tmpdir)
    return results


def _within_period_dataset():
    """Dataset with temporal + group structure (detect as within_period or time_based_split)."""
    from examples.make_toy_within_period_data import make_within_period_data
    return make_within_period_data(n_entities=20, n_periods=20, seed=42)


def _temporal_group_overlapping():
    """Dataset where predict timestamps genuinely overlap train (classic within-period)."""
    rng = np.random.default_rng(99)
    ts = pd.date_range("2023-01-01", periods=120, freq="h")
    n = len(ts)
    groups = rng.choice(["G01", "G02", "G03", "G04"], n)
    df = pd.DataFrame({
        "timestamp": ts,
        "group_id": groups,
        "numeric_feat": rng.normal(size=n),
        "target": rng.normal(size=n),
    })
    train = df.iloc[:100].copy()
    predict = df.iloc[90:].drop(columns=["target"]).copy()
    return train, predict


def test_feature_recommendation_has_required_keys():
    train, predict = _within_period_dataset()
    results = _run_auditor(train, predict, target_col="target",
                           datetime_col="timestamp", group_col="entity_id")
    fe = results["feature_engineering"]
    for key in ("features_to_add", "features_to_exclude", "interactions_to_try",
                "target_transforms_to_try", "warnings"):
        assert key in fe, f"Missing key in feature_recommendation: {key}"


def test_datetime_features_added_for_datetime_col():
    train, predict = _within_period_dataset()
    results = _run_auditor(train, predict, target_col="target",
                           datetime_col="timestamp", group_col="entity_id")
    to_add = results["feature_engineering"]["features_to_add"]
    types = {f["type"] for f in to_add}
    assert "datetime_derived" in types, "Expected datetime_derived features in features_to_add"
    derived_names = {f["feature"] for f in to_add if f["type"] == "datetime_derived"}
    assert any("hour" in n for n in derived_names)
    assert any("dayofweek" in n for n in derived_names)


def test_within_period_interactions_include_lag_features():
    """Temporal + group data should produce lag interaction suggestions."""
    train, predict = _temporal_group_overlapping()
    results = _run_auditor(train, predict, target_col="target",
                           datetime_col="timestamp", group_col="group_id")
    interactions = results["feature_engineering"]["interactions_to_try"]
    types = [i["type"] for i in interactions]
    assert "lag" in types, f"Expected lag interaction for temporal+group pattern; got types: {types}"


def test_within_period_interactions_include_datetime_interactions():
    """hour×dayofweek and hour×is_workingday should always appear for datetime data."""
    train, predict = _within_period_dataset()
    results = _run_auditor(train, predict, target_col="target",
                           datetime_col="timestamp", group_col="entity_id")
    interactions = results["feature_engineering"]["interactions_to_try"]
    interaction_names = {i["interaction"] for i in interactions}
    assert "hour × dayofweek" in interaction_names
    assert "hour × is_workingday" in interaction_names


def test_train_only_leakage_in_features_to_exclude():
    """Train-only columns (high leakage) must appear in features_to_exclude."""
    rng = np.random.default_rng(0)
    train = pd.DataFrame({
        "feat_a": rng.normal(size=200),
        "secret_train_only": rng.normal(size=200),
        "target": rng.normal(size=200),
    })
    predict = pd.DataFrame({"feat_a": rng.normal(size=40)})
    results = _run_auditor(train, predict, target_col="target")
    to_excl = {f["column"] for f in results["feature_engineering"]["features_to_exclude"]}
    assert "secret_train_only" in to_excl, "Train-only leakage column should be in features_to_exclude"


def test_iid_data_no_lag_interactions():
    """Pure iid data should not produce lag or group_encoding interactions."""
    rng = np.random.default_rng(5)
    train = pd.DataFrame({
        "x": rng.normal(size=200),
        "y": rng.normal(size=200),
        "target": rng.normal(size=200),
    })
    predict = pd.DataFrame({"x": rng.normal(size=40), "y": rng.normal(size=40)})
    results = _run_auditor(train, predict, target_col="target")
    interactions = results["feature_engineering"]["interactions_to_try"]
    types = [i["type"] for i in interactions]
    assert "lag" not in types, "iid data should not suggest lag features"


def test_group_split_interactions_include_group_encoding():
    """Group-split pattern should suggest group target encoding."""
    rng = np.random.default_rng(2)
    train_groups = [f"G{i:03d}" for i in range(60)]
    predict_groups = [f"G{i:03d}" for i in range(60, 100)]
    train = pd.DataFrame({
        "group_id": np.repeat(train_groups, 5),
        "val": rng.normal(size=300),
        "target": rng.normal(size=300),
    })
    predict = pd.DataFrame({
        "group_id": np.repeat(predict_groups, 5),
        "val": rng.normal(size=200),
    })
    results = _run_auditor(train, predict, target_col="target", group_col="group_id")
    interactions = results["feature_engineering"]["interactions_to_try"]
    types = [i["type"] for i in interactions]
    assert "group_encoding" in types, f"Group-split should suggest group_encoding; got: {types}"


def test_near_constant_in_features_to_exclude():
    """Near-constant columns should be in features_to_exclude."""
    rng = np.random.default_rng(7)
    n = 200
    train = pd.DataFrame({
        "good": rng.normal(size=n),
        "near_const": [1.0] * 198 + [2.0, 3.0],
        "target": rng.normal(size=n),
    })
    predict = pd.DataFrame({
        "good": rng.normal(size=40),
        "near_const": [1.0] * 40,
    })
    results = _run_auditor(train, predict, target_col="target")
    to_excl = {f["column"] for f in results["feature_engineering"]["features_to_exclude"]}
    assert "near_const" in to_excl


def test_feature_recommendation_json_structure_serializable():
    """feature_recommendation dict must be JSON-serializable."""
    import json
    train, predict = _within_period_dataset()
    results = _run_auditor(train, predict, target_col="target",
                           datetime_col="timestamp", group_col="entity_id")
    # Should not raise
    dumped = json.dumps(results["feature_engineering"])
    loaded = json.loads(dumped)
    assert "features_to_add" in loaded
