"""Tests for feature availability audit."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import tempfile
import numpy as np
import pandas as pd
import pytest
from pattern_auditor import PatternAuditor


def _run(train, predict, **kwargs) -> dict:
    with tempfile.TemporaryDirectory() as tmpdir:
        auditor = PatternAuditor(train, predict, **kwargs)
        return auditor.run(output_dir=tmpdir)


def test_availability_audit_has_required_keys():
    rng = np.random.default_rng(0)
    train = pd.DataFrame({"a": rng.normal(size=100), "target": rng.normal(size=100)})
    predict = pd.DataFrame({"a": rng.normal(size=20)})
    results = _run(train, predict, target_col="target")
    fa = results["feature_availability"]
    for key in ("columns", "summary", "columns_to_use", "columns_to_exclude", "columns_to_verify"):
        assert key in fa, f"Missing key: {key}"


def test_train_only_column_excluded():
    rng = np.random.default_rng(1)
    train = pd.DataFrame({
        "shared": rng.normal(size=100),
        "train_secret": rng.normal(size=100),
        "target": rng.normal(size=100),
    })
    predict = pd.DataFrame({"shared": rng.normal(size=20)})
    results = _run(train, predict, target_col="target")
    fa = results["feature_availability"]
    assert "train_secret" in fa["columns_to_exclude"], \
        "train-only column must be in columns_to_exclude"
    assert fa["columns"]["train_secret"]["available_at_prediction"] is False


def test_shared_column_is_usable():
    rng = np.random.default_rng(2)
    train = pd.DataFrame({"feat": rng.normal(size=100), "target": rng.normal(size=100)})
    predict = pd.DataFrame({"feat": rng.normal(size=20)})
    results = _run(train, predict, target_col="target")
    fa = results["feature_availability"]
    assert "feat" in fa["columns_to_use"]
    assert fa["columns"]["feat"]["action"] == "use"


def test_target_component_suspect_excluded():
    rng = np.random.default_rng(3)
    train = pd.DataFrame({
        "feature": rng.normal(size=100),
        "target_score": rng.normal(size=100),
        "target": rng.normal(size=100),
    })
    predict = pd.DataFrame({
        "feature": rng.normal(size=20),
        "target_score": rng.normal(size=20),
    })
    results = _run(train, predict, target_col="target")
    fa = results["feature_availability"]
    assert "target_score" in fa["columns_to_exclude"], \
        "target_component_suspect should be excluded"


def test_summary_counts_consistent():
    rng = np.random.default_rng(4)
    train = pd.DataFrame({
        "a": rng.normal(size=100),
        "b": rng.normal(size=100),
        "train_only": rng.normal(size=100),
        "target": rng.normal(size=100),
    })
    predict = pd.DataFrame({"a": rng.normal(size=20), "b": rng.normal(size=20)})
    results = _run(train, predict, target_col="target")
    fa = results["feature_availability"]
    s = fa["summary"]
    assert s["train_only"] >= 1
    assert s["available_at_prediction"] >= 2


def test_feature_availability_json_serializable():
    import json
    rng = np.random.default_rng(5)
    train = pd.DataFrame({"x": rng.normal(size=100), "target": rng.normal(size=100)})
    predict = pd.DataFrame({"x": rng.normal(size=20)})
    results = _run(train, predict, target_col="target")
    dumped = json.dumps(results["feature_availability"])
    loaded = json.loads(dumped)
    assert "columns_to_use" in loaded
