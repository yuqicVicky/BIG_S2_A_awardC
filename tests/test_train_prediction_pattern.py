"""Tests for train/prediction split pattern detection."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import pytest
from pattern_auditor.feature_types import FeatureTypeInferrer
from pattern_auditor.distribution_shift import DistributionShiftDetector


def _make_time_split():
    """Predict timestamps come strictly after train."""
    ts_train = pd.date_range("2023-01-01", periods=100, freq="D")
    ts_predict = pd.date_range("2023-04-11", periods=20, freq="D")
    rng = np.random.default_rng(0)
    train = pd.DataFrame({"timestamp": ts_train, "val": rng.normal(size=100), "target": rng.normal(size=100)})
    predict = pd.DataFrame({"timestamp": ts_predict, "val": rng.normal(size=20)})
    return train, predict


def _make_within_period_split():
    """Predict timestamps overlap with train."""
    ts_all = pd.date_range("2023-01-01", periods=100, freq="h")
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"timestamp": ts_all, "val": rng.normal(size=100), "target": rng.normal(size=100)})
    train = df.iloc[:80].copy()
    predict = df.iloc[70:].drop(columns=["target"]).copy()  # overlap!
    return train, predict


def _make_group_split():
    """Predict groups are disjoint from train groups."""
    rng = np.random.default_rng(2)
    train_groups = [f"G{i:03d}" for i in range(75)]
    predict_groups = [f"G{i:03d}" for i in range(75, 100)]
    train = pd.DataFrame({
        "group_id": np.repeat(train_groups, 10),
        "val": rng.normal(size=750),
        "target": rng.normal(size=750),
    })
    predict = pd.DataFrame({
        "group_id": np.repeat(predict_groups, 10),
        "val": rng.normal(size=250),
    })
    return train, predict


def _detect(train, predict, datetime_col=None, group_col=None):
    fti = FeatureTypeInferrer(train, predict, target_col="target",
                              datetime_col=datetime_col, group_col=group_col)
    ft_report = fti.infer()
    dsd = DistributionShiftDetector(train, predict, ft_report)
    result = dsd.detect()
    return result["split_pattern"]


def test_time_based_split_detected():
    train, predict = _make_time_split()
    sp = _detect(train, predict, datetime_col="timestamp")
    assert sp["pattern"] == "time_based_split"


def test_within_period_split_detected():
    train, predict = _make_within_period_split()
    sp = _detect(train, predict, datetime_col="timestamp")
    assert sp["pattern"] == "within_period"


def test_group_split_detected():
    train, predict = _make_group_split()
    sp = _detect(train, predict, group_col="group_id")
    assert sp["pattern"] == "group_based_split"


def test_iid_pattern_when_no_structure():
    rng = np.random.default_rng(3)
    train = pd.DataFrame({"a": rng.normal(size=200), "b": rng.normal(size=200), "target": rng.normal(size=200)})
    predict = pd.DataFrame({"a": rng.normal(size=50), "b": rng.normal(size=50)})
    sp = _detect(train, predict)
    assert sp["pattern"] == "iid_random"
