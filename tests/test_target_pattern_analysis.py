"""Tests for target pattern analysis."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import pytest
from pattern_auditor.feature_types import FeatureTypeInferrer
from pattern_auditor.target_patterns import TargetPatternAnalyser


def _make_correlated_df(seed=0):
    rng = np.random.default_rng(seed)
    n = 400
    x = rng.normal(0, 1, n)
    y = 3 * x + rng.normal(0, 0.5, n)
    return pd.DataFrame({
        "strong_feature": x,
        "noise_feature": rng.normal(size=n),
        "category": rng.choice(["A", "B", "C"], n),
        "target": y,
    })


def test_numeric_correlation_detected():
    df = _make_correlated_df()
    fti = FeatureTypeInferrer(df, target_col="target")
    ft = fti.infer()
    tpa = TargetPatternAnalyser(df, "target", ft)
    report = tpa.analyse()
    corrs = {c["column"]: c["pearson_r"] for c in report["numeric_correlations"]}
    assert "strong_feature" in corrs
    assert abs(corrs["strong_feature"]) > 0.9


def test_categorical_target_means():
    rng = np.random.default_rng(1)
    n = 300
    cats = rng.choice(["X", "Y"], n)
    target = np.where(cats == "X", rng.normal(10, 1, n), rng.normal(20, 1, n))
    df = pd.DataFrame({"category": cats, "target": target})
    fti = FeatureTypeInferrer(df, target_col="target")
    ft = fti.infer()
    tpa = TargetPatternAnalyser(df, "target", ft)
    report = tpa.analyse()
    cat_means = {
        item["column"]: item
        for item in report["categorical_target_means"]
    }
    assert "category" in cat_means
    records = cat_means["category"]["target_mean_by_category"]
    means_by_cat = {r["category"]: r["target_mean"] for r in records}
    assert means_by_cat["X"] < means_by_cat["Y"]


def test_correlations_sorted_by_abs():
    df = _make_correlated_df()
    fti = FeatureTypeInferrer(df, target_col="target")
    ft = fti.infer()
    tpa = TargetPatternAnalyser(df, "target", ft)
    report = tpa.analyse()
    corrs = report["numeric_correlations"]
    abs_vals = [abs(c["pearson_r"]) for c in corrs if c.get("pearson_r") is not None]
    assert abs_vals == sorted(abs_vals, reverse=True)


def test_target_stats_returned():
    df = _make_correlated_df()
    fti = FeatureTypeInferrer(df, target_col="target")
    ft = fti.infer()
    tpa = TargetPatternAnalyser(df, "target", ft)
    report = tpa.analyse()
    assert "mean" in report["target_stats"]
    assert "std" in report["target_stats"]


def test_datetime_patterns_populated():
    rng = np.random.default_rng(2)
    ts = pd.date_range("2023-01-01", periods=200, freq="h")
    df = pd.DataFrame({
        "timestamp": ts,
        "target": rng.normal(size=200),
    })
    fti = FeatureTypeInferrer(df, target_col="target", datetime_col="timestamp")
    ft = fti.infer()
    tpa = TargetPatternAnalyser(df, "target", ft)
    report = tpa.analyse()
    assert len(report["datetime_target_patterns"]) > 0
