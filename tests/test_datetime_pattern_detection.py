"""Tests for datetime pattern detection."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import pytest
from pattern_auditor.datetime_patterns import DatetimePatternAnalyser, _infer_granularity


def _make_hourly_df(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2023-01-01", periods=n, freq="h")
    return pd.DataFrame({
        "timestamp": ts,
        "target": rng.normal(10, 2, n) + 3 * np.sin(2 * np.pi * ts.hour / 24),
    })


def test_coverage_returns_min_max():
    df = _make_hourly_df()
    dpa = DatetimePatternAnalyser(df, "timestamp", "target")
    cov = dpa.coverage()
    assert "min" in cov
    assert "max" in cov
    assert cov["n_rows"] == 200


def test_granularity_hourly():
    ts = pd.date_range("2023-01-01", periods=100, freq="h")
    result = _infer_granularity(ts.to_series())
    assert result == "hour"


def test_granularity_daily():
    ts = pd.date_range("2023-01-01", periods=30, freq="D")
    result = _infer_granularity(ts.to_series())
    assert result == "day"


def test_extract_components_has_hour():
    df = _make_hourly_df()
    dpa = DatetimePatternAnalyser(df, "timestamp", "target")
    ext = dpa.extract_components()
    assert "__hour" in ext.columns
    assert "__dayofweek" in ext.columns
    assert "__month" in ext.columns


def test_target_by_component_hour():
    df = _make_hourly_df(n=500)
    dpa = DatetimePatternAnalyser(df, "timestamp", "target")
    by_comp = dpa.target_by_component()
    assert "hour" in by_comp
    assert len(by_comp["hour"]) == 24


def test_hour_x_dayofweek_interaction():
    df = _make_hourly_df(n=2000)
    dpa = DatetimePatternAnalyser(df, "timestamp", "target")
    by_comp = dpa.target_by_component()
    assert "hour_x_dayofweek" in by_comp


def test_string_datetime_parsed():
    df = pd.DataFrame({
        "timestamp": ["2023-01-01 00:00", "2023-01-01 01:00", "2023-01-01 02:00"] * 20,
        "target": np.random.default_rng(5).normal(size=60),
    })
    dpa = DatetimePatternAnalyser(df, "timestamp", "target")
    cov = dpa.coverage()
    assert "error" not in cov
