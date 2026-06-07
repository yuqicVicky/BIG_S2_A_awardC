"""Tests for MissingnessProfiler."""

import numpy as np
import pandas as pd
import pytest

from missingness_auditor.profiler import MissingnessProfiler


def _make_df():
    return pd.DataFrame({
        "numeric_a": [1.0, np.nan, 3.0, np.nan, 5.0],
        "numeric_b": [10.0, 20.0, 30.0, 40.0, 50.0],
        "cat_a": ["x", None, "y", "x", None],
        "target": [0, 1, 0, 1, 0],
    })


def test_missing_counts():
    df = _make_df()
    result = MissingnessProfiler(df, target_col="target").profile()
    cols = result["columns"]

    assert cols["numeric_a"]["n_missing"] == 2
    assert cols["numeric_b"]["n_missing"] == 0
    assert cols["cat_a"]["n_missing"] == 2


def test_missing_rates():
    df = _make_df()
    result = MissingnessProfiler(df, target_col="target").profile()
    cols = result["columns"]

    assert cols["numeric_a"]["missing_rate"] == pytest.approx(0.4, abs=0.01)
    assert cols["numeric_b"]["missing_rate"] == 0.0


def test_severity_labels():
    df = _make_df()
    result = MissingnessProfiler(df).profile()
    cols = result["columns"]

    assert cols["numeric_b"]["severity"] == "none"
    # 40% missing → "moderate"
    assert cols["numeric_a"]["severity"] == "moderate"


def test_dtype_category():
    df = _make_df()
    result = MissingnessProfiler(df).profile()
    cols = result["columns"]

    assert cols["numeric_a"]["dtype_category"] == "numeric"
    assert cols["cat_a"]["dtype_category"] == "categorical"


def test_target_flag():
    df = _make_df()
    result = MissingnessProfiler(df, target_col="target").profile()
    cols = result["columns"]

    assert cols["target"]["is_target"] is True
    assert cols["numeric_a"]["is_target"] is False


def test_summary_counts():
    df = _make_df()
    result = MissingnessProfiler(df, target_col="target").profile()
    s = result["summary"]

    assert s["total_rows"] == 5
    assert s["total_columns"] == 4
    assert s["columns_with_any_missing"] == 2  # numeric_a and cat_a


def test_columns_with_missing_list():
    df = _make_df()
    result = MissingnessProfiler(df).profile()
    assert set(result["columns_with_missing"]) == {"numeric_a", "cat_a"}


def test_predict_missing_rate_populated():
    train = _make_df()
    predict = pd.DataFrame({
        "numeric_a": [np.nan, np.nan, 3.0],
        "numeric_b": [10.0, 20.0, 30.0],
        "cat_a": [None, None, "y"],
        "target": [np.nan, np.nan, np.nan],
    })
    result = MissingnessProfiler(train, predict).profile()
    assert result["columns"]["numeric_a"]["predict_missing_rate"] == pytest.approx(2 / 3, abs=0.01)


def test_no_missing_df():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    result = MissingnessProfiler(df).profile()
    assert result["summary"]["columns_with_any_missing"] == 0
    assert result["summary"]["overall_missing_rate"] == 0.0
