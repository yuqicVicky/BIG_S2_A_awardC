"""Tests for MissingnessProfiler."""

import numpy as np
import pandas as pd
import pytest
from missingness_auditor.profiler import MissingnessProfiler


def _df():
    return pd.DataFrame({
        "num_a":   [1.0, np.nan, 3.0, np.nan, 5.0],
        "num_b":   [10.0, 20.0, 30.0, 40.0, 50.0],
        "cat_a":   ["x", None, "y", "x", None],
        "target":  [0, 1, 0, 1, 0],
    })


def test_missing_counts():
    result = MissingnessProfiler(_df(), target_col="target").profile()
    cols = result["columns"]
    assert cols["num_a"]["n_missing"] == 2
    assert cols["num_b"]["n_missing"] == 0
    assert cols["cat_a"]["n_missing"] == 2


def test_missing_rates():
    result = MissingnessProfiler(_df(), target_col="target").profile()
    cols = result["columns"]
    assert cols["num_a"]["missing_rate"] == pytest.approx(0.4, abs=0.01)
    assert cols["num_b"]["missing_rate"] == 0.0


def test_severity_labels():
    result = MissingnessProfiler(_df()).profile()
    cols = result["columns"]
    assert cols["num_b"]["severity"] == "none"
    assert cols["num_a"]["severity"] == "moderate"   # 40% → moderate


def test_dtype_category():
    result = MissingnessProfiler(_df()).profile()
    cols = result["columns"]
    assert cols["num_a"]["dtype_category"] == "numeric"
    assert cols["cat_a"]["dtype_category"] == "categorical"


def test_target_flag():
    result = MissingnessProfiler(_df(), target_col="target").profile()
    cols = result["columns"]
    assert cols["target"]["is_target"] is True
    assert cols["num_a"]["is_target"] is False


def test_summary_counts():
    result = MissingnessProfiler(_df(), target_col="target").profile()
    s = result["summary"]
    assert s["total_rows"] == 5
    assert s["total_columns"] == 4
    assert s["columns_with_any_missing"] == 2


def test_columns_with_missing_list():
    result = MissingnessProfiler(_df()).profile()
    assert set(result["columns_with_missing"]) == {"num_a", "cat_a"}


def test_predict_missing_rate():
    train = _df()
    predict = pd.DataFrame({
        "num_a":  [np.nan, np.nan, 3.0],
        "num_b":  [10.0, 20.0, 30.0],
        "cat_a":  [None, None, "y"],
        "target": [np.nan, np.nan, np.nan],
    })
    result = MissingnessProfiler(train, predict).profile()
    assert result["columns"]["num_a"]["predict_missing_rate"] == pytest.approx(2 / 3, abs=0.01)


def test_no_missing_df():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    result = MissingnessProfiler(df).profile()
    assert result["summary"]["columns_with_any_missing"] == 0
    assert result["summary"]["overall_missing_rate"] == 0.0


def test_complete_column_severity_none():
    result = MissingnessProfiler(_df()).profile()
    assert result["columns"]["num_b"]["severity"] == "none"
