"""Tests for feature type inference."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import pytest
from pattern_auditor.feature_types import FeatureTypeInferrer


def _make_df(**cols) -> pd.DataFrame:
    return pd.DataFrame(cols)


def test_numeric_continuous():
    df = _make_df(x=np.random.default_rng(0).normal(0, 1, 200))
    fti = FeatureTypeInferrer(df)
    result = fti.infer()
    assert result["columns"]["x"]["inferred_type"] == "numeric_continuous"


def test_boolean_binary():
    df = _make_df(flag=[0, 1, 0, 1, 0, 1] * 50)
    fti = FeatureTypeInferrer(df)
    result = fti.infer()
    assert result["columns"]["flag"]["inferred_type"] == "boolean_binary"


def test_categorical_low_cardinality():
    df = _make_df(color=["red", "green", "blue"] * 100)
    fti = FeatureTypeInferrer(df)
    result = fti.infer()
    assert result["columns"]["color"]["inferred_type"] == "categorical_low_cardinality"


def test_datetime_column_by_name():
    df = _make_df(timestamp=pd.date_range("2023-01-01", periods=100, freq="h"))
    fti = FeatureTypeInferrer(df)
    result = fti.infer()
    assert result["columns"]["timestamp"]["inferred_type"] == "datetime_like"


def test_explicit_target_col():
    df = _make_df(y=np.arange(100, dtype=float), x=np.random.default_rng(1).normal(size=100))
    fti = FeatureTypeInferrer(df, target_col="y")
    result = fti.infer()
    assert result["columns"]["y"]["inferred_type"] == "target"


def test_constant_column():
    df = _make_df(const=[5] * 100, feat=np.arange(100, dtype=float))
    fti = FeatureTypeInferrer(df)
    result = fti.infer()
    assert result["columns"]["const"]["inferred_type"] == "constant"


def test_near_constant_column():
    vals = [1] * 99 + [2]
    df = _make_df(near=[1] * 990 + [2] * 10, feat=np.arange(1000, dtype=float))
    fti = FeatureTypeInferrer(df)
    result = fti.infer()
    assert result["columns"]["near"]["inferred_type"] == "near_constant"


def test_train_only_column():
    train = _make_df(a=np.arange(50, dtype=float), b=np.arange(50, dtype=float))
    predict = _make_df(a=np.arange(10, dtype=float))
    fti = FeatureTypeInferrer(train, predict)
    result = fti.infer()
    assert result["columns"]["b"]["inferred_type"] == "train_only"


def test_text_like_column():
    texts = ["This is a long description of the event"] * 100
    df = _make_df(description=texts, val=np.arange(100, dtype=float))
    fti = FeatureTypeInferrer(df)
    result = fti.infer()
    assert result["columns"]["description"]["inferred_type"] == "text_like"


def test_numeric_discrete():
    df = _make_df(rating=np.random.default_rng(2).integers(1, 6, 200).astype(float))
    fti = FeatureTypeInferrer(df)
    result = fti.infer()
    assert result["columns"]["rating"]["inferred_type"] == "numeric_discrete"


def test_type_summary_counts():
    df = _make_df(
        x=np.random.default_rng(3).normal(size=100),
        y=np.random.default_rng(4).normal(size=100),
        c=["a", "b"] * 50,
    )
    fti = FeatureTypeInferrer(df, target_col="y")
    result = fti.infer()
    assert result["type_summary"]["target"] == 1
