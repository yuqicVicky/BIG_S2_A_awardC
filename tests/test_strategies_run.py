"""Every supported strategy applies cleanly: no errors, no residual nulls."""

import numpy as np
import pandas as pd
import pytest

from missingness_auditor import Imputer


def _frame():
    rng = np.random.RandomState(7)
    n = 60
    df = pd.DataFrame({
        "num":   rng.normal(0, 1, n),
        "num2":  rng.normal(3, 1, n),
        "grp":   rng.choice(["A", "B", "C"], n),
        "cat":   rng.choice(["x", "y", "z"], n),
        "text":  [f"id_{i}" for i in range(n)],  # high cardinality
        "sparse": rng.normal(0, 1, n),
    })
    for c in ["num", "num2", "cat", "text", "sparse"]:
        df.loc[rng.rand(n) < 0.25, c] = np.nan
    df.loc[rng.rand(n) < 0.9, "sparse"] = np.nan  # >80% missing → drop
    return df


STRATEGIES = [
    ("num",  "numeric_median", False, None),
    ("num",  "numeric_median_plus_indicator", True, None),
    ("num",  "model_based_imputation_optional", True, None),
    ("num",  "mice_multiple_imputation", True, None),
    ("num",  "time_series_ffill_bfill_plus_indicator", True, None),
    ("num",  "groupwise_numeric_median_plus_indicator", True, "grp"),
    ("cat",  "categorical_missing_token", False, None),
    ("cat",  "categorical_missing_token_plus_indicator", True, None),
    ("cat",  "categorical_mode_plus_indicator", True, None),
    ("cat",  "structural_none_token_plus_indicator", True, None),
    ("num",  "structural_zero_plus_indicator", True, None),
    ("num",  "structural_none_or_zero", True, None),
]


@pytest.mark.parametrize("col,strategy,add_ind,group_col", STRATEGIES)
def test_strategy_applies_without_error(col, strategy, add_ind, group_col):
    df = _frame()
    entry = {"strategy": strategy, "add_missing_indicator": add_ind}
    if group_col:
        entry["group_col"] = group_col
    out = Imputer().apply(df, {"columns": {col: entry}})
    assert out[col].isna().sum() == 0
    if add_ind:
        assert f"{col}_was_missing" in out.columns


def test_drop_column_strategy_removes_column():
    df = _frame()
    out = Imputer().apply(df, {"columns": {
        "sparse": {"strategy": "drop_column", "add_missing_indicator": False}
    }})
    assert "sparse" not in out.columns
