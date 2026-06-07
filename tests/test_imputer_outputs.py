"""Tests for Imputer: correct strategies applied, leakage-safe fit/transform."""

import numpy as np
import pandas as pd
import pytest
from missingness_auditor.imputer import Imputer
from missingness_auditor import MissingnessAuditor


def _plan(strategies: dict, add_indicators: dict | None = None) -> dict:
    cols = {}
    for col, strat in strategies.items():
        add_ind = (add_indicators or {}).get(col, False)
        fit_on = "N/A" if strat == "no_imputation_needed" else "train_only"
        cols[col] = {
            "strategy": strat,
            "add_missing_indicator": add_ind,
            "reason": "test",
            "fit_on": fit_on,
        }
    return {"columns": cols}


def test_numeric_median_fills_nulls():
    df = pd.DataFrame({"x": [1.0, np.nan, 3.0, 4.0, 5.0]})
    plan = _plan({"x": "numeric_median"})
    result = Imputer().apply(df, plan)
    assert result["x"].isna().sum() == 0
    assert result["x"].iloc[1] == pytest.approx(3.5)  # median of [1,3,4,5] = (3+4)/2


def test_numeric_median_plus_indicator_creates_column():
    df = pd.DataFrame({"x": [1.0, np.nan, 3.0, 4.0, np.nan]})
    plan = _plan({"x": "numeric_median_plus_indicator"}, {"x": True})
    result = Imputer().apply(df, plan)
    assert result["x"].isna().sum() == 0
    assert "x_was_missing" in result.columns
    assert set(result["x_was_missing"].unique()) <= {0, 1}
    assert result["x_was_missing"].iloc[1] == 1
    assert result["x_was_missing"].iloc[0] == 0


def test_categorical_missing_token():
    df = pd.DataFrame({"cat": ["a", None, "b", "a", None]})
    plan = _plan({"cat": "categorical_missing_token"})
    result = Imputer().apply(df, plan)
    assert result["cat"].isna().sum() == 0
    assert "MISSING" in result["cat"].values


def test_categorical_mode_plus_indicator():
    df = pd.DataFrame({"cat": ["a", None, "a", "b", None, "a"]})
    plan = _plan({"cat": "categorical_mode_plus_indicator"}, {"cat": True})
    result = Imputer().apply(df, plan)
    assert result["cat"].isna().sum() == 0
    assert "cat_was_missing" in result.columns
    assert set(result["cat_was_missing"].unique()) <= {0, 1}
    assert result["cat"].iloc[1] == "a"  # mode is "a"


def test_structural_none_or_zero_numeric():
    df = pd.DataFrame({"x": [1.0, np.nan, 3.0]})
    plan = _plan({"x": "structural_none_or_zero"}, {"x": True})
    result = Imputer().apply(df, plan)
    assert result["x"].isna().sum() == 0
    assert result["x"].iloc[1] == 0


def test_structural_none_or_zero_categorical():
    df = pd.DataFrame({"cat": ["good", None, "poor"]})
    plan = _plan({"cat": "structural_none_or_zero"}, {"cat": True})
    result = Imputer().apply(df, plan)
    assert result["cat"].isna().sum() == 0
    assert result["cat"].iloc[1] == "NONE"


def test_drop_column_removes_it():
    df = pd.DataFrame({"x": [np.nan] * 5, "y": [1.0] * 5})
    plan = _plan({"x": "drop_column"})
    result = Imputer().apply(df, plan)
    assert "x" not in result.columns
    assert "y" in result.columns


def test_no_imputation_leaves_column_unchanged():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    plan = _plan({"x": "no_imputation_needed"})
    result = Imputer().apply(df, plan)
    assert list(result["x"]) == [1.0, 2.0, 3.0]


def test_train_predict_split_uses_train_median():
    rng = np.random.default_rng(0)
    df_train = pd.DataFrame({"x": [10.0, np.nan, 10.0, 10.0]})
    df_pred  = pd.DataFrame({"x": [np.nan, 100.0]})
    plan = _plan({"x": "numeric_median"})
    train_out, pred_out = Imputer().apply(df_train, plan, df_pred)
    # train median of non-null values is 10.0 — must be used for predict, not 100
    assert pred_out["x"].iloc[0] == pytest.approx(10.0)


def test_indicator_added_before_imputation():
    """Indicator must reflect original nulls, not post-imputation state."""
    df = pd.DataFrame({"x": [1.0, np.nan, 3.0]})
    plan = _plan({"x": "numeric_median_plus_indicator"}, {"x": True})
    result = Imputer().apply(df, plan)
    # row 1 was missing → indicator should be 1
    assert result["x_was_missing"].iloc[1] == 1
    # row 0 was not missing → indicator should be 0
    assert result["x_was_missing"].iloc[0] == 0


def test_full_auditor_apply_imputation():
    rng = np.random.default_rng(9)
    n = 200
    df = pd.DataFrame({
        "num":  np.where(rng.random(n) < 0.15, np.nan, rng.normal(0, 1, n)),
        "cat":  np.where(rng.random(n) < 0.20, None, rng.choice(["a", "b"], n)),
        "target": rng.normal(0, 1, n),
    })
    auditor = MissingnessAuditor(df, target_col="target")
    results = auditor.run()
    imputed = auditor.apply_imputation(df, results["imputation_plan"])
    # All non-drop columns should have no NaN after imputation
    plan_cols = results["imputation_plan"]["columns"]
    for col, entry in plan_cols.items():
        if entry["strategy"] in ("no_imputation_needed", "drop_column"):
            continue
        if col in imputed.columns:
            assert imputed[col].isna().sum() == 0, f"{col} still has NaN"
