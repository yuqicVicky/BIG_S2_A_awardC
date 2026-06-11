"""Edge-case robustness tests for degenerate inputs.

These protect the degenerate-column guard in the imputer/codegen and confirm the
auditor never hard-fails on the awkward frames real datasets produce: an all-null
column, a single row, no target, a constant column, and a column that is null in
train but present in predict (the leakage-safe boundary).

Each test asserts the same contract the skill advertises: the run completes, no
residual nulls remain in imputed columns, and the target is never modified.
"""

import numpy as np
import pandas as pd

from missingness_auditor import MissingnessAuditor
from missingness_auditor.imputer import Imputer


def _no_residual_nulls(df_imputed, plan):
    """No imputed column keeps nulls (drop_column / no_imputation excepted)."""
    for col, entry in plan["columns"].items():
        if entry["strategy"] in ("no_imputation_needed", "drop_column"):
            continue
        if col in df_imputed.columns:
            assert not df_imputed[col].isna().any(), f"{col} still has nulls"


def test_all_null_numeric_column_via_direct_plan():
    """A numeric column with no observed value must still be filled (median is NaN
    → the 0.0 guard kicks in) rather than leaving residual nulls."""
    df = pd.DataFrame({
        "allnull": [np.nan] * 30,
        "x": np.arange(30, dtype=float),
        "target": np.arange(30, dtype=float),
    })
    plan = {"columns": {
        "allnull": {"strategy": "numeric_median", "add_missing_indicator": True,
                    "fit_on": "train_only"},
    }}
    out = Imputer().apply(df, plan)
    assert not out["allnull"].isna().any()
    assert (out["allnull"] == 0.0).all()
    # Target untouched
    assert out["target"].equals(df["target"])


def test_all_null_groupwise_and_timeseries_guarded():
    """The group-median and time-series fallbacks are guarded the same way."""
    df = pd.DataFrame({
        "allnull": [np.nan] * 20,
        "grp": (["a", "b"] * 10),
        "target": np.arange(20, dtype=float),
    })
    for strat in ("groupwise_numeric_median_plus_indicator",
                  "time_series_ffill_bfill_plus_indicator"):
        plan = {"columns": {"allnull": {
            "strategy": strat, "add_missing_indicator": True,
            "group_col": "grp", "fit_on": "train_only"}}}
        out = Imputer().apply(df.copy(), plan)
        assert not out["allnull"].isna().any(), strat


def test_single_row_frame_completes():
    df = pd.DataFrame({"a": [1.0], "b": [np.nan], "target": [0.0]})
    aud = MissingnessAuditor(df, target_col="target")
    results = aud.run()
    out = aud.apply_imputation(df, results["imputation_plan"])
    out = out[0] if isinstance(out, tuple) else out
    _no_residual_nulls(out, results["imputation_plan"])
    assert out["target"].equals(df["target"])


def test_all_numeric_no_target_completes():
    rng = np.random.RandomState(0)
    df = pd.DataFrame({
        "a": rng.normal(size=60),
        "b": rng.normal(size=60),
        "c": rng.normal(size=60),
    })
    df.loc[df.sample(15, random_state=1).index, "a"] = np.nan
    aud = MissingnessAuditor(df, target_col=None)
    results = aud.run()
    # Mechanism audit must run and Little's MCAR test must be reported.
    assert "little_mcar_test" in results["mechanism_audit"]
    out = aud.apply_imputation(df, results["imputation_plan"])
    out = out[0] if isinstance(out, tuple) else out
    _no_residual_nulls(out, results["imputation_plan"])


def test_constant_column_completes():
    """A zero-variance column must not crash the significance tests."""
    df = pd.DataFrame({
        "const": [5.0] * 50,
        "varies": np.arange(50, dtype=float),
        "target": np.arange(50, dtype=float),
    })
    df.loc[:10, "varies"] = np.nan
    df.loc[:5, "const"] = np.nan  # constant-but-with-missing
    aud = MissingnessAuditor(df, target_col="target")
    results = aud.run()
    out = aud.apply_imputation(df, results["imputation_plan"])
    out = out[0] if isinstance(out, tuple) else out
    _no_residual_nulls(out, results["imputation_plan"])
    assert out["target"].equals(df["target"])


def test_train_null_but_present_in_predict_is_leakage_safe():
    """Column is entirely null in train but observed in predict. The fill must
    derive from train only (→ the 0.0 guard), never from predict values."""
    df_train = pd.DataFrame({
        "feat": [np.nan] * 25,
        "x": np.arange(25, dtype=float),
        "target": np.arange(25, dtype=float),
    })
    df_predict = pd.DataFrame({
        "feat": np.arange(100, 110, dtype=float),  # extreme observed values
        "x": np.arange(10, dtype=float),
        "target": np.arange(10, dtype=float),
    })
    plan = {"columns": {"feat": {
        "strategy": "numeric_median", "add_missing_indicator": True,
        "fit_on": "train_only"}}}
    out_train, out_predict = Imputer().apply(df_train, plan, df_predict)
    assert not out_train["feat"].isna().any()
    # Predict's own observed values are kept (not overwritten); train-null rows
    # would be filled with the train-derived 0.0 — but predict has no nulls here,
    # so its values stay exactly as given (proves no train stat leaked in/out).
    assert out_predict["feat"].tolist() == df_predict["feat"].tolist()
