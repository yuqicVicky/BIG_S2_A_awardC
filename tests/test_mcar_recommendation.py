"""MCAR case: low random missingness → numeric_median, no indicator."""

import numpy as np
import pandas as pd
import pytest
from missingness_auditor.profiler import MissingnessProfiler
from missingness_auditor.mechanism import MechanismAuditor
from missingness_auditor.structural import StructuralMissingnessDetector
from missingness_auditor.planner import ImputationPlanner


def _mcar_df(n=1000, miss_prob=0.07, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "feature_a": rng.normal(0, 1, n),
        "feature_b": rng.normal(5, 2, n),
        "feature_c": rng.normal(-3, 1, n),
        "target":    rng.normal(0, 1, n),
    })
    mask = rng.random(n) < miss_prob
    df.loc[mask, "feature_a"] = np.nan
    return df


def _run(df, target="target"):
    profile = MissingnessProfiler(df, target_col=target).profile()
    mech = MechanismAuditor(df, target_col=target).audit()
    struct = StructuralMissingnessDetector(df).detect()
    plan = ImputationPlanner(profile, mech, struct).plan()
    return profile, mech, struct, plan


def test_mcar_mechanism_label():
    df = _mcar_df()
    _, mech, _, _ = _run(df)
    assert mech["columns"]["feature_a"]["mechanism_label"] == "MCAR-compatible"


def test_mcar_no_correlated_features():
    df = _mcar_df()
    _, mech, _, _ = _run(df)
    assert mech["columns"]["feature_a"]["correlated_features"] == []


def test_mcar_target_signal_false():
    df = _mcar_df()
    _, mech, _, _ = _run(df)
    assert mech["columns"]["feature_a"]["target_signal"] is False


def test_mcar_strategy_numeric_median():
    df = _mcar_df()
    _, _, _, plan = _run(df)
    assert plan["columns"]["feature_a"]["strategy"] == "numeric_median"


def test_mcar_no_indicator():
    df = _mcar_df()
    _, _, _, plan = _run(df)
    assert plan["columns"]["feature_a"]["add_missing_indicator"] is False


def test_complete_columns_no_imputation():
    df = _mcar_df()
    _, _, _, plan = _run(df)
    for col in ("feature_b", "feature_c"):
        assert plan["columns"][col]["strategy"] == "no_imputation_needed"


def test_fit_on_train_only():
    df = _mcar_df()
    _, _, _, plan = _run(df)
    assert plan["columns"]["feature_a"]["fit_on"] == "train_only"
