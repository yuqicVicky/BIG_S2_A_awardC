"""MCAR-compatible case: low, random missingness → numeric_median, no indicator."""

import numpy as np
import pandas as pd
import pytest

from missingness_auditor.profiler import MissingnessProfiler
from missingness_auditor.mechanism import MechanismAuditor
from missingness_auditor.structural import StructuralMissingnessDetector
from missingness_auditor.recommender import ImputationRecommender


def _make_mcar_df(n=1000, miss_prob=0.07, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "feature_a": rng.normal(0, 1, n),
        "feature_b": rng.normal(5, 2, n),
        "feature_c": rng.normal(-3, 1, n),
        "target": rng.normal(0, 1, n),
    })
    mask = rng.random(n) < miss_prob
    df.loc[mask, "feature_a"] = np.nan
    return df


def test_mcar_mechanism_label():
    df = _make_mcar_df()
    mech = MechanismAuditor(df, target_col="target").audit()
    label = mech["columns"]["feature_a"]["mechanism_label"]
    assert label == "MCAR-compatible"


def test_mcar_no_correlated_features():
    df = _make_mcar_df()
    mech = MechanismAuditor(df, target_col="target").audit()
    assert len(mech["columns"]["feature_a"]["correlated_features"]) == 0


def test_mcar_target_signal_false():
    df = _make_mcar_df()
    mech = MechanismAuditor(df, target_col="target").audit()
    assert mech["columns"]["feature_a"]["target_signal"] is False


def test_mcar_recommends_numeric_median():
    df = _make_mcar_df()
    profile = MissingnessProfiler(df, target_col="target").profile()
    mechanism = MechanismAuditor(df, target_col="target").audit()
    structural = StructuralMissingnessDetector(df).detect()
    plan = ImputationRecommender(profile, mechanism, structural).recommend()

    entry = plan["columns"]["feature_a"]
    assert entry["strategy"] == "numeric_median"
    assert entry["add_missing_indicator"] is False
    assert entry["fit_on"] == "train_only"


def test_no_imputation_for_complete_columns():
    df = _make_mcar_df()
    profile = MissingnessProfiler(df, target_col="target").profile()
    mechanism = MechanismAuditor(df, target_col="target").audit()
    structural = StructuralMissingnessDetector(df).detect()
    plan = ImputationRecommender(profile, mechanism, structural).recommend()

    for col in ("feature_b", "feature_c"):
        assert plan["columns"][col]["strategy"] == "no_imputation_needed"
