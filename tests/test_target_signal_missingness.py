"""MNAR case: missingness correlated with target → MNAR/structural concern."""

import numpy as np
import pandas as pd
import pytest
from missingness_auditor.mechanism import MechanismAuditor
from missingness_auditor.profiler import MissingnessProfiler
from missingness_auditor.structural import StructuralMissingnessDetector
from missingness_auditor.planner import ImputationPlanner


def _mnar_df(n=600, seed=7):
    rng = np.random.default_rng(seed)
    target = rng.binomial(1, 0.30, n).astype(float)
    risk_score = rng.normal(650, 90, n).astype(float)
    risk_score[target == 1] = np.nan   # MNAR: missing because of outcome
    return pd.DataFrame({
        "risk_score":    risk_score,
        "other_feature": rng.normal(0, 1, n),
        "target":        target,
    })


def test_mnar_mechanism_label():
    df = _mnar_df()
    mech = MechanismAuditor(df, target_col="target").audit()
    assert mech["columns"]["risk_score"]["mechanism_label"] == "target-associated missingness"


def test_mnar_target_signal_true():
    df = _mnar_df()
    mech = MechanismAuditor(df, target_col="target").audit()
    assert mech["columns"]["risk_score"]["target_signal"] is True


def test_mnar_correlation_high():
    df = _mnar_df()
    mech = MechanismAuditor(df, target_col="target").audit()
    r = mech["columns"]["risk_score"]["target_correlation"]
    assert r is not None and r > 0.10


def test_mnar_recommends_indicator():
    df = _mnar_df()
    profile = MissingnessProfiler(df, target_col="target").profile()
    mech = MechanismAuditor(df, target_col="target").audit()
    struct = StructuralMissingnessDetector(df).detect()
    plan = ImputationPlanner(profile, mech, struct).plan()
    assert plan["columns"]["risk_score"]["add_missing_indicator"] is True
    assert plan["columns"]["risk_score"]["strategy"] == "numeric_median_plus_indicator"


def test_complete_column_unaffected():
    df = _mnar_df()
    profile = MissingnessProfiler(df, target_col="target").profile()
    mech = MechanismAuditor(df, target_col="target").audit()
    struct = StructuralMissingnessDetector(df).detect()
    plan = ImputationPlanner(profile, mech, struct).plan()
    assert plan["columns"]["other_feature"]["strategy"] == "no_imputation_needed"
