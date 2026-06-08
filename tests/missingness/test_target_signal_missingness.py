"""MNAR case: missingness correlated with target → MNAR/structural concern."""

import numpy as np
import pandas as pd
import pytest

from missingness_auditor.mechanism import MechanismAuditor
from missingness_auditor.profiler import MissingnessProfiler
from missingness_auditor.structural import StructuralMissingnessDetector
from missingness_auditor.recommender import ImputationRecommender


def _make_mnar_df(n=600, seed=7):
    """credit_score is always missing when the customer defaulted (target=1)."""
    rng = np.random.default_rng(seed)
    target = rng.binomial(1, 0.30, n).astype(float)
    credit_score = rng.normal(650, 80, n).astype(float)
    credit_score[target == 1] = np.nan   # MNAR: missing because of outcome
    other = rng.normal(0, 1, n)

    return pd.DataFrame({
        "credit_score": credit_score,
        "other_feature": other,
        "target": target,
    })


def test_mnar_mechanism_label():
    df = _make_mnar_df()
    mech = MechanismAuditor(df, target_col="target").audit()
    label = mech["columns"]["credit_score"]["mechanism_label"]
    assert label == "target-associated missingness"


def test_mnar_target_signal_true():
    df = _make_mnar_df()
    mech = MechanismAuditor(df, target_col="target").audit()
    assert mech["columns"]["credit_score"]["target_signal"] is True


def test_mnar_target_correlation_high():
    df = _make_mnar_df()
    mech = MechanismAuditor(df, target_col="target").audit()
    r = mech["columns"]["credit_score"]["target_correlation"]
    assert r is not None
    assert r > 0.10


def test_mnar_recommends_indicator():
    df = _make_mnar_df()
    profile = MissingnessProfiler(df, target_col="target").profile()
    mechanism = MechanismAuditor(df, target_col="target").audit()
    structural = StructuralMissingnessDetector(df).detect()
    plan = ImputationRecommender(profile, mechanism, structural).recommend()

    entry = plan["columns"]["credit_score"]
    assert entry["add_missing_indicator"] is True
    assert entry["strategy"] in {"numeric_median_plus_indicator"}


def test_mnar_other_feature_unaffected():
    """other_feature has no missing values → no_imputation_needed."""
    df = _make_mnar_df()
    profile = MissingnessProfiler(df, target_col="target").profile()
    mechanism = MechanismAuditor(df, target_col="target").audit()
    structural = StructuralMissingnessDetector(df).detect()
    plan = ImputationRecommender(profile, mechanism, structural).recommend()

    assert plan["columns"]["other_feature"]["strategy"] == "no_imputation_needed"
