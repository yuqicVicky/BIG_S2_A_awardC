"""MAR-like case: missingness correlated with an observed feature → add indicator."""

import numpy as np
import pandas as pd
import pytest

from missingness_auditor.profiler import MissingnessProfiler
from missingness_auditor.mechanism import MechanismAuditor
from missingness_auditor.structural import StructuralMissingnessDetector
from missingness_auditor.recommender import ImputationRecommender


def _make_mar_df(n=800, seed=0):
    """income is missing when education_level == 0 (MAR: depends on observed feature)."""
    rng = np.random.default_rng(seed)
    education = rng.integers(0, 3, n).astype(float)   # 0=low, 1=mid, 2=high
    income = rng.normal(50_000, 10_000, n)
    income_mar = income.astype(float).copy()
    income_mar[education == 0] = np.nan
    target = rng.normal(0, 1, n)

    return pd.DataFrame({
        "education_level": education,
        "income": income_mar,
        "unrelated": rng.normal(0, 1, n),
        "target": target,
    })


def test_mar_mechanism_label():
    df = _make_mar_df()
    mech = MechanismAuditor(df, target_col="target").audit()
    label = mech["columns"]["income"]["mechanism_label"]
    assert label == "MAR-like evidence"


def test_mar_correlated_feature_found():
    df = _make_mar_df()
    mech = MechanismAuditor(df, target_col="target").audit()
    corr_features = [e["feature"] for e in mech["columns"]["income"]["correlated_features"]]
    assert "education_level" in corr_features


def test_mar_recommends_indicator():
    df = _make_mar_df()
    profile = MissingnessProfiler(df, target_col="target").profile()
    mechanism = MechanismAuditor(df, target_col="target").audit()
    structural = StructuralMissingnessDetector(df).detect()
    plan = ImputationRecommender(profile, mechanism, structural).recommend()

    entry = plan["columns"]["income"]
    assert entry["strategy"] == "numeric_median_plus_indicator"
    assert entry["add_missing_indicator"] is True


def test_mar_plan_summary_includes_income():
    df = _make_mar_df()
    profile = MissingnessProfiler(df, target_col="target").profile()
    mechanism = MechanismAuditor(df, target_col="target").audit()
    structural = StructuralMissingnessDetector(df).detect()
    plan = ImputationRecommender(profile, mechanism, structural).recommend()

    assert "income" in plan["summary"]["columns_needing_missing_indicator"]
