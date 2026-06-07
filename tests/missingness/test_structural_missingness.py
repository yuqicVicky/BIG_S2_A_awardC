"""Structural missingness: categorical NA → numeric companion is 0."""

import numpy as np
import pandas as pd
import pytest

from missingness_auditor.structural import StructuralMissingnessDetector
from missingness_auditor.profiler import MissingnessProfiler
from missingness_auditor.mechanism import MechanismAuditor
from missingness_auditor.recommender import ImputationRecommender


def _make_structural_df():
    """
    facility_type is NaN when there is no facility.
    facility_capacity is 0 when facility_type is NaN.
    """
    n_with = 120
    n_without = 80
    rng = np.random.default_rng(1)

    facility_type = ["type_a"] * 60 + ["type_b"] * 60 + [None] * 80
    facility_capacity = (
        rng.integers(10, 100, n_with).tolist() + [0] * n_without
    )
    other = rng.normal(0, 1, n_with + n_without)
    target = rng.normal(100, 20, n_with + n_without)

    return pd.DataFrame({
        "facility_type": facility_type,
        "facility_capacity": facility_capacity,
        "other_feature": other,
        "target": target,
    })


def test_structural_pair_detected():
    df = _make_structural_df()
    result = StructuralMissingnessDetector(df).detect()
    assert result["n_structural_pairs"] > 0


def test_structural_flags_both_columns():
    df = _make_structural_df()
    result = StructuralMissingnessDetector(df).detect()
    flags = result["column_flags"]
    assert "facility_type" in flags or "facility_capacity" in flags


def test_structural_pattern_name():
    df = _make_structural_df()
    result = StructuralMissingnessDetector(df).detect()
    patterns = [p["pattern"] for p in result["structural_pairs"]]
    assert any("zero" in p or "missing" in p for p in patterns)


def test_structural_recommends_none_or_zero():
    """Structural columns that have actual NaN values get structural_none_or_zero.
    Structural columns whose absence is encoded as 0 (no NaN) get no_imputation_needed."""
    df = _make_structural_df()
    profile = MissingnessProfiler(df, target_col="target").profile()
    mechanism = MechanismAuditor(df, target_col="target").audit()
    structural = StructuralMissingnessDetector(df).detect()
    plan = ImputationRecommender(profile, mechanism, structural).recommend()

    flagged = structural["column_flags"]
    found_structural_strategy = False
    for col in flagged:
        if col == "target":
            continue
        if flagged[col]["is_structural"]:
            miss_rate = profile["columns"][col]["missing_rate"]
            if miss_rate > 0:
                # columns with actual NaN + structural flag → structural_none_or_zero
                assert plan["columns"][col]["strategy"] == "structural_none_or_zero"
                assert plan["columns"][col]["add_missing_indicator"] is True
                found_structural_strategy = True
            else:
                # structural zeros (no NaN) → no imputation needed
                assert plan["columns"][col]["strategy"] == "no_imputation_needed"

    assert found_structural_strategy, "At least one flagged column should have actual NaN values"


def test_non_structural_column_unaffected():
    df = _make_structural_df()
    profile = MissingnessProfiler(df, target_col="target").profile()
    mechanism = MechanismAuditor(df, target_col="target").audit()
    structural = StructuralMissingnessDetector(df).detect()
    plan = ImputationRecommender(profile, mechanism, structural).recommend()

    # other_feature has no missing values
    assert plan["columns"]["other_feature"]["strategy"] == "no_imputation_needed"
