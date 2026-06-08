"""
Group-dependent numeric missingness must NOT be classified as structural absence.

When a numeric column is always missing for a specific categorical group
(e.g. temperature data missing for a jurisdiction that doesn't collect it),
this is group-dependent missingness — not a structural absence pattern.
The structural detector should not flag it, and the planner should
recommend groupwise imputation, not structural_zero_plus_indicator.
"""

import numpy as np
import pandas as pd
import pytest

from missingness_auditor.structural import StructuralMissingnessDetector
from missingness_auditor.mechanism import MechanismAuditor
from missingness_auditor.profiler import MissingnessProfiler
from missingness_auditor.planner import ImputationPlanner


def _make_group_dependent_df(n_per_group: int = 100, seed: int = 42) -> pd.DataFrame:
    """
    temp_avg_f is always missing for jurisdiction 'remote_territory'.
    It has real positive values for all other jurisdictions.
    This is group-dependent missingness, NOT structural absence
    (the zero value is not co-occurring — temp values are all positive).
    """
    rng = np.random.default_rng(seed)
    jurisdictions = ["state_a"] * n_per_group + ["state_b"] * n_per_group + ["remote_territory"] * n_per_group
    n = len(jurisdictions)

    temp = rng.uniform(40, 100, n)  # realistic temperatures, all positive
    temp[np.array(jurisdictions) == "remote_territory"] = np.nan

    return pd.DataFrame({
        "jurisdiction": jurisdictions,
        "temp_avg_f": temp,
        "population": rng.integers(10_000, 1_000_000, n).astype(float),
        "target": rng.normal(0, 1, n),
    })


def test_group_dependent_not_in_structural_pairs():
    df = _make_group_dependent_df()
    result = StructuralMissingnessDetector(df).detect()
    numeric_cols_in_pairs = [p["numeric_col"] for p in result["structural_pairs"]]
    assert "temp_avg_f" not in numeric_cols_in_pairs, (
        "temp_avg_f should not appear as a structural pair; "
        "it is group-dependent missingness, not structural absence"
    )


def test_group_dependent_not_flagged_as_structural():
    df = _make_group_dependent_df()
    result = StructuralMissingnessDetector(df).detect()
    assert "temp_avg_f" not in result["column_flags"], (
        "temp_avg_f should not be flagged as structural"
    )


def test_group_dependent_mechanism_label():
    df = _make_group_dependent_df()
    mech = MechanismAuditor(df, target_col="target").audit()
    label = mech["columns"]["temp_avg_f"]["mechanism_label"]
    assert label == "group-dependent missingness", (
        f"Expected 'group-dependent missingness', got {label!r}"
    )


def test_group_dependent_evidence_contains_jurisdiction():
    df = _make_group_dependent_df()
    mech = MechanismAuditor(df, target_col="target").audit()
    gdep = mech["columns"]["temp_avg_f"].get("group_dependency_evidence", {}) or {}
    assert gdep.get("detected") is True
    assert gdep.get("top_categorical_feature") == "jurisdiction"


def test_group_dependent_strategy_is_groupwise_not_structural_zero():
    df = _make_group_dependent_df()
    profile = MissingnessProfiler(df, target_col="target").profile()
    mech = MechanismAuditor(df, target_col="target").audit()
    struct = StructuralMissingnessDetector(df).detect()
    plan = ImputationPlanner(profile, mech, struct).plan()

    strategy = plan["columns"]["temp_avg_f"]["strategy"]
    assert strategy == "groupwise_numeric_median_plus_indicator", (
        f"Expected groupwise_numeric_median_plus_indicator, got {strategy!r}. "
        "Filling group-dependent missing temps with 0 is unsafe."
    )
    assert plan["columns"]["temp_avg_f"]["add_missing_indicator"] is True
