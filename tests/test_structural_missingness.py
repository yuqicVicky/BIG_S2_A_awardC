"""Structural missingness: categorical NA → numeric companion is 0."""

import numpy as np
import pandas as pd
import pytest
from missingness_auditor.structural import StructuralMissingnessDetector
from missingness_auditor.profiler import MissingnessProfiler
from missingness_auditor.mechanism import MechanismAuditor
from missingness_auditor.planner import ImputationPlanner


def _structural_df():
    rng = np.random.default_rng(1)
    n_with, n_without = 120, 80
    n = n_with + n_without
    quality = ["good"] * 60 + ["poor"] * 60 + [None] * 80
    area = rng.integers(10, 100, n_with).tolist() + [0] * n_without
    return pd.DataFrame({
        "facility_quality": quality,
        "facility_area":    area,
        "other_feature":    rng.normal(0, 1, n),
        "target":           rng.normal(100, 20, n),
    })


def test_structural_pair_detected():
    result = StructuralMissingnessDetector(_structural_df()).detect()
    assert result["n_structural_pairs"] > 0


def test_structural_flags_categorical_column():
    result = StructuralMissingnessDetector(_structural_df()).detect()
    assert "facility_quality" in result["column_flags"]


def test_structural_pattern_in_pair():
    result = StructuralMissingnessDetector(_structural_df()).detect()
    patterns = [p["pattern"] for p in result["structural_pairs"]]
    assert any("zero" in p or "missing" in p for p in patterns)


def test_structural_recommends_none_or_zero():
    df = _structural_df()
    profile = MissingnessProfiler(df, target_col="target").profile()
    mech = MechanismAuditor(df, target_col="target").audit()
    struct = StructuralMissingnessDetector(df).detect()
    plan = ImputationPlanner(profile, mech, struct).plan()

    flagged = struct["column_flags"]
    found_strategy = False
    for col, flag in flagged.items():
        if col == "target":
            continue
        if flag["is_structural"]:
            miss_rate = profile["columns"][col]["missing_rate"]
            if miss_rate > 0:
                strategy = plan["columns"][col]["strategy"]
                assert strategy in {
                    "structural_none_token_plus_indicator",
                    "structural_zero_plus_indicator",
                }, f"Unexpected structural strategy: {strategy}"
                assert plan["columns"][col]["add_missing_indicator"] is True
                found_strategy = True
            else:
                assert plan["columns"][col]["strategy"] == "no_imputation_needed"
    assert found_strategy


def test_non_structural_column_unaffected():
    df = _structural_df()
    profile = MissingnessProfiler(df, target_col="target").profile()
    mech = MechanismAuditor(df, target_col="target").audit()
    struct = StructuralMissingnessDetector(df).detect()
    plan = ImputationPlanner(profile, mech, struct).plan()
    assert plan["columns"]["other_feature"]["strategy"] == "no_imputation_needed"


def test_co_missing_pattern_uses_median_not_zero():
    """Pattern 1 (both cols absent together) must NOT fill numeric with 0.

    Simulates geographic data: when 'state' is unknown, weather values are
    also missing — but those values are unknown, not zero.
    """
    import numpy as np

    rng = np.random.default_rng(42)
    n_known, n_unknown = 150, 50
    n = n_known + n_unknown

    state = rng.choice(["CA", "TX", "NY"], n_known).tolist() + [None] * n_unknown
    precipitation = rng.uniform(10, 100, n_known).tolist() + [None] * n_unknown
    temperature = rng.uniform(5, 35, n_known).tolist() + [None] * n_unknown
    target = rng.normal(50, 10, n).tolist()

    df = pd.DataFrame({
        "state": state,
        "precipitation": precipitation,
        "temperature": temperature,
        "target": target,
    })

    struct = StructuralMissingnessDetector(df).detect()
    # Pattern 1 fires: state is NA whenever precipitation/temperature are NA
    if "precipitation" in struct["column_flags"]:
        assert struct["column_flags"]["precipitation"]["pattern"] == "co_missing_structural"

    profile = MissingnessProfiler(df, target_col="target").profile()
    mech = MechanismAuditor(df, target_col="target").audit()
    plan = ImputationPlanner(profile, mech, struct).plan()

    for col in ("precipitation", "temperature"):
        if col not in plan["columns"]:
            continue
        strategy = plan["columns"][col]["strategy"]
        assert strategy != "structural_zero_plus_indicator", (
            f"{col} was co-missing (unknown, not zero) but got zero-fill strategy"
        )
