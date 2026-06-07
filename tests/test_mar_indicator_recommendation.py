"""MAR-like case: missingness correlated with an observed feature → add indicator."""

import numpy as np
import pandas as pd
import pytest
from missingness_auditor.profiler import MissingnessProfiler
from missingness_auditor.mechanism import MechanismAuditor
from missingness_auditor.structural import StructuralMissingnessDetector
from missingness_auditor.planner import ImputationPlanner


def _mar_df(n=800, seed=0):
    rng = np.random.default_rng(seed)
    education = rng.integers(0, 3, n).astype(float)   # 0=low, 1=mid, 2=high
    income = rng.normal(50_000, 10_000, n).astype(float)
    income[education == 0] = np.nan                    # MAR: missing when education low
    return pd.DataFrame({
        "education_level": education,
        "income":          income,
        "unrelated":       rng.normal(0, 1, n),
        "target":          rng.normal(0, 1, n),
    })


def _run(df):
    profile = MissingnessProfiler(df, target_col="target").profile()
    mech = MechanismAuditor(df, target_col="target").audit()
    struct = StructuralMissingnessDetector(df).detect()
    plan = ImputationPlanner(profile, mech, struct).plan()
    return profile, mech, struct, plan


def test_mar_mechanism_label():
    _, mech, _, _ = _run(_mar_df())
    assert mech["columns"]["income"]["mechanism_label"] == "MAR-like evidence"


def test_mar_correlated_feature_found():
    _, mech, _, _ = _run(_mar_df())
    features = [e["feature"] for e in mech["columns"]["income"]["correlated_features"]]
    assert "education_level" in features


def test_mar_strategy_has_indicator():
    _, _, _, plan = _run(_mar_df())
    assert plan["columns"]["income"]["strategy"] == "numeric_median_plus_indicator"
    assert plan["columns"]["income"]["add_missing_indicator"] is True


def test_mar_indicator_in_summary():
    _, _, _, plan = _run(_mar_df())
    assert "income" in plan["summary"]["columns_needing_missing_indicator"]


def test_unrelated_column_no_indicator():
    _, _, _, plan = _run(_mar_df())
    assert plan["columns"]["unrelated"]["strategy"] == "no_imputation_needed"
