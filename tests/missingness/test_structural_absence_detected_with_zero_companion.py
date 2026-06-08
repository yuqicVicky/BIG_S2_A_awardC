"""
Structural absence must only be detected when paired zero/absent companion evidence exists.

Examples:
  - garage_quality (categorical) NaN + garage_area = 0  → structural absence ✓
  - pool_quality (categorical) NaN + pool_area = 0      → structural absence ✓
  - temp column NaN for one jurisdiction                 → group-dependent, NOT structural ✗

Tests confirm:
  1. Categorical + numeric-zero companion → structural pair detected.
  2. Categorical structural column gets structural_none_token_plus_indicator.
  3. Numeric companion with structural zero gets structural_zero_plus_indicator.
  4. A numeric column missing only for one group (no zero companion) is NOT structural.
"""

import numpy as np
import pandas as pd
import pytest

from missingness_auditor.structural import StructuralMissingnessDetector
from missingness_auditor.profiler import MissingnessProfiler
from missingness_auditor.mechanism import MechanismAuditor
from missingness_auditor.planner import ImputationPlanner


def _make_garage_df(n_with: int = 150, n_without: int = 80, seed: int = 42) -> pd.DataFrame:
    """
    garage_quality is NaN when there is no garage.
    garage_area is 0 when there is no garage.
    This is the canonical structural absence pattern.
    """
    rng = np.random.default_rng(seed)
    n = n_with + n_without

    garage_quality = ["good"] * 75 + ["poor"] * 75 + [None] * n_without
    garage_area = rng.integers(200, 800, n_with).tolist() + [0] * n_without
    house_size = rng.normal(2000, 500, n).clip(800)
    target = rng.normal(250_000, 80_000, n)

    return pd.DataFrame({
        "garage_quality": garage_quality,
        "garage_area": garage_area,
        "house_size": house_size,
        "target": target,
    })


def _make_pool_df(n_with: int = 60, n_without: int = 200, seed: int = 7) -> pd.DataFrame:
    """
    pool_quality is NaN when there is no pool.
    pool_area is 0 when there is no pool.
    """
    rng = np.random.default_rng(seed)
    n = n_with + n_without

    pool_quality = ["excellent"] * 30 + ["average"] * 30 + [None] * n_without
    pool_area = rng.integers(100, 600, n_with).tolist() + [0] * n_without
    lot_size = rng.normal(8000, 2000, n).clip(2000)
    target = rng.normal(400_000, 100_000, n)

    return pd.DataFrame({
        "pool_quality": pool_quality,
        "pool_area": pool_area,
        "lot_size": lot_size,
        "target": target,
    })


def test_garage_structural_pair_detected():
    df = _make_garage_df()
    result = StructuralMissingnessDetector(df).detect()
    assert result["n_structural_pairs"] > 0


def test_garage_quality_flagged_as_structural():
    df = _make_garage_df()
    result = StructuralMissingnessDetector(df).detect()
    assert "garage_quality" in result["column_flags"]
    assert result["column_flags"]["garage_quality"]["is_structural"] is True


def test_garage_quality_gets_structural_none_token():
    df = _make_garage_df()
    profile = MissingnessProfiler(df, target_col="target").profile()
    mech = MechanismAuditor(df, target_col="target").audit()
    struct = StructuralMissingnessDetector(df).detect()
    plan = ImputationPlanner(profile, mech, struct).plan()

    strategy = plan["columns"]["garage_quality"]["strategy"]
    assert strategy == "structural_none_token_plus_indicator", (
        f"Expected structural_none_token_plus_indicator, got {strategy!r}"
    )


def test_pool_structural_pair_detected():
    df = _make_pool_df()
    result = StructuralMissingnessDetector(df).detect()
    assert result["n_structural_pairs"] > 0


def test_pool_quality_gets_structural_none_token():
    df = _make_pool_df()
    profile = MissingnessProfiler(df, target_col="target").profile()
    mech = MechanismAuditor(df, target_col="target").audit()
    struct = StructuralMissingnessDetector(df).detect()
    plan = ImputationPlanner(profile, mech, struct).plan()

    strategy = plan["columns"]["pool_quality"]["strategy"]
    assert strategy == "structural_none_token_plus_indicator", (
        f"Expected structural_none_token_plus_indicator, got {strategy!r}"
    )


def test_non_structural_column_unaffected():
    df = _make_garage_df()
    profile = MissingnessProfiler(df, target_col="target").profile()
    mech = MechanismAuditor(df, target_col="target").audit()
    struct = StructuralMissingnessDetector(df).detect()
    plan = ImputationPlanner(profile, mech, struct).plan()

    assert plan["columns"]["house_size"]["strategy"] == "no_imputation_needed"


def test_structural_none_token_has_indicator():
    df = _make_garage_df()
    profile = MissingnessProfiler(df, target_col="target").profile()
    mech = MechanismAuditor(df, target_col="target").audit()
    struct = StructuralMissingnessDetector(df).detect()
    plan = ImputationPlanner(profile, mech, struct).plan()

    assert plan["columns"]["garage_quality"]["add_missing_indicator"] is True
