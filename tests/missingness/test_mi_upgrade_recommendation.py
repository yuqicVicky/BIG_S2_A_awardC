"""
Tests for mi_upgrade_recommended field in imputation plan.

Rule (Collins et al. 2001 via van Buuren FIMD Ch5):
  Upgrade to full MICE when: missing rate > 25%, strong covariate correlation > 0.4,
  or missingness correlated with the target variable.
"""

import pandas as pd
import numpy as np
import pytest

from missingness_auditor.profiler import MissingnessProfiler
from missingness_auditor.mechanism import MechanismAuditor
from missingness_auditor.structural import StructuralMissingnessDetector
from missingness_auditor.planner import ImputationPlanner


def _run_plan(df, target_col=None):
    profile = MissingnessProfiler(df, target_col=target_col).profile()
    mechanism = MechanismAuditor(df, target_col=target_col).audit()
    structural = StructuralMissingnessDetector(df).detect()
    planner = ImputationPlanner(profile, mechanism, structural)
    return planner.plan()


def test_high_missing_rate_triggers_mi_upgrade():
    """Column with >25% missing → mi_upgrade_recommended = True."""
    rng = np.random.default_rng(1)
    n = 400
    x = rng.normal(0, 1, n)
    y = rng.normal(0, 1, n)
    miss = rng.random(n) < 0.30
    y_miss = y.copy()
    y_miss[miss] = np.nan
    df = pd.DataFrame({"x": x, "y": y_miss})
    plan = _run_plan(df)
    assert plan["columns"]["y"]["mi_upgrade_recommended"] is True


def test_low_missing_rate_no_mi_upgrade():
    """Column with <10% MCAR missing → mi_upgrade_recommended = False."""
    rng = np.random.default_rng(2)
    n = 400
    x = rng.normal(0, 1, n)
    y = rng.normal(0, 1, n)
    miss = rng.random(n) < 0.05
    y_miss = y.copy()
    y_miss[miss] = np.nan
    df = pd.DataFrame({"x": x, "y": y_miss})
    plan = _run_plan(df)
    assert plan["columns"]["y"]["mi_upgrade_recommended"] is False


def test_target_signal_triggers_mi_upgrade():
    """Column whose missingness correlates with target → mi_upgrade_recommended = True."""
    rng = np.random.default_rng(3)
    n = 500
    target = rng.choice([0, 1], size=n, p=[0.5, 0.5])
    income = rng.normal(50000, 10000, n).astype(float)
    # Income missing more often when target=1
    miss = (target == 1) & (rng.random(n) < 0.50)
    income[miss] = np.nan
    df = pd.DataFrame({"income": income, "target": target})
    plan = _run_plan(df, target_col="target")
    assert plan["columns"]["income"]["mi_upgrade_recommended"] is True


def test_mi_upgrade_field_present_for_all_imputed_cols():
    """Every column that needs imputation must have mi_upgrade_recommended field."""
    rng = np.random.default_rng(42)
    n = 300
    df = pd.DataFrame({
        "a": rng.normal(0, 1, n),
        "b": rng.normal(0, 1, n),
        "c": pd.Categorical(rng.choice(["x", "y"], n)),
    })
    df.loc[rng.random(n) < 0.05, "a"] = np.nan
    df.loc[rng.random(n) < 0.40, "b"] = np.nan
    df["c"] = df["c"].astype(object)
    df.loc[rng.random(n) < 0.10, "c"] = np.nan

    plan = _run_plan(df)
    for col, entry in plan["columns"].items():
        if entry["strategy"] != "no_imputation_needed":
            assert "mi_upgrade_recommended" in entry, (
                f"mi_upgrade_recommended missing for column {col}"
            )


def test_summary_includes_mi_upgrade_list():
    """Plan summary should list columns where full MI is recommended."""
    rng = np.random.default_rng(7)
    n = 500
    x = rng.normal(0, 1, n)
    y = rng.normal(0, 1, n)
    miss = rng.random(n) < 0.35
    y_miss = y.copy()
    y_miss[miss] = np.nan
    df = pd.DataFrame({"x": x, "y": y_miss})
    plan = _run_plan(df)
    mi_cols = plan["summary"]["columns_where_full_mi_recommended"]
    assert isinstance(mi_cols, list)
    assert "y" in mi_cols
