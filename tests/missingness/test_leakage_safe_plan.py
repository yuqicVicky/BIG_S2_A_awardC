"""Leakage-safe imputation protocol checks."""

import numpy as np
import pandas as pd
import pytest

from missingness_auditor.profiler import MissingnessProfiler
from missingness_auditor.mechanism import MechanismAuditor
from missingness_auditor.structural import StructuralMissingnessDetector
from missingness_auditor.recommender import ImputationRecommender
from missingness_auditor.leakage_check import LeakageSafeImputationChecker


def _make_mixed_df():
    rng = np.random.default_rng(5)
    n = 200
    return pd.DataFrame({
        "num_col": np.where(rng.random(n) < 0.15, np.nan, rng.normal(0, 1, n)),
        "cat_col": np.where(rng.random(n) < 0.20, None, rng.choice(["a", "b", "c"], n)),
        "complete": rng.normal(5, 1, n),
        "target": rng.normal(0, 1, n),
    })


def _build_plan(df):
    profile = MissingnessProfiler(df, target_col="target").profile()
    mechanism = MechanismAuditor(df, target_col="target").audit()
    structural = StructuralMissingnessDetector(df).detect()
    return ImputationRecommender(profile, mechanism, structural).recommend()


def test_no_high_leakage_risk():
    df = _make_mixed_df()
    plan = _build_plan(df)
    checker = LeakageSafeImputationChecker(plan, has_predict_df=False)
    result = checker.check()
    assert result["summary"]["high_leakage_risk"] == 0


def test_imputed_columns_have_train_only_fit():
    df = _make_mixed_df()
    plan = _build_plan(df)
    for col, entry in plan["columns"].items():
        if entry["strategy"] != "no_imputation_needed":
            assert entry["fit_on"] == "train_only"


def test_global_protocol_present():
    df = _make_mixed_df()
    plan = _build_plan(df)
    checker = LeakageSafeImputationChecker(plan)
    result = checker.check()
    protocol = result["global_protocol"]
    assert "rule" in protocol
    assert "sklearn_pattern" in protocol
    assert "indicator_pattern" in protocol


def test_no_imputation_findings_have_none_risk():
    df = _make_mixed_df()
    plan = _build_plan(df)
    checker = LeakageSafeImputationChecker(plan)
    result = checker.check()
    for finding in result["findings"]:
        if finding["strategy"] == "no_imputation_needed":
            assert finding["leakage_risk"] == "none"


def test_summary_counts_correct():
    df = _make_mixed_df()
    plan = _build_plan(df)
    checker = LeakageSafeImputationChecker(plan)
    result = checker.check()
    s = result["summary"]
    total = s["high_leakage_risk"] + s["medium_leakage_risk"] + s["low_or_none_leakage_risk"]
    assert total == s["total_columns_checked"]
    assert s["total_columns_checked"] == len(plan["columns"])
