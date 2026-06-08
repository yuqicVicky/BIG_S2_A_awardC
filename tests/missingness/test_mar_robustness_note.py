"""
Tests for MAR robustness note derived from Collins et al. (2001) via van Buuren FIMD Ch5.

Rule: when missing rate < 25% and max covariate correlation < 0.4,
the MAR assumption is likely robust — label it as such.
When missing rate >= 25% or strong correlation found, flag for MI upgrade.
"""

import pandas as pd
import numpy as np
import pytest

from missingness_auditor.mechanism import MechanismAuditor


def _make_df_mcar_low_miss(seed=42):
    """5% MCAR missingness (~25 rows), no correlation with any feature.
    Uses 5% so n_missing > _MIN_ROWS_FOR_ANALYSIS=20 to avoid 'insufficient evidence' label.
    """
    rng = np.random.default_rng(seed)
    n = 500
    x = rng.normal(0, 1, n)
    y = rng.normal(0, 1, n)
    z = rng.normal(0, 1, n)
    miss_mask = rng.random(n) < 0.05
    z_miss = z.copy().astype(float)
    z_miss[miss_mask] = np.nan
    return pd.DataFrame({"x": x, "y": y, "z": z_miss})


def _make_df_high_miss_rate(seed=42):
    """30% missingness, no strong correlation — above Collins threshold."""
    rng = np.random.default_rng(seed)
    n = 500
    x = rng.normal(0, 1, n)
    y = rng.normal(0, 1, n)
    miss_mask = rng.random(n) < 0.30
    y_miss = y.copy()
    y_miss[miss_mask] = np.nan
    return pd.DataFrame({"x": x, "y": y_miss})


def _make_df_strong_corr_miss(seed=42):
    """15% missingness, but missingness strongly correlated with x (r > 0.4)."""
    rng = np.random.default_rng(seed)
    n = 500
    x = rng.normal(0, 1, n)
    y = rng.normal(0, 1, n)
    # Make missingness highly dependent on x
    prob = 1 / (1 + np.exp(-3 * x))  # high x → high probability of missing
    miss_mask = rng.random(n) < prob * 0.30
    y_miss = y.copy()
    y_miss[miss_mask] = np.nan
    return pd.DataFrame({"x": x, "y": y_miss})


def test_low_miss_mcar_gets_robust_note():
    """Low missing rate + MCAR-compatible → mar_robustness_note = likely_robust_per_collins2001."""
    df = _make_df_mcar_low_miss()
    auditor = MechanismAuditor(df)
    result = auditor.audit()
    col_info = result["columns"]["z"]
    assert col_info["mechanism_label"] == "MCAR-compatible"
    assert col_info["mar_robustness_note"] == "likely_robust_per_collins2001"


def test_high_miss_rate_gets_upgrade_note():
    """30% missing rate → mar_robustness_note flags mechanism matters."""
    df = _make_df_high_miss_rate()
    auditor = MechanismAuditor(df)
    result = auditor.audit()
    col_info = result["columns"]["y"]
    assert col_info["mar_robustness_note"] == "mechanism_assumption_may_matter_consider_full_mi"


def test_strong_corr_gets_upgrade_note():
    """Missingness strongly correlated with observed feature → upgrade note."""
    df = _make_df_strong_corr_miss()
    auditor = MechanismAuditor(df)
    result = auditor.audit()
    col_info = result["columns"]["y"]
    # Either MAR-like or MCAR depending on threshold, but robustness note should warn
    rob_note = col_info["mar_robustness_note"]
    # Strong correlation should push to "may matter" note
    assert rob_note in (
        "mechanism_assumption_may_matter_consider_full_mi",
        "insufficient_evidence_to_assess",
    ), f"Expected upgrade note but got: {rob_note}"


def test_mar_robustness_note_present_for_all_missing_cols():
    """Every column with missingness should have the mar_robustness_note field."""
    rng = np.random.default_rng(99)
    n = 300
    df = pd.DataFrame({
        "a": rng.normal(0, 1, n),
        "b": rng.normal(0, 1, n),
        "c": rng.normal(0, 1, n),
    })
    df.loc[rng.random(n) < 0.05, "a"] = np.nan
    df.loc[rng.random(n) < 0.30, "b"] = np.nan

    auditor = MechanismAuditor(df)
    result = auditor.audit()
    for col in ["a", "b"]:
        assert "mar_robustness_note" in result["columns"][col], (
            f"mar_robustness_note missing for column {col}"
        )
