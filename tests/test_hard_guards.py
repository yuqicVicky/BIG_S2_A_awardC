"""Hard guard #1: the target column is never imputed — anywhere."""

import numpy as np
import pandas as pd

from missingness_auditor import MissingnessAuditor, Imputer, MICEImputer


def test_target_excluded_from_plan(audited):
    _, _, results = audited
    plan = results["imputation_plan"]["columns"]
    assert plan["target"]["strategy"] == "no_imputation_needed"
    assert plan["target"]["add_missing_indicator"] is False
    assert results["imputation_plan"]["summary"]  # summary present


def test_target_values_unchanged_after_imputation(mixed_df):
    # Give the target some missing values to prove they are preserved, not filled.
    df = mixed_df.copy()
    df.loc[df.index[:5], "target"] = np.nan
    aud = MissingnessAuditor(df, target_col="target")
    results = aud.run()
    imputed = aud.apply_imputation(df, results["imputation_plan"])

    before = df["target"]
    after = imputed["target"]
    # Same missing positions and same observed values.
    assert before.isna().sum() == after.isna().sum() == 5
    assert np.allclose(
        before.dropna().to_numpy(), after.loc[before.notna()].to_numpy()
    )


def test_mice_imputer_preserves_target():
    rng = np.random.RandomState(0)
    n = 200
    df = pd.DataFrame({
        "a": rng.normal(0, 1, n),
        "b": rng.normal(5, 2, n),
        "target": rng.normal(10, 3, n),
    })
    df.loc[rng.rand(n) < 0.2, "a"] = np.nan
    df.loc[df.index[:8], "target"] = np.nan  # target has holes too

    out = MICEImputer(m=3).fit_transform(df, target_col="target")
    for completed in out:
        # 'a' is fully imputed, target's NaNs are left exactly as they were.
        assert completed["a"].isna().sum() == 0
        assert completed["target"].isna().sum() == 8
