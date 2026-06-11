"""Structural absence is filled with NONE/0 + indicator — never the mean/mode."""

import numpy as np
import pandas as pd

from missingness_auditor import (
    StructuralMissingnessDetector,
    MissingnessAuditor,
    Imputer,
)


def test_structural_pair_detected(mixed_df):
    result = StructuralMissingnessDetector(mixed_df).detect()
    flagged = {
        (p["categorical_col"], p["numeric_col"]) for p in result["structural_pairs"]
    }
    assert ("pool_type", "pool_area") in flagged


def test_structural_categorical_filled_with_none_token(mixed_df):
    aud = MissingnessAuditor(mixed_df, target_col="target")
    results = aud.run()
    plan = results["imputation_plan"]["columns"]
    # pool_type is missing and structural → NONE token, not mode imputation.
    assert plan["pool_type"]["strategy"] == "structural_none_token_plus_indicator"

    imputed = aud.apply_imputation(mixed_df, results["imputation_plan"])
    assert imputed["pool_type"].isna().sum() == 0
    # The filled rows carry the explicit NONE token (absence), not a real category.
    filled_token_rows = imputed.loc[mixed_df["pool_type"].isna(), "pool_type"]
    assert (filled_token_rows == "NONE").all()


def test_structural_zero_fill_is_not_mean():
    # A numeric column whose missing cells encode absence → must be 0, not the mean.
    df = pd.DataFrame({"area": [50.0, np.nan, 80.0, np.nan, 60.0]})
    plan = {"columns": {"area": {
        "strategy": "structural_zero_plus_indicator",
        "add_missing_indicator": True,
    }}}
    out = Imputer().apply(df, plan)
    assert out.loc[df["area"].isna(), "area"].eq(0).all()
    assert "area_was_missing" in out.columns
