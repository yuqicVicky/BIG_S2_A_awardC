"""Hard guard #2: imputation statistics are fitted on train only.

The decisive test: inject extreme values into the predict frame and prove the
fill value used for predict's own missing cells comes from TRAIN, not predict.
"""

import numpy as np
import pandas as pd

from missingness_auditor import Imputer


def test_numeric_median_fill_comes_from_train_only():
    df_train = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0, np.nan]})
    train_median = 3.0  # median of [1,2,3,4,5]

    # Predict has wild observed values; its own median would be ~1500.
    df_predict = pd.DataFrame({"x": [1000.0, np.nan, 2000.0, np.nan]})

    plan = {"columns": {"x": {
        "strategy": "numeric_median_plus_indicator",
        "add_missing_indicator": True,
    }}}

    _, pi = Imputer().apply(df_train, plan, df_predict)
    filled = pi.loc[df_predict["x"].isna(), "x"].to_numpy()
    # Every predict NaN is filled with the TRAIN median, not predict's own median.
    assert np.allclose(filled, train_median)


def test_group_median_fitted_on_train_only():
    df_train = pd.DataFrame({
        "g": ["A", "A", "A", "B", "B", "B"],
        "y": [10.0, 12.0, np.nan, 100.0, 102.0, np.nan],
    })
    # Train group medians: A -> 11, B -> 101
    df_predict = pd.DataFrame({
        "g": ["A", "B"],
        "y": [np.nan, np.nan],
    })
    plan = {"columns": {"y": {
        "strategy": "groupwise_numeric_median_plus_indicator",
        "add_missing_indicator": True,
        "group_col": "g",
    }}}

    _, pi = Imputer().apply(df_train, plan, df_predict)
    assert pi.loc[pi["g"] == "A", "y"].iloc[0] == 11.0
    assert pi.loc[pi["g"] == "B", "y"].iloc[0] == 101.0


def test_plan_records_fit_on_train_only(audited):
    _, _, results = audited
    for col, entry in results["imputation_plan"]["columns"].items():
        if entry["strategy"] not in ("no_imputation_needed",):
            assert entry["fit_on"] == "train_only"
