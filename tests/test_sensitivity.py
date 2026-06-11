"""MNAR delta-adjustment sensitivity: fragile columns tip early, robust ones don't."""

import numpy as np
import pandas as pd

from missingness_auditor import mnar_sensitivity


def _frame():
    rng = np.random.RandomState(99)
    n = 300
    df = pd.DataFrame({
        "fragile": rng.normal(0, 1, n),   # heavy missingness → tips early
        "robust":  rng.normal(0, 1, n),   # tiny missingness → robust
        "target":  rng.normal(0, 1, n),
    })
    df.loc[rng.rand(n) < 0.40, "fragile"] = np.nan
    df.loc[rng.rand(n) < 0.02, "robust"] = np.nan
    return df


def test_tipping_point_found_for_fragile_column():
    sens = mnar_sensitivity(_frame(), plan=None, target_col="target")
    cols = sens["columns"]
    assert "fragile" in cols
    tp = cols["fragile"]["tipping_point_delta_sd"]
    assert tp is not None
    # A high-missingness column tips within a modest delta.
    assert abs(tp) <= 0.75
    assert "fragile" in sens["fragile_columns"]


def test_low_missingness_column_is_robust():
    sens = mnar_sensitivity(_frame(), plan=None, target_col="target")
    robust = sens["columns"]["robust"]
    # With ~2% missing, no plausible delta in the grid moves the mean out of CI.
    assert robust["tipping_point_delta_sd"] is None
    assert robust["robustness"] == "robust"


def test_trajectory_is_monotonic_in_delta():
    sens = mnar_sensitivity(_frame(), plan=None, target_col="target")
    traj = sens["columns"]["fragile"]["trajectory"]
    deltas = [p["delta_sd"] for p in traj]
    means = [p["mean"] for p in traj]
    # Mean increases monotonically as the delta shift increases.
    assert deltas == sorted(deltas)
    assert all(means[i] <= means[i + 1] + 1e-9 for i in range(len(means) - 1))
