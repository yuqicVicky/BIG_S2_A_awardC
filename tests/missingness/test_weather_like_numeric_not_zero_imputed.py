"""
Weather-like numeric columns must never receive structural_zero imputation.

Continuous numeric columns with positive real-world values (temperatures,
precipitation, wind speed) should never be filled with 0 unless paired
zero-companion evidence specifically supports it.

Even if such a column appears to be missing for certain jurisdictions,
it is group-dependent missingness — and the correct strategy is
groupwise_numeric_median_plus_indicator, not structural_zero_plus_indicator.
"""

import numpy as np
import pandas as pd
import pytest

from missingness_auditor.profiler import MissingnessProfiler
from missingness_auditor.mechanism import MechanismAuditor
from missingness_auditor.structural import StructuralMissingnessDetector
from missingness_auditor.planner import ImputationPlanner


_WEATHER_LIKE_COLS = ["temp_avg_f", "wind_speed_mph", "humidity_pct", "solar_irradiance"]

_UNSAFE_STRATEGIES = {"structural_none_or_zero", "structural_zero_plus_indicator"}


def _make_weather_df(seed: int = 0) -> pd.DataFrame:
    """
    Multiple continuous weather columns.
    temp_avg_f is missing for 'arctic_station' (group-dependent).
    wind_speed_mph has 5% MCAR missingness.
    No column should be zero-imputed.
    """
    rng = np.random.default_rng(seed)
    n_per = 80
    regions = ["midwest"] * n_per + ["southeast"] * n_per + ["arctic_station"] * n_per
    n = len(regions)

    temp = rng.uniform(30, 95, n)
    temp[np.array(regions) == "arctic_station"] = np.nan  # group-dependent

    wind = rng.uniform(2, 40, n)
    wind_mask = rng.random(n) < 0.05
    wind[wind_mask] = np.nan  # MCAR

    humidity = rng.uniform(20, 90, n)

    return pd.DataFrame({
        "region": regions,
        "temp_avg_f": temp,
        "wind_speed_mph": wind,
        "humidity_pct": humidity,
        "target": rng.normal(50, 10, n),
    })


def test_no_weather_column_gets_zero_imputation():
    df = _make_weather_df()
    profile = MissingnessProfiler(df, target_col="target").profile()
    mech = MechanismAuditor(df, target_col="target").audit()
    struct = StructuralMissingnessDetector(df).detect()
    plan = ImputationPlanner(profile, mech, struct).plan()

    for col in _WEATHER_LIKE_COLS:
        if col not in plan["columns"]:
            continue
        strategy = plan["columns"][col]["strategy"]
        assert strategy not in _UNSAFE_STRATEGIES, (
            f"Column `{col}` got unsafe strategy `{strategy}`. "
            "Weather-like continuous columns must never be zero-imputed."
        )


def test_group_dependent_temp_not_structural():
    df = _make_weather_df()
    struct = StructuralMissingnessDetector(df).detect()
    assert "temp_avg_f" not in struct["column_flags"], (
        "temp_avg_f missing for arctic_station is group-dependent, not structural"
    )


def test_group_dependent_temp_gets_groupwise_strategy():
    df = _make_weather_df()
    profile = MissingnessProfiler(df, target_col="target").profile()
    mech = MechanismAuditor(df, target_col="target").audit()
    struct = StructuralMissingnessDetector(df).detect()
    plan = ImputationPlanner(profile, mech, struct).plan()

    strategy = plan["columns"]["temp_avg_f"]["strategy"]
    assert strategy == "groupwise_numeric_median_plus_indicator", (
        f"Expected groupwise_numeric_median_plus_indicator, got {strategy!r}"
    )


def test_mcar_wind_gets_median_not_zero():
    df = _make_weather_df()
    profile = MissingnessProfiler(df, target_col="target").profile()
    mech = MechanismAuditor(df, target_col="target").audit()
    struct = StructuralMissingnessDetector(df).detect()
    plan = ImputationPlanner(profile, mech, struct).plan()

    strategy = plan["columns"]["wind_speed_mph"]["strategy"]
    assert strategy in {"numeric_median", "numeric_median_plus_indicator"}, (
        f"wind_speed_mph (MCAR) should get median imputation, got {strategy!r}"
    )
    assert strategy not in _UNSAFE_STRATEGIES
