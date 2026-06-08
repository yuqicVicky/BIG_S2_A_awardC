"""
High-cardinality text/categorical columns must use missing-token imputation,
not mode imputation.

Mode imputation on a high-cardinality column (e.g. a column with hundreds of
unique string values) would create a spurious spike in one arbitrary category,
distorting the distribution and adding fake signal. The safe strategy is to
fill with "MISSING" (categorical_missing_token_plus_indicator).
"""

import numpy as np
import pandas as pd
import pytest

from missingness_auditor.profiler import MissingnessProfiler
from missingness_auditor.mechanism import MechanismAuditor
from missingness_auditor.structural import StructuralMissingnessDetector
from missingness_auditor.planner import ImputationPlanner


def _make_high_cardinality_df(n: int = 500, seed: int = 1) -> pd.DataFrame:
    """
    product_id: 200 unique string IDs — highly cardinal text-like column.
    city_name: 80 unique city strings — also high cardinality.
    region: 4 categories — low cardinality (mode imputation acceptable).
    """
    rng = np.random.default_rng(seed)

    # 200 unique product IDs
    all_product_ids = [f"SKU-{i:04d}" for i in range(200)]
    product_ids = rng.choice(all_product_ids, n).tolist()
    miss_mask_product = rng.random(n) < 0.20
    product_col = [None if m else v for m, v in zip(miss_mask_product, product_ids)]

    # 80 unique city names
    all_cities = [f"City_{i}" for i in range(80)]
    cities = rng.choice(all_cities, n).tolist()
    miss_mask_city = rng.random(n) < 0.15
    city_col = [None if m else v for m, v in zip(miss_mask_city, cities)]

    # 4 low-cardinality regions
    regions = rng.choice(["north", "south", "east", "west"], n).tolist()
    miss_mask_region = rng.random(n) < 0.08
    region_col = [None if m else v for m, v in zip(miss_mask_region, regions)]

    return pd.DataFrame({
        "product_id": product_col,
        "city_name": city_col,
        "region": region_col,
        "sales_amount": rng.normal(100, 30, n),
        "target": rng.normal(0, 1, n),
    })


def test_high_cardinality_product_id_flagged():
    df = _make_high_cardinality_df()
    profile = MissingnessProfiler(df, target_col="target").profile()
    assert profile["columns"]["product_id"]["is_high_cardinality"] is True


def test_high_cardinality_city_name_flagged():
    df = _make_high_cardinality_df()
    profile = MissingnessProfiler(df, target_col="target").profile()
    assert profile["columns"]["city_name"]["is_high_cardinality"] is True


def test_low_cardinality_region_not_flagged():
    df = _make_high_cardinality_df()
    profile = MissingnessProfiler(df, target_col="target").profile()
    assert profile["columns"]["region"]["is_high_cardinality"] is False


def test_high_cardinality_product_id_not_mode_imputed():
    df = _make_high_cardinality_df()
    profile = MissingnessProfiler(df, target_col="target").profile()
    mech = MechanismAuditor(df, target_col="target").audit()
    struct = StructuralMissingnessDetector(df).detect()
    plan = ImputationPlanner(profile, mech, struct).plan()

    strategy = plan["columns"]["product_id"]["strategy"]
    assert strategy != "categorical_mode_plus_indicator", (
        "Mode imputation on high-cardinality product_id creates spurious repeated values"
    )
    assert strategy == "categorical_missing_token_plus_indicator", (
        f"Expected categorical_missing_token_plus_indicator, got {strategy!r}"
    )


def test_high_cardinality_city_name_not_mode_imputed():
    df = _make_high_cardinality_df()
    profile = MissingnessProfiler(df, target_col="target").profile()
    mech = MechanismAuditor(df, target_col="target").audit()
    struct = StructuralMissingnessDetector(df).detect()
    plan = ImputationPlanner(profile, mech, struct).plan()

    strategy = plan["columns"]["city_name"]["strategy"]
    assert strategy != "categorical_mode_plus_indicator", (
        "Mode imputation on high-cardinality city_name creates spurious repeated values"
    )
    assert strategy == "categorical_missing_token_plus_indicator", (
        f"Expected categorical_missing_token_plus_indicator, got {strategy!r}"
    )


def test_high_cardinality_adds_missing_indicator():
    df = _make_high_cardinality_df()
    profile = MissingnessProfiler(df, target_col="target").profile()
    mech = MechanismAuditor(df, target_col="target").audit()
    struct = StructuralMissingnessDetector(df).detect()
    plan = ImputationPlanner(profile, mech, struct).plan()

    assert plan["columns"]["product_id"]["add_missing_indicator"] is True
    assert plan["columns"]["city_name"]["add_missing_indicator"] is True


def test_safety_warning_present_for_high_cardinality():
    df = _make_high_cardinality_df()
    profile = MissingnessProfiler(df, target_col="target").profile()
    mech = MechanismAuditor(df, target_col="target").audit()
    struct = StructuralMissingnessDetector(df).detect()
    plan = ImputationPlanner(profile, mech, struct).plan()

    warning = plan["columns"]["product_id"].get("safety_warning")
    assert warning is not None and "high_cardinality" in warning.lower(), (
        f"Expected safety warning mentioning high cardinality, got: {warning!r}"
    )
