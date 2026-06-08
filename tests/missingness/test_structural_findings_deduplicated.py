"""
Structural missingness findings must be deduplicated.

The same (categorical_col, numeric_col) pair must appear at most once in
structural_pairs, even if both Pattern 1 and Pattern 2 evidence could be
detected for the same pair.
"""

import numpy as np
import pandas as pd
import pytest

from missingness_auditor.structural import StructuralMissingnessDetector


def _make_structural_df_multi_pattern(seed: int = 3) -> pd.DataFrame:
    """
    facility_type is NaN when there is no facility.
    facility_capacity is 0 when there is no facility.
    facility_area is NaN when there is no facility.
    Both facility_capacity and facility_area could match the same cat column.
    """
    rng = np.random.default_rng(seed)
    n_with = 120
    n_without = 80
    n = n_with + n_without

    facility_type = ["type_a"] * 60 + ["type_b"] * 60 + [None] * n_without
    facility_capacity = rng.integers(10, 100, n_with).tolist() + [0] * n_without
    facility_area = rng.integers(50, 500, n_with).tolist() + [0] * n_without
    other = rng.normal(0, 1, n)

    return pd.DataFrame({
        "facility_type": facility_type,
        "facility_capacity": facility_capacity,
        "facility_area": facility_area,
        "other": other,
    })


def _make_multi_cat_df(seed: int = 5) -> pd.DataFrame:
    """
    Two separate structural patterns:
    garage_quality NaN → garage_area = 0
    pool_quality NaN  → pool_area = 0
    These are two distinct pairs; neither should appear more than once.
    """
    rng = np.random.default_rng(seed)
    n_with_g = 100
    n_without_g = 60
    n_with_p = 80
    n_without_p = 80
    n = n_with_g + n_without_g + n_with_p + n_without_p

    # Simplify: just make a flat dataset
    rng2 = np.random.default_rng(seed + 1)
    n2 = 200
    garage_q = ["good"] * 120 + [None] * 80
    garage_a = rng2.integers(100, 600, 120).tolist() + [0] * 80
    pool_q = ["blue"] * 100 + [None] * 100
    pool_a = rng2.integers(50, 300, 100).tolist() + [0] * 100

    return pd.DataFrame({
        "garage_quality": garage_q,
        "garage_area": garage_a,
        "pool_quality": pool_q,
        "pool_area": pool_a,
    })


def test_no_duplicate_pairs_in_structural():
    df = _make_structural_df_multi_pattern()
    result = StructuralMissingnessDetector(df).detect()
    pairs = result["structural_pairs"]
    seen = set()
    for pair in pairs:
        key = (pair["categorical_col"], pair["numeric_col"])
        assert key not in seen, (
            f"Duplicate structural pair found: {key}. "
            "Each (categorical, numeric) pair should appear at most once."
        )
        seen.add(key)


def test_multi_cat_no_duplicate_pairs():
    df = _make_multi_cat_df()
    result = StructuralMissingnessDetector(df).detect()
    pairs = result["structural_pairs"]
    seen = set()
    for pair in pairs:
        key = (pair["categorical_col"], pair["numeric_col"])
        assert key not in seen, (
            f"Duplicate structural pair found: {key}"
        )
        seen.add(key)


def test_pair_count_matches_unique_pairs():
    df = _make_structural_df_multi_pattern()
    result = StructuralMissingnessDetector(df).detect()
    pairs = result["structural_pairs"]
    unique_pairs = {(p["categorical_col"], p["numeric_col"]) for p in pairs}
    assert len(pairs) == len(unique_pairs), (
        f"n_pairs={len(pairs)} but n_unique={len(unique_pairs)}; duplicates present"
    )
    assert result["n_structural_pairs"] == len(pairs)


def test_n_structural_pairs_accurate():
    df = _make_multi_cat_df()
    result = StructuralMissingnessDetector(df).detect()
    assert result["n_structural_pairs"] == len(result["structural_pairs"])
