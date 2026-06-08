"""
Every imputation plan entry (except no_imputation_needed) must contain
the required evidence fields for explainability and auditability.

Required fields:
  - missing_rate
  - dtype
  - cardinality
  - mechanism_label
  - target_association_evidence
  - covariate_association_evidence
  - group_dependency_evidence
  - structural_evidence
  - safety_warning (may be None)
"""

import numpy as np
import pandas as pd
import pytest

from missingness_auditor.profiler import MissingnessProfiler
from missingness_auditor.mechanism import MechanismAuditor
from missingness_auditor.structural import StructuralMissingnessDetector
from missingness_auditor.planner import ImputationPlanner


_REQUIRED_EVIDENCE_FIELDS = [
    "missing_rate",
    "dtype",
    "cardinality",
    "mechanism_label",
    "target_association_evidence",
    "covariate_association_evidence",
    "group_dependency_evidence",
    "structural_evidence",
    "safety_warning",
]


def _make_mixed_df(n: int = 400, seed: int = 42) -> pd.DataFrame:
    """Dataset with MCAR numeric, MAR numeric, structural categorical, high-card text."""
    rng = np.random.default_rng(seed)

    # MCAR numeric
    feat_a = rng.normal(0, 1, n)
    feat_a[rng.random(n) < 0.07] = np.nan

    # Group-dependent numeric
    regions = rng.choice(["east", "west", "north", "south"], n)
    temp = rng.uniform(30, 90, n)
    temp[regions == "north"] = np.nan  # group-dependent

    # Structural categorical + numeric companion
    facility_q = np.where(rng.random(n) < 0.35, None, rng.choice(["A", "B"], n))
    facility_cap = np.where(
        np.array(facility_q) == None,  # noqa: E711 (intentional None check)
        0,
        rng.integers(10, 100, n)
    ).astype(float)

    # High-cardinality text
    product_ids = [f"SKU-{i}" for i in range(150)]
    sku_col = rng.choice(product_ids, n).tolist()
    sku_col_with_missing = [None if rng.random() < 0.15 else v for v in sku_col]

    target = rng.normal(0, 1, n)

    return pd.DataFrame({
        "feature_a": feat_a,
        "temp_c": temp,
        "region": regions.tolist(),
        "facility_quality": facility_q.tolist(),
        "facility_capacity": facility_cap,
        "product_sku": sku_col_with_missing,
        "target": target,
    })


def _run_plan(df: pd.DataFrame) -> dict:
    profile = MissingnessProfiler(df, target_col="target").profile()
    mech = MechanismAuditor(df, target_col="target").audit()
    struct = StructuralMissingnessDetector(df).detect()
    return ImputationPlanner(profile, mech, struct).plan()


def test_all_imputed_columns_have_evidence_fields():
    df = _make_mixed_df()
    plan = _run_plan(df)

    for col, entry in plan["columns"].items():
        if entry["strategy"] == "no_imputation_needed":
            continue
        for field in _REQUIRED_EVIDENCE_FIELDS:
            assert field in entry, (
                f"Column `{col}` is missing required evidence field `{field}`. "
                f"Present fields: {list(entry.keys())}"
            )


def test_missing_rate_is_float_in_range():
    df = _make_mixed_df()
    plan = _run_plan(df)

    for col, entry in plan["columns"].items():
        if entry["strategy"] == "no_imputation_needed":
            continue
        rate = entry["missing_rate"]
        assert isinstance(rate, float), f"`{col}` missing_rate is not float: {rate!r}"
        assert 0.0 <= rate <= 1.0, f"`{col}` missing_rate out of range: {rate}"


def test_mechanism_label_is_valid_string():
    df = _make_mixed_df()
    plan = _run_plan(df)

    valid_labels = {
        "MCAR-compatible",
        "MAR-like evidence",
        "group-dependent missingness",
        "target-associated missingness",
        "high-cardinality text/category missingness",
        "structural absence concern",
        "insufficient evidence",
        "N/A",
    }
    for col, entry in plan["columns"].items():
        lbl = entry.get("mechanism_label")
        assert isinstance(lbl, str), f"`{col}` mechanism_label is not a string: {lbl!r}"
        assert lbl in valid_labels, (
            f"`{col}` has unrecognized mechanism_label: {lbl!r}. "
            f"Valid labels: {valid_labels}"
        )


def test_group_dependency_evidence_is_dict_or_none():
    df = _make_mixed_df()
    plan = _run_plan(df)

    for col, entry in plan["columns"].items():
        if entry["strategy"] == "no_imputation_needed":
            continue
        gdep = entry.get("group_dependency_evidence")
        assert gdep is None or isinstance(gdep, dict), (
            f"`{col}` group_dependency_evidence must be dict or None, got {type(gdep)}"
        )


def test_target_association_evidence_is_dict_or_none():
    df = _make_mixed_df()
    plan = _run_plan(df)

    for col, entry in plan["columns"].items():
        if entry["strategy"] == "no_imputation_needed":
            continue
        tgt = entry.get("target_association_evidence")
        assert tgt is None or isinstance(tgt, dict), (
            f"`{col}` target_association_evidence must be dict or None, got {type(tgt)}"
        )


def test_covariate_association_evidence_is_list():
    df = _make_mixed_df()
    plan = _run_plan(df)

    for col, entry in plan["columns"].items():
        if entry["strategy"] == "no_imputation_needed":
            continue
        cov = entry.get("covariate_association_evidence")
        assert isinstance(cov, list), (
            f"`{col}` covariate_association_evidence must be a list, got {type(cov)}"
        )
