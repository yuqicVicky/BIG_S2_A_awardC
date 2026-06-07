"""Verify src/missingness_auditor/ contains no dataset-specific hardcoded names."""

import glob
import os
import sys
import numpy as np
import pandas as pd

_SRC = os.path.join(os.path.dirname(__file__), "..", "src", "missingness_auditor")

FORBIDDEN = [
    "SalePrice", "LotArea", "GrLivArea", "MSZoning", "HousePrice",
    "house_price", "titanic", "Titanic", "LotFrontage", "GarageCars",
    "PoolArea", "Neighborhood", "BldgType", "PassengerId",
]


def _src_files():
    return glob.glob(os.path.join(_SRC, "*.py"))


def test_no_competition_column_names():
    for path in _src_files():
        content = open(path).read()
        for name in FORBIDDEN:
            assert name not in content, (
                f"Forbidden name {name!r} found in {os.path.basename(path)}"
            )


def test_src_files_present():
    expected = ["auditor.py", "profiler.py", "mechanism.py", "structural.py",
                "planner.py", "imputer.py", "visualization.py", "reporting.py", "cli.py"]
    found = {os.path.basename(p) for p in _src_files()}
    for f in expected:
        assert f in found, f"Missing source file: {f}"


def test_strategies_are_generic():
    path = os.path.join(_SRC, "planner.py")
    content = open(path).read()
    for name in FORBIDDEN:
        assert name not in content


def test_imputer_no_hardcoding():
    path = os.path.join(_SRC, "imputer.py")
    content = open(path).read()
    for name in FORBIDDEN:
        assert name not in content


def test_auditor_accepts_arbitrary_columns(tmp_path):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from missingness_auditor import MissingnessAuditor

    rng = np.random.default_rng(99)
    df = pd.DataFrame({
        "zebra_metric": np.where(rng.random(60) < 0.20, np.nan, rng.normal(0, 1, 60)),
        "alpha_code":   np.where(rng.random(60) < 0.10, None, rng.choice(["p", "q"], 60)),
        "outcome_flag": rng.binomial(1, 0.4, 60).astype(float),
    })
    out = str(tmp_path / "out") + "/"
    results = MissingnessAuditor(df, target_col="outcome_flag").run()
    MissingnessAuditor(df, target_col="outcome_flag").save_outputs(results, out)
    assert "missingness_profile" in results
    assert "imputation_plan" in results
    assert os.path.exists(os.path.join(out, "logs/imputation_plan.json"))
