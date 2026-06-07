"""Verify src/missingness_auditor contains no dataset-specific column names."""

import glob
import os


SRC_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "src", "missingness_auditor"
)

FORBIDDEN_NAMES = [
    "SalePrice",
    "LotArea",
    "GrLivArea",
    "MSZoning",
    "HousePrice",
    "house_price",
    "titanic",
    "Titanic",
    "LotFrontage",
    "GarageCars",
    "PoolArea",
    "Neighborhood",
    "BldgType",
]


def _source_files():
    return glob.glob(os.path.join(SRC_DIR, "*.py"))


def test_no_competition_column_names():
    for py_file in _source_files():
        content = open(py_file).read()
        for name in FORBIDDEN_NAMES:
            assert name not in content, (
                f"Forbidden dataset-specific name {name!r} found in {py_file}"
            )


def test_src_files_exist():
    files = _source_files()
    assert len(files) >= 6, "Expected at least 6 source files in missingness_auditor"


def test_strategies_are_generic():
    """Imputation strategy names in recommender.py must not contain dataset-specific tokens."""
    rec_file = os.path.join(SRC_DIR, "recommender.py")
    content = open(rec_file).read()
    for name in FORBIDDEN_NAMES:
        assert name not in content


def test_auditor_works_on_arbitrary_columns(tmp_path):
    """MissingnessAuditor must accept any column names, not just known ones."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    import numpy as np
    import pandas as pd
    from missingness_auditor import MissingnessAuditor

    rng = np.random.default_rng(99)
    df = pd.DataFrame({
        "zebra_metric": np.where(rng.random(50) < 0.2, np.nan, rng.normal(0, 1, 50)),
        "alpha_code": np.where(rng.random(50) < 0.1, None, rng.choice(["p", "q"], 50)),
        "outcome_flag": rng.binomial(1, 0.4, 50).astype(float),
    })
    auditor = MissingnessAuditor(df, target_col="outcome_flag")
    results = auditor.run()
    assert "missingness_profile" in results
    assert "imputation_plan" in results
