"""Shared fixtures for the missingness-auditor test suite.

All datasets are synthetic and constructed in-memory — no external downloads.
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def rng():
    return np.random.RandomState(12345)


@pytest.fixture
def mixed_df(rng):
    """A dataset with four distinct missingness types and a binary target.

    - age            : ~8% MCAR-compatible
    - income         : ~30% MAR-like (missing when education is low)
    - risk_score     : ~25% target-associated (missing when target == 1)
    - pool_type/area : structural absence (pool_type NaN <=> pool_area == 0)
    """
    n = 400
    education = rng.randint(1, 5, n)
    target = rng.binomial(1, 0.4, n).astype(float)

    income = rng.normal(60000, 12000, n)
    mar_mask = (education <= 2) & (rng.rand(n) < 0.6)
    income[mar_mask] = np.nan

    risk_score = rng.normal(0.5, 0.15, n)
    tgt_mask = (target == 1) & (rng.rand(n) < 0.55)
    risk_score[tgt_mask] = np.nan

    age = rng.normal(50, 10, n)
    age[rng.rand(n) < 0.08] = np.nan

    # Structural: half the rows have no pool (pool_type missing, pool_area == 0)
    has_pool = rng.rand(n) < 0.5
    pool_type = np.where(has_pool, rng.choice(["concrete", "vinyl"], n), None)
    pool_area = np.where(has_pool, rng.normal(40, 8, n), 0.0)

    return pd.DataFrame({
        "age": age,
        "income": income,
        "education": education.astype(float),
        "risk_score": risk_score,
        "pool_type": pool_type,
        "pool_area": pool_area,
        "target": target,
    })


@pytest.fixture
def audited(mixed_df):
    """Run the full auditor once and return (df, results)."""
    from missingness_auditor import MissingnessAuditor
    aud = MissingnessAuditor(mixed_df, target_col="target")
    return mixed_df, aud, aud.run()
