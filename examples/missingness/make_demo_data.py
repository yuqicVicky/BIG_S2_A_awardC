"""
Generate the four toy datasets that illustrate different missingness mechanisms.

Case 1 — MCAR numeric:
    A numeric column has ~7% of values randomly dropped with no relationship to
    any other variable.

Case 2 — MAR-like:
    An income column is missing whenever education_level is low.  The missingness
    depends on an observed feature, a classic MAR-like pattern.

Case 3 — MNAR / target signal:
    A credit_score column is always missing for customers who defaulted (target=1).
    The missingness is directly related to the outcome being predicted.

Case 4 — Structural:
    A facility_type categorical column is NaN when there is no facility.
    Its companion facility_capacity is 0 in those same rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_mcar_dataset(n: int = 800, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "measurement_a": rng.normal(10, 2, n),
        "measurement_b": rng.normal(5, 1, n),
        "measurement_c": rng.normal(0, 3, n),
        "target": rng.normal(100, 15, n),
    })
    mcar_mask = rng.random(n) < 0.07
    df.loc[mcar_mask, "measurement_a"] = np.nan
    return df


def make_mar_dataset(n: int = 800, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    education = rng.integers(0, 3, n).astype(float)   # 0=low, 1=mid, 2=high
    income = rng.normal(60_000, 15_000, n)
    income_mar = income.astype(float).copy()
    income_mar[education == 0] = np.nan                # missing when education low
    target = rng.normal(0, 1, n)

    return pd.DataFrame({
        "education_level": education,
        "income": income_mar,
        "years_experience": rng.normal(8, 4, n).clip(0),
        "target": target,
    })


def make_mnar_dataset(n: int = 800, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    target = rng.binomial(1, 0.30, n).astype(float)   # 1 = defaulted
    credit_score = rng.normal(650, 80, n).astype(float)
    credit_score[target == 1] = np.nan                 # missing because defaulted
    account_age = rng.normal(5, 2, n).clip(0)

    return pd.DataFrame({
        "credit_score": credit_score,
        "account_age_years": account_age,
        "target": target,
    })


def make_structural_dataset(n_with: int = 120, n_without: int = 80, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_with + n_without

    facility_type = ["covered"] * 60 + ["open"] * 60 + [None] * n_without
    facility_capacity = (
        rng.integers(10, 100, n_with).tolist() + [0] * n_without
    )
    property_size = rng.normal(1_500, 400, n).clip(400)
    target = rng.normal(300_000, 80_000, n)

    return pd.DataFrame({
        "facility_type": facility_type,
        "facility_capacity": facility_capacity,
        "property_size_sqft": property_size,
        "target": target,
    })


if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)

    cases = {
        "mcar": make_mcar_dataset(),
        "mar":  make_mar_dataset(),
        "mnar": make_mnar_dataset(),
        "structural": make_structural_dataset(),
    }
    for name, df in cases.items():
        path = f"data/{name}_demo.csv"
        df.to_csv(path, index=False)
        miss = df.isna().mean()
        print(f"[{name}] saved {len(df)} rows to {path}")
        print(f"  missing rates: {miss[miss > 0].to_dict()}")
