"""
Generate examples/demo_missingness.csv — a toy dataset with four missingness patterns.

Case 1 — MCAR:         age is ~8% randomly missing (no pattern).
Case 2 — MAR-like:     income is missing when education_level == 0 (low education).
Case 3 — MNAR:         risk_score is missing whenever target == 1 (defaulted).
Case 4 — Structural:   facility_quality is missing when facility_area == 0 (no facility).

Run:
    python examples/make_demo_missingness_data.py
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd


def make_demo_df(n: int = 600, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Observed covariates (no missing)
    education_level = rng.integers(0, 3, n).astype(float)   # 0=low, 1=mid, 2=high
    facility_area   = np.where(rng.random(n) < 0.35, 0.0,   # 35% have no facility
                               rng.gamma(2, 50, n))

    # Target (binary)
    target = rng.binomial(1, 0.28, n).astype(float)

    # Case 1 — MCAR: age, ~8% randomly missing
    age = rng.normal(38, 12, n).clip(18, 80)
    age = np.where(rng.random(n) < 0.08, np.nan, age)

    # Case 2 — MAR-like: income missing when education_level == 0
    income = rng.normal(55_000, 18_000, n).clip(10_000)
    income = np.where(education_level == 0, np.nan, income)

    # Case 3 — MNAR: risk_score missing when customer defaulted (target == 1)
    risk_score = rng.normal(650, 90, n).clip(300, 850)
    risk_score = np.where(target == 1, np.nan, risk_score)

    # Case 4 — Structural: facility_quality missing when no facility
    quality_labels = rng.choice(["excellent", "good", "fair", "poor"], n,
                                 p=[0.15, 0.40, 0.30, 0.15])
    facility_quality = np.where(facility_area == 0, None, quality_labels)

    # Other complete column
    other_score = rng.normal(100, 15, n)

    return pd.DataFrame({
        "age":              age,
        "income":           income,
        "education_level":  education_level,
        "risk_score":       risk_score,
        "facility_area":    facility_area,
        "facility_quality": facility_quality,
        "other_score":      other_score,
        "target":           target,
    })


if __name__ == "__main__":
    os.makedirs("examples", exist_ok=True)
    df = make_demo_df()
    out = os.path.join("examples", "demo_missingness.csv")
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows to {out}")
    miss = df.isna().mean()
    print("Missing rates:")
    for col, rate in miss[miss > 0].items():
        print(f"  {col}: {rate:.1%}")
