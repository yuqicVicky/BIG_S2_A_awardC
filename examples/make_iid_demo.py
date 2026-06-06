"""Case B — i.i.d. split demo data generator.

Structure:
- 1 000 rows drawn from the same distribution, random 80/20 split.
- No datetime column, no group/entity column.
- No train-only leakage columns (besides the target itself).
- Features: two numeric + one low-cardinality categorical.

Expected auditor output:
  pattern            : iid_random
  strategy           : kfold  (or stratified_kfold if target is categorical)
  leakage            : low — no high-severity risks
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

_CASE_DIR = os.path.join(os.path.dirname(__file__), "data", "iid")

AUDITOR_KWARGS = dict(target_col="target")


def make_iid_data(
    n_rows: int = 1000,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    feature_a = rng.normal(0, 1, n_rows)
    feature_b = rng.uniform(0, 10, n_rows)
    category = rng.choice(["alpha", "beta", "gamma"], n_rows)
    target = 2.0 * feature_a - 0.5 * feature_b + rng.normal(0, 1, n_rows)
    target = target.round(3)

    df = pd.DataFrame({
        "feature_a": feature_a.round(4),
        "feature_b": feature_b.round(4),
        "category": category,
        "target": target,
    })

    split = int(0.8 * n_rows)
    train_df = df.iloc[:split].copy().reset_index(drop=True)
    predict_df = df.iloc[split:].drop(columns=["target"]).reset_index(drop=True)
    return train_df, predict_df


def write_data(out_dir: str = _CASE_DIR) -> tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    train_df, predict_df = make_iid_data()
    train_path = os.path.join(out_dir, "train.csv")
    predict_path = os.path.join(out_dir, "predict.csv")
    train_df.to_csv(train_path, index=False)
    predict_df.to_csv(predict_path, index=False)
    return train_path, predict_path


if __name__ == "__main__":
    train_path, predict_path = write_data()
    train_df = pd.read_csv(train_path)
    predict_df = pd.read_csv(predict_path)
    print(f"Case B — iid")
    print(f"  Train  : {train_df.shape[0]} rows × {train_df.shape[1]} cols")
    print(f"  Predict: {predict_df.shape[0]} rows × {predict_df.shape[1]} cols")
    print(f"  No time / group columns  →  expect iid_random pattern, kfold strategy")
