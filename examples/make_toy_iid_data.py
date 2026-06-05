"""Generate toy iid dataset (no temporal or group structure)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import os


def make_iid_data(n_rows: int = 2000, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    df = pd.DataFrame({
        "feature_a": rng.normal(0, 1, n_rows),
        "feature_b": rng.normal(5, 2, n_rows),
        "feature_c": rng.choice(["X", "Y", "Z"], n_rows),
        "feature_d": rng.integers(1, 10, n_rows).astype(float),
        "target": rng.normal(10, 3, n_rows),
    })

    # Add a near-constant column
    df["near_const"] = 1.0
    df.loc[rng.choice(n_rows, 10, replace=False), "near_const"] = 2.0

    split = int(0.8 * n_rows)
    train_df = df.iloc[:split].copy().reset_index(drop=True)
    predict_df = df.iloc[split:].drop(columns=["target"]).reset_index(drop=True)
    return train_df, predict_df


if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(out_dir, exist_ok=True)
    train_df, predict_df = make_iid_data()
    train_df.to_csv(os.path.join(out_dir, "iid_train.csv"), index=False)
    predict_df.to_csv(os.path.join(out_dir, "iid_predict.csv"), index=False)
    print(f"IID — Train: {train_df.shape}, Predict: {predict_df.shape}")
