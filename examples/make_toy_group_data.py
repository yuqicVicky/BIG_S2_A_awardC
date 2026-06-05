"""Generate toy group-split dataset (disjoint groups in train vs predict)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import os


def make_group_data(
    n_groups: int = 100,
    rows_per_group: int = 20,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    records = []
    for g in range(n_groups):
        group_effect = rng.normal(0, 5)
        for _ in range(rows_per_group):
            feat = rng.normal(0, 1)
            records.append({
                "group_id": f"G{g:04d}",
                "feature_x": round(feat, 3),
                "feature_y": round(rng.uniform(0, 10), 3),
                "category": rng.choice(["P", "Q", "R"]),
                "target": round(group_effect + 2 * feat + rng.normal(0, 1), 3),
            })

    df = pd.DataFrame(records)

    train_groups = [f"G{g:04d}" for g in range(int(0.75 * n_groups))]
    predict_groups = [f"G{g:04d}" for g in range(int(0.75 * n_groups), n_groups)]

    train_df = df[df["group_id"].isin(train_groups)].reset_index(drop=True)
    predict_df = (
        df[df["group_id"].isin(predict_groups)]
        .drop(columns=["target"])
        .reset_index(drop=True)
    )
    return train_df, predict_df


if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(out_dir, exist_ok=True)
    train_df, predict_df = make_group_data()
    train_df.to_csv(os.path.join(out_dir, "group_train.csv"), index=False)
    predict_df.to_csv(os.path.join(out_dir, "group_predict.csv"), index=False)
    print(f"Group split — Train: {train_df.shape}, Predict: {predict_df.shape}")
    print(f"Train groups: {train_df['group_id'].nunique()}, Predict groups: {predict_df['group_id'].nunique()}")
