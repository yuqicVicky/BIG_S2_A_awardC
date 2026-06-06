"""Case C — group (unseen-group) split demo data generator.

Structure:
- 6 stores total: stores 1–4 in train, stores 5–6 in predict (fully disjoint).
- Column name "store_id" → FeatureTypeInferrer labels it group_entity_id.
- Overlap ratio = 0 / 6 < 0.5 threshold → group_based_split detected.
- No datetime column; no train-only leakage columns besides the target.

Expected auditor output:
  pattern            : group_based_split
  strategy           : group_split
  strategies_to_avoid: includes kfold, random_holdout
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

_CASE_DIR = os.path.join(os.path.dirname(__file__), "data", "group")

AUDITOR_KWARGS = dict(target_col="target", group_col="store_id")


def make_group_data(
    rows_per_store: int = 120,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    train_stores = ["store_A", "store_B", "store_C", "store_D"]
    predict_stores = ["store_E", "store_F"]
    all_stores = train_stores + predict_stores

    records = []
    for store in all_stores:
        store_effect = rng.normal(20, 5)
        for _ in range(rows_per_store):
            sales_volume = rng.uniform(100, 500)
            promo = rng.choice([0, 1])
            target = round(
                store_effect + 0.05 * sales_volume + 10 * promo + rng.normal(0, 3),
                2,
            )
            records.append(dict(
                store_id=store,
                sales_volume=round(sales_volume, 2),
                promo_flag=int(promo),
                target=target,
            ))

    df = pd.DataFrame(records)
    train_df = df[df["store_id"].isin(train_stores)].reset_index(drop=True)
    predict_df = (
        df[df["store_id"].isin(predict_stores)]
        .drop(columns=["target"])
        .reset_index(drop=True)
    )
    return train_df, predict_df


def write_data(out_dir: str = _CASE_DIR) -> tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    train_df, predict_df = make_group_data()
    train_path = os.path.join(out_dir, "train.csv")
    predict_path = os.path.join(out_dir, "predict.csv")
    train_df.to_csv(train_path, index=False)
    predict_df.to_csv(predict_path, index=False)
    return train_path, predict_path


if __name__ == "__main__":
    train_path, predict_path = write_data()
    train_df = pd.read_csv(train_path)
    predict_df = pd.read_csv(predict_path)
    print(f"Case C — group")
    print(f"  Train  : {train_df.shape[0]} rows × {train_df.shape[1]} cols")
    print(f"  Predict: {predict_df.shape[0]} rows × {predict_df.shape[1]} cols")
    print(f"  Train stores  : {sorted(train_df['store_id'].unique())}")
    print(f"  Predict stores: {sorted(predict_df['store_id'].unique())}")
    print(f"  Overlap = 0  →  expect group_based_split, group_split strategy")
