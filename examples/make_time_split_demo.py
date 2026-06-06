"""Case D — future time-split demo data generator.

Structure:
- Daily rows from 2023-01-01 to 2023-12-31.
- Train: months Jan–Jun (days 1–181); Predict: months Jul–Dec (days 182–365).
- predict_min (2023-07-01) > train_max (2023-06-30) → time_based_split detected.
- No group column; no train-only leakage columns besides the target.
- Target has a linear trend + monthly seasonality.

Expected auditor output:
  pattern            : time_based_split
  strategy           : time_holdout
  strategies_to_avoid: includes kfold, random_holdout
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

_CASE_DIR = os.path.join(os.path.dirname(__file__), "data", "future_time_split")

AUDITOR_KWARGS = dict(target_col="target", datetime_col="date")


def make_time_split_data(seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    dates = pd.date_range("2023-01-01", "2023-12-31", freq="D")
    n = len(dates)

    day_of_year = np.arange(n)
    seasonality = 10 * np.sin(2 * np.pi * day_of_year / 365)
    trend = 0.05 * day_of_year
    numeric_feat = rng.normal(3, 1, n)
    category = rng.choice(["low", "mid", "high"], n)
    target = (50 + trend + seasonality + 2 * numeric_feat
              + rng.normal(0, 2, n)).round(3)

    df = pd.DataFrame({
        "date": dates,
        "numeric_feat": numeric_feat.round(4),
        "category": category,
        "target": target,
    })

    train_df = df[df["date"].dt.month <= 6].reset_index(drop=True)
    predict_df = (
        df[df["date"].dt.month > 6]
        .drop(columns=["target"])
        .reset_index(drop=True)
    )
    return train_df, predict_df


def write_data(out_dir: str = _CASE_DIR) -> tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    train_df, predict_df = make_time_split_data()
    train_path = os.path.join(out_dir, "train.csv")
    predict_path = os.path.join(out_dir, "predict.csv")
    train_df.to_csv(train_path, index=False)
    predict_df.to_csv(predict_path, index=False)
    return train_path, predict_path


if __name__ == "__main__":
    train_path, predict_path = write_data()
    train_df = pd.read_csv(train_path)
    predict_df = pd.read_csv(predict_path)
    print(f"Case D — future_time_split")
    print(f"  Train  : {train_df.shape[0]} rows × {train_df.shape[1]} cols")
    print(f"  Predict: {predict_df.shape[0]} rows × {predict_df.shape[1]} cols")
    print(f"  Train dates  : {train_df['date'].min()} → {train_df['date'].max()}")
    print(f"  Predict dates: {predict_df['date'].min()} → {predict_df['date'].max()}")
    print(f"  No overlap in time  →  expect time_based_split, time_holdout strategy")
