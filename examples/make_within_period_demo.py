"""Case A — within-period split demo data generator.

Structure:
- 3 months of daily data for 10 entities.
- Train: days 1-19 of each month; Predict: days 20-end of each month.
- This interleaving means predict_min (Jan-20) < train_max (Mar-19)
  → DistributionShiftDetector labels the pattern "within_period" or
    "within_period_group" (entity column is present in both splits).
- `demand_so_far` is a train-only running-total column that is absent
  from the predict CSV → flagged as high-severity leakage / unavailable.

Expected auditor output:
  pattern            : within_period  OR  within_period_group
  strategy           : within_period_latest_available_holdout
  strategies_to_avoid: includes time_holdout
  columns_to_exclude : includes demand_so_far
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

_CASE_DIR = os.path.join(os.path.dirname(__file__), "data", "within_period")

AUDITOR_KWARGS = dict(
    target_col="target",
    datetime_col="timestamp",
    group_col="entity_id",
)


def make_within_period_data(
    n_entities: int = 10,
    n_months: int = 3,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    train_rows: list[dict] = []
    predict_rows: list[dict] = []

    for entity_idx in range(n_entities):
        entity_id = f"ENT_{entity_idx:02d}"
        baseline = rng.normal(50, 10)

        for month in range(1, n_months + 1):
            days_in_month = [31, 28, 31][month - 1]
            running_total = 0.0

            for day in range(1, days_in_month + 1):
                ts = pd.Timestamp(f"2023-{month:02d}-{day:02d}")
                day_of_week = ts.dayofweek

                numeric_feat = rng.normal(5, 1)
                category = rng.choice(["P", "Q", "R"])
                target = round(
                    baseline
                    + 3 * np.sin(2 * np.pi * day_of_week / 7)
                    + 1.5 * numeric_feat
                    + rng.normal(0, 2),
                    2,
                )
                running_total += target

                row = dict(
                    timestamp=ts,
                    entity_id=entity_id,
                    numeric_feat=round(numeric_feat, 3),
                    category=category,
                    target=target,
                )

                if day <= 19:
                    row["demand_so_far"] = round(running_total, 2)
                    train_rows.append(row)
                else:
                    predict_rows.append({k: v for k, v in row.items()
                                         if k != "demand_so_far" and k != "target"})

    train_df = pd.DataFrame(train_rows).reset_index(drop=True)
    predict_df = pd.DataFrame(predict_rows).reset_index(drop=True)
    return train_df, predict_df


def write_data(out_dir: str = _CASE_DIR) -> tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    train_df, predict_df = make_within_period_data()
    train_path = os.path.join(out_dir, "train.csv")
    predict_path = os.path.join(out_dir, "predict.csv")
    train_df.to_csv(train_path, index=False)
    predict_df.to_csv(predict_path, index=False)
    return train_path, predict_path


if __name__ == "__main__":
    train_path, predict_path = write_data()
    train_df = pd.read_csv(train_path)
    predict_df = pd.read_csv(predict_path)
    print(f"Case A — within_period")
    print(f"  Train  : {train_df.shape[0]} rows × {train_df.shape[1]} cols")
    print(f"  Predict: {predict_df.shape[0]} rows × {predict_df.shape[1]} cols")
    print(f"  Train timestamps : {train_df['timestamp'].min()} → {train_df['timestamp'].max()}")
    print(f"  Predict timestamps: {predict_df['timestamp'].min()} → {predict_df['timestamp'].max()}")
    print(f"  Train-only col : demand_so_far  →  expect HIGH leakage flag")
