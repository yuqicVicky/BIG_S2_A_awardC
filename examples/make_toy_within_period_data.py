"""Generate toy within-period dataset.

In a within-period scenario, the prediction timestamps overlap with the
training period. The task is to predict a value for a future time step
given all prior observations within the same period.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import os


def make_within_period_data(
    n_entities: int = 50,
    n_periods: int = 30,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    records = []
    for entity in range(n_entities):
        base_level = rng.normal(100, 20)
        for period in range(n_periods):
            ts = pd.Timestamp("2023-01-01") + pd.Timedelta(hours=period)
            hour = ts.hour
            dow = ts.dayofweek
            month = ts.month

            # Synthetic feature: numeric signal correlated with target
            numeric_feature = rng.normal(50, 10)
            category = rng.choice(["A", "B", "C"])
            text_note = rng.choice(["normal operation", "elevated demand", "low activity"])

            # Target: influenced by hour, entity level, and numeric feature
            target = (
                base_level
                + 5 * np.sin(2 * np.pi * hour / 24)
                + 0.3 * numeric_feature
                + rng.normal(0, 5)
            )

            records.append({
                "timestamp": ts,
                "entity_id": f"E{entity:03d}",
                "hour": hour,
                "dayofweek": dow,
                "month": month,
                "numeric_feature": round(numeric_feature, 2),
                "category": category,
                "text_note": text_note,
                "target": round(target, 2),
            })

    df = pd.DataFrame(records)

    # Within-period split: latest observation per entity is the prediction set
    train_rows = []
    predict_rows = []
    for entity, grp in df.groupby("entity_id"):
        grp_sorted = grp.sort_values("timestamp")
        # Predict on last 3 rows per entity; train on all prior
        predict_rows.append(grp_sorted.tail(3))
        train_rows.append(grp_sorted.iloc[:-3])

    train_df = pd.concat(train_rows).reset_index(drop=True)
    predict_df = pd.concat(predict_rows).reset_index(drop=True)

    # Drop target from predict set (as in real prediction scenario)
    predict_df = predict_df.drop(columns=["target"])

    return train_df, predict_df


if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(out_dir, exist_ok=True)
    train_df, predict_df = make_within_period_data()
    train_df.to_csv(os.path.join(out_dir, "within_period_train.csv"), index=False)
    predict_df.to_csv(os.path.join(out_dir, "within_period_predict.csv"), index=False)
    print(f"Train: {train_df.shape}, Predict: {predict_df.shape}")
    print(f"Train timestamp range: {train_df['timestamp'].min()} → {train_df['timestamp'].max()}")
    print(f"Predict timestamp range: {predict_df['timestamp'].min()} → {predict_df['timestamp'].max()}")
