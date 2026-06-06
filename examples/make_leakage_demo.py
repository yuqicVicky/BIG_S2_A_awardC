"""Case E — deliberate leakage demo data generator.
   Case F — valid datetime (timestamp in both sets) is covered by the same data.

Structure:
  - `timestamp`       : datetime in BOTH train and predict  → NOT leakage (Case F)
  - `numeric_feat`    : numeric in BOTH sets               → safe to use
  - `target_component_1` : train-only; name matches ^target_ → HIGH leakage
  - `target_component_2` : train-only; name matches ^target_ → HIGH leakage
  - `target`          : target_component_1 + target_component_2 + noise, train-only

Case E expected:
  target_component_1 and target_component_2 in columns_to_exclude with leakage_risk="high"

Case F expected:
  timestamp is in columns_to_use  (datetime available at prediction time ≠ leakage)
  timestamp action = "use", leakage_risk = "none"
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

_CASE_DIR = os.path.join(os.path.dirname(__file__), "data", "leakage")

AUDITOR_KWARGS = dict(target_col="target", datetime_col="timestamp")


def make_leakage_data(
    n_train: int = 500,
    n_predict: int = 100,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    ts_train = pd.date_range("2023-01-01", periods=n_train, freq="h")
    ts_predict = pd.date_range("2023-02-01", periods=n_predict, freq="h")

    comp1_train = rng.normal(10, 2, n_train)
    comp2_train = rng.normal(5, 1, n_train)
    target_train = (comp1_train + comp2_train + rng.normal(0, 0.5, n_train)).round(3)

    train_df = pd.DataFrame({
        "timestamp": ts_train,
        "numeric_feat": rng.normal(3, 1, n_train).round(3),
        "target_component_1": comp1_train.round(3),
        "target_component_2": comp2_train.round(3),
        "target": target_train,
    })

    predict_df = pd.DataFrame({
        "timestamp": ts_predict,
        "numeric_feat": rng.normal(3, 1, n_predict).round(3),
    })

    return train_df, predict_df


def write_data(out_dir: str = _CASE_DIR) -> tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    train_df, predict_df = make_leakage_data()
    train_path = os.path.join(out_dir, "train.csv")
    predict_path = os.path.join(out_dir, "predict.csv")
    train_df.to_csv(train_path, index=False)
    predict_df.to_csv(predict_path, index=False)
    return train_path, predict_path


if __name__ == "__main__":
    train_path, predict_path = write_data()
    train_df = pd.read_csv(train_path)
    predict_df = pd.read_csv(predict_path)
    print(f"Case E — leakage  /  Case F — valid datetime")
    print(f"  Train  : {train_df.shape[0]} rows × {train_df.shape[1]} cols")
    print(f"  Predict: {predict_df.shape[0]} rows × {predict_df.shape[1]} cols")
    print(f"  Train-only leakage cols : target_component_1, target_component_2")
    print(f"  Predict cols: {list(predict_df.columns)}")
    print(f"  Timestamp in both sets  →  expect timestamp NOT flagged as leakage")
