"""Schema inspection: column types, null rates, unique counts, sample values."""

from __future__ import annotations

import pandas as pd
import numpy as np


class SchemaInspector:
    def __init__(self, train_df: pd.DataFrame, predict_df: pd.DataFrame | None = None):
        self.train_df = train_df
        self.predict_df = predict_df

    def inspect(self) -> dict:
        train_schema = self._inspect_df(self.train_df, "train")
        result: dict = {"train": train_schema}
        if self.predict_df is not None:
            result["predict"] = self._inspect_df(self.predict_df, "predict")
            result["comparison"] = self._compare_schemas(train_schema, result["predict"])
        return result

    def _inspect_df(self, df: pd.DataFrame, split: str) -> dict:
        columns = {}
        for col in df.columns:
            series = df[col]
            null_count = int(series.isna().sum())
            null_rate = round(null_count / max(len(series), 1), 4)
            unique_count = int(series.nunique(dropna=True))
            dtype_str = str(series.dtype)
            sample_vals = [
                _safe_str(v)
                for v in series.dropna().head(5).tolist()
            ]
            columns[col] = {
                "dtype": dtype_str,
                "null_count": null_count,
                "null_rate": null_rate,
                "unique_count": unique_count,
                "row_count": len(series),
                "sample_values": sample_vals,
            }
        return {"split": split, "row_count": len(df), "col_count": len(df.columns), "columns": columns}

    def _compare_schemas(self, train_schema: dict, predict_schema: dict) -> dict:
        train_cols = set(train_schema["columns"].keys())
        predict_cols = set(predict_schema["columns"].keys())
        train_only = sorted(train_cols - predict_cols)
        predict_only = sorted(predict_cols - train_cols)
        common = sorted(train_cols & predict_cols)
        dtype_mismatches = {}
        for col in common:
            td = train_schema["columns"][col]["dtype"]
            pd_ = predict_schema["columns"][col]["dtype"]
            if td != pd_:
                dtype_mismatches[col] = {"train_dtype": td, "predict_dtype": pd_}
        return {
            "train_only_columns": train_only,
            "predict_only_columns": predict_only,
            "common_columns": common,
            "dtype_mismatches": dtype_mismatches,
            "row_count_train": train_schema["row_count"],
            "row_count_predict": predict_schema["row_count"],
        }


def _safe_str(v):
    if isinstance(v, float) and np.isnan(v):
        return None
    if isinstance(v, (pd.Timestamp,)):
        return str(v)
    try:
        return str(v)
    except Exception:
        return None
