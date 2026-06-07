"""Missingness profile: per-column missing counts, rates, and severity."""

from __future__ import annotations

import pandas as pd
import numpy as np


def _severity(rate: float) -> str:
    if rate == 0.0:
        return "none"
    if rate >= 1.0:
        return "complete"
    if rate >= 0.80:
        return "high"
    if rate >= 0.20:
        return "moderate"
    if rate >= 0.05:
        return "low"
    return "trace"


def _dtype_category(series: pd.Series) -> str:
    return "numeric" if pd.api.types.is_numeric_dtype(series) else "categorical"


class MissingnessProfiler:
    """Builds a per-column missingness profile from one or two dataframes."""

    def __init__(
        self,
        df: pd.DataFrame,
        predict_df: pd.DataFrame | None = None,
        *,
        target_col: str | None = None,
    ):
        self.df = df
        self.predict_df = predict_df
        self.target_col = target_col

    def profile(self) -> dict:
        columns: dict[str, dict] = {}
        for col in self.df.columns:
            series = self.df[col]
            n_total = len(series)
            n_missing = int(series.isna().sum())
            missing_rate = round(n_missing / max(n_total, 1), 4)

            predict_missing_rate = None
            if self.predict_df is not None and col in self.predict_df.columns:
                pred_n = len(self.predict_df[col])
                pred_miss = int(self.predict_df[col].isna().sum())
                predict_missing_rate = round(pred_miss / max(pred_n, 1), 4)

            columns[col] = {
                "n_total": n_total,
                "n_missing": n_missing,
                "missing_rate": missing_rate,
                "severity": _severity(missing_rate),
                "dtype": str(series.dtype),
                "dtype_category": _dtype_category(series),
                "is_target": col == self.target_col,
                "predict_missing_rate": predict_missing_rate,
            }

        cols_with_missing = [c for c, v in columns.items() if v["n_missing"] > 0]
        total_cells = self.df.size
        total_missing = int(self.df.isna().sum().sum())

        return {
            "columns": columns,
            "summary": {
                "total_rows": len(self.df),
                "total_columns": len(self.df.columns),
                "columns_with_any_missing": len(cols_with_missing),
                "total_missing_cells": total_missing,
                "overall_missing_rate": round(total_missing / max(total_cells, 1), 4),
            },
            "columns_with_missing": cols_with_missing,
        }
