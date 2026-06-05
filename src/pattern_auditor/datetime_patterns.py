"""Datetime pattern analysis: coverage, granularity, derived-feature target signals."""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Optional


class DatetimePatternAnalyser:
    """Extracts datetime-derived features and computes target aggregations."""

    def __init__(
        self,
        df: pd.DataFrame,
        datetime_col: str,
        target_col: Optional[str] = None,
    ):
        self.df = df.copy()
        self.datetime_col = datetime_col
        self.target_col = target_col
        self._ensure_parsed()

    def _ensure_parsed(self):
        col = self.datetime_col
        if not pd.api.types.is_datetime64_any_dtype(self.df[col]):
            try:
                self.df[col] = pd.to_datetime(self.df[col], infer_datetime_format=True)
            except Exception:
                try:
                    self.df[col] = pd.to_datetime(self.df[col], unit="s")
                except Exception:
                    pass

    def extract_components(self) -> pd.DataFrame:
        """Return df with added temporal component columns."""
        ts = self.df[self.datetime_col]
        out = self.df.copy()
        if pd.api.types.is_datetime64_any_dtype(ts):
            out["__hour"] = ts.dt.hour
            out["__dayofweek"] = ts.dt.dayofweek
            out["__month"] = ts.dt.month
            out["__year"] = ts.dt.year
            out["__dayofyear"] = ts.dt.dayofyear
            out["__weekofyear"] = ts.dt.isocalendar().week.astype(int)
            out["__is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)
            out["__quarter"] = ts.dt.quarter
        return out

    def coverage(self) -> dict:
        ts = self.df[self.datetime_col]
        if not pd.api.types.is_datetime64_any_dtype(ts):
            return {"error": "Could not parse datetime column"}
        ts_clean = ts.dropna()
        return {
            "min": str(ts_clean.min()),
            "max": str(ts_clean.max()),
            "n_rows": int(ts_clean.count()),
            "n_unique_dates": int(ts_clean.dt.date.nunique()),
            "inferred_granularity": _infer_granularity(ts_clean),
        }

    def target_by_component(self) -> dict:
        if not self.target_col or self.target_col not in self.df.columns:
            return {}
        df = self.extract_components()
        target = df[self.target_col]
        result = {}
        for comp in ["__hour", "__dayofweek", "__month", "__year", "__quarter"]:
            if comp in df.columns:
                agg = df.groupby(comp)[self.target_col].agg(["mean", "count"]).reset_index()
                agg.columns = [comp, "target_mean", "count"]
                result[comp.lstrip("_")] = agg.to_dict(orient="records")

        # Interactions
        if "__hour" in df.columns and "__dayofweek" in df.columns:
            agg2 = (
                df.groupby(["__hour", "__dayofweek"])[self.target_col]
                .mean()
                .reset_index()
            )
            agg2.columns = ["hour", "dayofweek", "target_mean"]
            result["hour_x_dayofweek"] = agg2.to_dict(orient="records")

        if "__month" in df.columns and "__year" in df.columns:
            agg3 = (
                df.groupby(["__year", "__month"])[self.target_col]
                .mean()
                .reset_index()
            )
            agg3.columns = ["year", "month", "target_mean"]
            result["month_x_year"] = agg3.to_dict(orient="records")

        return result


def _infer_granularity(ts: pd.Series) -> str:
    if len(ts) < 2:
        return "unknown"
    diffs = ts.sort_values().diff().dropna()
    median_diff = diffs.median()
    if pd.isna(median_diff):
        return "unknown"
    secs = median_diff.total_seconds()
    if secs < 120:
        return "minute"
    if secs < 7200:
        return "hour"
    if secs < 172800:
        return "day"
    if secs < 1296000:
        return "week"
    return "month_or_more"
