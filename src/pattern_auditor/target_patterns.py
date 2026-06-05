"""Target-aware pattern discovery: correlations, target means by feature."""

from __future__ import annotations

import warnings
import pandas as pd
import numpy as np
from typing import Optional

from .datetime_patterns import DatetimePatternAnalyser


class TargetPatternAnalyser:
    def __init__(
        self,
        train_df: pd.DataFrame,
        target_col: str,
        feature_type_report: dict,
    ):
        self.df = train_df.copy()
        self.target_col = target_col
        self.ft = feature_type_report.get("columns", {})

    def analyse(self) -> dict:
        target = self.df[self.target_col]
        result: dict = {
            "target_col": self.target_col,
            "target_dtype": str(target.dtype),
            "target_stats": _safe_stats(target),
            "numeric_correlations": [],
            "categorical_target_means": [],
            "datetime_target_patterns": [],
            "group_target_patterns": [],
            "text_target_patterns": [],
        }

        for col, info in self.ft.items():
            ftype = info.get("inferred_type", "")
            if col == self.target_col or col not in self.df.columns:
                continue

            if ftype in ("numeric_continuous", "numeric_discrete"):
                result["numeric_correlations"].append(
                    _numeric_correlation(self.df, col, self.target_col)
                )

            elif ftype in (
                "categorical_low_cardinality",
                "categorical_high_cardinality",
                "boolean_binary",
                "numeric_discrete",
            ):
                result["categorical_target_means"].append(
                    _categorical_target_mean(self.df, col, self.target_col)
                )

            elif ftype == "datetime_like":
                dpa = DatetimePatternAnalyser(self.df, col, self.target_col)
                cov = dpa.coverage()
                by_comp = dpa.target_by_component()
                result["datetime_target_patterns"].append(
                    {"column": col, "coverage": cov, "by_component": by_comp}
                )

            elif ftype == "group_entity_id":
                result["group_target_patterns"].append(
                    _group_target_pattern(self.df, col, self.target_col)
                )

            elif ftype == "text_like":
                result["text_target_patterns"].append(
                    _text_target_pattern(self.df, col, self.target_col)
                )

        # Sort correlations by abs value
        result["numeric_correlations"].sort(
            key=lambda x: abs(x.get("pearson_r") or 0), reverse=True
        )
        return result


# ---------------------------------------------------------------------------
# Per-type helpers
# ---------------------------------------------------------------------------

def _safe_stats(series: pd.Series) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            desc = series.describe()
            return {k: _jval(v) for k, v in desc.to_dict().items()}
        except Exception:
            return {}


def _jval(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return None if np.isnan(v) else float(v)
    return v


def _numeric_correlation(df: pd.DataFrame, col: str, target_col: str) -> dict:
    sub = df[[col, target_col]].dropna()
    if len(sub) < 5:
        return {"column": col, "pearson_r": None, "note": "too few rows"}
    try:
        r = float(sub[col].corr(sub[target_col]))
        r = None if np.isnan(r) else r
    except Exception:
        r = None

    # Target mean by quantile bin
    try:
        sub = sub.copy()
        sub["__q"] = pd.qcut(sub[col], q=5, duplicates="drop")
        bin_means = (
            sub.groupby("__q", observed=True)[target_col]
            .agg(["mean", "count"])
            .reset_index()
        )
        bin_means.columns = ["bin", "target_mean", "count"]
        bin_means["bin"] = bin_means["bin"].astype(str)
        bins = bin_means.to_dict(orient="records")
    except Exception:
        bins = []

    return {"column": col, "pearson_r": r, "target_mean_by_quantile": bins}


def _categorical_target_mean(df: pd.DataFrame, col: str, target_col: str) -> dict:
    sub = df[[col, target_col]].dropna(subset=[col])
    agg = (
        sub.groupby(col)[target_col]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "target_mean", "count": "count"})
        .sort_values("count", ascending=False)
    )

    n_categories = int(agg[col].nunique())
    rare_categories = agg[agg["count"] < 5][col].tolist()
    records = agg.head(50).copy()
    records[col] = records[col].astype(str)
    records["target_mean"] = records["target_mean"].apply(
        lambda x: None if (isinstance(x, float) and np.isnan(x)) else float(x)
    )

    return {
        "column": col,
        "n_categories": n_categories,
        "target_mean_by_category": records.to_dict(orient="records"),
        "rare_categories_warning": rare_categories[:20],
    }


def _group_target_pattern(df: pd.DataFrame, col: str, target_col: str) -> dict:
    group_sizes = df[col].value_counts().describe().to_dict()
    agg = (
        df.groupby(col)[target_col]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "target_mean", "count": "group_size"})
    )
    return {
        "column": col,
        "n_groups": int(agg[col].nunique()),
        "group_size_stats": {k: _jval(v) for k, v in group_sizes.items()},
        "target_mean_by_group_sample": agg.head(30).astype(str).to_dict(orient="records"),
    }


def _text_target_pattern(df: pd.DataFrame, col: str, target_col: str) -> dict:
    sub = df[[col, target_col]].dropna(subset=[col]).copy()
    sub["__len"] = sub[col].astype(str).str.len()
    sub["__words"] = sub[col].astype(str).str.split().str.len()

    try:
        sub["__len_bin"] = pd.qcut(sub["__len"], q=4, duplicates="drop")
        bin_means = (
            sub.groupby("__len_bin", observed=True)[target_col]
            .mean()
            .reset_index()
        )
        bin_means.columns = ["length_bin", "target_mean"]
        bin_means["length_bin"] = bin_means["length_bin"].astype(str)
        bins = bin_means.to_dict(orient="records")
    except Exception:
        bins = []

    return {
        "column": col,
        "avg_length": float(sub["__len"].mean()),
        "avg_word_count": float(sub["__words"].mean()),
        "target_mean_by_length_bin": bins,
    }
