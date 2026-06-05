"""Train vs prediction distribution shift and split-pattern detection."""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Optional
from scipy import stats as scipy_stats


_KS_PVALUE_THRESH = 0.05
_OVERLAP_LOW_THRESH = 0.5   # < 50% group overlap → potential group split


class DistributionShiftDetector:
    def __init__(
        self,
        train_df: pd.DataFrame,
        predict_df: Optional[pd.DataFrame],
        feature_type_report: dict,
    ):
        self.train_df = train_df
        self.predict_df = predict_df
        self.ft = feature_type_report.get("columns", {})

    def detect(self) -> dict:
        if self.predict_df is None:
            return {"note": "No prediction set provided; skipping shift detection."}

        result: dict = {
            "row_counts": {
                "train": len(self.train_df),
                "predict": len(self.predict_df),
            },
            "numeric_shift": [],
            "categorical_overlap": [],
            "datetime_coverage": [],
            "split_pattern": _detect_split_pattern(
                self.train_df, self.predict_df, self.ft
            ),
        }

        for col, info in self.ft.items():
            ftype = info.get("inferred_type", "")
            if col not in self.train_df.columns or col not in self.predict_df.columns:
                continue

            if ftype in ("numeric_continuous", "numeric_discrete"):
                result["numeric_shift"].append(
                    _ks_shift(self.train_df[col], self.predict_df[col], col)
                )

            elif ftype in (
                "categorical_low_cardinality",
                "categorical_high_cardinality",
                "boolean_binary",
                "group_entity_id",
            ):
                result["categorical_overlap"].append(
                    _category_overlap(self.train_df[col], self.predict_df[col], col)
                )

            elif ftype == "datetime_like":
                result["datetime_coverage"].append(
                    _datetime_coverage(self.train_df[col], self.predict_df[col], col)
                )

        return result


# ---------------------------------------------------------------------------

def _ks_shift(train_s: pd.Series, pred_s: pd.Series, col: str) -> dict:
    t = train_s.dropna().astype(float)
    p = pred_s.dropna().astype(float)
    if len(t) < 3 or len(p) < 3:
        return {"column": col, "ks_statistic": None, "ks_pvalue": None, "shift_detected": False}
    try:
        stat, pval = scipy_stats.ks_2samp(t.values, p.values)
        shift = bool(pval < _KS_PVALUE_THRESH)
        return {
            "column": col,
            "ks_statistic": round(float(stat), 4),
            "ks_pvalue": round(float(pval), 4),
            "shift_detected": shift,
            "train_mean": round(float(t.mean()), 4),
            "predict_mean": round(float(p.mean()), 4),
        }
    except Exception as e:
        return {"column": col, "error": str(e)}


def _category_overlap(train_s: pd.Series, pred_s: pd.Series, col: str) -> dict:
    train_cats = set(train_s.dropna().unique())
    pred_cats = set(pred_s.dropna().unique())
    overlap = train_cats & pred_cats
    train_only = sorted(str(c) for c in train_cats - pred_cats)
    pred_only = sorted(str(c) for c in pred_cats - train_cats)
    overlap_ratio = len(overlap) / max(len(pred_cats), 1)
    return {
        "column": col,
        "n_train_categories": len(train_cats),
        "n_predict_categories": len(pred_cats),
        "n_overlap": len(overlap),
        "overlap_ratio": round(overlap_ratio, 4),
        "train_only_categories": train_only[:20],
        "predict_only_categories": pred_only[:20],
    }


def _datetime_coverage(train_s: pd.Series, pred_s: pd.Series, col: str) -> dict:
    def _parse(s):
        if pd.api.types.is_datetime64_any_dtype(s):
            return s
        try:
            return pd.to_datetime(s, infer_datetime_format=True)
        except Exception:
            return s

    t = _parse(train_s).dropna()
    p = _parse(pred_s).dropna()
    result: dict = {"column": col}
    try:
        result["train_min"] = str(t.min())
        result["train_max"] = str(t.max())
        result["predict_min"] = str(p.min())
        result["predict_max"] = str(p.max())
        result["predict_after_train"] = str(p.min()) > str(t.max())
        result["time_gap_days"] = _time_gap_days(t.max(), p.min())
    except Exception as e:
        result["error"] = str(e)
    return result


def _time_gap_days(train_max, pred_min) -> Optional[float]:
    try:
        gap = pred_min - train_max
        return round(gap.total_seconds() / 86400, 2)
    except Exception:
        return None


def _detect_split_pattern(
    train_df: pd.DataFrame,
    predict_df: pd.DataFrame,
    ft: dict,
) -> dict:
    """Heuristically classify the train/predict split pattern."""
    evidence: list[str] = []
    has_time = False
    has_group = False
    within_period = False
    time_based = False
    group_based = False

    # --- Detect time-based split ---
    datetime_cols = [c for c, i in ft.items() if i.get("inferred_type") == "datetime_like"
                     and c in train_df.columns and c in predict_df.columns]
    for col in datetime_cols:
        cov = _datetime_coverage(train_df[col], predict_df[col], col)
        train_max = cov.get("train_max", "")
        pred_min = cov.get("predict_min", "")
        pred_max = cov.get("predict_max", "")
        if train_max and pred_min and pred_max:
            if pred_min <= train_max:
                within_period = True
                evidence.append(f"{col}: predict overlaps or precedes train max → within-period pattern")
            elif cov.get("predict_after_train"):
                time_based = True
                evidence.append(f"{col}: predict starts after train ends → time-based split")
        has_time = True

    # --- Detect group-based split ---
    group_cols = [c for c, i in ft.items() if i.get("inferred_type") == "group_entity_id"
                  and c in train_df.columns and c in predict_df.columns]
    for col in group_cols:
        ov = _category_overlap(train_df[col], predict_df[col], col)
        if ov["overlap_ratio"] < _OVERLAP_LOW_THRESH:
            group_based = True
            evidence.append(f"{col}: low group overlap ({ov['overlap_ratio']:.2f}) → group split")
        has_group = True

    # --- Determine pattern ---
    if within_period and has_group:
        pattern = "within_period_group"
    elif within_period:
        pattern = "within_period"
    elif time_based and group_based:
        pattern = "group_time_split"
    elif time_based:
        pattern = "time_based_split"
    elif group_based:
        pattern = "group_based_split"
    else:
        pattern = "iid_random"
        evidence.append("No strong temporal or group signal → assume iid random split")

    return {"pattern": pattern, "evidence": evidence}
