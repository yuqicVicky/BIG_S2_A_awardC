"""Validation strategy recommender.

Strategies:
  random_holdout, kfold, stratified_kfold, time_holdout,
  group_split, group_time_split, within_period_latest_available_holdout,
  rolling_split
"""

from __future__ import annotations

import pandas as pd
from typing import Optional


_STRATEGIES = {
    "within_period": {
        "strategy": "within_period_latest_available_holdout",
        "confidence": "high",
        "reason": (
            "Prediction timestamps overlap with training period. "
            "The correct validation approach holds out the latest available "
            "observations within each period (e.g., the most recent record per "
            "entity or time window) and trains on earlier records in the same period."
        ),
        "fit_rule": "Train on all but the latest N observations within each period.",
        "validation_rule": "Validate on the latest observations that mirror the prediction scenario.",
        "avoid": ["time_holdout", "random_holdout"],
    },
    "within_period_group": {
        "strategy": "within_period_latest_available_holdout",
        "confidence": "high",
        "reason": (
            "Prediction period overlaps training AND groups have partial overlap. "
            "Use within-period latest-available holdout stratified by group."
        ),
        "fit_rule": "Train on all but the latest N observations per group per period.",
        "validation_rule": "Validate on the latest per-group observations.",
        "avoid": ["time_holdout", "kfold"],
    },
    "time_based_split": {
        "strategy": "time_holdout",
        "confidence": "high",
        "reason": "Prediction data comes strictly after training data — temporal ordering must be respected.",
        "fit_rule": "Train on all data up to a cutoff date.",
        "validation_rule": "Validate on data after the cutoff (simulating the prediction window).",
        "avoid": ["kfold", "random_holdout", "stratified_kfold"],
    },
    "group_based_split": {
        "strategy": "group_split",
        "confidence": "high",
        "reason": "Prediction groups are largely unseen at training time — group leakage must be prevented.",
        "fit_rule": "Train on a subset of groups; hold out a disjoint set of groups for validation.",
        "validation_rule": "Ensure no group appears in both train and validation splits.",
        "avoid": ["kfold", "random_holdout"],
    },
    "group_time_split": {
        "strategy": "group_time_split",
        "confidence": "high",
        "reason": "Both group and temporal structure exist; splits must respect both dimensions.",
        "fit_rule": "Train on groups up to time cutoff; validate on same groups after cutoff or new groups.",
        "validation_rule": "Ensure temporal ordering is preserved within each group.",
        "avoid": ["kfold", "random_holdout"],
    },
    "iid_random": {
        "strategy": "kfold",
        "confidence": "medium",
        "reason": "No strong temporal or group signal detected; iid assumption is reasonable.",
        "fit_rule": "Use standard K-Fold or stratified K-Fold cross-validation.",
        "validation_rule": "Randomly assign rows to folds.",
        "avoid": ["time_holdout"],
    },
}


class ValidationRecommender:
    def __init__(
        self,
        train_df: pd.DataFrame,
        predict_df: Optional[pd.DataFrame],
        feature_type_report: dict,
        distribution_shift_report: dict,
        *,
        target_col: Optional[str] = None,
    ):
        self.train_df = train_df
        self.predict_df = predict_df
        self.ft = feature_type_report.get("columns", {})
        self.shift_report = distribution_shift_report
        self.target_col = target_col

    def recommend(self) -> dict:
        split_pattern = self.shift_report.get("split_pattern", {})
        pattern = split_pattern.get("pattern", "iid_random")
        evidence = split_pattern.get("evidence", [])

        base = _STRATEGIES.get(pattern, _STRATEGIES["iid_random"]).copy()

        # Upgrade iid to stratified_kfold if target is categorical/binary
        if pattern == "iid_random" and self.target_col in self.ft:
            ftype = self.ft[self.target_col].get("inferred_type", "")
            if ftype in ("boolean_binary", "categorical_low_cardinality"):
                base = base.copy()
                base["strategy"] = "stratified_kfold"
                base["reason"] += " Target appears categorical — stratification recommended."

        # Add rolling split as alternative for time-based
        alternatives = []
        if pattern in ("time_based_split", "within_period", "group_time_split"):
            alternatives.append({
                "strategy": "rolling_split",
                "reason": "Rolling window validation can better capture temporal model decay.",
            })

        return {
            "detected_split_pattern": pattern,
            "evidence": evidence,
            "recommendation": base,
            "alternatives": alternatives,
        }
