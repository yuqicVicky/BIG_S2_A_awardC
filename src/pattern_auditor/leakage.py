"""Leakage and feature availability audit."""

from __future__ import annotations

import re
import pandas as pd
import numpy as np
from typing import Optional


_TARGET_COMPONENT_PATTERNS = re.compile(
    r"(_target$|_label$|_pred$|_score$|_outcome$|_response$"
    r"|^target_|^label_|^pred_|^forecast_|^outcome_|^response_"
    r"|^y_pred|^y_hat|_y_pred|_y_hat)",
    re.IGNORECASE,
)
_ID_LIKE = re.compile(r"(_id|_key|_hash|_uuid|rownum|index|_index)$", re.IGNORECASE)


class LeakageAuditor:
    def __init__(
        self,
        train_df: pd.DataFrame,
        predict_df: Optional[pd.DataFrame],
        feature_type_report: dict,
        *,
        target_col: Optional[str] = None,
    ):
        self.train_df = train_df
        self.predict_df = predict_df
        self.ft = feature_type_report.get("columns", {})
        self.target_col = target_col

    def audit(self) -> dict:
        risks: list[dict] = []

        for col, info in self.ft.items():
            ftype = info.get("inferred_type", "")
            if col == self.target_col:
                continue

            risk = self._assess_column(col, ftype)
            if risk:
                risks.append(risk)

        high = [r for r in risks if r["severity"] == "high"]
        medium = [r for r in risks if r["severity"] == "medium"]
        low = [r for r in risks if r["severity"] == "low"]

        return {
            "risks": risks,
            "summary": {
                "high": len(high),
                "medium": len(medium),
                "low": len(low),
            },
        }

    def _assess_column(self, col: str, ftype: str) -> Optional[dict]:
        # Train-only column available at training but not prediction → leakage risk
        if ftype == "train_only":
            return {
                "column": col,
                "risk_type": "train_only_feature",
                "severity": "high",
                "reason": (
                    "Column is present in training data but absent in prediction data. "
                    "If used as a feature, the model cannot generate predictions."
                ),
                "recommendation": "Drop this column from features or investigate if it should exist in predict data.",
            }

        # Target-named component suspect
        if _TARGET_COMPONENT_PATTERNS.search(col) and col != self.target_col:
            return {
                "column": col,
                "risk_type": "target_component_suspect",
                "severity": "high",
                "reason": "Column name suggests it may be derived from or closely related to the target.",
                "recommendation": "Verify whether this column is computed after the target is known. If so, drop it.",
            }

        # Raw ID leakage — high cardinality row identifier
        if ftype == "row_id":
            return {
                "column": col,
                "risk_type": "row_id_leakage",
                "severity": "medium",
                "reason": "Row identifiers can cause models to memorize training examples.",
                "recommendation": "Drop or exclude from features.",
            }

        if _ID_LIKE.search(col) and ftype in ("numeric_continuous", "numeric_discrete", "categorical_high_cardinality"):
            return {
                "column": col,
                "risk_type": "raw_id_leakage_risk",
                "severity": "medium",
                "reason": "Column name looks like a raw ID and may cause memorization.",
                "recommendation": "Exclude from features or hash/encode carefully.",
            }

        # High-cardinality memorization risk
        if ftype == "categorical_high_cardinality":
            n_unique = self.ft[col].get("n_unique", 0)
            n_rows = len(self.train_df)
            ratio = n_unique / max(n_rows, 1)
            if ratio > 0.9:
                return {
                    "column": col,
                    "risk_type": "high_cardinality_memorization",
                    "severity": "medium",
                    "reason": f"Cardinality ratio {ratio:.2f} — model may memorize training IDs.",
                    "recommendation": "Consider hashing, target encoding with proper CV, or dropping.",
                }

        # Near-constant in train but NOT datetime-derived
        if ftype == "near_constant":
            return {
                "column": col,
                "risk_type": "near_constant",
                "severity": "low",
                "reason": "Near-constant column provides almost no generalizable signal.",
                "recommendation": "Drop to reduce noise.",
            }

        # Note: datetime_like columns are intentionally NOT flagged as leakage here.
        # Timestamp features available at prediction time carry valid signal.

        return None
