"""
Missingness mechanism clue detection.

Labels are cautious and statistical, not causal:
- "MCAR-compatible"         : no significant correlation with observed features or target
- "MAR-like evidence"       : missingness correlates with at least one observed feature
- "MNAR/structural concern" : missingness correlates with the target variable
"""

from __future__ import annotations

import pandas as pd
import numpy as np

_CORR_THRESHOLD_MAR = 0.15
_CORR_THRESHOLD_TARGET = 0.10


class MechanismAuditor:
    """
    For each column with missing values, compute a missingness indicator and
    correlate it with observed features and the target.

    Results are labelled MCAR-compatible / MAR-like / MNAR-concern.
    These are statistical clues only — not causal mechanism assignments.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        target_col: str | None = None,
        corr_threshold_mar: float = _CORR_THRESHOLD_MAR,
        corr_threshold_target: float = _CORR_THRESHOLD_TARGET,
    ):
        self.df = df
        self.target_col = target_col
        self.corr_threshold_mar = corr_threshold_mar
        self.corr_threshold_target = corr_threshold_target
        self._numeric_feature_cols = [
            c for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c]) and c != target_col
        ]
        self._categorical_feature_cols = [
            c for c in df.columns
            if not pd.api.types.is_numeric_dtype(df[c])
            and c != target_col
            and df[c].notna().any()
        ]

    def audit(self) -> dict:
        missing_cols = [c for c in self.df.columns if self.df[c].isna().any()]
        results = {col: self._analyze(col) for col in missing_cols}
        return {
            "columns": results,
            "note": (
                "Mechanism labels are statistical clues from observational data only. "
                "They do not imply a causal missingness mechanism."
            ),
        }

    def _analyze(self, col: str) -> dict:
        miss_ind = self.df[col].isna().astype(int)

        # Correlate missingness with observed numeric features
        mar_correlations: list[dict] = []
        for other in self._numeric_feature_cols:
            if other == col:
                continue
            mask = self.df[other].notna()
            if mask.sum() < 20:
                continue
            try:
                r = float(miss_ind[mask].corr(self.df[other][mask]))
                if pd.notna(r) and abs(r) >= self.corr_threshold_mar:
                    mar_correlations.append({"feature": other, "correlation": round(abs(r), 4)})
            except Exception:
                pass
        mar_correlations.sort(key=lambda x: -x["correlation"])

        # Correlate missingness with categorical features via group missingness spread
        cat_correlations: list[dict] = []
        for cat_col in self._categorical_feature_cols:
            if cat_col == col:
                continue
            groups = self.df[cat_col].dropna().unique()
            if len(groups) < 2 or len(groups) > 50:
                continue
            rates: dict[str, float] = {}
            for g in groups:
                mask = self.df[cat_col] == g
                if mask.sum() >= 5:
                    rates[str(g)] = float(miss_ind[mask].mean())
            if len(rates) < 2:
                continue
            spread = max(rates.values()) - min(rates.values())
            if spread >= self.corr_threshold_mar:
                cat_correlations.append({"feature": cat_col, "correlation": round(spread, 4)})
        cat_correlations.sort(key=lambda x: -x["correlation"])

        # Correlate missingness with target
        target_correlation: float | None = None
        target_signal = False
        if self.target_col and self.target_col in self.df.columns:
            tgt = self.df[self.target_col]
            if pd.api.types.is_numeric_dtype(tgt):
                mask = tgt.notna()
                if mask.sum() >= 20:
                    try:
                        r = float(miss_ind[mask].corr(tgt[mask]))
                        if pd.notna(r):
                            target_correlation = round(abs(r), 4)
                            target_signal = abs(r) >= self.corr_threshold_target
                    except Exception:
                        pass
            else:
                groups = tgt.dropna().unique()
                if len(groups) >= 2:
                    rates = {
                        str(g): float(miss_ind[tgt == g].mean())
                        for g in groups
                    }
                    spread = max(rates.values()) - min(rates.values())
                    target_correlation = round(spread, 4)
                    target_signal = spread >= self.corr_threshold_target

        if target_signal:
            label = "MNAR/structural concern"
            evidence = "missingness_correlated_with_target"
        elif mar_correlations or cat_correlations:
            label = "MAR-like evidence"
            evidence = "missingness_correlated_with_observed_features"
        else:
            label = "MCAR-compatible"
            evidence = "no_significant_correlation_found"

        return {
            "mechanism_label": label,
            "mechanism_evidence": evidence,
            "correlated_features": mar_correlations[:5],
            "categorical_correlated_features": cat_correlations[:5],
            "target_correlation": target_correlation,
            "target_signal": target_signal,
        }
