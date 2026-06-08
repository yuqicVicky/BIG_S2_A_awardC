"""
Missingness mechanism clue detection.

Labels are cautious and evidence-based, not causal:
- "MCAR-compatible"                        : no significant correlation with observed features or target
- "MAR-like evidence"                      : missingness correlates with at least one observed numeric feature
- "group-dependent missingness"            : missingness concentrated in specific categorical groups;
                                             some groups have disproportionately high missing rates
- "target-associated missingness"          : missingness correlates with the target variable
- "high-cardinality text/category missingness": missingness in a high-cardinality text-like column
- "insufficient evidence"                  : too few rows to determine mechanism

Note: "structural absence concern" is determined by StructuralMissingnessDetector, not here.

MAR robustness rule (Collins et al., 2001 via van Buuren FIMD Ch5):
  When missing rate < 25% AND max observed correlation < 0.4, omitting a lurking variable
  from the imputation model has negligible effect on regression estimates. The MAR assumption
  is "likely robust" in that regime.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

_CORR_THRESHOLD_MAR = 0.15
_CORR_THRESHOLD_TARGET = 0.10
_GROUP_SPREAD_THRESHOLD = 0.20
_MIN_ROWS_FOR_ANALYSIS = 20
_HIGH_CARDINALITY_ABS = 50
_HIGH_CARDINALITY_RATIO = 0.20

# Collins et al. (2001) MAR robustness thresholds (via van Buuren FIMD Ch5)
_MAR_ROBUST_MAX_MISS_RATE = 0.25
_MAR_ROBUST_MAX_CORR = 0.40


class MechanismAuditor:
    """
    For each column with missing values, compute a missingness indicator and
    correlate it with observed features and the target.

    Labels are cautious statistical clues — not causal mechanism assignments.

    Priority order when multiple signals are present:
    1. group-dependent missingness (categorical group spread dominates)
    2. target-associated missingness (target correlation)
    3. MAR-like evidence (numeric feature correlation)
    4. MCAR-compatible (no significant correlation)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        target_col: str | None = None,
        corr_threshold_mar: float = _CORR_THRESHOLD_MAR,
        corr_threshold_target: float = _CORR_THRESHOLD_TARGET,
        group_spread_threshold: float = _GROUP_SPREAD_THRESHOLD,
    ):
        self.df = df
        self.target_col = target_col
        self.corr_threshold_mar = corr_threshold_mar
        self.corr_threshold_target = corr_threshold_target
        self.group_spread_threshold = group_spread_threshold
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
                "They do not imply a definitive causal missingness mechanism. "
                "MCAR/MAR/MNAR cannot be confirmed from observational data alone. "
                "MAR is the recommended default assumption (van Buuren FIMD Ch5)."
            ),
        }

    def _analyze(self, col: str) -> dict:
        miss_ind = self.df[col].isna().astype(int)
        n_obs = int(miss_ind.sum())

        # Correlate missingness with observed numeric features
        mar_correlations: list[dict] = []
        for other in self._numeric_feature_cols:
            if other == col:
                continue
            mask = self.df[other].notna()
            if mask.sum() < _MIN_ROWS_FOR_ANALYSIS:
                continue
            try:
                r = float(miss_ind[mask].corr(self.df[other][mask]))
                if pd.notna(r) and abs(r) >= self.corr_threshold_mar:
                    mar_correlations.append({"feature": other, "correlation": round(abs(r), 4)})
            except Exception:
                pass
        mar_correlations.sort(key=lambda x: -x["correlation"])

        # Detect group-dependent missingness via categorical feature spread
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
            if spread >= self.group_spread_threshold:
                high_miss_groups = {
                    grp: round(rate, 3)
                    for grp, rate in rates.items()
                    if rate >= 0.5
                }
                cat_correlations.append({
                    "feature": cat_col,
                    "correlation": round(spread, 4),
                    "high_missing_groups": high_miss_groups,
                    "all_group_rates": {k: round(v, 3) for k, v in rates.items()},
                })
        cat_correlations.sort(key=lambda x: -x["correlation"])

        # Correlate missingness with target
        target_correlation: float | None = None
        target_signal = False
        if self.target_col and self.target_col in self.df.columns:
            tgt = self.df[self.target_col]
            if pd.api.types.is_numeric_dtype(tgt):
                mask = tgt.notna()
                if mask.sum() >= _MIN_ROWS_FOR_ANALYSIS:
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
                    rates_t = {
                        str(g): float(miss_ind[tgt == g].mean())
                        for g in groups
                    }
                    spread = max(rates_t.values()) - min(rates_t.values())
                    target_correlation = round(spread, 4)
                    target_signal = spread >= self.corr_threshold_target

        # Check if this column itself is high-cardinality categorical
        is_cat = not pd.api.types.is_numeric_dtype(self.df[col])
        n_total = len(self.df[col])
        unique_count = int(self.df[col].dropna().nunique())
        is_high_card_col = is_cat and (
            unique_count > _HIGH_CARDINALITY_ABS
            or (n_total > 0 and unique_count / max(n_total, 1) > _HIGH_CARDINALITY_RATIO)
        )

        # Insufficient evidence when very few observed rows
        if n_obs < _MIN_ROWS_FOR_ANALYSIS and not cat_correlations and not mar_correlations and not target_signal:
            label = "insufficient evidence"
            evidence = "too_few_missing_rows_for_reliable_analysis"
        elif cat_correlations:
            label = "group-dependent missingness"
            evidence = "missingness_concentrated_in_categorical_groups"
        elif target_signal:
            label = "target-associated missingness"
            evidence = "missingness_correlated_with_target"
        elif mar_correlations:
            label = "MAR-like evidence"
            evidence = "missingness_correlated_with_observed_numeric_features"
        else:
            label = "MCAR-compatible"
            evidence = "no_significant_correlation_found"

        # Build evidence dicts
        group_dependency_evidence: dict | None = None
        if cat_correlations:
            top = cat_correlations[0]
            group_dependency_evidence = {
                "detected": True,
                "top_categorical_feature": top["feature"],
                "max_missingness_spread": top["correlation"],
                "high_missing_groups": top.get("high_missing_groups", {}),
            }
        else:
            group_dependency_evidence = {"detected": False}

        target_association_evidence: dict = {
            "detected": target_signal,
            "correlation": target_correlation,
        }

        covariate_association_evidence = mar_correlations[:5]

        # Collins et al. (2001) MAR robustness: below 25% missing and max corr < 0.4,
        # omitting a lurking variable has negligible effect on regression estimates.
        miss_rate = float(miss_ind.mean())
        max_observed_corr = max(
            (x["correlation"] for x in mar_correlations), default=0.0
        )
        if (
            label == "MCAR-compatible"
            and miss_rate < _MAR_ROBUST_MAX_MISS_RATE
            and max_observed_corr < _MAR_ROBUST_MAX_CORR
        ):
            mar_robustness_note = "likely_robust_per_collins2001"
        elif miss_rate >= _MAR_ROBUST_MAX_MISS_RATE or max_observed_corr >= _MAR_ROBUST_MAX_CORR:
            mar_robustness_note = "mechanism_assumption_may_matter_consider_full_mi"
        else:
            mar_robustness_note = "insufficient_evidence_to_assess"

        return {
            "mechanism_label": label,
            "mechanism_evidence": evidence,
            "mar_robustness_note": mar_robustness_note,
            "correlated_features": mar_correlations[:5],
            "categorical_correlated_features": cat_correlations[:5],
            "target_correlation": target_correlation,
            "target_signal": target_signal,
            "group_dependency_evidence": group_dependency_evidence,
            "target_association_evidence": target_association_evidence,
            "covariate_association_evidence": covariate_association_evidence,
        }
