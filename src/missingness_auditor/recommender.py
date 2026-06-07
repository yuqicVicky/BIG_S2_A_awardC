"""Column-specific imputation strategy recommendation."""

from __future__ import annotations

from collections import Counter

_HIGH_MISSING_THRESHOLD = 0.80
_MODERATE_MISSING_THRESHOLD = 0.10


class ImputationRecommender:
    """
    Recommends a per-column imputation strategy based on:
    - missingness profile (rate, dtype)
    - mechanism clue (MCAR-compatible / MAR-like / MNAR-concern)
    - structural patterns (categorical NA → numeric zero/absent)

    Available strategies
    --------------------
    no_imputation_needed           : column has no missing values
    numeric_median                 : fill with training-set median
    numeric_median_plus_indicator  : fill with median + add binary missing indicator
    categorical_missing_token      : fill with a literal "MISSING" token
    categorical_mode_plus_indicator: fill with mode + add binary missing indicator
    structural_none_or_zero        : encode as None/zero; keep a missing indicator
    drop_column                    : >80% missing, column too sparse to be useful
    model_based_imputation_optional: optional for complex MAR with moderate missingness
    """

    def __init__(
        self,
        profile: dict,
        mechanism_audit: dict,
        structural_audit: dict,
    ):
        self.profile = profile
        self.mechanism_audit = mechanism_audit
        self.structural_audit = structural_audit

    def recommend(self) -> dict:
        col_profiles = self.profile.get("columns", {})
        col_mechanisms = self.mechanism_audit.get("columns", {})
        col_structural = self.structural_audit.get("column_flags", {})

        plan: dict[str, dict] = {}
        for col, prof in col_profiles.items():
            if prof.get("is_target"):
                plan[col] = {
                    "strategy": "no_imputation_needed",
                    "add_missing_indicator": False,
                    "reason": "target_column_excluded_from_imputation",
                    "fit_on": "N/A",
                }
                continue

            miss_rate = prof["missing_rate"]
            dtype_cat = prof["dtype_category"]
            mech = col_mechanisms.get(col, {}).get("mechanism_label", "MCAR-compatible")
            target_signal = col_mechanisms.get(col, {}).get("target_signal", False)
            is_structural = col_structural.get(col, {}).get("is_structural", False)

            strategy, add_indicator, reason = self._decide(
                miss_rate, dtype_cat, mech, target_signal, is_structural
            )
            plan[col] = {
                "strategy": strategy,
                "add_missing_indicator": add_indicator,
                "reason": reason,
                "fit_on": "train_only" if strategy != "no_imputation_needed" else "N/A",
            }

        return {
            "columns": plan,
            "summary": self._summarize(plan),
            "leakage_protocol": (
                "All imputation statistics (median, mode, model parameters) "
                "must be fitted on training data only and applied to prediction data."
            ),
        }

    def _decide(
        self,
        miss_rate: float,
        dtype_cat: str,
        mech: str,
        target_signal: bool,
        is_structural: bool,
    ) -> tuple[str, bool, str]:
        if miss_rate == 0.0:
            return "no_imputation_needed", False, "no_missing_values"

        if miss_rate > _HIGH_MISSING_THRESHOLD:
            return "drop_column", False, "over_80_percent_missing"

        if is_structural:
            return "structural_none_or_zero", True, "structural_absence_pattern_detected"

        if dtype_cat == "numeric":
            mar_or_mnar = mech in ("MAR-like evidence", "MNAR/structural concern")
            if mar_or_mnar or target_signal or miss_rate >= _MODERATE_MISSING_THRESHOLD:
                return (
                    "numeric_median_plus_indicator",
                    True,
                    f"mechanism={mech!r}_or_moderate_missing_rate",
                )
            return "numeric_median", False, "mcar_compatible_low_missing_rate"

        # categorical
        if miss_rate >= _MODERATE_MISSING_THRESHOLD:
            return "categorical_mode_plus_indicator", True, "moderate_or_high_missing_categorical"
        return "categorical_missing_token", False, "low_missing_categorical"

    def _summarize(self, plan: dict) -> dict:
        strategy_counts = Counter(v["strategy"] for v in plan.values())
        indicator_cols = [c for c, v in plan.items() if v["add_missing_indicator"]]
        return {
            "strategy_counts": dict(strategy_counts),
            "columns_needing_missing_indicator": indicator_cols,
            "n_columns_to_impute": sum(
                1 for v in plan.values() if v["strategy"] != "no_imputation_needed"
            ),
        }
