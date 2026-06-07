"""Column-specific imputation strategy planning."""

from __future__ import annotations

from collections import Counter

_HIGH_MISSING_THRESHOLD = 0.80
_MODERATE_MISSING_THRESHOLD = 0.10


class ImputationPlanner:
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
    group_median                   : fill with per-group median of a correlated categorical
                                     feature (e.g. state/region); leakage-safe, fitted on train
    categorical_missing_token      : fill with a literal "MISSING" token
    categorical_mode_plus_indicator: fill with mode + add binary missing indicator
    structural_none_or_zero        : encode as None/zero; add a missing indicator
    drop_column                    : >80% missing, too sparse to be useful
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

    def plan(self) -> dict:
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
            mech_info = col_mechanisms.get(col, {})
            mech = mech_info.get("mechanism_label", "MCAR-compatible")
            target_signal = mech_info.get("target_signal", False)
            is_structural = col_structural.get(col, {}).get("is_structural", False)

            # Find the top correlated categorical feature for potential group imputation.
            # Prefer the dedicated categorical_correlated_features list; fall back to
            # checking the numeric correlated_features list for any categorical dtype.
            top_cat_feature = None
            cat_corr = mech_info.get("categorical_correlated_features", [])
            if cat_corr:
                top_cat_feature = cat_corr[0]["feature"]
            else:
                for feat_info in mech_info.get("correlated_features", []):
                    feat_col = feat_info["feature"]
                    feat_prof = col_profiles.get(feat_col, {})
                    if feat_prof.get("dtype_category") in ("categorical", "boolean"):
                        top_cat_feature = feat_col
                        break

            strategy, add_indicator, reason, group_col = self._decide(
                miss_rate, dtype_cat, mech, target_signal, is_structural, top_cat_feature
            )
            entry: dict = {
                "strategy": strategy,
                "add_missing_indicator": add_indicator,
                "reason": reason,
                "fit_on": "train_only" if strategy != "no_imputation_needed" else "N/A",
            }
            if group_col:
                entry["group_col"] = group_col
            plan[col] = entry

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
        top_cat_feature: str | None = None,
    ) -> tuple[str, bool, str, str | None]:
        if miss_rate == 0.0:
            return "no_imputation_needed", False, "no_missing_values", None

        if miss_rate > _HIGH_MISSING_THRESHOLD:
            return "drop_column", False, "over_80_percent_missing", None

        if is_structural:
            return "structural_none_or_zero", True, "structural_absence_pattern_detected", None

        if dtype_cat == "numeric":
            mar_or_mnar = mech in ("MAR-like evidence", "MNAR/structural concern")
            # Use group median when MAR is detected and a correlated categorical
            # feature exists (e.g. state/region for weather data).
            if mar_or_mnar and top_cat_feature and miss_rate >= _MODERATE_MISSING_THRESHOLD:
                return (
                    "group_median",
                    True,
                    f"MAR_grouped_by_correlated_feature={top_cat_feature!r}",
                    top_cat_feature,
                )
            if mar_or_mnar or target_signal or miss_rate >= _MODERATE_MISSING_THRESHOLD:
                return (
                    "numeric_median_plus_indicator",
                    True,
                    f"mechanism={mech!r}_or_moderate_missing_rate",
                    None,
                )
            return "numeric_median", False, "mcar_compatible_low_missing_rate", None

        # categorical
        if miss_rate >= _MODERATE_MISSING_THRESHOLD:
            return "categorical_mode_plus_indicator", True, "moderate_or_high_missing_categorical", None
        return "categorical_missing_token", False, "low_missing_categorical", None

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
