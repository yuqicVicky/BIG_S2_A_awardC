"""Leakage-safe imputation protocol checker."""

from __future__ import annotations


class LeakageSafeImputationChecker:
    """
    Verifies that the imputation plan does not risk data leakage.

    Core rule: all imputation statistics must be fitted on training data
    only, then applied to the prediction set without re-fitting.
    """

    def __init__(self, imputation_plan: dict, *, has_predict_df: bool = False):
        self.imputation_plan = imputation_plan
        self.has_predict_df = has_predict_df

    def check(self) -> dict:
        findings: list[dict] = []
        for col, entry in self.imputation_plan.get("columns", {}).items():
            findings.append(self._check_column(col, entry))

        high = [f for f in findings if f["leakage_risk"] == "high"]
        medium = [f for f in findings if f["leakage_risk"] == "medium"]

        return {
            "findings": findings,
            "summary": {
                "total_columns_checked": len(findings),
                "high_leakage_risk": len(high),
                "medium_leakage_risk": len(medium),
                "low_or_none_leakage_risk": len(findings) - len(high) - len(medium),
            },
            "global_protocol": {
                "rule": (
                    "Fit all imputation statistics on training data only. "
                    "Never use prediction/test data to compute medians, modes, "
                    "or model imputation parameters."
                ),
                "sklearn_pattern": (
                    "use Pipeline([('imputer', SimpleImputer()), ('model', ...)]); "
                    "pipeline.fit(X_train, y_train)"
                ),
                "indicator_pattern": (
                    "use MissingIndicator(features='missing-only') before imputation in pipeline"
                ),
            },
        }

    def _check_column(self, col: str, entry: dict) -> dict:
        strategy = entry.get("strategy", "")
        fit_on = entry.get("fit_on", "N/A")

        if strategy == "no_imputation_needed":
            return {
                "column": col,
                "strategy": strategy,
                "leakage_risk": "none",
                "leakage_note": "no_imputation_required",
                "recommendation": "no_action",
            }

        if strategy == "model_based_imputation_optional":
            return {
                "column": col,
                "strategy": strategy,
                "leakage_risk": "medium",
                "leakage_note": "model_based_imputation_requires_careful_pipeline_design",
                "recommendation": "use_sklearn_pipeline_fit_on_train_only",
            }

        if fit_on == "train_only":
            note = "statistics_fitted_on_train_only"
            risk = "low"
            rec = "fit_statistic_on_train_apply_to_predict"
        else:
            note = "imputation_fit_scope_unspecified"
            risk = "high"
            rec = "must_fit_on_train_only"

        return {
            "column": col,
            "strategy": strategy,
            "leakage_risk": risk,
            "leakage_note": note,
            "recommendation": rec,
        }
