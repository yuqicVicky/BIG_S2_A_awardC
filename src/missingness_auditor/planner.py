"""
Column-specific imputation strategy planning with evidence-based decisions.

Strategy catalogue is designed for ML feature-engineering pipelines (single imputation
with missing indicator). For downstream statistical inference, upgrade to full
Multiple Imputation by Chained Equations (MICE):
  - Numeric continuous → PMM (predictive mean matching, mice default)
  - Binary categorical → logreg
  - Nominal categorical → polyreg
  - Ordinal categorical → polr

Mean imputation is avoided: it underestimates variance and distorts correlations
(van Buuren FIMD Ch1, Table 1.1). Mode imputation is avoided for high-cardinality
columns: creates spurious repeated values.

Collins et al. (2001) via van Buuren FIMD Ch5: when miss_rate < 25% and max
predictor correlation < 0.4, MAR assumption is robust for regression estimates.
Above those thresholds, mechanism assumptions matter — flag `mi_upgrade_recommended`.
"""

from __future__ import annotations

from collections import Counter

_HIGH_MISSING_THRESHOLD = 0.80
_MODERATE_MISSING_THRESHOLD = 0.10

# Collins et al. (2001) thresholds for MI upgrade recommendation
_MI_UPGRADE_MISS_RATE = 0.25
_MI_UPGRADE_CORR = 0.40


class ImputationPlanner:
    """
    Recommends a per-column imputation strategy based on:
    - missingness profile (rate, dtype, cardinality)
    - mechanism clue (group-dependent / MAR-like / target-associated / MCAR)
    - structural patterns (categorical NA → numeric zero/absent)

    Strategy catalogue
    ------------------
    no_imputation_needed                 : column has no missing values
    numeric_median                       : fill with training-set median
    numeric_median_plus_indicator        : fill with median + binary missing indicator
    groupwise_numeric_median_plus_indicator : fill with per-group median + indicator;
                                           fitted on train only (leakage-safe)
    categorical_missing_token            : fill with literal "MISSING" token
    categorical_missing_token_plus_indicator : fill with "MISSING" + indicator;
                                           safe for high-cardinality text columns
    structural_none_token_plus_indicator : fill categorical with "NONE" + indicator;
                                           used when categorical NA encodes absence
    structural_zero_plus_indicator       : fill numeric with 0 + indicator;
                                           used only when zero-companion evidence exists
    drop_column                          : >80% missing, too sparse to be useful
    """

    def __init__(
        self,
        profile: dict,
        mechanism_audit: dict,
        structural_audit: dict,
        domain_tags: dict[str, str] | None = None,
        llm_client=None,
    ):
        self.profile = profile
        self.mechanism_audit = mechanism_audit
        self.structural_audit = structural_audit
        # domain_tags: {col_name: domain_tag} e.g. {"temp_avg_f": "weather_metric"}
        # Produced by SKILL.md Step 1.5 semantic analysis; optional.
        self.domain_tags: dict[str, str] = domain_tags or {}
        self.llm_client = llm_client

    def plan(self) -> dict:
        col_profiles = self.profile.get("columns", {})
        col_mechanisms = self.mechanism_audit.get("columns", {})
        col_structural = self.structural_audit.get("column_flags", {})
        structural_pairs = self.structural_audit.get("structural_pairs", [])

        # Build structural companion map: column → list of paired columns
        structural_companions: dict[str, list[str]] = {}
        for pair in structural_pairs:
            cat_c = pair["categorical_col"]
            num_c = pair["numeric_col"]
            structural_companions.setdefault(cat_c, []).append(num_c)
            structural_companions.setdefault(num_c, []).append(cat_c)

        plan_result: dict[str, dict] = {}
        for col, prof in col_profiles.items():
            if prof.get("is_target"):
                plan_result[col] = {
                    "strategy": "no_imputation_needed",
                    "add_missing_indicator": False,
                    "reason": "target_column_excluded_from_imputation",
                    "fit_on": "N/A",
                    "missing_rate": prof["missing_rate"],
                    "dtype": prof["dtype"],
                    "cardinality": prof.get("cardinality"),
                    "mechanism_label": "N/A",
                    "target_association_evidence": None,
                    "covariate_association_evidence": [],
                    "group_dependency_evidence": None,
                    "structural_evidence": None,
                    "safety_warning": None,
                }
                continue

            miss_rate = prof["missing_rate"]
            dtype_cat = prof["dtype_category"]
            is_high_cardinality = prof.get("is_high_cardinality", False)

            mech_info = col_mechanisms.get(col, {})
            mech = mech_info.get("mechanism_label", "MCAR-compatible")
            target_signal = mech_info.get("target_signal", False)
            group_dep = mech_info.get("group_dependency_evidence") or {}

            is_structural = col_structural.get(col, {}).get("is_structural", False)
            structural_pattern = col_structural.get(col, {}).get("pattern", None)

            # Find top correlated categorical feature for group imputation
            top_cat_feature: str | None = None
            if group_dep.get("detected"):
                top_cat_feature = group_dep.get("top_categorical_feature")
            if not top_cat_feature:
                for feat_info in mech_info.get("categorical_correlated_features", []):
                    top_cat_feature = feat_info["feature"]
                    break

            strategy, add_indicator, reason, group_col, safety_warning = self._decide(
                col=col,
                miss_rate=miss_rate,
                dtype_cat=dtype_cat,
                mech=mech,
                target_signal=target_signal,
                is_structural=is_structural,
                structural_pattern=structural_pattern,
                is_high_cardinality=is_high_cardinality,
                top_cat_feature=top_cat_feature,
                domain_tag=self.domain_tags.get(col),
            )

            # Effective mechanism label
            if is_structural:
                effective_label = "structural absence concern"
            elif is_high_cardinality and dtype_cat == "categorical":
                effective_label = "high-cardinality text/category missingness"
            else:
                effective_label = mech

            structural_evidence: dict | None = None
            if is_structural:
                companions = structural_companions.get(col, [])
                structural_evidence = {
                    "detected": True,
                    "pattern": structural_pattern,
                    "companion_columns": companions,
                }

            mi_upgrade_recommended = self._should_upgrade_to_mi(
                miss_rate=miss_rate,
                mech=mech,
                target_signal=target_signal,
                cov_evidence=mech_info.get("covariate_association_evidence", []),
            )

            entry: dict = {
                "strategy": strategy,
                "add_missing_indicator": add_indicator,
                "reason": reason,
                "fit_on": "train_only" if strategy != "no_imputation_needed" else "N/A",
                "missing_rate": miss_rate,
                "dtype": prof["dtype"],
                "cardinality": prof.get("cardinality"),
                "mechanism_label": effective_label,
                "target_association_evidence": mech_info.get("target_association_evidence"),
                "covariate_association_evidence": mech_info.get("covariate_association_evidence", []),
                "group_dependency_evidence": mech_info.get("group_dependency_evidence"),
                "structural_evidence": structural_evidence,
                "safety_warning": safety_warning,
                "mi_upgrade_recommended": mi_upgrade_recommended,
            }
            if group_col:
                entry["group_col"] = group_col
            plan_result[col] = entry

        if self.llm_client and plan_result:
            self._enrich_with_llm(plan_result)

        return {
            "columns": plan_result,
            "summary": self._summarize(plan_result),
            "leakage_protocol": (
                "All imputation statistics (median, mode, group medians, model parameters) "
                "must be fitted on training data only and applied to prediction data."
            ),
        }

    def _enrich_with_llm(self, plan_result: dict) -> None:
        """Add human-readable reason and alternative strategy suggestions per column."""
        from ._llm import call_llm_json

        col_lines = []
        for col, entry in plan_result.items():
            if entry["strategy"] == "no_imputation_needed":
                continue
            col_lines.append(
                f"- {col}: strategy={entry['strategy']}, "
                f"mechanism={entry.get('mechanism_label', '?')}, "
                f"missing_rate={entry.get('missing_rate', 0):.1%}, "
                f"add_indicator={entry['add_missing_indicator']}, "
                f"internal_reason={entry['reason']}"
            )

        if not col_lines:
            return

        prompt = (
            "You are explaining an imputation plan to a data scientist. "
            "For each column, write:\n"
            "1. readable_reason: 1-2 sentences explaining in plain English WHY this strategy "
            "   was chosen (mention the mechanism, missing rate, and what it means for the model).\n"
            "2. alternatives: list of 1-2 alternative strategies with a brief trade-off note each.\n\n"
            "Columns:\n" + "\n".join(col_lines) + "\n\n"
            'Return JSON: {"col_name": {"readable_reason": "...", '
            '"alternatives": ["alt1: trade-off", "alt2: trade-off"]}, ...}'
        )

        enrichments = call_llm_json(self.llm_client, prompt, max_tokens=1400)
        if not isinstance(enrichments, dict):
            return
        for col, enrich in enrichments.items():
            if col in plan_result and isinstance(enrich, dict):
                plan_result[col]["llm_readable_reason"] = enrich.get("readable_reason", "")
                plan_result[col]["llm_alternatives"] = enrich.get("alternatives", [])

    # Domain tags that indicate temporally-ordered continuous measurements.
    # The planner recommends time-series forward/backward fill for these when
    # group-dependent imputation is not available and miss_rate is moderate.
    _TEMPORAL_DOMAIN_TAGS: frozenset = frozenset({
        "weather_metric", "sensor_reading", "time_series_metric",
        "physiological_measurement", "environmental_metric",
        "financial_time_series", "temporal_measurement",
    })

    def _decide(
        self,
        col: str,
        miss_rate: float,
        dtype_cat: str,
        mech: str,
        target_signal: bool,
        is_structural: bool,
        structural_pattern: str | None,
        is_high_cardinality: bool,
        top_cat_feature: str | None,
        domain_tag: str | None = None,
    ) -> tuple[str, bool, str, str | None, str | None]:
        if miss_rate == 0.0:
            return "no_imputation_needed", False, "no_missing_values", None, None

        if miss_rate > _HIGH_MISSING_THRESHOLD:
            return "drop_column", False, "over_80_percent_missing", None, "column_too_sparse_consider_feature_engineering"

        if is_structural:
            if dtype_cat == "numeric":
                # Pattern 2 (zero-valued absence): numeric is near-zero when the
                # categorical is NA — zero-fill is semantically correct here.
                # Pattern 1 (co_missing_structural): both columns are absent together;
                # the numeric value is unknown (not zero), so fall through to normal
                # median / group-median logic below.
                if structural_pattern != "co_missing_structural":
                    return (
                        "structural_zero_plus_indicator",
                        True,
                        "structural_absence_companion_is_zero",
                        None,
                        "verify_true_zero_before_applying_structural_zero",
                    )
                # else: co-missing — fall through to standard numeric imputation
            else:
                return (
                    "structural_none_token_plus_indicator",
                    True,
                    "structural_absence_categorical_na_encodes_absence",
                    None,
                    None,
                )

        if dtype_cat == "numeric":
            if mech == "group-dependent missingness" and top_cat_feature:
                return (
                    "groupwise_numeric_median_plus_indicator",
                    True,
                    f"group_dependent_missingness_by_feature={top_cat_feature!r}",
                    top_cat_feature,
                    None,
                )

            # Domain-aware upgrade: temporally-ordered continuous measurements
            # (weather, sensor, physiological, financial time-series) benefit from
            # forward/backward fill over median because adjacent timestamps carry
            # more signal than the global central tendency.
            is_temporal_domain = domain_tag in self._TEMPORAL_DOMAIN_TAGS
            if is_temporal_domain:
                return (
                    "time_series_ffill_bfill_plus_indicator",
                    True,
                    f"domain={domain_tag!r}_temporal_measurement_ffill_preferred_over_median",
                    None,
                    "sort_by_time_column_before_applying_ffill_bfill",
                )

            mar_or_target = mech in ("MAR-like evidence", "target-associated missingness")
            if mar_or_target or target_signal or miss_rate >= _MODERATE_MISSING_THRESHOLD:
                return (
                    "numeric_median_plus_indicator",
                    True,
                    f"mechanism={mech!r}_or_moderate_missing_rate",
                    None,
                    None,
                )
            return "numeric_median", False, "mcar_compatible_low_missing_rate", None, None

        # Categorical
        if is_high_cardinality:
            return (
                "categorical_missing_token_plus_indicator",
                True,
                "high_cardinality_text_column_mode_imputation_unsafe",
                None,
                "high_cardinality_column_mode_creates_spurious_repeated_values",
            )
        if mech == "group-dependent missingness":
            return (
                "categorical_missing_token_plus_indicator",
                True,
                "group_dependent_missingness_categorical",
                None,
                None,
            )
        if miss_rate >= _MODERATE_MISSING_THRESHOLD:
            return (
                "categorical_missing_token_plus_indicator",
                True,
                "moderate_or_high_missing_categorical",
                None,
                None,
            )
        return "categorical_missing_token", False, "low_missing_categorical", None, None

    def _should_upgrade_to_mi(
        self,
        miss_rate: float,
        mech: str,
        target_signal: bool,
        cov_evidence: list,
    ) -> bool:
        """
        Returns True when simple median/token imputation may be insufficient and
        upgrading to full MICE is recommended.

        Criteria (Collins et al. 2001 via van Buuren FIMD Ch5):
        - missing rate > 25%
        - any covariate correlation > 0.4 with the missingness indicator
        - target_signal is present (missingness correlated with outcome)
        - mechanism is group-dependent or target-associated
        """
        if miss_rate >= _MI_UPGRADE_MISS_RATE:
            return True
        if target_signal:
            return True
        if mech in ("target-associated missingness", "group-dependent missingness"):
            return True
        max_corr = max((x.get("correlation", 0.0) for x in cov_evidence), default=0.0)
        if max_corr >= _MI_UPGRADE_CORR:
            return True
        return False

    def _summarize(self, plan: dict) -> dict:
        strategy_counts = Counter(v["strategy"] for v in plan.values())
        indicator_cols = [c for c, v in plan.items() if v["add_missing_indicator"]]
        mi_upgrade_cols = [c for c, v in plan.items() if v.get("mi_upgrade_recommended")]
        return {
            "strategy_counts": dict(strategy_counts),
            "columns_needing_missing_indicator": indicator_cols,
            "n_columns_to_impute": sum(
                1 for v in plan.values() if v["strategy"] != "no_imputation_needed"
            ),
            "columns_where_full_mi_recommended": mi_upgrade_cols,
        }
