"""
Data Pattern & Validation Auditor
==================================
A statistical agent skill that helps autonomous data-science agents understand
data patterns before modeling.

Main entry point:
    from pattern_auditor import PatternAuditor
    auditor = PatternAuditor(train_df, predict_df, target_col="target")
    report = auditor.run(output_dir="outputs/")
"""

from .schema import SchemaInspector
from .feature_types import FeatureTypeInferrer
from .datetime_patterns import DatetimePatternAnalyser
from .target_patterns import TargetPatternAnalyser
from .distribution_shift import DistributionShiftDetector
from .leakage import LeakageAuditor
from .validation import ValidationRecommender
from .visualization import Visualizer
from .reporting import ReportWriter

__version__ = "0.2.0"
__all__ = [
    "PatternAuditor",
    "SchemaInspector",
    "FeatureTypeInferrer",
    "DatetimePatternAnalyser",
    "TargetPatternAnalyser",
    "DistributionShiftDetector",
    "LeakageAuditor",
    "ValidationRecommender",
    "Visualizer",
    "ReportWriter",
]


class PatternAuditor:
    """Orchestrates all sub-auditors and writes outputs."""

    def __init__(
        self,
        train_df,
        predict_df=None,
        *,
        target_col: str | None = None,
        row_id_col: str | None = None,
        datetime_col: str | None = None,
        group_col: str | None = None,
    ):
        self.train_df = train_df
        self.predict_df = predict_df
        self.target_col = target_col
        self.row_id_col = row_id_col
        self.datetime_col = datetime_col
        self.group_col = group_col

        self.schema_report: dict = {}
        self.feature_type_report: dict = {}
        self.train_prediction_pattern: dict = {}
        self.distribution_shift_report: dict = {}
        self.target_pattern_report: dict = {}
        self.leakage_report: dict = {}
        self.validation_recommendation: dict = {}
        self.feature_recommendation: dict = {}

    def run(self, output_dir: str = "outputs/") -> dict:
        import os

        os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "reports"), exist_ok=True)

        # 1. Schema understanding
        schema = SchemaInspector(self.train_df, self.predict_df)
        self.schema_report = schema.inspect()

        # 2. Feature type inference
        fti = FeatureTypeInferrer(
            self.train_df,
            self.predict_df,
            target_col=self.target_col,
            row_id_col=self.row_id_col,
            datetime_col=self.datetime_col,
            group_col=self.group_col,
        )
        self.feature_type_report = fti.infer()

        # 3. Train/prediction pattern detection
        dsd = DistributionShiftDetector(
            self.train_df, self.predict_df, self.feature_type_report
        )
        self.distribution_shift_report = dsd.detect()
        self.train_prediction_pattern = _build_train_prediction_pattern(
            self.schema_report, self.distribution_shift_report
        )

        # 4. Target-aware pattern discovery
        if self.target_col and self.target_col in self.train_df.columns:
            tpa = TargetPatternAnalyser(
                self.train_df,
                self.target_col,
                self.feature_type_report,
            )
            self.target_pattern_report = tpa.analyse()
        else:
            self.target_pattern_report = {"warning": "No target column provided."}

        # 5. Leakage audit
        la = LeakageAuditor(
            self.train_df,
            self.predict_df,
            self.feature_type_report,
            target_col=self.target_col,
        )
        self.leakage_report = la.audit()

        # 6. Validation recommendation
        vr = ValidationRecommender(
            self.train_df,
            self.predict_df,
            self.feature_type_report,
            self.distribution_shift_report,
            target_col=self.target_col,
        )
        self.validation_recommendation = vr.recommend()

        # 7. Feature recommendation
        self.feature_recommendation = _build_feature_recommendation(
            self.feature_type_report,
            self.target_pattern_report,
            self.leakage_report,
            self.distribution_shift_report,
            self.train_prediction_pattern,
        )

        # 8. Visualizations
        viz = Visualizer(
            self.train_df,
            self.predict_df,
            self.feature_type_report,
            self.target_pattern_report,
            self.distribution_shift_report,
            self.leakage_report,
            target_col=self.target_col,
        )
        viz.generate_all(figures_dir=os.path.join(output_dir, "figures"))

        # 9. Write reports
        rw = ReportWriter(output_dir)
        rw.write_all(
            schema_report=self.schema_report,
            feature_type_report=self.feature_type_report,
            train_prediction_pattern=self.train_prediction_pattern,
            target_pattern_report=self.target_pattern_report,
            distribution_shift_report=self.distribution_shift_report,
            leakage_report=self.leakage_report,
            validation_recommendation=self.validation_recommendation,
            feature_recommendation=self.feature_recommendation,
        )

        return {
            "schema": self.schema_report,
            "feature_types": self.feature_type_report,
            "train_prediction_pattern": self.train_prediction_pattern,
            "distribution_shift": self.distribution_shift_report,
            "target_patterns": self.target_pattern_report,
            "leakage": self.leakage_report,
            "validation": self.validation_recommendation,
            "feature_engineering": self.feature_recommendation,
        }


# ---------------------------------------------------------------------------
# Train/prediction pattern builder
# ---------------------------------------------------------------------------

def _build_train_prediction_pattern(
    schema_report: dict,
    distribution_shift_report: dict,
) -> dict:
    """Produce a focused train/prediction comparison report."""
    cmp = schema_report.get("comparison", {})
    split = distribution_shift_report.get("split_pattern", {})
    numeric_shifts = distribution_shift_report.get("numeric_shift", [])
    cat_overlaps = distribution_shift_report.get("categorical_overlap", [])
    dt_coverage = distribution_shift_report.get("datetime_coverage", [])

    shifted_cols = [s["column"] for s in numeric_shifts if s.get("shift_detected")]
    category_mismatches = [
        c for c in cat_overlaps
        if c.get("predict_only_categories") or c.get("train_only_categories")
    ]

    return {
        "pattern": split.get("pattern", "unknown"),
        "evidence": split.get("evidence", []),
        "schema_mismatch": {
            "train_only_columns": cmp.get("train_only_columns", []),
            "predict_only_columns": cmp.get("predict_only_columns", []),
            "dtype_mismatches": cmp.get("dtype_mismatches", {}),
        },
        "datetime_coverage": dt_coverage,
        "category_mismatch_columns": [c["column"] for c in category_mismatches],
        "numeric_distribution_shift": {
            "total_columns_tested": len(numeric_shifts),
            "shift_detected_count": len(shifted_cols),
            "columns_with_shift": shifted_cols,
        },
    }


# ---------------------------------------------------------------------------
# Feature recommendation builder
# ---------------------------------------------------------------------------

def _build_feature_recommendation(
    feature_type_report: dict,
    target_pattern_report: dict,
    leakage_report: dict,
    distribution_shift_report: dict,
    train_prediction_pattern: dict,
) -> dict:
    ft = feature_type_report.get("columns", {})
    split_pattern = train_prediction_pattern.get("pattern", "iid_random")

    leakage_cols = {
        item["column"]
        for item in leakage_report.get("risks", [])
        if item.get("severity") == "high"
    }

    features_to_add: list[dict] = []
    features_to_exclude: list[dict] = []
    interactions_to_try: list[dict] = []
    target_transforms_to_try: list[dict] = []
    warnings: list[dict] = []

    # --- features_to_add: datetime-derived components ---
    datetime_cols = [c for c, i in ft.items() if i.get("inferred_type") == "datetime_like"]
    for col in datetime_cols:
        if col in leakage_cols:
            continue
        for comp, reason in [
            ("hour", "Hour of day captures intra-day demand cycles"),
            ("dayofweek", "Day of week captures weekly seasonality"),
            ("month", "Month captures annual seasonality"),
            ("is_weekend", "Binary weekday/weekend flag captures structural behavior change"),
            ("is_workingday", "Working-day flag isolates business-hours patterns"),
        ]:
            features_to_add.append({
                "feature": f"{col}__{comp}",
                "source_col": col,
                "type": "datetime_derived",
                "reason": reason,
            })

    # --- features_to_add: polynomial/log for high-signal numerics ---
    for item in target_pattern_report.get("numeric_correlations", []):
        r = item.get("pearson_r") or 0
        col = item["column"]
        if col in leakage_cols:
            continue
        if abs(r) > 0.3:
            features_to_add.append({
                "feature": f"{col}_sq",
                "source_col": col,
                "type": "polynomial",
                "reason": f"|r|={abs(r):.3f} — non-linear term may capture additional variance",
            })

    # --- features_to_exclude: leakage + useless ---
    for col, info in ft.items():
        ftype = info.get("inferred_type", "")
        if col in leakage_cols:
            features_to_exclude.append({
                "column": col,
                "reason": "High leakage risk — absent from prediction data or derived from target",
            })
        elif ftype in ("constant", "near_constant"):
            features_to_exclude.append({
                "column": col,
                "reason": f"{ftype.replace('_', ' ')} — carries no generalizable signal",
            })

    # --- interactions_to_try: pattern-driven ---
    group_cols = [c for c, i in ft.items() if i.get("inferred_type") == "group_entity_id"]

    if datetime_cols:
        interactions_to_try.append({
            "interaction": "hour × dayofweek",
            "type": "datetime_interaction",
            "reason": "Hour-of-day effect varies by day of week — key interaction for temporal models",
        })
        interactions_to_try.append({
            "interaction": "hour × is_workingday",
            "type": "datetime_interaction",
            "reason": "Hour patterns differ significantly between working days and weekends",
        })

    if datetime_cols and group_cols:
        interactions_to_try.append({
            "interaction": f"{datetime_cols[0]}__hour × {group_cols[0]}",
            "type": "datetime_x_group",
            "reason": "Temporal demand cycles typically differ by group/entity",
        })

    _is_temporal = (
        "within_period" in split_pattern
        or split_pattern in ("time_based_split", "group_time_split")
    )
    if _is_temporal and group_cols:
        interactions_to_try.append({
            "interaction": f"lag_target_by_{group_cols[0]}",
            "type": "lag",
            "reason": "Temporal pattern: recent target values per entity are typically the strongest predictor",
        })
        interactions_to_try.append({
            "interaction": f"rolling_mean_by_{group_cols[0]}",
            "type": "lag",
            "reason": "Rolling average per entity captures trend and level shift over time",
        })

    if "group" in split_pattern and group_cols:
        interactions_to_try.append({
            "interaction": f"group_target_encode_{group_cols[0]}",
            "type": "group_encoding",
            "reason": "Group split: target encoding with leave-one-group-out avoids group leakage",
        })

    # --- target_transforms_to_try ---
    target_stats = target_pattern_report.get("target_stats", {})
    if target_stats:
        min_val = target_stats.get("min") or 0
        mean_val = target_stats.get("mean") or 0
        std_val = target_stats.get("std") or 1
        if min_val > 0 and std_val > abs(mean_val) * 0.5:
            target_transforms_to_try.append({
                "transform": "log1p",
                "reason": "Target is strictly positive with high relative std — log1p stabilizes variance",
            })
        p25 = target_stats.get("25%") or 0
        p75 = target_stats.get("75%") or 0
        if p75 > 0 and (p75 / max(p25, 1e-9)) > 10:
            target_transforms_to_try.append({
                "transform": "sqrt",
                "reason": "IQR spans an order of magnitude — sqrt may reduce right skew",
            })

    # --- warnings ---
    shifted_cols = train_prediction_pattern.get("numeric_distribution_shift", {}).get("columns_with_shift", [])
    if shifted_cols:
        warnings.append({
            "type": "distribution_shift",
            "columns": shifted_cols[:10],
            "message": (
                "These features show significant distribution shift between train and predict. "
                "Models trained without accounting for this shift may degrade in production."
            ),
        })

    high_card_cols = [c for c, i in ft.items() if i.get("inferred_type") == "categorical_high_cardinality"]
    if high_card_cols:
        warnings.append({
            "type": "high_cardinality",
            "columns": high_card_cols[:10],
            "message": "Use target encoding with out-of-fold estimates or embeddings — naive one-hot will overfit.",
        })

    cat_mismatch = train_prediction_pattern.get("category_mismatch_columns", [])
    if cat_mismatch:
        warnings.append({
            "type": "unseen_categories",
            "columns": cat_mismatch[:10],
            "message": "Prediction set contains categories not seen in training — model will encounter OOV at inference.",
        })

    return {
        "features_to_add": features_to_add,
        "features_to_exclude": features_to_exclude,
        "interactions_to_try": interactions_to_try,
        "target_transforms_to_try": target_transforms_to_try,
        "warnings": warnings,
    }
