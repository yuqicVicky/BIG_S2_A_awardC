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

__version__ = "0.1.0"
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

        # Results populated by run()
        self.schema_report: dict = {}
        self.feature_type_report: dict = {}
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

        # 1. Schema
        schema = SchemaInspector(self.train_df, self.predict_df)
        self.schema_report = schema.inspect()

        # 2. Feature types
        fti = FeatureTypeInferrer(
            self.train_df,
            self.predict_df,
            target_col=self.target_col,
            row_id_col=self.row_id_col,
            datetime_col=self.datetime_col,
            group_col=self.group_col,
        )
        self.feature_type_report = fti.infer()

        # 3. Distribution shift / train-prediction comparison
        dsd = DistributionShiftDetector(
            self.train_df, self.predict_df, self.feature_type_report
        )
        self.distribution_shift_report = dsd.detect()

        # 4. Target patterns
        if self.target_col and self.target_col in self.train_df.columns:
            tpa = TargetPatternAnalyser(
                self.train_df,
                self.target_col,
                self.feature_type_report,
            )
            self.target_pattern_report = tpa.analyse()
        else:
            self.target_pattern_report = {"warning": "No target column provided."}

        # 5. Leakage
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
            distribution_shift_report=self.distribution_shift_report,
            target_pattern_report=self.target_pattern_report,
            leakage_report=self.leakage_report,
            validation_recommendation=self.validation_recommendation,
            feature_recommendation=self.feature_recommendation,
            train_prediction_pattern=self.distribution_shift_report,
        )

        return {
            "schema": self.schema_report,
            "feature_types": self.feature_type_report,
            "distribution_shift": self.distribution_shift_report,
            "target_patterns": self.target_pattern_report,
            "leakage": self.leakage_report,
            "validation": self.validation_recommendation,
            "feature_engineering": self.feature_recommendation,
        }


def _build_feature_recommendation(
    feature_type_report: dict,
    target_pattern_report: dict,
    leakage_report: dict,
) -> dict:
    recommendations = []
    leakage_cols = {
        item["column"]
        for item in leakage_report.get("risks", [])
        if item.get("severity") == "high"
    }

    ft = feature_type_report.get("columns", {})
    for col, info in ft.items():
        ftype = info.get("inferred_type", "")
        if col in leakage_cols:
            recommendations.append(
                {"column": col, "action": "DROP", "reason": "High leakage risk"}
            )
        elif ftype == "datetime_like":
            recommendations.append(
                {
                    "column": col,
                    "action": "EXTRACT_DATETIME_FEATURES",
                    "reason": "Extract hour, dayofweek, month, year, etc.",
                }
            )
        elif ftype == "categorical_high_cardinality":
            recommendations.append(
                {
                    "column": col,
                    "action": "TARGET_ENCODE_OR_EMBED",
                    "reason": "High-cardinality — consider target encoding or embedding",
                }
            )
        elif ftype == "text_like":
            recommendations.append(
                {
                    "column": col,
                    "action": "EXTRACT_TEXT_FEATURES",
                    "reason": "Extract length, word count, or embeddings",
                }
            )
        elif ftype in ("constant", "near_constant"):
            recommendations.append(
                {"column": col, "action": "DROP", "reason": "Near-constant — no signal"}
            )

    high_signal = []
    for item in target_pattern_report.get("numeric_correlations", []):
        if abs(item.get("pearson_r", 0)) > 0.3:
            high_signal.append(
                {
                    "column": item["column"],
                    "action": "KEEP_HIGH_SIGNAL",
                    "reason": f"Pearson r={item['pearson_r']:.3f}",
                }
            )

    return {"per_column": recommendations, "high_signal_numeric": high_signal}
