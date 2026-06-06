"""
Train-Test Pattern Auditor
==========================
A lightweight pre-modeling skill that helps autonomous data-science agents
avoid misleading validation and leakage before any model is trained.

Main entry point:
    from pattern_auditor import PatternAuditor
    auditor = PatternAuditor(train_df, predict_df, target_col="target")
    results = auditor.run(output_dir="outputs/")
"""

from .schema import SchemaInspector
from .feature_types import FeatureTypeInferrer
from .distribution_shift import DistributionShiftDetector
from .leakage import LeakageAuditor
from .validation import ValidationRecommender
from .visualization import Visualizer
from .reporting import ReportWriter

__version__ = "0.3.0"
__all__ = [
    "PatternAuditor",
    "SchemaInspector",
    "FeatureTypeInferrer",
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
        self.leakage_report: dict = {}
        self.feature_availability_audit: dict = {}
        self.validation_recommendation: dict = {}

    def run(self, output_dir: str = "outputs/") -> dict:
        import os

        os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "reports"), exist_ok=True)

        # 1. Schema audit
        schema = SchemaInspector(self.train_df, self.predict_df)
        self.schema_report = schema.inspect()

        # 2. Feature type inference (drives all downstream steps)
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
        distribution_shift = dsd.detect()
        self.train_prediction_pattern = _build_train_prediction_pattern(
            self.schema_report, distribution_shift
        )

        # 4. Leakage audit
        la = LeakageAuditor(
            self.train_df,
            self.predict_df,
            self.feature_type_report,
            target_col=self.target_col,
        )
        self.leakage_report = la.audit()

        # 5. Feature availability audit (answers: available vs unavailable at prediction time)
        self.feature_availability_audit = _build_feature_availability_audit(
            self.feature_type_report,
            self.leakage_report,
            self.schema_report,
            target_col=self.target_col,
        )

        # 6. Validation recommendation
        vr = ValidationRecommender(
            self.train_df,
            self.predict_df,
            self.feature_type_report,
            distribution_shift,
            target_col=self.target_col,
        )
        self.validation_recommendation = vr.recommend()

        # 7. Visualizations (3 figures)
        viz = Visualizer(
            self.train_df,
            self.predict_df,
            self.feature_type_report,
            distribution_shift,
            self.leakage_report,
        )
        viz.generate_all(figures_dir=os.path.join(output_dir, "figures"))

        # 8. Write reports
        rw = ReportWriter(output_dir)
        rw.write_all(
            schema_report=self.schema_report,
            feature_availability_audit=self.feature_availability_audit,
            train_prediction_pattern=self.train_prediction_pattern,
            leakage_report=self.leakage_report,
            validation_recommendation=self.validation_recommendation,
        )

        return {
            "schema": self.schema_report,
            "feature_availability": self.feature_availability_audit,
            "train_prediction_pattern": self.train_prediction_pattern,
            "leakage": self.leakage_report,
            "validation": self.validation_recommendation,
        }


# ---------------------------------------------------------------------------
# Train/prediction pattern builder
# ---------------------------------------------------------------------------

def _build_train_prediction_pattern(
    schema_report: dict,
    distribution_shift_report: dict,
) -> dict:
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
# Feature availability audit builder
# ---------------------------------------------------------------------------

def _build_feature_availability_audit(
    feature_type_report: dict,
    leakage_report: dict,
    schema_report: dict,
    *,
    target_col: str | None = None,
) -> dict:
    ft = feature_type_report.get("columns", {})

    leakage_by_col = {
        item["column"]: item
        for item in leakage_report.get("risks", [])
    }

    columns: dict = {}
    counts = {
        "total_train_columns": 0,
        "available_at_prediction": 0,
        "train_only": 0,
        "predict_only": 0,
        "high_leakage_risk": 0,
    }

    columns_to_use: list[str] = []
    columns_to_exclude: list[str] = []
    columns_to_verify: list[str] = []

    train_col_count = schema_report.get("train", {}).get("col_count", 0)
    counts["total_train_columns"] = train_col_count

    for col, info in ft.items():
        ftype = info.get("inferred_type", "unknown")

        if col == target_col:
            continue

        leakage_entry = leakage_by_col.get(col)
        leakage_severity = leakage_entry["severity"] if leakage_entry else "none"
        leakage_type = leakage_entry["risk_type"] if leakage_entry else None

        available = ftype not in ("train_only",)
        predict_only = ftype == "prediction_only"

        if ftype == "train_only":
            counts["train_only"] += 1
            action = "exclude"
            columns_to_exclude.append(col)
        elif predict_only:
            counts["predict_only"] += 1
            action = "exclude"
            columns_to_exclude.append(col)
        elif leakage_severity == "high":
            counts["high_leakage_risk"] += 1
            action = "exclude"
            columns_to_exclude.append(col)
        elif leakage_severity == "medium":
            action = "verify"
            columns_to_verify.append(col)
        else:
            counts["available_at_prediction"] += 1
            action = "use"
            columns_to_use.append(col)

        columns[col] = {
            "available_at_prediction": available and not predict_only,
            "inferred_type": ftype,
            "leakage_risk": leakage_severity,
            "leakage_type": leakage_type,
            "action": action,
        }

    return {
        "columns": columns,
        "summary": counts,
        "columns_to_use": columns_to_use,
        "columns_to_exclude": columns_to_exclude,
        "columns_to_verify": columns_to_verify,
    }
