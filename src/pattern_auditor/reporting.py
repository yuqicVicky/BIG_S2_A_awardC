"""Write JSON logs and a Markdown narrative report."""

from __future__ import annotations

import json
import os
import datetime
from typing import Any


_PATTERN_NAMES = {
    "within_period": "Within-Period Latest-Available",
    "within_period_group": "Within-Period + Group",
    "time_based_split": "Strict Time Split (Forecasting)",
    "group_based_split": "Group Split (Unseen Groups)",
    "group_time_split": "Group + Time Split",
    "iid_random": "i.i.d. Random",
    "unknown": "Unknown",
}


class ReportWriter:
    def __init__(self, output_dir: str = "outputs/"):
        self.output_dir = output_dir
        self.logs_dir = os.path.join(output_dir, "logs")
        self.reports_dir = os.path.join(output_dir, "reports")

    def write_all(
        self,
        *,
        schema_report: dict,
        feature_availability_audit: dict,
        train_prediction_pattern: dict,
        leakage_report: dict,
        validation_recommendation: dict,
    ):
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)

        self._write_json("schema_audit.json", schema_report)
        self._write_json("feature_availability_audit.json", feature_availability_audit)
        self._write_json("train_prediction_pattern.json", train_prediction_pattern)
        self._write_json("leakage_audit.json", leakage_report)
        self._write_json("validation_recommendation.json", validation_recommendation)

        md = self._build_markdown(
            schema_report=schema_report,
            feature_availability_audit=feature_availability_audit,
            train_prediction_pattern=train_prediction_pattern,
            leakage_report=leakage_report,
            validation_recommendation=validation_recommendation,
        )
        md_path = os.path.join(self.reports_dir, "pattern_audit.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md)

    def _write_json(self, filename: str, data: Any):
        path = os.path.join(self.logs_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=_json_default)

    def _build_markdown(
        self,
        *,
        schema_report: dict,
        feature_availability_audit: dict,
        train_prediction_pattern: dict,
        leakage_report: dict,
        validation_recommendation: dict,
    ) -> str:
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = [
            "# Train-Test Pattern Audit",
            "",
            f"_Generated: {ts}_",
            "",
            "---",
            "",
        ]

        # --- Section 1: Detected pattern ---
        pattern = train_prediction_pattern.get("pattern", "unknown")
        pattern_label = _PATTERN_NAMES.get(pattern, pattern)
        evidence = train_prediction_pattern.get("evidence", [])

        t_schema = schema_report.get("train", {})
        p_schema = schema_report.get("predict", {})

        lines += [
            "## 1. Detected Train/Prediction Pattern",
            "",
            f"**Pattern: `{pattern_label}`**",
            "",
        ]
        if evidence:
            lines.append("Evidence from train/prediction comparison:")
            for ev in evidence:
                lines.append(f"- {ev}")
        else:
            lines.append("- No strong structural signal detected; assuming i.i.d. split.")

        lines += [
            "",
            f"**Dataset shape:** train {t_schema.get('row_count', '?')} rows × "
            f"{t_schema.get('col_count', '?')} cols",
        ]
        if p_schema:
            lines[-1] += (
                f" | predict {p_schema.get('row_count', '?')} rows × "
                f"{p_schema.get('col_count', '?')} cols"
            )

        schema_mismatch = train_prediction_pattern.get("schema_mismatch", {})
        train_only_cols = schema_mismatch.get("train_only_columns", [])
        predict_only_cols = schema_mismatch.get("predict_only_columns", [])
        if train_only_cols:
            lines.append(f"\nTrain-only columns (absent from predict): `{'`, `'.join(train_only_cols)}`")
        if predict_only_cols:
            lines.append(f"Predict-only columns (absent from train): `{'`, `'.join(predict_only_cols)}`")

        shift_info = train_prediction_pattern.get("numeric_distribution_shift", {})
        n_shift = shift_info.get("shift_detected_count", 0)
        n_total = shift_info.get("total_columns_tested", 0)
        if n_total:
            lines.append(f"\nNumeric distribution shift: **{n_shift}/{n_total}** columns shifted (KS p < 0.05)")
        lines.append("")

        # --- Section 2: Feature availability ---
        fa = feature_availability_audit
        fa_summary = fa.get("summary", {})
        columns_to_use = fa.get("columns_to_use", [])
        columns_to_exclude = fa.get("columns_to_exclude", [])
        columns_to_verify = fa.get("columns_to_verify", [])

        lines += [
            "## 2. Feature Availability at Prediction Time",
            "",
            "| Category | Count |",
            "|----------|-------|",
            f"| Available at prediction | {fa_summary.get('available_at_prediction', 0)} |",
            f"| Train-only (must exclude) | {fa_summary.get('train_only', 0)} |",
            f"| High leakage risk (exclude) | {fa_summary.get('high_leakage_risk', 0)} |",
            f"| Needs verification | {len(columns_to_verify)} |",
            "",
        ]

        if columns_to_exclude:
            lines.append("**Columns to exclude (unavailable or high leakage risk):**")
            fa_cols = fa.get("columns", {})
            for col in columns_to_exclude[:20]:
                info = fa_cols.get(col, {})
                ftype = info.get("inferred_type", "?")
                risk = info.get("leakage_type") or ftype
                lines.append(f"- `{col}` [{risk}]")
            lines.append("")

        if columns_to_verify:
            lines.append("**Columns to verify before use:**")
            fa_cols = fa.get("columns", {})
            for col in columns_to_verify[:10]:
                info = fa_cols.get(col, {})
                risk = info.get("leakage_type", "?")
                lines.append(f"- `{col}` [{risk}]")
            lines.append("")

        # --- Section 3: Why generic validation is risky ---
        why_risky = validation_recommendation.get("why_generic_validation_is_risky", "")
        lines += [
            "## 3. Why Generic Validation Is Risky Here",
            "",
        ]
        if why_risky:
            lines.append(why_risky)
        else:
            lines.append("Standard random-split validation may be appropriate for this data structure.")

        avoid = validation_recommendation.get("recommendation", {}).get("strategies_to_avoid", [])
        if avoid:
            lines += [
                "",
                f"**Do not use:** {', '.join(f'`{s}`' for s in avoid)}",
            ]
        lines.append("")

        # --- Section 4: Validation recommendation ---
        rec = validation_recommendation.get("recommendation", {})
        confidence = rec.get("confidence", "?")
        strategy = rec.get("strategy", "?")
        alts = validation_recommendation.get("alternatives", [])

        lines += [
            "## 4. Recommended Validation Strategy",
            "",
            f"**Strategy: `{strategy}`** (Confidence: {confidence})",
            "",
            f"_{rec.get('reason', '')}_",
            "",
            f"**Fit rule:** {rec.get('fit_rule', '')}",
            "",
            f"**Validation rule:** {rec.get('validation_rule', '')}",
            "",
        ]
        if alts:
            lines.append("**Alternatives to consider:**")
            for a in alts:
                lines.append(f"- `{a['strategy']}`: {a['reason']}")
            lines.append("")

        # --- Section 5: Leakage audit detail ---
        summary = leakage_report.get("summary", {})
        high_risks = [r for r in leakage_report.get("risks", []) if r.get("severity") == "high"]
        med_risks = [r for r in leakage_report.get("risks", []) if r.get("severity") == "medium"]

        lines += [
            "## 5. Leakage Audit",
            "",
            "| Severity | Count |",
            "|----------|-------|",
            f"| High     | {summary.get('high', 0)} |",
            f"| Medium   | {summary.get('medium', 0)} |",
            f"| Low      | {summary.get('low', 0)} |",
            "",
        ]
        if high_risks:
            lines.append("**High-risk columns — exclude from features:**")
            for r in high_risks:
                lines.append(f"- `{r['column']}` ({r['risk_type']}): {r['reason']}")
            lines.append("")
        if med_risks:
            lines.append("**Medium-risk columns — verify before using:**")
            for r in med_risks[:10]:
                lines.append(f"- `{r['column']}` ({r['risk_type']}): {r['recommendation']}")
            lines.append("")

        return "\n".join(lines)


def _json_default(obj):
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    if hasattr(obj, "item"):
        return obj.item()
    return str(obj)
