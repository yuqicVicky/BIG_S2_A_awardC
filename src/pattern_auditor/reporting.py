"""Write JSON logs and a Markdown summary report."""

from __future__ import annotations

import json
import os
import datetime
from typing import Any


class ReportWriter:
    def __init__(self, output_dir: str = "outputs/"):
        self.output_dir = output_dir
        self.logs_dir = os.path.join(output_dir, "logs")
        self.reports_dir = os.path.join(output_dir, "reports")

    def write_all(
        self,
        *,
        schema_report: dict,
        feature_type_report: dict,
        train_prediction_pattern: dict,
        target_pattern_report: dict,
        distribution_shift_report: dict,
        leakage_report: dict,
        validation_recommendation: dict,
        feature_recommendation: dict,
    ):
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)

        self._write_json("schema_report.json", schema_report)
        self._write_json("feature_type_report.json", feature_type_report)
        self._write_json("train_prediction_pattern.json", train_prediction_pattern)
        self._write_json("target_pattern_report.json", target_pattern_report)
        self._write_json("distribution_shift_report.json", distribution_shift_report)
        self._write_json("leakage_feature_audit.json", leakage_report)
        self._write_json("validation_recommendation.json", validation_recommendation)
        self._write_json("feature_recommendation.json", feature_recommendation)

        md = self._build_markdown(
            schema_report=schema_report,
            feature_type_report=feature_type_report,
            distribution_shift_report=distribution_shift_report,
            target_pattern_report=target_pattern_report,
            leakage_report=leakage_report,
            validation_recommendation=validation_recommendation,
            feature_recommendation=feature_recommendation,
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
        feature_type_report: dict,
        distribution_shift_report: dict,
        target_pattern_report: dict,
        leakage_report: dict,
        validation_recommendation: dict,
        feature_recommendation: dict,
    ) -> str:
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = [f"# Data Pattern & Validation Audit Report\n\n_Generated: {ts}_\n"]

        # --- Schema ---
        lines.append("## 1. Schema Summary\n")
        t = schema_report.get("train", {})
        lines.append(f"- **Train rows:** {t.get('row_count', 'N/A')}")
        lines.append(f"- **Train columns:** {t.get('col_count', 'N/A')}")
        if "predict" in schema_report:
            p = schema_report["predict"]
            lines.append(f"- **Predict rows:** {p.get('row_count', 'N/A')}")
            lines.append(f"- **Predict columns:** {p.get('col_count', 'N/A')}")
        cmp = schema_report.get("comparison", {})
        if cmp.get("train_only_columns"):
            lines.append(f"- **Train-only columns:** {', '.join(cmp['train_only_columns'])}")
        if cmp.get("predict_only_columns"):
            lines.append(f"- **Predict-only columns:** {', '.join(cmp['predict_only_columns'])}")
        lines.append("")

        # --- Feature Types ---
        lines.append("## 2. Feature Type Summary\n")
        type_summary = feature_type_report.get("type_summary", {})
        if type_summary:
            lines.append("| Type | Count |")
            lines.append("|------|-------|")
            for t_name, n_of_type in sorted(type_summary.items(), key=lambda x: -x[1]):
                lines.append(f"| {t_name} | {n_of_type} |")
        lines.append("")

        # --- Split Pattern ---
        lines.append("## 3. Detected Split Pattern\n")
        sp = distribution_shift_report.get("split_pattern", {})
        lines.append(f"- **Pattern:** `{sp.get('pattern', 'unknown')}`")
        for ev in sp.get("evidence", []):
            lines.append(f"  - {ev}")
        lines.append("")

        # --- Distribution Shift ---
        lines.append("## 4. Distribution Shift\n")
        shifts = distribution_shift_report.get("numeric_shift", [])
        detected_shifts = [s for s in shifts if s.get("shift_detected")]
        if detected_shifts:
            lines.append(f"**{len(detected_shifts)} column(s) with detected shift:**\n")
            lines.append("| Column | KS Stat | p-value | Train Mean | Predict Mean |")
            lines.append("|--------|---------|---------|------------|--------------|")
            for s in detected_shifts[:20]:
                lines.append(
                    f"| {s['column']} | {s.get('ks_statistic','N/A')} "
                    f"| {s.get('ks_pvalue','N/A')} "
                    f"| {s.get('train_mean','N/A')} "
                    f"| {s.get('predict_mean','N/A')} |"
                )
        else:
            lines.append("No significant numeric distribution shifts detected.")
        lines.append("")

        # --- Target Patterns ---
        lines.append("## 5. Target Pattern Highlights\n")
        top_corrs = [
            c for c in target_pattern_report.get("numeric_correlations", [])
            if c.get("pearson_r") is not None
        ][:5]
        if top_corrs:
            lines.append("**Top numeric correlations with target:**\n")
            lines.append("| Column | Pearson r |")
            lines.append("|--------|-----------|")
            for c in top_corrs:
                lines.append(f"| {c['column']} | {c['pearson_r']:.4f} |")
        lines.append("")

        # --- Leakage ---
        lines.append("## 6. Leakage & Feature Availability Audit\n")
        summary = leakage_report.get("summary", {})
        lines.append(
            f"- **High risk:** {summary.get('high', 0)}  "
            f"**Medium risk:** {summary.get('medium', 0)}  "
            f"**Low risk:** {summary.get('low', 0)}"
        )
        high_risks = [r for r in leakage_report.get("risks", []) if r.get("severity") == "high"]
        if high_risks:
            lines.append("\n**High-risk columns:**\n")
            for r in high_risks:
                lines.append(f"- `{r['column']}` — {r['reason']}")
        lines.append("")

        # --- Validation Recommendation ---
        lines.append("## 7. Validation Recommendation\n")
        rec = validation_recommendation.get("recommendation", {})
        lines.append(f"**Recommended Strategy:** `{rec.get('strategy', 'N/A')}`\n")
        lines.append(f"- **Confidence:** {rec.get('confidence', 'N/A')}")
        lines.append(f"- **Reason:** {rec.get('reason', '')}")
        lines.append(f"- **Fit rule:** {rec.get('fit_rule', '')}")
        lines.append(f"- **Validation rule:** {rec.get('validation_rule', '')}")
        avoid = rec.get("avoid", [])
        if avoid:
            lines.append(f"- **Avoid:** {', '.join(avoid)}")
        alts = validation_recommendation.get("alternatives", [])
        if alts:
            lines.append("\n**Alternative strategies:**")
            for a in alts:
                lines.append(f"- `{a['strategy']}`: {a['reason']}")
        lines.append("")

        # --- Feature Engineering Recommendations ---
        lines.append("## 8. Feature Engineering Recommendations\n")
        per_col = feature_recommendation.get("per_column", [])
        high_signal = feature_recommendation.get("high_signal_numeric", [])
        if per_col:
            lines.append("| Column | Action | Reason |")
            lines.append("|--------|--------|--------|")
            for r in per_col[:30]:
                lines.append(f"| {r['column']} | {r['action']} | {r['reason']} |")
        if high_signal:
            lines.append("\n**High-signal numeric features (|r| > 0.3):**")
            for r in high_signal:
                lines.append(f"- `{r['column']}`: {r['reason']}")
        lines.append("")

        return "\n".join(lines)


def _json_default(obj):
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    if hasattr(obj, "item"):
        return obj.item()
    return str(obj)
