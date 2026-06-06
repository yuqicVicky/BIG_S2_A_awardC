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
            train_prediction_pattern=train_prediction_pattern,
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
        train_prediction_pattern: dict,
        target_pattern_report: dict,
        leakage_report: dict,
        validation_recommendation: dict,
        feature_recommendation: dict,
    ) -> str:
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = [
            f"# Data Pattern & Validation Audit",
            f"",
            f"_Generated: {ts}_",
            f"",
            f"---",
            f"",
        ]

        # --- Section 1: What pattern was detected ---
        pattern = train_prediction_pattern.get("pattern", "unknown")
        pattern_label = _PATTERN_NAMES.get(pattern, pattern)
        evidence = train_prediction_pattern.get("evidence", [])

        lines += [
            "## 1. Detected Data Pattern",
            "",
            f"**Pattern: `{pattern_label}`**",
            "",
        ]

        if evidence:
            lines.append("Evidence from train/prediction comparison:")
            for ev in evidence:
                lines.append(f"- {ev}")
        else:
            lines.append("- No strong structural signal detected in the data.")

        schema_mismatch = train_prediction_pattern.get("schema_mismatch", {})
        train_only = schema_mismatch.get("train_only_columns", [])
        predict_only = schema_mismatch.get("predict_only_columns", [])
        if train_only:
            lines.append(f"- Train-only columns (absent from predict): `{'`, `'.join(train_only)}`")
        if predict_only:
            lines.append(f"- Predict-only columns (absent from train): `{'`, `'.join(predict_only)}`")

        t_schema = schema_report.get("train", {})
        p_schema = schema_report.get("predict", {})
        lines += [
            "",
            f"**Dataset shape:** train {t_schema.get('row_count', '?')} rows × "
            f"{t_schema.get('col_count', '?')} cols",
        ]
        if p_schema:
            lines[-1] += f" | predict {p_schema.get('row_count', '?')} rows × {p_schema.get('col_count', '?')} cols"
        lines.append("")

        # --- Section 2: Why generic validation is risky ---
        why_risky = validation_recommendation.get("why_generic_validation_is_risky", "")
        lines += [
            "## 2. Why Generic Validation Is Risky Here",
            "",
        ]
        if why_risky:
            lines.append(why_risky)
        else:
            lines.append("Standard validation may be appropriate for this data structure.")

        avoid = validation_recommendation.get("recommendation", {}).get("strategies_to_avoid", [])
        if avoid:
            lines += [
                "",
                f"**Do not use:** {', '.join(f'`{s}`' for s in avoid)}",
            ]
        lines.append("")

        # --- Section 3: Validation Recommendation ---
        rec = validation_recommendation.get("recommendation", {})
        confidence = rec.get("confidence", "?")
        strategy = rec.get("strategy", "?")
        alts = validation_recommendation.get("alternatives", [])

        lines += [
            "## 3. Recommended Validation Strategy",
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

        # --- Section 4: Leakage Audit ---
        summary = leakage_report.get("summary", {})
        high_risks = [r for r in leakage_report.get("risks", []) if r.get("severity") == "high"]
        med_risks = [r for r in leakage_report.get("risks", []) if r.get("severity") == "medium"]

        lines += [
            "## 4. Leakage Audit",
            "",
            f"| Severity | Count |",
            f"|----------|-------|",
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

        # --- Section 5: Top Target Patterns ---
        top_corrs = [
            c for c in target_pattern_report.get("numeric_correlations", [])
            if c.get("pearson_r") is not None
        ][:8]
        dt_patterns = target_pattern_report.get("datetime_target_patterns", [])
        cat_patterns = target_pattern_report.get("categorical_target_means", [])[:3]
        group_patterns = target_pattern_report.get("group_target_patterns", [])

        lines += [
            "## 5. Target Pattern Highlights",
            "",
        ]

        if top_corrs:
            lines += [
                "**Numeric feature correlations with target:**",
                "",
                "| Column | Pearson r | Signal |",
                "|--------|-----------|--------|",
            ]
            for c in top_corrs:
                r = c["pearson_r"]
                signal = "strong" if abs(r) > 0.5 else "moderate" if abs(r) > 0.2 else "weak"
                lines.append(f"| `{c['column']}` | {r:+.4f} | {signal} |")
            lines.append("")

        if dt_patterns:
            lines.append("**Datetime feature target patterns detected:**")
            for dp in dt_patterns:
                comp_keys = [k for k in dp.get("by_component", {}).keys() if "_x_" not in k]
                inter_keys = [k for k in dp.get("by_component", {}).keys() if "_x_" in k]
                lines.append(f"- `{dp['column']}`: components={comp_keys}, interactions={inter_keys}")
            lines.append("")

        if cat_patterns:
            lines.append("**Categorical features with target variation:**")
            for cp in cat_patterns:
                n = cp.get("n_categories", "?")
                lines.append(f"- `{cp['column']}` ({n} categories)")
            lines.append("")

        if group_patterns:
            lines.append("**Group/entity features with target variation:**")
            for gp in group_patterns:
                n = gp.get("n_groups", "?")
                lines.append(f"- `{gp['column']}` ({n} groups)")
            lines.append("")

        # --- Section 6: Feature Recommendations ---
        fe = feature_recommendation
        lines += [
            "## 6. Feature Recommendations",
            "",
        ]

        features_to_add = fe.get("features_to_add", [])
        if features_to_add:
            lines += [
                "### Features to Add",
                "",
                "| Feature | Source | Type | Reason |",
                "|---------|--------|------|--------|",
            ]
            for f in features_to_add[:20]:
                lines.append(f"| `{f['feature']}` | `{f.get('source_col', '')}` | {f.get('type', '')} | {f['reason']} |")
            lines.append("")

        features_to_exclude = fe.get("features_to_exclude", [])
        if features_to_exclude:
            lines += [
                "### Features to Exclude",
                "",
            ]
            for f in features_to_exclude[:20]:
                lines.append(f"- `{f['column']}`: {f['reason']}")
            lines.append("")

        interactions = fe.get("interactions_to_try", [])
        if interactions:
            lines += [
                "### Interactions to Try",
                "",
            ]
            for i in interactions:
                lines.append(f"- **{i['interaction']}** ({i['type']}): {i['reason']}")
            lines.append("")

        transforms = fe.get("target_transforms_to_try", [])
        if transforms:
            lines += [
                "### Target Transforms to Try",
                "",
            ]
            for t in transforms:
                lines.append(f"- `{t['transform']}`: {t['reason']}")
            lines.append("")

        fe_warnings = fe.get("warnings", [])
        if fe_warnings:
            lines += [
                "### Warnings",
                "",
            ]
            for w in fe_warnings:
                cols = ", ".join(f"`{c}`" for c in w.get("columns", []))
                lines.append(f"- **{w['type']}** [{cols}]: {w['message']}")
            lines.append("")

        # --- Section 7: Feature Type Summary ---
        type_summary = feature_type_report.get("type_summary", {})
        if type_summary:
            lines += [
                "## 7. Feature Type Summary",
                "",
                "| Type | Count |",
                "|------|-------|",
            ]
            for t_name, n_of_type in sorted(type_summary.items(), key=lambda x: -x[1]):
                lines.append(f"| `{t_name}` | {n_of_type} |")
            lines.append("")

        return "\n".join(lines)


def _json_default(obj):
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    if hasattr(obj, "item"):
        return obj.item()
    return str(obj)
