"""Write JSON logs and markdown report; also provides in-memory markdown generation."""

from __future__ import annotations

import json
import os
from datetime import datetime


class ReportWriter:
    def __init__(self, output_dir: str = "."):
        self.logs_dir = os.path.join(output_dir, "logs")
        self.reports_dir = os.path.join(output_dir, "reports")

    def write_all(
        self,
        profile: dict,
        mechanism_audit: dict,
        structural_audit: dict,
        imputation_plan: dict,
        leakage_check: dict,
    ) -> None:
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)

        self._json("missingness_profile.json", profile)
        self._json("missingness_mechanism_audit.json", mechanism_audit)
        self._json("structural_missingness_audit.json", structural_audit)
        self._json("imputation_plan.json", imputation_plan)
        self._json("leakage_safe_imputation_check.json", leakage_check)

        md = self.generate_report_md(
            profile, mechanism_audit, structural_audit, imputation_plan, leakage_check
        )
        with open(os.path.join(self.reports_dir, "missing_data_report.md"), "w") as f:
            f.write(md)

    def generate_report_md(
        self,
        profile: dict,
        mechanism_audit: dict,
        structural_audit: dict,
        imputation_plan: dict,
        leakage_check: dict,
    ) -> str:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        summary = profile.get("summary", {})
        col_profiles = profile.get("columns", {})
        col_mechanisms = mechanism_audit.get("columns", {})
        plan_cols = imputation_plan.get("columns", {})
        plan_summary = imputation_plan.get("summary", {})
        protocol = leakage_check.get("global_protocol", {})
        struct_pairs = structural_audit.get("structural_pairs", [])

        lines = [
            "# Missing Data Report",
            "",
            f"*Generated: {ts}*",
            "",
            "## Overview",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total rows | {summary.get('total_rows', 'N/A')} |",
            f"| Total columns | {summary.get('total_columns', 'N/A')} |",
            f"| Columns with missing values | {summary.get('columns_with_any_missing', 0)} |",
            f"| Overall missing rate | {summary.get('overall_missing_rate', 0):.1%} |",
            "",
            "## Missingness Profile",
            "",
            "| Column | Missing Rate | Severity | Dtype | Mechanism Clue |",
            "|--------|-------------|----------|-------|----------------|",
        ]
        for col, prof in col_profiles.items():
            if prof["missing_rate"] == 0:
                continue
            mech = col_mechanisms.get(col, {}).get("mechanism_label", "—")
            lines.append(
                f"| `{col}` | {prof['missing_rate']:.1%} | {prof['severity']} "
                f"| {prof['dtype_category']} | {mech} |"
            )
        lines.append("")

        if struct_pairs:
            lines += [
                "## Structural Missingness",
                "",
                "Column pairs showing structural absence patterns:",
                "",
            ]
            for pair in struct_pairs:
                lines.append(
                    f"- `{pair['categorical_col']}` (categorical NA) → "
                    f"`{pair['numeric_col']}` ({pair['pattern']})"
                )
            lines.append("")

        lines += [
            "## Imputation Plan",
            "",
            "| Column | Strategy | Add Indicator | Fit On | Reason |",
            "|--------|----------|--------------|--------|--------|",
        ]
        for col, plan in plan_cols.items():
            if plan["strategy"] == "no_imputation_needed":
                continue
            ind = "Yes" if plan["add_missing_indicator"] else "No"
            lines.append(
                f"| `{col}` | `{plan['strategy']}` | {ind} "
                f"| {plan['fit_on']} | {plan['reason']} |"
            )
        lines.append("")

        sc = plan_summary.get("strategy_counts", {})
        if sc:
            lines += ["## Strategy Summary", ""]
            for strat, cnt in sorted(sc.items(), key=lambda x: -x[1]):
                lines.append(f"- `{strat}`: {cnt} column(s)")
            lines.append("")

        lines += [
            "## Leakage-Safe Imputation Protocol",
            "",
            f"> **Rule:** {protocol.get('rule', '')}",
            "",
            "**sklearn pattern:**",
            "```python",
            protocol.get("sklearn_pattern", ""),
            "```",
            "",
        ]

        return "\n".join(lines)

    def _json(self, filename: str, data: dict) -> None:
        with open(os.path.join(self.logs_dir, filename), "w") as f:
            json.dump(data, f, indent=2, default=str)
