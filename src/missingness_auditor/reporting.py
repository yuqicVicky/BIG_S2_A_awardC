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
            (
                "This report diagnoses missingness patterns and recommends a leakage-safe "
                "imputation strategy for each column. Mechanism labels are statistical clues "
                "from observational data — they do not confirm causal mechanisms."
            ),
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total rows | {summary.get('total_rows', 'N/A')} |",
            f"| Total columns | {summary.get('total_columns', 'N/A')} |",
            f"| Columns with missing values | {summary.get('columns_with_any_missing', 0)} |",
            f"| Overall missing rate | {summary.get('overall_missing_rate', 0):.1%} |",
            "",
            "---",
            "",
            "## Missingness Profile",
            "",
            "| Column | Missing Rate | Severity | Dtype | Cardinality | Mechanism Clue |",
            "|--------|-------------|----------|-------|-------------|----------------|",
        ]
        for col, prof in col_profiles.items():
            if prof["missing_rate"] == 0:
                continue
            mech = col_mechanisms.get(col, {}).get("mechanism_label", "—")
            card = prof.get("cardinality", "—")
            card_str = str(card) if card is not None else "—"
            lines.append(
                f"| `{col}` | {prof['missing_rate']:.1%} | {prof['severity']} "
                f"| {prof['dtype_category']} | {card_str} | {mech} |"
            )
        lines.append("")

        # ── Mechanism clues section
        lines += [
            "---",
            "",
            "## Mechanism Clues",
            "",
            "> **Caution:** These labels are observational clues, not causal claims. "
            "MCAR, MAR, and MNAR cannot be confirmed from observational data alone. "
            "MAR is the recommended default assumption (van Buuren FIMD). "
            "Use these clues to inform — not dictate — imputation choices.",
            "",
        ]
        _mech_descriptions = {
            "MCAR-compatible": (
                "No significant correlation with other features or target. "
                "Median/token imputation is sufficient. MAR assumption likely robust "
                "when missing rate < 25% and max correlation < 0.4 (Collins et al., 2001)."
            ),
            "MAR-like evidence": (
                "Missingness correlates with at least one observed numeric feature. "
                "Add missing indicator. Include correlated features as predictors "
                "in any downstream imputation model."
            ),
            "group-dependent missingness": (
                "Missingness concentrated in specific categorical groups. "
                "Use groupwise imputation or missing token. This is NOT structural absence — "
                "data exists in principle but was not collected for certain groups."
            ),
            "target-associated missingness": (
                "Missingness correlates with the target variable. Missing indicator is important. "
                "Consider full MICE for inference tasks (including target in imputation model "
                "is correct and reduces bias toward zero)."
            ),
            "high-cardinality text/category missingness": (
                "High-cardinality column (>50 unique values or >20% unique ratio). "
                "Mode imputation would create spurious repeated values. Use missing token."
            ),
            "insufficient evidence": (
                "Too few missing rows for reliable analysis. Default to conservative strategy."
            ),
        }
        label_counts: dict[str, int] = {}
        for col, info in col_mechanisms.items():
            lbl = info.get("mechanism_label", "—")
            label_counts[lbl] = label_counts.get(lbl, 0) + 1

        for lbl, cnt in sorted(label_counts.items(), key=lambda x: -x[1]):
            desc = _mech_descriptions.get(lbl, "")
            lines.append(f"- **{lbl}** ({cnt} column(s)): {desc}")
        lines.append("")

        # ── Group-dependent missingness section
        group_dep_cols = [
            (col, info) for col, info in col_mechanisms.items()
            if info.get("mechanism_label") == "group-dependent missingness"
        ]
        if group_dep_cols:
            lines += [
                "---",
                "",
                "## Group-Dependent Missingness",
                "",
                (
                    "The following columns have missingness concentrated in specific "
                    "categorical groups. This is **not** structural absence — the data exists "
                    "for some groups but not others (e.g., a measurement collected only in "
                    "certain jurisdictions). Recommended: groupwise imputation or missing token."
                ),
                "",
            ]
            for col, info in group_dep_cols:
                gdep = info.get("group_dependency_evidence", {}) or {}
                top_feat = gdep.get("top_categorical_feature", "unknown")
                spread = gdep.get("max_missingness_spread", None)
                spread_str = f"{spread:.1%}" if spread is not None else "unknown"
                high_groups = gdep.get("high_missing_groups", {})
                lines.append(f"### `{col}`")
                lines.append(f"- Correlated group feature: `{top_feat}`")
                lines.append(f"- Max missingness spread across groups: {spread_str}")
                if high_groups:
                    groups_str = ", ".join(f"`{g}` ({r:.0%})" for g, r in list(high_groups.items())[:5])
                    lines.append(f"- High-missing groups: {groups_str}")
                lines.append("")

        # ── Structural absence section
        if struct_pairs:
            lines += [
                "---",
                "",
                "## Structural Absence",
                "",
                (
                    "The following column pairs show structural absence: when the categorical "
                    "column is NaN, its numeric companion is 0 or also NaN. This pattern "
                    "suggests the NaN encodes the **absence of a facility or item**, not a "
                    "data collection error. Impute with a structural token/zero + indicator."
                ),
                "",
                "| Categorical (NA) | Numeric (companion) | Pattern | Evidence |",
                "|-----------------|---------------------|---------|----------|",
            ]
            for pair in struct_pairs:
                evidence = pair.get("evidence", pair.get("pattern", "—"))
                lines.append(
                    f"| `{pair['categorical_col']}` | `{pair['numeric_col']}` "
                    f"| {pair['pattern']} | {evidence} |"
                )
            lines.append("")

        # ── Imputation plan
        lines += [
            "---",
            "",
            "## Imputation Plan",
            "",
            "All imputation statistics must be fitted on **training data only**.",
            "",
            "| Column | Strategy | Indicator | Mechanism | Missing Rate | Reason |",
            "|--------|----------|-----------|-----------|-------------|--------|",
        ]
        for col, plan in plan_cols.items():
            if plan["strategy"] == "no_imputation_needed":
                continue
            ind = "Yes" if plan["add_missing_indicator"] else "No"
            mech_lbl = plan.get("mechanism_label", "—")
            miss_r = plan.get("missing_rate", "—")
            miss_r_str = f"{miss_r:.1%}" if isinstance(miss_r, float) else str(miss_r)
            lines.append(
                f"| `{col}` | `{plan['strategy']}` | {ind} "
                f"| {mech_lbl} | {miss_r_str} | {plan['reason']} |"
            )
        lines.append("")

        # Safety warnings
        warnings = [
            (col, plan["safety_warning"])
            for col, plan in plan_cols.items()
            if plan.get("safety_warning")
        ]
        if warnings:
            lines += ["### Safety Warnings", ""]
            for col, warn in warnings:
                lines.append(f"- `{col}`: {warn}")
            lines.append("")

        # Strategy summary
        sc = plan_summary.get("strategy_counts", {})
        if sc:
            lines += ["### Strategy Summary", ""]
            for strat, cnt in sorted(sc.items(), key=lambda x: -x[1]):
                lines.append(f"- `{strat}`: {cnt} column(s)")
            lines.append("")

        # ── Full MI upgrade recommendations
        mi_upgrade_cols = imputation_plan.get("summary", {}).get(
            "columns_where_full_mi_recommended", []
        )
        if mi_upgrade_cols:
            lines += [
                "---",
                "",
                "## When to Use Full Multiple Imputation (MICE)",
                "",
                (
                    "The following columns have conditions where simple median/token imputation "
                    "may be insufficient for **statistical inference** (confidence intervals, "
                    "hypothesis tests). For those use cases, upgrade to full MICE. "
                    "Criteria: missing rate > 25%, or any predictor correlation > 0.40, "
                    "or missingness correlated with target (Collins et al., 2001 via van Buuren FIMD Ch5)."
                ),
                "",
                "| Column | Missing Rate | Mechanism | Recommended MICE Method |",
                "|--------|-------------|-----------|------------------------|",
            ]
            for col in mi_upgrade_cols:
                plan_entry = plan_cols.get(col, {})
                miss_r = plan_entry.get("missing_rate", "—")
                miss_r_str = f"{miss_r:.1%}" if isinstance(miss_r, float) else str(miss_r)
                mech_lbl = plan_entry.get("mechanism_label", "—")
                dtype_v = plan_entry.get("dtype", "")
                if "int" in str(dtype_v) or "float" in str(dtype_v):
                    mice_method = "pmm (predictive mean matching)"
                elif plan_entry.get("cardinality") == 2:
                    mice_method = "logreg (logistic regression)"
                else:
                    mice_method = "polyreg/polr (multinomial/ordered logit)"
                lines.append(
                    f"| `{col}` | {miss_r_str} | {mech_lbl} | `{mice_method}` |"
                )
            lines += [
                "",
                "> **Note:** For ML feature engineering (prediction only), the simple "
                "strategies above are acceptable. The goal of full MI is to reflect "
                "uncertainty due to missing data, not to predict missing values accurately "
                "(Rubin 1987, van Buuren FIMD Ch2).",
                "",
            ]

        # Evidence details for each column
        lines += [
            "---",
            "",
            "## Per-Column Evidence",
            "",
            "Evidence collected for each recommendation:",
            "",
        ]
        for col, plan in plan_cols.items():
            if plan["strategy"] == "no_imputation_needed":
                continue
            lines.append(f"### `{col}`")
            lines.append(f"- **Strategy:** `{plan['strategy']}`")
            lines.append(f"- **Missing rate:** {plan.get('missing_rate', '—'):.1%}" if isinstance(plan.get('missing_rate'), float) else f"- **Missing rate:** {plan.get('missing_rate', '—')}")
            lines.append(f"- **Mechanism label:** {plan.get('mechanism_label', '—')}")
            # MAR robustness note from mechanism audit
            mech_info = col_mechanisms.get(col, {})
            rob_note = mech_info.get("mar_robustness_note")
            if rob_note:
                lines.append(f"- **MAR robustness:** {rob_note}")
            if plan.get("mi_upgrade_recommended"):
                lines.append("- **Full MI recommended:** Yes (consider MICE for inference tasks)")
            gdep = plan.get("group_dependency_evidence") or {}
            if gdep.get("detected"):
                lines.append(f"- **Group dependency:** detected via `{gdep.get('top_categorical_feature', '?')}` (spread={gdep.get('max_missingness_spread', '?'):.2f})" if isinstance(gdep.get('max_missingness_spread'), float) else f"- **Group dependency:** detected")
            tgt = plan.get("target_association_evidence") or {}
            if tgt.get("detected"):
                corr = tgt.get("correlation")
                lines.append(f"- **Target association:** correlation={corr:.3f}" if isinstance(corr, float) else "- **Target association:** detected")
            cov = plan.get("covariate_association_evidence") or []
            if cov:
                top_cov = cov[0]
                lines.append(f"- **Covariate association:** `{top_cov['feature']}` (r={top_cov['correlation']:.3f})")
            struct = plan.get("structural_evidence") or {}
            if struct.get("detected"):
                comp = struct.get("companion_columns", [])
                lines.append(f"- **Structural evidence:** companion columns={comp}")
            if plan.get("safety_warning"):
                lines.append(f"- **Safety warning:** {plan['safety_warning']}")
            lines.append("")

        # ── Leakage-safe protocol
        lines += [
            "---",
            "",
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

        # ── Limitations
        lines += [
            "---",
            "",
            "## Limitations",
            "",
            "- Mechanism labels (MCAR-compatible, MAR-like, group-dependent, target-associated) "
            "are statistical clues from observational data. They cannot confirm the true "
            "causal mechanism (van Buuren FIMD Ch1).",
            "- MAR is the default assumption. MNAR cannot be confirmed or ruled out from "
            "observational data alone. Sensitivity analysis (e.g., delta-adjustment) is "
            "needed when MNAR is suspected (van Buuren FIMD Ch5).",
            "- Simple median/token imputation (single imputation) underestimates variance "
            "and produces confidence intervals that are too narrow. Use full MICE with "
            "Rubin's rules when valid statistical inference is required (van Buuren FIMD Ch1, Table 1.1).",
            "- The MAR robustness note uses Collins et al. (2001) thresholds: missing rate "
            "< 25% and max correlation < 0.4. These are empirical guidelines, not hard cutoffs.",
            "- Group-dependent missingness may overlap with target-associated missingness "
            "when the grouping variable (e.g., jurisdiction) also correlates with the target.",
            "- Structural absence detection relies on categorical NA + numeric zero/absent "
            "co-occurrence patterns. False positives are possible for columns where "
            "zero is a common legitimate value.",
            "- High-cardinality detection uses heuristic thresholds (>50 unique values or "
            ">20% unique ratio). Domain knowledge should override these thresholds.",
            "- Groupwise imputation uses training-set group medians fitted on observed rows "
            "only. Groups with fewer than 5 observations fall back to the global median.",
            "",
        ]

        return "\n".join(lines)

    def _json(self, filename: str, data: dict) -> None:
        with open(os.path.join(self.logs_dir, filename), "w") as f:
            json.dump(data, f, indent=2, default=str)
