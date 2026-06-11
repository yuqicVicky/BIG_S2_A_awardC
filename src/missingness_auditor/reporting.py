"""Write JSON logs and markdown report; also provides in-memory markdown generation."""

from __future__ import annotations

import json
import os
from datetime import datetime


class ReportWriter:
    def __init__(self, output_dir: str = ".", llm_client=None):
        self.logs_dir = os.path.join(output_dir, "logs")
        self.reports_dir = os.path.join(output_dir, "reports")
        self.llm_client = llm_client

    def write_all(
        self,
        profile: dict,
        mechanism_audit: dict,
        structural_audit: dict,
        imputation_plan: dict,
        leakage_check: dict,
        mice_pooling: dict | None = None,
        mnar_sensitivity: dict | None = None,
    ) -> None:
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)

        self._json("missingness_profile.json", profile)
        self._json("missingness_mechanism_audit.json", mechanism_audit)
        self._json("structural_missingness_audit.json", structural_audit)
        self._json("imputation_plan.json", imputation_plan)
        self._json("leakage_safe_imputation_check.json", leakage_check)
        if mice_pooling is not None:
            self._json("mice_pooling.json", mice_pooling)
        if mnar_sensitivity is not None:
            self._json("mnar_sensitivity.json", mnar_sensitivity)

        md = self.generate_report_md(
            profile, mechanism_audit, structural_audit, imputation_plan, leakage_check,
            mice_pooling=mice_pooling, mnar_sensitivity=mnar_sensitivity,
        )
        with open(os.path.join(self.reports_dir, "missing_data_report.md"), "w") as f:
            f.write(md)
        # Best-effort PDF methods appendix; never let it break the core outputs.
        try:
            self.generate_report_pdf(
                md, os.path.join(self.reports_dir, "missing_data_report.pdf")
            )
        except Exception as exc:  # pragma: no cover - optional artifact
            print(f"[reporting] PDF generation skipped: {exc}")

    def generate_report_md(
        self,
        profile: dict,
        mechanism_audit: dict,
        structural_audit: dict,
        imputation_plan: dict,
        leakage_check: dict,
        mice_pooling: dict | None = None,
        mnar_sensitivity: dict | None = None,
    ) -> str:
        """Generate a markdown report. Includes LLM narratives when present in results."""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        summary = profile.get("summary", {})
        col_profiles = profile.get("columns", {})
        col_mechanisms = mechanism_audit.get("columns", {})
        plan_cols = imputation_plan.get("columns", {})
        plan_summary = imputation_plan.get("summary", {})
        protocol = leakage_check.get("global_protocol", {})
        struct_pairs = structural_audit.get("structural_pairs", [])

        # LLM-generated executive narrative (only if llm_client provided)
        llm_executive = self._generate_executive_narrative(
            summary, col_mechanisms, struct_pairs, plan_cols
        ) if self.llm_client else None

        lines = [
            "# Missing Data Report",
            "",
            f"*Generated: {ts}*",
            "",
            "## Overview",
            "",
        ]

        if llm_executive:
            lines += [llm_executive, ""]
        else:
            lines.append(
                "This report diagnoses missingness patterns and recommends a leakage-safe "
                "imputation strategy for each column. Mechanism labels are statistical clues "
                "from observational data — they do not confirm causal mechanisms."
            )

        lines += [
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
        # Global Little's MCAR test verdict (Little 1988).
        _lm = mechanism_audit.get("little_mcar_test", {}) or {}
        if _lm.get("applicable"):
            _p = _lm.get("p_value", 1.0)
            _verdict = (
                "**rejects MCAR** (p < 0.05) — missingness is not completely at random; "
                "treat as MAR and model the missingness."
                if _p < 0.05 else
                "**does not reject MCAR** (p ≥ 0.05) — no global evidence against MCAR."
            )
            lines += [
                f"**Little's MCAR test (Little, 1988):** χ² = {_lm.get('statistic')}, "
                f"df = {_lm.get('df')}, p = {_p:.4g} over {_lm.get('n_numeric_cols')} "
                f"numeric column(s). The test {_verdict}",
                "",
            ]
        elif _lm.get("reason"):
            lines += [f"*Little's MCAR test not run: {_lm['reason']}.*", ""]
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

        # Per-column significance tests that drive each label (not bare thresholds).
        def _driving_test(info: dict) -> tuple[str, str]:
            lbl = info.get("mechanism_label", "")
            if lbl == "group-dependent missingness":
                p = (info.get("group_dependency_evidence") or {}).get("chi2_p_value")
                return "χ² independence (vs group)", (f"{p:.4g}" if p is not None else "—")
            if lbl == "target-associated missingness":
                tt = info.get("target_test") or {}
                p = tt.get("p_value")
                return tt.get("method", "target test"), (f"{p:.4g}" if p is not None else "—")
            mt = info.get("mar_test") or {}
            p = mt.get("p_value")
            method = "logistic LR (vs covariates)" if mt else "correlation (no test)"
            return method, (f"{p:.4g}" if p is not None else "—")

        lines += [
            "**Significance tests behind each label** "
            "(labels are driven by these p-values, with correlations kept as effect size):",
            "",
            "| Column | Mechanism | Driving test | p-value |",
            "|--------|-----------|--------------|---------|",
        ]
        for col, info in col_mechanisms.items():
            method, pstr = _driving_test(info)
            lines.append(
                f"| `{col}` | {info.get('mechanism_label', '—')} | {method} | {pstr} |"
            )
        lines.append("")

        # Per-column LLM narratives (present when MechanismAuditor ran with llm_client)
        llm_mech_cols = [
            (col, info) for col, info in col_mechanisms.items()
            if info.get("llm_narrative")
        ]
        if llm_mech_cols:
            lines += ["### AI Interpretation", ""]
            for col, info in llm_mech_cols:
                lines.append(f"**`{col}`** — {info['llm_narrative']}")
                if info.get("llm_suggestion"):
                    lines.append(f"  *Suggestion: {info['llm_suggestion']}*")
                if info.get("llm_anomaly"):
                    lines.append("  ⚠️ *Complex or anomalous pattern — review carefully.*")
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
            llm_struct_summary = structural_audit.get("llm_summary")
            lines += [
                "---",
                "",
                "## Structural Absence",
                "",
            ]
            if llm_struct_summary:
                lines += [llm_struct_summary, ""]
            else:
                lines += [
                    "The following column pairs show structural absence: when the categorical "
                    "column is NaN, its numeric companion is 0 or also NaN. This pattern "
                    "suggests the NaN encodes the **absence of a facility or item**, not a "
                    "data collection error. Impute with a structural token/zero + indicator.",
                    "",
                ]
            lines += [
                "| Categorical (NA) | Numeric (companion) | Pattern | Explanation |",
                "|-----------------|---------------------|---------|-------------|",
            ]
            for pair in struct_pairs:
                explanation = (
                    pair.get("llm_explanation")
                    or pair.get("evidence")
                    or pair.get("pattern", "—")
                )
                lines.append(
                    f"| `{pair['categorical_col']}` | `{pair['numeric_col']}` "
                    f"| {pair['pattern']} | {explanation} |"
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
            reason = plan.get("llm_readable_reason") or plan["reason"]
            lines.append(
                f"| `{col}` | `{plan['strategy']}` | {ind} "
                f"| {mech_lbl} | {miss_r_str} | {reason} |"
            )
        lines.append("")

        # LLM alternatives
        alt_cols = [
            (col, plan) for col, plan in plan_cols.items()
            if plan.get("llm_alternatives")
        ]
        if alt_cols:
            lines += ["### Alternative Strategy Options", ""]
            for col, plan in alt_cols:
                alts = plan["llm_alternatives"]
                lines.append(f"**`{col}`** (current: `{plan['strategy']}`):")
                for alt in alts:
                    lines.append(f"  - {alt}")
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
            if plan.get("llm_readable_reason"):
                lines.append(f"> {plan['llm_readable_reason']}")
                lines.append("")
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

        # ── Multiple Imputation (Rubin pooling) results
        if mice_pooling and mice_pooling.get("columns"):
            lines += [
                "---",
                "",
                "## Multiple Imputation Results (Rubin's Rules)",
                "",
                f"*Method: {mice_pooling.get('method', 'MICE')}; m = {mice_pooling.get('m')} imputations; "
                f"estimand = {mice_pooling.get('estimand', 'per-column mean')}.*",
                "",
                "Single imputation treats filled values as certain and understates variance. "
                "The pooled standard error below propagates the extra uncertainty from "
                "missingness via Rubin's rules. **FMI** is the fraction of information about "
                "the estimand lost to missing data.",
                "",
                "| Column | Missing | Pooled mean | Naive SE (single) | Pooled SE (MI) | 95% CI | FMI |",
                "|--------|---------|------------|-------------------|----------------|--------|-----|",
            ]
            for col, p in mice_pooling["columns"].items():
                ci = p.get("ci_95", [None, None])
                ci_str = (f"[{ci[0]:.3g}, {ci[1]:.3g}]"
                          if ci and ci[0] is not None else "—")
                lines.append(
                    f"| `{col}` | {p.get('missing_rate', 0):.1%} "
                    f"| {p.get('pooled_estimate', float('nan')):.4g} "
                    f"| {p.get('naive_single_imputation_std_error', float('nan')):.4g} "
                    f"| {p.get('std_error', float('nan')):.4g} "
                    f"| {ci_str} | {p.get('fraction_missing_information', 0):.2f} |"
                )
            lines += [
                "",
                "> The pooled (MI) standard error is ≥ the naive single-imputation SE by "
                "construction — that gap is exactly the uncertainty single imputation hides "
                "(Rubin 1987; van Buuren FIMD Ch2).",
                "",
            ]

        # ── MNAR sensitivity analysis
        if mnar_sensitivity and mnar_sensitivity.get("columns"):
            fragile = mnar_sensitivity.get("fragile_columns", [])
            lines += [
                "---",
                "",
                "## MNAR Sensitivity Analysis (Delta-Adjustment)",
                "",
                f"*{mnar_sensitivity.get('method', 'delta-adjustment')}; "
                f"tipping-point rule: {mnar_sensitivity.get('tipping_point_rule', '')}.*",
                "",
                "Imputation assumes MAR, which cannot be verified from observed data. Each "
                "column's imputed values are shifted by `delta` standard deviations; the "
                "**tipping point** is the smallest |delta| at which the mean leaves the "
                "complete-case 95% CI. A small tipping point ⇒ the conclusion hinges on the "
                "untestable MAR assumption.",
                "",
                "| Column | Missing | Tipping point (|δ| SD) | Robustness |",
                "|--------|---------|------------------------|------------|",
            ]
            for col, a in mnar_sensitivity["columns"].items():
                tp = a.get("tipping_point_delta_sd")
                tp_str = f"{abs(tp)}" if tp is not None else "none (robust)"
                lines.append(
                    f"| `{col}` | {a.get('missing_rate', 0):.1%} | {tp_str} "
                    f"| {a.get('robustness', '—')} |"
                )
            if fragile:
                lines += [
                    "",
                    f"> ⚠️ **Fragile columns** (tip at |δ| ≤ 0.5 SD): "
                    + ", ".join(f"`{c}`" for c in fragile)
                    + ". MAR-based estimates for these are sensitive to plausible MNAR "
                    "departures — gather domain evidence on why values are missing "
                    "(van Buuren FIMD Ch9).",
                ]
            lines.append("")

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

    def generate_report_pdf(self, md_text: str, out_path: str) -> str:
        """Render the markdown report to a PDF "methods appendix" via reportlab.

        This is a pragmatic markdown renderer: headings, bullet lists, blockquotes,
        and pipe-tables are styled; everything else flows as body text. The goal is a
        shareable, attach-to-a-paper artifact, not a pixel-perfect typesetter.
        """
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors
        from reportlab.lib.units import inch
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Preformatted,
        )

        styles = getSampleStyleSheet()
        body = styles["BodyText"]
        h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontSize=16, spaceAfter=8)
        h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=13, spaceAfter=6)
        h3 = ParagraphStyle("h3", parent=styles["Heading3"], fontSize=11, spaceAfter=4)
        quote = ParagraphStyle(
            "quote", parent=body, leftIndent=14, textColor=colors.HexColor("#555555"),
            fontName="Helvetica-Oblique",
        )

        def esc(s: str) -> str:
            return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                    .replace("`", ""))

        def inline(s: str) -> str:
            # markdown bold **x** → <b>x</b>; strip backticks for code spans.
            import re
            s = esc(s)
            s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
            return s

        flow = []
        lines = md_text.split("\n")
        i = 0
        table_buf: list[str] = []

        def flush_table():
            nonlocal table_buf
            if not table_buf:
                return
            rows = []
            for r in table_buf:
                cells = [c.strip() for c in r.strip().strip("|").split("|")]
                rows.append(cells)
            # Drop the markdown separator row (---|---).
            rows = [r for r in rows if not all(set(c) <= set("-: ") for c in r)]
            if rows:
                data = [[Paragraph(inline(c), body) for c in r] for r in rows]
                tbl = Table(data, repeatRows=1, hAlign="LEFT")
                tbl.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eaf2fb")),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#bbbbbb")),
                    ("FONTSIZE", (0, 0), (-1, -1), 7),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 3),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ]))
                flow.append(tbl)
                flow.append(Spacer(1, 8))
            table_buf = []

        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            if stripped.startswith("|") and stripped.endswith("|"):
                table_buf.append(stripped)
                i += 1
                continue
            else:
                flush_table()

            if not stripped or stripped == "---":
                flow.append(Spacer(1, 6))
            elif stripped.startswith("### "):
                flow.append(Paragraph(inline(stripped[4:]), h3))
            elif stripped.startswith("## "):
                flow.append(Paragraph(inline(stripped[3:]), h2))
            elif stripped.startswith("# "):
                flow.append(Paragraph(inline(stripped[2:]), h1))
            elif stripped.startswith(">"):
                flow.append(Paragraph(inline(stripped.lstrip("> ").strip()), quote))
            elif stripped.startswith("```"):
                # consume a fenced code block
                i += 1
                code = []
                while i < len(lines) and not lines[i].strip().startswith("```"):
                    code.append(lines[i])
                    i += 1
                flow.append(Preformatted("\n".join(code), styles["Code"]))
            elif stripped.startswith("- ") or stripped.startswith("* "):
                flow.append(Paragraph("• " + inline(stripped[2:]), body))
            else:
                flow.append(Paragraph(inline(stripped), body))
            i += 1
        flush_table()

        doc = SimpleDocTemplate(
            out_path, pagesize=letter,
            leftMargin=0.8 * inch, rightMargin=0.8 * inch,
            topMargin=0.8 * inch, bottomMargin=0.8 * inch,
            title="Missing Data Report — Methods Appendix",
        )
        doc.build(flow)
        return out_path

    def _generate_executive_narrative(
        self,
        summary: dict,
        col_mechanisms: dict,
        struct_pairs: list,
        plan_cols: dict,
    ) -> str | None:
        """Call LLM to write a 3-5 sentence executive narrative for the report overview."""
        from ._llm import call_llm_text

        mech_summary = {}
        for col, info in col_mechanisms.items():
            lbl = info.get("mechanism_label", "?")
            mech_summary[lbl] = mech_summary.get(lbl, 0) + 1

        strategy_counts = {}
        for col, entry in plan_cols.items():
            s = entry.get("strategy", "?")
            strategy_counts[s] = strategy_counts.get(s, 0) + 1

        prompt = (
            "Write a 3-5 sentence executive narrative for a missing-data audit report. "
            "Cover: overall data quality, the most concerning missingness mechanisms, "
            "the imputation strategy selected, and the main risk the analyst should address. "
            "Write in plain English for a non-expert audience.\n\n"
            f"Dataset: {summary.get('total_rows', '?')} rows × {summary.get('total_columns', '?')} columns\n"
            f"Overall missing rate: {summary.get('overall_missing_rate', 0):.1%}\n"
            f"Columns with missing: {summary.get('columns_with_any_missing', 0)}\n"
            f"Mechanism breakdown: {mech_summary}\n"
            f"Structural pairs detected: {len(struct_pairs)}\n"
            f"Strategy counts: {strategy_counts}"
        )
        text = call_llm_text(self.llm_client, prompt, max_tokens=250)
        return text if text else None

    def _json(self, filename: str, data: dict) -> None:
        with open(os.path.join(self.logs_dir, filename), "w") as f:
            json.dump(data, f, indent=2, default=str)
