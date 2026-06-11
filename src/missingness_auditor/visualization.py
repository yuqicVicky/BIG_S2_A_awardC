"""Missingness visualizations — returns Figure objects or saves to disk."""

from __future__ import annotations

import os
import textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np


_SEVERITY_COLORS = {
    "high":     "#d62728",
    "moderate": "#ff7f0e",
    "low":      "#ffbb78",
    "trace":    "#aec7e8",
}

_CHART_METHODS: dict[str, str] = {
    "missingness_bar":    "_bar_chart_fig",
    "pattern_matrix":     "_pattern_matrix_fig",
    "target_signal":      "_target_signal_fig",
    "missing_correlation": "_missing_correlation_fig",
}


class MissingnessVisualizer:
    def __init__(self, df: pd.DataFrame, target_col: str | None = None, llm_client=None):
        self.df = df
        self.target_col = target_col
        self.llm_client = llm_client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def suggest_charts(
        self,
        ctx: str,
        has_target: bool,
        n_missing_cols: int,
    ) -> tuple[list[str], str]:
        """Use LLM (if available) to choose the 2-3 most informative charts.

        Falls back to a heuristic selection when llm_client is None.
        Returns (chart_names, rationale_string).
        """
        available = ["missingness_bar", "pattern_matrix"]
        if has_target:
            available.append("target_signal")
        if n_missing_cols >= 3:
            available.append("missing_correlation")

        if self.llm_client:
            from ._llm import call_llm_json
            prompt = (
                f"Missing-data audit summary:\n{ctx}\n\n"
                f"Available chart types: {available}\n"
                "- missingness_bar: bar chart of missing % per column — always useful\n"
                "- pattern_matrix: row×column heatmap of co-missingness — useful when 2+ columns missing\n"
                "- target_signal: compare target value for rows with/without a feature — "
                "  include if any target_signal=True\n"
                "- missing_correlation: correlation between missingness indicators — "
                "  useful when 3+ columns missing\n\n"
                "Choose 2-3 charts that give the most insight for THIS specific dataset. "
                'Return ONLY valid JSON: {"charts": ["name1", "name2"], "rationale": "brief reason"}'
            )
            result = call_llm_json(self.llm_client, prompt, max_tokens=200)
            if isinstance(result, dict):
                valid = [c for c in result.get("charts", []) if c in available]
                if valid:
                    return valid, result.get("rationale", "")

        # Heuristic fallback
        charts = ["missingness_bar"]
        if n_missing_cols >= 2:
            charts.append("pattern_matrix")
        if has_target:
            charts.append("target_signal")
        return charts, ""

    def generate_figures(self, charts: list[str] | None = None) -> dict[str, plt.Figure]:
        """Return {name: Figure} for the requested charts. Caller is responsible for closing."""
        if charts is None:
            charts = list(_CHART_METHODS.keys())
        out: dict[str, plt.Figure] = {}
        for name in charts:
            if name not in _CHART_METHODS:
                continue
            try:
                out[name] = getattr(self, _CHART_METHODS[name])()
            except Exception as exc:
                out[name] = self._blank_fig(f"Chart '{name}' could not be rendered: {exc}")
        return out

    def generate_all(self, figures_dir: str) -> None:
        """Save all figures to disk and close them."""
        os.makedirs(figures_dir, exist_ok=True)
        for name, fig in self.generate_figures().items():
            fig.savefig(
                os.path.join(figures_dir, f"{name}.png"),
                bbox_inches="tight", dpi=100,
            )
            plt.close(fig)

    # ------------------------------------------------------------------
    # Figure builders (each returns a Figure)
    # ------------------------------------------------------------------

    def _bar_chart_fig(self) -> plt.Figure:
        rates = self.df.isna().mean().sort_values(ascending=False)
        rates = rates[rates > 0]

        if rates.empty:
            return self._blank_fig("No missing values detected")

        def _color(r):
            if r >= 0.80: return _SEVERITY_COLORS["high"]
            if r >= 0.20: return _SEVERITY_COLORS["moderate"]
            if r >= 0.05: return _SEVERITY_COLORS["low"]
            return _SEVERITY_COLORS["trace"]

        colors = [_color(r) for r in rates]
        fig, ax = plt.subplots(figsize=(max(6, len(rates) * 0.55 + 2), 5))
        ax.bar(range(len(rates)), rates.values, color=colors)
        ax.set_xticks(range(len(rates)))
        ax.set_xticklabels(rates.index, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("Missing Rate")
        ax.set_title("Missingness Rate by Column")
        ax.set_ylim(0, 1.05)
        ax.axhline(0.20, color="orange", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.axhline(0.80, color="red",    linestyle="--", linewidth=0.8, alpha=0.7)
        legend_patches = [
            mpatches.Patch(color=_SEVERITY_COLORS["high"],     label="High (≥80%)"),
            mpatches.Patch(color=_SEVERITY_COLORS["moderate"], label="Moderate (20–80%)"),
            mpatches.Patch(color=_SEVERITY_COLORS["low"],      label="Low (5–20%)"),
            mpatches.Patch(color=_SEVERITY_COLORS["trace"],    label="Trace (<5%)"),
        ]
        ax.legend(handles=legend_patches, loc="upper right", fontsize=8)
        fig.tight_layout()

        n_high = int((rates >= 0.80).sum())
        n_mod  = int(((rates >= 0.20) & (rates < 0.80)).sum())
        n_low  = int((rates < 0.20).sum())
        top_col  = rates.index[0]
        top_rate = rates.iloc[0]
        parts = [f"{len(rates)} column(s) have missing values "
                 f"(highest: '{top_col}' at {top_rate:.1%})."]
        if n_high:
            parts.append(f"{n_high} column(s) exceed 80% — too sparse to impute reliably; consider dropping.")
        if n_mod:
            parts.append(f"{n_mod} moderate (20–80%) and {n_low} low (<20%) severity column(s) — imputation with a missing indicator is recommended.")
        else:
            parts.append(f"All missing rates are below 20% — median imputation is generally sufficient; add a missing indicator if mechanism is non-MCAR.")
        self._add_caption(fig, " ".join(parts))
        return fig

    def _pattern_matrix_fig(self) -> plt.Figure:
        missing_cols = [c for c in self.df.columns if self.df[c].isna().any()]
        if not missing_cols:
            return self._blank_fig("No missing values detected")

        show_cols = missing_cols[:50]
        df_s = self.df[show_cols]
        if len(df_s) > 300:
            df_s = df_s.sample(300, random_state=42)

        mat = df_s.isna().astype(int)
        fig, ax = plt.subplots(figsize=(max(6, len(show_cols) * 0.45 + 2), 6))
        ax.imshow(mat.T, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=1, interpolation="nearest")
        ax.set_yticks(range(len(show_cols)))
        ax.set_yticklabels(show_cols, fontsize=8)
        ax.set_xlabel("Rows (sample)")
        ax.set_title("Missingness Pattern Matrix\n(green = present, red = missing)")
        fig.tight_layout()

        row_miss_counts = mat.sum(axis=1)
        n_multi = int((row_miss_counts >= 2).sum())
        pct_multi = n_multi / len(mat)
        n_any = int((row_miss_counts >= 1).sum())
        pct_any = n_any / len(mat)
        if pct_multi > 0.05:
            caption = (
                f"{pct_any:.1%} of rows (sample) have at least one missing value; "
                f"{pct_multi:.1%} are missing in 2+ columns simultaneously. "
                "Clustered red bands across columns indicate systematic co-missingness — "
                "likely MAR or MNAR rather than independent random dropout."
            )
        else:
            caption = (
                f"{pct_any:.1%} of rows (sample) have at least one missing value. "
                "Missingness appears largely independent across columns — "
                "consistent with MCAR (random dropout). "
                "No strong co-missingness pattern detected."
            )
        self._add_caption(fig, caption)
        return fig

    def _target_signal_fig(self) -> plt.Figure:
        if self.target_col is None or self.target_col not in self.df.columns:
            return self._blank_fig("No target column — target signal plot skipped")

        missing_cols = [
            c for c in self.df.columns
            if self.df[c].isna().any() and c != self.target_col
        ]
        if not missing_cols:
            return self._blank_fig("No missing columns to compare against target")

        tgt = self.df[self.target_col]
        is_num = pd.api.types.is_numeric_dtype(tgt)
        show_cols = missing_cols[:12]

        present_vals, missing_vals, labels = [], [], []
        for col in show_cols:
            is_miss = self.df[col].isna()
            if is_miss.sum() < 3 or (~is_miss).sum() < 3:
                continue
            if is_num:
                pv = float(tgt[~is_miss].mean())
                mv = float(tgt[is_miss].mean())
            else:
                pv = float(tgt[~is_miss].value_counts(normalize=True).iloc[0]) if (~is_miss).sum() else 0.0
                mv = float(tgt[is_miss].value_counts(normalize=True).iloc[0]) if is_miss.sum() else 0.0
            present_vals.append(pv)
            missing_vals.append(mv)
            labels.append(col)

        if not labels:
            return self._blank_fig("Insufficient data for target signal plot")

        x = range(len(labels))
        w = 0.35
        fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.85 + 2), 5))
        ax.bar([i - w / 2 for i in x], present_vals, w, label="Feature present", color="#2196F3", alpha=0.85)
        ax.bar([i + w / 2 for i in x], missing_vals, w, label="Feature missing", color="#F44336", alpha=0.85)
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("Mean target" if is_num else "Most-common target fraction")
        ax.set_title("Target signal by missingness\n(difference → MAR-like or MNAR/structural pattern)")
        ax.legend(fontsize=9)
        fig.tight_layout()

        tgt_std = float(tgt.std()) if is_num else 1.0
        diffs = [abs(mv - pv) for pv, mv in zip(present_vals, missing_vals)]
        if diffs:
            max_idx  = int(np.argmax(diffs))
            max_col  = labels[max_idx]
            max_diff = diffs[max_idx]
            norm_diff = max_diff / tgt_std if tgt_std > 0 else 0.0
            if norm_diff >= 0.10:
                direction = "higher" if missing_vals[max_idx] > present_vals[max_idx] else "lower"
                caption = (
                    f"Strongest signal: '{max_col}' — rows where this feature is missing have a "
                    f"{direction} mean target (Δ = {max_diff:.2f}, {norm_diff:.2f}σ). "
                    "A substantial gap suggests MAR-like or MNAR missingness: "
                    "the missingness indicator carries predictive information and should be retained."
                )
            else:
                caption = (
                    f"Target means are similar between missing and present rows across all columns "
                    f"(max Δ = {max(diffs):.2f}, {norm_diff:.2f}σ for '{labels[int(np.argmax(diffs))]}'), "
                    "consistent with MCAR. Missing indicators may add little predictive value."
                )
        else:
            caption = "No target signal could be computed."
        self._add_caption(fig, caption)
        return fig

    def _missing_correlation_fig(self) -> plt.Figure:
        missing_cols = [c for c in self.df.columns if self.df[c].isna().any()]
        if len(missing_cols) < 2:
            return self._blank_fig("Need ≥ 2 columns with missing values for correlation plot")

        mat = self.df[missing_cols].isna().astype(float)
        # Drop constant columns (all-missing or never-missing) before corr() to avoid NaN
        varying = [c for c in missing_cols if mat[c].std() > 0]
        if len(varying) < 2:
            return self._blank_fig("Insufficient variance in missingness indicators for correlation plot")
        mat = mat[varying]
        missing_cols = varying
        corr = mat.corr()
        # Replace any residual NaN (e.g. near-constant columns) with 0 for display
        corr = corr.fillna(0.0)
        n = len(missing_cols)

        fig, ax = plt.subplots(figsize=(max(5, n * 0.9 + 2), max(4, n * 0.8 + 1)))
        im = ax.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
        plt.colorbar(im, ax=ax, shrink=0.8, label="correlation")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(missing_cols, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(missing_cols, fontsize=8)
        ax.set_title("Missingness Indicator Correlation\n(red = co-missing, green = mutually exclusive)")

        for i in range(n):
            for j in range(n):
                val = corr.values[i, j]
                ax.text(
                    j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if abs(val) > 0.7 else "black",
                )
        fig.tight_layout()

        pairs = [
            (missing_cols[i], missing_cols[j], corr.values[i, j])
            for i in range(n) for j in range(i + 1, n)
            if abs(corr.values[i, j]) > 0.5
        ]
        if pairs:
            top = sorted(pairs, key=lambda x: -abs(x[2]))[:3]
            desc = ", ".join(f"'{a}'↔'{b}' ({v:.2f})" for a, b, v in top)
            caption = (
                f"Strong co-missingness: {desc}. "
                "These columns tend to be missing together — likely a shared cause. "
                "Consider imputing them jointly or flagging their combined indicator."
            )
        else:
            caption = (
                "No strong co-missingness correlations detected (all |r| < 0.5). "
                "Each column's missingness appears largely independent of the others."
            )
        self._add_caption(fig, caption)
        return fig

    def decision_flow(self, imputation_plan: dict, out_path: str) -> str:
        """Render a CONSORT-style flow diagram of the imputation decision pipeline.

        Reads the imputation plan and draws the funnel: dataset → columns with
        missing values → mechanism breakdown → strategy assignment → leakage-safe
        imputation, with side boxes for the columns excluded at each stage (fully
        observed, dropped). Saves a PNG and returns its path.
        """
        from collections import Counter

        cols = imputation_plan.get("columns", {})
        total_cols = len(cols)
        n_rows = len(self.df)
        missing_cols = {c: e for c, e in cols.items()
                        if e.get("strategy") not in ("no_imputation_needed", None)}
        n_missing = len(missing_cols)
        n_observed = total_cols - n_missing

        mech_counts = Counter(e.get("mechanism_label", "—") for e in missing_cols.values())
        strat_counts = Counter(e.get("strategy", "—") for e in missing_cols.values())
        dropped = [c for c, e in missing_cols.items() if e.get("strategy") == "drop_column"]
        n_imputed = n_missing - len(dropped)
        n_indicator = sum(1 for e in missing_cols.values() if e.get("add_missing_indicator"))

        def _fmt_counts(counter, limit=6):
            items = counter.most_common(limit)
            lines = [f"{k}: {v}" for k, v in items]
            extra = len(counter) - len(items)
            if extra > 0:
                lines.append(f"+{extra} more")
            return "\n".join(lines) if lines else "(none)"

        fig, ax = plt.subplots(figsize=(9, 11))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 13)
        ax.axis("off")

        def box(x, y, w, h, text, fc="#eaf2fb", ec="#1f77b4"):
            from matplotlib.patches import FancyBboxPatch
            patch = FancyBboxPatch(
                (x - w / 2, y - h / 2), w, h,
                boxstyle="round,pad=0.1,rounding_size=0.12",
                facecolor=fc, edgecolor=ec, linewidth=1.4,
            )
            ax.add_patch(patch)
            ax.text(x, y, text, ha="center", va="center", fontsize=9, wrap=True)

        def arrow(x0, y0, x1, y1):
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle="-|>", color="#444", lw=1.6))

        main_x = 3.6
        # Tier 0 — dataset
        box(main_x, 12.0, 5.0, 1.1,
            f"Dataset\n{n_rows} rows × {total_cols} columns")
        arrow(main_x, 11.45, main_x, 10.85)
        # Tier 1 — columns with missing values (+ exclusion: fully observed)
        box(main_x, 10.2, 5.0, 1.2,
            f"{n_missing} of {total_cols} columns\nhave missing values")
        box(8.3, 10.2, 3.0, 1.1,
            f"Excluded:\n{n_observed} fully observed\n(no action)",
            fc="#f2f2f2", ec="#999999")
        arrow(5.85, 10.2, 6.8, 10.2)
        arrow(main_x, 9.6, main_x, 9.0)
        # Tier 2 — mechanism breakdown
        box(main_x, 8.0, 5.0, 1.9,
            "Mechanism clue per column\n" + _fmt_counts(mech_counts),
            fc="#fff3e6", ec="#ff7f0e")
        arrow(main_x, 7.05, main_x, 6.45)
        # Tier 3 — strategy assignment (+ exclusion: dropped)
        box(main_x, 5.3, 5.0, 2.0,
            "Leakage-safe strategy assigned\n" + _fmt_counts(strat_counts),
            fc="#eef7ee", ec="#2ca02c")
        if dropped:
            box(8.3, 5.3, 3.0, 1.3,
                f"Excluded:\n{len(dropped)} dropped\n(>80% missing)",
                fc="#fdecec", ec="#d62728")
            arrow(5.85, 5.3, 6.8, 5.3)
        arrow(main_x, 4.3, main_x, 3.7)
        # Tier 4 — terminal
        box(main_x, 2.8, 5.4, 1.9,
            f"Imputed (fit on train only)\n{n_imputed} columns imputed\n"
            f"{n_indicator} missing-indicator columns added",
            fc="#eaf2fb", ec="#1f77b4")

        ax.set_title("Imputation decision flow (CONSORT-style)", fontsize=13, pad=12)
        fig.text(0.5, 0.02,
                 "Statistics fitted on training data only · target column never imputed",
                 ha="center", fontsize=8, style="italic", color="#555555")
        fig.savefig(out_path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        return out_path

    @staticmethod
    def _add_caption(fig: plt.Figure, text: str) -> None:
        """Render a data-driven interpretation below the figure axes."""
        wrapped = "\n".join(textwrap.wrap(text, width=100))
        n_lines = wrapped.count("\n") + 1
        fig.text(
            0.5, -0.03 - 0.035 * (n_lines - 1),
            wrapped,
            ha="center", va="top",
            fontsize=8, style="italic",
            color="#555555",
        )

    @staticmethod
    def _blank_fig(message: str) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=11, wrap=True)
        ax.set_axis_off()
        return fig
