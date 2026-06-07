"""Missingness visualizations — returns Figure objects or saves to disk."""

from __future__ import annotations

import os
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


class MissingnessVisualizer:
    def __init__(self, df: pd.DataFrame, target_col: str | None = None):
        self.df = df
        self.target_col = target_col

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_figures(self) -> dict[str, plt.Figure]:
        """Return {name: Figure} — caller is responsible for closing."""
        return {
            "missingness_bar":           self._bar_chart_fig(),
            "missingness_matrix":        self._pattern_matrix_fig(),
            "missingness_target_signal": self._target_signal_fig(),
        }

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
        return fig

    @staticmethod
    def _blank_fig(message: str) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=11, wrap=True)
        ax.set_axis_off()
        return fig
