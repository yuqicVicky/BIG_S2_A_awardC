"""Visualization: generate 3 focused figures using matplotlib only."""

from __future__ import annotations

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Optional


_DPI = 120
_FIG_W = 12
_FIG_H = 6


class Visualizer:
    def __init__(
        self,
        train_df: pd.DataFrame,
        predict_df: Optional[pd.DataFrame],
        feature_type_report: dict,
        distribution_shift_report: dict,
        leakage_report: dict,
    ):
        self.train_df = train_df
        self.predict_df = predict_df
        self.ft = feature_type_report.get("columns", {})
        self.shift = distribution_shift_report
        self.leakage = leakage_report

    def generate_all(self, figures_dir: str = "outputs/figures") -> list[str]:
        os.makedirs(figures_dir, exist_ok=True)
        generated = []

        paths = {
            "train_prediction_coverage": os.path.join(figures_dir, "train_prediction_coverage.png"),
            "feature_availability": os.path.join(figures_dir, "feature_availability.png"),
            "distribution_shift_summary": os.path.join(figures_dir, "distribution_shift_summary.png"),
        }

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            _save(self._fig_train_prediction_coverage(), paths["train_prediction_coverage"])
            generated.append(paths["train_prediction_coverage"])

            _save(self._fig_feature_availability(), paths["feature_availability"])
            generated.append(paths["feature_availability"])

            _save(self._fig_distribution_shift_summary(), paths["distribution_shift_summary"])
            generated.append(paths["distribution_shift_summary"])

        return generated

    # ------------------------------------------------------------------
    # Figure 1: row/column counts for train vs predict
    # ------------------------------------------------------------------

    def _fig_train_prediction_coverage(self) -> plt.Figure:
        fig, axes = plt.subplots(1, 2, figsize=(_FIG_W, _FIG_H))
        fig.suptitle("Train vs Prediction Coverage", fontsize=14)

        labels = ["Train", "Predict"]
        row_counts = [len(self.train_df), len(self.predict_df) if self.predict_df is not None else 0]
        colors = ["steelblue", "tomato"]
        axes[0].bar(labels, row_counts, color=colors)
        axes[0].set_title("Row Counts")
        axes[0].set_ylabel("Rows")
        for i, v in enumerate(row_counts):
            axes[0].text(i, v + 0.01 * max(row_counts, default=1), str(v), ha="center", fontsize=10)

        col_counts = [len(self.train_df.columns)]
        col_labels = ["Train"]
        if self.predict_df is not None:
            col_counts.append(len(self.predict_df.columns))
            col_labels.append("Predict")
        axes[1].bar(col_labels, col_counts, color=colors[: len(col_labels)])
        axes[1].set_title("Column Counts")
        axes[1].set_ylabel("Columns")

        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Figure 2: feature availability (leakage severity pie + risk types)
    # ------------------------------------------------------------------

    def _fig_feature_availability(self) -> plt.Figure:
        risks = self.leakage.get("risks", [])
        summary = self.leakage.get("summary", {})
        fig, axes = plt.subplots(1, 2, figsize=(_FIG_W, _FIG_H))
        fig.suptitle("Feature Availability & Leakage Audit", fontsize=13)

        labels = []
        sizes = []
        colors_pie = []
        color_map = {"high": "tomato", "medium": "orange", "low": "steelblue"}
        for sev, color in color_map.items():
            n_sev = summary.get(sev, 0)
            if n_sev > 0:
                labels.append(f"{sev.capitalize()} ({n_sev})")
                sizes.append(n_sev)
                colors_pie.append(color)

        if sizes:
            axes[0].pie(sizes, labels=labels, colors=colors_pie, autopct="%1.0f%%", startangle=90)
            axes[0].set_title("Leakage Risk by Severity")
        else:
            axes[0].text(0.5, 0.5, "No risks detected", ha="center", va="center")
            axes[0].set_title("Leakage Risk by Severity")

        if risks:
            risk_types: dict = {}
            for r in risks:
                rt = r.get("risk_type", "unknown")
                risk_types[rt] = risk_types.get(rt, 0) + 1
            types = list(risk_types.keys())
            counts = list(risk_types.values())
            axes[1].barh(types, counts, color="steelblue", alpha=0.8)
            axes[1].set_title("Risk Counts by Type")
            axes[1].set_xlabel("Count")
        else:
            axes[1].text(0.5, 0.5, "No risks detected", ha="center", va="center")
            axes[1].set_title("Risk Counts by Type")

        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Figure 3: distribution shift summary for numeric columns
    # ------------------------------------------------------------------

    def _fig_distribution_shift_summary(self) -> plt.Figure:
        shifts = self.shift.get("numeric_shift", [])
        detected = [s for s in shifts if s.get("shift_detected")]
        cols_plot = detected[:6] if detected else shifts[:6]

        fig, axes = plt.subplots(1, max(len(cols_plot), 1), figsize=(_FIG_W, _FIG_H))
        fig.suptitle("Train vs Prediction Distribution Shift", fontsize=13)

        if not cols_plot:
            ax = axes if not hasattr(axes, "__len__") else axes
            ax.text(0.5, 0.5, "No numeric columns to compare", ha="center", va="center")
            fig.tight_layout()
            return fig

        if len(cols_plot) == 1:
            axes = [axes]

        for ax, entry in zip(axes, cols_plot):
            col = entry["column"]
            if col not in self.train_df.columns:
                continue
            t = self.train_df[col].dropna().astype(float)
            kwargs = dict(bins=40, alpha=0.6, density=True)
            ax.hist(t.values, label="Train", color="steelblue", **kwargs)
            if self.predict_df is not None and col in self.predict_df.columns:
                p = self.predict_df[col].dropna().astype(float)
                ax.hist(p.values, label="Predict", color="tomato", **kwargs)
            title = col[:20]
            if entry.get("shift_detected"):
                title += " *"
            ax.set_title(title, fontsize=9)
            ax.legend(fontsize=7)

        fig.tight_layout()
        return fig


def _save(fig: plt.Figure, path: str):
    try:
        fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    finally:
        plt.close(fig)
