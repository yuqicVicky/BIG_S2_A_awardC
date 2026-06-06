"""Visualization: generate all figures using matplotlib only."""

from __future__ import annotations

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
        target_pattern_report: dict,
        distribution_shift_report: dict,
        leakage_report: dict,
        *,
        target_col: Optional[str] = None,
    ):
        self.train_df = train_df
        self.predict_df = predict_df
        self.ft = feature_type_report.get("columns", {})
        self.tp = target_pattern_report
        self.shift = distribution_shift_report
        self.leakage = leakage_report
        self.target_col = target_col

    def generate_all(self, figures_dir: str = "outputs/figures") -> list[str]:
        os.makedirs(figures_dir, exist_ok=True)
        generated = []

        paths = {
            "train_prediction_coverage": os.path.join(figures_dir, "train_prediction_coverage.png"),
            "target_distribution": os.path.join(figures_dir, "target_distribution.png"),
            "feature_target_signal": os.path.join(figures_dir, "feature_target_signal.png"),
            "train_prediction_shift": os.path.join(figures_dir, "train_prediction_shift.png"),
            "leakage_feature_availability": os.path.join(figures_dir, "leakage_feature_availability.png"),
        }

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            _save(self._fig_train_prediction_coverage(), paths["train_prediction_coverage"])
            generated.append(paths["train_prediction_coverage"])

            _save(self._fig_target_distribution(), paths["target_distribution"])
            generated.append(paths["target_distribution"])

            _save(self._fig_feature_target_signal(), paths["feature_target_signal"])
            generated.append(paths["feature_target_signal"])

            _save(self._fig_train_prediction_shift(), paths["train_prediction_shift"])
            generated.append(paths["train_prediction_shift"])

            _save(self._fig_leakage(), paths["leakage_feature_availability"])
            generated.append(paths["leakage_feature_availability"])

            # Optional: datetime figures
            dt_cols = [c for c, i in self.ft.items() if i.get("inferred_type") == "datetime_like"
                       and c in self.train_df.columns]
            if dt_cols:
                p = os.path.join(figures_dir, "target_by_time_pattern.png")
                _save(self._fig_target_by_time(dt_cols[0]), p)
                generated.append(p)

                df_ext = self._extended_df(dt_cols[0])

                # Hour × dayofweek heatmap
                if df_ext is not None and "__hour" in df_ext.columns and "__dayofweek" in df_ext.columns:
                    p2 = os.path.join(figures_dir, "hour_by_dayofweek_heatmap.png")
                    _save(self._fig_hour_dow_heatmap(df_ext), p2)
                    generated.append(p2)

                # Hour × workingday heatmap
                if df_ext is not None and "__hour" in df_ext.columns and "__is_workingday" in df_ext.columns:
                    p3 = os.path.join(figures_dir, "hour_by_workingday_heatmap.png")
                    _save(self._fig_hour_workingday_heatmap(df_ext), p3)
                    generated.append(p3)

        return generated

    # ------------------------------------------------------------------
    # Individual figures
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

    def _fig_target_distribution(self) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
        if not self.target_col or self.target_col not in self.train_df.columns:
            ax.text(0.5, 0.5, "No target column available", ha="center", va="center")
            ax.set_title("Target Distribution")
            return fig

        target = self.train_df[self.target_col].dropna()
        if pd.api.types.is_numeric_dtype(target):
            ax.hist(target, bins=50, color="steelblue", edgecolor="white", alpha=0.85)
            ax.axvline(target.mean(), color="tomato", linestyle="--", label=f"mean={target.mean():.2f}")
            ax.axvline(target.median(), color="orange", linestyle="--", label=f"median={target.median():.2f}")
            ax.legend()
        else:
            vc = target.value_counts().head(20)
            ax.bar(range(len(vc)), vc.values, color="steelblue")
            ax.set_xticks(range(len(vc)))
            ax.set_xticklabels([str(v) for v in vc.index], rotation=45, ha="right")
        ax.set_title(f"Target Distribution: {self.target_col}")
        ax.set_xlabel(self.target_col)
        ax.set_ylabel("Count")
        fig.tight_layout()
        return fig

    def _fig_feature_target_signal(self) -> plt.Figure:
        corrs = self.tp.get("numeric_correlations", [])
        fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
        if not corrs:
            ax.text(0.5, 0.5, "No numeric correlations available", ha="center", va="center")
            ax.set_title("Feature-Target Signal")
            return fig

        valid = [(c["column"], c["pearson_r"]) for c in corrs if c.get("pearson_r") is not None]
        if not valid:
            ax.text(0.5, 0.5, "All correlations are null", ha="center", va="center")
            ax.set_title("Feature-Target Signal")
            return fig

        valid.sort(key=lambda x: x[1])
        cols, vals = zip(*valid)
        colors = ["tomato" if v < 0 else "steelblue" for v in vals]
        y_pos = range(len(cols))
        ax.barh(y_pos, vals, color=colors, alpha=0.85)
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(cols, fontsize=8)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title("Pearson Correlation with Target")
        ax.set_xlabel("Pearson r")
        fig.tight_layout()
        return fig

    def _fig_train_prediction_shift(self) -> plt.Figure:
        shifts = self.shift.get("numeric_shift", [])
        detected = [s for s in shifts if s.get("shift_detected")]
        n = max(len(detected), 1)
        cols_plot = detected[:6] if detected else shifts[:6]

        fig, axes = plt.subplots(1, len(cols_plot) if cols_plot else 1, figsize=(_FIG_W, _FIG_H))
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
                title += " ⚠"
            ax.set_title(title, fontsize=9)
            ax.legend(fontsize=7)

        fig.tight_layout()
        return fig

    def _fig_leakage(self) -> plt.Figure:
        risks = self.leakage.get("risks", [])
        summary = self.leakage.get("summary", {})
        fig, axes = plt.subplots(1, 2, figsize=(_FIG_W, _FIG_H))
        fig.suptitle("Leakage & Feature Availability Audit", fontsize=13)

        # Pie chart of severity
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
            axes[0].set_title("Risk Severity Distribution")
        else:
            axes[0].text(0.5, 0.5, "No risks detected", ha="center", va="center")
            axes[0].set_title("Risk Severity Distribution")

        # Horizontal bar of risk types
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

    def _fig_target_by_time(self, dt_col: str) -> plt.Figure:
        from .datetime_patterns import DatetimePatternAnalyser
        dpa = DatetimePatternAnalyser(self.train_df, dt_col, self.target_col)
        by_comp = dpa.target_by_component()

        components = [k for k in ["hour", "dayofweek", "month"] if k in by_comp]
        n = len(components) if components else 1
        fig, axes = plt.subplots(1, n, figsize=(_FIG_W, _FIG_H))
        fig.suptitle(f"Target Mean by Datetime Component ({dt_col})", fontsize=13)
        if n == 1:
            axes = [axes]

        for ax, comp in zip(axes, components):
            records = by_comp[comp]
            key = f"__{comp}"
            xs = [r.get(key, r.get(comp, i)) for i, r in enumerate(records)]
            ys = [r.get("target_mean", 0) for r in records]
            ax.bar([str(x) for x in xs], ys, color="steelblue", alpha=0.85)
            ax.set_title(comp.replace("_", " ").title())
            ax.set_xlabel(comp)
            ax.set_ylabel("Mean Target")
            ax.tick_params(axis="x", rotation=45)

        fig.tight_layout()
        return fig

    def _extended_df(self, dt_col: str) -> Optional[pd.DataFrame]:
        from .datetime_patterns import DatetimePatternAnalyser
        try:
            dpa = DatetimePatternAnalyser(self.train_df, dt_col, self.target_col)
            return dpa.extract_components()
        except Exception:
            return None

    def _fig_hour_dow_heatmap(self, df_ext: pd.DataFrame) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
        fig.suptitle("Mean Target: Hour × Day of Week", fontsize=13)

        if self.target_col not in df_ext.columns:
            ax.text(0.5, 0.5, "No target column", ha="center", va="center")
            return fig

        pivot = (
            df_ext.groupby(["__hour", "__dayofweek"])[self.target_col]
            .mean()
            .unstack(fill_value=np.nan)
        )

        im = ax.imshow(pivot.values, aspect="auto", cmap="coolwarm", origin="lower")
        ax.set_xticks(range(pivot.shape[1]))
        ax.set_xticklabels(
            [["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][i] for i in pivot.columns],
            fontsize=9,
        )
        ax.set_yticks(range(pivot.shape[0]))
        ax.set_yticklabels(list(pivot.index), fontsize=8)
        ax.set_xlabel("Day of Week")
        ax.set_ylabel("Hour")
        fig.colorbar(im, ax=ax, label="Mean Target")
        fig.tight_layout()
        return fig


    def _fig_hour_workingday_heatmap(self, df_ext: pd.DataFrame) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(_FIG_W, _FIG_H))
        fig.suptitle("Mean Target: Hour × Working Day", fontsize=13)

        if self.target_col not in df_ext.columns:
            ax.text(0.5, 0.5, "No target column", ha="center", va="center")
            return fig

        pivot = (
            df_ext.groupby(["__hour", "__is_workingday"])[self.target_col]
            .mean()
            .unstack(fill_value=np.nan)
        )

        im = ax.imshow(pivot.values, aspect="auto", cmap="coolwarm", origin="lower")
        ax.set_xticks(range(pivot.shape[1]))
        ax.set_xticklabels(
            ["Weekend" if c == 0 else "Workday" for c in pivot.columns],
            fontsize=10,
        )
        ax.set_yticks(range(pivot.shape[0]))
        ax.set_yticklabels(list(pivot.index), fontsize=8)
        ax.set_xlabel("Day Type")
        ax.set_ylabel("Hour")
        fig.colorbar(im, ax=ax, label="Mean Target")
        fig.tight_layout()
        return fig


def _save(fig: plt.Figure, path: str):
    try:
        fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    finally:
        plt.close(fig)
