"""
MNAR sensitivity analysis via delta-adjustment (pattern-mixture / tipping-point).

Every imputation method in this package fills missing cells under a Missing-At-Random
(MAR) working assumption. MAR cannot be verified from the observed data alone
(van Buuren FIMD Ch1). When missingness may depend on the unobserved value itself
(MNAR) — for example, the highest overdose-rate weeks being the ones that go
unreported — the honest question is not "what is the right fill?" but "how wrong
would my conclusion be if the missing values are systematically different?".

Delta-adjustment answers that. The imputed values are shifted by ``delta`` standard
deviations and the estimand (here, the column mean) is recomputed across a grid of
deltas. The **tipping point** is the smallest |delta| at which the full-sample mean
leaves the 95% confidence interval of the complete-case mean — i.e. the magnitude of
MNAR departure that would overturn a conclusion drawn under MAR. A large tipping point
means the result is robust; a small one means it hinges on an untestable assumption.

References
----------
van Buuren, S. FIMD Ch9 (Sensitivity analysis under MNAR).
Cro, S. et al. (2020). Reference-based / delta-adjustment sensitivity analysis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_DEFAULT_DELTAS = [-2.0, -1.5, -1.0, -0.75, -0.5, -0.25, 0.0,
                   0.25, 0.5, 0.75, 1.0, 1.5, 2.0]


def _select_columns(df: pd.DataFrame, plan: dict, target_col: str | None) -> list[str]:
    """Columns most in need of MNAR scrutiny: numeric, missing, and either
    target-associated or flagged for an MI upgrade by the planner."""
    cols = []
    plan_cols = (plan or {}).get("columns", {})
    for c in df.columns:
        if c == target_col:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        if not df[c].isna().any():
            continue
        entry = plan_cols.get(c, {})
        mech = entry.get("mechanism_label", "")
        flagged = (
            entry.get("mi_upgrade_recommended")
            or "target-associated" in mech
            or "MNAR" in mech
            or "structural" in mech
        )
        # When we have no plan signal at all, still analyse every missing numeric
        # column — sensitivity analysis is cheap and always informative.
        if flagged or not plan_cols:
            cols.append(c)
    # Fall back to all missing numeric columns if nothing was specifically flagged.
    if not cols:
        cols = [
            c for c in df.columns
            if c != target_col
            and pd.api.types.is_numeric_dtype(df[c])
            and df[c].isna().any()
        ]
    return cols


def _analyze_column(series: pd.Series, deltas: list[float]) -> dict | None:
    observed = series.dropna().astype(float)
    n_obs = int(observed.size)
    n_missing = int(series.isna().sum())
    if n_obs < 2 or n_missing == 0:
        return None

    sd = float(observed.std(ddof=1))
    base_fill = float(observed.median())
    mean_obs = float(observed.mean())
    se_obs = sd / np.sqrt(n_obs)
    ci_lo, ci_hi = mean_obs - 1.96 * se_obs, mean_obs + 1.96 * se_obs

    miss_mask = series.isna()
    trajectory = []
    for delta in deltas:
        filled = series.copy().astype(float)
        filled[miss_mask] = base_fill + delta * sd
        mean_delta = float(filled.mean())
        trajectory.append({
            "delta_sd": delta,
            "mean": mean_delta,
            "outside_complete_case_ci": bool(mean_delta < ci_lo or mean_delta > ci_hi),
        })

    # Tipping point: smallest |delta| (delta != 0) whose full-sample mean falls
    # outside the complete-case 95% CI.
    tipping = None
    for point in sorted(trajectory, key=lambda p: abs(p["delta_sd"])):
        if point["delta_sd"] == 0.0:
            continue
        if point["outside_complete_case_ci"]:
            tipping = point["delta_sd"]
            break

    if tipping is None:
        robustness = "robust"
        interpretation = (
            "Across the full delta grid the imputed mean stays within the "
            "complete-case 95% CI — the mean estimate is robust to MNAR departures "
            "of this magnitude."
        )
    elif abs(tipping) <= 0.5:
        robustness = "fragile"
        interpretation = (
            f"The conclusion tips at only |delta|={abs(tipping)} SD: a modest MNAR "
            "departure would move the mean outside its complete-case CI. Treat any "
            "MAR-based estimate with caution and collect domain evidence on why values "
            "are missing."
        )
    else:
        robustness = "moderately_robust"
        interpretation = (
            f"The conclusion tips at |delta|={abs(tipping)} SD — only a sizeable MNAR "
            "departure would overturn the MAR-based mean."
        )

    return {
        "n_observed": n_obs,
        "n_missing": n_missing,
        "missing_rate": float(series.isna().mean()),
        "observed_sd": sd,
        "complete_case_mean": mean_obs,
        "complete_case_ci_95": [ci_lo, ci_hi],
        "tipping_point_delta_sd": tipping,
        "robustness": robustness,
        "interpretation": interpretation,
        "trajectory": trajectory,
    }


def mnar_sensitivity(
    df_train: pd.DataFrame,
    plan: dict | None = None,
    target_col: str | None = None,
    deltas: list[float] | None = None,
) -> dict:
    """
    Run delta-adjustment sensitivity analysis on the columns most exposed to MNAR.

    Returns a JSON-serialisable dict suitable for ``logs/mnar_sensitivity.json``.
    """
    deltas = list(deltas) if deltas is not None else list(_DEFAULT_DELTAS)
    cols = _select_columns(df_train, plan or {}, target_col)

    columns: dict[str, dict] = {}
    for c in cols:
        analysis = _analyze_column(df_train[c], deltas)
        if analysis is not None:
            columns[c] = analysis

    fragile = [c for c, a in columns.items() if a["robustness"] == "fragile"]
    return {
        "method": "delta-adjustment (pattern-mixture) MNAR sensitivity analysis",
        "estimand": "column mean",
        "deltas_sd": deltas,
        "tipping_point_rule": (
            "smallest |delta| (in observed-SD units) at which the full-sample mean "
            "leaves the complete-case 95% CI"
        ),
        "columns": columns,
        "fragile_columns": fragile,
        "note": (
            "MAR cannot be verified from observed data. A small tipping point means a "
            "conclusion hinges on the untestable MAR assumption; a large one means it "
            "is robust to plausible MNAR departures (van Buuren FIMD Ch9)."
        ),
    }


def plot_tipping_point(sensitivity: dict, out_path: str, max_panels: int = 4) -> str | None:
    """
    Render delta-adjustment trajectories with the complete-case CI band and the
    tipping point marked. Saves a PNG and returns its path (or None if nothing to plot).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    columns = sensitivity.get("columns", {})
    if not columns:
        return None

    # Show the most fragile columns first (smallest tipping point), capped at max_panels.
    def _sort_key(item):
        tp = item[1].get("tipping_point_delta_sd")
        return abs(tp) if tp is not None else float("inf")

    items = sorted(columns.items(), key=_sort_key)[:max_panels]
    n = len(items)
    ncols = min(2, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)

    for ax in axes.flat:
        ax.set_visible(False)

    for idx, (col, a) in enumerate(items):
        ax = axes.flat[idx]
        ax.set_visible(True)
        deltas = [p["delta_sd"] for p in a["trajectory"]]
        means = [p["mean"] for p in a["trajectory"]]
        ax.plot(deltas, means, marker="o", color="#1f77b4", label="imputed mean")
        lo, hi = a["complete_case_ci_95"]
        ax.axhspan(lo, hi, color="#2ca02c", alpha=0.15,
                   label="complete-case 95% CI")
        ax.axhline(a["complete_case_mean"], color="#2ca02c", lw=1, ls="--")
        tp = a.get("tipping_point_delta_sd")
        if tp is not None:
            ax.axvline(tp, color="#d62728", lw=1.5, ls=":",
                       label=f"tipping point (|δ|={abs(tp)} SD)")
            ax.axvline(-tp, color="#d62728", lw=1.0, ls=":")
        ax.set_title(f"{col}  ({a['robustness']}, miss={a['missing_rate']:.0%})")
        ax.set_xlabel("delta (SD units applied to imputed values)")
        ax.set_ylabel("estimated column mean")
        ax.legend(fontsize=8, loc="best")

    fig.suptitle("MNAR sensitivity: delta-adjustment tipping-point analysis",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out_path
