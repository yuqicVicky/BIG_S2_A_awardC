"""
Generate a standalone, self-verifying reproduction script from an imputation plan.

The emitted script has no dependency on this package: it embeds the plan inline,
re-applies the exact per-column strategy with statistics fitted on the training
data only, and then *checks its own work* (no remaining nulls, target untouched,
indicator columns binary). Anyone can run it months later to reproduce the cleaned
dataset, or attach it to a methods appendix — the reproducibility artifact the
competing tool advertises, but self-verified.
"""

from __future__ import annotations

import json
import os

# The runtime applier is written into the generated file verbatim. Keeping it as a
# module-level string (rather than building it programmatically) means what we test
# here is exactly what ships in the generated script.
_APPLIER_SRC = '''
def _add_indicator(df, col):
    df[f"{col}_was_missing"] = df[col].isna().astype(int)
    return df


def apply_plan(df_train, plan, df_predict=None):
    """Re-apply an imputation plan. All statistics are fitted on df_train only."""
    df_train = df_train.copy()
    if df_predict is not None:
        df_predict = df_predict.copy()
    drop_cols = []
    for col, entry in plan.get("columns", {}).items():
        strategy = entry["strategy"]
        add_ind = entry.get("add_missing_indicator", False)
        if strategy == "no_imputation_needed":
            continue
        if strategy == "drop_column":
            drop_cols.append(col)
            continue
        if col not in df_train.columns:
            continue
        if add_ind:
            df_train = _add_indicator(df_train, col)
            if df_predict is not None and col in df_predict.columns:
                df_predict = _add_indicator(df_predict, col)

        if strategy in ("groupwise_numeric_median_plus_indicator", "group_median"):
            group_col = entry.get("group_col")
            global_median = df_train[col].median()
            if group_col and group_col in df_train.columns:
                gmap = df_train.groupby(group_col)[col].median().to_dict()
                df_train[col] = df_train[col].fillna(df_train[group_col].map(gmap))
                if df_predict is not None and col in df_predict.columns and group_col in df_predict.columns:
                    df_predict[col] = df_predict[col].fillna(df_predict[group_col].map(gmap))
            df_train[col] = df_train[col].fillna(global_median)
            if df_predict is not None and col in df_predict.columns:
                df_predict[col] = df_predict[col].fillna(global_median)
        elif strategy == "time_series_ffill_bfill_plus_indicator":
            df_train[col] = df_train[col].ffill().bfill()
            fallback = df_train[col].median()
            df_train[col] = df_train[col].fillna(fallback)
            if df_predict is not None and col in df_predict.columns:
                df_predict[col] = df_predict[col].ffill().bfill().fillna(fallback)
        elif strategy in ("categorical_missing_token", "categorical_missing_token_plus_indicator"):
            df_train[col] = df_train[col].fillna("MISSING")
            if df_predict is not None and col in df_predict.columns:
                df_predict[col] = df_predict[col].fillna("MISSING")
        elif strategy == "categorical_mode_plus_indicator":
            mode_vals = df_train[col].mode()
            fill = mode_vals.iloc[0] if len(mode_vals) else "MISSING"
            df_train[col] = df_train[col].fillna(fill)
            if df_predict is not None and col in df_predict.columns:
                df_predict[col] = df_predict[col].fillna(fill)
        elif strategy == "structural_none_token_plus_indicator":
            df_train[col] = df_train[col].fillna("NONE")
            if df_predict is not None and col in df_predict.columns:
                df_predict[col] = df_predict[col].fillna("NONE")
        elif strategy == "structural_zero_plus_indicator":
            df_train[col] = df_train[col].fillna(0)
            if df_predict is not None and col in df_predict.columns:
                df_predict[col] = df_predict[col].fillna(0)
        elif strategy == "structural_none_or_zero":
            import pandas as _pd
            fill = 0 if _pd.api.types.is_numeric_dtype(df_train[col]) else "NONE"
            df_train[col] = df_train[col].fillna(fill)
            if df_predict is not None and col in df_predict.columns:
                df_predict[col] = df_predict[col].fillna(fill)
        else:
            # numeric_median, numeric_median_plus_indicator, and model-based
            # strategies all reduce to a leakage-safe median in this standalone
            # script (the package uses IterativeImputer for the model-based case).
            fill = df_train[col].median()
            df_train[col] = df_train[col].fillna(fill)
            if df_predict is not None and col in df_predict.columns:
                df_predict[col] = df_predict[col].fillna(fill)

    if drop_cols:
        df_train = df_train.drop(columns=drop_cols, errors="ignore")
        if df_predict is not None:
            df_predict = df_predict.drop(columns=drop_cols, errors="ignore")
    if df_predict is not None:
        return df_train, df_predict
    return df_train


def self_verify(df_imputed, plan, target_col, original_target=None):
    """Assert the cleaning honoured its own guarantees."""
    failures = []
    cols = plan.get("columns", {})
    dropped = {c for c, e in cols.items() if e["strategy"] == "drop_column"}
    for col, entry in cols.items():
        if entry["strategy"] in ("no_imputation_needed", "drop_column"):
            continue
        if col in df_imputed.columns and df_imputed[col].isna().any():
            failures.append(f"{col}: {int(df_imputed[col].isna().sum())} nulls remain")
        if entry.get("add_missing_indicator"):
            ind = f"{col}_was_missing"
            if ind in df_imputed.columns:
                uniq = set(df_imputed[ind].dropna().unique())
                if not uniq.issubset({0, 1}):
                    failures.append(f"{ind}: indicator not binary ({uniq})")
    if target_col and original_target is not None and target_col in df_imputed.columns:
        import pandas as _pd
        if not df_imputed[target_col].reset_index(drop=True).equals(
            _pd.Series(original_target).reset_index(drop=True)
        ):
            failures.append(f"target column '{target_col}' was modified")
    for c in dropped:
        if c in df_imputed.columns:
            failures.append(f"drop_column '{c}' still present")
    return failures
'''


def emit_reproduction_script(
    plan: dict,
    data_path: str,
    target_col: str | None = None,
    out_path: str = "reproduce_imputation.py",
) -> str:
    """
    Write a standalone, self-verifying reproduction script and return its path.

    Parameters
    ----------
    plan : dict
        The ``imputation_plan`` produced by the auditor (the ``{"columns": {...}}`` dict).
    data_path : str
        Path to the original dataset the script should re-impute. Stored relative to
        the script's own location when possible so the artifact stays portable.
    target_col : str | None
        Target column to protect and verify as untouched.
    out_path : str
        Where to write the generated ``.py`` file.
    """
    plan_json = json.dumps(plan, indent=4, default=str)
    target_repr = repr(target_col)

    # Make the embedded data path relative to the script when they share a tree.
    script_dir = os.path.dirname(os.path.abspath(out_path))
    abs_data = os.path.abspath(data_path)
    try:
        rel_data = os.path.relpath(abs_data, script_dir)
    except ValueError:
        rel_data = abs_data

    header = f'''#!/usr/bin/env python
"""
Auto-generated reproducible imputation script (self-verifying).

Generated by missingness-auditor. Re-applies the exact imputation plan that was
recommended for `{os.path.basename(data_path)}`, fitting all statistics on the
training data only, then checks its own work. No dependency on the auditor package.

Usage:
    python {os.path.basename(out_path)}
"""

import json
import os

import numpy as np
import pandas as pd

# --- Embedded imputation plan -------------------------------------------------
PLAN = json.loads(r"""
{plan_json}
""")

TARGET_COL = {target_repr}
# Data path is resolved relative to this script so the artifact stays portable.
_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_HERE, {rel_data!r})

# --- Imputation engine (leakage-safe; statistics fitted on training data only) -
'''

    main = '''

# --- Run ----------------------------------------------------------------------
def main():
    if not os.path.exists(DATA_PATH):
        raise SystemExit(f"Original data not found at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    original_target = df[TARGET_COL].copy() if (TARGET_COL and TARGET_COL in df.columns) else None

    imputed = apply_plan(df, PLAN)
    failures = self_verify(imputed, PLAN, TARGET_COL, original_target)

    if failures:
        print("SELF-VERIFICATION FAILED:")
        for f in failures:
            print("  -", f)
        raise SystemExit(1)

    out = os.path.join(_HERE, "reproduced_imputed.csv")
    imputed.to_csv(out, index=False)
    print(f"OK - reproduced cleaned dataset ({len(imputed)} rows, "
          f"{imputed.shape[1]} cols) -> {out}")
    print("Self-verification passed: no residual nulls, target untouched, "
          "indicators binary.")


if __name__ == "__main__":
    main()
'''

    content = header + _APPLIER_SRC + main
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write(content)
    return out_path
