"""
Execution template for missingness-auditor skill.
Run after user confirms the imputation plan (Step 6 of SKILL.md).

Variables expected in caller scope:
    df_train          pd.DataFrame  Training data (required)
    df_predict        pd.DataFrame | None  Predict/test data (optional; default None)
    target_col        str | None    Target column name (excluded from imputation)
    plan_path         str           Path to imputation_plan.json
                                    (default: "outputs/logs/imputation_plan.json")
"""

import json
import re
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer

# ── Configuration ────────────────────────────────────────────────────────────
plan_path  = plan_path if "plan_path" in dir() else "outputs/logs/imputation_plan.json"
df_predict = df_predict if "df_predict" in dir() else None

# ── HARD GUARD 1: preserve unimputed copy for validation ─────────────────────
df_train_original = df_train.copy()

# ── HARD GUARD 2: build type-skip set ────────────────────────────────────────
# These column types must never be imputed.
TYPE_SKIP: set = set()

if target_col and target_col in df_train.columns:
    TYPE_SKIP.add(target_col)                          # target: never impute

for col in df_train.columns:
    if col == target_col:
        continue
    if (re.search(r"(?i)(^id$|_id$|^id_)", col)       # ID-like name
            or df_train[col].nunique() == len(df_train)):  # all-unique → surrogate key
        TYPE_SKIP.add(col)
    elif pd.api.types.is_datetime64_any_dtype(df_train[col]):
        TYPE_SKIP.add(col)                             # datetime: handle separately

# ── Load plan ────────────────────────────────────────────────────────────────
with open(plan_path) as f:
    plan = json.load(f)["columns"]

# ── Helper ───────────────────────────────────────────────────────────────────
def _add_indicator(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Add binary indicator BEFORE imputing so the flag captures true nulls."""
    df[f"{col}_was_missing"] = df[col].isna().astype(int)
    return df

# ── Apply per-column strategy ─────────────────────────────────────────────────
drop_cols: list = []

for col, entry in plan.items():
    strategy = entry["strategy"]
    add_ind   = entry["add_missing_indicator"]

    # ── HARD GUARD: skip target and type-protected columns ───────────────────
    if col in TYPE_SKIP:
        print(f"[GUARD] Skipping {col!r} (type-protected or target column)")
        continue

    if strategy == "no_imputation_needed":
        continue

    elif strategy == "drop_column":
        drop_cols.append(col)

    elif strategy == "numeric_median":
        median = df_train[col].median()                          # fit on train only
        df_train[col]   = df_train[col].fillna(median)
        if df_predict is not None:
            df_predict[col] = df_predict[col].fillna(median)    # apply same value

    elif strategy == "numeric_median_plus_indicator":
        if add_ind:
            df_train  = _add_indicator(df_train, col)
            if df_predict is not None:
                df_predict = _add_indicator(df_predict, col)
        median = df_train[col].median()                          # fit on train only
        df_train[col]   = df_train[col].fillna(median)
        if df_predict is not None:
            df_predict[col] = df_predict[col].fillna(median)

    elif strategy in ("categorical_missing_token", "categorical_missing_token_plus_indicator"):
        if add_ind:
            df_train  = _add_indicator(df_train, col)
            if df_predict is not None:
                df_predict = _add_indicator(df_predict, col)
        df_train[col]   = df_train[col].fillna("MISSING")
        if df_predict is not None:
            df_predict[col] = df_predict[col].fillna("MISSING")

    elif strategy == "groupwise_numeric_median_plus_indicator":
        # Geo-group imputation: uses per-group median, falls back to global.
        # Applied ONLY when between-group variance condition was met (see evidence.geo_grouping_applied).
        # evidence.geo_grouping_skipped_reason is non-null when condition was not met.
        group_col = entry.get("group_col")
        if add_ind:
            df_train  = _add_indicator(df_train, col)
            if df_predict is not None:
                df_predict = _add_indicator(df_predict, col)
        global_median = df_train[col].median()                   # fit on train only
        if group_col and group_col in df_train.columns:
            group_map = df_train.groupby(group_col)[col].median().to_dict()  # fit on train only
            df_train[col] = df_train[col].fillna(df_train[group_col].map(group_map))
            if df_predict is not None and group_col in df_predict.columns:
                df_predict[col] = df_predict[col].fillna(df_predict[group_col].map(group_map))
        df_train[col]   = df_train[col].fillna(global_median)    # fallback: unknown groups
        if df_predict is not None:
            df_predict[col] = df_predict[col].fillna(global_median)

    elif strategy == "structural_zero_plus_indicator":
        if add_ind:
            df_train  = _add_indicator(df_train, col)
            if df_predict is not None:
                df_predict = _add_indicator(df_predict, col)
        df_train[col]   = df_train[col].fillna(0)
        if df_predict is not None:
            df_predict[col] = df_predict[col].fillna(0)

    elif strategy == "structural_none_token_plus_indicator":
        if add_ind:
            df_train  = _add_indicator(df_train, col)
            if df_predict is not None:
                df_predict = _add_indicator(df_predict, col)
        df_train[col]   = df_train[col].fillna("NONE")
        if df_predict is not None:
            df_predict[col] = df_predict[col].fillna("NONE")

    elif strategy == "structural_none_or_zero":
        # Legacy alias — dispatches by dtype. Prefer explicit strategies above.
        if add_ind:
            df_train  = _add_indicator(df_train, col)
            if df_predict is not None:
                df_predict = _add_indicator(df_predict, col)
        fill_val = 0 if pd.api.types.is_numeric_dtype(df_train[col]) else "NONE"
        df_train[col]   = df_train[col].fillna(fill_val)
        if df_predict is not None:
            df_predict[col] = df_predict[col].fillna(fill_val)

    elif strategy == "model_based_imputation_optional":
        # IterativeImputer (MICE-like). Fit on df_train only.
        # HARD GUARD: df_predict only receives .transform(), never .fit_transform().
        if add_ind:
            df_train  = _add_indicator(df_train, col)
            if df_predict is not None:
                df_predict = _add_indicator(df_predict, col)
        num_cols = [c for c in df_train.select_dtypes(include="number").columns
                    if c not in TYPE_SKIP]
        imp = IterativeImputer(max_iter=10, random_state=0)
        df_train[num_cols] = imp.fit_transform(df_train[num_cols])   # fit on train only
        if df_predict is not None:
            df_predict[num_cols] = imp.transform(df_predict[num_cols])  # transform only

# ── Drop flagged columns ─────────────────────────────────────────────────────
if drop_cols:
    df_train  = df_train.drop(columns=drop_cols, errors="ignore")
    if df_predict is not None:
        df_predict = df_predict.drop(columns=drop_cols, errors="ignore")

# ── Write outputs ─────────────────────────────────────────────────────────────
df_train.to_csv("outputs/train_imputed.csv", index=False)
if df_predict is not None:
    df_predict.to_csv("outputs/predict_imputed.csv", index=False)

print(f"\nImputation complete. Train: {df_train.shape}. "
      f"Predict: {df_predict.shape if df_predict is not None else 'N/A'}.")
print(f"Type-protected (skipped): {sorted(TYPE_SKIP)}")
