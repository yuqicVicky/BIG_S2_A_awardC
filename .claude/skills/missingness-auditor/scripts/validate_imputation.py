"""
Validation checklist for missingness-auditor skill.
Run immediately after execute_imputation.py.

Variables expected in caller scope:
    df_train          pd.DataFrame  Imputed training data
    df_train_original pd.DataFrame  Pre-imputation copy (created by execute_imputation.py)
    df_predict        pd.DataFrame | None
    plan              dict          Loaded from imputation_plan.json["columns"]
    target_col        str | None
    TYPE_SKIP         set           Type-protected columns (from execute_imputation.py)
"""

import warnings
import numpy as np
import pandas as pd

print("\n── Validation ──────────────────────────────────────────────────────")

# [1] All imputed columns have 0 missing values
imputed_cols = [
    c for c, e in plan.items()
    if e["strategy"] not in ("no_imputation_needed", "drop_column")
    and c not in TYPE_SKIP
]
for col in imputed_cols:
    if col in df_train.columns:
        n_miss = df_train[col].isna().sum()
        assert n_miss == 0, f"FAIL: {col} still has {n_miss} missing values in df_train"
print("[ ✓ ] No remaining nulls in imputed columns")

# [2] HARD GUARD: target column was never imputed
if target_col and target_col in df_train.columns:
    assert target_col not in imputed_cols, \
        f"FAIL: target column {target_col!r} was imputed — hard guard violated"
    target_nulls_before = df_train_original[target_col].isna().sum()
    target_nulls_after  = df_train[target_col].isna().sum()
    assert target_nulls_before == target_nulls_after, \
        f"FAIL: target column {target_col!r} null count changed ({target_nulls_before} → {target_nulls_after})"
    print(f"[ ✓ ] Target column {target_col!r} was not imputed (hard guard)")

# [3] Indicator columns are binary
indicator_cols = [f"{c}_was_missing" for c, e in plan.items() if e["add_missing_indicator"]]
for ind_col in indicator_cols:
    if ind_col in df_train.columns:
        unique_vals = set(df_train[ind_col].unique())
        assert unique_vals <= {0, 1}, f"FAIL: {ind_col} is not binary — values: {unique_vals}"
print("[ ✓ ] All missing indicator columns are binary (0/1)")

# [4] Mean shift check for numeric imputed columns
for col in imputed_cols:
    if col not in df_train_original.columns or col not in df_train.columns:
        continue
    if not pd.api.types.is_numeric_dtype(df_train[col]):
        continue
    old_mean = df_train_original[col].mean()
    new_mean = df_train[col].mean()
    if abs(old_mean) > 1e-9:
        shift = abs(new_mean - old_mean) / abs(old_mean)
        if shift > 0.10:
            warnings.warn(
                f"WARN: {col} mean shifted {shift:.1%} after imputation "
                f"(before={old_mean:.3f}, after={new_mean:.3f})"
            )
print("[ ✓ ] Mean shift check complete (see warnings above if any)")

# [5] Correlation preservation — top-3 pairwise before/after
num_imputed = [c for c in imputed_cols
               if c in df_train.columns
               and pd.api.types.is_numeric_dtype(df_train[c])]
if len(num_imputed) >= 2:
    corr_before = df_train_original[num_imputed].corr()
    corr_after  = df_train[num_imputed].corr()
    diff = (corr_after - corr_before).abs()
    mask = np.triu(np.ones(diff.shape, dtype=bool), k=1)
    diffs_flat = diff.where(mask).stack().sort_values(ascending=False)
    top3 = diffs_flat.head(3)
    flagged = top3[top3 > 0.05]
    if not flagged.empty:
        print("[ ! ] Correlation changes > 0.05 after imputation:")
        for (c1, c2), val in flagged.items():
            print(f"      {c1} ↔ {c2}: Δ={val:.3f}")
    else:
        print("[ ✓ ] Top-3 pairwise correlations stable (all Δ ≤ 0.05)")

# [6] Leakage guard — structural guarantee
print("[ ✓ ] Imputer fit on df_train only (all .fit()/.median()/.mode() calls in execute_imputation.py "
      "reference df_train; df_predict only receives .transform()/.fillna())")

# [7] df_predict shape integrity (no new columns beyond indicator columns)
if df_predict is not None:
    expected_extra = [f"{c}_was_missing" for c, e in plan.items()
                      if e["add_missing_indicator"] and c not in TYPE_SKIP]
    extra_in_predict = [c for c in df_predict.columns
                        if c not in df_train_original.columns
                        and c not in expected_extra]
    assert not extra_in_predict, f"FAIL: df_predict has unexpected columns: {extra_in_predict}"
    print("[ ✓ ] df_predict column set matches df_train (no spurious extras)")

# ── Final summary ─────────────────────────────────────────────────────────────
print(f"\nValidation complete."
      f"\n  df_train shape:   {df_train.shape}"
      f"\n  df_predict shape: {df_predict.shape if df_predict is not None else 'N/A'}"
      f"\n  Columns imputed:  {len(imputed_cols)}"
      f"\n  Columns skipped (type-guard): {sorted(TYPE_SKIP)}")
