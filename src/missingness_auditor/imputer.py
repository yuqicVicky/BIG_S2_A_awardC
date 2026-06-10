"""Leakage-safe imputation execution: fit on train, apply to train and predict."""

from __future__ import annotations

import pandas as pd
import numpy as np


def _add_indicator(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Add a binary missing indicator column BEFORE imputing."""
    df[f"{col}_was_missing"] = df[col].isna().astype(int)
    return df


class Imputer:
    """
    Applies an imputation plan to a DataFrame.

    All statistics (median, mode, group medians) are fitted on df_train only,
    then applied to both df_train and df_predict. This prevents information
    from the test/predict set from leaking into imputation parameters.

    Supported strategies
    --------------------
    no_imputation_needed                     : skip
    drop_column                              : remove column from both datasets
    numeric_median                           : fill with training-set median
    numeric_median_plus_indicator            : fill with median + binary indicator
    groupwise_numeric_median_plus_indicator  : fill with per-group median + indicator
    group_median                             : alias for groupwise_numeric_median_plus_indicator
    time_series_ffill_bfill_plus_indicator   : forward/backward fill + indicator (temporal domain)
    categorical_missing_token                : fill with "MISSING"
    categorical_missing_token_plus_indicator : fill with "MISSING" + indicator
    categorical_mode_plus_indicator          : fill with mode + indicator (legacy)
    structural_none_token_plus_indicator     : fill categorical with "NONE" + indicator
    structural_zero_plus_indicator           : fill numeric with 0 + indicator
    structural_none_or_zero                  : legacy alias (auto-dispatches by dtype)
    model_based_imputation_optional          : falls back to median
    """

    def apply(
        self,
        df_train: pd.DataFrame,
        imputation_plan: dict,
        df_predict: pd.DataFrame | None = None,
    ) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
        df_train = df_train.copy()
        if df_predict is not None:
            df_predict = df_predict.copy()

        drop_cols: list[str] = []

        for col, entry in imputation_plan.get("columns", {}).items():
            strategy = entry["strategy"]
            add_ind = entry["add_missing_indicator"]

            if strategy == "no_imputation_needed":
                continue

            if strategy == "drop_column":
                drop_cols.append(col)
                continue

            if col not in df_train.columns:
                continue

            # Add indicator BEFORE imputing so it captures true nulls
            if add_ind:
                df_train = _add_indicator(df_train, col)
                if df_predict is not None and col in df_predict.columns:
                    df_predict = _add_indicator(df_predict, col)

            if strategy in ("groupwise_numeric_median_plus_indicator", "group_median"):
                group_col = entry.get("group_col")
                global_median = df_train[col].median()    # fit on train
                if group_col and group_col in df_train.columns:
                    group_map = df_train.groupby(group_col)[col].median().to_dict()
                    df_train[col] = df_train[col].fillna(
                        df_train[group_col].map(group_map)
                    )
                    if df_predict is not None and col in df_predict.columns and group_col in df_predict.columns:
                        df_predict[col] = df_predict[col].fillna(
                            df_predict[group_col].map(group_map)
                        )
                # Fall back to global median for rows where group is unknown/NA
                df_train[col] = df_train[col].fillna(global_median)
                if df_predict is not None and col in df_predict.columns:
                    df_predict[col] = df_predict[col].fillna(global_median)

            elif strategy == "time_series_ffill_bfill_plus_indicator":
                # Temporal domain: neighbouring rows carry more signal than global median.
                # Train: ffill then bfill in row order (assumes data is sorted by time).
                # Predict: ffill from predict data then backfill using last train value.
                df_train[col] = df_train[col].ffill().bfill()
                # Any remaining NaN (all-null column) → global median fallback
                fallback = df_train[col].median()
                df_train[col] = df_train[col].fillna(fallback)
                if df_predict is not None and col in df_predict.columns:
                    df_predict[col] = df_predict[col].ffill().bfill().fillna(fallback)

            elif strategy in ("numeric_median", "numeric_median_plus_indicator"):
                fill = df_train[col].median()              # fit on train
                df_train[col] = df_train[col].fillna(fill)
                if df_predict is not None and col in df_predict.columns:
                    df_predict[col] = df_predict[col].fillna(fill)

            elif strategy in ("categorical_missing_token", "categorical_missing_token_plus_indicator"):
                df_train[col] = df_train[col].fillna("MISSING")
                if df_predict is not None and col in df_predict.columns:
                    df_predict[col] = df_predict[col].fillna("MISSING")

            elif strategy == "categorical_mode_plus_indicator":
                mode_vals = df_train[col].mode()
                fill = mode_vals.iloc[0] if len(mode_vals) > 0 else "MISSING"  # fit on train
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
                # Legacy strategy: dispatch by dtype
                fill = 0 if pd.api.types.is_numeric_dtype(df_train[col]) else "NONE"
                df_train[col] = df_train[col].fillna(fill)
                if df_predict is not None and col in df_predict.columns:
                    df_predict[col] = df_predict[col].fillna(fill)

            elif strategy == "model_based_imputation_optional":
                # Falls back to median; MICE via IterativeImputer is out of scope
                fill = df_train[col].median()              # fit on train
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
