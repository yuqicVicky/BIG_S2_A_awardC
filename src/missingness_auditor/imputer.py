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

    All statistics (median, mode) are fitted on df_train only, then applied
    to both df_train and df_predict.  This prevents information from the
    test/predict set from leaking into imputation parameters.

    Usage
    -----
    imputed_train = Imputer().apply(df_train, plan)
    imputed_train, imputed_predict = Imputer().apply(df_train, plan, df_predict)
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

            if strategy == "group_median":
                group_col = entry.get("group_col")
                global_median = df_train[col].median()    # fit on train
                if group_col and group_col in df_train.columns:
                    # Build group → median map from train only (leakage-safe)
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

            elif strategy in ("numeric_median", "numeric_median_plus_indicator"):
                fill = df_train[col].median()              # fit on train
                df_train[col] = df_train[col].fillna(fill)
                if df_predict is not None and col in df_predict.columns:
                    df_predict[col] = df_predict[col].fillna(fill)

            elif strategy == "categorical_missing_token":
                df_train[col] = df_train[col].fillna("MISSING")
                if df_predict is not None and col in df_predict.columns:
                    df_predict[col] = df_predict[col].fillna("MISSING")

            elif strategy == "categorical_mode_plus_indicator":
                mode_vals = df_train[col].mode()
                fill = mode_vals.iloc[0] if len(mode_vals) > 0 else "MISSING"  # fit on train
                df_train[col] = df_train[col].fillna(fill)
                if df_predict is not None and col in df_predict.columns:
                    df_predict[col] = df_predict[col].fillna(fill)

            elif strategy == "structural_none_or_zero":
                fill = 0 if pd.api.types.is_numeric_dtype(df_train[col]) else "NONE"
                df_train[col] = df_train[col].fillna(fill)
                if df_predict is not None and col in df_predict.columns:
                    df_predict[col] = df_predict[col].fillna(fill)

            elif strategy == "model_based_imputation_optional":
                # Falls back to median; full MICE via sklearn IterativeImputer is out of scope
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
