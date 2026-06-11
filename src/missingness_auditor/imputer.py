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

    When llm_client is provided, after applying imputation the attribute
    llm_distribution_summary is populated with a natural language before/after
    comparison narrative.

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
    model_based_imputation_optional          : leakage-safe IterativeImputer (MICE engine)
    mice_multiple_imputation                 : alias for model_based_imputation_optional
    """

    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.llm_distribution_summary: str | None = None
        # Lazily-fitted IterativeImputer shared across all model-based columns.
        # Fitted on df_train only (leakage-safe); cached so we fit at most once.
        self._mice_model = None
        self._mice_numeric_cols: list[str] = []

    def apply(
        self,
        df_train: pd.DataFrame,
        imputation_plan: dict,
        df_predict: pd.DataFrame | None = None,
    ) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
        df_train = df_train.copy()
        if df_predict is not None:
            df_predict = df_predict.copy()

        # Snapshot pre-imputation series for LLM comparison
        _pre_snap: dict[str, pd.Series] = {}
        if self.llm_client:
            for col, entry in imputation_plan.get("columns", {}).items():
                if entry["strategy"] not in ("no_imputation_needed", "drop_column") and col in df_train.columns:
                    _pre_snap[col] = df_train[col].copy()

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
                if pd.isna(global_median):                # all-null column guard
                    global_median = 0.0
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
                if pd.isna(fallback):                      # all-null column guard
                    fallback = 0.0
                df_train[col] = df_train[col].fillna(fallback)
                if df_predict is not None and col in df_predict.columns:
                    df_predict[col] = df_predict[col].ffill().bfill().fillna(fallback)

            elif strategy in ("numeric_median", "numeric_median_plus_indicator"):
                fill = df_train[col].median()              # fit on train
                if pd.isna(fill):                          # all-null column guard
                    fill = 0.0
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

            elif strategy in ("model_based_imputation_optional", "mice_multiple_imputation"):
                # Leakage-safe model-based single imputation: one IterativeImputer
                # (the MICE engine) fitted on df_train's numeric columns predicts the
                # missing cells from the other features. Far better than median when
                # the column is MAR-correlated with observed covariates. Falls back to
                # median if the column is non-numeric or the model cannot be fitted.
                df_train, df_predict = self._model_based_fill(
                    col, df_train, df_predict
                )

        if drop_cols:
            df_train = df_train.drop(columns=drop_cols, errors="ignore")
            if df_predict is not None:
                df_predict = df_predict.drop(columns=drop_cols, errors="ignore")

        if self.llm_client and _pre_snap:
            self._generate_distribution_summary(_pre_snap, df_train, drop_cols)

        if df_predict is not None:
            return df_train, df_predict
        return df_train

    def _model_based_fill(
        self,
        col: str,
        df_train: pd.DataFrame,
        df_predict: pd.DataFrame | None,
    ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        """Fill a single numeric column with a leakage-safe IterativeImputer.

        The imputer is fitted once on df_train's numeric columns and cached, then
        reused for every model-based column. Only ``col`` is written back from the
        model output, so other columns keep their plan-specific strategies. Falls
        back to the training-set median when the column is non-numeric or there are
        too few predictor columns to fit a model.
        """
        if not pd.api.types.is_numeric_dtype(df_train[col]):
            fill = df_train[col].median()
            df_train[col] = df_train[col].fillna(fill)
            if df_predict is not None and col in df_predict.columns:
                df_predict[col] = df_predict[col].fillna(fill)
            return df_train, df_predict

        if self._mice_model is None:
            numeric_cols = [
                c for c in df_train.columns
                if pd.api.types.is_numeric_dtype(df_train[c])
                and df_train[c].notna().any()
            ]
            # Need at least one predictor besides the target column to model from.
            if len(numeric_cols) < 2:
                fill = df_train[col].median()
                df_train[col] = df_train[col].fillna(fill)
                if df_predict is not None and col in df_predict.columns:
                    df_predict[col] = df_predict[col].fillna(fill)
                return df_train, df_predict
            from sklearn.experimental import enable_iterative_imputer  # noqa: F401
            from sklearn.impute import IterativeImputer

            model = IterativeImputer(max_iter=10, random_state=0)
            model.fit(df_train[numeric_cols].to_numpy(dtype=float))  # train only
            self._mice_model = model
            self._mice_numeric_cols = numeric_cols

        cols = self._mice_numeric_cols
        if col not in cols:
            fill = df_train[col].median()
            df_train[col] = df_train[col].fillna(fill)
            if df_predict is not None and col in df_predict.columns:
                df_predict[col] = df_predict[col].fillna(fill)
            return df_train, df_predict

        ci = cols.index(col)
        filled_train = self._mice_model.transform(df_train[cols].to_numpy(dtype=float))
        df_train[col] = filled_train[:, ci]
        if df_predict is not None and all(c in df_predict.columns for c in cols):
            filled_pred = self._mice_model.transform(df_predict[cols].to_numpy(dtype=float))
            df_predict[col] = filled_pred[:, ci]
        elif df_predict is not None and col in df_predict.columns:
            # Predict frame lacks some predictor columns — safe median fallback.
            df_predict[col] = df_predict[col].fillna(df_train[col].median())
        return df_train, df_predict

    def _generate_distribution_summary(
        self,
        pre_snap: dict[str, pd.Series],
        df_after: pd.DataFrame,
        dropped_cols: list[str],
    ) -> None:
        """Call LLM to narrate the before/after distribution changes."""
        from ._llm import call_llm_text

        col_stats = []
        for col, before in pre_snap.items():
            n_missing_before = int(before.isna().sum())
            if col not in df_after.columns:
                col_stats.append(f"- {col}: DROPPED ({n_missing_before} missing values removed)")
                continue
            after = df_after[col]
            n_missing_after = int(after.isna().sum())
            if pd.api.types.is_numeric_dtype(before):
                col_stats.append(
                    f"- {col} (numeric): missing {n_missing_before}→{n_missing_after}, "
                    f"mean {before.mean():.3g}→{after.mean():.3g}, "
                    f"std {before.std():.3g}→{after.std():.3g}, "
                    f"median {before.median():.3g}→{after.median():.3g}"
                )
            else:
                top_before = before.value_counts().head(3).to_dict()
                top_after = after.value_counts().head(3).to_dict()
                col_stats.append(
                    f"- {col} (categorical): missing {n_missing_before}→{n_missing_after}, "
                    f"top_before={top_before}, top_after={top_after}"
                )

        if dropped_cols:
            for c in dropped_cols:
                if c not in pre_snap:
                    col_stats.append(f"- {c}: DROPPED (>80% missing)")

        prompt = (
            "You are reviewing imputation results. Summarize what changed in 3-5 sentences:\n"
            "- Were distribution shapes preserved or distorted?\n"
            "- Any columns where the fill caused a noticeable shift in mean or introduced a dominant category?\n"
            "- Any distribution change the analyst should verify before training?\n\n"
            "Per-column stats (before → after):\n" + "\n".join(col_stats)
        )
        self.llm_distribution_summary = call_llm_text(self.llm_client, prompt, max_tokens=350)
