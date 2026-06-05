"""Feature type inference engine.

Classifies each column into one of:
  target, row_id, datetime_like, numeric_continuous, numeric_discrete,
  categorical_low_cardinality, categorical_high_cardinality, text_like,
  boolean_binary, group_entity_id, train_only, prediction_only,
  constant, near_constant, leakage_risk
"""

from __future__ import annotations

import re
import pandas as pd
import numpy as np

_DATETIME_PATTERNS = re.compile(
    r"(date|time|timestamp|dt|year|month|day|hour|minute|second|week|quarter)",
    re.IGNORECASE,
)

_ID_PATTERNS = re.compile(
    r"(_id|_key|_code|_num|^id$|_uuid|_hash)", re.IGNORECASE
)

_TEXT_PATTERNS = re.compile(
    r"(description|comment|note|text|title|name|message|summary|body)",
    re.IGNORECASE,
)

_GROUP_PATTERNS = re.compile(
    r"(group|entity|site|location|station|user|customer|store|product|region|zone|area|district)",
    re.IGNORECASE,
)

# Thresholds
_BOOL_THRESH = 2
_LOW_CARD_THRESH = 20
_HIGH_CARD_THRESH = 0.5   # fraction of rows
_NEAR_CONST_THRESH = 0.99  # top value covers >= 99% of rows
_DISCRETE_MAX_UNIQUE = 30  # numeric unique count below this → discrete
_AVG_WORDS_THRESH = 3.0    # avg words/cell above this → text_like


class FeatureTypeInferrer:
    def __init__(
        self,
        train_df: pd.DataFrame,
        predict_df: pd.DataFrame | None = None,
        *,
        target_col: str | None = None,
        row_id_col: str | None = None,
        datetime_col: str | None = None,
        group_col: str | None = None,
    ):
        self.train_df = train_df
        self.predict_df = predict_df
        self.target_col = target_col
        self.row_id_col = row_id_col
        self.datetime_col = datetime_col
        self.group_col = group_col

        if predict_df is not None:
            train_cols = set(train_df.columns)
            pred_cols = set(predict_df.columns)
            self._train_only_cols = train_cols - pred_cols
            self._predict_only_cols = pred_cols - train_cols
        else:
            self._train_only_cols = set()
            self._predict_only_cols = set()

    def infer(self) -> dict:
        all_cols = list(self.train_df.columns)
        if self.predict_df is not None:
            all_cols = list(
                dict.fromkeys(all_cols + list(self.predict_df.columns))
            )

        columns: dict = {}
        for col in all_cols:
            if col in self.train_df.columns:
                series = self.train_df[col]
            else:
                series = self.predict_df[col]  # type: ignore[index]
            columns[col] = self._classify(col, series)

        # Summary counts per type
        type_counts: dict[str, int] = {}
        for info in columns.values():
            t = info["inferred_type"]
            type_counts[t] = type_counts.get(t, 0) + 1

        return {"columns": columns, "type_summary": type_counts}

    def _classify(self, col: str, series: pd.Series) -> dict:
        n = len(series)
        null_rate = series.isna().mean()
        n_unique = series.nunique(dropna=True)

        reasons: list[str] = []
        inferred: str

        # --- Explicit hints from caller ---
        if col == self.target_col:
            return {"inferred_type": "target", "null_rate": null_rate, "n_unique": n_unique, "reasons": ["explicitly provided as target"]}
        if col == self.row_id_col:
            return {"inferred_type": "row_id", "null_rate": null_rate, "n_unique": n_unique, "reasons": ["explicitly provided as row_id"]}
        if col == self.datetime_col:
            return {"inferred_type": "datetime_like", "null_rate": null_rate, "n_unique": n_unique, "reasons": ["explicitly provided as datetime_col"]}
        if col == self.group_col:
            return {"inferred_type": "group_entity_id", "null_rate": null_rate, "n_unique": n_unique, "reasons": ["explicitly provided as group_col"]}

        # --- Train/predict-only ---
        if col in self._train_only_cols:
            inferred = "train_only"
            reasons.append("present only in train")
            return {"inferred_type": inferred, "null_rate": null_rate, "n_unique": n_unique, "reasons": reasons}
        if col in self._predict_only_cols:
            inferred = "prediction_only"
            reasons.append("present only in predict")
            return {"inferred_type": inferred, "null_rate": null_rate, "n_unique": n_unique, "reasons": reasons}

        dtype = series.dtype

        # --- Text-like check must precede constant for object columns ---
        # A column of identical long strings is text_like, not constant.
        if dtype == object or pd.api.types.is_string_dtype(dtype):
            if _is_text_like(series):
                return {"inferred_type": "text_like", "null_rate": null_rate, "n_unique": n_unique, "reasons": ["high average word count suggests free text"]}

        # --- Constant / near-constant ---
        if n_unique <= 1:
            return {"inferred_type": "constant", "null_rate": null_rate, "n_unique": n_unique, "reasons": ["single unique value"]}
        if n > 0:
            top_freq = series.value_counts(dropna=False).iloc[0] / n
            if top_freq >= _NEAR_CONST_THRESH:
                return {"inferred_type": "near_constant", "null_rate": null_rate, "n_unique": n_unique, "reasons": [f"top value covers {top_freq:.1%}"]}

        # --- Datetime ---
        if _is_datetime(series, col):
            return {"inferred_type": "datetime_like", "null_rate": null_rate, "n_unique": n_unique, "reasons": _datetime_reasons(series, col)}

        # --- Boolean/binary ---
        if n_unique == _BOOL_THRESH:
            non_null = series.dropna()
            unique_vals = set(non_null.unique())
            binary_sets = [
                {0, 1}, {True, False}, {"0", "1"}, {"yes", "no"},
                {"true", "false"}, {"Y", "N"}, {"y", "n"},
            ]
            if any(unique_vals == s or {str(v).lower() for v in unique_vals} == {str(v).lower() for v in s} for s in binary_sets):
                return {"inferred_type": "boolean_binary", "null_rate": null_rate, "n_unique": n_unique, "reasons": ["2 unique values matching boolean pattern"]}

        # --- Numeric ---
        if pd.api.types.is_numeric_dtype(dtype):
            # Check if it looks like a group/entity ID
            if _ID_PATTERNS.search(col) and n_unique > _LOW_CARD_THRESH:
                reasons.append("numeric but name matches ID pattern")
                inferred = "group_entity_id"
                return {"inferred_type": inferred, "null_rate": null_rate, "n_unique": n_unique, "reasons": reasons}

            if n_unique <= _DISCRETE_MAX_UNIQUE:
                inferred = "numeric_discrete"
                reasons.append(f"numeric with only {n_unique} unique values")
            else:
                # Check if values are all integers even if float dtype
                non_null = series.dropna()
                if len(non_null) > 0 and np.all(non_null == non_null.astype(int)):
                    if n_unique <= _DISCRETE_MAX_UNIQUE * 2:
                        inferred = "numeric_discrete"
                        reasons.append("all integer values with moderate cardinality")
                    else:
                        inferred = "numeric_continuous"
                        reasons.append("numeric, all integer but high cardinality")
                else:
                    inferred = "numeric_continuous"
                    reasons.append("numeric float with high cardinality")
            return {"inferred_type": inferred, "null_rate": null_rate, "n_unique": n_unique, "reasons": reasons}

        # --- String/object columns ---
        if dtype == object or pd.api.types.is_string_dtype(dtype):
            # Try to parse as datetime
            sample = series.dropna().head(100)

            # Text-like check first
            if _is_text_like(series):
                inferred = "text_like"
                reasons.append("high average word count suggests free text")
                return {"inferred_type": inferred, "null_rate": null_rate, "n_unique": n_unique, "reasons": reasons}

            # Group/entity name check
            if _GROUP_PATTERNS.search(col) and n_unique > _BOOL_THRESH:
                inferred = "group_entity_id"
                reasons.append("column name suggests group/entity")
                return {"inferred_type": inferred, "null_rate": null_rate, "n_unique": n_unique, "reasons": reasons}

            # Cardinality-based categorical
            card_ratio = n_unique / max(n, 1)
            if n_unique <= _LOW_CARD_THRESH:
                inferred = "categorical_low_cardinality"
                reasons.append(f"string with {n_unique} unique values (<= {_LOW_CARD_THRESH})")
            elif card_ratio >= _HIGH_CARD_THRESH:
                inferred = "categorical_high_cardinality"
                reasons.append(f"string with high cardinality ratio {card_ratio:.2f}")
            else:
                inferred = "categorical_low_cardinality"
                reasons.append(f"string with {n_unique} unique values")

            # Upgrade to group_entity_id if ID-like name
            if _ID_PATTERNS.search(col) and inferred == "categorical_high_cardinality":
                inferred = "group_entity_id"
                reasons.append("column name matches ID pattern")

            return {"inferred_type": inferred, "null_rate": null_rate, "n_unique": n_unique, "reasons": reasons}

        # --- Category dtype ---
        if hasattr(dtype, "name") and dtype.name == "category":
            n_cats = len(series.cat.categories)
            inferred = "categorical_low_cardinality" if n_cats <= _LOW_CARD_THRESH else "categorical_high_cardinality"
            return {"inferred_type": inferred, "null_rate": null_rate, "n_unique": n_unique, "reasons": ["pandas category dtype"]}

        return {"inferred_type": "unknown", "null_rate": null_rate, "n_unique": n_unique, "reasons": ["unrecognised dtype"]}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_datetime(series: pd.Series, col: str) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if _DATETIME_PATTERNS.search(col):
        # Try to parse a sample
        sample = series.dropna().head(50)
        if len(sample) == 0:
            return True
        if pd.api.types.is_numeric_dtype(series):
            # Could be unix timestamp; check range
            med = float(series.dropna().median())
            # Unix timestamps for years 2000-2100 roughly 946_684_800 – 4_102_444_800
            if 946_684_800 <= med <= 4_102_444_800:
                return True
        try:
            pd.to_datetime(sample, infer_datetime_format=True)
            return True
        except Exception:
            return False
    if series.dtype == object:
        sample = series.dropna().head(30)
        try:
            converted = pd.to_datetime(sample, infer_datetime_format=True)
            return converted.notna().mean() > 0.8
        except Exception:
            return False
    return False


def _datetime_reasons(series: pd.Series, col: str) -> list[str]:
    reasons = []
    if pd.api.types.is_datetime64_any_dtype(series):
        reasons.append("datetime64 dtype")
    if _DATETIME_PATTERNS.search(col):
        reasons.append("column name contains datetime keyword")
    return reasons or ["parsed as datetime"]


def _is_text_like(series: pd.Series) -> bool:
    sample = series.dropna().head(200).astype(str)
    if len(sample) == 0:
        return False
    avg_words = sample.str.split().str.len().mean()
    return float(avg_words) >= _AVG_WORDS_THRESH
