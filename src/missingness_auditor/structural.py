"""
Structural missingness detection.

Looks for patterns where a categorical column being NaN signals the
absence of a facility or item, and a companion numeric column is
correspondingly zero or also missing.

This detector identifies only TRUE structural absence:
  - categorical quality/type/status column is NaN
  - companion numeric amount/count/area is 0 or NaN for the same rows

Group-dependent missingness (a numeric column always missing for one
specific category GROUP VALUE, e.g. weather data missing by jurisdiction)
is NOT structural absence — it is handled by the mechanism detector.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

_NA_DOMINANCE_THRESHOLD = 0.85
_ZERO_DOMINANCE_THRESHOLD = 0.85
_MEAN_RATIO_THRESHOLD = 0.10
_MIN_NA_ROWS = 5
_MIN_ZERO_LIFT = 0.40


class StructuralMissingnessDetector:
    """
    Detects structural missingness patterns between column pairs.

    Only Pattern 1 and Pattern 2 are classified as structural absence:
      1. When a categorical column is NaN, a numeric companion is mostly NaN too
         (both absent together — e.g. no parking facility → no parking spots).
      2. When a categorical column is NaN, a numeric companion is mostly 0
         (absence maps to zero — e.g. no pool type → pool_area = 0).

    Pattern 3 (numeric column always missing for a specific category group value)
    is group-dependent missingness, NOT structural absence, and is excluded here.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self._cat_with_missing = [
            c for c in df.columns
            if not pd.api.types.is_numeric_dtype(df[c]) and df[c].isna().any()
        ]
        self._num_cols = [
            c for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c])
        ]

    def detect(self) -> dict:
        structural_pairs: list[dict] = []
        column_flags: dict[str, dict] = {}
        seen_pairs: set[tuple[str, str]] = set()

        self._check_cat_na_patterns(structural_pairs, column_flags, seen_pairs)

        return {
            "structural_pairs": structural_pairs,
            "column_flags": column_flags,
            "n_structural_pairs": len(structural_pairs),
        }

    def _check_cat_na_patterns(
        self, pairs: list, flags: dict, seen_pairs: set
    ) -> None:
        for cat_col in self._cat_with_missing:
            cat_na = self.df[cat_col].isna()
            if cat_na.sum() < _MIN_NA_ROWS:
                continue

            for num_col in self._num_cols:
                pair_key = (cat_col, num_col)
                if pair_key in seen_pairs:
                    continue

                num_when_na = self.df.loc[cat_na, num_col]
                num_when_present = self.df.loc[~cat_na, num_col]

                if len(num_when_na) == 0:
                    continue

                # Pattern 1: numeric is mostly NaN when cat is NA.
                # This is co-missing (both absent together), NOT zero-valued absence.
                # Flag with "co_missing_structural" so the planner uses median/group
                # imputation rather than zero-fill.
                na_frac = float(num_when_na.isna().mean())
                if na_frac >= _NA_DOMINANCE_THRESHOLD:
                    seen_pairs.add(pair_key)
                    pairs.append({
                        "categorical_col": cat_col,
                        "numeric_col": num_col,
                        "pattern": "numeric_all_missing_when_cat_na",
                        "evidence": "companion_numeric_absent_when_categorical_na",
                        "na_fraction_when_cat_na": round(na_frac, 4),
                        "zero_fraction_when_cat_na": None,
                    })
                    flags.setdefault(cat_col, {"is_structural": True, "pattern": "co_missing_structural"})
                    flags.setdefault(num_col, {"is_structural": True, "pattern": "co_missing_structural"})
                    continue

                # Pattern 2: numeric is mostly 0 when cat is NA (with lift check)
                filled = num_when_na.fillna(0)
                zero_frac = float((filled == 0).mean())

                zero_frac_when_present = float(
                    (num_when_present.fillna(0) == 0).mean()
                ) if len(num_when_present) > 0 else 0.0
                zero_lift = zero_frac - zero_frac_when_present

                # Require zeros to be disproportionately concentrated when the
                # categorical is NA (lift check). Prevents flagging globally-common
                # zeros like precipitation=0 which appear across all groups.
                if zero_lift < _MIN_ZERO_LIFT:
                    continue

                mean_na = float(num_when_na.abs().mean()) if num_when_na.notna().any() else 0.0
                mean_present = float(num_when_present.abs().mean()) if num_when_present.notna().any() else 0.0

                ratio_low = (
                    mean_present > 0
                    and pd.notna(mean_na)
                    and mean_na / mean_present < _MEAN_RATIO_THRESHOLD
                )
                if zero_frac >= _ZERO_DOMINANCE_THRESHOLD or ratio_low:
                    seen_pairs.add(pair_key)
                    pairs.append({
                        "categorical_col": cat_col,
                        "numeric_col": num_col,
                        "pattern": "numeric_near_zero_when_cat_na",
                        "evidence": "companion_numeric_zero_when_categorical_na",
                        "na_fraction_when_cat_na": round(float(num_when_na.isna().mean()), 4),
                        "zero_fraction_when_cat_na": round(zero_frac, 4),
                        "zero_lift_over_baseline": round(zero_lift, 4),
                    })
                    flags.setdefault(cat_col, {"is_structural": True, "pattern": "structural_absence"})
                    flags.setdefault(num_col, {"is_structural": True, "pattern": "structural_absence"})
