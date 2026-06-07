"""
Structural missingness detection.

Looks for patterns where a categorical column being NaN signals the
absence of a facility or item, and a companion numeric column is
correspondingly zero or also missing.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

_NA_DOMINANCE_THRESHOLD = 0.85
_ZERO_DOMINANCE_THRESHOLD = 0.85
_MEAN_RATIO_THRESHOLD = 0.10
_MIN_NA_ROWS = 5
# Zeros when cat is NA must be at least this much more common than zeros when
# cat is present. Prevents flagging columns like precipitation where 0 is a
# legitimate value distributed across all categories (lift ≈ 0).
_MIN_ZERO_LIFT = 0.40


class StructuralMissingnessDetector:
    """
    Detects structural missingness patterns between column pairs:

    1. When a categorical column is NaN, a numeric companion is mostly NaN too
       (both absent together — e.g. no parking facility → no parking spots).
    2. When a categorical column is NaN, a numeric companion is mostly 0
       (absence maps to zero — e.g. no pool type → pool_area = 0).
    3. A numeric column is always NaN for one specific category group.
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

        self._check_cat_na_patterns(structural_pairs, column_flags)
        self._check_category_group_patterns(structural_pairs, column_flags)

        return {
            "structural_pairs": structural_pairs,
            "column_flags": column_flags,
            "n_structural_pairs": len(structural_pairs),
        }

    def _check_cat_na_patterns(
        self, pairs: list, flags: dict
    ) -> None:
        for cat_col in self._cat_with_missing:
            cat_na = self.df[cat_col].isna()
            if cat_na.sum() < _MIN_NA_ROWS:
                continue

            for num_col in self._num_cols:
                num_when_na = self.df.loc[cat_na, num_col]
                num_when_present = self.df.loc[~cat_na, num_col]

                if len(num_when_na) == 0:
                    continue

                # Pattern 1: numeric is mostly NaN when cat is NA
                na_frac = float(num_when_na.isna().mean())
                if na_frac >= _NA_DOMINANCE_THRESHOLD:
                    pairs.append({
                        "categorical_col": cat_col,
                        "numeric_col": num_col,
                        "pattern": "numeric_all_missing_when_cat_na",
                        "na_fraction_when_cat_na": round(na_frac, 4),
                        "zero_fraction_when_cat_na": None,
                    })
                    flags.setdefault(cat_col, {"is_structural": True, "pattern": "structural_absence"})
                    flags.setdefault(num_col, {"is_structural": True, "pattern": "structural_absence"})
                    continue

                # Pattern 2: numeric is mostly 0 when cat is NA
                filled = num_when_na.fillna(0)
                zero_frac = float((filled == 0).mean())
                mean_na = float(num_when_na.abs().mean()) if num_when_na.notna().any() else 0.0
                mean_present = float(num_when_present.abs().mean()) if num_when_present.notna().any() else 0.0

                # Require that zeros are disproportionately concentrated when the
                # categorical is NA, not just common globally.  For example,
                # precipitation=0 is a real meteorological value spread across all
                # states (lift ≈ 0), whereas facility_capacity=0 only appears when
                # facility_type is absent (lift ≈ 1).
                zero_frac_when_present = float(
                    (num_when_present.fillna(0) == 0).mean()
                ) if len(num_when_present) > 0 else 0.0
                zero_lift = zero_frac - zero_frac_when_present
                if zero_lift < _MIN_ZERO_LIFT:
                    continue

                ratio_low = (
                    mean_present > 0
                    and pd.notna(mean_na)
                    and mean_na / mean_present < _MEAN_RATIO_THRESHOLD
                )
                if zero_frac >= _ZERO_DOMINANCE_THRESHOLD or ratio_low:
                    pairs.append({
                        "categorical_col": cat_col,
                        "numeric_col": num_col,
                        "pattern": "numeric_near_zero_when_cat_na",
                        "na_fraction_when_cat_na": round(float(num_when_na.isna().mean()), 4),
                        "zero_fraction_when_cat_na": round(zero_frac, 4),
                    })
                    flags.setdefault(cat_col, {"is_structural": True, "pattern": "structural_absence"})
                    flags.setdefault(num_col, {"is_structural": True, "pattern": "structural_absence"})

    def _check_category_group_patterns(
        self, pairs: list, flags: dict
    ) -> None:
        cat_present_cols = [
            c for c in self.df.columns
            if not pd.api.types.is_numeric_dtype(self.df[c]) and self.df[c].notna().any()
        ]
        for cat_col in cat_present_cols:
            for num_col in self._num_cols:
                if num_col in flags:
                    continue
                overall_na = float(self.df[num_col].isna().mean())
                if overall_na >= 0.90:
                    continue
                for grp_val in self.df[cat_col].dropna().unique():
                    mask = self.df[cat_col] == grp_val
                    if mask.sum() < _MIN_NA_ROWS:
                        continue
                    grp_na = float(self.df.loc[mask, num_col].isna().mean())
                    if grp_na >= 0.99:
                        pairs.append({
                            "categorical_col": cat_col,
                            "numeric_col": num_col,
                            "pattern": f"numeric_always_missing_for_category",
                            "group_value": str(grp_val),
                            "na_fraction_for_group": 1.0,
                        })
                        flags.setdefault(
                            num_col,
                            {"is_structural": True, "pattern": "structural_category_group"},
                        )
