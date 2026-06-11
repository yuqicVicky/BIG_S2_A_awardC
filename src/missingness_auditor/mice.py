"""
Multiple Imputation by Chained Equations (MICE) with Rubin's Rules pooling.

This module turns the report's standing recommendation ("upgrade to full MICE for
inference") into an actual, runnable capability. Single imputation (median/token)
fills the holes but treats the filled values as if they were observed — it
understates variance and produces confidence intervals that are too narrow
(van Buuren FIMD Ch1, Table 1.1). Multiple imputation instead draws `m` plausible
completions, computes the estimand on each, and pools them with Rubin's rules so the
extra uncertainty from missingness is propagated into the standard error.

Leakage-safe by construction
-----------------------------
The IterativeImputer is fitted on training data only. The same fitted imputers are
applied (``.transform``) to predict data. The target column is included as a
*predictor* in the imputation model (statistically correct — van Buuren FIMD Ch5),
but its own values are never overwritten: they are snapshot and restored after each
draw, honouring the skill's hard guard "target column is never imputed".

References
----------
Rubin, D.B. (1987). Multiple Imputation for Nonresponse in Surveys.
van Buuren, S. Flexible Imputation of Missing Data (FIMD), Ch 2 & Ch 5.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

# IterativeImputer is still behind the experimental flag in scikit-learn.
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge


def pool_rubin(estimates, variances) -> dict:
    """
    Pool ``m`` point estimates and their within-imputation variances with
    Rubin's (1987) rules.

    Parameters
    ----------
    estimates : sequence of float
        The estimand (e.g. a column mean or a regression coefficient) computed on
        each of the ``m`` completed datasets, ``Q_i``.
    variances : sequence of float
        The estimated variance of each ``Q_i`` (e.g. ``var/n`` for a mean), ``U_i``.

    Returns
    -------
    dict with the pooled quantities:
        m                : number of imputations
        pooled_estimate  : Q-bar, the average of the estimates
        within_variance  : U-bar, mean within-imputation variance
        between_variance : B, variance *between* the m estimates (ddof=1)
        total_variance   : T = U-bar + (1 + 1/m) * B
        std_error        : sqrt(T)
        df               : Rubin-Barnard-Rubin degrees of freedom
        relative_increase_variance : r = (1 + 1/m) * B / U-bar
        fraction_missing_information : FMI, share of information lost to missingness
        ci_95            : [lower, upper] using a t-distribution with `df`
    """
    est = np.asarray(list(estimates), dtype=float)
    var = np.asarray(list(variances), dtype=float)
    m = int(est.size)
    if m == 0:
        raise ValueError("pool_rubin requires at least one estimate")

    q_bar = float(np.mean(est))
    u_bar = float(np.mean(var))
    # Between-imputation variance: with a single imputation there is no spread.
    b = float(np.var(est, ddof=1)) if m > 1 else 0.0
    total = u_bar + (1.0 + 1.0 / m) * b
    std_error = math.sqrt(total) if total > 0 else 0.0

    # Relative increase in variance due to nonresponse and fraction of missing info.
    if u_bar > 0:
        r = (1.0 + 1.0 / m) * b / u_bar
    else:
        r = float("inf") if b > 0 else 0.0

    if m > 1 and r > 0 and math.isfinite(r):
        df = (m - 1) * (1.0 + 1.0 / r) ** 2
        fmi = (r + 2.0 / (df + 3.0)) / (r + 1.0)
    else:
        df = float("inf")
        fmi = 0.0

    # 95% interval. Use a normal quantile when df is large/infinite, otherwise t.
    if math.isfinite(df) and df < 1000:
        try:
            from scipy import stats  # optional; only used for the t-quantile

            tcrit = float(stats.t.ppf(0.975, df))
        except Exception:
            tcrit = 1.96
    else:
        tcrit = 1.96
    ci = [q_bar - tcrit * std_error, q_bar + tcrit * std_error]

    return {
        "m": m,
        "pooled_estimate": q_bar,
        "within_variance": u_bar,
        "between_variance": b,
        "total_variance": total,
        "std_error": std_error,
        "df": df if math.isfinite(df) else None,
        "relative_increase_variance": r if math.isfinite(r) else None,
        "fraction_missing_information": fmi,
        "ci_95": ci,
    }


class MICEImputer:
    """
    Leakage-safe multiple imputation over the numeric columns of a DataFrame.

    ``m`` independent :class:`~sklearn.impute.IterativeImputer` instances are fitted
    on the training data with ``sample_posterior=True`` and distinct random seeds, so
    each ``transform`` draws a different plausible completion. Non-numeric columns are
    passed through untouched. The target column (if numeric) participates as a
    predictor but is never imputed — its observed values are restored after each draw.
    """

    def __init__(self, m: int = 5, max_iter: int = 10, random_state: int = 0):
        self.m = int(m)
        self.max_iter = int(max_iter)
        self.random_state = int(random_state)
        self._imputers: list[IterativeImputer] = []
        self._numeric_cols: list[str] = []
        self._other_cols: list[str] = []
        self._target_col: str | None = None
        self._fitted = False

    def fit(self, df_train: pd.DataFrame, target_col: str | None = None) -> "MICEImputer":
        self._numeric_cols = [
            c for c in df_train.columns
            if pd.api.types.is_numeric_dtype(df_train[c])
        ]
        self._other_cols = [c for c in df_train.columns if c not in self._numeric_cols]
        self._target_col = target_col if target_col in self._numeric_cols else None

        if not self._numeric_cols:
            raise ValueError("MICEImputer needs at least one numeric column")

        x = df_train[self._numeric_cols].to_numpy(dtype=float)
        self._imputers = []
        for i in range(self.m):
            imp = IterativeImputer(
                estimator=BayesianRidge(),
                sample_posterior=True,
                max_iter=self.max_iter,
                random_state=self.random_state + i,
            )
            imp.fit(x)  # fit on TRAIN only
            self._imputers.append(imp)
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> list[pd.DataFrame]:
        """Return ``m`` completed copies of ``df`` (target values preserved)."""
        if not self._fitted:
            raise RuntimeError("call fit() before transform()")
        x = df[self._numeric_cols].to_numpy(dtype=float)
        completed: list[pd.DataFrame] = []
        for imp in self._imputers:
            filled = imp.transform(x)
            out = df.copy()
            out[self._numeric_cols] = filled
            # Hard guard: never overwrite the target column's own values.
            if self._target_col is not None and self._target_col in df.columns:
                out[self._target_col] = df[self._target_col].to_numpy()
            completed.append(out)
        return completed

    def fit_transform(
        self,
        df_train: pd.DataFrame,
        df_predict: pd.DataFrame | None = None,
        target_col: str | None = None,
    ):
        self.fit(df_train, target_col=target_col)
        train_sets = self.transform(df_train)
        if df_predict is None:
            return train_sets
        return train_sets, self.transform(df_predict)


def mice_pool_column_means(
    df_train: pd.DataFrame,
    target_col: str | None = None,
    columns: list[str] | None = None,
    m: int = 5,
    random_state: int = 0,
) -> dict:
    """
    Run MICE on ``df_train`` and pool the **mean of each imputed column** with
    Rubin's rules. The column mean is a simple, universally meaningful estimand that
    makes the value of multiple imputation concrete: the reported ``std_error``
    reflects both sampling variability *and* the extra uncertainty from missingness,
    which single imputation cannot express.

    Returns a JSON-serialisable dict suitable for ``logs/mice_pooling.json``.
    """
    numeric_cols = [
        c for c in df_train.columns if pd.api.types.is_numeric_dtype(df_train[c])
    ]
    # Columns worth pooling: numeric, have missing values, and are not the target.
    candidate = [
        c for c in numeric_cols
        if c != target_col and df_train[c].isna().any()
    ]
    if columns is not None:
        candidate = [c for c in candidate if c in columns]

    result: dict = {
        "method": "MICE (IterativeImputer + BayesianRidge, sample_posterior) "
                  "pooled with Rubin's rules",
        "m": int(m),
        "estimand": "per-column mean",
        "leakage_safe": "imputers fitted on training data only; target preserved",
        "columns": {},
        "note": (
            "Pooled std_error reflects sampling variability AND between-imputation "
            "uncertainty from missing data. fraction_missing_information (FMI) is the "
            "share of information about the estimand lost to missingness "
            "(Rubin 1987; van Buuren FIMD Ch2)."
        ),
    }
    if not candidate:
        result["note"] += " No numeric columns with missing values to pool."
        return result

    imputer = MICEImputer(m=m, random_state=random_state)
    completed = imputer.fit_transform(df_train, target_col=target_col)

    for col in candidate:
        estimates, variances = [], []
        for d in completed:
            series = d[col].astype(float)
            n = int(series.notna().sum())
            if n < 2:
                continue
            mean = float(series.mean())
            # Variance of the sample mean = s^2 / n.
            var_of_mean = float(series.var(ddof=1)) / n
            estimates.append(mean)
            variances.append(var_of_mean)
        if not estimates:
            continue
        pooled = pool_rubin(estimates, variances)
        pooled["missing_rate"] = float(df_train[col].isna().mean())
        # Naive single-imputation SE (treats one completed dataset as certain),
        # shown alongside to make the MI variance inflation visible.
        pooled["naive_single_imputation_std_error"] = math.sqrt(pooled["within_variance"])
        result["columns"][col] = pooled

    return result
