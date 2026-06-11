"""
Missingness mechanism clue detection.

Labels are cautious and evidence-based, not causal:
- "MCAR-compatible"                        : no significant correlation with observed features or target
- "MAR-like evidence"                      : missingness correlates with at least one observed numeric feature
- "group-dependent missingness"            : missingness concentrated in specific categorical groups;
                                             some groups have disproportionately high missing rates
- "target-associated missingness"          : missingness correlates with the target variable
- "high-cardinality text/category missingness": missingness in a high-cardinality text-like column
- "insufficient evidence"                  : too few rows to determine mechanism

Note: "structural absence concern" is determined by StructuralMissingnessDetector, not here.

MAR robustness rule (Collins et al., 2001 via van Buuren FIMD Ch5):
  When missing rate < 25% AND max observed correlation < 0.4, omitting a lurking variable
  from the imputation model has negligible effect on regression estimates. The MAR assumption
  is "likely robust" in that regime.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

_CORR_THRESHOLD_MAR = 0.15
_CORR_THRESHOLD_TARGET = 0.10
_GROUP_SPREAD_THRESHOLD = 0.20
_MIN_ROWS_FOR_ANALYSIS = 20
_HIGH_CARDINALITY_ABS = 50
_HIGH_CARDINALITY_RATIO = 0.20

# Collins et al. (2001) MAR robustness thresholds (via van Buuren FIMD Ch5)
_MAR_ROBUST_MAX_MISS_RATE = 0.25
_MAR_ROBUST_MAX_CORR = 0.40

# Significance level for the hypothesis tests that now drive the mechanism labels.
_ALPHA = 0.05


# ───────────────────────────── hypothesis tests ──────────────────────────────
# The mechanism labels are driven by significance tests, not bare correlation
# cut-offs. Each helper degrades gracefully (returns None) when scipy/sklearn are
# unavailable or the sample is too small, so the auditor never hard-fails — the
# caller falls back to the legacy correlation rule in that case.


def _logistic_lr_test(y: np.ndarray, X: np.ndarray) -> dict | None:
    """
    Likelihood-ratio test for whether missingness (binary ``y``) depends on the
    observed numeric covariates ``X`` — the standard MAR diagnostic
    (van Buuren FIMD Ch2: model the *response* indicator on the data).

    Fits a near-unpenalised logistic regression (full model) and compares its
    log-likelihood to an intercept-only null model. Returns
    ``{statistic, df, p_value, n_covariates}`` or ``None`` on failure.
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from scipy.stats import chi2
    except Exception:
        return None

    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] != y.shape[0] or X.shape[1] == 0:
        return None
    # Need both classes present and enough rows to estimate the coefficients.
    n_pos = int(y.sum())
    if n_pos < 2 or n_pos > len(y) - 2 or len(y) < _MIN_ROWS_FOR_ANALYSIS:
        return None
    # Drop covariates with no variance (would contribute nothing / break scaling).
    keep = X.std(axis=0) > 0
    X = X[:, keep]
    if X.shape[1] == 0:
        return None
    try:
        Xs = StandardScaler().fit_transform(X)
        # Large C ⇒ minimal regularisation, so the fit approximates the MLE the
        # likelihood-ratio test assumes.
        clf = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
        clf.fit(Xs, y)
        p_full = np.clip(clf.predict_proba(Xs)[:, 1], 1e-12, 1 - 1e-12)
        ll_full = float(np.sum(y * np.log(p_full) + (1 - y) * np.log(1 - p_full)))
        p0 = np.clip(y.mean(), 1e-12, 1 - 1e-12)
        ll_null = float(np.sum(y * np.log(p0) + (1 - y) * np.log(1 - p0)))
        stat = 2.0 * (ll_full - ll_null)
        df = int(Xs.shape[1])
        if stat < 0 or df < 1:
            return None
        p_value = float(chi2.sf(stat, df))
        return {"statistic": round(stat, 4), "df": df,
                "p_value": round(p_value, 6), "n_covariates": df}
    except Exception:
        return None


def _chi2_independence(miss_ind: pd.Series, group: pd.Series) -> dict | None:
    """χ² test of independence between a missingness indicator and a categorical
    grouping variable. Returns ``{statistic, df, p_value}`` or ``None``."""
    try:
        from scipy.stats import chi2_contingency
    except Exception:
        return None
    try:
        table = pd.crosstab(miss_ind, group)
        if table.shape[0] < 2 or table.shape[1] < 2:
            return None
        stat, p_value, dof, _ = chi2_contingency(table)
        return {"statistic": round(float(stat), 4), "df": int(dof),
                "p_value": round(float(p_value), 6)}
    except Exception:
        return None


def _little_mcar_test(df: pd.DataFrame) -> dict:
    """
    Little's (1988) χ² test for MCAR over the numeric columns of ``df``.

    Estimates the mean and covariance by EM under multivariate normality, then
    compares each missingness pattern's observed-variable means to those
    estimates. A small p-value is evidence *against* MCAR. Best-effort: returns
    ``{"applicable": False, "reason": ...}`` when it cannot be computed.
    """
    num = df.select_dtypes(include="number")
    if num.shape[1] < 2:
        return {"applicable": False, "reason": "fewer_than_two_numeric_columns"}
    if not num.isna().any().any():
        return {"applicable": False, "reason": "no_missing_values_in_numeric_columns"}
    X = num.to_numpy(dtype=float)
    n, p = X.shape
    if n < _MIN_ROWS_FOR_ANALYSIS:
        return {"applicable": False, "reason": "too_few_rows"}
    try:
        from scipy.stats import chi2
        mu, cov = _em_mean_cov(X)
        miss = np.isnan(X)
        d2 = 0.0
        df_total = 0
        # Group rows by missingness pattern; each pattern contributes to d² and df.
        patterns: dict[tuple, list[int]] = {}
        for i in range(n):
            patterns.setdefault(tuple(miss[i].tolist()), []).append(i)
        for key, idx in patterns.items():
            obs = ~np.array(key)
            k = int(obs.sum())
            if k == 0:
                continue
            Xj = X[np.ix_(idx, np.where(obs)[0])]
            xbar = np.nanmean(Xj, axis=0)
            mu_obs = mu[obs]
            cov_obs = cov[np.ix_(obs, obs)]
            diff = xbar - mu_obs
            inv = np.linalg.pinv(cov_obs)
            d2 += len(idx) * float(diff @ inv @ diff)
            df_total += k
        df_stat = df_total - p
        if df_stat < 1:
            return {"applicable": False, "reason": "insufficient_pattern_degrees_of_freedom"}
        p_value = float(chi2.sf(d2, df_stat))
        return {
            "applicable": True,
            "statistic": round(float(d2), 4),
            "df": int(df_stat),
            "p_value": round(p_value, 6),
            "n_numeric_cols": int(p),
            "n_patterns": len(patterns),
            "interpretation": (
                "p < 0.05 is evidence against MCAR (data not missing completely at "
                "random); MAR is then the recommended working assumption."
            ),
        }
    except Exception as exc:  # pragma: no cover - defensive
        return {"applicable": False, "reason": f"computation_failed: {type(exc).__name__}"}


def _em_mean_cov(X: np.ndarray, max_iter: int = 100, tol: float = 1e-5):
    """EM estimate of the mean and covariance of a multivariate normal with
    missing entries (``np.nan``). Used by Little's MCAR test."""
    n, p = X.shape
    mu = np.nanmean(X, axis=0)
    mu = np.where(np.isnan(mu), 0.0, mu)
    filled = np.where(np.isnan(X), mu, X)
    cov = np.cov(filled, rowvar=False)
    cov = np.atleast_2d(cov)
    cov += np.eye(p) * 1e-6  # ridge for numerical stability
    for _ in range(max_iter):
        mu_old = mu.copy()
        T1 = np.zeros(p)
        T2 = np.zeros((p, p))
        for i in range(n):
            row = X[i]
            mis = np.isnan(row)
            xi = row.copy()
            if mis.any():
                obs = ~mis
                cov_oo = cov[np.ix_(obs, obs)]
                cov_mo = cov[np.ix_(mis, obs)]
                inv_oo = np.linalg.pinv(cov_oo)
                xi[mis] = mu[mis] + cov_mo @ inv_oo @ (row[obs] - mu[obs])
                T2_corr = cov[np.ix_(mis, mis)] - cov_mo @ inv_oo @ cov_mo.T
                midx = np.where(mis)[0]
                T2[np.ix_(midx, midx)] += T2_corr
            T1 += xi
            T2 += np.outer(xi, xi)
        mu = T1 / n
        cov = T2 / n - np.outer(mu, mu)
        cov += np.eye(p) * 1e-6
        if np.max(np.abs(mu - mu_old)) < tol:
            break
    return mu, cov


class MechanismAuditor:
    """
    For each column with missing values, compute a missingness indicator and
    correlate it with observed features and the target.

    Labels are cautious statistical clues — not causal mechanism assignments.

    Priority order when multiple signals are present:
    1. group-dependent missingness (categorical group spread dominates)
    2. target-associated missingness (target correlation)
    3. MAR-like evidence (numeric feature correlation)
    4. MCAR-compatible (no significant correlation)

    When llm_client is provided, each column gets LLM-generated fields:
      llm_narrative   : natural language explanation of the detected mechanism
      llm_suggestion  : imputation recommendation based on the pattern
      llm_anomaly     : True when the pattern is complex or unusual
    """

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        target_col: str | None = None,
        llm_client=None,
        corr_threshold_mar: float = _CORR_THRESHOLD_MAR,
        corr_threshold_target: float = _CORR_THRESHOLD_TARGET,
        group_spread_threshold: float = _GROUP_SPREAD_THRESHOLD,
    ):
        self.df = df
        self.target_col = target_col
        self.llm_client = llm_client
        self.corr_threshold_mar = corr_threshold_mar
        self.corr_threshold_target = corr_threshold_target
        self.group_spread_threshold = group_spread_threshold
        self._numeric_feature_cols = [
            c for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c]) and c != target_col
        ]
        self._categorical_feature_cols = [
            c for c in df.columns
            if not pd.api.types.is_numeric_dtype(df[c])
            and c != target_col
            and df[c].notna().any()
        ]

    def audit(self) -> dict:
        missing_cols = [c for c in self.df.columns if self.df[c].isna().any()]
        results = {col: self._analyze(col) for col in missing_cols}
        if self.llm_client and results:
            self._enrich_with_llm(results)
        return {
            "columns": results,
            "little_mcar_test": _little_mcar_test(self.df),
            "note": (
                "Mechanism labels are statistical clues from observational data only. "
                "They do not imply a definitive causal missingness mechanism. "
                "MCAR/MAR/MNAR cannot be confirmed from observational data alone. "
                "MAR is the recommended default assumption (van Buuren FIMD Ch5)."
            ),
        }

    def _enrich_with_llm(self, results: dict) -> None:
        """Batch LLM call: natural language narrative, imputation suggestion, anomaly flag."""
        from ._llm import call_llm_json

        col_lines = []
        for col, info in results.items():
            miss_rate = float(self.df[col].isna().mean())
            top_feat = (
                info["correlated_features"][0]["feature"]
                if info["correlated_features"] else "none"
            )
            top_cat = (
                info["categorical_correlated_features"][0]["feature"]
                if info.get("categorical_correlated_features") else "none"
            )
            col_lines.append(
                f"- {col}: mechanism={info['mechanism_label']}, "
                f"missing_rate={miss_rate:.1%}, "
                f"target_signal={info['target_signal']}, "
                f"top_numeric_corr={top_feat}, "
                f"top_group_corr={top_cat}"
            )

        prompt = (
            "You are a senior data scientist interpreting missing data mechanisms. "
            "For each column below, provide:\n"
            "1. narrative: 1-2 sentences explaining what the mechanism label means "
            "   and what it implies for the downstream model.\n"
            "2. suggestion: one concise imputation action recommendation.\n"
            "3. anomaly: true if the pattern is complex, ambiguous, or has multiple "
            "   overlapping signals that warrant special attention.\n\n"
            "Columns:\n" + "\n".join(col_lines) + "\n\n"
            'Return JSON: {"col_name": {"narrative": "...", "suggestion": "...", "anomaly": true|false}, ...}'
        )

        enrichments = call_llm_json(self.llm_client, prompt, max_tokens=1200)
        if not isinstance(enrichments, dict):
            return
        for col, enrich in enrichments.items():
            if col in results and isinstance(enrich, dict):
                results[col]["llm_narrative"] = enrich.get("narrative", "")
                results[col]["llm_suggestion"] = enrich.get("suggestion", "")
                results[col]["llm_anomaly"] = bool(enrich.get("anomaly", False))

    def _analyze(self, col: str) -> dict:
        miss_ind = self.df[col].isna().astype(int)
        n_obs = int(miss_ind.sum())

        # Correlate missingness with observed numeric features
        mar_correlations: list[dict] = []
        for other in self._numeric_feature_cols:
            if other == col:
                continue
            mask = self.df[other].notna()
            if mask.sum() < _MIN_ROWS_FOR_ANALYSIS:
                continue
            try:
                r = float(miss_ind[mask].corr(self.df[other][mask]))
                if pd.notna(r) and abs(r) >= self.corr_threshold_mar:
                    mar_correlations.append({"feature": other, "correlation": round(abs(r), 4)})
            except Exception:
                pass
        mar_correlations.sort(key=lambda x: -x["correlation"])

        # Significance test for MAR: logistic regression of the missingness
        # indicator on the observed numeric covariates + likelihood-ratio test.
        # This drives the label; the correlations above are kept as effect size.
        mar_test: dict | None = None
        cov_cols = [c for c in self._numeric_feature_cols if c != col]
        if cov_cols:
            X_cov = self.df[cov_cols].to_numpy(dtype=float)
            # Mean-impute covariates so every row is usable; the indicator y is
            # fully observed. Columns that are entirely NaN are dropped.
            col_means = np.nanmean(np.where(np.isnan(X_cov), np.nan, X_cov), axis=0)
            usable = ~np.isnan(col_means)
            if usable.any():
                X_cov = X_cov[:, usable]
                col_means = col_means[usable]
                inds = np.where(np.isnan(X_cov))
                X_cov[inds] = np.take(col_means, inds[1])
                mar_test = _logistic_lr_test(miss_ind.to_numpy(), X_cov)
        mar_significant = (
            mar_test is not None and mar_test["p_value"] < _ALPHA
            if mar_test is not None
            else bool(mar_correlations)  # legacy fallback when test not computable
        )

        # Detect group-dependent missingness via categorical feature spread
        cat_correlations: list[dict] = []
        for cat_col in self._categorical_feature_cols:
            if cat_col == col:
                continue
            groups = self.df[cat_col].dropna().unique()
            if len(groups) < 2 or len(groups) > 50:
                continue
            rates: dict[str, float] = {}
            for g in groups:
                mask = self.df[cat_col] == g
                if mask.sum() >= 5:
                    rates[str(g)] = float(miss_ind[mask].mean())
            if len(rates) < 2:
                continue
            spread = max(rates.values()) - min(rates.values())
            if spread >= self.group_spread_threshold:
                high_miss_groups = {
                    grp: round(rate, 3)
                    for grp, rate in rates.items()
                    if rate >= 0.5
                }
                # χ² test of independence: is missingness significantly associated
                # with this categorical grouping (not just a large raw spread)?
                chi2_res = _chi2_independence(miss_ind, self.df[cat_col])
                cat_correlations.append({
                    "feature": cat_col,
                    "correlation": round(spread, 4),
                    "p_value": chi2_res["p_value"] if chi2_res else None,
                    "chi2_statistic": chi2_res["statistic"] if chi2_res else None,
                    "high_missing_groups": high_miss_groups,
                    "all_group_rates": {k: round(v, 3) for k, v in rates.items()},
                })
        cat_correlations.sort(key=lambda x: -x["correlation"])
        # Group label requires a significant χ² (or falls back to spread-only when
        # the test could not be computed).
        group_significant = any(
            (c["p_value"] is None) or (c["p_value"] < _ALPHA)
            for c in cat_correlations
        )

        # Associate missingness with the target, backed by a significance test:
        #   numeric target      → point-biserial correlation + its p-value
        #   categorical target  → χ² test of independence
        target_correlation: float | None = None
        target_signal = False
        target_test: dict | None = None
        if self.target_col and self.target_col in self.df.columns:
            tgt = self.df[self.target_col]
            if pd.api.types.is_numeric_dtype(tgt):
                mask = tgt.notna()
                if mask.sum() >= _MIN_ROWS_FOR_ANALYSIS:
                    try:
                        r = float(miss_ind[mask].corr(tgt[mask]))
                        if pd.notna(r):
                            target_correlation = round(abs(r), 4)
                            p_val = None
                            try:
                                from scipy.stats import pointbiserialr
                                pb = pointbiserialr(miss_ind[mask].to_numpy(),
                                                    tgt[mask].to_numpy())
                                p_val = round(float(pb.pvalue), 6)
                            except Exception:
                                p_val = None
                            target_test = {"method": "point_biserial",
                                           "correlation": target_correlation,
                                           "p_value": p_val}
                            target_signal = (
                                p_val < _ALPHA if p_val is not None
                                else abs(r) >= self.corr_threshold_target
                            )
                    except Exception:
                        pass
            else:
                groups = tgt.dropna().unique()
                if len(groups) >= 2:
                    rates_t = {
                        str(g): float(miss_ind[tgt == g].mean())
                        for g in groups
                    }
                    spread = max(rates_t.values()) - min(rates_t.values())
                    target_correlation = round(spread, 4)
                    chi2_res = _chi2_independence(miss_ind, tgt)
                    p_val = chi2_res["p_value"] if chi2_res else None
                    target_test = {"method": "chi2_contingency",
                                   "spread": target_correlation, "p_value": p_val}
                    target_signal = (
                        p_val < _ALPHA if p_val is not None
                        else spread >= self.corr_threshold_target
                    )

        # Check if this column itself is high-cardinality categorical
        is_cat = not pd.api.types.is_numeric_dtype(self.df[col])
        n_total = len(self.df[col])
        unique_count = int(self.df[col].dropna().nunique())
        is_high_card_col = is_cat and (
            unique_count > _HIGH_CARDINALITY_ABS
            or (n_total > 0 and unique_count / max(n_total, 1) > _HIGH_CARDINALITY_RATIO)
        )

        # Label is driven by significance tests, in priority order. The legacy
        # correlation lists act as a fallback only when a test is not computable.
        if (n_obs < _MIN_ROWS_FOR_ANALYSIS
                and not group_significant and not mar_significant and not target_signal):
            label = "insufficient evidence"
            evidence = "too_few_missing_rows_for_reliable_analysis"
        elif group_significant and cat_correlations:
            label = "group-dependent missingness"
            evidence = "missingness_significantly_associated_with_categorical_group_chi2"
        elif target_signal:
            label = "target-associated missingness"
            evidence = "missingness_significantly_associated_with_target"
        elif mar_significant:
            label = "MAR-like evidence"
            evidence = "missingness_significantly_predicted_by_observed_covariates_logistic_lr"
        else:
            label = "MCAR-compatible"
            evidence = "no_significant_association_found"

        # Build evidence dicts
        group_dependency_evidence: dict | None = None
        if cat_correlations:
            top = cat_correlations[0]
            group_dependency_evidence = {
                "detected": group_significant,
                "top_categorical_feature": top["feature"],
                "max_missingness_spread": top["correlation"],
                "chi2_p_value": top.get("p_value"),
                "high_missing_groups": top.get("high_missing_groups", {}),
            }
        else:
            group_dependency_evidence = {"detected": False}

        target_association_evidence: dict = {
            "detected": target_signal,
            "correlation": target_correlation,
            "test": target_test,
        }

        covariate_association_evidence = mar_correlations[:5]

        # Collins et al. (2001) MAR robustness: below 25% missing and max corr < 0.4,
        # omitting a lurking variable has negligible effect on regression estimates.
        miss_rate = float(miss_ind.mean())
        max_observed_corr = max(
            (x["correlation"] for x in mar_correlations), default=0.0
        )
        if (
            label == "MCAR-compatible"
            and miss_rate < _MAR_ROBUST_MAX_MISS_RATE
            and max_observed_corr < _MAR_ROBUST_MAX_CORR
        ):
            mar_robustness_note = "likely_robust_per_collins2001"
        elif miss_rate >= _MAR_ROBUST_MAX_MISS_RATE or max_observed_corr >= _MAR_ROBUST_MAX_CORR:
            mar_robustness_note = "mechanism_assumption_may_matter_consider_full_mi"
        else:
            mar_robustness_note = "insufficient_evidence_to_assess"

        return {
            "mechanism_label": label,
            "mechanism_evidence": evidence,
            "mar_robustness_note": mar_robustness_note,
            "correlated_features": mar_correlations[:5],
            "categorical_correlated_features": cat_correlations[:5],
            "target_correlation": target_correlation,
            "target_signal": target_signal,
            "mar_test": mar_test,
            "target_test": target_test,
            "group_dependency_evidence": group_dependency_evidence,
            "target_association_evidence": target_association_evidence,
            "covariate_association_evidence": covariate_association_evidence,
        }
