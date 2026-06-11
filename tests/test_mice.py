"""MICE + Rubin pooling: multiple imputation inflates variance over single imputation.

That inflation is the entire point — it is the uncertainty that single imputation
hides (Rubin 1987; van Buuren FIMD Ch2).
"""

import numpy as np
import pandas as pd

from missingness_auditor import pool_rubin, mice_pool_column_means, MICEImputer


def test_pool_rubin_total_variance_exceeds_within():
    estimates = [1.0, 2.0, 3.0, 4.0, 5.0]
    variances = [0.1, 0.1, 0.1, 0.1, 0.1]
    pooled = pool_rubin(estimates, variances)

    assert pooled["m"] == 5
    assert pooled["pooled_estimate"] == 3.0
    assert pooled["between_variance"] > 0
    # T = U-bar + (1 + 1/m) * B  must exceed U-bar whenever estimates disagree.
    assert pooled["total_variance"] > pooled["within_variance"]
    assert 0.0 < pooled["fraction_missing_information"] < 1.0
    lo, hi = pooled["ci_95"]
    assert lo < pooled["pooled_estimate"] < hi


def test_pool_rubin_no_between_variance_when_single():
    pooled = pool_rubin([2.0], [0.25])
    assert pooled["between_variance"] == 0.0
    assert pooled["total_variance"] == 0.25
    assert pooled["fraction_missing_information"] == 0.0


def test_mice_pooling_reports_inflated_se():
    rng = np.random.RandomState(1)
    n = 300
    df = pd.DataFrame({
        "a": rng.normal(10, 3, n),
        "b": rng.normal(-2, 1, n),
        "c": rng.normal(100, 20, n),
    })
    # 'a' is MAR-correlated with 'b'; impose ~30% missingness on 'a'.
    df.loc[df["b"] < df["b"].median(), "a"] = np.where(
        rng.rand((df["b"] < df["b"].median()).sum()) < 0.6, np.nan,
        df.loc[df["b"] < df["b"].median(), "a"],
    )

    result = mice_pool_column_means(df, m=5)
    assert result["m"] == 5
    assert "a" in result["columns"]
    col = result["columns"]["a"]
    # Pooled (MI) SE must be >= the naive single-imputation SE by construction.
    assert col["std_error"] >= col["naive_single_imputation_std_error"]
    assert col["total_variance"] >= col["within_variance"]
    assert 0.0 <= col["fraction_missing_information"] < 1.0


def test_mice_imputer_produces_m_distinct_draws():
    rng = np.random.RandomState(2)
    n = 200
    df = pd.DataFrame({"a": rng.normal(0, 1, n), "b": rng.normal(0, 1, n)})
    df.loc[rng.rand(n) < 0.3, "a"] = np.nan

    draws = MICEImputer(m=4).fit_transform(df)
    assert len(draws) == 4
    assert all(d["a"].isna().sum() == 0 for d in draws)
    # sample_posterior draws should not be identical on the imputed cells.
    miss = df["a"].isna().to_numpy()
    fills = np.array([d["a"].to_numpy()[miss] for d in draws])
    assert fills.std(axis=0).sum() > 0
