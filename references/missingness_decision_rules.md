# Missingness Decision Rules

Practical rules for the Missingness Audit & Imputation Planner, derived from:
- van Buuren, S. (2012). *Flexible Imputation of Missing Data*. Chapman & Hall/CRC. (fimd.pdf)
- Collins, L.M., Schafer, J.L., & Kam, C.-M. (2001). A comparison of inclusive and restrictive strategies. *Psychological Methods*.

Rules are extracted for actionable use only. No long text is reproduced.

---

## Rule 1 — Mean imputation: avoid

**Rule:** Do not use mean (or mode) imputation as a general strategy.

**Evidence:** Mean imputation underestimates variance, distorts correlations, and produces
confidence intervals that are too narrow. (van Buuren Ch1, Table 1.1: stochastic regression
achieves correct coverage; mean imputation does not.)

**In our planner:** Numeric columns use `numeric_median` or `numeric_median_plus_indicator`,
never mean imputation. Categorical columns use `categorical_missing_token`, never mode.

---

## Rule 2 — Listwise deletion: no safe "critical missing rate"

**Rule:** There is no threshold (e.g., 5% or 10%) below which listwise deletion is known
to be safe. Listwise deletion is unbiased for means only under MCAR; it produces biased
estimates under MAR and MNAR (van Buuren Ch1).

**In our planner:** We do not recommend listwise deletion. The `drop_column` strategy
(>80% missing) is a feature-engineering signal only, not a deletion recommendation.

---

## Rule 3 — Mechanism labels are statistical clues, not diagnoses

**Rule:** MCAR, MAR, and MNAR cannot be confirmed from observational data alone.
Mechanism labels derived from correlations are statistical clues, not causal assignments.
(van Buuren Ch1, Section 1.2: "while convenient, MCAR is often unrealistic.")

**In our planner:** All labels are prefixed by their evidence source. Reports carry the
caution: "These labels are observational clues, not causal claims."

---

## Rule 4 — MAR as default; robustness threshold (Collins et al., 2001)

**Rule:** Use MAR as the default assumption. MAR is "a suitable starting point" (van Buuren
Ch5, Section 5.2). When the missing data rate does not exceed 25% AND the correlation
between any lurking variable and the incomplete variable is below 0.4, omitting that variable
from the imputation model has negligible effect on regression estimates.

**In our planner:** Below 25% missing with no strong correlations detected → label
`MCAR-compatible` and set `mar_robustness_note = "likely_robust_per_collins2001"`.
Above 25% OR correlation > 0.4 → note that mechanism assumptions matter more.

---

## Rule 5 — Numeric imputation: PMM is the gold standard for full MI

**Rule:** For continuous numeric variables in full Multiple Imputation, Predictive Mean
Matching (PMM) is the default method (van Buuren Table 5.1, Ch3 Section 3.4). PMM
draws imputed values from the observed data, preserving range and distribution without
requiring a distributional assumption.

**In our planner:** For simple ML pipelines: use `numeric_median_plus_indicator`.
When full MI is warranted, recommend upgrading to MICE with `pmm` method.

---

## Rule 6 — Categorical imputation: use logistic/multinomial regression for full MI

**Rule:** For full MI: binary → `logreg`, nominal → `polyreg`, ordinal → `polr` (van Buuren
Table 5.1). Mean/mode imputation is "included for completeness and should not be generally
used" (van Buuren Ch5).

**In our planner:** For simple ML pipelines: `categorical_missing_token` or
`categorical_missing_token_plus_indicator`. High-cardinality columns never use mode.

---

## Rule 7 — Regression imputation (predict-only) underestimates variance

**Rule:** Using regression predictions as imputations (without noise) yields correct
mean estimates but confidence intervals that are too narrow (coverage ~65% vs. 95%
nominal in van Buuren Table 3.1). Adding residual noise and parameter uncertainty
(Bayesian or bootstrap MI) restores correct coverage.

**In our planner:** For ML feature engineering, median + indicator captures direction
of missingness without overclaiming precision. For inference tasks, note that MICE
is needed for correct uncertainty quantification.

---

## Rule 8 — The goal of MI is to reflect uncertainty, not predict accurately

**Rule:** "The goal of MI is not accurately predicting the missing values, it is properly
reflecting the uncertainty due to missing data" (JSS review of FIMD, 2018). Use m ≥ 5
imputed datasets and pool with Rubin's rules for valid inference.

**In our planner:** The imputation plan targets ML feature engineering (single imputation
with indicator). For inference tasks, flag `mi_upgrade_recommendation` when uncertainty
reflection matters.

---

## Rule 9 — Include the target variable in the imputation model

**Rule:** When imputing, include all variables in the complete-data analysis model,
including the outcome/target. "Not including the complete data model variables will tend
to bias the results toward zero" (Little 1992 via van Buuren Ch5).

**In our planner:** Target-associated missingness is flagged as a signal, not a
disqualifier. The target correlation evidence is reported so downstream analysis can
use the target when fitting imputation models.

---

## Rule 10 — Predictor selection: 15–25 variables, prioritised

**Rule:** For imputation model predictors (van Buuren Ch5, Section 5.3.2):
1. All variables in the complete-data model (including target/outcome)
2. Variables correlated with the missingness indicator
3. Variables that explain variance in the incomplete column
4. Remove variables with too many missing values themselves
5. Limit to ~15–25 predictors; increase in explained variance is typically negligible beyond 15

**In our planner:** The `covariate_association_evidence` list (top-5 by correlation)
identifies which variables should be prioritised as imputation predictors.

---

## Rule 11 — Structural absence: domain-driven, not purely statistical

**Rule:** Structural absence occurs when a categorical NA encodes a real-world absence
(e.g., `garage_type = NaN` when the property has no garage → `garage_area = 0`). This
is a domain concept confirmed by a companion numeric being 0 or absent (van Buuren
Ch7 domain examples context). False positives arise when zero is a common legitimate
value across all groups.

**In our planner:** Structural detection requires both the categorical NA and a zero-lift
check on the companion numeric (zero concentrated under NA vs. elsewhere). Globally
common zeros (precipitation = 0) are excluded by the lift threshold.

---

## Rule 12 — Group-dependent missingness ≠ structural absence

**Rule:** When a numeric column is entirely missing for one categorical group value
(e.g., temperature measurements absent for one jurisdiction), this is group-dependent
missingness (a MAR-like pattern). It is NOT structural absence. The data exists in
principle but was not collected for that group.

**In our planner:** `_check_category_group_patterns` was removed from
`StructuralMissingnessDetector`. Group-dependent patterns are detected and labelled
by `MechanismAuditor` using categorical spread analysis.

---

## Rule 13 — Derived variables: impute originals, compute derived afterward

**Rule:** Ratio, interaction, and sum-score variables derived from incomplete base
variables should not be imputed directly. Impute the base variables, then compute
derived variables afterward. If derived variables are needed during imputation,
use passive imputation to maintain consistency and avoid feedback loops (van Buuren Ch5).

**In our planner:** The planner operates per-column on raw features. Derived/transformed
columns should be flagged by the user; the plan's `safety_warning` field can carry this note.

---

## Rule 14 — Indicator method in regression: biased even under MCAR

**Rule:** The "indicator method" (replace missing with 0, add binary flag as covariate
in downstream regression) yields severely biased regression estimates even under MCAR
and for low amounts of missing data (Vach & Blettner 1991 via van Buuren Ch1).

**IMPORTANT distinction:** Adding a *separate* missing indicator as an auxiliary feature
for a downstream ML model (our `add_missing_indicator = True`) is different and acceptable.
The banned pattern is: fill with 0 AND use that filled column directly in the same regression
without separating the missingness mechanism.

**In our planner:** `add_missing_indicator` adds a separate `_missing` flag column.
It does not replace the value with 0 for direct regression use.

---

## Rule 15 — LOCF and BOCF: avoid unless justified

**Rule:** Last-observation-carried-forward and baseline-observation-carried-forward
"should not be used as the primary approach for handling missing data unless the
assumptions that underlie them are scientifically justified" (National Research Council,
2010 via van Buuren Ch1). Biased even under MCAR.

**In our planner:** No LOCF/BOCF strategy is offered. Not applicable to cross-sectional
tabular data common in this planner's scope.

---

## Rule 16 — When to upgrade to full MICE (upgrade signal thresholds)

**Rule:** Simple median/token imputation is sufficient for ML feature engineering when
the missing rate is low and correlations are weak. Upgrade to full MICE when:
- Missing rate > 25% on key analysis variables (Collins et al. 2001 threshold)
- Any covariate correlation with missingness indicator > 0.4
- Target variable is itself partially missing and used in downstream inference
- Derived variables are needed during imputation (use passive imputation)

**In our planner:** The `mi_upgrade_recommendation` field per column signals when
upgrading from simple imputation to MICE/PMM is warranted.

---

## Summary table

| Scenario | Simple ML pipeline | Full inference (MI) |
|---|---|---|
| Continuous numeric, MCAR-compat | `numeric_median` | PMM (mice default) |
| Continuous numeric, MAR/target | `numeric_median_plus_indicator` | PMM + indicator |
| Numeric, group-dependent | `groupwise_numeric_median_plus_indicator` | MICE with group predictor |
| Numeric, structural zero | `structural_zero_plus_indicator` | 0-fill + indicator |
| Binary categorical | `categorical_missing_token` | logreg (mice) |
| Nominal categorical | `categorical_missing_token` | polyreg (mice) |
| High-cardinality text | `categorical_missing_token_plus_indicator` | missing token (pmm won't help) |
| Structural categorical NA | `structural_none_token_plus_indicator` | NONE token + indicator |
| >80% missing | `drop_column` | drop column |
