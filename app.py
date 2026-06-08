"""Streamlit web demo — Missingness Audit & Imputation Planner."""

from __future__ import annotations

import io
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from missingness_auditor import MissingnessAuditor
from missingness_auditor.visualization import MissingnessVisualizer
from missingness_auditor.reporting import ReportWriter

# ─────────────────────────────── page config ────────────────────────────────
st.set_page_config(
    page_title="Missingness Audit & Imputation Planner",
    page_icon="🔍",
    layout="wide",
)

# ─────────────────────────────── helpers ────────────────────────────────────

def _demo_df() -> pd.DataFrame:
    demo_path = os.path.join(os.path.dirname(__file__), "examples", "demo_missingness.csv")
    if os.path.exists(demo_path):
        return pd.read_csv(demo_path)
    rng = np.random.default_rng(42)
    n = 300
    target = rng.binomial(1, 0.3, n).astype(float)
    age = np.where(rng.random(n) < 0.08, np.nan, rng.integers(20, 70, n).astype(float))
    income = np.where(rng.random(n) < 0.35, np.nan, rng.normal(50000, 15000, n))
    risk = rng.normal(650, 90, n).astype(float)
    risk[target == 1] = np.nan
    return pd.DataFrame({
        "age": age, "income": income, "risk_score": risk, "target": target,
    })


def _fig_to_bytes(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    buf.seek(0)
    return buf.read()


def _df_to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode()


def _reset_audit() -> None:
    for key in ["audit_results", "imputed_df", "imputed_df_predict"]:
        st.session_state.pop(key, None)


# ─────────────────────────────── sidebar ────────────────────────────────────
with st.sidebar:
    st.title("🔍 Missingness Auditor")
    st.caption("Upload a CSV or load the demo dataset to diagnose missing data.")

    source = st.radio("Data source", ["Upload CSV", "Use demo dataset"], index=1)

    df_train: pd.DataFrame | None = None
    df_predict: pd.DataFrame | None = None

    if source == "Upload CSV":
        uploaded = st.file_uploader("Training CSV", type="csv")
        if uploaded:
            file_key = f"{uploaded.name}_{uploaded.size}"
            if st.session_state.get("_train_file_key") != file_key:
                st.session_state["_train_file_key"] = file_key
                st.session_state["_df_train_cached"] = pd.read_csv(uploaded)
                _reset_audit()
            df_train = st.session_state.get("_df_train_cached")
        uploaded_pred = st.file_uploader("Predict CSV (optional)", type="csv")
        if uploaded_pred:
            pred_key = f"{uploaded_pred.name}_{uploaded_pred.size}"
            if st.session_state.get("_pred_file_key") != pred_key:
                st.session_state["_pred_file_key"] = pred_key
                st.session_state["_df_pred_cached"] = pd.read_csv(uploaded_pred)
            df_predict = st.session_state.get("_df_pred_cached")
    else:
        if st.button("Load demo dataset"):
            _reset_audit()
            st.session_state.pop("_train_file_key", None)
            st.session_state.pop("_pred_file_key", None)
        df_train = _demo_df()

    if df_train is not None:
        all_cols = ["(none)"] + list(df_train.columns)
        target_col = st.selectbox("Target column", all_cols)
        target_col = None if target_col == "(none)" else target_col

        run_btn = st.button("▶ Run Audit", type="primary", use_container_width=True)
    else:
        target_col = None
        run_btn = False

    st.divider()
    st.markdown("**About**")
    st.caption(
        "Diagnoses missingness mechanisms (MCAR-compatible, MAR-like, group-dependent, "
        "target-associated, structural absence), and produces a leakage-safe imputation plan."
    )

# ─────────────────────────────── run audit ──────────────────────────────────
if run_btn and df_train is not None:
    with st.spinner("Running missingness audit…"):
        auditor = MissingnessAuditor(df_train, df_predict, target_col=target_col)
        st.session_state["audit_results"] = auditor.run()
        st.session_state["auditor"] = auditor
        st.session_state["df_train"] = df_train
        st.session_state["df_predict"] = df_predict
        st.session_state.pop("imputed_df", None)
        st.session_state.pop("imputed_df_predict", None)

# ─────────────────────────────── main area ──────────────────────────────────
if "audit_results" not in st.session_state:
    st.title("Missingness Audit & Imputation Planner")
    st.info("Select a dataset in the sidebar and click **▶ Run Audit** to begin.")
    st.stop()

results = st.session_state["audit_results"]
auditor: MissingnessAuditor = st.session_state["auditor"]
df_train = st.session_state["df_train"]
df_predict = st.session_state.get("df_predict")

profile = results["missingness_profile"]
mech = results["mechanism_audit"]
struct = results["structural_missingness"]
plan = results["imputation_plan"]
leakage = results["leakage_safe_check"]
summary = profile["summary"]

st.title("Missingness Audit Report")

# ── top-level KPIs
col1, col2, col3, col4 = st.columns(4)
col1.metric("Rows", f"{summary['total_rows']:,}")
col2.metric("Columns", summary["total_columns"])
col3.metric("With missing", summary["columns_with_any_missing"])
col4.metric("Overall rate", f"{summary['overall_missing_rate']:.1%}")

# ── tabs
tab_profile, tab_mech, tab_struct, tab_plan, tab_figures = st.tabs([
    "📊 Profile", "🧩 Mechanisms", "🏗 Structural", "📋 Imputation Plan", "🖼 Figures"
])

# ── Tab 1: Profile
with tab_profile:
    st.subheader("Per-column missingness profile")
    rows = []
    for col, info in profile["columns"].items():
        rows.append({
            "Column": col,
            "Missing Rate": f"{info['missing_rate']:.1%}",
            "Severity": info["severity"],
            "Dtype": info["dtype_category"],
            "Is Target": "✓" if info.get("is_target") else "",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    if "predict_missing_rates" in profile and profile["predict_missing_rates"]:
        st.subheader("Predict-set missing rates")
        pred_rows = [
            {"Column": c, "Predict Missing Rate": f"{r:.1%}"}
            for c, r in profile["predict_missing_rates"].items()
        ]
        st.dataframe(pd.DataFrame(pred_rows), use_container_width=True, hide_index=True)

# ── Tab 2: Mechanisms
with tab_mech:
    st.subheader("Missingness mechanism clues")
    st.caption(
        "Labels are statistical clues, not causal claims. "
        "MCAR-compatible = no significant correlation; "
        "MAR-like = correlated with observed numeric features; "
        "group-dependent = concentrated in specific categorical groups; "
        "target-associated = correlated with target variable; "
        "structural absence concern = set by structural detector."
    )
    mech_rows = []
    for col, info in mech["columns"].items():
        mech_rows.append({
            "Column": col,
            "Mechanism Clue": info["mechanism_label"],
            "Target Signal": "Yes" if info.get("target_signal") else "No",
            "Target Corr": (
                f"{info['target_correlation']:.3f}"
                if info.get("target_correlation") is not None
                else "—"
            ),
            "Top Correlated Feature": (
                info["correlated_features"][0]["feature"]
                if info.get("correlated_features")
                else "—"
            ),
        })
    st.dataframe(pd.DataFrame(mech_rows), use_container_width=True, hide_index=True)

# ── Tab 3: Structural
with tab_struct:
    st.subheader("Structural absence pairs")
    if struct["n_structural_pairs"] == 0:
        st.success("No structural missingness pairs detected.")
    else:
        st.warning(f"{struct['n_structural_pairs']} structural pair(s) detected.")
        pair_rows = [
            {
                "Categorical (NA)": p["categorical_col"],
                "Numeric (companion)": p["numeric_col"],
                "Pattern": p["pattern"],
                "NA fraction": f"{(p.get('na_fraction_when_cat_na') or p.get('na_fraction_for_group', 0)):.1%}",
            }
            for p in struct["structural_pairs"]
        ]
        st.dataframe(pd.DataFrame(pair_rows), use_container_width=True, hide_index=True)

    flag_rows = [
        {"Column": c, "Is Structural": "✓" if f["is_structural"] else ""}
        for c, f in struct.get("column_flags", {}).items()
        if f["is_structural"]
    ]
    if flag_rows:
        st.subheader("Flagged columns")
        st.dataframe(pd.DataFrame(flag_rows), use_container_width=True, hide_index=True)

# ── Tab 4: Imputation Plan
with tab_plan:
    st.subheader("Recommended imputation plan")

    leakage_risk = leakage.get("high_risk_count", 0) + leakage.get("medium_risk_count", 0)
    if leakage_risk > 0:
        st.warning(
            f"Leakage risk: {leakage.get('high_risk_count', 0)} high, "
            f"{leakage.get('medium_risk_count', 0)} medium risk finding(s). "
            "See details below."
        )
    else:
        st.success("No leakage risk detected. Safe to proceed with the plan.")

    plan_rows = []
    for col, entry in plan["columns"].items():
        plan_rows.append({
            "Column": col,
            "Strategy": entry["strategy"],
            "Group By": entry.get("group_col", "—"),
            "Add Indicator": "Yes" if entry["add_missing_indicator"] else "No",
            "Fit On": entry["fit_on"],
            "Reason": entry["reason"],
        })
    st.dataframe(pd.DataFrame(plan_rows), use_container_width=True, hide_index=True)

    sc = plan.get("summary", {}).get("strategy_counts", {})
    if sc:
        st.subheader("Strategy counts")
        st.dataframe(
            pd.DataFrame(
                [{"Strategy": k, "Count": v} for k, v in sorted(sc.items(), key=lambda x: -x[1])]
            ),
            use_container_width=True, hide_index=True,
        )

    with st.expander("Leakage-safe protocol"):
        proto = leakage.get("global_protocol", {})
        st.markdown(f"**Rule:** {proto.get('rule', '')}")
        st.code(proto.get("sklearn_pattern", ""), language="python")

    # Apply imputation
    st.divider()
    if st.button("⚡ Apply Imputation", type="primary"):
        with st.spinner("Applying imputation plan…"):
            result = auditor.apply_imputation(df_train, plan, df_predict)
            if isinstance(result, tuple):
                st.session_state["imputed_df"], st.session_state["imputed_df_predict"] = result
            else:
                st.session_state["imputed_df"] = result

    if "imputed_df" in st.session_state:
        imputed = st.session_state["imputed_df"]
        st.success(f"Imputation complete — {imputed.shape[0]:,} rows × {imputed.shape[1]} columns")
        st.dataframe(imputed.head(20), use_container_width=True)

        remaining_na = imputed.isna().sum().sum()
        if remaining_na > 0:
            st.warning(f"{remaining_na} missing values remain (expected for drop_column or no_imputation_needed).")
        else:
            st.success("No missing values remain in imputed data.")

        if "imputed_df_predict" in st.session_state:
            st.subheader("Predict set (imputed)")
            pred_imp = st.session_state["imputed_df_predict"]
            st.dataframe(pred_imp.head(20), use_container_width=True)

# ── Tab 5: Figures
with tab_figures:
    st.subheader("Visualizations")
    viz = MissingnessVisualizer(df_train, target_col=target_col)
    figs = viz.generate_figures()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Missingness Rate by Column**")
        st.pyplot(figs["missingness_bar"])
    with c2:
        st.markdown("**Pattern Matrix**")
        st.pyplot(figs["missingness_matrix"])

    st.markdown("**Target Signal by Missingness**")
    st.pyplot(figs["missingness_target_signal"])

    for fig in figs.values():
        plt.close(fig)

# ─────────────────────────────── download buttons ───────────────────────────
st.divider()
st.subheader("Downloads")

rw = ReportWriter()
report_md = rw.generate_report_md(profile, mech, struct, plan, leakage)

d1, d2, d3, d4 = st.columns(4)

with d1:
    st.download_button(
        "📄 Markdown report",
        data=report_md.encode(),
        file_name="missing_data_report.md",
        mime="text/markdown",
    )

with d2:
    st.download_button(
        "🗂 imputation_plan.json",
        data=json.dumps(plan, indent=2).encode(),
        file_name="imputation_plan.json",
        mime="application/json",
    )

with d3:
    if "imputed_df" in st.session_state:
        st.download_button(
            "📊 Imputed train CSV",
            data=_df_to_csv(st.session_state["imputed_df"]),
            file_name="train_imputed.csv",
            mime="text/csv",
        )
    else:
        st.button("📊 Imputed train CSV", disabled=True, help="Apply imputation first")

with d4:
    if "imputed_df_predict" in st.session_state:
        st.download_button(
            "📊 Imputed predict CSV",
            data=_df_to_csv(st.session_state["imputed_df_predict"]),
            file_name="predict_imputed.csv",
            mime="text/csv",
        )
    else:
        st.button("📊 Imputed predict CSV", disabled=True, help="Provide predict CSV and apply imputation")
