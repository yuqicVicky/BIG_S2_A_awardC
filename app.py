"""Streamlit web demo — AI-Guided Missingness Audit & Imputation Planner."""

from __future__ import annotations

import json
import os
import sys
from typing import Optional

import anthropic
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

st.set_page_config(
    page_title="AI Missingness Auditor",
    page_icon="🤖",
    layout="wide",
)

# ──────────────────────────────────────────────── helpers ────────────────────

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
    return pd.DataFrame({"age": age, "income": income, "risk_score": risk, "target": target})


def _df_to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode()


def _reset() -> None:
    for key in [
        "audit_results", "auditor", "df_train", "df_predict", "target_col",
        "imputed_df", "imputed_df_predict", "ai_sections",
        "chat_history", "imputation_summary",
    ]:
        st.session_state.pop(key, None)


def _build_context(results: dict, target_col: Optional[str]) -> str:
    """Compact text summary of all audit results for LLM context."""
    profile = results["missingness_profile"]
    mech = results["mechanism_audit"]
    struct = results["structural_missingness"]
    plan = results["imputation_plan"]
    leakage = results["leakage_safe_check"]
    summary = profile["summary"]

    col_lines = []
    for col, info in profile["columns"].items():
        m = mech["columns"].get(col, {})
        p = plan["columns"].get(col, {})
        col_lines.append(
            f"  - {col}: {info['missing_rate']:.1%} missing, severity={info['severity']}, "
            f"dtype={info['dtype_category']}, mechanism={m.get('mechanism_label', '?')}, "
            f"target_signal={m.get('target_signal', False)}, strategy={p.get('strategy', '?')}"
        )

    if struct["n_structural_pairs"] > 0:
        struct_str = (
            f"{struct['n_structural_pairs']} structural pairs: "
            + ", ".join(
                f"{p['categorical_col']}→{p['numeric_col']}"
                for p in struct["structural_pairs"]
            )
        )
    else:
        struct_str = "none"

    return (
        f"Dataset: {summary['total_rows']:,} rows × {summary['total_columns']} columns | "
        f"Target: {target_col or 'none'} | "
        f"Overall missing rate: {summary['overall_missing_rate']:.1%} | "
        f"Columns with missing: {summary['columns_with_any_missing']}/{summary['total_columns']} | "
        f"Structural absence: {struct_str} | "
        f"Leakage: {leakage.get('high_risk_count', 0)} high-risk, "
        f"{leakage.get('medium_risk_count', 0)} medium-risk\n\n"
        f"Per-column breakdown:\n"
        + "\n".join(col_lines)
        + f"\n\nStrategy counts: {json.dumps(plan.get('summary', {}).get('strategy_counts', {}))}"
    )


def _claude_call(client: anthropic.Anthropic, prompt: str, max_tokens: int = 400) -> str:
    try:
        resp = client.messages.create(
            model="claude-opus-4-8",
            max_tokens=max_tokens,
            thinking={"type": "adaptive"},
            messages=[{"role": "user", "content": prompt}],
        )
        return next((b.text for b in resp.content if b.type == "text"), "")
    except Exception as e:
        return f"_(AI insight unavailable: {e})_"


def _show_insight(key: str, ai_sections: dict, api_key: str) -> None:
    if text := ai_sections.get(key):
        st.info(f"🤖 **Claude:** {text}")
    elif not api_key:
        st.caption("_Add an API key in the sidebar for AI-guided insights._")


# ──────────────────────────────────────────────── sidebar ────────────────────
with st.sidebar:
    st.title("🤖 AI Missingness Auditor")
    st.caption("Claude narrates every step of your missing-data analysis.")

    st.divider()
    api_key_input = st.text_input(
        "Anthropic API Key",
        type="password",
        placeholder="sk-ant-...",
        help="Required for AI-guided analysis.",
    )
    if api_key_input:
        st.success("API key ready ✓", icon="🔑")
    else:
        st.info("Add an API key for AI analysis.", icon="🔑")

    st.divider()
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
                _reset()
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
            _reset()
            st.session_state.pop("_train_file_key", None)
        df_train = _demo_df()

    if df_train is not None:
        all_cols = ["(none)"] + list(df_train.columns)
        target_col = st.selectbox("Target column", all_cols)
        target_col = None if target_col == "(none)" else target_col
        run_btn = st.button("▶ Run AI Audit", type="primary", use_container_width=True)
    else:
        target_col = None
        run_btn = False

    st.divider()
    st.caption(
        "This demo uses Claude to diagnose missingness mechanisms, interpret "
        "structural patterns, and recommend a leakage-safe imputation plan — "
        "with AI narration at every step."
    )

# ─────────────────────────────────────────────── run audit ───────────────────
if run_btn and df_train is not None:
    with st.spinner("Running missingness audit…"):
        auditor = MissingnessAuditor(df_train, predict_df=df_predict, target_col=target_col)
        results = auditor.run()
    st.session_state.update(
        {
            "audit_results": results,
            "auditor": auditor,
            "df_train": df_train,
            "df_predict": df_predict,
            "target_col": target_col,
        }
    )
    for k in ("imputed_df", "imputed_df_predict", "ai_sections",
               "chat_history", "imputation_summary"):
        st.session_state.pop(k, None)

# ────────────────────────────────────────────── landing page ─────────────────
if "audit_results" not in st.session_state:
    st.title("AI-Guided Missingness Audit")
    st.markdown(
        "Upload your dataset and let **Claude** walk you through every step: "
        "profile → mechanisms → structural patterns → imputation plan."
    )
    st.info("Select a dataset in the sidebar and click **▶ Run AI Audit** to begin.")
    st.stop()

# ──────────────────────────────────────────── retrieve state ─────────────────
results = st.session_state["audit_results"]
auditor: MissingnessAuditor = st.session_state["auditor"]
df_train = st.session_state["df_train"]
df_predict = st.session_state.get("df_predict")
target_col = st.session_state.get("target_col")

profile = results["missingness_profile"]
mech = results["mechanism_audit"]
struct = results["structural_missingness"]
plan = results["imputation_plan"]
leakage = results["leakage_safe_check"]
summary = profile["summary"]
sc = plan.get("summary", {}).get("strategy_counts", {})

# ───────────────────────────────── generate AI section insights (once) ───────
if api_key_input and "ai_sections" not in st.session_state:
    client = anthropic.Anthropic(api_key=api_key_input)
    ctx = _build_context(results, target_col)
    BASE = f"You are a senior data scientist reviewing this missing-data audit:\n\n{ctx}\n\n"

    with st.spinner("Claude is analyzing your dataset…"):
        ai_sections: dict[str, str] = {}
        try:
            ai_sections["overview"] = _claude_call(
                client,
                BASE
                + "Write a 3-sentence executive summary covering: (1) overall severity, "
                "(2) the most concerning column(s), (3) the main modeling risk. "
                "Be specific to these numbers.",
                400,
            )
            ai_sections["profile"] = _claude_call(
                client,
                BASE
                + "In 3 sentences, explain the per-column missingness profile. "
                "Which columns have critical or high severity, and what does that mean for "
                "downstream model quality?",
                350,
            )
            ai_sections["mechanisms"] = _claude_call(
                client,
                BASE
                + "In 4 sentences, explain what the detected missingness mechanisms mean for "
                "this dataset. Focus on any target-associated or MAR-like columns and their "
                "bias implications for a predictive model.",
                450,
            )
            ai_sections["structural"] = _claude_call(
                client,
                BASE
                + (
                    "In 2-3 sentences, interpret the structural absence patterns found and "
                    "explain what they imply for imputation."
                    if struct["n_structural_pairs"] > 0
                    else "In 2 sentences, confirm no structural absence was found and explain "
                    "why that simplifies imputation."
                ),
                300,
            )
            ai_sections["plan"] = _claude_call(
                client,
                BASE
                + "In 4 sentences, explain the imputation strategies recommended. Why do "
                "specific columns get their strategies? What must the analyst do before "
                "model training to avoid leakage?",
                450,
            )
        except anthropic.AuthenticationError:
            st.error("Invalid API key. Please check your key in the sidebar.")
            ai_sections = {}
        except Exception as e:
            st.warning(f"Could not generate AI insights: {e}")
            ai_sections = {}

    st.session_state["ai_sections"] = ai_sections

ai_sections: dict[str, str] = st.session_state.get("ai_sections", {})

def _insight(key: str) -> None:
    _show_insight(key, ai_sections, api_key_input)

# ──────────────────────────────────────────────── main report ────────────────
st.title("AI Missingness Audit Report")

# KPIs
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", f"{summary['total_rows']:,}")
c2.metric("Columns", summary["total_columns"])
c3.metric("With missing", summary["columns_with_any_missing"])
c4.metric("Overall rate", f"{summary['overall_missing_rate']:.1%}")

_insight("overview")

st.divider()

# ── Section 1: Profile ────────────────────────────────────────────────────────
st.subheader("📊 Missingness Profile")
st.dataframe(
    pd.DataFrame(
        [
            {
                "Column": col,
                "Missing Rate": f"{info['missing_rate']:.1%}",
                "Severity": info["severity"],
                "Dtype": info["dtype_category"],
                "Is Target": "✓" if info.get("is_target") else "",
            }
            for col, info in profile["columns"].items()
        ]
    ),
    use_container_width=True,
    hide_index=True,
)
_insight("profile")

if profile.get("predict_missing_rates"):
    st.markdown("**Predict-set missing rates**")
    st.dataframe(
        pd.DataFrame(
            [
                {"Column": c, "Predict Missing Rate": f"{r:.1%}"}
                for c, r in profile["predict_missing_rates"].items()
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )

st.divider()

# ── Section 2: Mechanisms ─────────────────────────────────────────────────────
st.subheader("🧩 Missingness Mechanisms")
st.caption(
    "MCAR-compatible: no detectable correlation. "
    "MAR-like: correlated with other observed features. "
    "Target-associated: correlated with the outcome — bias risk. "
    "Structural absence: driven by categorical group membership."
)
st.dataframe(
    pd.DataFrame(
        [
            {
                "Column": col,
                "Mechanism": info["mechanism_label"],
                "Target Signal": "Yes" if info.get("target_signal") else "No",
                "Target Corr": (
                    f"{info['target_correlation']:.3f}"
                    if info.get("target_correlation") is not None
                    else "—"
                ),
                "Top Correlated": (
                    info["correlated_features"][0]["feature"]
                    if info.get("correlated_features")
                    else "—"
                ),
            }
            for col, info in mech["columns"].items()
        ]
    ),
    use_container_width=True,
    hide_index=True,
)
_insight("mechanisms")

st.divider()

# ── Section 3: Structural ─────────────────────────────────────────────────────
st.subheader("🏗 Structural Absence")
if struct["n_structural_pairs"] == 0:
    st.success("No structural missingness pairs detected.")
else:
    st.warning(f"{struct['n_structural_pairs']} structural pair(s) detected.")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Categorical (NA)": p["categorical_col"],
                    "Numeric (companion)": p["numeric_col"],
                    "Pattern": p["pattern"],
                    "NA fraction": f"{(p.get('na_fraction_when_cat_na') or p.get('na_fraction_for_group', 0)):.1%}",
                }
                for p in struct["structural_pairs"]
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )
_insight("structural")

st.divider()

# ── Section 4: Imputation Plan ────────────────────────────────────────────────
st.subheader("📋 Imputation Plan")

if leakage.get("high_risk_count", 0) + leakage.get("medium_risk_count", 0) > 0:
    st.warning(
        f"Leakage check: {leakage.get('high_risk_count', 0)} high-risk, "
        f"{leakage.get('medium_risk_count', 0)} medium-risk finding(s)."
    )
else:
    st.success("Leakage check passed — safe to proceed.")

st.dataframe(
    pd.DataFrame(
        [
            {
                "Column": col,
                "Strategy": entry["strategy"],
                "Group By": entry.get("group_col", "—"),
                "Add Indicator": "Yes" if entry["add_missing_indicator"] else "No",
                "Fit On": entry["fit_on"],
                "Reason": entry["reason"],
            }
            for col, entry in plan["columns"].items()
        ]
    ),
    use_container_width=True,
    hide_index=True,
)

if sc:
    st.markdown(
        "**Strategy counts:** "
        + " · ".join(f"`{k}`: {v}" for k, v in sorted(sc.items(), key=lambda x: -x[1]))
    )

_insight("plan")

with st.expander("Leakage-safe sklearn pattern"):
    proto = leakage.get("global_protocol", {})
    st.markdown(f"**Rule:** {proto.get('rule', '')}")
    st.code(proto.get("sklearn_pattern", ""), language="python")

# ── Apply Imputation ──────────────────────────────────────────────────────────
st.divider()
if st.button("⚡ Apply Imputation", type="primary"):
    with st.spinner("Applying imputation plan…"):
        result = auditor.apply_imputation(df_train, plan, df_predict)
        if isinstance(result, tuple):
            st.session_state["imputed_df"], st.session_state["imputed_df_predict"] = result
        else:
            st.session_state["imputed_df"] = result

    if api_key_input and "imputed_df" in st.session_state:
        imp = st.session_state["imputed_df"]
        remaining = int(imp.isna().sum().sum())
        client = anthropic.Anthropic(api_key=api_key_input)
        st.session_state["imputation_summary"] = _claude_call(
            client,
            f"Imputation was applied: {imp.shape[0]} rows × {imp.shape[1]} columns, "
            f"{remaining} NAs remain. Strategies used: {json.dumps(sc)}. "
            "In 2 sentences, confirm what was done and flag one thing to verify before training.",
            250,
        )

if "imputed_df" in st.session_state:
    imp = st.session_state["imputed_df"]
    st.success(f"Imputation complete — {imp.shape[0]:,} rows × {imp.shape[1]} columns")
    if summ := st.session_state.get("imputation_summary"):
        st.info(f"🤖 **Claude:** {summ}")
    st.dataframe(imp.head(20), use_container_width=True)
    remaining = int(imp.isna().sum().sum())
    if remaining:
        st.warning(
            f"{remaining} missing values remain "
            "(expected for drop_column / no_imputation_needed strategies)."
        )
    else:
        st.success("No missing values remain.")
    if "imputed_df_predict" in st.session_state:
        st.subheader("Predict set (imputed)")
        st.dataframe(st.session_state["imputed_df_predict"].head(20), use_container_width=True)

st.divider()

# ── Section 5: Visualizations ─────────────────────────────────────────────────
st.subheader("🖼 Visualizations")
viz = MissingnessVisualizer(df_train, target_col=target_col)
figs = viz.generate_figures()
col_a, col_b = st.columns(2)
with col_a:
    st.markdown("**Missingness Rate by Column**")
    st.pyplot(figs["missingness_bar"], bbox_inches="tight")
with col_b:
    st.markdown("**Pattern Matrix**")
    st.pyplot(figs["missingness_matrix"], bbox_inches="tight")
st.markdown("**Target Signal by Missingness**")
st.pyplot(figs["missingness_target_signal"], bbox_inches="tight")
for fig in figs.values():
    plt.close(fig)

st.divider()

# ── Section 6: Chat ───────────────────────────────────────────────────────────
st.subheader("💬 Ask Claude")
if not api_key_input:
    st.info(
        "Add your Anthropic API key in the sidebar to ask follow-up questions about your data.",
        icon="🔑",
    )
else:
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    ctx = _build_context(results, target_col)

    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about your missing data…"):
        st.session_state["chat_history"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Prepend audit context to the first user message only
        api_messages = []
        for i, m in enumerate(st.session_state["chat_history"]):
            if i == 0:
                api_messages.append(
                    {
                        "role": "user",
                        "content": f"Audit context:\n{ctx}\n\n---\n{m['content']}",
                    }
                )
            else:
                api_messages.append(m)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            client = anthropic.Anthropic(api_key=api_key_input)
            try:
                full = ""
                with client.messages.stream(
                    model="claude-opus-4-8",
                    max_tokens=1024,
                    thinking={"type": "adaptive"},
                    messages=api_messages,
                ) as stream:
                    for chunk in stream.text_stream:
                        full += chunk
                        placeholder.markdown(full + "▌")
                placeholder.markdown(full)
                st.session_state["chat_history"].append(
                    {"role": "assistant", "content": full}
                )
            except anthropic.AuthenticationError:
                st.error("Invalid API key. Please check your key in the sidebar.")
            except Exception as e:
                st.error(f"Error: {e}")

# ── Downloads ─────────────────────────────────────────────────────────────────
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
        st.button(
            "📊 Imputed predict CSV",
            disabled=True,
            help="Provide predict CSV and apply imputation",
        )
