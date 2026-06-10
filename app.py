"""Streamlit web demo — AI-Guided Missingness Audit & Imputation Planner."""

from __future__ import annotations

import json
import os
import re
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

_CHART_TITLES = {
    "missingness_bar":    "Missingness Rate by Column",
    "pattern_matrix":     "Co-Missingness Pattern Matrix",
    "target_signal":      "Target Signal by Missingness",
    "missing_correlation": "Missingness Correlation",
}

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
        "chart_list", "chart_rationale", "domain_tags",
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


def _claude_structured(
    client: anthropic.Anthropic, prompt: str, max_tokens: int = 600
) -> dict | None:
    """Call Claude and return parsed JSON insight dict, or None on failure."""
    json_prompt = (
        prompt
        + "\n\nReturn ONLY valid JSON (no markdown fences, no extra text):\n"
        + '{"severity":"high|medium|low","headline":"one sentence summary",'
        + '"bullets":["finding 1","finding 2","finding 3"],"recommendation":"one action to take"}'
    )
    try:
        resp = client.messages.create(
            model="claude-opus-4-8",
            max_tokens=max_tokens,
            thinking={"type": "adaptive"},
            messages=[{"role": "user", "content": json_prompt}],
        )
        text = next((b.text for b in resp.content if b.type == "text"), "")
        text = re.sub(r"```[a-z]*\n?", "", text).strip().rstrip("`")
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            return json.loads(m.group())
    except Exception:
        pass
    return None


def _render_insight(data: dict | None, label: str = "Claude's Analysis") -> None:
    """Render a structured AI insight as a severity-aware card with bullets."""
    if data is None:
        return

    severity = data.get("severity", "low")
    headline = data.get("headline", "")
    bullets = data.get("bullets", [])
    rec = data.get("recommendation", "")

    icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(severity, "🔵")
    st.markdown(f"**{icon} {label}**")
    if headline:
        st.markdown(f"> _{headline}_")
    if bullets:
        st.markdown("\n".join(f"- {b}" for b in bullets))
    if rec:
        if severity == "high":
            st.error(f"**Action required:** {rec}")
        elif severity == "medium":
            st.warning(f"**Recommendation:** {rec}")
        else:
            st.success(f"**Suggested:** {rec}")


def _extract_domain_tags(
    client: anthropic.Anthropic, df: pd.DataFrame, target_col: str | None
) -> dict[str, str]:
    """Ask Claude (haiku) to assign a domain_tag to each column.
    Returns {col_name: domain_tag}; empty dict on failure.
    """
    col_summary = pd.DataFrame({
        "dtype":     df.dtypes.astype(str),
        "n_unique":  df.nunique(),
        "sample":    [df[c].dropna().head(3).tolist() if df[c].notna().any() else []
                      for c in df.columns],
        "min":       df.select_dtypes("number").min().reindex(df.columns),
        "max":       df.select_dtypes("number").max().reindex(df.columns),
    }).to_string()

    prompt = (
        f"Assign a domain_tag to every column in this dataset.\n"
        f"Target column (excluded from imputation): {target_col or 'none'}\n\n"
        f"Column summary:\n{col_summary}\n\n"
        f"Domain tag examples (use these or invent a fitting label):\n"
        f"  geographic_identifier, administrative_code, weather_metric, financial_indicator,\n"
        f"  health_outcome, demographic_rate, binary_flag, free_text, id_or_key, datetime,\n"
        f"  model_score, sensor_reading, temporal_measurement, environmental_metric.\n\n"
        f"Reason from column names, dtypes, value ranges, and sample values — do NOT match\n"
        f"against fixed abbreviation lists. If ambiguous, pick the most plausible tag.\n\n"
        f"Return ONLY valid JSON mapping every column name to its tag:\n"
        f'{{"{df.columns[0]}": "domain_tag", ...}}'
    )
    try:
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        text = next((b.text for b in resp.content if b.type == "text"), "")
        text = re.sub(r"```[a-z]*\n?", "", text).strip().rstrip("`")
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            d = json.loads(m.group())
            return {k: v for k, v in d.items() if isinstance(k, str) and isinstance(v, str)}
    except Exception:
        pass
    return {}


def _plan_charts(
    client: anthropic.Anthropic, ctx: str, has_target: bool, n_missing_cols: int
) -> tuple[list[str], str]:
    """Ask Claude (haiku) which 2-3 charts are most informative for this dataset."""
    available = ["missingness_bar", "pattern_matrix"]
    if has_target:
        available.append("target_signal")
    if n_missing_cols >= 3:
        available.append("missing_correlation")

    prompt = (
        f"Missing-data audit summary:\n{ctx}\n\n"
        f"Available chart types: {available}\n"
        "- missingness_bar: bar chart of missing % per column — always useful\n"
        "- pattern_matrix: row×column heatmap of co-missingness — useful when 2+ columns missing\n"
        "- target_signal: compare target value for rows with/without a feature — include if any target_signal=True\n"
        "- missing_correlation: correlation between missingness indicators — useful when 3+ columns missing\n\n"
        "Choose 2-3 charts that give the most insight for THIS specific dataset. "
        "Return ONLY valid JSON: {\"charts\":[\"name1\",\"name2\"],\"rationale\":\"brief reason\"}"
    )
    try:
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        text = next((b.text for b in resp.content if b.type == "text"), "")
        text = re.sub(r"```[a-z]*\n?", "", text).strip().rstrip("`")
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            d = json.loads(m.group())
            valid = [c for c in d.get("charts", []) if c in available]
            if valid:
                return valid, d.get("rationale", "")
    except Exception:
        pass
    # Sensible fallback
    charts = ["missingness_bar"]
    if n_missing_cols >= 2:
        charts.append("pattern_matrix")
    if has_target:
        charts.append("target_signal")
    return charts, ""


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
               "chat_history", "imputation_summary", "chart_list", "chart_rationale"):
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
    n_missing_cols = sum(1 for c in df_train.columns if df_train[c].isna().any())

    with st.spinner("Claude is analyzing your dataset…"):
        ai_sections: dict[str, dict | None] = {}
        chart_list: list[str] = []
        chart_rationale: str = ""
        try:
            ai_sections["overview"] = _claude_structured(
                client,
                BASE + "Assess the overall severity of missingness in this dataset. "
                "Identify the most concerning column and the main risk for a downstream model.",
                600,
            )
            ai_sections["profile"] = _claude_structured(
                client,
                BASE + "Assess the per-column missingness profile. "
                "Which columns have critical or high severity, and what does that mean for model quality?",
                600,
            )
            ai_sections["mechanisms"] = _claude_structured(
                client,
                BASE + "Interpret the detected missingness mechanisms. "
                "Focus on any target-associated or MAR-like columns and their bias implications for a predictive model.",
                700,
            )
            if struct["n_structural_pairs"] > 0:
                ai_sections["structural"] = _claude_structured(
                    client,
                    BASE + "Interpret the structural absence patterns found. "
                    "What do they imply for imputation strategy?",
                    500,
                )
            # Extract domain tags and re-run planner with domain-aware strategies.
            # This upgrades e.g. weather/sensor columns from median to ffill/bfill.
            domain_tags = _extract_domain_tags(client, df_train, target_col)
            if domain_tags:
                from missingness_auditor.planner import ImputationPlanner
                updated_plan = ImputationPlanner(
                    results["missingness_profile"],
                    results["mechanism_audit"],
                    results["structural_missingness"],
                    domain_tags=domain_tags,
                ).plan()
                results = {**results, "imputation_plan": updated_plan}
                st.session_state["audit_results"] = results
                st.session_state["domain_tags"] = domain_tags

            # Rebuild context with updated plan for the plan AI section
            ctx = _build_context(results, target_col)
            BASE = f"You are a senior data scientist reviewing this missing-data audit:\n\n{ctx}\n\n"
            ai_sections["plan"] = _claude_structured(
                client,
                BASE + "Evaluate the imputation plan. "
                "For each column explain why its strategy was chosen based on its real-world meaning "
                "(not just statistics), and what the analyst must do to avoid leakage.",
                800,
            )
            chart_list, chart_rationale = _plan_charts(client, ctx, bool(target_col), n_missing_cols)
        except anthropic.AuthenticationError:
            st.error("Invalid API key. Please check your key in the sidebar.")
            ai_sections = {}
        except Exception as e:
            st.warning(f"Could not generate AI insights: {e}")
            ai_sections = {}

    st.session_state["ai_sections"] = ai_sections
    st.session_state["chart_list"] = chart_list
    st.session_state["chart_rationale"] = chart_rationale

ai_sections: dict[str, dict | None] = st.session_state.get("ai_sections", {})

# Reload plan/results in case domain-aware re-run updated them
results = st.session_state["audit_results"]
plan = results["imputation_plan"]
sc = plan.get("summary", {}).get("strategy_counts", {})


def _insight(key: str, label: str = "Claude's Analysis") -> None:
    data = ai_sections.get(key)
    if data is None:
        if not api_key_input:
            st.caption("_Add an API key in the sidebar for AI-guided insights._")
        return
    _render_insight(data, label)


# ──────────────────────────────────────────────── main report ────────────────
st.title("AI Missingness Audit Report")

# KPIs
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", f"{summary['total_rows']:,}")
c2.metric("Columns", summary["total_columns"])
c3.metric("With missing", summary["columns_with_any_missing"])
c4.metric("Overall rate", f"{summary['overall_missing_rate']:.1%}")

_insight("overview", "Overall Assessment")

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
_insight("profile", "Profile Interpretation")

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
_insight("mechanisms", "Mechanism Interpretation")

st.divider()

# ── Section 3: Structural (only shown when pairs are detected) ────────────────
if struct["n_structural_pairs"] > 0:
    st.subheader("🏗 Structural Absence")
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
    _insight("structural", "Structural Pattern Analysis")
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

_insight("plan", "Plan Evaluation")

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
        raw = _claude_structured(
            client,
            f"Imputation was applied: {imp.shape[0]} rows × {imp.shape[1]} columns, "
            f"{remaining} NAs remain. Strategies used: {json.dumps(sc)}. "
            "Confirm what was done and flag one thing to verify before training.",
            400,
        )
        st.session_state["imputation_summary"] = raw

if "imputed_df" in st.session_state:
    imp = st.session_state["imputed_df"]
    st.success(f"Imputation complete — {imp.shape[0]:,} rows × {imp.shape[1]} columns")
    if summ := st.session_state.get("imputation_summary"):
        _render_insight(summ, "Post-Imputation Check")
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

# ── Section 5: Visualizations (LLM-planned) ───────────────────────────────────
st.subheader("🖼 Visualizations")

chart_list: list[str] = st.session_state.get("chart_list", [])
chart_rationale: str = st.session_state.get("chart_rationale", "")

# Fallback when no API key (or chart planning failed)
if not chart_list:
    n_mc = sum(1 for c in df_train.columns if df_train[c].isna().any())
    chart_list = ["missingness_bar"]
    if n_mc >= 2:
        chart_list.append("pattern_matrix")
    if target_col:
        chart_list.append("target_signal")

if chart_rationale:
    st.caption(f"🤖 _Claude selected these charts: {chart_rationale}_")

viz = MissingnessVisualizer(df_train, target_col=target_col)
figs = viz.generate_figures(charts=chart_list)
items = list(figs.items())

if len(items) == 1:
    name, fig = items[0]
    st.markdown(f"**{_CHART_TITLES.get(name, name)}**")
    st.pyplot(fig, bbox_inches="tight")
elif len(items) == 2:
    col_a, col_b = st.columns(2)
    for col, (name, fig) in zip([col_a, col_b], items):
        with col:
            st.markdown(f"**{_CHART_TITLES.get(name, name)}**")
            st.pyplot(fig, bbox_inches="tight")
else:
    col_a, col_b = st.columns(2)
    with col_a:
        name, fig = items[0]
        st.markdown(f"**{_CHART_TITLES.get(name, name)}**")
        st.pyplot(fig, bbox_inches="tight")
    with col_b:
        name, fig = items[1]
        st.markdown(f"**{_CHART_TITLES.get(name, name)}**")
        st.pyplot(fig, bbox_inches="tight")
    for name, fig in items[2:]:
        st.markdown(f"**{_CHART_TITLES.get(name, name)}**")
        st.pyplot(fig, bbox_inches="tight")

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
