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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from missingness_auditor import MissingnessAuditor
from missingness_auditor.visualization import MissingnessVisualizer
from missingness_auditor.reporting import ReportWriter
from missingness_auditor.mice import mice_pool_column_means
from missingness_auditor.sensitivity import mnar_sensitivity, plot_tipping_point

st.set_page_config(
    page_title="AI Missingness Auditor",
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
        "playground_df", "playground_strategies",
        "last_applied_source",
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
            model="claude-sonnet-4-6",
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

    st.markdown(f"**{label}**")
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
            model="claude-sonnet-4-6",
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
            model="claude-sonnet-4-6",
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


# ──────────────────── playground & before/after helpers ──────────────────────

_PLAN_TO_PLAYGROUND: dict[str, str] = {
    "numeric_median":                            "median",
    "numeric_median_plus_indicator":             "median",
    "groupwise_numeric_median_plus_indicator":   "groupwise_median",
    "group_median":                              "groupwise_median",
    "time_series_ffill_bfill_plus_indicator":    "median",
    "categorical_missing_token":                 "missing_token",
    "categorical_missing_token_plus_indicator":  "missing_token",
    "categorical_mode_plus_indicator":           "mode",
    "structural_none_token_plus_indicator":      "missing_token",
    "structural_zero_plus_indicator":            "median",
    "structural_none_or_zero":                   "median",
    "model_based_imputation_optional":           "mice",
    "no_imputation_needed":                      "none",
    "drop_column":                               "drop",
}


def _apply_playground_impute(
    df_train: pd.DataFrame,
    col: str,
    strategy: str,
    group_col: str | None = None,
) -> tuple[pd.Series | None, bool]:
    """Fit+apply a single-column imputation strategy on train data."""
    series = df_train[col].copy()
    is_num = pd.api.types.is_numeric_dtype(series)

    if strategy == "drop":
        return None, True
    if strategy == "none":
        return series, False

    if strategy == "mean" and is_num:
        return series.fillna(series.mean()), False
    if strategy == "median":
        return series.fillna(series.median() if is_num else series.mode().iloc[0] if series.mode().size else "MISSING"), False
    if strategy == "knn" and is_num:
        try:
            from sklearn.impute import KNNImputer
            num_cols = df_train.select_dtypes(include="number").columns.tolist()
            imp = KNNImputer(n_neighbors=5)
            arr = imp.fit_transform(df_train[num_cols].values.astype(float))
            return pd.Series(arr[:, num_cols.index(col)], index=df_train.index, name=col), False
        except Exception:
            return series.fillna(series.median()), False
    if strategy == "mice" and is_num:
        try:
            from sklearn.experimental import enable_iterative_imputer  # noqa: F401
            from sklearn.impute import IterativeImputer
            num_cols = df_train.select_dtypes(include="number").columns.tolist()
            imp = IterativeImputer(max_iter=10, random_state=42)
            arr = imp.fit_transform(df_train[num_cols].values.astype(float))
            return pd.Series(arr[:, num_cols.index(col)], index=df_train.index, name=col), False
        except Exception:
            return series.fillna(series.median()), False
    if strategy == "groupwise_median" and is_num:
        if group_col and group_col in df_train.columns:
            gmap = df_train.groupby(group_col)[col].median().to_dict()
            filled = series.fillna(df_train[group_col].map(gmap))
            return filled.fillna(series.median()), False
        return series.fillna(series.median()), False
    if strategy == "missing_token":
        return series.fillna("MISSING"), False
    if strategy == "mode":
        modes = series.mode()
        fill = modes.iloc[0] if len(modes) > 0 else "MISSING"
        return series.fillna(fill), False

    # fallback
    return (series.fillna(series.median()) if is_num else series.fillna("MISSING")), False


def _render_numeric_comparison(before: pd.Series, after: pd.Series, col: str) -> None:
    bv = before.dropna().values.astype(float)
    av = after.dropna().values.astype(float)
    if len(bv) == 0 or len(av) == 0:
        st.info("Not enough data to compare.")
        return

    def fmt(x: float) -> str:
        return f"{x:.4g}"

    stats_df = pd.DataFrame({
        "Stat":   ["Mean", "Median", "Std Dev", "Min", "Max"],
        "Before": [fmt(bv.mean()), fmt(float(np.median(bv))), fmt(bv.std()), fmt(bv.min()), fmt(bv.max())],
        "After":  [fmt(av.mean()), fmt(float(np.median(av))), fmt(av.std()), fmt(av.min()), fmt(av.max())],
    })
    st.dataframe(stats_df, use_container_width=False, hide_index=True)

    mean_shift = abs(av.mean() - bv.mean()) / (abs(bv.mean()) + 1e-9)
    std_ratio = av.std() / (bv.std() + 1e-9)

    try:
        from scipy.stats import ks_2samp
        ks_stat, ks_p = ks_2samp(bv, av)
    except ImportError:
        ks_stat, ks_p = 0.0, 1.0

    alerts = []
    if mean_shift > 0.10:
        alerts.append(f"Mean shifted {mean_shift:.1%} — imputed fill may not be representative.")
    if std_ratio < 0.80:
        alerts.append(f"Std dev shrank {(1-std_ratio):.1%} — imputation compressed the spread.")
    if ks_p < 0.05:
        alerts.append(f"KS test: distributions differ significantly (p={ks_p:.3f}, stat={ks_stat:.3f}).")
    if alerts:
        for a in alerts:
            st.warning(a)
    else:
        st.success("Distribution stable — no significant drift detected.")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3), sharey=False)
    axes[0].hist(bv, bins=30, color="#4C78A8", edgecolor="white", alpha=0.9)
    axes[0].set_title(f"Before  (n={len(bv):,})")
    axes[0].set_xlabel(col)
    axes[0].set_ylabel("Count")
    axes[1].hist(av, bins=30, color="#72B7B2", edgecolor="white", alpha=0.9)
    axes[1].set_title(f"After  (n={len(av):,})")
    axes[1].set_xlabel(col)
    axes[1].set_ylabel("Count")
    plt.tight_layout()
    st.pyplot(fig, bbox_inches="tight")
    plt.close(fig)


def _render_categorical_comparison(before: pd.Series, after: pd.Series, col: str) -> None:
    TOP_N = 8
    bvc = before.value_counts(dropna=True)
    avc = after.value_counts(dropna=False)
    missing_pct_before = before.isna().mean()
    missing_token_pct = (after == "MISSING").mean()

    ca, cb = st.columns(2)
    with ca:
        st.markdown(f"**Before** — {missing_pct_before:.1%} missing")
        tbl = bvc.head(TOP_N).reset_index()
        tbl.columns = ["Category", "Count"]
        n_valid = len(before.dropna()) or 1
        tbl["Pct"] = (tbl["Count"] / n_valid * 100).map("{:.1f}%".format)
        st.dataframe(tbl, use_container_width=True, hide_index=True)
    with cb:
        st.markdown(f"**After** — {missing_token_pct:.1%} MISSING token")
        tbl2 = avc.head(TOP_N).reset_index()
        tbl2.columns = ["Category", "Count"]
        n_total = len(after) or 1
        tbl2["Pct"] = (tbl2["Count"] / n_total * 100).map("{:.1f}%".format)
        st.dataframe(tbl2, use_container_width=True, hide_index=True)

    if missing_token_pct > 0.30:
        st.warning(
            f"MISSING token is {missing_token_pct:.1%} of all values "
            "— check that your model handles this category meaningfully."
        )

    valid_before = before.dropna()
    if len(valid_before) > 0 and len(valid_before.mode()) > 0:
        mode_val = valid_before.mode().iloc[0]
        pct_b = (before == mode_val).mean()
        pct_a = (after == mode_val).mean()
        if pct_a > 2 * pct_b and pct_a > 0.30:
            st.warning(
                f"Mode fill caused `{mode_val}` to inflate from {pct_b:.1%} → {pct_a:.1%} "
                "— risk of class-imbalance bias in downstream model."
            )

    cats = list(set(bvc.head(TOP_N).index) | set(avc.head(TOP_N).index))
    n_b = len(before.dropna()) or 1
    n_a = len(after) or 1
    bvals = [bvc.get(c, 0) / n_b for c in cats]
    avals = [avc.get(c, 0) / n_a for c in cats]

    x = np.arange(len(cats))
    w = 0.35
    fig, ax = plt.subplots(figsize=(max(7, len(cats) * 1.2), 3.5))
    ax.bar(x - w / 2, bvals, w, label="Before", color="#4C78A8", alpha=0.9)
    ax.bar(x + w / 2, avals, w, label="After",  color="#72B7B2", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([str(c)[:12] for c in cats], rotation=30, ha="right")
    ax.set_ylabel("Proportion")
    ax.set_title(f"{col} — Category Distribution Before vs After")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig, bbox_inches="tight")
    plt.close(fig)


def _render_before_after_section(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    profile: dict,
    target_col: str | None,
    label: str = "Before vs After Simulation",
) -> None:
    st.subheader(label)

    imputed_cols = [
        col for col in df_before.columns
        if col != target_col and df_before[col].isna().any()
    ]
    dropped = [c for c in df_before.columns if c not in df_after.columns and c != target_col]
    if dropped:
        st.info(f"Dropped columns: {', '.join(f'`{c}`' for c in dropped)}")

    if not imputed_cols:
        st.info("No missing values were present before imputation.")
        return

    for col in imputed_cols:
        dtype_cat = profile["columns"].get(col, {}).get("dtype_category", "numeric")
        n_miss = int(df_before[col].isna().sum())

        if col not in df_after.columns:
            st.markdown(f"**{col}** — dropped ({n_miss} missing values removed)")
            continue

        n_remain = int(df_after[col].isna().sum())
        with st.expander(f"**{col}** — {dtype_cat} | {n_miss} missing → {n_remain} remaining", expanded=True):
            if dtype_cat == "numeric":
                _render_numeric_comparison(df_before[col], df_after[col], col)
            else:
                _render_categorical_comparison(df_before[col], df_after[col], col)


# ──────────────────────────────────────────────── sidebar ────────────────────
with st.sidebar:
    st.title("AI Missingness Auditor")
    st.caption("Claude narrates every step of your missing-data analysis.")

    st.divider()
    api_key_input = st.text_input(
        "Anthropic API Key",
        type="password",
        placeholder="sk-ant-...",
        help="Required for AI-guided analysis.",
    )
    if api_key_input:
        st.success("API key ready")
    else:
        st.info("Add an API key for AI analysis.")

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
        run_btn = st.button("Run AI Audit", type="primary", use_container_width=True)
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
    _run_llm_client = anthropic.Anthropic(api_key=api_key_input) if api_key_input else None
    try:
        with st.spinner("Running missingness audit…"):
            auditor = MissingnessAuditor(
                df_train,
                predict_df=df_predict,
                target_col=target_col,
                llm_client=_run_llm_client,
            )
            results = auditor.run()
    except Exception as _audit_err:
        st.error(f"Audit failed: {type(_audit_err).__name__}: {_audit_err}")
        import traceback
        st.code(traceback.format_exc(), language="python")
        st.stop()
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
               "chat_history", "imputation_summary", "chart_list", "chart_rationale",
               "playground_df", "playground_strategies", "last_applied_source"):
        st.session_state.pop(k, None)

# ────────────────────────────────────────────── landing page ─────────────────
if "audit_results" not in st.session_state:
    st.title("AI-Guided Missingness Audit")
    st.markdown(
        "Upload your dataset and let **Claude** walk you through every step: "
        "profile → mechanisms → structural patterns → imputation plan."
    )
    st.info("Select a dataset in the sidebar and click **Run AI Audit** to begin.")
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
            viz_tmp = MissingnessVisualizer(df_train, target_col=target_col, llm_client=client)
            chart_list, chart_rationale = viz_tmp.suggest_charts(ctx, bool(target_col), n_missing_cols)
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

# ─── Main tabs ────────────────────────────────────────────────────────────────
tab_audit, tab_impute, tab_viz, tab_chat = st.tabs([
    "Audit",
    "Imputation",
    "Visualizations",
    "Ask Claude",
])

# ── Tab 1: Audit ──────────────────────────────────────────────────────────────
with tab_audit:
    st.subheader("Missingness Profile")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Column": col,
                    "Missing Rate": f"{info['missing_rate']:.1%}",
                    "Severity": info["severity"],
                    "Dtype": info["dtype_category"],
                    "Is Target": "Yes" if info.get("is_target") else "",
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

    st.subheader("Missingness Mechanisms")
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

    # Per-column LLM narratives from MechanismAuditor
    llm_mech_cols = [
        (col, info) for col, info in mech["columns"].items()
        if info.get("llm_narrative")
    ]
    if llm_mech_cols:
        with st.expander("AI — per-column mechanism narratives", expanded=True):
            for col, info in llm_mech_cols:
                st.markdown(f"**`{col}`** — {info['llm_narrative']}")
                if info.get("llm_suggestion"):
                    st.caption(f"Suggestion: {info['llm_suggestion']}")

    if struct["n_structural_pairs"] > 0:
        st.divider()
        st.subheader("Structural Absence")
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

        # LLM explanations per pair from StructuralMissingnessDetector
        if struct.get("llm_summary"):
            st.info(struct["llm_summary"])
        llm_pair_exps = [p for p in struct["structural_pairs"] if p.get("llm_explanation")]
        if llm_pair_exps:
            with st.expander("AI — structural pair explanations"):
                for p in llm_pair_exps:
                    st.markdown(
                        f"**`{p['categorical_col']}` → `{p['numeric_col']}`**: "
                        f"{p['llm_explanation']}"
                    )

# ── Tab 2: Imputation ─────────────────────────────────────────────────────────
with tab_impute:
    # Plan summary (read-only)
    if leakage.get("high_risk_count", 0) + leakage.get("medium_risk_count", 0) > 0:
        st.warning(
            f"Leakage check: {leakage.get('high_risk_count', 0)} high-risk, "
            f"{leakage.get('medium_risk_count', 0)} medium-risk finding(s)."
        )
    else:
        st.success("Leakage check passed — safe to proceed.")

    # ── Decision Cards — one card per column with missing values ──────────────
    # Each card surfaces the judgment call (mechanism + recommended strategy +
    # why + alternatives) the way a reviewer would want to see it before approving.
    _card_cols = [
        (col, entry) for col, entry in plan["columns"].items()
        if entry["strategy"] != "no_imputation_needed"
    ]
    if _card_cols:
        st.subheader("Decision Cards")
        st.caption(
            "One card per column with missing values — the recommended call, why it "
            "was made, and the alternatives. Statistics are fitted on training data "
            "only; the target column is never imputed."
        )
        for _i in range(0, len(_card_cols), 2):
            _row = _card_cols[_i:_i + 2]
            _cols = st.columns(len(_row))
            for _slot, (_col, _entry) in zip(_cols, _row):
                with _slot:
                    with st.container(border=True):
                        _prof = profile["columns"].get(_col, {})
                        _sev = _prof.get("severity", "—")
                        _rate = _entry.get("missing_rate", _prof.get("missing_rate", 0))
                        st.markdown(f"#### `{_col}`")
                        st.markdown(
                            f"**{_rate:.1%} missing** "
                            f"· {_sev} severity · {_prof.get('dtype_category', '—')}"
                        )
                        st.markdown(f"**Mechanism clue:** {_entry.get('mechanism_label', '—')}")
                        st.markdown(f"**Recommended:** `{_entry['strategy']}`")
                        _bits = []
                        if _entry.get("group_col"):
                            _bits.append(f"grouped by `{_entry['group_col']}`")
                        if _entry.get("add_missing_indicator"):
                            _bits.append("+ missing indicator")
                        if _entry.get("mi_upgrade_recommended"):
                            _bits.append("MI upgrade advised")
                        if _bits:
                            st.caption(" · ".join(_bits))
                        _why = _entry.get("llm_readable_reason") or _entry.get("reason", "")
                        if _why:
                            st.markdown(f"*Why:* {_why}")
                        _alts = _entry.get("llm_alternatives", [])
                        if _alts:
                            with st.expander("Alternatives"):
                                for _a in _alts:
                                    st.markdown(f"- {_a}")

    with st.expander("Plan as table"):
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

    # ── Inference-grade analysis: multiple imputation + MNAR sensitivity ──────
    with st.expander("Inference-grade analysis — multiple imputation & MNAR sensitivity"):
        st.caption(
            "Single imputation understates variance; MAR cannot be verified from data. "
            "These two analyses quantify what the plan above cannot: the extra "
            "uncertainty from missingness (Rubin's rules) and how robust a conclusion "
            "is to MNAR departures (delta-adjustment)."
        )
        if st.button("Run MI + MNAR sensitivity", key="run_inference"):
            with st.spinner("Running multiple imputation and sensitivity analysis…"):
                try:
                    _mice = mice_pool_column_means(df_train, target_col=target_col)
                    if _mice.get("columns"):
                        st.markdown("**Multiple imputation (Rubin pooling) — per-column mean**")
                        st.dataframe(
                            pd.DataFrame([
                                {
                                    "Column": c,
                                    "Pooled mean": round(p["pooled_estimate"], 4),
                                    "Naive SE": round(p["naive_single_imputation_std_error"], 4),
                                    "MI SE": round(p["std_error"], 4),
                                    "FMI": round(p["fraction_missing_information"], 3),
                                }
                                for c, p in _mice["columns"].items()
                            ]),
                            use_container_width=True, hide_index=True,
                        )
                        st.caption("MI SE ≥ Naive SE by construction — that gap is the "
                                   "uncertainty single imputation hides.")
                    _sens = mnar_sensitivity(df_train, plan, target_col=target_col)
                    if _sens.get("columns"):
                        st.markdown("**MNAR sensitivity — delta-adjustment tipping points**")
                        st.dataframe(
                            pd.DataFrame([
                                {
                                    "Column": c,
                                    "Missing": f"{a['missing_rate']:.1%}",
                                    "Tipping |δ| (SD)": (abs(a["tipping_point_delta_sd"])
                                                          if a["tipping_point_delta_sd"] is not None
                                                          else "robust"),
                                    "Robustness": a["robustness"],
                                }
                                for c, a in _sens["columns"].items()
                            ]),
                            use_container_width=True, hide_index=True,
                        )
                        if _sens.get("fragile_columns"):
                            st.warning("Fragile (MAR-sensitive): "
                                       + ", ".join(_sens["fragile_columns"]))
                        import tempfile
                        _fig_path = os.path.join(tempfile.mkdtemp(), "tip.png")
                        if plot_tipping_point(_sens, _fig_path):
                            st.image(_fig_path)
                except Exception as _exc:
                    st.error(f"Analysis failed: {_exc}")

    _insight("plan", "Plan Evaluation")

    # Per-column LLM readable reasons from ImputationPlanner
    llm_plan_cols = [
        (col, entry) for col, entry in plan["columns"].items()
        if entry.get("llm_readable_reason")
    ]
    if llm_plan_cols:
        with st.expander("AI — why each strategy was chosen", expanded=False):
            for col, entry in llm_plan_cols:
                st.markdown(f"**`{col}`** (`{entry['strategy']}`): {entry['llm_readable_reason']}")
                alts = entry.get("llm_alternatives", [])
                if alts:
                    st.caption("Alternatives: " + " · ".join(alts))

    with st.expander("Leakage-safe sklearn pattern"):
        proto = leakage.get("global_protocol", {})
        st.markdown(f"**Rule:** {proto.get('rule', '')}")
        st.code(proto.get("sklearn_pattern", ""), language="python")

    st.divider()

    # Strategy override (replaces separate playground section)
    st.subheader("Strategy Override")
    st.caption(
        "Leave dropdowns at **recommended** to use the plan above, or override per column. "
        "**Apply Recommended Plan** runs the full auditor pipeline; "
        "**Apply Custom** applies your overrides."
    )

    _missing_cols_pg = [
        col for col in df_train.columns
        if col != target_col and df_train[col].isna().any()
    ]

    if not _missing_cols_pg:
        st.info("No columns with missing values — nothing to impute.")
    else:
        _NUMERIC_OPTS = [
            "recommended",
            "mean",
            "median",
            "KNN  (k=5)",
            "MICE  (IterativeImputer)",
            "groupwise median",
            "drop column",
        ]
        _CAT_OPTS = [
            "recommended",
            "MISSING token",
            "mode fill",
            "drop column",
        ]

        _pg_strategies: dict[str, tuple[str, str, str | None]] = {}

        header_cols = st.columns([2, 3, 3])
        header_cols[0].markdown("**Column**")
        header_cols[1].markdown("**AI Recommendation**")
        header_cols[2].markdown("**Your Choice**")
        st.divider()

        for _col in _missing_cols_pg:
            _col_info = profile["columns"].get(_col, {})
            _dtype_cat = _col_info.get("dtype_category", "numeric")
            _plan_entry = plan["columns"].get(_col, {})
            _rec = _plan_entry.get("strategy", "numeric_median")
            _group_col = _plan_entry.get("group_col")

            _opts = _NUMERIC_OPTS if _dtype_cat == "numeric" else _CAT_OPTS

            _c1, _c2, _c3 = st.columns([2, 3, 3])
            with _c1:
                _miss_pct = df_train[_col].isna().mean()
                st.markdown(f"**{_col}**")
                st.caption(f"{_dtype_cat} · {_miss_pct:.1%} missing")
            with _c2:
                st.code(_rec, language=None)
                if _group_col:
                    st.caption(f"group by `{_group_col}`")
            with _c3:
                _choice = st.selectbox(
                    f"Strategy — {_col}",
                    _opts,
                    key=f"pg_{_col}",
                    label_visibility="collapsed",
                )
            _pg_strategies[_col] = (_choice, _dtype_cat, _group_col)

        _btn1, _btn2 = st.columns(2)
        with _btn1:
            _apply_rec = st.button("Apply Recommended Plan", type="primary", use_container_width=True)
        with _btn2:
            _apply_custom = st.button("Apply Custom Strategies", use_container_width=True)

        if _apply_rec:
            with st.spinner("Applying imputation plan…"):
                result = auditor.apply_imputation(df_train, plan, df_predict)
                if isinstance(result, tuple):
                    st.session_state["imputed_df"], st.session_state["imputed_df_predict"] = result
                else:
                    st.session_state["imputed_df"] = result
            st.session_state["last_applied_source"] = "plan"

            # LLM distribution summary generated by Imputer (when llm_client was set)
            if auditor.llm_imputation_summary:
                st.session_state["imputation_summary"] = {
                    "severity": "low",
                    "headline": "Distribution comparison after imputation",
                    "bullets": [],
                    "recommendation": auditor.llm_imputation_summary,
                }
            elif api_key_input:
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

        if _apply_custom:
            with st.spinner("Applying custom strategies…"):
                _df_pg = df_train.copy()
                _drop_pg: list[str] = []

                for _col, (_choice, _dtype_cat, _group_col) in _pg_strategies.items():
                    if _choice == "recommended":
                        _rec_strat = _PLAN_TO_PLAYGROUND.get(
                            plan["columns"].get(_col, {}).get("strategy", "numeric_median"),
                            "median",
                        )
                        _strat_key = _rec_strat
                    elif _choice == "drop column":
                        _drop_pg.append(_col)
                        continue
                    else:
                        _strat_key = {
                            "mean":                        "mean",
                            "median":                      "median",
                            "KNN  (k=5)":                  "knn",
                            "MICE  (IterativeImputer)":    "mice",
                            "groupwise median":            "groupwise_median",
                            "MISSING token":               "missing_token",
                            "mode fill":                   "mode",
                        }.get(_choice, "median")

                    _filled, _dropped = _apply_playground_impute(_df_pg, _col, _strat_key, _group_col)
                    if _dropped:
                        _drop_pg.append(_col)
                    elif _filled is not None:
                        _df_pg[_col] = _filled

                if _drop_pg:
                    _df_pg = _df_pg.drop(columns=_drop_pg, errors="ignore")

                st.session_state["playground_df"] = _df_pg
                st.session_state["playground_strategies"] = {
                    col: vals[0] for col, vals in _pg_strategies.items()
                }
            st.session_state["last_applied_source"] = "playground"

    # Unified before/after — shows result of whichever apply was last run
    _last_source = st.session_state.get("last_applied_source")
    if _last_source == "plan" and "imputed_df" in st.session_state:
        st.divider()
        imp = st.session_state["imputed_df"]
        st.success(f"Imputation complete — {imp.shape[0]:,} rows × {imp.shape[1]} columns")
        if summ := st.session_state.get("imputation_summary"):
            _render_insight(summ, "Post-Imputation Check")
        remaining = int(imp.isna().sum().sum())
        if remaining:
            st.warning(
                f"{remaining} missing values remain "
                "(expected for drop_column / no_imputation_needed strategies)."
            )
        else:
            st.success("No missing values remain.")
        _render_before_after_section(df_train, imp, profile, target_col)
        with st.expander("View imputed data (first 20 rows)"):
            st.dataframe(imp.head(20), use_container_width=True)
        if "imputed_df_predict" in st.session_state:
            with st.expander("Predict set (imputed, first 20 rows)"):
                st.dataframe(st.session_state["imputed_df_predict"].head(20), use_container_width=True)

    elif _last_source == "playground" and "playground_df" in st.session_state:
        st.divider()
        _pg_df = st.session_state["playground_df"]
        _render_before_after_section(
            df_train, _pg_df, profile, target_col,
            label="Custom Strategies — Before vs After",
        )
        _remain_pg = int(_pg_df.isna().sum().sum())
        if _remain_pg:
            st.warning(f"{_remain_pg} missing values remain after custom strategies.")
        else:
            st.success("No missing values remain with your selected strategies.")
        with st.expander("View result (first 20 rows)"):
            st.dataframe(_pg_df.head(20), use_container_width=True)

# ── Tab 3: Visualizations ──────────────────────────────────────────────────────
with tab_viz:
    chart_list: list[str] = st.session_state.get("chart_list", [])
    chart_rationale: str = st.session_state.get("chart_rationale", "")

    if not chart_list:
        n_mc = sum(1 for c in df_train.columns if df_train[c].isna().any())
        chart_list = ["missingness_bar"]
        if n_mc >= 2:
            chart_list.append("pattern_matrix")
        if target_col:
            chart_list.append("target_signal")

    if chart_rationale:
        st.caption(f"_Claude selected these charts: {chart_rationale}_")

    _viz_client = anthropic.Anthropic(api_key=api_key_input) if api_key_input else None
    viz = MissingnessVisualizer(df_train, target_col=target_col, llm_client=_viz_client)
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

# ── Tab 4: Ask Claude ──────────────────────────────────────────────────────────
with tab_chat:
    if not api_key_input:
        st.info(
            "Add your Anthropic API key in the sidebar to ask follow-up questions about your data.",
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
                        model="claude-sonnet-4-6",
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

# ── Downloads (outside tabs) ───────────────────────────────────────────────────
st.divider()
st.subheader("Downloads")
_dl_client = anthropic.Anthropic(api_key=api_key_input) if api_key_input else None
rw = ReportWriter(llm_client=_dl_client)
report_md = rw.generate_report_md(profile, mech, struct, plan, leakage)

d1, d2, d3, d4 = st.columns(4)
with d1:
    st.download_button(
        "Markdown report",
        data=report_md.encode(),
        file_name="missing_data_report.md",
        mime="text/markdown",
    )
with d2:
    st.download_button(
        "imputation_plan.json",
        data=json.dumps(plan, indent=2).encode(),
        file_name="imputation_plan.json",
        mime="application/json",
    )
with d3:
    if "imputed_df" in st.session_state:
        st.download_button(
            "Imputed train CSV",
            data=_df_to_csv(st.session_state["imputed_df"]),
            file_name="train_imputed.csv",
            mime="text/csv",
        )
    else:
        st.button("Imputed train CSV", disabled=True, help="Apply imputation first")
with d4:
    if "imputed_df_predict" in st.session_state:
        st.download_button(
            "Imputed predict CSV",
            data=_df_to_csv(st.session_state["imputed_df_predict"]),
            file_name="predict_imputed.csv",
            mime="text/csv",
        )
    else:
        st.button(
            "Imputed predict CSV",
            disabled=True,
            help="Provide predict CSV and apply imputation",
        )
