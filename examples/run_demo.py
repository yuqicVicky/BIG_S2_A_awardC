"""Demo: run the full Data Pattern & Validation Auditor on within-period toy data.

Usage:
    cd data-pattern-validation-auditor
    python examples/run_demo.py
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_ROOT, "..", "src"))
sys.path.insert(0, os.path.join(_ROOT, ".."))

from examples.make_toy_within_period_data import make_within_period_data
from pattern_auditor import PatternAuditor

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")

_SEP = "=" * 64
_SUBSEP = "-" * 64


def _hr(title: str = ""):
    if title:
        pad = (64 - len(title) - 2) // 2
        print(f"{'─' * pad} {title} {'─' * (64 - pad - len(title) - 2)}")
    else:
        print(_SUBSEP)


def main():
    print(_SEP)
    print("  Data Pattern & Validation Auditor  —  Demo")
    print("  Scenario: Bike-sharing demand  |  within-period split")
    print(_SEP)

    # ── 1. Generate toy data ──────────────────────────────────────────────
    print("\n[Step 1/5]  Generating within-period toy dataset …")
    train_df, predict_df = make_within_period_data(n_entities=50, n_periods=30)
    print(f"  Train  : {train_df.shape[0]:>5} rows × {train_df.shape[1]} cols")
    print(f"  Predict: {predict_df.shape[0]:>5} rows × {predict_df.shape[1]} cols")
    print(f"  Train timestamps : {train_df['timestamp'].min()} → {train_df['timestamp'].max()}")
    print(f"  Predict timestamps: {predict_df['timestamp'].min()} → {predict_df['timestamp'].max()}")

    # ── 2. Run auditor ────────────────────────────────────────────────────
    print(f"\n[Step 2/5]  Running PatternAuditor …")
    auditor = PatternAuditor(
        train_df,
        predict_df,
        target_col="target",
        row_id_col=None,
        datetime_col="timestamp",
        group_col="entity_id",
    )
    results = auditor.run(output_dir=OUTPUT_DIR)
    print("  Audit complete.")

    # ── 3. Detected Pattern ───────────────────────────────────────────────
    print(f"\n[Step 3/5]  Train / Prediction Pattern")
    _hr()
    tpp = results["train_prediction_pattern"]
    pattern = tpp.get("pattern", "unknown")
    print(f"  Detected pattern : {pattern}")
    print(f"  Evidence:")
    for ev in tpp.get("evidence", [])[:5]:
        print(f"    • {ev}")
    schema_mismatch = tpp.get("schema_mismatch", {})
    train_only = [c for c in schema_mismatch.get("train_only_columns", []) if c != auditor.target_col]
    if train_only:
        print(f"  Train-only columns (leakage candidates): {train_only}")
    shift_info = tpp.get("numeric_distribution_shift", {})
    n_shift = shift_info.get("shift_detected_count", 0)
    n_total = shift_info.get("total_columns_tested", 0)
    print(f"  Numeric shift: {n_shift}/{n_total} columns have significant shift")
    if shift_info.get("columns_with_shift"):
        print(f"    Shifted: {shift_info['columns_with_shift']}")

    # ── 4. Validation Recommendation ─────────────────────────────────────
    print(f"\n[Step 4/5]  Validation Recommendation")
    _hr()
    val = results["validation"]
    rec = val.get("recommendation", {})
    print(f"  Recommended strategy  : {rec.get('strategy', '?')}")
    print(f"  Confidence            : {rec.get('confidence', '?')}")
    print()
    print(f"  Why this pattern is risky:")
    why = val.get("why_generic_validation_is_risky", "")
    for line in _wrap(why, 56):
        print(f"    {line}")
    print()
    print(f"  Fit rule       : {rec.get('fit_rule', '')}")
    print(f"  Validate on    : {rec.get('validation_rule', '')}")
    avoid = rec.get("strategies_to_avoid", [])
    if avoid:
        print(f"  Avoid          : {', '.join(avoid)}")
    alts = val.get("alternatives", [])
    if alts:
        print(f"  Alternatives   : {', '.join(a['strategy'] for a in alts)}")

    # ── 5. Leakage, Target Patterns, Features ────────────────────────────
    print(f"\n[Step 5/5]  Leakage + Target Patterns + Feature Recommendations")
    _hr("Leakage Audit")
    leak = results["leakage"]
    summary = leak.get("summary", {})
    print(f"  High risk: {summary.get('high', 0)}  |  Medium: {summary.get('medium', 0)}  |  Low: {summary.get('low', 0)}")
    high_risks = [r for r in leak.get("risks", []) if r["severity"] == "high"]
    if high_risks:
        print("  High-risk columns (exclude from features):")
        for r in high_risks:
            print(f"    ⚠  {r['column']:25s}  [{r['risk_type']}]")
    else:
        print("  No high-risk leakage columns detected.")

    _hr("Top Target Patterns")
    tp = results["target_patterns"]
    top_corrs = sorted(
        [c for c in tp.get("numeric_correlations", []) if c.get("pearson_r") is not None],
        key=lambda x: abs(x["pearson_r"]),
        reverse=True,
    )[:5]
    if top_corrs:
        print("  Top numeric correlations with target:")
        for c in top_corrs:
            bar = "█" * int(abs(c["pearson_r"]) * 20)
            print(f"    {c['column']:20s}  r={c['pearson_r']:+.3f}  {bar}")

    dt_pats = tp.get("datetime_target_patterns", [])
    if dt_pats:
        print("  Datetime patterns detected:")
        for dp in dt_pats:
            comps = list(dp.get("by_component", {}).keys())
            print(f"    {dp['column']} → {', '.join(comps)}")

    _hr("Feature Recommendations")
    fe = results["feature_engineering"]
    to_add = fe.get("features_to_add", [])
    to_excl = fe.get("features_to_exclude", [])
    interactions = fe.get("interactions_to_try", [])
    transforms = fe.get("target_transforms_to_try", [])
    fe_warns = fe.get("warnings", [])

    if to_add:
        print(f"  Features to add ({len(to_add)} total, showing first 6):")
        for f in to_add[:6]:
            print(f"    + {f['feature']:30s}  [{f['type']}]")
    if to_excl:
        print(f"  Features to exclude:")
        for f in to_excl[:6]:
            print(f"    ✗ {f['column']:30s}  {f['reason']}")
    if interactions:
        print(f"  Interactions to try:")
        for i in interactions[:4]:
            print(f"    ↔ {i['interaction']:30s}  [{i['type']}]")
    if transforms:
        print(f"  Target transforms to try:")
        for t in transforms:
            print(f"    ~ {t['transform']:30s}  {t['reason'][:50]}")
    if fe_warns:
        print(f"  Warnings:")
        for w in fe_warns:
            print(f"    ⚡ {w['type']}: {w['message'][:60]}")

    # ── Generated Outputs ─────────────────────────────────────────────────
    print(f"\n{_SEP}")
    print("  Generated Outputs")
    print(_SEP)
    fig_dir = os.path.join(OUTPUT_DIR, "figures")
    log_dir = os.path.join(OUTPUT_DIR, "logs")
    report_dir = os.path.join(OUTPUT_DIR, "reports")

    for label, d in [("Figures", fig_dir), ("JSON logs", log_dir), ("Reports", report_dir)]:
        if os.path.isdir(d):
            files = sorted(os.listdir(d))
            print(f"\n  {label}:")
            for f in files:
                full = os.path.join(d, f)
                size = os.path.getsize(full)
                print(f"    {f:50s}  {size:>8,} bytes")

    print(f"\n{'─' * 64}")
    print("  Open outputs/reports/pattern_audit.md for the full narrative.")
    print(f"{'─' * 64}\n")


def _wrap(text: str, width: int) -> list[str]:
    words = text.split()
    lines = []
    current = ""
    for w in words:
        if len(current) + len(w) + 1 > width:
            lines.append(current)
            current = w
        else:
            current = f"{current} {w}".strip()
    if current:
        lines.append(current)
    return lines


if __name__ == "__main__":
    main()
