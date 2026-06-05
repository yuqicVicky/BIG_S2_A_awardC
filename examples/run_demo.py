"""Demo: run the full Data Pattern & Validation Auditor on within-period toy data.

Usage:
    cd data-pattern-validation-auditor
    python examples/run_demo.py
"""

from __future__ import annotations

import os
import sys
import json

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_ROOT, "..", "src"))
sys.path.insert(0, os.path.join(_ROOT, ".."))

from examples.make_toy_within_period_data import make_within_period_data
from pattern_auditor import PatternAuditor

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")


def main():
    print("=" * 60)
    print("  Data Pattern & Validation Auditor — Demo")
    print("=" * 60)

    # --- Generate toy data ---
    print("\n[1/4] Generating within-period toy dataset …")
    train_df, predict_df = make_within_period_data(n_entities=50, n_periods=30)
    print(f"      Train: {train_df.shape}  |  Predict: {predict_df.shape}")
    print(f"      Train timestamps: {train_df['timestamp'].min()} → {train_df['timestamp'].max()}")
    print(f"      Predict timestamps: {predict_df['timestamp'].min()} → {predict_df['timestamp'].max()}")

    # --- Run auditor ---
    print("\n[2/4] Running PatternAuditor …")
    auditor = PatternAuditor(
        train_df,
        predict_df,
        target_col="target",
        row_id_col=None,
        datetime_col="timestamp",
        group_col="entity_id",
    )
    results = auditor.run(output_dir=OUTPUT_DIR)
    print("      Audit complete.")

    # --- Print validation recommendation ---
    print("\n[3/4] Validation Recommendation")
    print("-" * 40)
    val = results["validation"]
    sp = val.get("detected_split_pattern", "N/A")
    rec = val.get("recommendation", {})
    print(f"  Detected split pattern : {sp}")
    print(f"  Recommended strategy   : {rec.get('strategy', 'N/A')}")
    print(f"  Confidence             : {rec.get('confidence', 'N/A')}")
    print(f"  Reason                 : {rec.get('reason', '')}")
    print(f"  Fit rule               : {rec.get('fit_rule', '')}")
    print(f"  Validation rule        : {rec.get('validation_rule', '')}")
    avoid = rec.get("avoid", [])
    if avoid:
        print(f"  Strategies to avoid    : {', '.join(avoid)}")
    alts = val.get("alternatives", [])
    if alts:
        print(f"  Alternatives           : {', '.join(a['strategy'] for a in alts)}")

    # --- Print leakage summary ---
    print("\n[4/4] Leakage Summary")
    print("-" * 40)
    leak = results["leakage"]
    summary = leak.get("summary", {})
    print(f"  High risk  : {summary.get('high', 0)}")
    print(f"  Medium risk: {summary.get('medium', 0)}")
    print(f"  Low risk   : {summary.get('low', 0)}")
    high_risks = [r for r in leak.get("risks", []) if r["severity"] == "high"]
    for r in high_risks:
        print(f"  ⚠  {r['column']}: {r['risk_type']}")

    # --- List outputs ---
    print("\n" + "=" * 60)
    print("  Generated Outputs")
    print("=" * 60)
    for root, dirs, files in os.walk(OUTPUT_DIR):
        for f in sorted(files):
            full = os.path.join(root, f)
            rel = os.path.relpath(full, OUTPUT_DIR)
            size = os.path.getsize(full)
            print(f"  {rel:55s}  {size:>8,} bytes")

    print("\nDone. Open outputs/reports/pattern_audit.md for the full report.")


if __name__ == "__main__":
    main()
