"""CLI entry point: python -m missingness_auditor.cli --data ... --target ... --out ..."""

from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

from .auditor import MissingnessAuditor


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="missingness_auditor",
        description=(
            "Missingness Audit & Imputation Planner — "
            "diagnose missing data before imputation."
        ),
    )
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument("--data",  help="Path to a single CSV file.")
    data_group.add_argument("--train", help="Path to training CSV file (use with --predict).")
    parser.add_argument("--predict", default=None, help="Path to prediction CSV (used with --train).")
    parser.add_argument("--target", default=None, help="Target column name.")
    parser.add_argument("--out", default="outputs/", help="Output directory (default: outputs/).")
    parser.add_argument(
        "--json-summary", action="store_true",
        help="Print imputation_plan JSON to stdout after run.",
    )
    args = parser.parse_args(argv)

    if args.data:
        print(f"[MissingnessAuditor] Loading data from: {args.data}")
        df = pd.read_csv(args.data)
        predict_df = None
    else:
        print(f"[MissingnessAuditor] Loading train from: {args.train}")
        df = pd.read_csv(args.train)
        predict_df = None
        if args.predict:
            print(f"[MissingnessAuditor] Loading predict from: {args.predict}")
            predict_df = pd.read_csv(args.predict)

    auditor = MissingnessAuditor(df, predict_df, target_col=args.target)

    print(f"[MissingnessAuditor] Running audit…")
    results = auditor.run()

    print(f"[MissingnessAuditor] Writing outputs → {args.out}")
    auditor.save_outputs(results, args.out)

    plan = results["imputation_plan"]
    summary = plan.get("summary", {})
    print(f"\n[MissingnessAuditor] Columns requiring imputation: {summary.get('n_columns_to_impute', 0)}")
    for strat, cnt in sorted(summary.get("strategy_counts", {}).items(), key=lambda x: -x[1]):
        print(f"  {strat}: {cnt}")

    lk = results["leakage_safe_check"].get("summary", {})
    print(f"\n[MissingnessAuditor] Leakage risk: "
          f"high={lk.get('high_leakage_risk', 0)}, "
          f"medium={lk.get('medium_leakage_risk', 0)}")
    print(f"[MissingnessAuditor] Report: {os.path.join(args.out, 'reports', 'missing_data_report.md')}")

    if args.json_summary:
        print(json.dumps(plan, indent=2))

    return results


if __name__ == "__main__":
    main()
