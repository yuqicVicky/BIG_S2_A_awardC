"""CLI entry point: python -m pattern_auditor.cli --train ... --predict ... --target ..."""

from __future__ import annotations

import argparse
import json
import sys
import pandas as pd

from . import PatternAuditor


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="pattern_auditor",
        description="Data Pattern & Validation Auditor — understand data before modeling.",
    )
    parser.add_argument("--train", required=True, help="Path to training CSV file.")
    parser.add_argument("--predict", default=None, help="Path to prediction CSV file.")
    parser.add_argument("--target", default=None, help="Target column name.")
    parser.add_argument("--row-id", dest="row_id", default=None, help="Row ID column name.")
    parser.add_argument("--datetime", dest="datetime_col", default=None, help="Primary datetime column name.")
    parser.add_argument("--group", dest="group_col", default=None, help="Group/entity column name.")
    parser.add_argument("--out", default="outputs/", help="Output directory (default: outputs/).")
    parser.add_argument(
        "--json-summary",
        action="store_true",
        help="Print JSON validation recommendation to stdout after run.",
    )

    args = parser.parse_args(argv)

    print(f"[PatternAuditor] Loading train data from: {args.train}")
    train_df = pd.read_csv(args.train)

    predict_df = None
    if args.predict:
        print(f"[PatternAuditor] Loading predict data from: {args.predict}")
        predict_df = pd.read_csv(args.predict)

    auditor = PatternAuditor(
        train_df,
        predict_df,
        target_col=args.target,
        row_id_col=args.row_id,
        datetime_col=args.datetime_col,
        group_col=args.group_col,
    )

    print(f"[PatternAuditor] Running audit → outputs in: {args.out}")
    results = auditor.run(output_dir=args.out)

    rec = results.get("validation", {}).get("recommendation", {})
    strategy = rec.get("strategy", "N/A")
    confidence = rec.get("confidence", "N/A")
    print(f"\n[PatternAuditor] Recommended validation strategy: {strategy!r}  (confidence: {confidence})")
    print(f"[PatternAuditor] Report written to: {args.out}reports/pattern_audit.md")

    if args.json_summary:
        print(json.dumps(results["validation"], indent=2))

    return results


if __name__ == "__main__":
    main()
