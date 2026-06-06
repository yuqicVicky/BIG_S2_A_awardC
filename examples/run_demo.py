"""Train-Test Pattern Auditor — demo runner.

Usage:
    python examples/run_demo.py --case within_period --out outputs/demo_within_period
    python examples/run_demo.py --case iid           --out outputs/demo_iid
    python examples/run_demo.py --case group         --out outputs/demo_group
    python examples/run_demo.py --case future_time_split --out outputs/demo_time
    python examples/run_demo.py --case leakage       --out outputs/demo_leakage
"""

from __future__ import annotations

import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_ROOT, "..", "src"))
sys.path.insert(0, os.path.join(_ROOT, ".."))

from pattern_auditor import PatternAuditor

_SEP = "=" * 66
_SUBSEP = "-" * 66

# ── Case registry ──────────────────────────────────────────────────────────

def _cases() -> dict:
    from examples.make_within_period_demo import make_within_period_data, AUDITOR_KWARGS as _WP
    from examples.make_iid_demo import make_iid_data, AUDITOR_KWARGS as _IID
    from examples.make_group_demo import make_group_data, AUDITOR_KWARGS as _GRP
    from examples.make_time_split_demo import make_time_split_data, AUDITOR_KWARGS as _TIME
    from examples.make_leakage_demo import make_leakage_data, AUDITOR_KWARGS as _LEAK

    return {
        "within_period": {
            "label": "Case A — Within-Period Split",
            "description": "Days 1-19 train, days 20-31 predict per month across 3 months.\n"
                           "Random CV would be misleading: predict timestamps overlap train.",
            "make_fn": lambda: make_within_period_data(n_entities=15, n_months=3),
            "kwargs": _WP,
        },
        "iid": {
            "label": "Case B — i.i.d. Random Split",
            "description": "Train and predict from the same distribution, no time or group structure.\n"
                           "Standard k-fold is appropriate here.",
            "make_fn": make_iid_data,
            "kwargs": _IID,
        },
        "group": {
            "label": "Case C — Unseen-Group Split",
            "description": "Stores A-D in train, unseen stores E-F in predict.\n"
                           "Random CV leaks group signal; group holdout is required.",
            "make_fn": make_group_data,
            "kwargs": _GRP,
        },
        "future_time_split": {
            "label": "Case D — Future Time Split",
            "description": "Train = Jan-Jun 2023; Predict = Jul-Dec 2023.\n"
                           "Strict time holdout required; k-fold would train on future.",
            "make_fn": make_time_split_data,
            "kwargs": _TIME,
        },
        "leakage": {
            "label": "Case E+F — Deliberate Leakage + Valid Datetime",
            "description": "target_component_1 and _2 are train-only (high leakage).\n"
                           "timestamp is in both sets and should NOT be flagged as leakage.",
            "make_fn": make_leakage_data,
            "kwargs": _LEAK,
        },
    }


# ── Printing helpers ───────────────────────────────────────────────────────

def _hr(title: str = ""):
    if title:
        pad = max((66 - len(title) - 2) // 2, 2)
        print(f"{'─' * pad} {title} {'─' * (66 - pad - len(title) - 2)}")
    else:
        print(_SUBSEP)


def _wrap(text: str, width: int = 58, indent: str = "    ") -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for w in words:
        if len(current) + len(w) + 1 > width:
            lines.append(indent + current)
            current = w
        else:
            current = f"{current} {w}".strip()
    if current:
        lines.append(indent + current)
    return lines


def _print_results(case_name: str, case_meta: dict, results: dict, out_dir: str):
    print()
    print(_SEP)
    print(f"  {case_meta['label']}")
    print(_SEP)
    for line in _wrap(case_meta["description"], width=62, indent="  "):
        print(line)
    print()

    # ── 1. Detected pattern ────────────────────────────────────────────
    _hr("1. Detected Pattern")
    tpp = results["train_prediction_pattern"]
    schema = results["schema"]
    t_rows = schema.get("train", {}).get("row_count", "?")
    p_rows = schema.get("predict", {}).get("row_count", "?")
    t_cols = schema.get("train", {}).get("col_count", "?")
    p_cols = schema.get("predict", {}).get("col_count", "?")
    pattern = tpp.get("pattern", "unknown")
    print(f"  Pattern type    : {pattern}")
    print(f"  Train coverage  : {t_rows} rows × {t_cols} cols")
    print(f"  Predict coverage: {p_rows} rows × {p_cols} cols")
    print(f"  Evidence:")
    for ev in tpp.get("evidence", [])[:4]:
        for line in _wrap(ev, width=58, indent="    • "):
            print(line)
    train_only = tpp.get("schema_mismatch", {}).get("train_only_columns", [])
    if train_only:
        print(f"  Train-only cols : {', '.join(train_only)}")
    shift = tpp.get("numeric_distribution_shift", {})
    if shift.get("total_columns_tested", 0):
        n_shift = shift["shift_detected_count"]
        n_total = shift["total_columns_tested"]
        print(f"  Numeric shift   : {n_shift}/{n_total} columns (KS test)")

    # ── 2. Feature availability ────────────────────────────────────────
    _hr("2. Feature Availability at Prediction Time")
    fa = results["feature_availability"]
    fa_sum = fa.get("summary", {})
    print(f"  Available at prediction : {fa_sum.get('available_at_prediction', 0)}")
    print(f"  Train-only (exclude)    : {fa_sum.get('train_only', 0)}")
    print(f"  High leakage risk       : {fa_sum.get('high_leakage_risk', 0)}")
    to_excl = fa.get("columns_to_exclude", [])
    if to_excl:
        print(f"\n  Columns to EXCLUDE:")
        fa_cols = fa.get("columns", {})
        for col in to_excl[:8]:
            info = fa_cols.get(col, {})
            tag = info.get("leakage_type") or info.get("inferred_type", "?")
            print(f"    ✗  {col:30s}  [{tag}]")
    to_verify = fa.get("columns_to_verify", [])
    if to_verify:
        print(f"\n  Columns to VERIFY before use:")
        for col in to_verify[:5]:
            print(f"    ?  {col}")

    # ── 3. Leakage risk level ──────────────────────────────────────────
    _hr("3. Leakage Risk Level")
    leakage = results["leakage"]
    lsum = leakage.get("summary", {})
    print(f"  High   : {lsum.get('high', 0)}")
    print(f"  Medium : {lsum.get('medium', 0)}")
    print(f"  Low    : {lsum.get('low', 0)}")

    # ── 4. Validation recommendation ──────────────────────────────────
    _hr("4. Validation Recommendation")
    val = results["validation"]
    rec = val.get("recommendation", {})
    print(f"  Recommended strategy : {rec.get('strategy', '?')}")
    print(f"  Confidence           : {rec.get('confidence', '?')}")
    print()
    print(f"  Why random split is misleading:")
    for line in _wrap(val.get("why_generic_validation_is_risky", ""), width=58):
        print(line)
    print()
    print(f"  Fit rule    : {rec.get('fit_rule', '')}")
    print(f"  Validate on : {rec.get('validation_rule', '')}")
    avoid = rec.get("strategies_to_avoid", [])
    if avoid:
        print(f"  AVOID       : {', '.join(avoid)}")
    alts = val.get("alternatives", [])
    if alts:
        print(f"  Alternatives: {', '.join(a['strategy'] for a in alts)}")

    # ── 5. Generated outputs ───────────────────────────────────────────
    _hr("5. Generated Output Files")
    for sub in ("logs", "figures", "reports"):
        d = os.path.join(out_dir, sub)
        if os.path.isdir(d):
            files = sorted(os.listdir(d))
            print(f"\n  {sub}/")
            for fn in files:
                sz = os.path.getsize(os.path.join(d, fn))
                print(f"    {fn:48s}  {sz:>8,} B")

    print()
    print(f"{'─' * 66}")
    print(f"  Full report: {os.path.join(out_dir, 'reports', 'pattern_audit.md')}")
    print(f"{'─' * 66}")
    print()


# ── Entry point ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train-Test Pattern Auditor — demo runner",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--case",
        required=True,
        choices=["within_period", "iid", "group", "future_time_split", "leakage"],
        help="Demo case to run.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output directory. Defaults to outputs/demo_<case>.",
    )
    args = parser.parse_args()

    cases = _cases()
    case_meta = cases[args.case]
    out_dir = args.out or os.path.join(
        os.path.dirname(_ROOT), "outputs", f"demo_{args.case}"
    )

    print(f"\n[PatternAuditor] Generating data for case: {args.case!r}")
    train_df, predict_df = case_meta["make_fn"]()
    print(f"  Train  : {train_df.shape[0]} rows × {train_df.shape[1]} cols")
    print(f"  Predict: {predict_df.shape[0]} rows × {predict_df.shape[1]} cols")

    print(f"[PatternAuditor] Running audit → output dir: {out_dir}")
    auditor = PatternAuditor(train_df, predict_df, **case_meta["kwargs"])
    results = auditor.run(output_dir=out_dir)

    _print_results(args.case, case_meta, results, out_dir)


if __name__ == "__main__":
    main()
