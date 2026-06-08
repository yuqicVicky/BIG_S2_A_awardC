"""
Run the Missingness Audit & Imputation Planner on all four toy demo cases.

Usage (from the repo root):
    python -m examples.missingness.run_demo
or:
    cd examples/missingness && python run_demo.py
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))
sys.path.insert(0, _HERE)

from make_demo_data import (
    make_mcar_dataset,
    make_mar_dataset,
    make_mnar_dataset,
    make_structural_dataset,
)
from missingness_auditor import MissingnessAuditor


CASES = [
    ("mcar",       make_mcar_dataset,       "target", "Case 1 — MCAR numeric missing"),
    ("mar",        make_mar_dataset,         "target", "Case 2 — MAR-like (education → income)"),
    ("mnar",       make_mnar_dataset,        "target", "Case 3 — MNAR / target signal (default → credit_score)"),
    ("structural", make_structural_dataset,  "target", "Case 4 — Structural (no facility → capacity = 0)"),
]


def _sep(title: str) -> None:
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def run_all(base_out: str = "outputs/missingness_demo") -> None:
    for case_id, make_fn, target_col, description in CASES:
        _sep(description)
        df = make_fn()
        out_dir = os.path.join(base_out, case_id)

        auditor = MissingnessAuditor(df, target_col=target_col)
        results = auditor.run()
        auditor.save_outputs(results, out_dir)

        profile = results["missingness_profile"]
        mechanism = results["mechanism_audit"]
        plan = results["imputation_plan"]
        leakage = results["leakage_safe_check"]

        print(f"\n  Dataset shape: {df.shape}")
        missing_cols = profile["columns_with_missing"]
        print(f"  Columns with missing: {missing_cols}")

        print("\n  Per-column summary:")
        for col in missing_cols:
            prof = profile["columns"][col]
            mech = mechanism["columns"].get(col, {})
            p_entry = plan["columns"].get(col, {})
            print(
                f"    {col:30s}  rate={prof['missing_rate']:.0%}  "
                f"severity={prof['severity']:8s}  "
                f"mechanism={mech.get('mechanism_label', '—'):30s}  "
                f"strategy={p_entry.get('strategy', '—')}"
            )

        lk_summary = leakage["summary"]
        print(
            f"\n  Leakage check: high={lk_summary['high_leakage_risk']}, "
            f"medium={lk_summary['medium_leakage_risk']}"
        )
        print(f"  Outputs written to: {out_dir}/")


if __name__ == "__main__":
    run_all()
    print("\n[Done] All demo cases complete.")
