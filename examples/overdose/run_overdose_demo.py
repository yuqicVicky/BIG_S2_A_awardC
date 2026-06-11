"""
End-to-end worked example on the STAI-X 2026 challenge data SHAPE.

Runs the missingness-auditor on synthetic state x week overdose-rate panel data and
shows the four capabilities that matter for this public-health forecasting problem:

  1. group-dependent missingness  -> per-STATE median imputation (ed_visit_rate)
  2. time-series reporting gaps    -> forward/backward fill (naloxone_admin_rate)
  3. structural absence            -> NONE token + indicator (subprogram_type)
  4. MNAR-like under-reporting      -> delta-adjustment sensitivity / tipping point

The data is SYNTHETIC and illustrative — it is not the official competition dataset
and uses no external data (see make_overdose_demo.py).

Run:
    python examples/overdose/run_overdose_demo.py
"""

from __future__ import annotations

import json
import os

import pandas as pd

from missingness_auditor import MissingnessAuditor
from make_overdose_demo import make_overdose_demo


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(here, "outputs")

    df = make_overdose_demo()
    # Panel data must be ordered by unit then time so forward/backward fill is valid.
    df = df.sort_values(["state", "week"]).reset_index(drop=True)

    # naloxone_admin_rate is a temporal measurement → ffill beats global median.
    domain_tags = {"naloxone_admin_rate": "time_series_metric"}

    # No single supervised target here: we are cleaning the historical panel, so the
    # auditor is free to diagnose and impute every feature (and run MNAR sensitivity
    # on the headline rate). target_col is left None.
    aud = MissingnessAuditor(df, target_col=None, domain_tags=domain_tags)
    results = aud.run()
    aud.save_outputs(results, out_dir)

    plan = results["imputation_plan"]["columns"]
    print("\n=== Imputation plan (columns with missing values) ===")
    print(f"{'column':22} {'missing':>8}  {'mechanism':32} strategy")
    for col, e in plan.items():
        if e["strategy"] == "no_imputation_needed":
            continue
        print(f"{col:22} {e['missing_rate']:>7.1%}  "
              f"{e.get('mechanism_label', '—'):32} {e['strategy']}"
              + (f"  (group={e['group_col']})" if e.get("group_col") else ""))

    # Show the MNAR tipping points for the headline rate.
    sens_path = os.path.join(out_dir, "logs", "mnar_sensitivity.json")
    if os.path.exists(sens_path):
        sens = json.load(open(sens_path))
        print("\n=== MNAR sensitivity (delta-adjustment tipping points) ===")
        for col, a in sens["columns"].items():
            tp = a["tipping_point_delta_sd"]
            tp_s = f"|δ|={abs(tp)} SD" if tp is not None else "robust"
            print(f"{col:22} {a['robustness']:18} {tp_s}")
        if sens["fragile_columns"]:
            print("Fragile (MAR-sensitive):", ", ".join(sens["fragile_columns"]))

    # Show MI pooling for one column to make the variance inflation concrete.
    mice_path = os.path.join(out_dir, "logs", "mice_pooling.json")
    if os.path.exists(mice_path):
        mice = json.load(open(mice_path))
        print("\n=== Multiple imputation (Rubin pooling) ===")
        for col, p in mice["columns"].items():
            print(f"{col:22} pooled_mean={p['pooled_estimate']:.2f}  "
                  f"naive_SE={p['naive_single_imputation_std_error']:.3f}  "
                  f"MI_SE={p['std_error']:.3f}  FMI={p['fraction_missing_information']:.2f}")

    print(f"\nAll artifacts written to: {out_dir}")
    print("  reports/missing_data_report.md / .pdf")
    print("  figures/decision_flow.png, figures/mnar_tipping_point.png")
    print("  reproduce_imputation.py (standalone, self-verifying)")


if __name__ == "__main__":
    main()
