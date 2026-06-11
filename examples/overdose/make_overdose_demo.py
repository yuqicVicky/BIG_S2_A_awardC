"""
Generate a SYNTHETIC dataset shaped like the STAI-X 2026 challenge problem:
state x week suspected nonfatal overdose ED-visit rates.

IMPORTANT — this data is entirely synthetic and illustrative. It is generated from
random numbers, contains no real surveillance data, and is NOT the official
competition dataset. Its only purpose is to exercise the missingness-auditor on the
*shape* of the challenge data (panel data: many states observed weekly, with
public-health-style missingness) without using any external or competition data.

Four missingness types are embedded, each mapping to a capability of the auditor:

  ed_visit_rate   : group-dependent by state — a few states under-report some weeks
                    -> groupwise (per-state) median imputation
  naloxone_admin  : time-series gaps (reporting outages span consecutive weeks)
                    -> forward/backward fill (temporal domain)
  subprogram_type : structural absence — a categorical that is absent (NaN) where no
                    sub-program exists, and its companion rate is then 0
                    -> structural NONE token + indicator
  high_week_flag  : MNAR-like — the highest-rate weeks are the most likely to be
                    missing (the signal hides itself)
                    -> flagged by MNAR sensitivity analysis
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

STATES = ["AL", "CA", "CO", "FL", "KY", "MA", "OH", "PA", "TX", "WA",
          "WV", "AZ", "NC", "MI", "NY"]
N_WEEKS = 52


def make_overdose_demo(seed: int = 2026) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    rows = []
    # Each state has its own baseline overdose burden and seasonal amplitude.
    base_rate = {s: rng.uniform(3, 18) for s in STATES}
    season_amp = {s: rng.uniform(0.5, 3.0) for s in STATES}
    # A sub-program (e.g. a co-located treatment service) exists only in some states.
    # Where it exists it has a type; where it does not, the type is absent (NaN) and
    # the companion rate is a structural zero.
    subprogram = {
        s: (rng.choice(["clinic", "mobile", "pharmacy"]) if rng.rand() < 0.55 else None)
        for s in STATES
    }
    # A few states are chronic under-reporters of the headline rate.
    under_reporters = set(rng.choice(STATES, size=3, replace=False))

    for s in STATES:
        for w in range(1, N_WEEKS + 1):
            seasonal = season_amp[s] * np.sin(2 * np.pi * w / 52.0)
            rate = max(0.0, base_rate[s] + seasonal + rng.normal(0, 1.0))
            naloxone = max(0.0, rate * rng.uniform(1.5, 3.0) + rng.normal(0, 2))
            has_sub = subprogram[s] is not None
            subrate = round(rate * rng.uniform(0.2, 0.5), 2) if has_sub else 0.0
            rows.append({
                "state": s,
                "week": w,
                "population_100k": round(base_rate[s] * 2 + rng.uniform(5, 50), 1),
                "ed_visit_rate": round(rate, 2),
                "naloxone_admin_rate": round(naloxone, 2),
                "subprogram_type": subprogram[s],
                "copresent_subrate": subrate,
            })

    df = pd.DataFrame(rows)
    n = len(df)

    # 1) group-dependent missingness: under-reporter states drop ~35% of weeks.
    gd_mask = df["state"].isin(under_reporters) & (rng.rand(n) < 0.35)
    df.loc[gd_mask, "ed_visit_rate"] = np.nan

    # 2) time-series gaps: each state has 1-2 multi-week reporting outages of naloxone.
    for s in STATES:
        idx = df.index[df["state"] == s].to_numpy()
        for _ in range(rng.randint(1, 3)):
            start = rng.randint(0, N_WEEKS - 6)
            length = rng.randint(2, 6)
            df.loc[idx[start:start + length], "naloxone_admin_rate"] = np.nan

    # 3) structural absence is already encoded: subprogram_type is NaN exactly where
    #    there is no sub-program, and copresent_subrate is 0 on those same rows.

    # 4) MNAR-like: the highest-rate weeks are the most likely to be missing.
    obs = df["ed_visit_rate"].notna()
    thresh = df.loc[obs, "ed_visit_rate"].quantile(0.85)
    mnar_mask = obs & (df["ed_visit_rate"] >= thresh) & (rng.rand(n) < 0.5)
    df.loc[mnar_mask, "ed_visit_rate"] = np.nan

    return df


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "overdose_demo.csv")
    df = make_overdose_demo()
    df.to_csv(out, index=False)
    miss = df.isna().mean().round(3)
    print(f"Wrote {out}  ({len(df)} rows, {df.shape[1]} cols)")
    print("Missing rates:\n", miss[miss > 0].to_string())
