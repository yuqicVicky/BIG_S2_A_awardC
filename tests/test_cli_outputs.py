"""CLI produces all required output files for missingness_auditor."""

import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

_SRC = os.path.join(os.path.dirname(__file__), "..", "src")
_ENV = {**os.environ, "PYTHONPATH": _SRC + os.pathsep + os.environ.get("PYTHONPATH", "")}

REQUIRED_FILES = [
    "logs/missingness_profile.json",
    "logs/missingness_mechanism_audit.json",
    "logs/structural_missingness_audit.json",
    "logs/imputation_plan.json",
    "logs/leakage_safe_imputation_check.json",
    "reports/missing_data_report.md",
    "figures/missingness_bar.png",
    "figures/missingness_matrix.png",
    "figures/missingness_target_signal.png",
]


@pytest.fixture
def csv_path(tmp_path):
    rng = np.random.default_rng(0)
    n = 60
    df = pd.DataFrame({
        "num_a":  np.where(rng.random(n) < 0.20, np.nan, rng.normal(0, 1, n)),
        "cat_a":  np.where(rng.random(n) < 0.20, None, rng.choice(["x", "y"], n)),
        "target": rng.normal(0, 1, n),
    })
    p = str(tmp_path / "data.csv")
    df.to_csv(p, index=False)
    return p


def test_cli_data_mode(csv_path, tmp_path):
    out = str(tmp_path / "out") + "/"
    result = subprocess.run(
        [sys.executable, "-m", "missingness_auditor.cli",
         "--data", csv_path, "--target", "target", "--out", out],
        capture_output=True, text=True, env=_ENV,
    )
    assert result.returncode == 0, result.stderr
    for f in REQUIRED_FILES:
        assert os.path.exists(os.path.join(out, f)), f"Missing output: {f}"


def test_cli_train_predict_mode(tmp_path):
    rng = np.random.default_rng(1)
    n = 40
    df = pd.DataFrame({
        "feat":   np.where(rng.random(n) < 0.15, np.nan, rng.normal(0, 1, n)),
        "target": rng.normal(0, 1, n),
    })
    train_p = str(tmp_path / "train.csv")
    pred_p  = str(tmp_path / "pred.csv")
    df.to_csv(train_p, index=False)
    df.drop(columns=["target"]).to_csv(pred_p, index=False)

    out = str(tmp_path / "out") + "/"
    result = subprocess.run(
        [sys.executable, "-m", "missingness_auditor.cli",
         "--train", train_p, "--predict", pred_p,
         "--target", "target", "--out", out],
        capture_output=True, text=True, env=_ENV,
    )
    assert result.returncode == 0, result.stderr
    for f in REQUIRED_FILES:
        assert os.path.exists(os.path.join(out, f)), f"Missing output: {f}"


def test_cli_json_summary_valid(csv_path, tmp_path):
    out = str(tmp_path / "out") + "/"
    result = subprocess.run(
        [sys.executable, "-m", "missingness_auditor.cli",
         "--data", csv_path, "--target", "target",
         "--out", out, "--json-summary"],
        capture_output=True, text=True, env=_ENV,
    )
    assert result.returncode == 0, result.stderr
    lines = result.stdout.strip().split("\n")
    start = next((i for i, l in enumerate(lines) if l.strip().startswith("{")), None)
    assert start is not None
    parsed = json.loads("\n".join(lines[start:]))
    assert "columns" in parsed


def test_cli_no_target_runs_ok(tmp_path):
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"a": np.where(rng.random(30) < 0.2, np.nan, rng.normal(0, 1, 30))})
    p   = str(tmp_path / "d.csv")
    out = str(tmp_path / "out") + "/"
    df.to_csv(p, index=False)
    result = subprocess.run(
        [sys.executable, "-m", "missingness_auditor.cli", "--data", p, "--out", out],
        capture_output=True, text=True, env=_ENV,
    )
    assert result.returncode == 0, result.stderr


def test_imputation_plan_json_structure(csv_path, tmp_path):
    out = str(tmp_path / "out") + "/"
    subprocess.run(
        [sys.executable, "-m", "missingness_auditor.cli",
         "--data", csv_path, "--target", "target", "--out", out],
        env=_ENV, capture_output=True,
    )
    with open(os.path.join(out, "logs/imputation_plan.json")) as f:
        plan = json.load(f)
    assert "columns" in plan
    assert "summary" in plan
    for col, entry in plan["columns"].items():
        assert "strategy" in entry
        assert "add_missing_indicator" in entry
        assert "fit_on" in entry
