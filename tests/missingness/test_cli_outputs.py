"""CLI produces all required output files."""

import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

# The subprocess needs to find missingness_auditor in src/
_SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "src")
_CLI_ENV = {**os.environ, "PYTHONPATH": _SRC_DIR + os.pathsep + os.environ.get("PYTHONPATH", "")}


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
def sample_csv(tmp_path):
    rng = np.random.default_rng(0)
    n = 50
    df = pd.DataFrame({
        "num_a": np.where(rng.random(n) < 0.2, np.nan, rng.normal(0, 1, n)),
        "cat_a": np.where(rng.random(n) < 0.2, None, rng.choice(["x", "y"], n)),
        "complete": rng.normal(5, 1, n),
        "target": rng.normal(0, 1, n),
    })
    path = str(tmp_path / "data.csv")
    df.to_csv(path, index=False)
    return path


def test_cli_single_data_mode(sample_csv, tmp_path):
    out = str(tmp_path / "outputs") + "/"
    result = subprocess.run(
        [sys.executable, "-m", "missingness_auditor.cli",
         "--data", sample_csv, "--target", "target", "--out", out],
        capture_output=True, text=True, env=_CLI_ENV,
    )
    assert result.returncode == 0, result.stderr

    for fname in REQUIRED_FILES:
        full = os.path.join(out, fname)
        assert os.path.exists(full), f"Missing output file: {fname}"


def test_cli_train_predict_mode(tmp_path):
    rng = np.random.default_rng(1)
    n = 40
    df = pd.DataFrame({
        "feat": np.where(rng.random(n) < 0.15, np.nan, rng.normal(0, 1, n)),
        "target": rng.normal(0, 1, n),
    })
    train_path = str(tmp_path / "train.csv")
    predict_path = str(tmp_path / "predict.csv")
    df.to_csv(train_path, index=False)
    df.drop(columns=["target"]).to_csv(predict_path, index=False)

    out = str(tmp_path / "outputs") + "/"
    result = subprocess.run(
        [sys.executable, "-m", "missingness_auditor.cli",
         "--train", train_path, "--predict", predict_path,
         "--target", "target", "--out", out],
        capture_output=True, text=True, env=_CLI_ENV,
    )
    assert result.returncode == 0, result.stderr

    for fname in REQUIRED_FILES:
        assert os.path.exists(os.path.join(out, fname)), f"Missing: {fname}"


def test_cli_json_summary_flag(sample_csv, tmp_path):
    out = str(tmp_path / "outputs") + "/"
    result = subprocess.run(
        [sys.executable, "-m", "missingness_auditor.cli",
         "--data", sample_csv, "--target", "target",
         "--out", out, "--json-summary"],
        capture_output=True, text=True, env=_CLI_ENV,
    )
    assert result.returncode == 0, result.stderr
    import json
    # stdout should contain valid JSON somewhere
    lines = result.stdout.strip().split("\n")
    json_start = next((i for i, l in enumerate(lines) if l.strip().startswith("{")), None)
    assert json_start is not None, "No JSON output found in stdout"
    parsed = json.loads("\n".join(lines[json_start:]))
    assert "columns" in parsed


def test_cli_no_target(tmp_path):
    rng = np.random.default_rng(2)
    df = pd.DataFrame({
        "a": np.where(rng.random(30) < 0.2, np.nan, rng.normal(0, 1, 30)),
        "b": rng.normal(5, 1, 30),
    })
    path = str(tmp_path / "data.csv")
    out = str(tmp_path / "outputs") + "/"
    df.to_csv(path, index=False)
    result = subprocess.run(
        [sys.executable, "-m", "missingness_auditor.cli", "--data", path, "--out", out],
        capture_output=True, text=True, env=_CLI_ENV,
    )
    assert result.returncode == 0, result.stderr
