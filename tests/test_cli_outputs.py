"""CLI integration tests — run each demo case through the CLI and verify all 9 outputs."""

import sys, os, tempfile, subprocess, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_PYTHON = sys.executable

REQUIRED_OUTPUTS = [
    "logs/schema_audit.json",
    "logs/feature_availability_audit.json",
    "logs/train_prediction_pattern.json",
    "logs/leakage_audit.json",
    "logs/validation_recommendation.json",
    "reports/pattern_audit.md",
    "figures/train_prediction_coverage.png",
    "figures/feature_availability.png",
    "figures/distribution_shift_summary.png",
]


def _run_cli(data_dir: str, out_dir: str, extra_args: list[str]) -> subprocess.CompletedProcess:
    cmd = [
        _PYTHON, "-m", "pattern_auditor.cli",
        "--train", os.path.join(data_dir, "train.csv"),
        "--predict", os.path.join(data_dir, "predict.csv"),
        "--out", out_dir + "/",
    ] + extra_args
    env = os.environ.copy()
    src_path = os.path.join(_REPO_ROOT, "src")
    env["PYTHONPATH"] = src_path + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
        env=env,
    )


def _assert_outputs(out_dir: str, case_name: str):
    for rel in REQUIRED_OUTPUTS:
        path = os.path.join(out_dir, rel)
        assert os.path.exists(path), f"[{case_name}] Missing output: {rel}"
        assert os.path.getsize(path) > 20, f"[{case_name}] Empty output: {rel}"


# ── Case A: within_period ──────────────────────────────────────────────────

def test_cli_within_period():
    from examples.make_within_period_demo import write_data
    with tempfile.TemporaryDirectory() as data_dir, tempfile.TemporaryDirectory() as out_dir:
        write_data(data_dir)
        result = _run_cli(data_dir, out_dir,
                          ["--target", "target", "--datetime", "timestamp", "--group", "entity_id"])
        assert result.returncode == 0, f"CLI failed:\n{result.stderr}"
        _assert_outputs(out_dir, "within_period")

        with open(os.path.join(out_dir, "logs", "validation_recommendation.json")) as f:
            val = json.load(f)
        assert val["recommendation"]["strategy"] == "within_period_latest_available_holdout"


# ── Case B: iid ────────────────────────────────────────────────────────────

def test_cli_iid():
    from examples.make_iid_demo import write_data
    with tempfile.TemporaryDirectory() as data_dir, tempfile.TemporaryDirectory() as out_dir:
        write_data(data_dir)
        result = _run_cli(data_dir, out_dir, ["--target", "target"])
        assert result.returncode == 0, f"CLI failed:\n{result.stderr}"
        _assert_outputs(out_dir, "iid")

        with open(os.path.join(out_dir, "logs", "validation_recommendation.json")) as f:
            val = json.load(f)
        assert val["recommendation"]["strategy"] in ("kfold", "stratified_kfold")


# ── Case C: group ──────────────────────────────────────────────────────────

def test_cli_group():
    from examples.make_group_demo import write_data
    with tempfile.TemporaryDirectory() as data_dir, tempfile.TemporaryDirectory() as out_dir:
        write_data(data_dir)
        result = _run_cli(data_dir, out_dir,
                          ["--target", "target", "--group", "store_id"])
        assert result.returncode == 0, f"CLI failed:\n{result.stderr}"
        _assert_outputs(out_dir, "group")

        with open(os.path.join(out_dir, "logs", "validation_recommendation.json")) as f:
            val = json.load(f)
        assert val["recommendation"]["strategy"] == "group_split"


# ── Case D: future_time_split ──────────────────────────────────────────────

def test_cli_future_time_split():
    from examples.make_time_split_demo import write_data
    with tempfile.TemporaryDirectory() as data_dir, tempfile.TemporaryDirectory() as out_dir:
        write_data(data_dir)
        result = _run_cli(data_dir, out_dir,
                          ["--target", "target", "--datetime", "date"])
        assert result.returncode == 0, f"CLI failed:\n{result.stderr}"
        _assert_outputs(out_dir, "future_time_split")

        with open(os.path.join(out_dir, "logs", "validation_recommendation.json")) as f:
            val = json.load(f)
        assert val["recommendation"]["strategy"] == "time_holdout"


# ── Case E: leakage ────────────────────────────────────────────────────────

def test_cli_leakage():
    from examples.make_leakage_demo import write_data
    with tempfile.TemporaryDirectory() as data_dir, tempfile.TemporaryDirectory() as out_dir:
        write_data(data_dir)
        result = _run_cli(data_dir, out_dir,
                          ["--target", "target", "--datetime", "timestamp"])
        assert result.returncode == 0, f"CLI failed:\n{result.stderr}"
        _assert_outputs(out_dir, "leakage")

        with open(os.path.join(out_dir, "logs", "feature_availability_audit.json")) as f:
            fa = json.load(f)
        excluded = fa["columns_to_exclude"]
        assert "target_component_1" in excluded
        assert "target_component_2" in excluded


# ── JSON outputs are valid JSON ────────────────────────────────────────────

def test_all_json_outputs_valid():
    from examples.make_iid_demo import write_data
    with tempfile.TemporaryDirectory() as data_dir, tempfile.TemporaryDirectory() as out_dir:
        write_data(data_dir)
        _run_cli(data_dir, out_dir, ["--target", "target"])
        for log_name in [
            "schema_audit.json",
            "feature_availability_audit.json",
            "train_prediction_pattern.json",
            "leakage_audit.json",
            "validation_recommendation.json",
        ]:
            path = os.path.join(out_dir, "logs", log_name)
            with open(path) as f:
                data = json.load(f)
            assert isinstance(data, dict), f"{log_name} should be a JSON object"
