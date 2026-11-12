import subprocess
import sys
from pathlib import Path

import pytest

from brainsim.bench.__main__ import run_bench

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_run_bench_returns_positive_rates():
    report = run_bench(n=100, k=10, steps=20, seed=0)
    assert report["steps_per_sec"] > 0
    assert report["synaptic_events_per_sec"] >= 0


def test_run_bench_torch_backend_positive_rates():
    pytest.importorskip("torch")
    report = run_bench(n=100, k=10, steps=20, seed=0, backend="torch", device="cpu")
    assert report["steps_per_sec"] > 0
    assert report["backend"].startswith("torch-cpu")


def test_bench_forecast_cli_exits_zero():
    result = subprocess.run(
        [sys.executable, "-m", "brainsim.bench", "--forecast", "5e10", "--k", "10000"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "GB/rank(fp16)" in result.stdout


def test_bench_smoke_cli_exits_zero():
    result = subprocess.run(
        [sys.executable, "-m", "brainsim.bench", "--smoke"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "steps/sec" in result.stdout
