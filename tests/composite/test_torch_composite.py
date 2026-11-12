"""Torch composite adapter smoke: the numpy run_smoke_cycle schedule on
TorchNetwork projections + torch rules, same PipelineConfig and seeds.
Code check, never science (project rule): gates are cycle completion and the
cascade-protection mechanism, not effect sizes.
"""
import pytest

torch = pytest.importorskip("torch")

from brainsim.composite.torch_adapter import run_torch_smoke_cycle


def test_torch_smoke_cycle_completes_and_protection_wins():
    report = run_torch_smoke_cycle(seed=0)
    assert report["cycle_completed"] is True
    assert report["n_items"] == 5
    assert len(report["overlaps"]) == 5
    assert report["protected_overlap_after"] > report["unprotected_overlap_after"]


def test_torch_smoke_cycle_deterministic_under_seed():
    report_a = run_torch_smoke_cycle(seed=3)
    report_b = run_torch_smoke_cycle(seed=3)
    assert report_a["overlaps"] == report_b["overlaps"]
    assert report_a["protected_overlap_after"] == report_b["protected_overlap_after"]
