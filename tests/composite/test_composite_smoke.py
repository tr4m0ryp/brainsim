import numpy as np

from brainsim.composite.pipeline import run_smoke_cycle


def test_end_to_end_smoke_cycle():
    report = run_smoke_cycle(seed=0)

    assert report["cycle_completed"] is True
    assert report["n_items"] == 5
    assert len(report["overlaps"]) == 5

    mean_written = float(np.mean(report["overlaps"]))
    mean_control = float(np.mean(report["control_overlaps"]))
    assert mean_written > mean_control

    assert report["protected_overlap_after"] > report["unprotected_overlap_after"]


def test_smoke_cycle_deterministic_under_seed():
    report_a = run_smoke_cycle(seed=3)
    report_b = run_smoke_cycle(seed=3)
    assert report_a["overlaps"] == report_b["overlaps"]
    assert report_a["protected_overlap_after"] == report_b["protected_overlap_after"]


def test_protection_survives_storage_noise_better_than_baseline():
    report = run_smoke_cycle(seed=1)
    protected_drop = report["protected_overlap_before"] - report["protected_overlap_after"]
    unprotected_drop = report["unprotected_overlap_before"] - report["unprotected_overlap_after"]
    assert protected_drop <= unprotected_drop
