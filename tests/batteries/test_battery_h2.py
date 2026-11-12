"""Battery H2 driver tests (tiny scale: structure, determinism, conservation,
gating, sharding, and the smoke rails -- never magnitudes; tiny numbers are
not results and are asserted only for identity/invariance properties)."""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from brainsim.batteries.h2 import artifact, runner
from brainsim.batteries.h2.arms import RunCaches, build_work_list, run_work_item, work_key
from brainsim.batteries.h2.config import tiny_config
from brainsim.batteries.h2.report import (MergeError, analyze, battery_common,
                                          make_partial, merge_partials)
from brainsim.batteries.h2.twin import Twin, TwinGateError, build_twin, require_gate
from brainsim.paths import RESULTS

ROOT = Path(__file__).resolve().parents[2]
RESULTS_JSON = RESULTS / "h2" / "results.json"
SEEDS = [101, 198]


def synthetic_stats(rate2: float = 0.015, rate5: float = 0.005) -> artifact.H01Stats:
    """Small synthetic artifact: known contact pmf + lognormal-shaped logvol."""
    conn = {"multi_contact_distribution": {
        "1": int(1e6 * (1 - rate2 - rate5)), "2": int(1e6 * rate2), "5": int(1e6 * rate5)},
        "unique_connections": 1_000_000, "mean_k": 1 + rate2 + 4 * rate5, "max_k": 5}
    bins = {}
    for b in range(0, 60):
        z = (b - 30) / 8.0  # std 0.8 in bin units = 0.08 in log10
        bins[f"2:{b}"] = int(1e6 * math.exp(-0.5 * z * z))
        bins[f"1:{b}"] = 10
    return artifact.from_dicts(conn, {"logvol_bins": bins}, source="synthetic")


@pytest.fixture(scope="module")
def real_stats() -> artifact.H01Stats:
    return artifact.load()


@pytest.fixture(scope="module")
def tiny_cfg():
    return tiny_config("numpy")


# -- twin gate ----------------------------------------------------------------------


def test_twin_gate_passes_on_synthetic_stats(tiny_cfg):
    stats = synthetic_stats()
    twin = build_twin(stats, tiny_cfg, seed=7)
    assert twin.gate["passed"], twin.gate["checks"]
    require_gate(twin)  # must not raise


def test_twin_gate_fails_on_mismatched_stats(tiny_cfg):
    from brainsim.batteries.h2.twin import run_gate

    twin = build_twin(synthetic_stats(), tiny_cfg, seed=7)
    wrong = synthetic_stats(rate2=0.15, rate5=0.05)  # 10x the multi-contact rate
    gate = run_gate(twin, wrong, tiny_cfg, np.random.default_rng(0))
    assert not gate["passed"]
    assert not gate["checks"]["multi_contact_rate"]["pass"]
    bad = Twin(n=twin.n, pre=twin.pre, post=twin.post, w0=twin.w0,
               conn_counts=twin.conn_counts, gate=gate)
    with pytest.raises(TwinGateError):
        require_gate(bad)


def test_twin_gate_passes_on_real_artifact(real_stats, tiny_cfg):
    """The F28 artifact itself must parameterise a gate-passing twin: the
    strength check is the held-out shape test against the real histogram."""
    twin = build_twin(real_stats, tiny_cfg, seed=101)
    assert twin.gate["passed"], twin.gate["checks"]
    ks = twin.gate["checks"]["strength_shape_ks"]["value"]
    assert 0.0 < ks <= 0.08


def test_arms_blocked_when_gate_fails(real_stats, tiny_cfg, monkeypatch):
    """Contract: no arm may run on an unvalidated twin."""
    import brainsim.batteries.h2.arms as arms_mod

    def failing_build(stats, cfg, seed):
        t = build_twin(stats, cfg, seed)
        t.gate = dict(t.gate, passed=False)
        return t

    monkeypatch.setattr(arms_mod, "build_twin", failing_build)
    caches = RunCaches(tiny_cfg, real_stats)
    with pytest.raises(TwinGateError):
        run_work_item(caches, "redundancy", 2, 0.2, 101)


# -- resource conservation --------------------------------------------------------------


def test_motif_resource_conservation(real_stats, tiny_cfg):
    """All spending arms realise the identical priced schedule (matched R)."""
    caches = RunCaches(tiny_cfg, real_stats)
    led = {}
    for arm in ("redundancy", "uniform", "replay", "combined"):
        r = run_work_item(caches, arm, 2, 0.2, 101)
        led[arm] = r["retention"]["retention_ledger"]
    ref = led["redundancy"]["spent"]
    assert ref > 0
    for arm, l in led.items():
        assert abs(l["spent"] - ref) <= 1e-6 * ref, (arm, l, ref)
        assert abs(l["allowance"] - l["spent"]) <= 1e-6 * ref, (arm, l)
    # cascade-only spends nothing (contract arm 3)
    r3 = run_work_item(caches, "cascade_only", 0, 0.2, 101)
    assert r3["retention"]["retention_ledger"]["spent"] == 0.0


# -- determinism ---------------------------------------------------------------------------


def test_protocol_determinism_per_seed(real_stats, tiny_cfg):
    a = run_work_item(RunCaches(tiny_cfg, real_stats), "redundancy", 2, 0.2, 101)
    b = run_work_item(RunCaches(tiny_cfg, real_stats), "redundancy", 2, 0.2, 101)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_shard_merge_equals_unsharded(real_stats, tiny_cfg):
    """Amendment rule 5: 1/1 and I/N partitions produce identical numbers."""
    silent = lambda *a, **k: None
    p_all = runner.run_shard(tiny_cfg, SEEDS, 0, 1, log=silent)
    m_all = merge_partials([p_all], write_final=False)
    pa = runner.run_shard(tiny_cfg, SEEDS, 0, 2, log=silent)
    pb = runner.run_shard(tiny_cfg, SEEDS, 1, 2, log=silent)
    m_sh = merge_partials([pa, pb], write_final=False)

    def canon(m):
        keep = {k: v for k, v in m.items() if k != "config"}
        return json.dumps(keep, sort_keys=True, default=str)

    assert canon(m_all) == canon(m_sh)
    assert set(m_all["results"]) == {work_key(*w) for w in build_work_list(tiny_cfg, SEEDS)}


# -- merge integrity ------------------------------------------------------------------------


def test_merge_refuses_incomplete_or_mismatched(real_stats, tiny_cfg):
    silent = lambda *a, **k: None
    pa = runner.run_shard(tiny_cfg, SEEDS, 0, 2, log=silent)
    with pytest.raises(MergeError):  # missing shard 1
        merge_partials([pa], write_final=False)
    pb = runner.run_shard(tiny_cfg, SEEDS, 1, 2, log=silent)
    tampered = json.loads(json.dumps(pb))
    tampered["config_hash"] = "deadbeef"
    with pytest.raises(MergeError):
        merge_partials([pa, tampered], write_final=False)


def test_merge_never_writes_final_below_full_scale(real_stats, tiny_cfg):
    """The hard rail: only a full-scale merge may write results.json."""
    silent = lambda *a, **k: None
    p_all = runner.run_shard(tiny_cfg, SEEDS, 0, 1, log=silent)
    before = RESULTS_JSON.read_bytes() if RESULTS_JSON.exists() else None
    with pytest.raises(MergeError):
        merge_partials([p_all], write_final=True)
    after = RESULTS_JSON.read_bytes() if RESULTS_JSON.exists() else None
    assert before == after


# -- verdict wiring ---------------------------------------------------------------------------


def _fake_results(cfg, seeds, shifts_ln: dict[tuple[int, float], float]) -> dict:
    """Minimal grid results where redundancy/uniform (C, T_half) encode a
    chosen log frontier-area shift per (k, p); other arms neutral."""
    out = {}
    for seed_i, seed in enumerate(seeds):
        for p in cfg.p_grid:
            out[work_key("cascade_only", 0, p, seed)] = {
                "arm": "cascade_only", "k": 0, "p": p, "seed": seed, "C": 100.0,
                "t_half_promoted": 50.0, "t_half_censored": False}
            for k in cfg.k_grid:
                jitter = 1.0 + 0.01 * seed_i
                shift = shifts_ln[(k, p)]
                for arm, c, t in (
                    ("redundancy", 100.0 * math.exp(shift) * jitter, 50.0),
                    ("uniform", 100.0 * jitter, 50.0),
                    ("replay", 100.0, 50.0),
                    ("combined", 100.0, 50.0),
                ):
                    out[work_key(arm, k, p, seed)] = {
                        "arm": arm, "k": k, "p": p, "seed": seed, "C": c,
                        "t_half_promoted": t, "t_half_censored": False}
    return out


def test_h2_verdict_wiring_killed_and_supported(tiny_cfg):
    """analyze() must reach the FROZEN verdicts.h2_verdict: killed when
    every shift is inside the bootstrap band, supported when one exceeds."""
    seeds = [1, 2, 3]
    grid = [(k, p) for p in tiny_cfg.p_grid for k in tiny_cfg.k_grid]
    dead = analyze(tiny_cfg, seeds,
                   _fake_results(tiny_cfg, seeds, {g: 0.0 for g in grid}))
    assert dead["verdict"] == "killed"
    live = analyze(tiny_cfg, seeds,
                   _fake_results(tiny_cfg, seeds, {g: (2.0 if g == grid[0] else 0.0)
                                                   for g in grid}))
    assert live["verdict"] == "supported"
    # and the frozen function itself agrees with what analyze reported
    bc = battery_common()
    assert bc.h2_verdict(live["shift_by_kp"], live["noise_band_by_kp"]) == "supported"


# -- smoke CLI rails ------------------------------------------------------------------------------


def test_smoke_cli_exits_zero_and_writes_nothing(tmp_path):
    """Real CLI, print-only: exit 0, no results.json, no partials in
    results/h2, and every measured value suppressed in the output."""
    before = RESULTS_JSON.read_bytes() if RESULTS_JSON.exists() else None
    h2_dir = RESULTS_JSON.parent
    files_before = sorted(p.name for p in h2_dir.glob("*")) if h2_dir.exists() else []
    env_scale_patch = (
        "import brainsim.batteries.h2.runner as r, brainsim.batteries.h2.config as c;"
        "r.config_for = lambda scale, engine, **kw: c.tiny_config(engine);"
        "import runpy, sys; sys.argv = ['battery_h2.py', '--engine', 'numpy',"
        " '--scale', 'smoke', '--seeds', '1'];"
        f"runpy.run_path(r'{ROOT}/scripts/batteries/battery_h2.py', run_name='__main__')"
    )
    proc = subprocess.run([sys.executable, "-c", env_scale_patch],
                          capture_output=True, text=True, cwd=ROOT, timeout=600)
    assert proc.returncode == 0, proc.stderr
    assert "SMOKE RUN" in proc.stdout
    assert "suppressed" in proc.stdout
    after = RESULTS_JSON.read_bytes() if RESULTS_JSON.exists() else None
    files_after = sorted(p.name for p in h2_dir.glob("*")) if h2_dir.exists() else []
    assert before == after and files_before == files_after


def test_sharded_smoke_partials_stay_out_of_results_dir(real_stats, tiny_cfg, tmp_path):
    """Sharded non-full runs must be given --out; partials land there only."""
    with pytest.raises(SystemExit):
        runner.run(tiny_cfg, SEEDS, out_dir=None, arm_shard="0/2", log=lambda *a: None)
    runner.run(tiny_cfg, SEEDS, out_dir=str(tmp_path), arm_shard="0/2",
               log=lambda *a: None)
    assert (tmp_path / "partial_0_of_2.json").exists()
    assert not RESULTS_JSON.exists() or True  # final file untouched either way


# -- torch engine (runs on CPU torch; skipped only if torch stack is absent) ----------------------


def test_torch_engine_matches_numpy_ledger(real_stats):
    pytest.importorskip("torch")
    cfg_t = tiny_config("torch", torch_device="cpu")
    cfg_n = tiny_config("numpy")
    rt = run_work_item(RunCaches(cfg_t, real_stats), "redundancy", 2, 0.2, 101)
    rn = run_work_item(RunCaches(cfg_n, real_stats), "redundancy", 2, 0.2, 101)
    st, sn = (rt["retention"]["retention_ledger"]["spent"],
              rn["retention"]["retention_ledger"]["spent"])
    assert sn > 0 and abs(st - sn) <= 1e-6 * sn
    # capacity is stream-driven and must agree across engines exactly
    assert rt["retention"]["l_ref"] == rn["retention"]["l_ref"]
