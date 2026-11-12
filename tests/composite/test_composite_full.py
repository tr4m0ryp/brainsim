"""Full-scale composite runtime (composite.full): smoke cycle mechanism gate
(world 1, cpu, fp32), 2-rank gloo cycle + metrics merge, full-scale memory
forecast fit (C8 node: 8x141 GB), and config/build determinism.

Smoke values are CODE CHECKS, never science. The equivalence anchor to
composite.pipeline is the MECHANISM GATE, not the numbers (cycle.py module
docstring records why values differ): cascade deep-push on the protected
item's touched synapses -> turnover immunity -> the protected item's recall
survives the storage phase that erases the unprotected item's.
"""
import json
import os
from datetime import timedelta

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import torch.distributed as dist
import torch.multiprocessing as mp

from brainsim.composite.full.build import (
    check_fit,
    edge_specs,
    edge_synapse_count,
    forecast,
)
from brainsim.composite.full.config import STAGES, CompositeConfig
from brainsim.composite.full.cycle import run_full_smoke_cycle
from brainsim.composite.full.runtime import CompositeRuntime

WORLD = 2
# Transport-error substrings seen when a sandbox forbids gloo's local
# sockets (same guard as test_torch_shard); anything else propagates.
_SANDBOX_HINTS = ("socket", "address", "connect", "bind", "network",
                  "unreachable", "no route")


# ---- world 1: mechanism gate --------------------------------------------------

def test_smoke_cycle_mechanism_gate():
    report = run_full_smoke_cycle(seed=0)
    assert report["cycle_completed"] is True
    assert report["n_items"] == 5
    assert len(report["overlaps"]) == 5
    assert all(0.0 <= o <= 1.0 for o in report["overlaps"])

    # THE gate (pipeline.run_smoke_cycle's protection mechanism): after the
    # 60-step leak-dominated storage phase, the cascade-protected item's
    # recall beats the eroded unprotected item's, with real margin.
    assert report["protected_overlap_after"] > report["unprotected_overlap_after"]
    assert (report["protected_overlap_after"]
            - report["unprotected_overlap_after"]) >= 0.3
    # pipeline test analog: protection loses no more than non-protection.
    prot_drop = report["protected_overlap_before"] - report["protected_overlap_after"]
    unprot_drop = (report["unprotected_overlap_before"]
                   - report["unprotected_overlap_after"])
    assert prot_drop <= unprot_drop

    snap = report["snapshot"]
    assert snap["world_size"] == 1 and snap["n_items"] == 5
    assert snap["steps_total"] > 0 and snap["n_synapses_global"] > 0


def test_smoke_cycle_deterministic_under_seed():
    a = run_full_smoke_cycle(seed=3)
    b = run_full_smoke_cycle(seed=3)
    assert a["overlaps"] == b["overlaps"]
    assert a["control_overlaps"] == b["control_overlaps"]
    assert a["protected_overlap_after"] == b["protected_overlap_after"]
    assert a["snapshot"]["n_synapses_global"] == b["snapshot"]["n_synapses_global"]


# ---- runtime structure + replay transfer --------------------------------------

def _write_two_items(rt: CompositeRuntime, seed: int = 42) -> list[int]:
    rng = np.random.default_rng(seed)
    items = [
        (rng.choice([-1.0, 1.0], size=rt.config.n_input), 1.0, True),
        (rng.choice([-1.0, 1.0], size=rt.config.n_input), 1.5, False),
    ]
    return rt.write_phase(items)


def test_runtime_structure_and_recall_phase():
    cfg = CompositeConfig.smoke(seed=7)
    rt = CompositeRuntime(cfg)
    assert set(rt.edges) == {e.name for e in edge_specs(cfg)}
    ids = _write_two_items(rt)
    for i in ids:
        codes = rt.items[i].codes
        for stage in ("separation", "formation", "output", "relay", "storage"):
            assert codes[stage].size > 0, f"empty {stage} code: silent stage"

    recalls = rt.recall_phase(ids, drop_fraction=0.5)
    assert len(recalls) == 2
    assert all(0.0 <= r["store_overlap"] <= 1.0 for r in recalls)

    snap = rt.snapshot()
    analytic = sum(edge_synapse_count(cfg, e) for e in edge_specs(cfg))
    assert abs(snap["n_synapses_global"] - analytic) / analytic < 0.05
    assert snap["forecast_per_rank_bytes"] > 0


def test_replay_transfers_formation_to_storage():
    def run(budget: int) -> float:
        rt = CompositeRuntime(CompositeConfig.smoke(seed=11))
        _write_two_items(rt)
        rep = rt.storage_phase(steps=60, replay_budget=budget)
        assert rep["replay_events"] == budget
        return rt.snapshot()["edges"]["store_rec"]["w_abs_sum_global"]

    # replay re-writes sampled items' codes into the eroded store at the
    # slow lr: surviving weight mass must exceed the no-replay control.
    assert run(10) > run(0)


def test_encoding_mode_restores_weights_bitwise():
    rt = CompositeRuntime(CompositeConfig.smoke(seed=5))
    _write_two_items(rt)
    part = rt.edges["form_rec"].parts[0]
    before = rt.net.projections[part].weights.clone()
    with rt.encoding_mode(("form_rec", "store_rec")):
        assert not torch.equal(before, rt.net.projections[part].weights)
    assert torch.equal(before, rt.net.projections[part].weights)


# ---- memory forecast at full scale (C8 fit) ------------------------------------

def test_full_scale_forecast_fits_the_node():
    cfg = CompositeConfig.full_scale()
    fc = forecast(cfg, world_size=8)
    assert fc["per_rank_total"] <= 141e9, f"{fc['per_rank_total']/1e9:.1f} GB/rank"
    assert fc["all_ranks_total"] <= 1128e9
    assert fc["bytes_per_synapse"] <= 16.0  # F24 requirement
    assert 2e10 <= fc["n_synapses_global"] <= 5e10  # C8 envelope
    assert fc["n_neurons"] >= 1.5e6  # F21 census + T8 twin


def test_check_fit_raises_on_undersized_device():
    from dataclasses import replace

    cfg = replace(CompositeConfig.full_scale(), device_capacity_bytes=10**9)
    with pytest.raises(MemoryError):
        check_fit(cfg, world_size=8, device="cpu")


def test_config_layout_and_determinism():
    cfg = CompositeConfig.smoke()
    ranges = cfg.stage_ranges()
    assert list(ranges) == list(STAGES)
    lo = 0
    for name in STAGES:
        assert ranges[name][0] == lo
        lo = ranges[name][1]
    assert lo == cfg.n_global
    specs_a = edge_specs(cfg)
    specs_b = edge_specs(CompositeConfig.smoke())
    assert [(e.name, e.k, e.scale) for e in specs_a] == \
           [(e.name, e.k, e.scale) for e in specs_b]
    assert cfg.btsp_w_max() > 0 and cfg.store_w_unit() > 0 and cfg.mb_w_max() > 0


# ---- 2-rank gloo: cross-rank cycle + metrics merge -----------------------------

def _worker(rank: int, out_dir: str) -> None:
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    dist.init_process_group(
        "gloo", init_method=f"file://{out_dir}/pg", rank=rank,
        world_size=WORLD, timeout=timedelta(seconds=300),
    )
    try:
        report = run_full_smoke_cycle(seed=0, rank=rank, world_size=WORLD)
        with open(os.path.join(out_dir, f"report_{rank}.json"), "w") as fh:
            json.dump(report, fh)
    finally:
        dist.destroy_process_group()


def test_two_rank_cycle_completes_and_metrics_merge(tmp_path):
    assert not dist.is_initialized()
    try:
        mp.spawn(_worker, args=(str(tmp_path),), nprocs=WORLD, join=True)
    except Exception as exc:
        msg = str(exc).lower()
        if any(hint in msg for hint in _SANDBOX_HINTS):
            pytest.skip(f"distributed spawn unavailable in this sandbox: {exc}")
        raise

    reports = [json.loads((tmp_path / f"report_{r}.json").read_text())
               for r in range(WORLD)]
    for rep in reports:
        assert rep["cycle_completed"] is True
        assert rep["protected_overlap_after"] > rep["unprotected_overlap_after"]

    # SPMD identity: codes/overlaps and all-reduced globals agree exactly
    # across ranks (all ranks compute from the same reduced tensors).
    r0, r1 = reports
    assert r0["overlaps"] == r1["overlaps"]
    assert r0["control_overlaps"] == r1["control_overlaps"]
    s0, s1 = r0["snapshot"], r1["snapshot"]
    assert s0["n_synapses_global"] == s1["n_synapses_global"]
    assert s0["cascade_mean_depth_global"] == s1["cascade_mean_depth_global"]

    # the merge itself: per-rank local synapse counts sum to the global.
    for edge in s0["edges"]:
        local_sum = s0["edges"][edge]["n_syn_local"] + s1["edges"][edge]["n_syn_local"]
        assert local_sum == s0["edges"][edge]["n_syn_global"]
        w_local_sum = (s0["edges"][edge]["w_abs_sum_local"]
                       + s1["edges"][edge]["w_abs_sum_local"])
        assert w_local_sum == pytest.approx(
            s0["edges"][edge]["w_abs_sum_global"], rel=1e-6)

    # rank ranges partition the composite disjointly.
    assert s0["local_range"][1] == s1["local_range"][0]
    assert s0["local_range"][0] == 0 and s1["local_range"][1] == s0["n_global"]
