"""H1 composite-arm tests, part 1: budget invariants, control surgeries,
protocol determinism (tiny scale, never magnitudes)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from brainsim.batteries.h1 import composite_controls as controls
from brainsim.batteries.h1 import composite_report as report
from brainsim.batteries.h1 import composite_runner as runner
from brainsim.batteries.h1.composite_arm import (ACTIVE_PATH_EDGES,
                                                 base_cc, budget_rows,
                                                 forecast_table,
                                                 full_comp_config,
                                                 item_pattern,
                                                 smoke_comp_config,
                                                 tiny_comp_config)
from brainsim.batteries.h1.composite_tf import TorchTF
from brainsim.batteries.h3.budget import TOL, total_synapses
from brainsim.composite.full.runtime import CompositeRuntime
from brainsim.paths import RESULTS

ROOT = Path(__file__).resolve().parents[3]
H1_DIR = RESULTS / "h1"
SEED, SILENT = 0, (lambda *a, **k: None)

# the fly arm's real per-seed ratios (results/h1/fly_arm.json): far
# below the kill band -- the anti-premium case the tests pin
FLY_RATIOS_REAL = [0.4689537566456013, 0.0, 0.0, 0.0, 0.0]


# ---- budget invariants (matched resources: assertions, not comments) ---------

def test_full_scale_budget_rows_and_frozen_floor():
    cfg = full_comp_config()
    rows = budget_rows(cfg, SEED)
    assert rows["measured"]["target"] >= 1e10  # frozen validate.py h1 floor
    for name, row in rows.items():
        assert abs(row["ratio"] - 1.0) <= TOL, (name, row)
    assert len([n for n in rows if n.startswith("config:")]) == 5  # contract
    fc = forecast_table(cfg, SEED)
    assert fc["graph_arm_gb_per_rank_ws8"] <= 141  # fits one H200 per rank
    assert fc["tf_gb_single_device"] <= 141        # packed TF fits one H200
    assert fc["n_synapses_total"] == rows["measured"]["target"]
    assert fc["n_synapses_active_path"] < fc["n_synapses_total"]


def test_tiny_budget_rows_within_tol(tiny_cfg):
    for name, row in budget_rows(tiny_cfg, SEED).items():
        assert abs(row["ratio"] - 1.0) <= TOL, (name, row)


def test_formation_stage_is_as_configured_in_the_composite():
    """The thinning touches ONLY relay/storage: the formation loop keeps the
    composite's full-scale census and in-degrees field for field."""
    from brainsim.composite.full.config import CompositeConfig
    full = CompositeConfig.full_scale()
    cc = base_cc(full_comp_config(), SEED)
    for f in ("n_input", "n_separation", "n_formation", "n_output",
              "k_sep_in", "k_form_in", "k_form_direct", "k_form_rec",
              "k_out_in", "f_formation", "f_separation"):
        assert getattr(cc, f) == getattr(full, f), f
    assert cc.n_relay < full.n_relay and cc.n_storage < full.n_storage


# ---- control surgeries: exact invariants on the built CSR --------------------

def test_config_shuffle_preserves_degrees_and_per_source_weights(tiny_cfg, edge_state):
    rt = CompositeRuntime(base_cc(tiny_cfg, SEED), device="cpu")
    before = {n: edge_state(rt, n) for n in ACTIVE_PATH_EDGES}
    controls.config_shuffle(rt, SEED, draw=0)
    changed = 0
    for name in ACTIVE_PATH_EDGES:
        b, a = before[name], edge_state(rt, name)
        assert a["count"] == b["count"]
        for ib, ia in zip(b["indptr"], a["indptr"]):
            assert torch.equal(ib, ia)          # out-degrees preserved
        assert torch.equal(torch.sort(a["indices"]).values,
                           torch.sort(b["indices"]).values)  # in-degree multiset
        assert torch.equal(a["weights"], b["weights"])  # weights travel with pre
        changed += int(not torch.equal(a["indices"], b["indices"]))
    assert changed >= 3  # the shuffle really rewires


def test_config_shuffle_leaves_no_self_loops_on_form_rec(tiny_cfg):
    from brainsim.rules.torch_rules import synapse_pre_ids
    rt = CompositeRuntime(base_cc(tiny_cfg, SEED), device="cpu")
    controls.config_shuffle(rt, SEED, draw=1)
    for p in rt.edges["form_rec"].parts:
        proj = rt.net.projections[p]
        pre_local = synapse_pre_ids(proj) - rt.net.lo
        assert not torch.any(pre_local == proj.indices.to(torch.int64))
    # plastic caches were re-bound to the new topology
    es = rt.edges["form_rec"]
    for p in es.parts:
        assert torch.equal(es.post_idx[p],
                           rt.net.projections[p].indices.to(torch.int64))
        assert es.btsp[p].W is rt.net.projections[p].weights


def test_weight_shuffle_keeps_topology_and_multiset(tiny_cfg, edge_state):
    rt = CompositeRuntime(base_cc(tiny_cfg, SEED), device="cpu")
    before = {n: edge_state(rt, n) for n in ACTIVE_PATH_EDGES}
    controls.weight_shuffle(rt, SEED)
    for name in ACTIVE_PATH_EDGES:
        b, a = before[name], edge_state(rt, name)
        assert torch.equal(a["indices"], b["indices"])  # topology untouched
        assert torch.allclose(torch.sort(a["weights"]).values,
                              torch.sort(b["weights"]).values)
        if name != "form_rec":  # zero-init plastic: identity by definition
            assert not torch.equal(a["weights"], b["weights"])


def test_er_rewire_preserves_counts_and_weight_multiset(tiny_cfg, edge_state):
    from brainsim.rules.torch_rules import synapse_pre_ids
    cc = base_cc(tiny_cfg, SEED)
    rt = CompositeRuntime(cc, device="cpu")
    before = {n: edge_state(rt, n) for n in ACTIVE_PATH_EDGES}
    controls.er_rewire(rt, SEED)
    ranges = cc.stage_ranges()
    for name in ACTIVE_PATH_EDGES:
        b, a = before[name], edge_state(rt, name)
        assert a["count"] == b["count"]                 # matched exactly
        assert torch.allclose(torch.sort(a["weights"]).values,
                              torch.sort(b["weights"]).values)
        spec = rt.edges[name].spec
        for p in rt.edges[name].parts:
            proj = rt.net.projections[p]
            pre = synapse_pre_ids(proj)
            qlo, qhi = ranges[spec.pre_stage]
            assert int(pre.min()) >= qlo and int(pre.max()) < qhi
            post_g = proj.indices.to(torch.int64) + rt.net.lo
            plo, phi = ranges[spec.post_stage]
            assert int(post_g.min()) >= plo and int(post_g.max()) < phi
            if spec.pre_stage == spec.post_stage:
                assert not torch.any(pre == post_g)
        assert not torch.equal(a["indices"], b["indices"])


def test_surgeries_are_deterministic(tiny_cfg, edge_state):
    outs = []
    for _ in range(2):
        rt = CompositeRuntime(base_cc(tiny_cfg, SEED), device="cpu")
        controls.config_shuffle(rt, SEED, draw=0)
        outs.append(edge_state(rt, "sep_in")["indices"])
    assert torch.equal(outs[0], outs[1])
    rt = CompositeRuntime(base_cc(tiny_cfg, SEED), device="cpu")
    controls.config_shuffle(rt, SEED, draw=1)  # different draw differs
    assert not torch.equal(edge_state(rt, "sep_in")["indices"], outs[0])


# ---- protocol determinism and seed sensitivity --------------------------------

def test_measured_item_is_deterministic_and_seed_sensitive(tiny_cfg, canon):
    it = {"name": "measured", "seed": SEED, "exec_class": "sharded"}
    a = runner.run_work_item(tiny_cfg, it, log=SILENT)
    b = runner.run_work_item(tiny_cfg, it, log=SILENT)
    assert canon(a) == canon(b)
    n_in = base_cc(tiny_cfg, SEED).n_input
    assert not np.array_equal(item_pattern(0, 0, n_in), item_pattern(3, 0, n_in))
    assert np.array_equal(item_pattern(0, 0, n_in), item_pattern(0, 0, n_in))


def test_ffi_ko_uses_h3_topf_off_and_differs_from_measured(tiny_cfg, partial_all, canon):
    from brainsim.batteries.h1.composite_arm import FFI_KO_STAGES
    from brainsim.batteries.h3.systems import H3Runtime
    rt = H3Runtime(base_cc(tiny_cfg, SEED), device="cpu",
                   topf_off=FFI_KO_STAGES)
    drive = rt.drive_for(np.arange(10), 4.0)
    w = rt.run_window(drive, 10, ("separation",))
    code = rt.code_from_window("separation", w["separation"])
    fired = torch.nonzero(w["separation"][0] > 0).flatten().numpy()
    assert np.array_equal(code, np.sort(fired) + rt.ranges["separation"][0])
    res = partial_all["results"]
    meas = res[report.work_key({"name": "measured", "seed": SEED})]
    ko = res[report.work_key({"name": "measured_ffi_ko", "seed": SEED})]
    assert ko["meta"]["ffi_ko_stages"] == list(FFI_KO_STAGES)
    assert canon(meas["capacity"]) != canon(ko["capacity"])  # a real KO
