"""Battery H1 unit tests. SYNTHETIC TINY GRAPHS ONLY: per the frozen
contract, no real-circuit capacity number may exist outside the full-scale
run, so nothing here touches data/connectomes."""
from __future__ import annotations

import numpy as np
import pytest

from brainsim.batteries.h1 import controls, tf_arm
from brainsim.batteries.h1.circuit import (DELTA_W, RULE_PARAMS,
                                           T_WRITE_STEPS, MBCircuit,
                                           closed_form_delta, topk_rows)
from brainsim.batteries.h1.fly_arm import load_battery_common
from brainsim.batteries.h1.mb_extract import build_graph
from brainsim.batteries.h1.protocol import (ItemBank, ProtocolCfg,
                                            capacity_from_curve, rng_for,
                                            run_mb_capacity)
from brainsim.batteries.h1.synthetic import make_synthetic_graph
from brainsim.rules import mb_rule

CFG = ProtocolCfg(p_cap=48, eval_sample=12, write_chunk=16)


def small_graph():
    return make_synthetic_graph(seed=3)


# ---- extraction ------------------------------------------------------------

def test_extraction_deterministic():
    a, b = make_synthetic_graph(seed=1), make_synthetic_graph(seed=1)
    for pa, pb in [(a.pn_kc, b.pn_kc), (a.kc_mbon, b.kc_mbon),
                   (a.kc_apl, b.kc_apl), (a.apl_kc, b.apl_kc)]:
        assert np.array_equal(pa.pre, pb.pre)
        assert np.array_equal(pa.post, pb.post)
        assert np.array_equal(pa.w, pb.w)
    assert a.counts == b.counts


def test_extraction_census_and_classes():
    g = small_graph()
    assert g.counts["neurons"] == {"PN": 24, "KC": 120, "MBON": 10,
                                   "DAN": 6, "APL": 2}
    census = g.counts["class_pairs"]
    assert census["PN->KC"]["pairs"] == g.pn_kc.n_pairs
    assert census["KC->MBON"]["synapses"] == g.kc_mbon.n_synapses
    assert "KC->DAN" in census          # census counts non-circuit edges too
    assert g.counts["circuit_synapses"] == (
        g.pn_kc.n_synapses + g.kc_mbon.n_synapses
        + g.kc_apl.n_synapses + g.apl_kc.n_synapses)


def test_build_graph_aggregates_are_indexed_in_class_space():
    g = small_graph()
    assert g.pn_kc.pre.max() < g.n_pn and g.pn_kc.post.max() < g.n_kc
    assert g.kc_mbon.pre.max() < g.n_kc and g.kc_mbon.post.max() < g.n_mbon


# ---- write rule: closed form === iterated mb_rule.step ---------------------

def test_closed_form_matches_iterated_mb_rule_step():
    rng = np.random.default_rng(7)
    n_pre, n_syn = 30, 90
    pre_idx = rng.integers(0, n_pre, size=n_syn)
    active = rng.random(n_pre) < 0.3
    dan = (rng.random(n_syn) < 0.5).astype(np.float64)
    state = mb_rule.MBRuleState.zeros(n_pre, n_syn)
    params = RULE_PARAMS
    for _ in range(T_WRITE_STEPS):
        mb_rule.step(state, params, active, dan, pre_idx)
    expected = np.clip(
        params.w_max - closed_form_delta() * active[pre_idx] * dan,
        params.beta_floor, params.w_max)
    assert np.allclose(state.W, expected, atol=1e-12)
    assert closed_form_delta() == pytest.approx(DELTA_W)


# ---- protocol --------------------------------------------------------------

def test_protocol_deterministic_and_seed_sensitive():
    g = small_graph()
    bank = ItemBank.generate(0, CFG, g.n_pn, g.n_mbon)
    c1 = run_mb_capacity(MBCircuit.from_graph(g, 0.05, 1.0), bank, 0, CFG)
    c2 = run_mb_capacity(MBCircuit.from_graph(g, 0.05, 1.0), bank, 0, CFG)
    assert c1["C"] == c2["C"] and c1["curve_overlap"] == c2["curve_overlap"]
    bank3 = ItemBank.generate(3, CFG, g.n_pn, g.n_mbon)
    assert not np.array_equal(bank.pn_idx, bank3.pn_idx)


def test_single_item_is_retrievable_on_synthetic_graph():
    # f_z=0.3 here: on a 10-MBON toy graph the real-circuit default (0.125)
    # collapses to a 1-column target that a degraded cue can miss entirely;
    # chance overlap is f_z, so the assertion is 2x chance.
    g = small_graph()
    cfg = ProtocolCfg(p_cap=1, eval_sample=4, write_chunk=4, f_z=0.3)
    bank = ItemBank.generate(0, cfg, g.n_pn, g.n_mbon)
    res = run_mb_capacity(MBCircuit.from_graph(g, 0.05, 1.0), bank, 0, cfg)
    assert res["curve_overlap"][0] > 0.6   # 2x chance at P=1


def test_capacity_from_curve_interpolation_and_flags():
    r = capacity_from_curve([1, 10, 20], [1.0, 0.95, 0.85])
    assert 10 < r["C"] < 20 and not r["degenerate"] and not r["right_censored"]
    assert capacity_from_curve([1, 2], [0.5, 0.4])["degenerate"]
    cens = capacity_from_curve([1, 2], [1.0, 0.95])
    assert cens["right_censored"] and cens["C"] == 2.0


def test_cue_is_partial_pattern_deletion_only():
    g = small_graph()
    bank = ItemBank.generate(0, CFG, g.n_pn, g.n_mbon)
    cues = bank.cues(0, np.arange(5))
    k = bank.pn_idx.shape[1]
    n_keep = k - round(0.5 * k)
    assert np.all(cues.sum(axis=1) == n_keep)      # half deleted, none added
    for i in range(5):
        active = np.nonzero(cues[i])[0]
        assert np.isin(active, bank.pn_idx[i]).all()  # subset of the original
    c2 = bank.cues(0, np.arange(5))
    assert np.array_equal(cues, c2)                # deterministic per item


def test_topk_deterministic_tiebreak():
    h = np.array([[1.0, 2.0, 2.0, 0.5]])
    out = topk_rows(h, 2)
    assert out.tolist() == [[False, True, True, False]]


# ---- controls: resource matching -------------------------------------------

def test_config_shuffle_preserves_degrees_and_weights():
    g = small_graph()
    gs, stats = controls.config_shuffle(g, rng_for(0, 10, 0))
    for orig, shuf in [(g.pn_kc, gs.pn_kc), (g.kc_mbon, gs.kc_mbon)]:
        assert np.array_equal(np.bincount(orig.pre, minlength=orig.n_pre),
                              np.bincount(shuf.pre, minlength=orig.n_pre))
        assert np.array_equal(np.bincount(orig.post, minlength=orig.n_post),
                              np.bincount(shuf.post, minlength=orig.n_post))
        assert np.array_equal(np.sort(orig.w), np.sort(shuf.w))
        keys = shuf.pre.astype(np.int64) * (orig.n_post + 1) + shuf.post
        assert np.unique(keys).size == keys.size   # still a simple graph
    assert stats["accept_rate_pn_kc"] > 0
    assert not np.array_equal(gs.pn_kc.post, g.pn_kc.post)


def test_config_shuffle_draws_differ_and_are_deterministic():
    g = small_graph()
    a1, _ = controls.config_shuffle(g, rng_for(0, 10, 0))
    a2, _ = controls.config_shuffle(g, rng_for(0, 10, 0))
    b, _ = controls.config_shuffle(g, rng_for(0, 10, 1))
    assert np.array_equal(a1.pn_kc.post, a2.pn_kc.post)
    assert not np.array_equal(a1.pn_kc.post, b.pn_kc.post)


def test_weight_shuffle_keeps_topology():
    g = small_graph()
    gw = controls.weight_shuffle(g, rng_for(1, 10, 0))
    assert np.array_equal(gw.pn_kc.pre, g.pn_kc.pre)
    assert np.array_equal(gw.pn_kc.post, g.pn_kc.post)
    assert np.array_equal(np.sort(gw.pn_kc.w), np.sort(g.pn_kc.w))
    assert not np.array_equal(gw.pn_kc.w, g.pn_kc.w)


def test_er_control_matches_edge_count_and_weight_multiset():
    g = small_graph()
    ge = controls.er_control(g, rng_for(2, 20))
    for orig, er in [(g.pn_kc, ge.pn_kc), (g.kc_mbon, ge.kc_mbon)]:
        assert er.n_pairs == orig.n_pairs
        assert np.array_equal(np.sort(er.w), np.sort(orig.w))
        keys = er.pre.astype(np.int64) * er.n_post + er.post
        assert np.unique(keys).size == keys.size


def test_tf_arm_matches_synapse_count_and_is_deterministic():
    g = small_graph()
    entries = g.circuit_synapses()
    net1 = tf_arm.TFAttractor(g.circuit_neurons(), entries, 0.05,
                              rng_for(0, 30))
    net2 = tf_arm.TFAttractor(g.circuit_neurons(), entries, 0.05,
                              rng_for(0, 30))
    assert net1.n_directed_entries == 2 * (entries // 2)
    assert abs(net1.n_directed_entries - entries) <= 1
    assert np.array_equal(net1.rows, net2.rows)
    assert np.array_equal(net1.cols, net2.cols)


def test_tf_streamed_stats_match_batch_definition():
    rng = rng_for(9, 30)
    net = tf_arm.TFAttractor(40, 200, 0.1, rng)
    pats = [rng_for(9, 31, m).choice(40, size=4, replace=False)
            for m in range(6)]
    for p in pats:
        net.store(p)
    j = net.assemble().toarray()
    x = np.zeros((6, 40))
    for r, p in enumerate(pats):
        x[r, p] = 1.0
    dense = ((x - 0.1).T @ (x - 0.1))
    mask = np.zeros((40, 40), dtype=bool)
    mask[net.rows, net.cols] = True
    assert np.allclose(j[mask], dense[mask], atol=1e-9)


# ---- verdict wiring (frozen battery_common only) ----------------------------

def test_verdict_wiring_uses_frozen_battery_common():
    bc = load_battery_common()
    assert bc.h1_verdict([1.2, 1.3], [1.0]) == "supported"
    assert bc.h1_verdict([1.0, 1.01], [1.0]) == "killed"
    assert bc.h1_verdict([1.07, 1.08], [1.0]) == "partial"
    lo, hi = bc.bootstrap_ci([1.0, 1.1, 1.2])
    assert lo <= 1.1 <= hi


def test_fly_only_assessment_probes_not_reimplements():
    from brainsim.batteries.h1.fly_arm import _fly_only_assessment
    bc = load_battery_common()
    assert _fly_only_assessment(bc, [1.2, 1.25])["fly_only_assessment"] == \
        "clears_claim_bar"
    assert _fly_only_assessment(bc, [1.0, 1.02])["fly_only_assessment"] == \
        "in_kill_band"
    assert _fly_only_assessment(bc, [1.07, 1.08])["fly_only_assessment"] == \
        "between_band_and_bar"
    assert _fly_only_assessment(bc, [float("inf")])["fly_only_assessment"] == \
        "degenerate"


def test_smoke_run_writes_outside_results_and_flags_smoke(tmp_path):
    from brainsim.batteries.h1.fly_arm import run_fly_arm
    out = tmp_path / "fly_arm_smoke.json"
    rep = run_fly_arm(make_synthetic_graph(), out_path=out, seeds=(0, 1),
                      cfg=CFG, smoke=True, provenance={"note": "test"})
    assert out.exists() and rep["smoke"] is True
    assert rep["n_seeds"] == 2
    with pytest.raises(ValueError):
        run_fly_arm(make_synthetic_graph(), out_path=tmp_path / "results.json",
                    seeds=(0,), cfg=CFG, smoke=True)


def test_composite_full_scale_enforces_contract_seed_floor():
    """The composite arm is implemented (test_battery_h1_composite.py); the
    driver must refuse a full-scale invocation below the frozen >= 3 seed
    floor before touching any hardware."""
    import subprocess
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    r = subprocess.run(
        [sys.executable, str(root / "scripts/batteries/battery_h1.py"),
         "--arm", "composite", "--scale", "full", "--seeds", "0"],
        capture_output=True, text=True)
    assert r.returncode == 2
    assert "3" in r.stdout and "seed" in r.stdout.lower()
