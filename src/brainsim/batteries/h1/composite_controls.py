"""Frozen H1 controls realized on the sharded composite runtime.

The fly controls (batteries.h1.controls) operate on dense per-projection
COO tables; at composite scale the graph lives as post-owned CSR parts on
8 ranks, so each surgery is applied RANK-LOCALLY to the rank's parts of the
four encode/recall edges (composite_arm.ACTIVE_PATH_EDGES). The preserved
invariants are exact and global; the randomization SCOPE is the rank's
post-block (documented deviation from the fly's whole-projection mixing,
same spirit as the runtime's rank-keyed cascade/turnover draws):

Arm 2  config_shuffle : joint permutation of the POST endpoints across all
       of an edge's rank-local synapse slots. In the pre-keyed CSR the
       weight stays in its (pre, slot) cell, so weights travel with the
       presynaptic source -- the fly _ms_swap semantics (out-degrees,
       in-degree multiset, and per-source weight multisets all exactly
       preserved). Self-loops created on recurrent edges are repaired by
       pairwise swaps (the measured build drops self-loops); duplicate
       (pre, post) pairs are NOT rejected because the measured build itself
       draws with replacement (duplicates are benign multi-contacts).
Arm 3  weight_shuffle : Damicelli control -- topology untouched, the weight
       multiset permuted across the edge's rank-local slots (this also
       undoes the build's homeostatic row normalization; that destruction
       of the realized weight-to-edge assignment is the control's content).
       form_rec is zero-init plastic (permuting zeros is the identity), so
       the surgery acts on the static afferent edges.
Arm 4  erdos_renyi    : endpoints redrawn uniformly (pre over the pre
       stage, post over the rank's owned stage slice; with replacement,
       matching the measured build's convention; self-pairs rejected on
       recurrents), weights = the permuted real multiset. Edge counts per
       projection preserved EXACTLY; in-degrees binomialize.

Every surgery asserts the per-rank synapse count is unchanged (the matched
-budget mechanism, not a comment) and re-caches the plastic EdgeState
bindings it invalidates. Surgery rngs are keyed (H1SHUF, seed, draw,
edge_id, rank): deterministic per layout, identical across reruns.
"""
from __future__ import annotations

import numpy as np
import torch

from brainsim.batteries.h1.composite_arm import ACTIVE_PATH_EDGES
from brainsim.rules.torch_rules import (TorchBTSPState, synapse_post_ids,
                                        synapse_pre_ids)

H1SHUF = 0x481F5C  # surgery rng namespace
_MAX_REPAIR_ROUNDS = 64


def _generator(rt, *key: int) -> torch.Generator:
    ss = np.random.SeedSequence([H1SHUF, *map(int, key)])
    g = torch.Generator(device=rt.device)
    g.manual_seed(int(ss.generate_state(1, dtype=np.uint64)[0]) >> 1)
    return g


def _parts(rt, name: str) -> list[str]:
    return rt.edges[name].parts


def _cat_indices(rt, name: str) -> torch.Tensor:
    return torch.cat([rt.net.projections[p].indices.to(torch.int64)
                      for p in _parts(rt, name)])


def _split_back(rt, name: str, values: torch.Tensor, attr: str) -> None:
    off = 0
    for p in _parts(rt, name):
        proj = rt.net.projections[p]
        n = proj.n_synapses
        getattr(proj, attr).copy_(values[off:off + n].to(getattr(proj, attr).dtype))
        off += n


def _recache_plastic(rt, name: str) -> None:
    """Rebuild the EdgeState caches a topology surgery invalidates. Only
    fresh (pre-write) runtimes are ever operated on, so zeroed rule state
    is the correct state."""
    es = rt.edges[name]
    if not es.spec.plastic:
        return
    es.pre_idx.clear()
    es.post_idx.clear()
    es.btsp.clear()
    for p in es.parts:
        proj = rt.net.projections[p]
        es.pre_idx[p] = synapse_pre_ids(proj)
        es.post_idx[p] = synapse_post_ids(proj)
        if es.spec.plastic == "btsp":
            es.btsp[p] = TorchBTSPState.for_projection(proj)


def _assert_counts(rt, before: dict[str, int], arm: str) -> None:
    for name, n0 in before.items():
        n1 = rt.edges[name].n_synapses_local(rt.net)
        if n1 != n0:
            raise AssertionError(
                f"{arm}: edge {name} local synapse count changed {n0} -> {n1}")


def _local_counts(rt) -> dict[str, int]:
    return {n: rt.edges[n].n_synapses_local(rt.net) for n in ACTIVE_PATH_EDGES}


def _repair_self_loops(rt, name: str, idx: torch.Tensor,
                       gen: torch.Generator) -> torch.Tensor:
    """Pairwise-swap post endpoints until no slot has pre == post. Each
    round swaps DISJOINT, duplicate-free position sets, so the operation is
    a permutation and every multiset invariant survives exactly."""
    pre_local = torch.cat([synapse_pre_ids(rt.net.projections[p])
                           for p in _parts(rt, name)]) - rt.net.lo
    for _ in range(_MAX_REPAIR_ROUNDS):
        bad = torch.nonzero(pre_local == idx).flatten()
        if bad.numel() == 0:
            return idx
        j = torch.randint(idx.numel(), (bad.numel(),), generator=gen,
                          device=idx.device)
        order = torch.argsort(j)
        j_s, b_s = j[order], bad[order]
        first = torch.ones_like(j_s, dtype=torch.bool)
        first[1:] = j_s[1:] != j_s[:-1]          # j duplicates: keep first
        sel = first & ~torch.isin(j_s, bad)      # j must avoid bad positions
        b_s, j_s = b_s[sel], j_s[sel]
        tmp = idx[b_s].clone()
        idx[b_s] = idx[j_s]
        idx[j_s] = tmp
    raise AssertionError(f"{name}: self-loop repair did not converge")


# ---- surgeries ---------------------------------------------------------------

def config_shuffle(rt, seed: int, draw: int) -> None:
    before = _local_counts(rt)
    for edge_id, name in enumerate(ACTIVE_PATH_EDGES):
        if not _parts(rt, name):
            continue
        gen = _generator(rt, seed, draw, edge_id, rt.rank)
        idx = _cat_indices(rt, name)
        idx = idx[torch.randperm(idx.numel(), generator=gen, device=idx.device)]
        spec = rt.edges[name].spec
        if spec.pre_stage == spec.post_stage:
            idx = _repair_self_loops(rt, name, idx, gen)
        _split_back(rt, name, idx, "indices")
        _recache_plastic(rt, name)
    _assert_counts(rt, before, f"config_shuffle[{draw}]")


def weight_shuffle(rt, seed: int) -> None:
    before = _local_counts(rt)
    for edge_id, name in enumerate(ACTIVE_PATH_EDGES):
        if rt.edges[name].spec.plastic or not _parts(rt, name):
            continue  # zero-init plastic skeleton: permutation is identity
        gen = _generator(rt, seed, 100, edge_id, rt.rank)
        w = torch.cat([rt.net.projections[p].weights.to(torch.float32)
                       for p in _parts(rt, name)])
        w = w[torch.randperm(w.numel(), generator=gen, device=w.device)]
        _split_back(rt, name, w, "weights")
    _assert_counts(rt, before, "weight_shuffle")


def er_rewire(rt, seed: int) -> None:
    """Replace each active-path edge's rank-local parts by one ER part at
    the identical synapse count (endpoints uniform, weights permuted)."""
    cfg = rt.config
    ranges = cfg.stage_ranges()
    before = _local_counts(rt)
    for edge_id, name in enumerate(ACTIVE_PATH_EDGES):
        es = rt.edges[name]
        if not es.parts:
            continue
        gen = _generator(rt, seed, 200, edge_id, rt.rank)
        spec = es.spec
        qlo, qhi = ranges[spec.pre_stage]
        plo, phi = ranges[spec.post_stage]
        olo, ohi = max(plo, rt.net.lo), min(phi, rt.net.hi)
        m = es.n_synapses_local(rt.net)
        w = torch.cat([rt.net.projections[p].weights.to(torch.float32)
                       for p in es.parts])
        w = w[torch.randperm(m, generator=gen, device=w.device)]
        pre = torch.randint(qlo, qhi, (m,), generator=gen, device=w.device)
        post = torch.randint(olo, ohi, (m,), generator=gen, device=w.device)
        if spec.pre_stage == spec.post_stage:
            for _ in range(_MAX_REPAIR_ROUNDS):
                bad = torch.nonzero(pre == post).flatten()
                if bad.numel() == 0:
                    break
                pre[bad] = torch.randint(qlo, qhi, (bad.numel(),),
                                         generator=gen, device=w.device)
            else:
                raise AssertionError(f"{name}: ER self-pair redraw stuck")
        fp16 = cfg.fp16_plastic if spec.plastic else cfg.fp16_static
        for p in es.parts:
            del rt.net.projections[p]
        new_name = f"{name}.er"
        rt.net.add_projection(new_name, pre.cpu().numpy(), post.cpu().numpy(),
                              w.cpu().numpy(), np.ones(m, dtype=np.int8),
                              delay_steps=cfg.delay_steps, fp16=fp16)
        es.parts = [new_name]
        _recache_plastic(rt, name)
    _assert_counts(rt, before, "erdos_renyi")


def apply_control(rt, instance: str, seed: int) -> None:
    """Dispatch an arm-instance name to its surgery (no-op for measured and
    the FFI-KO, whose manipulation lives in the runtime class)."""
    if instance in ("measured", "measured_ffi_ko"):
        return
    if instance.startswith("config:"):
        config_shuffle(rt, seed, int(instance.split(":")[1]))
    elif instance == "weight_shuffle":
        weight_shuffle(rt, seed)
    elif instance == "erdos_renyi":
        er_rewire(rt, seed)
    else:
        raise ValueError(f"unknown graph-arm instance {instance!r}")


__all__ = ["H1SHUF", "apply_control", "config_shuffle", "er_rewire",
           "weight_shuffle"]
