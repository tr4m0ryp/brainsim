"""Frozen H1 controls: graph surgeries at matched resources.

Arm 2  config_shuffle : degree-preserving configuration model via
       Maslov-Sneppen double-edge swaps per projection (PN->KC and KC->MBON;
       out-degree, in-degree, and the per-source weight multiset all
       preserved; 10 x E swap attempts, duplicate-creating swaps rejected).
       The KC<->APL star projections cannot be rewired by edge swaps (only 2
       APL nodes), so the configuration arm permutes their weight vectors
       across KCs: degrees preserved, weight multiset preserved, assignment
       randomized. Five independent draws per seed (contract).
Arm 3  weight_shuffle : Damicelli control -- real topology untouched, the
       weight multiset permuted across the existing edges of each projection.
Arm 4  er_control     : Erdos-Renyi at matched density -- same node sets,
       same edge count per projection, uniform random pairs (no duplicates),
       weights = permuted real multiset (total synapse count preserved).
Arm 5  lives in tf_arm.py (Tsodyks-Feigelman sparse attractor).

Every surgery consumes an explicit rng; per-seed/per-draw generators are
built by the driver so each arm is deterministic.
"""
from __future__ import annotations

import numpy as np

from brainsim.batteries.h1.mb_extract import MBGraph, Projection


def _ms_swap(proj: Projection, rng: np.random.Generator,
             swaps_per_edge: int = 10) -> tuple[Projection, float]:
    """Maslov-Sneppen double-edge swaps; returns (shuffled, accept_rate)."""
    pre = proj.pre.copy()
    post = proj.post.copy()
    w = proj.w.copy()
    e = pre.size
    if e < 2:
        return Projection(pre, post, w, proj.n_pre, proj.n_post), 0.0
    key = proj.n_post + 1
    existing = set((pre.astype(np.int64) * key + post).tolist())
    n_attempts = swaps_per_edge * e
    picks = rng.integers(0, e, size=(n_attempts, 2))
    accepted = 0
    for i, j in picks:
        if i == j:
            continue
        a, b = int(pre[i]), int(post[i])
        c, d = int(pre[j]), int(post[j])
        if a == c or b == d:
            continue
        k1, k2 = a * key + d, c * key + b
        if k1 in existing or k2 in existing:
            continue
        existing.discard(a * key + b)
        existing.discard(c * key + d)
        existing.add(k1)
        existing.add(k2)
        post[i], post[j] = d, b
        accepted += 1
    out = Projection(pre, post, w, proj.n_pre, proj.n_post).sort()
    return out, accepted / n_attempts


def _permute_weights(proj: Projection, rng: np.random.Generator) -> Projection:
    return Projection(proj.pre.copy(), proj.post.copy(),
                      rng.permutation(proj.w), proj.n_pre, proj.n_post)


def _er_projection(proj: Projection, rng: np.random.Generator) -> Projection:
    """Same edge count, uniform pairs without duplicates, permuted weights."""
    n_keys = proj.n_pre * proj.n_post
    e = proj.n_pairs
    if e > n_keys:
        raise ValueError("cannot draw more unique edges than pre x post pairs")
    keys = rng.choice(n_keys, size=e, replace=False)
    pre = (keys // proj.n_post).astype(np.int32)
    post = (keys % proj.n_post).astype(np.int32)
    return Projection(pre, post, rng.permutation(proj.w),
                      proj.n_pre, proj.n_post).sort()


def _clone(g: MBGraph, pn_kc: Projection, kc_mbon: Projection,
           kc_apl: Projection, apl_kc: Projection) -> MBGraph:
    return MBGraph(roots=g.roots, pn_kc=pn_kc, kc_mbon=kc_mbon,
                   kc_apl=kc_apl, apl_kc=apl_kc, counts=g.counts)


def _permute_apl_across_kc(kc_apl: Projection, apl_kc: Projection,
                           rng: np.random.Generator) -> tuple[Projection, Projection]:
    """Permute which KC carries which APL weight profile (degrees preserved)."""
    perm = rng.permutation(kc_apl.n_pre)  # n_pre of kc_apl == n_kc
    ka = Projection(perm[kc_apl.pre].astype(np.int32), kc_apl.post.copy(),
                    kc_apl.w.copy(), kc_apl.n_pre, kc_apl.n_post).sort()
    ak = Projection(apl_kc.pre.copy(), perm[apl_kc.post].astype(np.int32),
                    apl_kc.w.copy(), apl_kc.n_pre, apl_kc.n_post).sort()
    return ka, ak


def config_shuffle(g: MBGraph, rng: np.random.Generator) -> tuple[MBGraph, dict]:
    pk, acc_pk = _ms_swap(g.pn_kc, rng)
    km, acc_km = _ms_swap(g.kc_mbon, rng)
    ka, ak = _permute_apl_across_kc(g.kc_apl, g.apl_kc, rng)
    stats = {"accept_rate_pn_kc": round(acc_pk, 4),
             "accept_rate_kc_mbon": round(acc_km, 4)}
    return _clone(g, pk, km, ka, ak), stats


def weight_shuffle(g: MBGraph, rng: np.random.Generator) -> MBGraph:
    return _clone(g, _permute_weights(g.pn_kc, rng),
                  _permute_weights(g.kc_mbon, rng),
                  _permute_weights(g.kc_apl, rng),
                  _permute_weights(g.apl_kc, rng))


def er_control(g: MBGraph, rng: np.random.Generator) -> MBGraph:
    ka, ak = _permute_apl_across_kc(g.kc_apl, g.apl_kc, rng)
    return _clone(g, _er_projection(g.pn_kc, rng),
                  _er_projection(g.kc_mbon, rng), ka, ak)


__all__ = ["config_shuffle", "er_control", "weight_shuffle"]
