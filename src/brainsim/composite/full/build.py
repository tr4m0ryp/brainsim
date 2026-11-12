"""Chunked, rank-local assembly of the composite ShardedNetwork (C8, F17).

Each edge's connectivity is generated PROCEDURALLY per fixed global
post-chunk -- never as one global COO table (hundreds of GB at 2-5e10
synapses) -- with the same statistics circuits.builders implements: uniform
convergent pre draws, lognormal weights quantile-quantised to the 26-state
ceiling (F5, builders.quantize_to_states applied per chunk; at full scale a
chunk holds >=1e6 draws so the quantile estimate converges), self-loops
dropped on recurrent edges, and the storage multi-contact tail (F20/F28).

Layout invariance: the chunk grid is anchored to GLOBAL stage coordinates
and every chunk's rng is keyed by (seed, edge, chunk index), so any rank
regenerates identical wiring for the posts it owns regardless of
world_size. Deviations from circuits.builders, documented once:
- pre partners are drawn WITH replacement (builders draw without); the
  duplicate probability is ~k/n_pre (negligible at full scale, <=5% at
  smoke) and duplicates act as benign multi-contacts.
- the separation interneuron pool is not built (config docstring: top-f
  readout is the inhibition abstraction shared with composite.pipeline).

Memory: the projected per-rank footprint (torch_core.memory constants) is
asserted against device capacity BEFORE any allocation. Plastic parts carry
cached int64 pre+post per-synapse index vectors (+16 B/syn): every T4 rule
gathers through COO index vectors; that cost is itemized, never hidden.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np

from brainsim.circuits.builders import quantize_to_states
from brainsim.composite.full.config import CompositeConfig
from brainsim.engine.torch_core import memory as mem

INT64 = np.int64


@dataclass(frozen=True)
class EdgeSpec:
    name: str
    pre_stage: str
    post_stage: str
    k: int
    scale: float  # multiplies the quantised lognormal draw (0.0 => zero-init)
    plastic: str | None  # None | "btsp" | "mb" | "cascade"
    multi_contact: bool = False


def edge_specs(cfg: CompositeConfig) -> list[EdgeSpec]:
    """The composite loop (Architecture / T10): separation <- input,
    formation <- separation (mossy), formation recurrent (BTSP+allocation),
    output <- formation, relay <- output (MB-gated coupling), storage <-
    relay, storage recurrent (cascade+turnover+replay). Plastic recurrents
    are zero-init skeletons: wiring fixed, only weights move (Formal Model
    W_plastic)."""
    mb_scale = 0.8 * cfg.mb_w_max() / cfg.lognormal_mean
    return [
        EdgeSpec("sep_in", "input", "separation", cfg.k_sep_in,
                 cfg.static_scale(cfg.k_sep_in, "input"), None),
        EdgeSpec("form_in", "separation", "formation", cfg.k_form_in,
                 cfg.static_scale(cfg.k_form_in, "separation"), None),
        # Treves-Rolls retrieval path (EC-II -> CA3 direct projection, see
        # config.k_form_direct): sub-dominant current target at write,
        # correlation carrier at retrieval.
        EdgeSpec("form_direct", "input", "formation", cfg.k_form_direct,
                 cfg.static_scale(cfg.k_form_direct, "input",
                                  cfg.current_target_direct), None),
        EdgeSpec("form_rec", "formation", "formation", cfg.k_form_rec,
                 0.0, "btsp"),
        EdgeSpec("out_in", "formation", "output", cfg.k_out_in,
                 cfg.static_scale(cfg.k_out_in, "formation"), None),
        EdgeSpec("relay_in", "output", "relay", cfg.k_relay_in,
                 mb_scale, "mb"),
        EdgeSpec("store_in", "relay", "storage", cfg.k_store_in,
                 cfg.static_scale(cfg.k_store_in, "relay"), None),
        EdgeSpec("store_rec", "storage", "storage", cfg.k_store_rec,
                 0.0, "cascade", multi_contact=True),
    ]


def edge_synapse_count(cfg: CompositeConfig, edge: EdgeSpec) -> int:
    """Expected global synapse count (upper bound: self-loop drop not
    subtracted, ~k/n relative; multi-contact tail added at its mean)."""
    n_post = cfg.stage_sizes()[edge.post_stage]
    n = n_post * edge.k
    if edge.multi_contact:
        mean_extra = cfg.multi_contact_frac * ((2 + cfg.multi_contact_tail) / 2 - 1)
        n = int(round(n * (1.0 + mean_extra)))
    return n


def _chunk_rows(cfg: CompositeConfig, edge: EdgeSpec) -> int:
    return max(1, cfg.chunk_synapses // max(1, edge.k))


def _generate_chunk(
    cfg: CompositeConfig, edge: EdgeSpec, edge_id: int, chunk_idx: int,
    post_lo: int, post_hi: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One full grid chunk of `edge` in GLOBAL coordinates: posts
    [post_lo, post_hi), k pre draws each, weights scaled-quantised lognormal
    (zeros for plastic skeletons), plus the multi-contact tail. rng keyed by
    (seed, edge, chunk) => identical on every rank."""
    rng = np.random.default_rng([cfg.seed, 0x0C0FFEE, edge_id, chunk_idx])
    ranges = cfg.stage_ranges()
    qlo, qhi = ranges[edge.pre_stage]
    rows = post_hi - post_lo
    pre = rng.integers(qlo, qhi, size=rows * edge.k, dtype=INT64)
    post = np.repeat(np.arange(post_lo, post_hi, dtype=INT64), edge.k)
    if edge.pre_stage == edge.post_stage:
        keep = pre != post
        pre, post = pre[keep], post[keep]
    if edge.scale > 0.0:
        w = quantize_to_states(
            rng.lognormal(cfg.weight_mu, cfg.weight_sigma, size=pre.size),
            cfg.n_weight_states,
        ).astype(np.float32) * edge.scale
    else:
        w = np.zeros(pre.size, dtype=np.float32)

    if edge.multi_contact and cfg.multi_contact_frac > 0 and cfg.multi_contact_tail > 1:
        chosen = np.nonzero(rng.random(pre.size) < cfg.multi_contact_frac)[0]
        if chosen.size:
            extra = rng.integers(2, cfg.multi_contact_tail + 1, size=chosen.size) - 1
            pre = np.concatenate([pre, np.repeat(pre[chosen], extra)])
            post = np.concatenate([post, np.repeat(post[chosen], extra)])
            n_extra = int(extra.sum())
            if edge.scale > 0.0:
                w_extra = quantize_to_states(
                    rng.lognormal(cfg.weight_mu, cfg.weight_sigma, size=n_extra),
                    cfg.n_weight_states,
                ).astype(np.float32) * edge.scale
            else:
                w_extra = np.zeros(n_extra, dtype=np.float32)
            w = np.concatenate([w, w_extra])

    if edge.scale > 0.0 and cfg.row_normalize:
        # Homeostatic synaptic scaling (engineering choice, flagged): equalize
        # each post cell's TOTAL in-weight to k * scale * lognormal_mean.
        # Without it, lognormal row-mass variance makes the same "hub" cells
        # win the top-f readout under ANY input, collapsing all items onto
        # correlated codes downstream (measured at smoke: formation cross-item
        # overlap 0.69 un-normalized). Per-synapse weights keep their
        # lognormal/26-state shape WITHIN each row.
        rows = post - post_lo
        sums = np.bincount(rows, weights=w, minlength=post_hi - post_lo)
        target = edge.k * edge.scale * cfg.lognormal_mean
        nz = sums > 0
        factor = np.ones_like(sums)
        factor[nz] = target / sums[nz]
        w = (w * factor[rows]).astype(np.float32)
    return pre, post, w


def add_edge_parts(cfg: CompositeConfig, net, edge: EdgeSpec, edge_id: int,
                   fp16: bool) -> list[str]:
    """Generate the grid chunks intersecting this rank's post slice and add
    each as one projection part named '<edge>.<chunk>'. Returns part names.
    Slicing to the owned sub-range happens via a post mask, so the chunk's
    rng stream is identical on every rank (layout invariance)."""
    plo, phi = cfg.stage_ranges()[edge.post_stage]
    lo, hi = net.local_range
    olo, ohi = max(plo, lo), min(phi, hi)
    parts: list[str] = []
    if olo >= ohi:
        return parts
    rows = _chunk_rows(cfg, edge)
    first = (olo - plo) // rows
    last = (ohi - 1 - plo) // rows
    for c in range(first, last + 1):
        c_lo, c_hi = plo + c * rows, min(plo + (c + 1) * rows, phi)
        pre, post, w = _generate_chunk(cfg, edge, edge_id, c, c_lo, c_hi)
        own = (post >= olo) & (post < ohi)
        if not own.any():
            continue
        name = f"{edge.name}.{c}"
        net.add_projection(
            name, pre[own], post[own], w[own], np.ones(int(own.sum()), dtype=np.int8),
            delay_steps=cfg.delay_steps, fp16=fp16,
        )
        parts.append(name)
    return parts


# ---- memory forecast (F24; no torch needed) ---------------------------------

def forecast(cfg: CompositeConfig, world_size: int) -> dict:
    """Projected bytes per rank under post-owned sharding, itemized with the
    torch_core.memory constants. Plastic parts additionally carry the cached
    int64 per-synapse pre-id vector the rule gathers need (+8 B/syn) and
    store_rec its int8 cascade state (+1 B/syn). Transients (build chunk COO,
    per-call int64 post upcasts) are reported, not added to the resident
    total."""
    per_edge: dict[str, dict] = {}
    resident = 0
    n_parts_total = 0
    for edge in edge_specs(cfg):
        n_syn = edge_synapse_count(cfg, edge)
        local_syn = math.ceil(n_syn / world_size)
        fp16 = cfg.fp16_plastic if edge.plastic else cfg.fp16_static
        per_syn = mem.synapse_bytes(fp16=fp16, cascade=(edge.plastic == "cascade"))
        extra = 16 * local_syn if edge.plastic else 0  # cached int64 pre+post ids
        n_parts = max(1, math.ceil(local_syn / max(1, cfg.chunk_synapses)))
        n_parts_total += n_parts
        indptr = mem.INDPTR_BYTES * (cfg.n_global + 1) * n_parts
        ring = mem.RING_BYTES * cfg.delay_steps * math.ceil(cfg.n_global / world_size) * n_parts
        total = per_syn * local_syn + extra + indptr + ring
        per_edge[edge.name] = {
            "n_syn_global": n_syn, "n_syn_rank": local_syn, "parts": n_parts,
            "synapse_arrays": per_syn * local_syn, "plastic_extra": extra,
            "indptr": indptr, "ring": ring, "total": total,
        }
        resident += total
    n_local = math.ceil(cfg.n_global / world_size)
    state = mem.STATE_BYTES * n_local
    # per-neuron rule state: BTSP et_pre + MB e_kc (global-pre fp32) + is_post
    # + excitability + drive/count vectors; small next to synapse arrays.
    neuron_side = 4 * cfg.n_global * 2 + 4 * n_local * 4
    resident += state + neuron_side
    n_syn_global = sum(e["n_syn_global"] for e in per_edge.values())
    return {
        "edges": per_edge,
        "world_size": world_size,
        "n_neurons": cfg.n_global,
        "n_synapses_global": n_syn_global,
        "lif_state": state,
        "neuron_side": neuron_side,
        "per_rank_total": resident,
        "all_ranks_total": resident * world_size,
        "bytes_per_synapse": resident / max(1, math.ceil(n_syn_global / world_size)),
        "transient_chunk_bytes": cfg.chunk_synapses * 21,  # int64 pre+post, fp32 w, int8 s
    }


def device_capacity_bytes(cfg: CompositeConfig, device) -> int | None:
    if cfg.device_capacity_bytes is not None:
        return cfg.device_capacity_bytes
    try:
        import torch
        dev = torch.device(device)
        if dev.type == "cuda":
            return int(torch.cuda.get_device_properties(dev).total_memory)
    except Exception:
        pass
    try:  # host RAM for cpu ranks
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError):
        return None


def check_fit(cfg: CompositeConfig, world_size: int, device) -> dict:
    """Raise MemoryError with the full itemization if the projected per-rank
    footprint exceeds 92% of device capacity. Returns the forecast."""
    fc = forecast(cfg, world_size)
    cap = device_capacity_bytes(cfg, device)
    fc["device_capacity_bytes"] = cap
    if cap is not None and fc["per_rank_total"] > 0.92 * cap:
        lines = [f"  {k}: {v['total'] / 1e9:.2f} GB" for k, v in fc["edges"].items()]
        raise MemoryError(
            f"forecast {fc['per_rank_total'] / 1e9:.1f} GB/rank exceeds 92% of "
            f"device capacity {cap / 1e9:.1f} GB at world_size={world_size}.\n"
            + "\n".join(lines)
        )
    return fc


__all__ = ["EdgeSpec", "edge_specs", "edge_synapse_count", "add_edge_parts",
           "forecast", "device_capacity_bytes", "check_fit"]
