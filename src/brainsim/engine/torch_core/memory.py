"""Bytes-per-synapse accounting and per-rank memory forecast (F24).

Static budget per synapse: fp16 weight (2 B) + int32 post index (4 B) = 6 B;
fp32 weights raise it to 8 B. Optional plasticity add-ons from the F24
budget: uint8 cascade state (+1 B, rules.cascade) and fp16 eligibility trace
(+2 B, plastic subsets only) -- the full plastic fp16 layout is 9 B/synapse,
inside the <=16 B/synapse requirement. Per-neuron overheads -- the int64
indptr (kept over the FULL global pre space on every rank under post-owned
sharding), the fp32 delivery ring, and the fp32/int32 LIF state -- amortize
to noise at composite in-degrees (k >= 100) and are itemized here anyway.

No torch import in this module: the bench --forecast mode must run (and
allocate nothing) on boxes without torch.
"""
from __future__ import annotations

import math

WEIGHT_BYTES_FP16 = 2
WEIGHT_BYTES_FP32 = 4
INDEX_BYTES = 4  # int32 column index (uint32 semantics)
INDPTR_BYTES = 8  # int64 row pointer, one per pre neuron (+1)
RING_BYTES = 4  # fp32 delivery ring cell, per post neuron per delay slot
STATE_BYTES = 12  # fp32 v + fp32 I + int32 refractory, per neuron
CASCADE_BYTES = 1  # uint8 Fusi-Abbott cascade state
TRACE_BYTES = 2  # fp16 eligibility trace


def synapse_bytes(fp16: bool = True, cascade: bool = False, trace: bool = False) -> int:
    """Per-synapse bytes of the flat synapse arrays (weight + index + add-ons)."""
    total = (WEIGHT_BYTES_FP16 if fp16 else WEIGHT_BYTES_FP32) + INDEX_BYTES
    if cascade:
        total += CASCADE_BYTES
    if trace:
        total += TRACE_BYTES
    return total


def projection_bytes(
    n_pre: int,
    n_post: int,
    n_syn: int,
    fp16: bool = True,
    delay_steps: int = 1,
    cascade: bool = False,
    trace: bool = False,
) -> dict[str, int]:
    """Itemized bytes of one CSR projection with the torch_core layout."""
    syn = synapse_bytes(fp16=fp16, cascade=cascade, trace=trace) * n_syn
    indptr = INDPTR_BYTES * (n_pre + 1)
    ring = RING_BYTES * delay_steps * n_post
    return {
        "synapse_arrays": syn,
        "indptr": indptr,
        "ring": ring,
        "total": syn + indptr + ring,
    }


def measured_projection_bytes(proj) -> dict[str, int]:
    """Same itemization read off a live TorchCSRProjection's tensors
    (duck-typed on element_size/nelement; no torch import needed).
    """

    def nbytes(t) -> int:
        return t.element_size() * t.nelement()

    syn = nbytes(proj.weights) + nbytes(proj.indices)
    indptr = nbytes(proj.indptr)
    ring = nbytes(proj._ring)
    return {
        "synapse_arrays": syn,
        "indptr": indptr,
        "ring": ring,
        "total": syn + indptr + ring,
    }


def forecast_rank_bytes(
    n_syn: int,
    n_neurons: int,
    world_size: int,
    fp16: bool = True,
    delay_steps: int = 1,
    cascade: bool = False,
    trace: bool = False,
) -> dict[str, float]:
    """Projected bytes on ONE rank under post-owned neuron-range sharding:
    ~n_syn/world_size synapses and n_neurons/world_size neurons of state and
    ring locally, but the FULL n_pre indptr (rows are global pre ids).
    Models the network as one recurrent projection (the composite backbone).
    """
    local_syn = math.ceil(n_syn / world_size)
    local_neurons = math.ceil(n_neurons / world_size)
    per_syn = synapse_bytes(fp16=fp16, cascade=cascade, trace=trace)
    syn = per_syn * local_syn
    indptr = INDPTR_BYTES * (n_neurons + 1)
    ring = RING_BYTES * delay_steps * local_neurons
    state = STATE_BYTES * local_neurons
    total = syn + indptr + ring + state
    return {
        "synapse_arrays": syn,
        "indptr": indptr,
        "ring": ring,
        "state": state,
        "total": total,
        "bytes_per_synapse": total / local_syn,
    }


def forecast_table(
    n_syn: int, n_neurons: int, max_ranks: int = 8, delay_steps: int = 1
) -> str:
    """Human-readable GB/rank projection for 1..max_ranks ranks, fp16 and
    fp32 weight layouts side by side. GB is decimal (1e9), matching F24.
    """
    lines = [
        f"forecast: n_syn={n_syn:.3e} n_neurons={n_neurons:.3e} delay_steps={delay_steps}",
        "ranks  GB/rank(fp16)  B/syn(fp16)  GB/rank(fp32)  B/syn(fp32)",
    ]
    for ranks in range(1, max_ranks + 1):
        f16 = forecast_rank_bytes(n_syn, n_neurons, ranks, fp16=True, delay_steps=delay_steps)
        f32 = forecast_rank_bytes(n_syn, n_neurons, ranks, fp16=False, delay_steps=delay_steps)
        lines.append(
            f"{ranks:>5}  {f16['total'] / 1e9:>13.2f}  {f16['bytes_per_synapse']:>11.2f}"
            f"  {f32['total'] / 1e9:>13.2f}  {f32['bytes_per_synapse']:>11.2f}"
        )
    return "\n".join(lines)


__all__ = [
    "synapse_bytes",
    "projection_bytes",
    "measured_projection_bytes",
    "forecast_rank_bytes",
    "forecast_table",
]
