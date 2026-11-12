"""Torch port of rules.stdp (F4/T4, Bi & Poo pair-based STDP). The numpy
module is the spec; STDPParams is imported from it so defaults cannot drift.
Same operation order: decay traces, potentiate on post spikes from the pre
trace, depress on pre spikes from the post trace (two-stage clip, exactly as
numpy applies it), then increment traces by +1 on this step's spikes.

The numpy rule skips each weight stage under `np.any(...)` guards; with no
spikes that stage adds exactly 0 and its clip is a no-op on an in-bounds W,
so evaluating both stages unconditionally is semantically identical and
avoids a host synchronisation per step on GPU.

Precision (F24): all arithmetic in fp32; fp16 W is read up to fp32 and cast
once on the final write (W.copy_), never accumulated in fp16.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from brainsim.engine.torch_core.propagate import TorchCSRProjection
from brainsim.rules.stdp import STDPParams


@dataclass
class TorchSTDPState:
    trace_pre: torch.Tensor  # fp32, per presynaptic neuron
    trace_post: torch.Tensor  # fp32, per postsynaptic neuron
    W: torch.Tensor  # per synapse; fp16 or fp32; may alias projection weights

    @classmethod
    def zeros(
        cls,
        n_pre: int,
        n_post: int,
        n_syn: int,
        w_init: float = 0.0,
        device: torch.device | str = "cpu",
        w_dtype: torch.dtype = torch.float32,
    ) -> "TorchSTDPState":
        device = torch.device(device)
        return cls(
            trace_pre=torch.zeros(n_pre, dtype=torch.float32, device=device),
            trace_post=torch.zeros(n_post, dtype=torch.float32, device=device),
            W=torch.full((n_syn,), float(w_init), dtype=w_dtype, device=device),
        )

    @classmethod
    def for_projection(cls, proj: TorchCSRProjection) -> "TorchSTDPState":
        """Bind to a projection: W ALIASES proj.weights (in-place updates).
        On a ShardedNetwork slice the pre trace is global-sized (driven by
        the exchanged fired ids) and the post trace local-sized; no remote
        state beyond the spike exchange is required.
        """
        return cls(
            trace_pre=torch.zeros(proj.n_pre, dtype=torch.float32, device=proj.device),
            trace_post=torch.zeros(proj.n_post, dtype=torch.float32, device=proj.device),
            W=proj.weights,
        )


def step(
    state: TorchSTDPState,
    params: STDPParams,
    pre_spikes: torch.Tensor,
    post_spikes: torch.Tensor,
    pre_idx: torch.Tensor,
    post_idx: torch.Tensor,
) -> torch.Tensor:
    """Advance traces and weights by one dt in place; returns the updated W."""
    state.trace_pre.mul_(math.exp(-params.dt / params.tau_plus))
    state.trace_post.mul_(math.exp(-params.dt / params.tau_minus))

    pre_f = pre_spikes.to(torch.float32)
    post_f = post_spikes.to(torch.float32)

    w = state.W.to(torch.float32)
    dW_pot = params.a_plus * state.trace_pre.index_select(0, pre_idx) * post_f.index_select(0, post_idx)
    w = torch.clamp(w + dW_pot, params.w_min, params.w_max)
    dW_dep = params.a_minus * state.trace_post.index_select(0, post_idx) * pre_f.index_select(0, pre_idx)
    w = torch.clamp(w - dW_dep, params.w_min, params.w_max)
    state.W.copy_(w)

    state.trace_pre.add_(pre_f)
    state.trace_post.add_(post_f)
    return state.W


__all__ = ["STDPParams", "TorchSTDPState", "step"]
