"""Torch port of rules.btsp (F2, Milstein-2021 BTSP). The numpy module is
the spec: identical update semantics and operation order, and BTSPParams is
imported from it so parameter defaults cannot drift. Flat per-synapse
tensors, batched gathers, no python loop over synapses.

Precision (F24): traces and all update arithmetic are fp32 (numpy reference
integrates in fp64; equivalence holds to <1e-5 over smoke-length
histories). When W is an fp16 projection weight tensor, the update is
computed fully in fp32 and cast ONCE on the final write (W.copy_), so
within-step increments never underflow at fp16 storage resolution; fp16
stays a storage format, never an accumulation format.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from brainsim.engine.torch_core.propagate import TorchCSRProjection
from brainsim.rules.btsp import BTSPParams


def _q(x: torch.Tensor, slope: float, mid: float) -> torch.Tensor:
    """Sigmoid of the coincidence signal, shifted so q(0) == 0 exactly
    (numpy rules.btsp._q, same normalisation)."""
    raw = torch.sigmoid(slope * (x - mid))
    baseline = 1.0 / (1.0 + math.exp(slope * mid))
    return torch.clamp((raw - baseline) / (1.0 - baseline), min=0.0)


@dataclass
class TorchBTSPState:
    et_pre: torch.Tensor  # fp32, per presynaptic neuron, in [0, 1]
    is_post: torch.Tensor  # fp32, per postsynaptic neuron, in [0, 1]
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
    ) -> "TorchBTSPState":
        device = torch.device(device)
        return cls(
            et_pre=torch.zeros(n_pre, dtype=torch.float32, device=device),
            is_post=torch.zeros(n_post, dtype=torch.float32, device=device),
            W=torch.full((n_syn,), float(w_init), dtype=w_dtype, device=device),
        )

    @classmethod
    def for_projection(cls, proj: TorchCSRProjection) -> "TorchBTSPState":
        """Bind to a projection: W ALIASES proj.weights, so step() updates the
        projection in place. A ShardedNetwork slice binds identically -- its
        n_pre is global (trace fed from exchanged fired ids) and its n_post
        local, so the rule needs no remote state beyond the spike exchange.
        """
        return cls(
            et_pre=torch.zeros(proj.n_pre, dtype=torch.float32, device=proj.device),
            is_post=torch.zeros(proj.n_post, dtype=torch.float32, device=proj.device),
            W=proj.weights,
        )


def step(
    state: TorchBTSPState,
    params: BTSPParams,
    pre_spikes: torch.Tensor,
    post_plateau: torch.Tensor,
    pre_idx: torch.Tensor,
    post_idx: torch.Tensor,
) -> torch.Tensor:
    """Advance traces and weights by one dt in place; returns the updated W.
    Same operation order as the numpy rule: decay traces, reset on events,
    then read the coincidence and move W.
    """
    state.et_pre.mul_(math.exp(-params.dt / params.tau_et))
    state.is_post.mul_(math.exp(-params.dt / params.tau_is))
    state.et_pre.masked_fill_(pre_spikes.to(torch.bool), 1.0)
    state.is_post.masked_fill_(post_plateau.to(torch.bool), 1.0)

    coincidence = state.et_pre.index_select(0, pre_idx) * state.is_post.index_select(0, post_idx)
    q_plus = _q(coincidence, params.q_plus_slope, params.q_plus_mid)
    q_minus = _q(coincidence, params.q_minus_slope, params.q_minus_mid)

    w = state.W.to(torch.float32)
    dW = (params.w_max - w) * params.k_plus * q_plus - w * params.k_minus * q_minus
    state.W.copy_(torch.clamp(w + dW * (params.dt / 1000.0), 0.0, params.w_max))
    return state.W


__all__ = ["BTSPParams", "TorchBTSPState", "step"]
