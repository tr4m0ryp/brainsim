"""Torch port of rules.mb_rule (F1, Jiang & Litwin-Kumar dopamine-gated
eligibility, depression-dominant). The numpy module is the spec; MBRuleParams
is imported from it so defaults cannot drift. Same operation order: decay
the KC eligibility trace, reset it on KC spikes, then drain W toward
beta_floor by alpha * (e_KC * e_DAN) / tau_W * dt.

Precision (F24): arithmetic in fp32; fp16 W is read up to fp32 and cast once
on the final write (W.copy_), never accumulated in fp16.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from brainsim.engine.torch_core.propagate import TorchCSRProjection
from brainsim.rules.mb_rule import MBRuleParams


@dataclass
class TorchMBRuleState:
    e_kc: torch.Tensor  # fp32, per presynaptic (KC) neuron
    W: torch.Tensor  # per synapse; fp16 or fp32; may alias projection weights

    @classmethod
    def zeros(
        cls,
        n_pre: int,
        n_syn: int,
        w_init: float | None = None,
        device: torch.device | str = "cpu",
        w_dtype: torch.dtype = torch.float32,
    ) -> "TorchMBRuleState":
        device = torch.device(device)
        w0 = 0.05 if w_init is None else w_init  # numpy MBRuleState.zeros default
        return cls(
            e_kc=torch.zeros(n_pre, dtype=torch.float32, device=device),
            W=torch.full((n_syn,), float(w0), dtype=w_dtype, device=device),
        )

    @classmethod
    def for_projection(cls, proj: TorchCSRProjection) -> "TorchMBRuleState":
        """Bind to a projection: W ALIASES proj.weights (in-place updates).
        On a ShardedNetwork slice e_kc is global-pre-sized, fed from the
        exchanged fired ids; the per-synapse dan_signal is local by
        construction, so no remote state beyond the spike exchange is needed.
        """
        return cls(
            e_kc=torch.zeros(proj.n_pre, dtype=torch.float32, device=proj.device),
            W=proj.weights,
        )


def step(
    state: TorchMBRuleState,
    params: MBRuleParams,
    kc_spikes: torch.Tensor,
    dan_signal: torch.Tensor | float,
    pre_idx: torch.Tensor,
) -> torch.Tensor:
    """Advance the KC eligibility trace and depress W by one dt in place.

    `dan_signal` is per-synapse (or broadcastable), the dopaminergic drive
    gating each synapse's depression this step, exactly as in the numpy rule.
    """
    state.e_kc.mul_(math.exp(-params.dt / params.tau_e))
    state.e_kc.masked_fill_(kc_spikes.to(torch.bool), 1.0)

    w = state.W.to(torch.float32)
    dan = torch.as_tensor(dan_signal, dtype=torch.float32, device=w.device)
    dw = params.alpha * (state.e_kc.index_select(0, pre_idx) * dan) / params.tau_w * params.dt
    state.W.copy_(torch.clamp(w - dw, params.beta_floor, params.w_max))
    return state.W


__all__ = ["MBRuleParams", "TorchMBRuleState", "step"]
