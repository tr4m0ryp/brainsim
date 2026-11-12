"""Torch port of rules.allocation (F6, Cai-2016 excitability allocation).
The numpy module is the spec; AllocationParams is imported from it. One
scalar excitability per neuron, exponential decay, top-k(excitability +
noise) selection with a jump for the winners, indices returned in
descending score order (torch.topk is sorted, matching numpy's
argpartition + argsort composition).

Deviation, documented: `noise` is either a torch.Generator (standard-normal
draws scaled by params.noise_std internally, the numpy rng.normal
equivalent) or a pre-drawn ADDITIVE noise tensor used as-is (already
scaled). Pre-drawn noise is what makes cross-engine equivalence exact.

Sharding note (F25): excitability shards cleanly per rank, but a GLOBAL
top-k over a ShardedNetwork needs a small cross-rank candidate merge
(all_gather of each rank's local top-k), which is beyond the spike
exchange; this module deliberately implements only the local-slice
selection.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from brainsim.rules.allocation import SECONDS_PER_HOUR, AllocationParams


@dataclass
class TorchAllocationState:
    excitability: torch.Tensor

    @classmethod
    def zeros(
        cls, n: int, device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32
    ) -> "TorchAllocationState":
        return cls(excitability=torch.zeros(n, dtype=dtype, device=torch.device(device)))


def decay(state: TorchAllocationState, params: AllocationParams, dt_seconds: float) -> None:
    """Exponentially decay excitability by `dt_seconds` of elapsed time, in place."""
    state.excitability.mul_(math.exp(-dt_seconds / params.tau))


def allocate(
    state: TorchAllocationState,
    params: AllocationParams,
    noise: torch.Tensor | torch.Generator,
    k: int | None = None,
) -> torch.Tensor:
    """Select and boost the top-k(excitability + noise) neurons; returns their
    int64 indices in descending score order. Mutates excitability in place.
    Scores are formed in fp32 regardless of storage dtype (fp16 supported as
    storage; accumulation stays fp32 per F24).
    """
    exc = state.excitability
    n = exc.shape[0]
    k = params.k if k is None else k
    k = min(k, n)
    if isinstance(noise, torch.Generator):
        noise_t = torch.randn(n, generator=noise, dtype=torch.float32, device=exc.device)
        noise_t = noise_t * params.noise_std
    else:
        noise_t = torch.as_tensor(noise, device=exc.device).to(torch.float32)
    noisy = exc.to(torch.float32) + noise_t
    idx = torch.topk(noisy, k).indices
    state.excitability.index_add_(
        0, idx, torch.full((k,), params.jump, dtype=exc.dtype, device=exc.device)
    )
    return idx


__all__ = ["AllocationParams", "TorchAllocationState", "decay", "allocate", "SECONDS_PER_HOUR"]
