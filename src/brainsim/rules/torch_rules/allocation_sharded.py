"""Cross-rank global top-k allocation for ShardedNetwork (closes the F25
sharding gap flagged by the torch_rules port: excitability shards per rank,
but engram allocation selects the GLOBAL top-k).

Protocol (exact, not approximate): each rank scores its local slice
(excitability + noise, fp32), takes its local top-min(k, n_local)
candidates, and all-gathers (score, global_index) pairs padded to k. The
merged 2k*world candidates provably contain the global top-k (each rank
contributed its k best, and any global winner is one of its rank's k best).
Every rank then selects the same global winners and boosts only the ones it
owns. Determinism: noise must be pre-drawn per-rank from a layout-
independent seed mapping (same rule as the batteries' (arm, seed) mapping)
so 1-rank and N-rank runs select identical winners.
"""
from __future__ import annotations

import torch
import torch.distributed as dist

from brainsim.rules.allocation import AllocationParams
from brainsim.rules.torch_rules.allocation import TorchAllocationState


def allocate_global(
    state: TorchAllocationState,
    params: AllocationParams,
    noise_local: torch.Tensor,
    local_range: tuple[int, int],
    n_global: int,
    k: int | None = None,
) -> torch.Tensor:
    """Global top-k over a sharded excitability vector. `state` holds ONLY the
    local slice [lo, hi); `noise_local` is the pre-drawn additive noise for
    that slice. Returns the GLOBAL winner indices (int64, descending score),
    identical on every rank; boosts the locally-owned winners in place.
    world_size=1 (no dist init) degrades to plain local top-k.
    """
    lo, hi = local_range
    exc = state.excitability
    if exc.shape[0] != hi - lo:
        raise ValueError(f"excitability len {exc.shape[0]} != local range {hi - lo}")
    k = params.k if k is None else k
    k = min(k, n_global)

    noisy = exc.to(torch.float32) + torch.as_tensor(
        noise_local, device=exc.device
    ).to(torch.float32)

    world = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    if world == 1:
        idx = torch.topk(noisy, min(k, noisy.shape[0])).indices
        state.excitability.index_add_(
            0, idx, torch.full((idx.shape[0],), params.jump,
                               dtype=exc.dtype, device=exc.device)
        )
        return idx + lo

    # local candidates, padded to k with -inf scores / -1 indices
    k_local = min(k, noisy.shape[0])
    top = torch.topk(noisy, k_local)
    scores = torch.full((k,), float("-inf"), dtype=torch.float32, device=exc.device)
    gidx = torch.full((k,), -1, dtype=torch.int64, device=exc.device)
    scores[:k_local] = top.values
    gidx[:k_local] = top.indices + lo

    all_scores = [torch.empty_like(scores) for _ in range(world)]
    all_gidx = [torch.empty_like(gidx) for _ in range(world)]
    dist.all_gather(all_scores, scores)
    dist.all_gather(all_gidx, gidx)
    cand_scores = torch.cat(all_scores)
    cand_gidx = torch.cat(all_gidx)

    # deterministic tie-break, exact: sort by index ascending first, then a
    # STABLE descending sort on score -- equal scores keep index-ascending
    # order, non-equal scores are untouched (no perturbation tricks).
    ord1 = torch.argsort(cand_gidx)
    s1, g1 = cand_scores[ord1], cand_gidx[ord1]
    ord2 = torch.argsort(s1, descending=True, stable=True)
    winners = g1[ord2][:k]
    winners = winners[winners >= 0]

    mine = winners[(winners >= lo) & (winners < hi)] - lo
    if mine.numel():
        state.excitability.index_add_(
            0, mine, torch.full((mine.numel(),), params.jump,
                                dtype=exc.dtype, device=exc.device)
        )
    return winners


__all__ = ["allocate_global"]
