"""Single-collective fired-index exchange with fixed-capacity buffers.

The slow exchange (ShardedNetwork._exchange_fired) launches TWO all_gathers
per step -- counts, then a max-count-padded payload -- and allocates fresh
tensors for both. At biological firing rates the payload is tiny, so the
per-step cost is dominated by collective LAUNCH overhead; halving the
number of collectives halves that term. Here every rank all_gathers ONE
fixed-shape int64 row -- slot 0 carries the fired count, slots 1..cap the
fired global indices -- into a single preallocated flat tensor via
all_gather_into_tensor (NCCL-native on cuda, verified working on gloo under
torch 2.2.2). Fixed shapes are also the precondition for CUDA-graph capture
later; graphs are deliberately not used here (no GPU on the dev box to
validate them).

Capacity is a hard contract: a rank whose fired count exceeds `cap` RAISES,
never truncates -- a silently dropped spike would corrupt the raster. The
count slot makes concatenation rank-major, exactly the slow path's
rank-ordered torch.cat, so both exchanges yield the same global fired list
and therefore the same delivery order (bit-identical rasters).
"""
from __future__ import annotations

import torch
import torch.distributed as dist


class FastExchange:
    """Preallocated one-collective all_gather of fired global indices.

    `cap` must be identical on every rank (fixed-shape collective); the
    constructor all_gathers the caps once and raises on mismatch.
    world_size=1 allocates nothing and short-circuits to the local list,
    matching ShardedNetwork._exchange_fired.
    """

    def __init__(
        self, world_size: int, cap: int, device: torch.device | str, rank: int = 0
    ) -> None:
        self.world_size = int(world_size)
        self.cap = int(cap)
        self.rank = int(rank)
        self.device = torch.device(device)
        if self.cap < 1:
            raise ValueError(f"fast_exchange_cap must be >= 1, got {self.cap}")
        if self.world_size == 1:
            return
        w, c = self.world_size, self.cap
        self._send = torch.zeros(c + 1, dtype=torch.int64, device=self.device)
        self._recv = torch.empty(w * (c + 1), dtype=torch.int64, device=self.device)
        self._rows = self._recv.view(w, c + 1)
        self._slot = torch.arange(c, dtype=torch.int64, device=self.device).unsqueeze(0)
        self._mask = torch.empty((w, c), dtype=torch.bool, device=self.device)
        self._assert_uniform_cap()

    def _assert_uniform_cap(self) -> None:
        """One-time collective (init, never per step): mismatched caps would
        desynchronize the fixed-shape all_gather into a hang or corruption.
        """
        mine = torch.tensor([self.cap], dtype=torch.int64, device=self.device)
        caps = torch.empty(self.world_size, dtype=torch.int64, device=self.device)
        dist.all_gather_into_tensor(caps, mine)
        vals: list[int] = caps.tolist()
        if any(v != self.cap for v in vals):
            raise ValueError(
                "fast_exchange_cap must be identical on every rank "
                f"(fixed-shape collective), got {vals}"
            )

    def exchange(self, fired: torch.Tensor, step_idx: int | None = None) -> torch.Tensor:
        """All-gather `fired` (int64 GLOBAL indices, ascending) from every
        rank; returns the rank-major concatenation, globally ascending
        because rank ranges are ordered. Raises on capacity overflow BEFORE
        the collective (peer ranks then block until their timeout: an
        overflow is fatal to the run by design, size the cap up instead).
        """
        n = fired.numel()
        if n > self.cap:
            at = "" if step_idx is None else f" at t={step_idx}"
            raise RuntimeError(
                f"fast_exchange_cap overflow{at}: rank {self.rank} fired {n} "
                f"> cap {self.cap}; raise fast_exchange_cap (the default, "
                "ceil(n_global/world_size), cannot overflow) -- truncation "
                "is never allowed"
            )
        if self.world_size == 1:
            return fired
        self._send[0] = n
        if n:
            self._send[1 : 1 + n] = fired
        dist.all_gather_into_tensor(self._recv, self._send)
        torch.lt(self._slot, self._rows[:, :1], out=self._mask)
        # Row-major masked select == cat of per-rank [:count] slices.
        return self._rows[:, 1:][self._mask]


__all__ = ["FastExchange"]
