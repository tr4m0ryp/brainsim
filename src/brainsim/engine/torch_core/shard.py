"""Post-owned neuron-range sharding over torch.distributed (T5, F25).

Rank r owns a contiguous slice [lo, hi) of the global neuron index space:
the slice's LIF state and EVERY synapse whose POSTsynaptic neuron lies in it
(post-owned CSR, rows still keyed by GLOBAL presynaptic id). Per step: local
LIF update -> all ranks exchange fired GLOBAL indices via all_gather (NCCL
when the tensors live on cuda, so the exchange is NVLink-native on an 8-GPU
node; gloo when cpu) -> each rank walks the global spike list through its
local CSR slice. Spike tensors are padded to the per-step max count because
NCCL all_gather requires equal shapes; counts travel in a first 1-element
all_gather. world_size=1 runs with NO torch.distributed init at all: the
exchange short-circuits to the local fired list.

step() is the reference path and is unchanged; run() is the opt-in fast
path (engine.torch_core.fastpath): on-device multi-step loop, single
fixed-capacity all_gather per step, bit-identical rasters.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.distributed as dist

from brainsim.engine.lif import LIFParams
from brainsim.engine.torch_core.fastpath import FastExchange, run_sharded
from brainsim.engine.torch_core.lif_step import TorchLIFState, step as lif_step
from brainsim.engine.torch_core.network import resolve_device
from brainsim.engine.torch_core.propagate import TorchCSRProjection


def partition_ranges(n_global: int, world_size: int) -> list[tuple[int, int]]:
    """Contiguous [lo, hi) neuron ranges; the remainder goes to the first ranks."""
    if world_size < 1 or n_global < world_size:
        raise ValueError(f"cannot partition {n_global} neurons over {world_size} ranks")
    base, rem = divmod(n_global, world_size)
    ranges, lo = [], 0
    for r in range(world_size):
        hi = lo + base + (1 if r < rem else 0)
        ranges.append((lo, hi))
        lo = hi
    return ranges


def default_backend(device: torch.device | str) -> str:
    """NCCL for cuda ranks (NVLink-native collectives), gloo for cpu."""
    return "nccl" if torch.device(device).type == "cuda" else "gloo"


class ShardedNetwork:
    """One global LIF population sharded across ranks. Global connectivity
    tables go in via add_projection; each rank keeps only its post-owned
    slice. step() takes the GLOBAL external current and returns the LOCAL
    spike bool array (callers stitch rasters by local_range).
    """

    def __init__(
        self,
        n_global: int,
        params: LIFParams,
        rank: int = 0,
        world_size: int = 1,
        device: str = "auto",
        fast_exchange_cap: int | None = None,
    ) -> None:
        if not 0 <= rank < world_size:
            raise ValueError(f"rank {rank} outside world_size {world_size}")
        if world_size > 1 and not (dist.is_available() and dist.is_initialized()):
            raise RuntimeError(
                "world_size > 1 requires an initialized torch.distributed process group"
            )
        self.n_global = int(n_global)
        self.params = params
        self.rank = rank
        self.world_size = world_size
        self.device = resolve_device(device)
        self.lo, self.hi = partition_ranges(self.n_global, world_size)[rank]
        self.state = TorchLIFState.zeros(self.n_local, params, device=self.device)
        self.projections: dict[str, TorchCSRProjection] = {}
        self.t = 0
        # Per-rank fired capacity of run()'s single-collective exchange. The
        # default, ceil(n_global/world_size), is UNIFORM across ranks (the
        # fixed-shape collective requires it; n_local is not uniform when the
        # partition is uneven) and >= every n_local, so it cannot overflow.
        # Callers size it down to the expected max fired per rank per step;
        # overflow raises, never truncates. step() ignores it entirely.
        cap = (
            -(-self.n_global // world_size)
            if fast_exchange_cap is None
            else int(fast_exchange_cap)
        )
        if cap < 1:
            raise ValueError(f"fast_exchange_cap must be >= 1, got {cap}")
        self.fast_exchange_cap = cap
        self._fast_exchange: FastExchange | None = None

    @property
    def n_local(self) -> int:
        return self.hi - self.lo

    @property
    def local_range(self) -> tuple[int, int]:
        return (self.lo, self.hi)

    def add_projection(
        self,
        name: str,
        pre_id,
        post_id,
        weight,
        sign,
        delay_steps: int = 1,
        fp16: bool = False,
    ) -> TorchCSRProjection:
        """Take the GLOBAL COO table, keep rows with post in [lo, hi), localize
        the post index, and build the pre-global/post-local CSR slice.
        """
        if name in self.projections:
            raise ValueError(f"projection '{name}' already exists")
        pre = torch.as_tensor(pre_id, dtype=torch.int64)
        post = torch.as_tensor(post_id, dtype=torch.int64)
        w = torch.as_tensor(weight, dtype=torch.float32)
        s = torch.as_tensor(sign, dtype=torch.float32)
        mask = (post >= self.lo) & (post < self.hi)
        proj = TorchCSRProjection.from_table(
            pre[mask],
            post[mask] - self.lo,
            w[mask],
            s[mask],
            n_pre=self.n_global,
            n_post=self.n_local,
            delay_steps=delay_steps,
            fp16=fp16,
            device=self.device,
        )
        self.projections[name] = proj
        return proj

    def _exchange_fired(self, fired: torch.Tensor) -> torch.Tensor:
        """All-gather fired global indices. Rank-order concatenation keeps the
        list ascending (ranges are ordered), matching the world_size=1 order.
        """
        if self.world_size == 1:
            return fired
        n_fired = torch.tensor([fired.numel()], dtype=torch.int64, device=self.device)
        counts = [torch.zeros_like(n_fired) for _ in range(self.world_size)]
        dist.all_gather(counts, n_fired)
        max_n = max(int(c) for c in counts)
        if max_n == 0:
            return torch.empty(0, dtype=torch.int64, device=self.device)
        padded = torch.zeros(max_n, dtype=torch.int64, device=self.device)
        padded[: fired.numel()] = fired
        gathered = [torch.empty_like(padded) for _ in range(self.world_size)]
        dist.all_gather(gathered, padded)
        return torch.cat([gathered[r][: int(counts[r])] for r in range(self.world_size)])

    def step(self, external_I=None) -> np.ndarray:
        """Advance the local slice by one dt; returns the LOCAL spike bool array.
        Same order as engine.network.Network.step: pop delayed currents, add
        external, integrate, spike, exchange, deliver.
        """
        currents = torch.zeros(self.n_local, dtype=torch.float32, device=self.device)
        for proj in self.projections.values():
            currents += proj.pop_current()
        if external_I is not None:
            ext = torch.as_tensor(external_I, dtype=torch.float32, device=self.device)
            currents += ext[self.lo : self.hi]

        self.state.I = self.state.I + currents
        spikes = lif_step(self.state, self.params)

        fired_global = torch.nonzero(spikes).flatten() + self.lo
        fired_global = self._exchange_fired(fired_global)
        for proj in self.projections.values():
            proj.deliver_fired(fired_global)

        self.t += 1
        return spikes.cpu().numpy()

    def _ensure_fast_exchange(self) -> FastExchange:
        """Lazily (re)build run()'s exchange buffers; rebuilds if
        fast_exchange_cap was changed between run() calls."""
        ex = self._fast_exchange
        if ex is None or ex.cap != self.fast_exchange_cap:
            ex = FastExchange(
                self.world_size, self.fast_exchange_cap, self.device, rank=self.rank
            )
            self._fast_exchange = ex
        return ex

    def run(self, steps: int, drive=None, record: str | None = None):
        """Fast on-device loop (engine.torch_core.fastpath): no per-step
        device-to-host download, ONE all_gather per step into fixed-capacity
        buffers (fast_exchange_cap; overflow raises), preallocated current
        buffers. `drive` is the LOCAL external current -- None, an (n_local,)
        fp32 tensor on this net's device reused every step, or
        callable(step_idx) -> such a tensor -- unlike step(external_I=...)
        which takes the GLOBAL vector and slices it. Returns (counts,
        raster): per-step LOCAL spike counts as int64 numpy (downloaded once
        at the end; all_reduce for global totals), plus the (steps, n_local)
        bool raster when record='raster', else None. Rasters are
        bit-identical to a step() loop fed the same drive values.
        """
        return run_sharded(self, steps, drive=drive, record=record)

    def synapse_count(self) -> int:
        return sum(proj.n_synapses for proj in self.projections.values())


__all__ = ["ShardedNetwork", "default_backend", "partition_ranges"]
