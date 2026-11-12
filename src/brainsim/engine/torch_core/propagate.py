"""CSR spike propagation on torch (T5, F24): int32 column indices (uint32
semantics -- per-rank post ids never reach 2^31), fp16 or fp32 weights with
the polarity sign folded in at load time, per-projection integer delay via
the same ring-buffer scheme as the numpy engine (engine.propagate).

Byte layout per synapse is weight (2 B fp16 / 4 B fp32) + index (4 B int32);
the int64 indptr and the fp32 delivery ring are per-neuron, not per-synapse
(accounted in engine.torch_core.memory). Accumulation is always fp32: fp16
is a storage format, gathered contributions are upcast before index_add_.
"""
from __future__ import annotations

import torch


def _expand_ranges(starts: torch.Tensor, ends: torch.Tensor) -> torch.Tensor:
    """Vectorized concatenation of [starts[i], ends[i]) ranges, no python loop
    over the (possibly many) spiking neurons. Torch mirror of
    engine.propagate._expand_ranges.
    """
    counts = ends - starts
    total = int(counts.sum())
    if total == 0:
        return torch.empty(0, dtype=torch.int64, device=starts.device)
    idx = torch.repeat_interleave(starts, counts)
    run_start = torch.repeat_interleave(torch.cumsum(counts, 0) - counts, counts)
    offset = torch.arange(total, dtype=torch.int64, device=starts.device) - run_start
    return idx + offset


class TorchCSRProjection:
    """CSR projection keyed by presynaptic neuron. `weights` carry the sign
    (weight * sign, folded once in fp32 before any fp16 downcast), so
    delivery is a single gather + index_add_ on the owning device.
    """

    def __init__(
        self,
        indptr: torch.Tensor,
        indices: torch.Tensor,
        weights: torch.Tensor,
        n_post: int,
        delay_steps: int = 1,
    ) -> None:
        self.indptr = indptr.to(torch.int64)
        self.indices = indices.to(torch.int32)
        self.weights = weights  # fp16 or fp32, signed
        self.n_post = int(n_post)
        self.delay_steps = max(1, int(delay_steps))
        device = self.weights.device
        self._ring = torch.zeros(
            (self.delay_steps, self.n_post), dtype=torch.float32, device=device
        )
        self._ring_ptr = 0

    @property
    def n_pre(self) -> int:
        return self.indptr.shape[0] - 1

    @property
    def n_synapses(self) -> int:
        return self.indices.shape[0]

    @property
    def device(self) -> torch.device:
        return self.weights.device

    @classmethod
    def from_table(
        cls,
        pre_id,
        post_id,
        weight,
        sign,
        n_pre: int,
        n_post: int,
        delay_steps: int = 1,
        fp16: bool = False,
        device: torch.device | str = "cpu",
    ) -> "TorchCSRProjection":
        """Build from a COO connectivity table (numpy arrays or tensors).
        Stable sort by pre id, exactly like engine.propagate.from_table, so
        equal-pre synapses keep their table order on both engines.
        """
        device = torch.device(device)
        pre = torch.as_tensor(pre_id, dtype=torch.int64)
        post = torch.as_tensor(post_id, dtype=torch.int64)
        w = torch.as_tensor(weight, dtype=torch.float32)
        s = torch.as_tensor(sign, dtype=torch.float32)
        order = torch.argsort(pre, stable=True)

        indptr = torch.zeros(n_pre + 1, dtype=torch.int64)
        counts = torch.bincount(pre[order], minlength=n_pre)
        indptr[1:] = torch.cumsum(counts, 0)

        signed = (w * s)[order]  # fold polarity once, in fp32
        weights = signed.to(torch.float16 if fp16 else torch.float32)
        return cls(
            indptr=indptr.to(device),
            indices=post[order].to(torch.int32).to(device),
            weights=weights.to(device),
            n_post=n_post,
            delay_steps=delay_steps,
        )

    def deliver(self, pre_spikes: torch.Tensor) -> None:
        """Schedule contributions of every spiking presynaptic neuron for
        delivery `delay_steps` steps from now. `pre_spikes` is a bool tensor.
        """
        self.deliver_fired(torch.nonzero(pre_spikes).flatten())

    def deliver_fired(self, fired: torch.Tensor) -> None:
        """Same as deliver, but takes the int64 indices of fired presynaptic
        neurons directly (the form the sharded exchange produces, F25).
        """
        if fired.numel() == 0:
            return
        starts = self.indptr[fired]
        ends = self.indptr[fired + 1]
        syn_idx = _expand_ranges(starts, ends)
        if syn_idx.numel() == 0:
            return
        post_idx = self.indices.index_select(0, syn_idx)
        contrib = self.weights.index_select(0, syn_idx).to(torch.float32)
        target_slot = (self._ring_ptr + self.delay_steps - 1) % self.delay_steps
        self._ring[target_slot].index_add_(0, post_idx, contrib)

    def pop_current(self) -> torch.Tensor:
        """Return the contributions due *now* and advance the ring buffer."""
        out = self._ring[self._ring_ptr].clone()
        self._ring[self._ring_ptr].zero_()
        self._ring_ptr = (self._ring_ptr + 1) % self.delay_steps
        return out

    def accumulate_current_into(self, out: torch.Tensor) -> None:
        """Fast-path form of pop_current (engine.torch_core.fastpath): add the
        due ring slot into `out` in place -- no per-step clone allocation --
        then zero it and advance. `out += slot` performs the same fp32
        addition step() applies to the cloned row, so both paths produce
        bit-identical currents.
        """
        due = self._ring[self._ring_ptr]
        out += due
        due.zero_()
        self._ring_ptr = (self._ring_ptr + 1) % self.delay_steps


__all__ = ["TorchCSRProjection"]
