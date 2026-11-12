"""Sparse synaptic propagation on CSR arrays (T1).

A CSRProjection is built once from a canonical connectivity table (indptr,
indices, weights, signs indexed by presynaptic neuron) and delivers
I[post] += weight*sign for every synapse of a spiking presynaptic neuron,
after an integer-step delay via a small ring buffer.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _expand_ranges(starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """Vectorized concatenation of [starts[i], ends[i]) ranges, no python loop
    over the (possibly many) spiking neurons.
    """
    counts = ends - starts
    total = int(counts.sum())
    if total == 0:
        return np.empty(0, dtype=np.int64)
    idx = np.repeat(starts, counts)
    run_start = np.repeat(np.cumsum(counts) - counts, counts)
    offset = np.arange(total, dtype=np.int64) - run_start
    return idx + offset


@dataclass
class CSRProjection:
    indptr: np.ndarray  # length n_pre + 1
    indices: np.ndarray  # post ids, length n_synapses
    weights: np.ndarray  # length n_synapses, >= 0
    signs: np.ndarray  # length n_synapses, in {-1, +1}
    n_post: int
    delay_steps: int = 1
    _ring: np.ndarray = field(init=False, repr=False)
    _ring_ptr: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.indptr = np.asarray(self.indptr, dtype=np.int64)
        self.indices = np.asarray(self.indices, dtype=np.int64)
        self.weights = np.asarray(self.weights, dtype=np.float64)
        self.signs = np.asarray(self.signs, dtype=np.int8)
        self.delay_steps = max(1, int(self.delay_steps))
        self._ring = np.zeros((self.delay_steps, self.n_post), dtype=np.float64)
        self._ring_ptr = 0

    @property
    def n_pre(self) -> int:
        return self.indptr.shape[0] - 1

    @property
    def n_synapses(self) -> int:
        return self.indices.shape[0]

    @classmethod
    def from_table(
        cls,
        pre_id: np.ndarray,
        post_id: np.ndarray,
        weight: np.ndarray,
        sign: np.ndarray,
        n_pre: int,
        n_post: int,
        delay_steps: int = 1,
    ) -> "CSRProjection":
        pre_id = np.asarray(pre_id, dtype=np.int64)
        order = np.argsort(pre_id, kind="stable")
        pre_sorted = pre_id[order]
        indptr = np.zeros(n_pre + 1, dtype=np.int64)
        counts = np.bincount(pre_sorted, minlength=n_pre)
        indptr[1:] = np.cumsum(counts)
        return cls(
            indptr=indptr,
            indices=np.asarray(post_id, dtype=np.int64)[order],
            weights=np.asarray(weight, dtype=np.float64)[order],
            signs=np.asarray(sign, dtype=np.int8)[order],
            n_post=n_post,
            delay_steps=delay_steps,
        )

    def deliver(self, pre_spikes: np.ndarray) -> None:
        """Schedule the synaptic contributions of every spiking presynaptic
        neuron for delivery `delay_steps` steps from now.
        """
        spiking = np.nonzero(pre_spikes)[0]
        if spiking.size == 0:
            return
        starts = self.indptr[spiking]
        ends = self.indptr[spiking + 1]
        syn_idx = _expand_ranges(starts, ends)
        if syn_idx.size == 0:
            return
        post_idx = self.indices[syn_idx]
        contrib = self.weights[syn_idx] * self.signs[syn_idx]
        target_slot = (self._ring_ptr + self.delay_steps - 1) % self.delay_steps
        np.add.at(self._ring[target_slot], post_idx, contrib)

    def pop_current(self) -> np.ndarray:
        """Return the contributions due *now* and advance the ring buffer."""
        out = self._ring[self._ring_ptr].copy()
        self._ring[self._ring_ptr] = 0.0
        self._ring_ptr = (self._ring_ptr + 1) % self.delay_steps
        return out


__all__ = ["CSRProjection"]
