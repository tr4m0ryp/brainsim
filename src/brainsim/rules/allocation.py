"""Excitability-based engram allocation, per-NEURON state (F6, Cai et al.
2016, Nature 10.1038/nature17955).

Each neuron carries one scalar excitability e_i. Allocation jumps e_i for the
neurons selected this event, then e_i decays exponentially with tau (default
36h, inside the measured 24-48h window). Selection is top-k on excitability
plus noise, so neurons allocated recently (elevated e_i) preferentially win
the next allocation -- co-allocation ("shared engrams") for events close in
time emerges purely from the decay, with no explicit linking mechanism.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SECONDS_PER_HOUR = 3600.0


@dataclass
class AllocationParams:
    tau: float = 36.0 * SECONDS_PER_HOUR  # seconds, Cai et al. 2016 (24-48h range)
    jump: float = 1.0  # excitability increment on allocation
    noise_std: float = 0.1
    k: int = 10  # neurons allocated per event


@dataclass
class AllocationState:
    excitability: np.ndarray

    @classmethod
    def zeros(cls, n: int) -> "AllocationState":
        return cls(excitability=np.zeros(n, dtype=np.float64))


def decay(state: AllocationState, params: AllocationParams, dt_seconds: float) -> None:
    """Exponentially decay excitability by `dt_seconds` of elapsed time, in place."""
    state.excitability = state.excitability * np.exp(-dt_seconds / params.tau)


def allocate(
    state: AllocationState,
    params: AllocationParams,
    rng: np.random.Generator,
    k: int | None = None,
) -> np.ndarray:
    """Select and boost the top-k(excitability + noise) neurons; returns their
    indices (descending score order). Mutates `state.excitability` in place.
    """
    k = params.k if k is None else k
    n = state.excitability.shape[0]
    k = min(k, n)
    noisy = state.excitability + rng.normal(0.0, params.noise_std, size=n)
    top = np.argpartition(-noisy, k - 1)[:k]
    idx = top[np.argsort(-noisy[top])]
    state.excitability[idx] = state.excitability[idx] + params.jump
    return idx


__all__ = ["AllocationParams", "AllocationState", "decay", "allocate", "SECONDS_PER_HOUR"]
