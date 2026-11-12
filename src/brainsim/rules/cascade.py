"""Fusi-Abbott cascade synapse, uint8 state (F3, Fusi & Abbott 2005).

Each synapse holds one of 2*n discrete metaplastic states: n "depressed"
levels [0 .. n-1] (n-1 = shallowest, 0 = deepest) and n "potentiated" levels
[n .. 2n-1] (n = shallowest, 2n-1 = deepest). The effective binary weight is
the branch (depressed=0 / potentiated=1). A plasticity or noise event flips
the state with probability q_i = q_1 * x^(i-1) at depth i (1 = shallowest),
so deep/"protected" states resist both correct-direction and noise-driven
transitions -- the metaplastic push/pull operations below let deep states be
installed deliberately (the H01 redundancy-motif proxy at rule level).
State is one byte per synapse (2n <= 256), O(1) per event, fully vectorized.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_N_LEVELS = 8  # Fusi & Abbott 2005, n ~ 5-10, fits one byte (2n states)


@dataclass
class CascadeParams:
    n_levels: int = DEFAULT_N_LEVELS
    q1: float = 0.5  # transition probability at the shallowest level
    x: float = 0.5  # geometric decay factor, 0 < x < 1


def depth(state: np.ndarray, params: CascadeParams) -> np.ndarray:
    """Depth i in [1, n_levels], 1 = shallowest (boundary-adjacent)."""
    n = params.n_levels
    state = np.asarray(state)
    return np.where(state < n, n - state, state - n + 1)


def transition_prob(state: np.ndarray, params: CascadeParams) -> np.ndarray:
    i = depth(state, params).astype(np.float64)
    return params.q1 * params.x ** (i - 1.0)


def effective_weight(state: np.ndarray, params: CascadeParams) -> np.ndarray:
    """Binary readout: 0 = depressed branch, 1 = potentiated branch."""
    return (np.asarray(state) >= params.n_levels).astype(np.uint8)


def _apply_direction(
    state: np.ndarray, fire_mask: np.ndarray, params: CascadeParams, sign: int
) -> np.ndarray:
    """sign=+1 (push, drive toward potentiated) or sign=-1 (pull, drive toward
    depressed). Firing synapses move one level in that direction; crossing the
    branch boundary happens naturally because the two branches are encoded
    adjacently (n-1 <-> n).
    """
    n = params.n_levels
    new_state = state.astype(np.int64).copy()
    pot_branch = state >= n
    if sign > 0:
        m = fire_mask & pot_branch
        new_state[m] = np.minimum(new_state[m] + 1, 2 * n - 1)
        m = fire_mask & ~pot_branch
        new_state[m] = new_state[m] + 1
    else:
        m = fire_mask & ~pot_branch
        new_state[m] = np.maximum(new_state[m] - 1, 0)
        m = fire_mask & pot_branch
        new_state[m] = new_state[m] - 1
    return new_state.astype(np.uint8)


def push(state: np.ndarray, params: CascadeParams, rng: np.random.Generator, mask: np.ndarray | None = None) -> np.ndarray:
    """Metaplastic push: drive `state` toward/deeper into the potentiated
    branch, level-dependent probability. `mask` restricts which synapses are
    eligible this call (default: all).
    """
    p = transition_prob(state, params)
    fire = rng.random(state.shape) < p
    if mask is not None:
        fire &= mask
    return _apply_direction(state, fire, params, sign=+1)


def pull(state: np.ndarray, params: CascadeParams, rng: np.random.Generator, mask: np.ndarray | None = None) -> np.ndarray:
    """Metaplastic pull: drive `state` toward/deeper into the depressed
    branch, level-dependent probability.
    """
    p = transition_prob(state, params)
    fire = rng.random(state.shape) < p
    if mask is not None:
        fire &= mask
    return _apply_direction(state, fire, params, sign=-1)


def deepen(state: np.ndarray, params: CascadeParams, steps: int = 1) -> np.ndarray:
    """Deterministic protection op: push `state` `steps` levels deeper within
    its current branch (used to install the H01 redundancy/permanence motif
    at rule level, no randomness).
    """
    n = params.n_levels
    new_state = state.astype(np.int64).copy()
    pot_branch = state >= n
    new_state[pot_branch] = np.minimum(new_state[pot_branch] + steps, 2 * n - 1)
    new_state[~pot_branch] = np.maximum(new_state[~pot_branch] - steps, 0)
    return new_state.astype(np.uint8)


__all__ = [
    "DEFAULT_N_LEVELS",
    "CascadeParams",
    "depth",
    "transition_prob",
    "effective_weight",
    "push",
    "pull",
    "deepen",
]
