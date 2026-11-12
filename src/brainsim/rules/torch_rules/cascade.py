"""Torch port of rules.cascade (F3, Fusi-Abbott cascade synapse), int8 state
per synapse (F22 byte budget: cascade1). The numpy module is the spec;
CascadeParams and the state encoding (n depressed levels [0..n-1], n
potentiated levels [n..2n-1], depth 1 = shallowest) are identical, and all
ops are batched over any state shape with no python loop over synapses.

Documented deviations from the numpy module, all equivalence-preserving:
- state dtype is torch.int8 per the F22 layout (numpy uses uint8), so
  2*n_levels <= 128 (n_levels <= 64); the default n=8 uses states 0..15.
  Level arithmetic runs in int64 and casts back, mirroring numpy's int64
  round-trip, so int8 overflow can never occur mid-update.
- the stochastic ops take `rand` = pre-drawn uniform draws (a tensor, any
  shape matching `state`) OR a torch.Generator, instead of consuming a
  hidden numpy Generator. Pre-drawn uniforms are what makes cross-engine
  exactness testable and let one seeded stream drive both engines.
- transition probabilities are gathered from a per-depth float64 table
  computed with the numpy expression itself, so `u < p` decisions are
  bit-identical to the numpy rule under identical uniforms for ANY params,
  not only the dyadic defaults.
"""
from __future__ import annotations

import numpy as np
import torch

from brainsim.rules.cascade import DEFAULT_N_LEVELS, CascadeParams

STATE_DTYPE = torch.int8


def _check_params(params: CascadeParams) -> None:
    if 2 * params.n_levels > 128:
        raise ValueError(f"n_levels={params.n_levels} needs states beyond int8 (2n <= 128)")


def depth(state: torch.Tensor, params: CascadeParams) -> torch.Tensor:
    """Depth i in [1, n_levels], 1 = shallowest (boundary-adjacent); int64."""
    n = params.n_levels
    s = state.to(torch.int64)
    return torch.where(s < n, n - s, s - n + 1)


def _prob_table(params: CascadeParams, device: torch.device) -> torch.Tensor:
    """q_i = q1 * x^(i-1) per depth i, evaluated with the numpy float64
    expression so gathered values match rules.cascade.transition_prob bitwise."""
    depths = np.arange(1, params.n_levels + 1, dtype=np.float64)
    return torch.from_numpy(params.q1 * params.x ** (depths - 1.0)).to(device)


def transition_prob(state: torch.Tensor, params: CascadeParams) -> torch.Tensor:
    """float64 transition probability per synapse (table gather)."""
    return _prob_table(params, state.device)[depth(state, params) - 1]


def effective_weight(state: torch.Tensor, params: CascadeParams) -> torch.Tensor:
    """Binary readout: 0 = depressed branch, 1 = potentiated branch (uint8)."""
    return (state.to(torch.int64) >= params.n_levels).to(torch.uint8)


def _uniforms(
    rand: torch.Tensor | torch.Generator, shape: torch.Size, device: torch.device
) -> torch.Tensor:
    if isinstance(rand, torch.Generator):
        return torch.rand(shape, generator=rand, dtype=torch.float64, device=device)
    u = torch.as_tensor(rand, device=device)
    if u.shape != shape:
        raise ValueError(f"uniform draws shape {tuple(u.shape)} != state shape {tuple(shape)}")
    return u


def _move(state: torch.Tensor, fire: torch.Tensor, params: CascadeParams, sign: int) -> torch.Tensor:
    """One level in `sign` direction for firing synapses. A single clamp per
    direction is exact: for sign=+1 only the potentiated branch can reach the
    2n-1 rail (depressed tops out at n-1+1 = n <= 2n-1), and symmetrically
    for sign=-1 at the 0 rail, matching numpy _apply_direction branch by
    branch."""
    n = params.n_levels
    s = state.to(torch.int64)
    moved = torch.clamp(s + 1, max=2 * n - 1) if sign > 0 else torch.clamp(s - 1, min=0)
    return torch.where(fire, moved, s).to(STATE_DTYPE)


def push(
    state: torch.Tensor,
    params: CascadeParams,
    rand: torch.Tensor | torch.Generator,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Metaplastic push toward/deeper into the potentiated branch with
    level-dependent probability; `mask` restricts eligibility this call."""
    _check_params(params)
    fire = _uniforms(rand, state.shape, state.device) < transition_prob(state, params)
    if mask is not None:
        fire = fire & mask
    return _move(state, fire, params, sign=+1)


def pull(
    state: torch.Tensor,
    params: CascadeParams,
    rand: torch.Tensor | torch.Generator,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Metaplastic pull toward/deeper into the depressed branch."""
    _check_params(params)
    fire = _uniforms(rand, state.shape, state.device) < transition_prob(state, params)
    if mask is not None:
        fire = fire & mask
    return _move(state, fire, params, sign=-1)


def deepen(state: torch.Tensor, params: CascadeParams, steps: int = 1) -> torch.Tensor:
    """Deterministic protection op: `steps` levels deeper within the current
    branch (H01 redundancy/permanence motif at rule level, no randomness)."""
    _check_params(params)
    n = params.n_levels
    s = state.to(torch.int64)
    deeper = torch.where(
        s >= n, torch.clamp(s + steps, max=2 * n - 1), torch.clamp(s - steps, min=0)
    )
    return deeper.to(STATE_DTYPE)


__all__ = [
    "DEFAULT_N_LEVELS",
    "STATE_DTYPE",
    "CascadeParams",
    "depth",
    "transition_prob",
    "effective_weight",
    "push",
    "pull",
    "deepen",
]
