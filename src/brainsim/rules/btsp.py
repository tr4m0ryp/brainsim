"""Behavioral timescale synaptic plasticity, Milstein-2021 form (F2).

Bittner & Magee 2017 (10.1126/science.aan3846) plateau-gated one-shot
formation, formalised as an eligibility trace (ET, tau ~864 ms) times an
instructive signal (IS, tau ~543 ms) driving a self-limiting weight update
(Milstein et al. 2021, eLife):
    dW = (Wmax - W) * k_plus  * q_plus(ET * IS)
       -        W  * k_minus * q_minus(ET * IS)

q_plus/q_minus are sigmoidal nonlinearities of the ET*IS coincidence signal,
normalised so q(0) = 0: with no plateau, IS == 0 everywhere, so the product is
0 and the weight is exactly unchanged (no silent leak between plateau events).
ET is tracked per presynaptic neuron, IS per postsynaptic neuron; W is
per-synapse and gathers both via pre_idx/post_idx.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BTSPParams:
    tau_et: float = 864.0  # ms, Milstein et al. 2021
    tau_is: float = 543.0  # ms, Milstein et al. 2021
    k_plus: float = 1.0  # per-second potentiation rate
    k_minus: float = 1.0  # per-second depression rate
    w_max: float = 1.0
    q_plus_slope: float = 12.0
    q_plus_mid: float = 0.5
    q_minus_slope: float = 12.0
    q_minus_mid: float = 0.5
    dt: float = 1.0  # ms


def _q(x: np.ndarray, slope: float, mid: float) -> np.ndarray:
    """Sigmoid of the coincidence signal, shifted so q(0) == 0 exactly (no
    plateau/no-recent-activity must mean no plasticity, not a small leak).
    """
    raw = 1.0 / (1.0 + np.exp(-slope * (x - mid)))
    baseline = 1.0 / (1.0 + np.exp(-slope * (0.0 - mid)))
    return np.clip((raw - baseline) / (1.0 - baseline), 0.0, None)


@dataclass
class BTSPState:
    et_pre: np.ndarray  # per presynaptic neuron, in [0, 1]
    is_post: np.ndarray  # per postsynaptic neuron, in [0, 1]
    W: np.ndarray  # per synapse

    @classmethod
    def zeros(cls, n_pre: int, n_post: int, n_syn: int, w_init: float = 0.0) -> "BTSPState":
        return cls(
            et_pre=np.zeros(n_pre, dtype=np.float64),
            is_post=np.zeros(n_post, dtype=np.float64),
            W=np.full(n_syn, w_init, dtype=np.float64),
        )


def step(
    state: BTSPState,
    params: BTSPParams,
    pre_spikes: np.ndarray,
    post_plateau: np.ndarray,
    pre_idx: np.ndarray,
    post_idx: np.ndarray,
) -> np.ndarray:
    """Advance traces and weights by one dt in place; returns the updated W."""
    state.et_pre = state.et_pre * np.exp(-params.dt / params.tau_et)
    state.is_post = state.is_post * np.exp(-params.dt / params.tau_is)
    state.et_pre = np.where(pre_spikes, 1.0, state.et_pre)
    state.is_post = np.where(post_plateau, 1.0, state.is_post)

    coincidence = state.et_pre[pre_idx] * state.is_post[post_idx]
    q_plus = _q(coincidence, params.q_plus_slope, params.q_plus_mid)
    q_minus = _q(coincidence, params.q_minus_slope, params.q_minus_mid)

    dW = (params.w_max - state.W) * params.k_plus * q_plus - state.W * params.k_minus * q_minus
    state.W = np.clip(state.W + dW * (params.dt / 1000.0), 0.0, params.w_max)
    return state.W


__all__ = ["BTSPParams", "BTSPState", "step"]
