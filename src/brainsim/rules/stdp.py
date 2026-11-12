"""Pair-based spike-timing-dependent plasticity (Bi & Poo 1998 windows, F4/T4).

Classic all-to-all pair-based STDP: exponential pre/post traces, potentiation
driven by the presynaptic trace at a postsynaptic spike (pre-before-post),
depression driven by the postsynaptic trace at a presynaptic spike
(post-before-pre). Default time constants and asymmetric amplitudes follow
Bi & Poo (1998, J. Neurosci): tau_plus=17ms, tau_minus=34ms.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class STDPParams:
    tau_plus: float = 17.0  # ms, Bi & Poo 1998
    tau_minus: float = 34.0  # ms, Bi & Poo 1998
    a_plus: float = 0.01
    a_minus: float = 0.012
    w_min: float = 0.0
    w_max: float = 1.0
    dt: float = 1.0  # ms


@dataclass
class STDPState:
    trace_pre: np.ndarray  # per presynaptic neuron
    trace_post: np.ndarray  # per postsynaptic neuron
    W: np.ndarray  # per synapse

    @classmethod
    def zeros(cls, n_pre: int, n_post: int, n_syn: int, w_init: float = 0.0) -> "STDPState":
        return cls(
            trace_pre=np.zeros(n_pre, dtype=np.float64),
            trace_post=np.zeros(n_post, dtype=np.float64),
            W=np.full(n_syn, w_init, dtype=np.float64),
        )


def step(
    state: STDPState,
    params: STDPParams,
    pre_spikes: np.ndarray,
    post_spikes: np.ndarray,
    pre_idx: np.ndarray,
    post_idx: np.ndarray,
) -> np.ndarray:
    """Advance traces and weights by one dt in place; returns the updated W."""
    state.trace_pre = state.trace_pre * np.exp(-params.dt / params.tau_plus)
    state.trace_post = state.trace_post * np.exp(-params.dt / params.tau_minus)

    if np.any(post_spikes):
        dW_pot = params.a_plus * state.trace_pre[pre_idx] * post_spikes[post_idx]
        state.W = np.clip(state.W + dW_pot, params.w_min, params.w_max)
    if np.any(pre_spikes):
        dW_dep = params.a_minus * state.trace_post[post_idx] * pre_spikes[pre_idx]
        state.W = np.clip(state.W - dW_dep, params.w_min, params.w_max)

    state.trace_pre = np.where(pre_spikes, state.trace_pre + 1.0, state.trace_pre)
    state.trace_post = np.where(post_spikes, state.trace_post + 1.0, state.trace_post)
    return state.W


__all__ = ["STDPParams", "STDPState", "step"]
