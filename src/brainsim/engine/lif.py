"""Vectorized leaky integrate-and-fire (LIF) neuron core (T1).

v' = v + dt * (-(v - v_rest) + R*I) / tau_mem; spikes at v >= v_th, reset to
v_reset, then a refractory counter blocks integration for `refractory` ms.
Synaptic current I decays with time constant tau_syn between deliveries
(engine.propagate adds to it; this module only decays and integrates).

Default v_rest/v_th/refractory/tau_syn follow the whole-fly LIF parameters in
Shiu et al. 2024 (Nature, 10.1038/s41586-024-07763-9), used on an analog mV
scale (T1); tau_mem is a standard cortical/hippocampal LIF value not pinned
by that paper.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class LIFParams:
    v_rest: float = -52.0  # mV, Shiu et al. 2024
    v_th: float = -45.0  # mV, Shiu et al. 2024
    v_reset: float = -52.0  # mV, reset to rest
    tau_mem: float = 20.0  # ms, membrane time constant
    tau_syn: float = 5.0  # ms, Shiu et al. 2024
    refractory: float = 2.2  # ms, Shiu et al. 2024
    R: float = 1.0  # membrane resistance (normalized units)
    dt: float = 1.0  # ms, integration step

    @property
    def refractory_steps(self) -> int:
        return max(1, int(round(self.refractory / self.dt)))


@dataclass
class LIFState:
    v: np.ndarray
    refractory_counter: np.ndarray
    I: np.ndarray

    @classmethod
    def zeros(cls, n: int, params: LIFParams) -> "LIFState":
        return cls(
            v=np.full(n, params.v_rest, dtype=np.float64),
            refractory_counter=np.zeros(n, dtype=np.int64),
            I=np.zeros(n, dtype=np.float64),
        )

    @property
    def n(self) -> int:
        return self.v.shape[0]


def step(state: LIFState, params: LIFParams) -> np.ndarray:
    """Advance `state` by one dt in place; returns the boolean spike vector."""
    active = state.refractory_counter <= 0
    dv = params.dt * (-(state.v - params.v_rest) + params.R * state.I) / params.tau_mem
    state.v = np.where(active, state.v + dv, params.v_reset)

    spikes = active & (state.v >= params.v_th)
    state.v = np.where(spikes, params.v_reset, state.v)

    state.refractory_counter = np.maximum(state.refractory_counter - 1, 0)
    state.refractory_counter = np.where(spikes, params.refractory_steps, state.refractory_counter)

    state.I = state.I * np.exp(-params.dt / params.tau_syn)
    return spikes


__all__ = ["LIFParams", "LIFState", "step"]
