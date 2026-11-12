"""Device-agnostic torch LIF core (T5): the numpy `engine.lif` dynamics
reproduced op-for-op on torch tensors.

Neuron state is fp32 by design (F24): state is per-neuron and small next to
the synapse arrays, so fp16 belongs to projection weights (engine.torch_core
.propagate), never to membrane state. Semantics mirror engine.lif.step
exactly -- leak toward v_rest, spike at v >= v_th, reset to v_reset, integer
refractory counter that clamps v to v_reset while it runs, synaptic current
decays by exp(-dt/tau_syn) each step -- in the same operation order, so the
two engines produce identical spike rasters wherever fp32-vs-fp64 rounding
cannot flip a threshold comparison.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from brainsim.engine.lif import LIFParams


@dataclass
class TorchLIFState:
    v: torch.Tensor  # fp32, membrane potential (mV)
    refractory_counter: torch.Tensor  # int32, refractory steps remaining
    I: torch.Tensor  # fp32, synaptic current

    @classmethod
    def zeros(
        cls, n: int, params: LIFParams, device: torch.device | str = "cpu"
    ) -> "TorchLIFState":
        device = torch.device(device)
        return cls(
            v=torch.full((n,), params.v_rest, dtype=torch.float32, device=device),
            refractory_counter=torch.zeros(n, dtype=torch.int32, device=device),
            I=torch.zeros(n, dtype=torch.float32, device=device),
        )

    @property
    def n(self) -> int:
        return self.v.shape[0]

    @property
    def device(self) -> torch.device:
        return self.v.device


def step(state: TorchLIFState, params: LIFParams) -> torch.Tensor:
    """Advance `state` by one dt in place; returns the boolean spike tensor."""
    active = state.refractory_counter <= 0
    dv = params.dt * (-(state.v - params.v_rest) + params.R * state.I) / params.tau_mem
    state.v = torch.where(active, state.v + dv, params.v_reset)

    spikes = active & (state.v >= params.v_th)
    state.v = torch.where(spikes, params.v_reset, state.v)

    state.refractory_counter = torch.clamp(state.refractory_counter - 1, min=0)
    state.refractory_counter = torch.where(
        spikes, params.refractory_steps, state.refractory_counter
    )

    state.I = state.I * math.exp(-params.dt / params.tau_syn)
    return spikes


__all__ = ["TorchLIFState", "step"]
