"""TorchNetwork: the numpy engine.network API on torch tensors (T5).

Public surface mirrors engine.network.Network -- add_population,
add_projection, step(external_I=...) returning {pop: bool ndarray} -- so the
bench and the composite pipeline can swap engines behind a factory. External
currents come in as numpy arrays or tensors; spikes go out as numpy bool
arrays (one small device-to-host copy per step; the sharded path in
engine.torch_core.shard keeps the exchange itself on-device).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import torch

from brainsim.engine.lif import LIFParams
from brainsim.engine.torch_core.fastpath.runners import run_network
from brainsim.engine.torch_core.lif_step import TorchLIFState, step as lif_step
from brainsim.engine.torch_core.propagate import TorchCSRProjection


def resolve_device(device: str = "auto") -> torch.device:
    """'auto' picks cuda when available, else cpu; explicit names pass through."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


@dataclass
class TorchPopulation:
    name: str
    n: int
    params: LIFParams
    device: torch.device
    state: TorchLIFState = field(init=False)

    def __post_init__(self) -> None:
        self.state = TorchLIFState.zeros(self.n, self.params, device=self.device)


@dataclass
class TorchProjection:
    name: str
    pre: str
    post: str
    csr: TorchCSRProjection


class TorchNetwork:
    """Wires named LIF populations via named CSR projections and steps them
    together. Same step order as the numpy Network: pop delayed currents,
    add external, integrate, spike, deliver.
    """

    def __init__(
        self, seed: int = 0, spike_buffer_size: int = 10_000, device: str = "auto"
    ) -> None:
        self.populations: dict[str, TorchPopulation] = {}
        self.projections: dict[str, TorchProjection] = {}
        self.rng = np.random.default_rng(seed)
        self.device = resolve_device(device)
        self.t = 0
        self.spike_buffer: deque = deque(maxlen=spike_buffer_size)

    def add_population(self, name: str, n: int, params: LIFParams) -> TorchPopulation:
        if name in self.populations:
            raise ValueError(f"population '{name}' already exists")
        pop = TorchPopulation(name=name, n=n, params=params, device=self.device)
        self.populations[name] = pop
        return pop

    def add_projection(
        self, name: str, pre: str, post: str, csr: TorchCSRProjection
    ) -> TorchProjection:
        if pre not in self.populations or post not in self.populations:
            raise ValueError(f"projection '{name}': unknown population(s) {pre!r}/{post!r}")
        if csr.n_pre != self.populations[pre].n or csr.n_post != self.populations[post].n:
            raise ValueError(f"projection '{name}': CSR shape does not match population sizes")
        proj = TorchProjection(name=name, pre=pre, post=post, csr=csr)
        self.projections[name] = proj
        return proj

    def step(self, external_I: dict | None = None) -> dict[str, np.ndarray]:
        """Advance every population by one dt. Returns {pop_name: spike_bool_array}."""
        external_I = external_I or {}

        currents = {
            name: torch.zeros(pop.n, dtype=torch.float32, device=self.device)
            for name, pop in self.populations.items()
        }
        for proj in self.projections.values():
            currents[proj.post] += proj.csr.pop_current()
        for name, extra in external_I.items():
            currents[name] += torch.as_tensor(
                extra, dtype=torch.float32, device=self.device
            )

        for name, pop in self.populations.items():
            pop.state.I = pop.state.I + currents[name]

        spikes_t: dict[str, torch.Tensor] = {}
        for name, pop in self.populations.items():
            spikes_t[name] = lif_step(pop.state, pop.params)

        for proj in self.projections.values():
            proj.csr.deliver(spikes_t[proj.pre])

        spikes = {name: s.cpu().numpy() for name, s in spikes_t.items()}
        self.spike_buffer.append((self.t, {k: v.copy() for k, v in spikes.items()}))
        self.t += 1
        return spikes

    def run(self, steps: int, drive=None, record: str | None = None):
        """Fast on-device loop (engine.torch_core.fastpath): no per-step
        device-to-host download, no spike_buffer appends, preallocated
        current buffers. `drive` is None, an fp32 device tensor (single-
        population nets), {pop_name: tensor}, or callable(step_idx) ->
        either. Returns (counts, rasters): per-step total spike counts as
        int64 numpy (downloaded once at the end) and, when record='raster',
        {pop_name: (steps, n) bool array}, else None. Rasters are
        bit-identical to a step() loop fed the same drive values; step()
        itself is unchanged.
        """
        return run_network(self, steps, drive=drive, record=record)

    def synapse_count(self) -> int:
        return sum(proj.csr.n_synapses for proj in self.projections.values())

    def neuron_count(self) -> int:
        return sum(pop.n for pop in self.populations.values())


__all__ = ["TorchNetwork", "TorchPopulation", "TorchProjection", "resolve_device"]
