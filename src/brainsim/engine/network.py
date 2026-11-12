"""Network: named populations + projections, stepped together (T1).

Deterministic under a seeded numpy Generator: nothing here draws randomness
unless a caller supplies external noise, so repeated runs from the same
seed/config are bit-identical.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from brainsim.engine.lif import LIFParams, LIFState, step as lif_step
from brainsim.engine.propagate import CSRProjection


@dataclass
class Population:
    name: str
    n: int
    params: LIFParams
    state: LIFState = field(init=False)

    def __post_init__(self) -> None:
        self.state = LIFState.zeros(self.n, self.params)


@dataclass
class Projection:
    name: str
    pre: str
    post: str
    csr: CSRProjection


class Network:
    """Wires named LIF populations via named CSR projections and steps them
    together. Spike history is kept in a bounded deque (oldest dropped first).
    """

    def __init__(self, seed: int = 0, spike_buffer_size: int = 10_000) -> None:
        self.populations: dict[str, Population] = {}
        self.projections: dict[str, Projection] = {}
        self.rng = np.random.default_rng(seed)
        self.t = 0
        self.spike_buffer: deque = deque(maxlen=spike_buffer_size)

    def add_population(self, name: str, n: int, params: LIFParams) -> Population:
        if name in self.populations:
            raise ValueError(f"population '{name}' already exists")
        pop = Population(name=name, n=n, params=params)
        self.populations[name] = pop
        return pop

    def add_projection(self, name: str, pre: str, post: str, csr: CSRProjection) -> Projection:
        if pre not in self.populations or post not in self.populations:
            raise ValueError(f"projection '{name}': unknown population(s) {pre!r}/{post!r}")
        if csr.n_pre != self.populations[pre].n or csr.n_post != self.populations[post].n:
            raise ValueError(f"projection '{name}': CSR shape does not match population sizes")
        proj = Projection(name=name, pre=pre, post=post, csr=csr)
        self.projections[name] = proj
        return proj

    def step(self, external_I: dict[str, np.ndarray] | None = None) -> dict[str, np.ndarray]:
        """Advance every population by one dt. Returns {pop_name: spike_bool_array}."""
        external_I = external_I or {}

        currents = {name: np.zeros(pop.n) for name, pop in self.populations.items()}
        for proj in self.projections.values():
            currents[proj.post] += proj.csr.pop_current()
        for name, extra in external_I.items():
            currents[name] += extra

        for name, pop in self.populations.items():
            pop.state.I = pop.state.I + currents[name]

        spikes: dict[str, np.ndarray] = {}
        for name, pop in self.populations.items():
            spikes[name] = lif_step(pop.state, pop.params)

        for proj in self.projections.values():
            proj.csr.deliver(spikes[proj.pre])

        self.spike_buffer.append((self.t, {k: v.copy() for k, v in spikes.items()}))
        self.t += 1
        return spikes

    def synapse_count(self) -> int:
        return sum(proj.csr.n_synapses for proj in self.projections.values())

    def neuron_count(self) -> int:
        return sum(pop.n for pop in self.populations.values())


__all__ = ["Network", "Population", "Projection"]
