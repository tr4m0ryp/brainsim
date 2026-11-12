"""Prioritized offline replay scheduler (F11, Wilson & McNaughton 1994;
Kali & Dayan 2004, salience-tagged sharp-wave-ripple replay).

Items register with a salience tag; each offline phase samples a fixed
budget of replay events with probability proportional to salience (sampling
with replacement, so a high-salience item can be replayed multiple times per
phase). Replay events carry a slow learning-rate multiplier so the network
can apply them as reduced-rate, replay-driven writes (F4: eta_ctx ~
0.01-0.1 x eta_hpc).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ReplayParams:
    budget: int = 10  # replay events per offline phase
    lr_multiplier: float = 0.05  # Kali & Dayan 2004, eta_ctx ~ 0.01-0.1 x eta_hpc


@dataclass
class ReplayEvent:
    item_id: int
    lr_multiplier: float


class ReplayScheduler:
    def __init__(self, params: ReplayParams, rng: np.random.Generator) -> None:
        self.params = params
        self.rng = rng
        self._salience: dict[int, float] = {}

    def register(self, item_id: int, salience: float) -> None:
        if salience < 0:
            raise ValueError("salience must be >= 0")
        self._salience[item_id] = float(salience)

    def unregister(self, item_id: int) -> None:
        self._salience.pop(item_id, None)

    def items(self) -> dict[int, float]:
        return dict(self._salience)

    def run_offline_phase(self, budget: int | None = None) -> list[ReplayEvent]:
        budget = self.params.budget if budget is None else budget
        if not self._salience:
            return []
        ids = np.array(list(self._salience.keys()))
        saliences = np.array([self._salience[i] for i in ids], dtype=np.float64)
        total = saliences.sum()
        probs = np.full(ids.shape, 1.0 / ids.shape[0]) if total <= 0 else saliences / total
        chosen = self.rng.choice(ids, size=budget, replace=True, p=probs)
        return [ReplayEvent(item_id=int(i), lr_multiplier=self.params.lr_multiplier) for i in chosen]


__all__ = ["ReplayParams", "ReplayEvent", "ReplayScheduler"]
