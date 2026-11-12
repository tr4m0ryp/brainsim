"""Torch port of rules.replay (F11, Kali-Dayan prioritised replay). Same
registration semantics and probability law as the numpy scheduler: salience
must be >= 0, sampling is with replacement proportional to salience, uniform
when total salience is 0, and every event carries the slow lr multiplier.
ReplayParams/ReplayEvent are imported from the numpy module.

Sampling uses torch.multinomial under an explicit torch.Generator, so the
event stream is DISTRIBUTION-equivalent to the numpy scheduler, not
stream-identical (different RNG); the equivalence test is statistical, plus
exact degenerate paths. The scheduler itself is control-plane state, O(#items)
python, never per-synapse; the per-synapse replay-driven write is applied by
the caller (e.g. the composite adapter's Hebbian write scaled by
event.lr_multiplier), fully batched there.

Sharding note (F25): replay needs no remote per-synapse state; ranks either
replicate the scheduler under a shared seed or broadcast rank-0's events.

`dtype` controls the salience/probability tensor (fp16 supported for the
storage-format sanity path); probabilities are upcast to fp32 for
torch.multinomial, which rejects half on CPU.
"""
from __future__ import annotations

import torch

from brainsim.rules.replay import ReplayEvent, ReplayParams


class TorchReplayScheduler:
    def __init__(
        self,
        params: ReplayParams,
        generator: torch.Generator | None = None,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.params = params
        self.generator = generator
        self.device = torch.device(device)
        self.dtype = dtype
        self._salience: dict[int, float] = {}

    def register(self, item_id: int, salience: float) -> None:
        if salience < 0:
            raise ValueError("salience must be >= 0")
        self._salience[item_id] = float(salience)

    def unregister(self, item_id: int) -> None:
        self._salience.pop(item_id, None)

    def items(self) -> dict[int, float]:
        return dict(self._salience)

    def _probs(self) -> tuple[list[int], torch.Tensor]:
        ids = list(self._salience.keys())
        sal = torch.tensor(
            [self._salience[i] for i in ids], dtype=self.dtype, device=self.device
        )
        total = sal.sum()
        if float(total) <= 0.0:
            probs = torch.full((len(ids),), 1.0 / len(ids), dtype=self.dtype, device=self.device)
        else:
            probs = sal / total
        return ids, probs

    def run_offline_phase(self, budget: int | None = None) -> list[ReplayEvent]:
        budget = self.params.budget if budget is None else budget
        if not self._salience:
            return []
        ids, probs = self._probs()
        chosen = torch.multinomial(
            probs.to(torch.float32), budget, replacement=True, generator=self.generator
        )
        return [
            ReplayEvent(item_id=ids[int(j)], lr_multiplier=self.params.lr_multiplier)
            for j in chosen.cpu()
        ]


__all__ = ["ReplayParams", "ReplayEvent", "TorchReplayScheduler"]
