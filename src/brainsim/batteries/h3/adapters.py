"""System adapters: the protocol's uniform interface over one composite
runtime (SingleSystem) or a group of independent bag modules (BagSystem).
Scoring: 'union' readout = max(formation, storage) overlap per formal-model
#5 (fast OR slow store); a stage readout scores that stage's recall alone.
"""
from __future__ import annotations

from brainsim.composite.full import phases
from brainsim.composite.full.codes import overlap_sets
from brainsim.composite.full.runtime import CompositeRuntime

from .binding import BindingSwap
from .flags import RunFlags


# ---- System adapters (the protocol's uniform interface) --------------------

class SingleSystem:
    """One runtime + a readout. readout 'union' scores an item on
    max(formation, storage) overlap (formal-model #5: fast OR slow store);
    a stage name scores on that stage's recalled code alone."""

    def __init__(self, rt, readout: str, flags: RunFlags):
        self.rt = rt
        self.readout = readout
        self.flags = flags
        self.swap = BindingSwap(rt) if flags.binding == "stdp" else None
        self.ids: dict[int, int] = {}  # stream index -> runtime item id

    def write_item(self, index: int, pattern, salience, protect) -> None:
        if self.swap is not None:
            with self.swap:
                (iid,) = self.rt.write_phase([(pattern, salience, protect)])
        else:
            (iid,) = self.rt.write_phase([(pattern, salience, protect)])
        self.ids[index] = iid
        if self.flags.uniform_replay:  # ablation: flatten replay priority
            for k in self.rt.replay.items():
                self.rt.replay.register(k, 0.0)

    def owns(self, index: int) -> bool:
        return True

    def _scores(self, codes, rec) -> dict:
        if self.readout == "union":
            fast = overlap_sets(codes["formation"], rec.codes["formation"])
            slow = overlap_sets(codes["storage"], rec.codes["storage"])
            return {"score": max(fast, slow), "fast": fast, "slow": slow}
        key = self.readout
        ov = overlap_sets(codes[key], rec.codes[key])
        return {"score": ov, "fast": ov, "slow": 0.0}

    def recall(self, index: int, cue) -> dict:
        rec = self.rt.items[self.ids[index]]
        return self._scores(phases.recall_codes(self.rt, cue), rec)

    def recall_pair(self, index: int, other: int, cue) -> tuple[float, float]:
        codes = phases.recall_codes(self.rt, cue)
        key = "formation" if self.readout == "union" else self.readout
        self_ov = overlap_sets(codes[key],
                               self.rt.items[self.ids[index]].codes[key])
        cross_ov = overlap_sets(codes[key],
                                self.rt.items[self.ids[other]].codes[key])
        return self_ov, cross_ov

    def pattern_of(self, index: int):
        return self.rt.items[self.ids[index]].pattern

    def day(self, replay_budget: int) -> None:
        self.rt.storage_phase(steps=1, replay_budget=replay_budget)


class BagSystem:
    """N independent formation-copies; item (and interference) stream index
    i routes to module i mod N; this instance holds one GROUP of modules
    (m % n_groups == group). Modules never interact, so group results merge
    by summation. No storage function: day() is a no-op and the readout is
    formation (module memories erode only via routed interference)."""

    readout = "formation"

    def __init__(self, modules: dict[int, CompositeRuntime], n_modules: int):
        self.modules = modules  # module_id -> runtime (this group's only)
        self.n_modules = n_modules
        self.ids: dict[int, tuple[int, int]] = {}

    def home(self, index: int) -> int:
        return index % self.n_modules

    def owns(self, index: int) -> bool:
        return self.home(index) in self.modules

    def write_item(self, index: int, pattern, salience, protect) -> None:
        if not self.owns(index):
            return
        rt = self.modules[self.home(index)]
        (iid,) = rt.write_phase([(pattern, salience, protect)])
        self.ids[index] = (self.home(index), iid)

    def recall(self, index: int, cue) -> dict:
        m, iid = self.ids[index]
        rt = self.modules[m]
        codes = phases.recall_codes(rt, cue)
        ov = overlap_sets(codes["formation"], rt.items[iid].codes["formation"])
        return {"score": ov, "fast": ov, "slow": 0.0}

    def recall_pair(self, index: int, other: int, cue) -> tuple[float, float]:
        m, iid = self.ids[index]
        rt = self.modules[m]
        codes = phases.recall_codes(rt, cue)
        self_ov = overlap_sets(codes["formation"], rt.items[iid].codes["formation"])
        if self.home(other) != m:  # disjoint modules: routing IS separation
            return self_ov, 0.0
        _, oid = self.ids[other]
        return self_ov, overlap_sets(codes["formation"],
                                     rt.items[oid].codes["formation"])

    def pattern_of(self, index: int):
        m, iid = self.ids[index]
        return self.modules[m].items[iid].pattern

    def day(self, replay_budget: int) -> None:
        return None  # no consolidation store in the bag baseline

__all__ = ["SingleSystem", "BagSystem"]
