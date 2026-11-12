"""run_full_smoke_cycle: run_smoke_cycle's exact schedule (same item count,
protect-first-item rule, salience range, 0.5-drop cues, 2 unwritten
controls, before/after around a 60-step leak-dominated storage phase) on the
full spiking runtime. Code check, never science.

Why the values differ from composite.pipeline's (the GATE, not the numbers,
is the anchor): codes here are top-f sets of cells that FIRED in real LIF
windows (ranked by integrated synaptic current) rather than signs of a
masked matvec; stages have different sizes (pipeline reuses n_exp
everywhere); overlaps are indicator cosines of id sets rather than
bipolar-vector cosines (no sign cancellation, so smoke-scale crowding sets
a nonzero control floor); drive carries only the +1 half of the bipolar
pattern (a spiking cell cannot be driven negative); and the retrieval
images carry an evidence floor that sends fully-eroded or novel recalls to
EMPTY (overlap 0) where pipeline's scale-invariant sparsify returns
residual-noise values. The mechanism gate is identical: protection =
cascade deep-push on the protected item's touched synapses -> turnover
immunity -> protected recall survives the storage phase that erases the
unprotected item.

Known smoke-scale behavior, stated plainly: pre-storage recall fidelity at
this crowded 2k-neuron scale is noisy (per-item overlaps ~0.1-0.6 across
seeds), and mean written-vs-control separation holds on most but not all
seeds; the pre-registered batteries measure capacity/retention at full
scale and do not consume this diagnostic.
"""
from __future__ import annotations

import numpy as np

from brainsim.composite.full.config import CompositeConfig
from brainsim.composite.full.phases import (
    encode_codes,
    make_cue,
    overlap_sets,
    recall_codes,
)
from brainsim.composite.full.runtime import CompositeRuntime
from brainsim.composite.pipeline import random_pattern

N_ITEMS = 5  # pipeline.run_smoke_cycle schedule
N_CONTROLS = 2
STORAGE_STEPS = 60
DROP_FRACTION = 0.5


def run_full_smoke_cycle(seed: int = 0, device: str = "cpu",
                         config: CompositeConfig | None = None,
                         rank: int = 0, world_size: int = 1) -> dict:
    """Returns the pipeline-shaped report dict plus a runtime snapshot.
    SPMD: under world_size > 1 every rank calls this identically (patterns,
    salience and cues come from the shared seed) and receives identical
    codes/overlaps; per-rank fields live under report['snapshot']."""
    cfg = (config or CompositeConfig.smoke()).with_seed(seed)
    rt = CompositeRuntime(cfg, rank=rank, world_size=world_size, device=device)
    rng = np.random.default_rng(seed + 1000)  # pipeline.run_smoke_cycle rng

    items = []
    for i in range(N_ITEMS):
        pattern = random_pattern(rng, cfg.n_input)
        salience = float(rng.uniform(0.5, 2.0))
        items.append((pattern, salience, i == 0))
    item_ids = rt.write_phase(items)

    def cued_overlap(item_id: int) -> float:
        rec = rt.items[item_id]
        cue = make_cue(rec.pattern, DROP_FRACTION, rng)
        codes = recall_codes(rt, cue)
        return overlap_sets(codes["storage"], rec.codes["storage"])

    overlaps = [cued_overlap(i) for i in item_ids]

    control_overlaps = []
    for _ in range(N_CONTROLS):
        # pipeline's control semantics: reference is the CLEAN encoding-mode
        # image of the unwritten pattern; recall retrieves the nearest
        # STORED memory for any input, so the retrieval of an unwritten
        # cue lands far from its own clean encoding.
        unwritten = random_pattern(rng, cfg.n_input)
        reference = encode_codes(rt, unwritten)["storage"]
        cue = make_cue(unwritten, DROP_FRACTION, rng)
        recalled = recall_codes(rt, cue)["storage"]
        control_overlaps.append(overlap_sets(recalled, reference))

    protected_id, unprotected_id = item_ids[0], item_ids[1]
    protected_before = cued_overlap(protected_id)
    unprotected_before = cued_overlap(unprotected_id)

    storage_report = rt.storage_phase(STORAGE_STEPS)

    protected_after = cued_overlap(protected_id)
    unprotected_after = cued_overlap(unprotected_id)

    return {
        "cycle_completed": True,
        "overlaps": overlaps,
        "control_overlaps": control_overlaps,
        "n_items": N_ITEMS,
        "protected_overlap_before": protected_before,
        "unprotected_overlap_before": unprotected_before,
        "protected_overlap_after": protected_after,
        "unprotected_overlap_after": unprotected_after,
        "storage_report": storage_report,
        "snapshot": rt.snapshot(),
    }


__all__ = ["run_full_smoke_cycle", "N_ITEMS", "STORAGE_STEPS", "DROP_FRACTION"]
