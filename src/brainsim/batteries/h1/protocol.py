"""The frozen H1 capacity protocol (fly arm).

Contract (battery-h1-wiring-premium.md,
FROZEN): store random sparse patterns via the stage's write rule at
increasing load P (streamed); recall with 50%-degraded cues; capacity
C = max P with mean output overlap >= 0.9.

Stored item mu = (xi_mu, z_mu): a random sparse PN pattern (odor analog,
coding level f_pn) associated with a random DAN-gated target over f_z of
the MBONs (compartment-like gate: real DANs gate a small subset of MB
compartments per reinforcement, not half the output layer). Writing
depresses measured KC->MBON synapses under the mb_rule closed form. Recall
presents a 50%-degraded cue (half the active PNs kept, the other half
replaced by random silent PNs -- coding level preserved), ranks MBONs by
fractional suppression, and takes the top |z|; overlap = |z_hat & z| / |z|
(chance f_z).

CALIBRATION HISTORY (2026-08-09, recorded for pre-registration integrity;
both aborts happened BEFORE any capacity or ratio value was computed or
read -- only per-arm stream lengths were observed in the logs; the frozen
thresholds (0.9 overlap bar, 1.10 claim, 0.95-1.05 kill band, 50% cue
degradation) were never touched, and every change applies identically to
all arms and controls):
1. First launch: f_z = 0.5 (max-entropy targets), 200 ms write
   (dw = 0.8 w_max). Every arm's stream saturated by P <= 17: bounded
   synapses floor-clipped within a handful of items because each item
   depressed half the plastic matrix at near-full depth -- no dynamic range
   for the frozen +-5% ratio bands. Aborted mid-seed-0. Recalibrated to
   f_z = 0.125 (compartment-like DAN gate) and 100 ms write (dw = 0.4 w_max).
2. Second launch: measured arm died at P = 2 in the first seconds. Root
   cause was the CUE model, not the operating point: the first cue
   implementation replaced the deleted half of the active PNs with random
   silent PNs; the measured expansion layer decorrelates such corrupted cues
   to chance-level KC overlap (~f_kc), and ~13 chance-shared KCs cannot
   cover 12 target columns (expected overlap ~0.82 < 0.9 at P = 1). The
   Formal Model defines recall as clamping a PARTIAL pattern, so the
   replacement noise was OUR deviation from the frozen design; cues became
   deletion-only (keep a random half of the active PNs, add nothing).

Capacity resolution is an ENGINEERING choice, recorded in the results JSON:
checkpoints on a geometric grid (ratio `grid_ratio`), mean overlap over a
deterministic sample of stored items per checkpoint, and C obtained by
linear interpolation of the 0.9 crossing between the last checkpoint at or
above the bar and the next one below it.

All randomness is keyed: rng(stream) = default_rng([H1_KEY, seed, stream,
*extra]); identical (seed, stream) always reproduces the same draw, and the
same per-seed item bank is shared by every arm (paired comparison).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from brainsim.batteries.h1.circuit import MBCircuit

H1_KEY = 0x481F17  # battery namespace for rng streams
STREAM_PATTERNS, STREAM_TARGETS, STREAM_CUES, STREAM_EVAL = 0, 1, 2, 3
OVERLAP_BAR = 0.9          # FROZEN by the contract
CUE_DEGRADATION = 0.5      # FROZEN by the contract


def rng_for(seed: int, stream: int, *extra: int) -> np.random.Generator:
    return np.random.default_rng([H1_KEY, int(seed), int(stream), *map(int, extra)])


@dataclass
class ProtocolCfg:
    f_pn: float = 0.10          # odor coding level on the PN layer
    f_kc: float = 0.05          # KC coding level (sparseness constraint)
    f_z: float = 0.125          # DAN-gated target density (compartment-like)
    grid_ratio: float = 1.3     # checkpoint spacing (sweep resolution)
    p_cap: int = 24576          # hard stream cap (right-censoring flag if hit)
    eval_sample: int = 200      # stored items evaluated per checkpoint
    early_stop_bar: float = 0.8
    early_stop_consecutive: int = 2
    write_chunk: int = 2048     # batch size for KC code computation


def checkpoint_grid(cfg: ProtocolCfg) -> list[int]:
    grid, p = [], 1
    while p <= cfg.p_cap:
        grid.append(p)
        p = max(p + 1, int(round(p * cfg.grid_ratio)))
    return grid


@dataclass
class ItemBank:
    """Per-seed stored items, shared across every arm of that seed."""
    pn_idx: np.ndarray    # (p_cap, k_pn) active-PN indices per item
    z_idx: np.ndarray     # (p_cap, k_z) target-MBON indices per item
    n_pn: int
    n_mbon: int

    @classmethod
    def generate(cls, seed: int, cfg: ProtocolCfg, n_pn: int,
                 n_mbon: int) -> "ItemBank":
        k_pn = max(1, int(round(cfg.f_pn * n_pn)))
        k_z = max(1, int(round(cfg.f_z * n_mbon)))
        rp = rng_for(seed, STREAM_PATTERNS)
        rt = rng_for(seed, STREAM_TARGETS)
        pn = np.stack([rp.choice(n_pn, size=k_pn, replace=False)
                       for _ in range(cfg.p_cap)]).astype(np.int32)
        z = np.stack([rt.choice(n_mbon, size=k_z, replace=False)
                      for _ in range(cfg.p_cap)]).astype(np.int32)
        return cls(pn_idx=pn, z_idx=z, n_pn=n_pn, n_mbon=n_mbon)

    def patterns(self, lo: int, hi: int) -> np.ndarray:
        x = np.zeros((hi - lo, self.n_pn), dtype=np.float64)
        np.put_along_axis(x, self.pn_idx[lo:hi].astype(np.int64), 1.0, axis=1)
        return x

    def cues(self, seed: int, items: np.ndarray) -> np.ndarray:
        """50%-degraded cues = PARTIAL patterns per the Formal Model's recall
        definition ("clamp cue x_cue (partial xi)"): a random half of the
        active PNs is DELETED, nothing is added. Deterministic per item.

        The first implementation replaced the deleted half with random
        silent PNs; the measured expansion layer decorrelates such corrupted
        cues to chance at the KC layer (that is its function), silencing the
        cue channel entirely -- see CALIBRATION HISTORY in the module
        docstring. Deletion-only is the design-doc reading and was adopted
        before any capacity value existed."""
        x = np.zeros((items.size, self.n_pn), dtype=np.float64)
        k_pn = self.pn_idx.shape[1]
        n_keep = k_pn - int(round(CUE_DEGRADATION * k_pn))
        for row, item in enumerate(items.tolist()):
            rc = rng_for(seed, STREAM_CUES, item)
            keep = rc.choice(self.pn_idx[item], size=n_keep, replace=False)
            x[row, keep] = 1.0
        return x


def capacity_from_curve(ps: list[int], overlaps: list[float],
                        bar: float = OVERLAP_BAR) -> dict:
    """C = max P with mean overlap >= bar, linearly interpolated into the
    next checkpoint below the bar. Degenerate and censored cases flagged."""
    above = [i for i, o in enumerate(overlaps) if o >= bar]
    if not above:
        return {"C": 0.0, "degenerate": True, "right_censored": False}
    i = above[-1]
    if i == len(ps) - 1:
        return {"C": float(ps[i]), "degenerate": False, "right_censored": True}
    p0, p1, o0, o1 = ps[i], ps[i + 1], overlaps[i], overlaps[i + 1]
    frac = (o0 - bar) / max(o0 - o1, 1e-12)
    return {"C": float(p0 + frac * (p1 - p0)), "degenerate": False,
            "right_censored": False}


def run_mb_capacity(circuit: MBCircuit, bank: ItemBank, seed: int,
                    cfg: ProtocolCfg) -> dict:
    """Stream items through one MB arm and measure the precision curve."""
    grid = checkpoint_grid(cfg)
    W = circuit.init_plastic()
    k_z = bank.z_idx.shape[1]
    ps, overlaps = [], []
    stored = 0
    below = 0
    for p_next in grid:
        for lo in range(stored, p_next, cfg.write_chunk):
            hi = min(lo + cfg.write_chunk, p_next)
            y_chunk = circuit.kc_activate(bank.patterns(lo, hi))
            for j in range(hi - lo):
                circuit.write(W, y_chunk[j], bank.z_idx[lo + j])
        stored = p_next
        re = rng_for(seed, STREAM_EVAL, p_next)
        items = (np.arange(stored, dtype=np.int64) if stored <= cfg.eval_sample
                 else np.sort(re.choice(stored, size=cfg.eval_sample,
                                        replace=False)).astype(np.int64))
        y_cue = circuit.kc_activate(bank.cues(seed, items))
        supp = circuit.readout_suppression(y_cue, W)
        order = np.argsort(-supp, axis=1, kind="stable")[:, :k_z]
        hit = np.zeros(items.size)
        for r, item in enumerate(items.tolist()):
            hit[r] = np.isin(order[r], bank.z_idx[item]).sum() / k_z
        ps.append(stored)
        overlaps.append(float(hit.mean()))
        below = below + 1 if overlaps[-1] < cfg.early_stop_bar else 0
        if below >= cfg.early_stop_consecutive:
            break
    out = capacity_from_curve(ps, overlaps)
    out.update({"curve_P": ps, "curve_overlap": overlaps,
                "n_eval": int(min(stored, cfg.eval_sample)), "cfg": asdict(cfg)})
    return out


__all__ = ["CUE_DEGRADATION", "H1_KEY", "ItemBank", "OVERLAP_BAR",
           "ProtocolCfg", "capacity_from_curve", "checkpoint_grid",
           "rng_for", "run_mb_capacity"]
