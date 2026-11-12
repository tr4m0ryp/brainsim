"""Work-item execution: one (arm, k, p, seed) cell of the contract grid.

Every work item is a pure function of (config, seeds) -- no state crosses
work items except immutable caches (the twin per seed, and the cascade-only
capacity per (seed, p), both deterministic recomputations) -- so any
arm-shard partition of the work list reproduces identical numbers
(amendment rule 5). The L_ref dependency on the cascade-only baseline is
resolved by pure recomputation in whichever shard needs it; the cascade-only
work item's owner emits the authoritative baseline results.
"""
from __future__ import annotations

import numpy as np

from brainsim.rules.cascade import CascadeParams

from .config import H2Config
from .engines import get_ops
from .motif import make_behavior
from .protocol import (ARM_CODES, PHASE_CAP, PHASE_RET, H2System, backend_seed,
                       host_rng, p_code, run_capacity, run_retention)
from .twin import Twin, build_twin, require_gate


def _cascade_params(cfg: H2Config) -> CascadeParams:
    return CascadeParams(n_levels=cfg.cascade_n_levels, q1=cfg.cascade_q1,
                         x=cfg.cascade_x)


def make_system(cfg: H2Config, twin: Twin, arm: str, k: int, p: float,
                seed: int, phase: int) -> H2System:
    ops = get_ops(cfg.engine, backend_seed(seed, p, k, arm, phase),
                  _cascade_params(cfg), device=cfg.torch_device,
                  weight_dtype=cfg.weight_dtype)
    stream_rng = host_rng(1, seed, p_code(p), phase)
    arm_rng = host_rng(3, seed, p_code(p), k, ARM_CODES[arm])
    behavior = make_behavior(arm, cfg, cfg.cascade_n_levels,
                             k if k > 0 else None, arm_rng)
    return H2System(ops, cfg, twin, behavior, stream_rng)


class RunCaches:
    """Immutable, deterministic intermediates shared across this process's
    work items. Twins are gate-checked at build (contract: no arms on an
    unvalidated twin); identical recomputation in other shards is a
    determinism property, not a rerun of the authoritative gate (shard 0
    emits the gate block)."""

    def __init__(self, cfg: H2Config, stats) -> None:
        self.cfg = cfg
        self.stats = stats
        self._twins: dict[int, Twin] = {}
        self._cap3: dict[tuple[int, int], dict] = {}

    def twin(self, seed: int) -> Twin:
        if seed not in self._twins:
            t = build_twin(self.stats, self.cfg, seed)
            require_gate(t)
            self._twins[seed] = t
        return self._twins[seed]

    def gates(self) -> dict:
        return {str(s): t.gate for s, t in sorted(self._twins.items())}

    def cascade_capacity(self, seed: int, p: float) -> dict:
        key = (seed, p_code(p))
        if key not in self._cap3:
            system = make_system(self.cfg, self.twin(seed), "cascade_only",
                                 0, p, seed, PHASE_CAP)
            self._cap3[key] = run_capacity(system, seed, p)
        return self._cap3[key]

    def l_ref(self, seed: int, p: float) -> int:
        cap3 = self.cascade_capacity(seed, p)
        return max(self.cfg.lref_floor,
                   int(round(self.cfg.lref_frac * cap3["C"])))


def run_work_item(caches: RunCaches, arm: str, k: int, p: float, seed: int) -> dict:
    cfg = caches.cfg
    twin = caches.twin(seed)
    cap3 = caches.cascade_capacity(seed, p)
    if cfg.scale == "full" and cap3["C"] <= 0:
        raise RuntimeError(
            f"cascade-only capacity is 0 at seed={seed} p={p} on the full-scale "
            "twin: protocol is degenerate, refusing to produce arm results")
    l_ref = caches.l_ref(seed, p)
    if arm == "cascade_only":
        capacity = dict(cap3)
    else:
        system = make_system(cfg, twin, arm, k, p, seed, PHASE_CAP)
        capacity = run_capacity(system, seed, p)
    ret_system = make_system(cfg, twin, arm, k, p, seed, PHASE_RET)
    retention = run_retention(ret_system, seed, p, l_ref)
    return {
        "arm": arm, "k": int(k), "p": float(p), "seed": int(seed),
        "capacity": capacity, "retention": retention,
        "C": capacity["C"], "t_half_promoted": retention["t_half_promoted"],
        "t_half_censored": retention["t_half_censored"],
    }


def build_work_list(cfg: H2Config, seeds: list[int]) -> list[tuple[str, int, float, int]]:
    """Canonical enumeration; sharding slices this list round-robin."""
    work: list[tuple[str, int, float, int]] = []
    for seed in seeds:
        for p in cfg.p_grid:
            work.append(("cascade_only", 0, float(p), int(seed)))
        for p in cfg.p_grid:
            for k in cfg.k_grid:
                for arm in ("redundancy", "uniform", "replay", "combined"):
                    work.append((arm, int(k), float(p), int(seed)))
    return work


def work_key(arm: str, k: int, p: float, seed: int) -> str:
    return f"{arm}|k={int(k)}|p={p:g}|seed={int(seed)}"


__all__ = ["RunCaches", "run_work_item", "build_work_list", "work_key",
           "make_system"]
