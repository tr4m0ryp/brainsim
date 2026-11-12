"""Arm catalog (contract Arms 1-5, driver-spec realizations). Arms
parameterize CompositeConfig ONLY (driver-spec rule); the two manipulations
without a config knob (binding swap, boundary surrogate) run through
systems.RunFlags on an unmodified config. Matched-budget arms are built by
budget.py, which asserts the 1% rule at construction.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from brainsim.composite.full.config import CompositeConfig
from brainsim.rules.cascade import CascadeParams

from . import budget
from .config import HEADLINE_LAMBDA, SEED_ROOT, H3Config, base_composite
from .protocol import arm_protocol, lambda_star_pred, shared_turnover_constant
from .systems import BagSystem, RunFlags, SingleSystem, build_runtime

# name -> (kind, readout). Readout 'union' = max(formation, storage) per
# formal-model #5; stage-alone arms read the stage under test; context rows
# read the published model's readout stage.
ARMS: dict[str, tuple[str, str]] = {
    "composite": ("composite", "union"),
    "separation_alone": ("stage", "separation"),
    "formation_alone": ("stage", "formation"),
    "relay_alone": ("stage", "relay"),
    "storage_alone": ("stage", "storage"),
    "bag": ("bag", "formation"),
    "btsp_to_stdp": ("ablation", "union"),
    "cascade_n1": ("ablation", "union"),
    "cascade_n8": ("ablation", "union"),
    "cascade_n16": ("ablation", "union"),
    "replay_off": ("ablation", "union"),
    "replay_uniform": ("ablation", "union"),
    "allocation_off": ("ablation", "union"),
    "relay_removed": ("ablation", "union"),
    "boundary_poisson": ("ablation", "union"),
    "row_normalize_off": ("ablation", "union"),
    "ctx_ecdgca3": ("context", "formation"),
    "ctx_kassab": ("context", "output"),
}

_STAGE_OF = {"separation_alone": "separation", "formation_alone": "formation",
             "relay_alone": "relay", "storage_alone": "storage"}
_CASCADE_N = {"cascade_n1": 1, "cascade_n8": 8, "cascade_n16": 16}


def derived_seed(seed: int) -> int:
    """Runtime/wiring seed for battery seed `seed`; shared by the composite
    and every ablation (paired circuits), layout-independent by construction."""
    ss = np.random.SeedSequence([SEED_ROOT, 6, int(seed)])
    return int(ss.generate_state(1)[0] % (2**31 - 1))


def _sigma_patch(cc: CompositeConfig) -> CompositeConfig:
    """Apply the H2-shared per-day turnover volatility to every arm."""
    return replace(cc, turnover_noise_sigma=shared_turnover_constant())


def arm_config(cfg: H3Config, seed: int, name: str) -> CompositeConfig:
    """The arm's CompositeConfig (single-runtime arms; bag is per-module)."""
    rs = derived_seed(seed)
    kind, _ = ARMS[name]
    if kind == "stage":
        cc = budget.stage_alone_config(cfg, seed, _STAGE_OF[name])
    elif kind == "context":
        cc = budget.context_config(cfg, seed, name)
    elif name == "relay_removed":
        cc = replace(budget.relay_removed_config(cfg, seed),
                     mb_write_steps=0, mb_gate_scale=0.0)
    else:
        cc = base_composite(cfg, seed)
        if name in _CASCADE_N:
            base_c = cc.cascade
            cc = replace(cc, cascade=CascadeParams(
                n_levels=_CASCADE_N[name], q1=base_c.q1, x=base_c.x))
        elif name == "allocation_off":
            cc = replace(cc, alloc_k=0, exc_drive_gain=0.0)
        elif name == "row_normalize_off":
            cc = replace(cc, row_normalize=False)
    return _sigma_patch(cc.with_seed(rs))


def arm_flags(name: str) -> RunFlags:
    if name == "btsp_to_stdp":
        return RunFlags(binding="stdp")
    if name == "replay_uniform":
        return RunFlags(uniform_replay=True)
    if name == "boundary_poisson":
        return RunFlags(poisson=True)
    return RunFlags()


def headline_lambda_pred(cfg: H3Config, seed: int) -> float:
    """lambda*_pred on the composite base config (the rate every
    storage-bearing arm shares at the headline factor)."""
    cc = base_composite(cfg, seed)
    return lambda_star_pred(cc, arm_protocol(cfg, 0.0))


def arm_lambda_rate(cfg: H3Config, seed: int, name: str,
                    lambda_factor: float) -> float:
    kind, _ = ARMS[name]
    if name == "replay_off" or kind == "bag":
        return 0.0
    pred = headline_lambda_pred(cfg, seed)
    if kind == "composite":
        return lambda_factor * pred
    return HEADLINE_LAMBDA * pred


def make_arm_system(cfg: H3Config, seed: int, name: str,
                    lambda_factor: float, group: int | None = None,
                    rank: int = 0, world_size: int = 1,
                    device: str = "cpu"):
    """Build the arm's System + protocol params + budget meta. Bag arms take
    `group` (this work item's module subset) and require world_size == 1
    (driver-spec: bag packs per GPU; composite-scale arms shard instead)."""
    kind, readout = ARMS[name]
    lam = arm_lambda_rate(cfg, seed, name, lambda_factor)
    pp = arm_protocol(cfg, lam)
    target = budget.total_synapses(base_composite(cfg, seed))

    if kind == "bag":
        if world_size != 1:
            raise RuntimeError("bag arms are single-process packed work "
                               "items; run them outside torchrun")
        n_mod, tgt = budget.bag_layout(cfg, seed)
        if group is None or not (0 <= group < cfg.bag_groups):
            raise ValueError(f"bag arm needs a group in [0, {cfg.bag_groups})")
        modules = {}
        group_total = 0
        for m in range(n_mod):
            mc = _sigma_patch(budget.bag_module_config(cfg, seed, m, n_mod, tgt))
            group_total += budget.total_synapses(mc)
            if m % cfg.bag_groups == group:
                modules[m] = build_runtime(mc, RunFlags(), device=device)
        budget.assert_matched("bag", group_total, tgt)  # sum over ALL modules
        meta = {"arm": name, "kind": kind, "n_modules": n_mod,
                "group": group, "n_groups": cfg.bag_groups,
                "budget_total": group_total, "budget_target": tgt,
                "lambda_rate": lam}
        return BagSystem(modules, n_mod), pp, meta

    cc = arm_config(cfg, seed, name)
    total = budget.total_synapses(cc)
    if kind in ("stage", "context") or name == "relay_removed":
        budget.assert_matched(name, total, target)
    rt = build_runtime(cc, arm_flags(name), rank=rank,
                       world_size=world_size, device=device)
    meta = {"arm": name, "kind": kind, "budget_total": total,
            "budget_target": target, "lambda_rate": lam,
            "lambda_star_pred": headline_lambda_pred(cfg, seed),
            "n_neurons": cc.n_global}
    return SingleSystem(rt, readout, arm_flags(name)), pp, meta


__all__ = ["ARMS", "arm_config", "arm_flags", "arm_lambda_rate",
           "headline_lambda_pred", "make_arm_system", "derived_seed"]
