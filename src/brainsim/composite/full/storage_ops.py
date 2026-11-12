"""Storage-stage plasticity operations: the slow Hebbian write with cascade
protection (composite.pipeline's write/cascade schedule on the sharded
per-synapse tensors) and the storage phase (leak-dominated turnover scaled
by cascade-depth resistance, then the prioritised-replay transfer, F11)."""
from __future__ import annotations

import numpy as np
import torch

from brainsim.composite.full.codes import bool_at, local_ids
from brainsim.rules.torch_rules import cascade as t_cascade


def store_write(rt, code: np.ndarray, eta_units: float, protect: bool) -> None:
    """Hebbian write on co-active storage pairs (in units of the derived
    store weight), then cascade push on touched synapses and a deterministic
    deepen when protected -- pipeline._advance_cascade_on_write exactly.
    Cascade uniforms come from the rank-keyed generator: deterministic per
    (config, world_size); the only cross-world-size-variant randomness in
    the runtime (documented in runtime.py)."""
    cfg = rt.config
    es = rt.edges["store_rec"]
    ind_pre = bool_at(cfg.n_global, code, rt.device)
    ind_post = bool_at(rt.net.n_local, local_ids(rt, code), rt.device)
    w_max = cfg.w_max_store * rt.store_unit
    for p in es.parts:
        proj = rt.net.projections[p]
        act = ind_pre[es.pre_idx[p]] & ind_post[es.post_idx[p]]
        w = proj.weights.to(torch.float32)
        w = torch.clamp(w + eta_units * rt.store_unit * act.to(torch.float32),
                        -w_max, w_max)
        proj.weights.copy_(w)
        idx = act.nonzero().flatten()
        if idx.numel():
            u = torch.rand(idx.shape, generator=rt.rank_gen,
                           dtype=torch.float64, device=rt.device)
            sub = t_cascade.push(es.cascade[p][idx], cfg.cascade, u)
            if protect:
                sub = t_cascade.deepen(sub, cfg.cascade, steps=cfg.cascade.n_levels)
            es.cascade[p][idx] = sub


def storage_phase(rt, steps: int, noise_sigma: float | None = None,
                  replay_budget: int | None = None) -> dict:
    """Turnover noise on touched storage synapses (leak-dominated, cascade-
    resistance scaled: pipeline.run_storage_phase semantics on per-synapse
    tensors), then the prioritised-replay transfer (F11): each sampled
    item's storage code is re-written at lr_multiplier rate with a cascade
    push -- the formation-held item re-consolidating into the slow store.
    replay_budget=0 (the default config) mirrors the numpy smoke cycle,
    which registers items but never runs an offline phase."""
    cfg = rt.config
    sigma_units = cfg.turnover_noise_sigma if noise_sigma is None else noise_sigma
    sigma = sigma_units * rt.store_unit
    w_max = cfg.w_max_store * rt.store_unit
    es = rt.edges["store_rec"]
    n_touched = 0
    for p in es.parts:
        proj = rt.net.projections[p]
        w = proj.weights.to(torch.float32)
        idx = (w != 0.0).nonzero().flatten()
        if idx.numel() == 0:
            continue
        n_touched += int(idx.numel())
        sub = w[idx]
        depth = t_cascade.depth(es.cascade[p][idx], cfg.cascade).to(torch.float32)
        resistance = depth / cfg.cascade.n_levels
        decay = 1.0 - cfg.leak_rate * (1.0 - resistance)
        noise_scale = sigma * (1.0 - resistance)
        for _ in range(steps):
            noise = torch.randn(sub.shape, generator=rt.rank_gen,
                                dtype=torch.float32, device=rt.device)
            sub = torch.clamp(sub * decay + noise * noise_scale, -w_max, w_max)
        w[idx] = sub
        proj.weights.copy_(w)

    budget = cfg.replay_budget if replay_budget is None else replay_budget
    events = rt.replay.run_offline_phase(budget) if budget > 0 else []
    for ev in events:  # identical stream on every rank (shared generator)
        rec = rt.items[ev.item_id]
        store_write(rt, rec.codes["storage"],
                    cfg.eta_store * ev.lr_multiplier, protect=False)
    return {"steps": steps, "touched_local": n_touched,
            "replay_events": len(events)}


__all__ = ["store_write", "storage_phase"]
