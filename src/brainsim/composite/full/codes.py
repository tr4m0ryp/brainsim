"""Code algebra for the composite runtime: indicator-set overlaps, cue
degradation, and the iterated recurrent-image completion (the spiking
runtime's version of composite.pipeline's relax step)."""
from __future__ import annotations

import numpy as np
import torch


def overlap_sets(a: np.ndarray, b: np.ndarray) -> float:
    """Indicator cosine |A n B| / sqrt(|A| |B|); 0.0 if either code is empty
    (pipeline._overlap's zero-denominator convention)."""
    if a.size == 0 or b.size == 0:
        return 0.0
    inter = np.intersect1d(a, b, assume_unique=True).size
    return float(inter) / float(np.sqrt(a.size * b.size))


def make_cue(pattern: np.ndarray, drop_fraction: float,
             rng: np.random.Generator) -> np.ndarray:
    """pipeline.make_cue semantics: zero a random fraction of entries."""
    cue = pattern.astype(np.float64).copy()
    n_drop = int(round(drop_fraction * pattern.size))
    idx = rng.choice(pattern.size, size=n_drop, replace=False)
    cue[idx] = 0.0
    return cue


def bool_at(n: int, ids, device) -> torch.Tensor:
    out = torch.zeros(n, dtype=torch.bool, device=device)
    idx = torch.as_tensor(np.asarray(ids), dtype=torch.int64, device=device)
    if idx.numel():
        out[idx] = True
    return out


def local_ids(rt, ids: np.ndarray) -> np.ndarray:
    lo, hi = rt.net.local_range
    return ids[(ids >= lo) & (ids < hi)] - lo


def recurrent_image_code(rt, edge_name: str, stage: str, seeds: np.ndarray,
                         iters: int | None = None) -> np.ndarray:
    """Retrieval readout through a plastic recurrent edge: the synaptic
    image of the seed set through the LIVE weights -- composite.pipeline's
    relax step (x_out = sparsify(W @ x)), ITERATED completion_iters times
    (classic associative settling; pipeline documents iterative settling as
    this runtime's territory) -- computed as batched gathers over the
    sharded CSR parts. Monotone in the stored weights, so BTSP-bound
    engrams win the formation image and cascade-protected rows survive the
    storage image exactly as the numpy gate requires. Cells enter a
    completed code only above the evidence floor
    (config.image_evidence_frac of a full member's expected current), so
    novel cues and fully-eroded assemblies image to EMPTY instead of
    confabulating the nearest stored item. Free-running spiking attractor
    completion was measured threshold-marginal at smoke scale (ignition
    cliffs, mixed final states under mean-field inhibition);
    run_window(inhibit=...) keeps that mode available for battery
    experiments, but the v1 readout is the image."""
    cfg = rt.config
    iters = cfg.completion_iters if iters is None else iters
    es = rt.edges[edge_name]
    plo = rt.ranges[stage][0]
    code = np.asarray(seeds, dtype=np.int64)
    n = rt.ranges[stage][1] - plo
    f_stage = cfg.coding_level(stage)
    if edge_name == "form_rec":
        w_full, k_rec = cfg.btsp_w_max(), cfg.k_form_rec
    else:
        w_full, k_rec = cfg.w_max_store * rt.store_unit, cfg.k_store_rec
    theta = cfg.image_evidence_frac * k_rec * f_stage * w_full
    arange_pen = torch.arange(n, dtype=torch.float64) * 1e-9
    for _ in range(max(1, iters)):
        ind_pre = torch.zeros(cfg.n_global, dtype=torch.float32, device=rt.device)
        idx = torch.as_tensor(code, dtype=torch.int64, device=rt.device)
        if idx.numel():
            ind_pre[idx] = 1.0
        image = torch.zeros(rt.net.n_local, dtype=torch.float32, device=rt.device)
        for p in es.parts:
            proj = rt.net.projections[p]
            contrib = proj.weights.to(torch.float32) * ind_pre[es.pre_idx[p]]
            image.index_add_(0, es.post_idx[p], contrib)
        g = rt.stage_global(stage, image)
        score = torch.where(
            g > theta,
            g.to(torch.float64) - arange_pen.to(g.device),
            torch.tensor(float("-inf"), dtype=torch.float64, device=g.device),
        )
        top = torch.topk(score, min(cfg.code_size(stage), n))
        keep = torch.isfinite(top.values)
        code = np.sort(top.indices[keep].cpu().numpy()) + plo
        if code.size == 0:
            break
    return code


def storage_image_code(rt, seeds: np.ndarray) -> np.ndarray:
    return recurrent_image_code(rt, "store_rec", "storage", seeds)


__all__ = ["overlap_sets", "make_cue", "bool_at", "local_ids",
           "recurrent_image_code", "storage_image_code"]
