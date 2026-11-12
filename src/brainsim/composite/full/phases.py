"""Phase implementations for CompositeRuntime (write / recall), mirroring
composite.pipeline's cycle semantics at spiking fidelity.

Mechanism map (numpy pipeline -> this runtime):
- encode_separation      -> the shared FRONT window (_front_window): input
                            pattern drive, free separation spiking, mossy
                            detonators + Treves-Rolls direct path into
                            formation, top-f readouts from spike windows
- _hebbian_write(form)   -> BTSP one-shot binding (F2) onto the engram,
                            allocation-biased (F6, global top-k, F25-sharded)
- _hebbian_write(store)  -> storage_ops.store_write (slow Hebbian +
                            cascade push, deepen when protected)
- relax(W_form/W_store)  -> codes.recurrent_image_code: the synaptic image
                            of the spiking-derived seed through the live
                            plastic weights, iterated (associative settling)
- run_storage_phase      -> storage_ops.storage_phase (turnover + cascade
                            resistance + prioritised replay transfer)
- MB-gated relay coupling (Formal Model "MB-style gated depression at
  readout", T10): applied after the relay window each write with dan =
  salience * mb_gate_scale. At F1 default alpha/tau_w the per-write
  depression is ~1e-4 of the derived ceiling: the coupling is exercised
  mechanically, not calibrated for effect; batteries own that sweep.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from brainsim.composite.full.codes import (
    bool_at,
    make_cue,
    overlap_sets,
    recurrent_image_code,
    storage_image_code,
)
from brainsim.composite.full.storage_ops import storage_phase, store_write
from brainsim.rules.torch_rules import (
    allocation as t_allocation,
    btsp as t_btsp,
    mb_rule as t_mb,
)
from brainsim.rules.torch_rules.allocation_sharded import allocate_global


def _rest_traces(rt) -> None:
    """Decay plasticity traces by the inter-item rest interval (config):
    consecutive items are distinct memories, not one behavioral episode."""
    ms = rt.config.inter_item_seconds * 1000.0
    f_et = math.exp(-ms / rt.btsp_params.tau_et)
    f_is = math.exp(-ms / rt.btsp_params.tau_is)
    for st in rt.edges["form_rec"].btsp.values():
        st.et_pre.mul_(f_et)
        st.is_post.mul_(f_is)
    f_kc = math.exp(-ms / rt.mb_params.tau_e)
    for st in rt.edges["relay_in"].mb.values():
        st.e_kc.mul_(f_kc)


def _btsp_write(rt, engram: np.ndarray) -> None:
    cfg = rt.config
    es = rt.edges["form_rec"]
    lo, hi = rt.net.local_range
    pre = bool_at(cfg.n_global, engram, rt.device)
    post = bool_at(rt.net.n_local,
                   engram[(engram >= lo) & (engram < hi)] - lo, rt.device)
    for p in es.parts:
        for _ in range(cfg.btsp_write_steps):
            t_btsp.step(es.btsp[p], rt.btsp_params, pre, post,
                        es.pre_idx[p], es.post_idx[p])


def _mb_write(rt, out_code: np.ndarray, salience: float) -> None:
    cfg = rt.config
    es = rt.edges["relay_in"]
    kc = bool_at(cfg.n_global, out_code, rt.device)
    dan = float(salience) * cfg.mb_gate_scale
    for p in es.parts:
        for _ in range(cfg.mb_write_steps):
            t_mb.step(es.mb[p], rt.mb_params, kc, dan, es.pre_idx[p])


def _front_window(rt, pattern: np.ndarray, exc_bonus: bool,
                  steps: int) -> tuple[np.ndarray, np.ndarray]:
    """The shared encode/retrieve FRONT of the loop, one natural forward
    window: drive the input stage with the pattern, let separation spike
    freely, and read separation and formation codes together. Formation
    input mixes the mossy detonators (selection variance, decorrelating)
    with the direct path (correlation carrier), under encoding-mode
    suppression of form_rec. Write and recall use the SAME functional, so
    a cue's seed correlates with the write-time code at the input-overlap
    level instead of its square (the mossy-only chain squares cue
    degradation; measured at smoke)."""
    cfg = rt.config
    in_lo, _ = rt.ranges["input"]
    in_ref = max(1, int(round(0.5 * cfg.n_input)))  # bipolar: ~half active
    drive = rt.drive_for(np.nonzero(pattern > 0)[0] + in_lo, cfg.drive_gain,
                         norm_ref=in_ref)
    if exc_bonus:
        a, b = rt.form_local
        if b > a:
            drive[a:b] += cfg.exc_drive_gain * rt.alloc_state.excitability
    with rt.encoding_mode(("form_rec",)):
        w = rt.run_window(drive, steps, ("separation", "formation"))
    return (rt.code_from_window("separation", w["separation"]),
            rt.code_from_window("formation", w["formation"]))


def _back_windows(rt, form_code: np.ndarray, steps: int,
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """formation -> output -> relay -> storage feedforward windows (the
    storage read is input-dominant under encoding-mode suppression)."""
    cfg = rt.config
    drive = rt.drive_for(form_code, cfg.clamp_gain,
                         norm_ref=cfg.code_size("formation"))
    out = rt.run_window(drive, steps, ("output",))["output"]
    out_code = rt.code_from_window("output", out)

    drive = rt.drive_for(out_code, cfg.clamp_gain,
                         norm_ref=cfg.code_size("output"))
    relay = rt.run_window(drive, steps, ("relay",))["relay"]
    relay_code = rt.code_from_window("relay", relay)

    drive = rt.drive_for(relay_code, cfg.clamp_gain,
                         norm_ref=cfg.code_size("relay"))
    with rt.encoding_mode(("store_rec",)):
        store = rt.run_window(drive, steps, ("storage",))["storage"]
    store_code = rt.code_from_window("storage", store)
    return out_code, relay_code, store_code


def write_phase(rt, items) -> list[int]:
    """items: sequence of (pattern[n_input], salience, protect). SPMD:
    identical on every rank. Returns item ids."""
    from brainsim.composite.full.runtime import ItemRecord

    cfg = rt.config
    flo = rt.ranges["formation"][0]
    ids: list[int] = []
    for pattern, salience, protect in items:
        item_id = rt._next_id
        rt._next_id += 1
        pattern = np.asarray(pattern, dtype=np.float64)
        _rest_traces(rt)

        sep_code, form_code = _front_window(rt, pattern, exc_bonus=True,
                                            steps=cfg.write_steps)
        a, b = rt.form_local

        t_allocation.decay(rt.alloc_state, cfg.allocation,
                           dt_seconds=cfg.alloc_interval_seconds)
        noise = np.random.default_rng([cfg.seed, 71, item_id]).normal(
            0.0, cfg.allocation.noise_std, cfg.n_formation)
        rel = (a - flo, b - flo)
        noise_local = torch.as_tensor(noise[rel[0]:rel[1]], dtype=torch.float32,
                                      device=rt.device)
        winners = allocate_global(rt.alloc_state, cfg.allocation, noise_local,
                                  rel, cfg.n_formation, k=cfg.alloc_k)
        engram = np.union1d(form_code, winners.cpu().numpy() + flo)
        _btsp_write(rt, engram)

        out_code, relay_code, store_code = _back_windows(
            rt, engram, cfg.write_steps)
        _mb_write(rt, out_code, salience)
        store_write(rt, store_code, cfg.eta_store, protect)

        rt.items[item_id] = ItemRecord(
            pattern=pattern.copy(),
            codes={"separation": sep_code, "formation": engram,
                   "output": out_code, "relay": relay_code,
                   "storage": store_code},
            salience=float(salience), protected=bool(protect),
        )
        rt.replay.register(item_id, float(salience))
        ids.append(item_id)
    return ids


def encode_codes(rt, pattern: np.ndarray) -> dict[str, np.ndarray]:
    """Clean ENCODING-MODE forward pass (write_phase's windows without any
    plasticity or allocation): every recurrent stage suppressed, codes are
    pure feedforward images. The composite analog of
    pipeline.encode_separation for reference codes: recall retrieves the
    nearest stored memory for ANY input (no novelty detector exists in the
    loop), so an unwritten pattern's reference must come from the encode
    pass, exactly as the numpy pipeline compares recall output against the
    cue's clean encoding."""
    cfg = rt.config
    pattern = np.asarray(pattern, dtype=np.float64)
    out: dict[str, np.ndarray] = {}
    out["separation"], out["formation"] = _front_window(
        rt, pattern, exc_bonus=False, steps=cfg.write_steps)
    out["output"], out["relay"], out["storage"] = _back_windows(
        rt, out["formation"], cfg.write_steps)
    return out


def recall_codes(rt, pattern: np.ndarray) -> dict[str, np.ndarray]:
    """Read-only retrieval pass: front window (shared functional with the
    write path), formation completion as the ITERATED synaptic image
    through the BTSP-bound recurrents (formation is the loop's designated
    error corrector), feedforward relay to the storage seed, storage
    completion as the iterated image through the cascade-protected
    weights. No plasticity, no allocation."""
    cfg = rt.config
    pattern = np.asarray(pattern, dtype=np.float64)
    sep_code, form_seed = _front_window(rt, pattern, exc_bonus=False,
                                        steps=cfg.recall_steps)
    form_code = recurrent_image_code(rt, "form_rec", "formation", form_seed)
    out_code, relay_code, seed_code = _back_windows(
        rt, form_code, cfg.recall_steps)
    store_code = storage_image_code(rt, seed_code)
    return {"separation": sep_code, "formation": form_code,
            "output": out_code, "relay": relay_code,
            "storage_seed": seed_code, "storage": store_code}


def recall_phase(rt, item_ids, drop_fraction: float = 0.5,
                 rng: np.random.Generator | None = None) -> list[dict]:
    """Cued recall of written items; overlaps against their stored codes.
    SPMD: pass a shared rng (or none for the seed-derived default)."""
    rng = rng or np.random.default_rng([rt.config.seed, 917])
    out = []
    for item_id in item_ids:
        rec = rt.items[item_id]
        cue = make_cue(rec.pattern, drop_fraction, rng)
        codes = recall_codes(rt, cue)
        out.append({
            "item_id": item_id,
            "store_overlap": overlap_sets(codes["storage"], rec.codes["storage"]),
            "form_overlap": overlap_sets(codes["formation"], rec.codes["formation"]),
        })
    return out


__all__ = ["write_phase", "encode_codes", "recall_codes", "recall_phase",
           "storage_phase", "store_write", "make_cue", "overlap_sets",
           "recurrent_image_code", "storage_image_code"]
