"""H1 composite arm: the frozen capacity protocol on the composite's
formation stage (contract arm 1b, battery-h1-wiring-premium.md FROZEN).

Arm definition (config arithmetic, checked by tests):
- full scale = CompositeConfig.full_scale() with relay+storage thinned to
  the H3 pass-through floors (batteries.h3.budget.thin_stages): the
  formation LOOP as configured in the composite -- input (EC), separation
  (DG), formation (CA3, BTSP+allocation recurrent), output (CA1) all at the
  design-doc census -- ~1.22e10 synapses, inside F21's loop-total envelope
  and above the frozen 1e10 results floor. Relay and storage carry no H1
  protocol role; their thin floors (~1.3e5 syn) keep the runtime machinery
  intact and are counted in the totals they negligibly perturb.
- out_in (formation->output, 9.3e9 syn) carries real spiking in every
  window (CA1 integrates the formation code) but no protocol readout; it is
  left at the measured realization in EVERY arm, so it cancels in all
  comparisons. n_synapses_active_path in the results names the four
  surgery-bearing edges separately (honesty rail).

Protocol (frozen semantics, engineering knobs recorded):
- items: random bipolar input patterns (the composite's native encoding
  boundary; the STORED sparse pattern is the formation engram at
  f_formation, per the contract's "stage's write rule" reading).
- write: the composite write path's front half exactly as write_phase runs
  it (rest traces -> front window with allocation excitability bonus ->
  allocation winners -> BTSP one-shot on the engram). Back windows (relay/
  storage writes) are not part of the formation stage's write rule and are
  not run.
- recall: front window on a 50%-degraded cue (make_cue deletion semantics:
  dropped coordinates are zeroed, nothing added -- the Formal Model's
  partial-pattern definition, same as the fly arm's calibrated cue), then
  the iterated recurrent image through form_rec. Overlap = overlap_sets
  (indicator cosine) vs the stored engram; C = max P with mean overlap >=
  0.9 by the SAME capacity_from_curve/checkpoint_grid machinery as the fly
  arm (grid_ratio 1.3, early stop 0.8 x2 -- fly parity).
- FFI-KO ablation: the composite's feedforward-inhibition abstraction IS
  the top-f readout (composite.full.config docstring); the knockout is the
  H3-established realization (batteries.h3.systems.H3Runtime topf_off) on
  the front-window readouts (separation, formation). Unlike the fly's
  gamma=0 KO the coding level cannot be held fixed -- the cap is the
  mechanism -- so the KO measures the loss of write-time sparseness
  enforcement (documented deviation in KO semantics, same realization as
  the H3 screen).

All randomness is keyed through h1.protocol.rng_for (H1_KEY namespace,
composite streams 40+): layout-independent, identical on every rank (SPMD).
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch

from brainsim.batteries.h3.budget import assert_matched, thin_stages, total_synapses
from brainsim.composite.full import phases
from brainsim.composite.full.build import forecast
from brainsim.composite.full.codes import make_cue, overlap_sets, recurrent_image_code
from brainsim.composite.full.config import CompositeConfig
from brainsim.batteries.h1.protocol import (CUE_DEGRADATION, OVERLAP_BAR,
                                            capacity_from_curve,
                                            checkpoint_grid, rng_for)
from brainsim.rules.torch_rules import allocation as t_allocation
from brainsim.rules.torch_rules.allocation_sharded import allocate_global

# rng streams in the H1_KEY namespace (fly arm uses 0-3, 10, 20, 30-33)
STREAM_C_ITEMS, STREAM_C_CUES, STREAM_C_EVAL = 40, 41, 42
COMP_SEED_TAG = 7  # SeedSequence tag for derived runtime seeds (h3 idiom)

# The four edges that carry the encode/recall computation; controls operate
# on exactly these (out_in disclosed above; thin back edges protocol-inert).
ACTIVE_PATH_EDGES = ("sep_in", "form_in", "form_direct", "form_rec")
LOOP_STAGES = ("input", "separation", "formation", "output")
FFI_KO_STAGES = ("separation", "formation")

INSTANCE_KINDS = ("measured", "measured_ffi_ko", "config", "weight_shuffle",
                  "erdos_renyi", "tf_attractor")


@dataclass(frozen=True)
class CompH1Config:
    """Engineering knobs of the composite arm (frozen thresholds live in
    h1.protocol / battery_common and are NOT repeated here)."""
    scale: str = "full"              # smoke | full
    engine: str = "torch"
    # sweep resolution (fly parity where stated)
    grid_ratio: float = 1.3          # fly parity
    p_cap: int = 24576               # fly parity; < 32768 (TF n11 int32 headroom)
    eval_sample: int = 96            # engineering: each eval is a full spiking
    #                                  window here (fly used 200 cheap matmuls)
    early_stop_bar: float = 0.8      # fly parity
    early_stop_consecutive: int = 2  # fly parity
    n_shuffle_draws: int = 5         # contract: 5 independent shuffles
    # thin pass-through floors for relay+storage (h3 full-scale values)
    thin_n: int = 2000
    thin_k: int = 32

    def config_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def instances(self) -> tuple[str, ...]:
        """Arm-instance names, frozen order (the canonical work list)."""
        draws = tuple(f"config:{d}" for d in range(self.n_shuffle_draws))
        if self.scale == "smoke":
            # driver-spec smoke scope: measured + KO + 2 controls, print-only
            return ("measured", "measured_ffi_ko") + draws[:1] + ("tf_attractor",)
        return (("measured", "measured_ffi_ko") + draws
                + ("weight_shuffle", "erdos_renyi", "tf_attractor"))


def full_comp_config(**kw) -> CompH1Config:
    return CompH1Config(scale="full", **kw)


def smoke_comp_config(**kw) -> CompH1Config:
    kw.setdefault("p_cap", 8)
    kw.setdefault("eval_sample", 6)
    kw.setdefault("n_shuffle_draws", 1)
    kw.setdefault("thin_n", 20)
    kw.setdefault("thin_k", 4)
    return CompH1Config(scale="smoke", **kw)


def tiny_comp_config(**kw) -> CompH1Config:
    """Test-suite scale; smaller than smoke, same code paths."""
    kw.setdefault("p_cap", 4)
    kw.setdefault("eval_sample", 4)
    kw.setdefault("n_shuffle_draws", 1)
    kw.setdefault("thin_n", 12)
    kw.setdefault("thin_k", 3)
    return CompH1Config(scale="tiny", **kw)


def derived_seed(seed: int) -> int:
    """Runtime/wiring seed for battery seed `seed` (h3.arms.derived_seed
    idiom, H1 namespace): layout-independent by construction."""
    from brainsim.batteries.h1.protocol import H1_KEY
    ss = np.random.SeedSequence([H1_KEY, COMP_SEED_TAG, int(seed)])
    return int(ss.generate_state(1)[0] % (2**31 - 1))


def base_cc(cfg: CompH1Config, seed: int) -> CompositeConfig:
    """The composite arm's CompositeConfig: formation loop as configured in
    the composite, relay+storage at pass-through floors."""
    if cfg.scale == "full":
        cc = CompositeConfig.full_scale(seed=derived_seed(seed))
    else:
        n = 1 if cfg.scale == "smoke" else 0  # tiny is a notch smaller
        cc = CompositeConfig(
            n_input=100 + 20 * n, n_separation=200 + 60 * n,
            n_formation=160 + 40 * n, n_output=100 + 40 * n,
            n_relay=40, n_storage=80,
            k_sep_in=20 + 4 * n, k_form_in=10 + 2 * n,
            k_form_direct=20 + 4 * n, k_form_rec=20 + 4 * n,
            k_out_in=20 + 4 * n, k_relay_in=8, k_store_in=8, k_store_rec=12,
            multi_contact_frac=0.01, multi_contact_tail=3,
            f_separation=0.1, f_formation=0.1, f_output=0.1, f_relay=0.1,
            f_storage=0.1, chunk_synapses=100_000, seed=derived_seed(seed))
    return thin_stages(cc, ("relay", "storage"), cfg.thin_n, cfg.thin_k)


# ---- protocol ----------------------------------------------------------------

def item_pattern(seed: int, i: int, n_input: int) -> np.ndarray:
    """Bipolar input pattern for stream item i (composite boundary model)."""
    return rng_for(seed, STREAM_C_ITEMS, i).choice(
        np.array([-1.0, 1.0]), size=n_input)


def item_cue(seed: int, i: int, n_input: int) -> np.ndarray:
    """50%-degraded cue: deterministic per item, deletion-only in drive terms
    (make_cue zeroes coordinates; only +1 cells are ever driven)."""
    return make_cue(item_pattern(seed, i, n_input), CUE_DEGRADATION,
                    rng_for(seed, STREAM_C_CUES, i))


def write_composite_item(rt, pattern: np.ndarray, item_id: int) -> np.ndarray:
    """The composite write path's FRONT half, line-for-line the same
    mechanism sequence as composite.full.phases.write_phase (rest traces,
    front window with excitability bonus, allocation decay + global top-k
    winners, BTSP one-shot on the engram). Returns the stored engram
    (sorted global ids). The allocation noise key [cfg.seed, 71, item_id]
    is write_phase's own convention, so paired arms share draws."""
    cfg = rt.config
    flo = rt.ranges["formation"][0]
    phases._rest_traces(rt)
    _, form_code = phases._front_window(rt, pattern, exc_bonus=True,
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
    phases._btsp_write(rt, engram)
    return engram


def recall_formation(rt, cue: np.ndarray) -> np.ndarray:
    """Read-only formation retrieval: front window on the cue (shared
    functional with the write path), then the iterated recurrent image
    through the BTSP-bound recurrents (the composite's designated error
    corrector). No plasticity, no allocation."""
    _, seed_code = phases._front_window(rt, cue, exc_bonus=False,
                                        steps=rt.config.recall_steps)
    return recurrent_image_code(rt, "form_rec", "formation", seed_code)


def run_graph_capacity(rt, seed: int, cfg: CompH1Config, log=print) -> dict:
    """Stream items through one composite-graph arm instance and measure the
    frozen precision curve (same checkpoint/interpolation machinery as the
    fly arm)."""
    t0 = time.time()
    grid = checkpoint_grid(cfg)
    n_input = rt.config.n_input
    engrams: dict[int, np.ndarray] = {}
    ps: list[int] = []
    overlaps: list[float] = []
    stored, below = 0, 0
    for p_next in grid:
        for i in range(stored, p_next):
            engrams[i] = write_composite_item(
                rt, item_pattern(seed, i, n_input), i)
        stored = p_next
        re = rng_for(seed, STREAM_C_EVAL, p_next)
        items = (np.arange(stored, dtype=np.int64)
                 if stored <= cfg.eval_sample
                 else np.sort(re.choice(stored, size=cfg.eval_sample,
                                        replace=False)).astype(np.int64))
        hit = np.zeros(items.size)
        for r, i in enumerate(items.tolist()):
            code = recall_formation(rt, item_cue(seed, i, n_input))
            hit[r] = overlap_sets(code, engrams[i])
        ps.append(stored)
        overlaps.append(float(hit.mean()))
        log(f"    P={stored}: overlap={overlaps[-1]:.4f} "
            f"({time.time() - t0:.0f}s)", flush=True)
        below = below + 1 if overlaps[-1] < cfg.early_stop_bar else 0
        if below >= cfg.early_stop_consecutive:
            break
    out = capacity_from_curve(ps, overlaps, bar=OVERLAP_BAR)
    out.update({"curve_P": ps, "curve_overlap": overlaps,
                "n_eval": int(min(stored, cfg.eval_sample)),
                "wall_s": round(time.time() - t0, 1)})
    return out


# ---- budget + memory forecast (config arithmetic; asserted, not assumed) -----

def loop_neurons(cc: CompositeConfig) -> int:
    sizes = cc.stage_sizes()
    return int(sum(sizes[s] for s in LOOP_STAGES))


def active_path_synapses(cc: CompositeConfig) -> int:
    from brainsim.composite.full.build import edge_specs, edge_synapse_count
    per = {e.name: edge_synapse_count(cc, e) for e in edge_specs(cc)}
    return int(sum(per[n] for n in ACTIVE_PATH_EDGES))


def budget_rows(cfg: CompH1Config, seed: int) -> dict:
    """Per-instance synapse totals vs the composite arm target. Graph
    surgeries preserve every edge count exactly (asserted again at runtime
    on the built network); the TF attractor matches the total directed-entry
    count. All rows must sit within the 1% envelope."""
    cc = base_cc(cfg, seed)
    target = total_synapses(cc)
    rows = {}
    for inst in cfg.instances():
        if inst == "tf_attractor":
            total = 2 * (target // 2)  # directed entries, rounded to pairs
        else:
            total = target  # count-preserving surgeries / measured build
        assert_matched(inst, total, target)
        rows[inst] = {"total": int(total), "target": int(target),
                      "ratio": total / target}
    return rows


def forecast_table(cfg: CompH1Config, seed: int) -> dict:
    """Memory forecast per execution class (build.forecast for the graph
    arms; explicit byte model for the TF attractor: col int32 + n11 int16
    + fp16 value buffer + indptr; composite_tf's probed fallbacks raise
    either the n11 or the value term to 4 B, each still on-device, and the
    double-fallback case would exceed one H200 and fail loudly)."""
    cc = base_cc(cfg, seed)
    target = total_synapses(cc)
    n_units = loop_neurons(cc)
    entries = 2 * (target // 2)
    tf_bytes = entries * (4 + 2 + 2) + 8 * (n_units + 1) + 8 * n_units
    return {
        "graph_arm_gb_per_rank_ws8": round(forecast(cc, 8)["per_rank_total"] / 1e9, 2),
        "graph_arm_gb_single_rank": round(forecast(cc, 1)["per_rank_total"] / 1e9, 2),
        "tf_gb_single_device": round(tf_bytes / 1e9, 2),
        "tf_gb_single_device_fallback": round((tf_bytes + 2 * entries) / 1e9, 2),
        "n_synapses_total": int(target),
        "n_synapses_active_path": active_path_synapses(cc),
        "n_neurons_global": cc.n_global,
        "n_units_tf": n_units,
        "note": ("graph arms run 8-rank sharded (one instance at a time, "
                 "sequenced); the TF attractor is single-device (packed). "
                 "The single-rank graph column shows packing graph arms is "
                 "an available upside when it fits the device."),
    }


__all__ = ["ACTIVE_PATH_EDGES", "CompH1Config", "FFI_KO_STAGES",
           "INSTANCE_KINDS", "LOOP_STAGES", "STREAM_C_CUES", "STREAM_C_EVAL",
           "STREAM_C_ITEMS", "active_path_synapses", "base_cc", "budget_rows",
           "derived_seed", "forecast_table", "full_comp_config", "item_cue",
           "item_pattern", "loop_neurons", "recall_formation",
           "run_graph_capacity", "smoke_comp_config", "tiny_comp_config",
           "write_composite_item"]
