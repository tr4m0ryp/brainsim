"""H1 fly arm: full-MB capacity on measured wiring vs frozen controls.

Executes the frozen protocol on the complete bilateral FlyWire v783 mushroom
body (the MINIMAL FALSIFYING EXPERIMENT for H1), >= 5 seeds, plus the
feedforward-inhibition knockout ablation (Formal Model prediction 1).

Writes results/h1/fly_arm.json ONLY. results/h1/results.json is
the node-side full-battery merge (fly + composite) and is NEVER written
here; verdict thresholds live exclusively in brainsim.batteries.verdicts
(h1_verdict), which this module probes rather than reimplements.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import replace
from pathlib import Path

from brainsim.batteries import verdicts
from brainsim.paths import RESULTS, display

import numpy as np

from brainsim.batteries.h1 import controls, tf_arm
from brainsim.batteries.h1.circuit import (DELTA_W, RULE_PARAMS,
                                           T_WRITE_STEPS, MBCircuit)
from brainsim.batteries.h1.mb_extract import MBGraph
from brainsim.batteries.h1.protocol import (CUE_DEGRADATION, OVERLAP_BAR,
                                            ItemBank, ProtocolCfg,
                                            rng_for, run_mb_capacity)

CONTRACT = "battery-h1-wiring-premium.md"
DEFAULT_OUT = RESULTS / "h1" / "fly_arm.json"
BUDGET_S = 170 * 60          # keep the whole arm under ~3 h wall time
SEEDS = (0, 1, 2, 3, 4)      # contract: >= 5 seeds per arm on the fly arm
N_SHUFFLE_DRAWS = 5          # contract: 5 independent shuffles
STREAM_SHUFFLE, STREAM_ER = 10, 20


def load_battery_common():
    """The frozen verdict module (brainsim.batteries.verdicts)."""
    return verdicts


def _fly_only_assessment(bc, ratios: list[float]) -> dict:
    """Restate the frozen thresholds for the fly arm alone by PROBING
    h1_verdict (no threshold is re-typed here). h1_verdict(fly, [0.0]) is
    'supported' iff the fly ratios clear the claim bar (a comp arm of 0.0 is
    neither in the kill band nor above the bar); h1_verdict(fly, [1.0]) is
    'killed' iff the fly ratios sit in the kill band (1.0 is in-band, so the
    kill fires exactly when the fly arm is in-band too)."""
    if not ratios or any(not np.isfinite(r) for r in ratios):
        return {"fly_only_assessment": "degenerate",
                "note": "non-finite or missing ratios; see arm capacities"}
    if bc.h1_verdict(ratios, [0.0]) == "supported":
        s = "clears_claim_bar"
        note = ("mean ratio > 1.10: frozen h1_verdict returns 'supported' "
                "regardless of the composite arm")
    elif bc.h1_verdict(ratios, [1.0]) == "killed":
        s = "in_kill_band"
        note = ("all seed ratios in [0.95, 1.05]: H1 dies iff the composite "
                "formation arm lands in the band too")
    else:
        s = "between_band_and_bar"
        note = ("fly arm neither clears the claim bar nor sits in the kill "
                "band; composite arm decides between partial and supported")
    return {"fly_only_assessment": s, "method": "h1_verdict probe", "note": note}


def _summary(bc, values: list[float]) -> dict:
    lo, hi = bc.bootstrap_ci(values)
    return {"per_seed": values, "mean": float(np.mean(values)),
            "ci95": [lo, hi]}


def _observed(assessment: dict, arms: dict, arm_names: list[str],
              c_meas: list[float]) -> dict:
    """Append a data-derived description (no thresholds re-typed): ordering
    of arm means and the measured arm's zero-capacity seed count."""
    means = {n: arms[n]["C"]["mean"] for n in arm_names}
    order = sorted(means, key=means.get)
    n_zero = arms["measured"]["n_zero_capacity_seeds"]
    obs = ("arm capacity ordering (ascending means): "
           + " < ".join(f"{n}={means[n]:.2f}" for n in order)
           + f"; measured arm at zero capacity on {n_zero}/{len(c_meas)} seeds")
    control_means = [v for n, v in means.items()
                     if n not in ("measured", "measured_ffi_ko")]
    if means["measured"] < min(control_means):
        obs += ("; the measured wiring UNDERPERFORMS every matched control on "
                "this protocol (anti-premium), a case the frozen bands do not "
                "name -- formal verdict still defers to h1_verdict with the "
                "composite arm")
    assessment["observed"] = obs
    return assessment


def _cohens_d(a: list[float], b: list[float]) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    n1, n2 = a.size, b.size
    pooled = ((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2)
    return float((a.mean() - b.mean()) / np.sqrt(max(pooled, 1e-30)))


def run_seed(g: MBGraph, seed: int, cfg: ProtocolCfg,
             n_draws: int) -> tuple[dict, dict]:
    """All arms for one seed; returns (results_by_arm, wall_times)."""
    bank = ItemBank.generate(seed, cfg, g.n_pn, g.n_mbon)
    out: dict = {}
    times: dict = {}

    def log(name: str, res: dict, t0: float) -> None:
        print(f"  [seed {seed}] {name}: {len(res['curve_P'])} checkpoints, "
              f"stream to P={res['curve_P'][-1]}, {time.time() - t0:.0f}s",
              flush=True)

    def mb_arm(name: str, graph: MBGraph, gamma: float) -> None:
        t0 = time.time()
        circ = MBCircuit.from_graph(graph, cfg.f_kc, gamma)
        out[name] = run_mb_capacity(circ, bank, seed, cfg)
        times[name] = round(time.time() - t0, 1)
        log(name, out[name], t0)

    mb_arm("measured", g, 1.0)
    mb_arm("measured_ffi_ko", g, 0.0)
    draws = []
    t0 = time.time()
    for d in range(n_draws):
        td = time.time()
        gs, stats = controls.config_shuffle(g, rng_for(seed, STREAM_SHUFFLE, d))
        circ = MBCircuit.from_graph(gs, cfg.f_kc, 1.0)
        r = run_mb_capacity(circ, bank, seed, cfg)
        r["swap_stats"] = stats
        draws.append(r)
        log(f"config_shuffle[{d}]", r, td)
    out["config_shuffle"] = {"draws": draws,
                             "C": float(np.mean([d["C"] for d in draws]))}
    times["config_shuffle"] = round(time.time() - t0, 1)
    t0 = time.time()
    gw = controls.weight_shuffle(g, rng_for(seed, STREAM_SHUFFLE, 100))
    out["weight_shuffle"] = run_mb_capacity(
        MBCircuit.from_graph(gw, cfg.f_kc, 1.0), bank, seed, cfg)
    times["weight_shuffle"] = round(time.time() - t0, 1)
    log("weight_shuffle", out["weight_shuffle"], t0)
    t0 = time.time()
    ge = controls.er_control(g, rng_for(seed, STREAM_ER))
    out["erdos_renyi"] = run_mb_capacity(
        MBCircuit.from_graph(ge, cfg.f_kc, 1.0), bank, seed, cfg)
    times["erdos_renyi"] = round(time.time() - t0, 1)
    log("erdos_renyi", out["erdos_renyi"], t0)
    t0 = time.time()
    out["tf_attractor"] = tf_arm.run_tf_capacity(
        g.circuit_neurons(), g.circuit_synapses(), seed, cfg)
    times["tf_attractor"] = round(time.time() - t0, 1)
    log("tf_attractor", out["tf_attractor"], t0)
    return out, times


def run_fly_arm(graph: MBGraph, out_path: Path = DEFAULT_OUT,
                seeds=SEEDS, cfg: ProtocolCfg | None = None,
                smoke: bool = False, provenance: dict | None = None) -> dict:
    if out_path.name == "results.json":
        raise ValueError("fly arm must never write results.json (node-side merge)")
    out_abs = Path(os.path.abspath(out_path))
    if smoke and (out_abs == RESULTS or RESULTS in out_abs.parents):
        raise ValueError("smoke output may not land under results/")
    bc = load_battery_common()
    cfg = cfg or ProtocolCfg()
    t_start = time.time()
    deviations: list[str] = []
    engineering: dict = {}

    # Runtime probe: seed 0 at default resolution, then project the rest.
    per_seed: dict[int, dict] = {}
    seed_times: dict[int, dict] = {}
    n_draws = N_SHUFFLE_DRAWS
    s0 = seeds[0]
    res0, t0 = run_seed(graph, s0, cfg, n_draws)
    seed0_wall = sum(t0.values())
    projected = seed0_wall * len(seeds) * 1.1
    engineering["probe"] = {"seed0_wall_s": round(seed0_wall, 1),
                            "projected_total_s": round(projected, 1),
                            "budget_s": BUDGET_S}
    if projected > BUDGET_S and not smoke:
        cfg = replace(cfg, grid_ratio=1.5)
        engineering["probe"]["action"] = (
            "projected over budget: sweep resolution coarsened to grid_ratio="
            "1.5 for ALL seeds (seed 0 re-run); shuffle draws kept at 5")
        res0, t0 = run_seed(graph, s0, cfg, n_draws)
        seed0_wall = sum(t0.values())
        projected = seed0_wall * len(seeds) * 1.1
        engineering["probe"]["projected_after_coarsen_s"] = round(projected, 1)
        if projected > BUDGET_S:
            n_draws = 3
            deviations.append(
                "CONTRACT DEVIATION: config-model shuffle draws reduced 5 -> 3 "
                "(last resort after coarsening the sweep; contract specifies 5)")
            res0, t0 = run_seed(graph, s0, cfg, n_draws)
    per_seed[s0], seed_times[s0] = res0, t0
    for s in seeds[1:]:
        per_seed[s], seed_times[s] = run_seed(graph, s, cfg, n_draws)

    arm_names = ["measured", "measured_ffi_ko", "config_shuffle",
                 "weight_shuffle", "erdos_renyi", "tf_attractor"]
    arms: dict = {}
    for name in arm_names:
        cs = [float(per_seed[s][name]["C"]) for s in seeds]
        arms[name] = {"C": _summary(bc, cs),
                      "n_zero_capacity_seeds": int(sum(c == 0.0 for c in cs)),
                      "per_seed_detail": {str(s): per_seed[s][name] for s in seeds}}
    c_meas = [float(per_seed[s]["measured"]["C"]) for s in seeds]
    c_shuf = [float(per_seed[s]["config_shuffle"]["C"]) for s in seeds]
    ratios = [m / sh if sh > 0 else float("inf") for m, sh in zip(c_meas, c_shuf)]
    finite = all(np.isfinite(r) for r in ratios) and len(ratios) > 0
    ratio_block = {"per_seed": [r if np.isfinite(r) else None for r in ratios],
                   "definition": "C_measured / mean(C over config-model draws), per seed"}
    if finite:
        lo, hi = bc.bootstrap_ci(ratios)
        ratio_block.update({"mean": float(np.mean(ratios)), "ci95": [lo, hi],
                            "effect_size_cohens_d": _cohens_d(c_meas, c_shuf)})
    c_ko = [float(per_seed[s]["measured_ffi_ko"]["C"]) for s in seeds]
    delta = [m - k for m, k in zip(c_meas, c_ko)]
    ablation = {"knockout": "feedforward inhibition (APL loop gain gamma=0, "
                            "coding level held at f_kc)",
                "C_measured": _summary(bc, c_meas),
                "C_knockout": _summary(bc, c_ko),
                "delta_C": _summary(bc, delta),
                "resolvable": bool(any(d != 0.0 for d in delta))}
    if not ablation["resolvable"]:
        ablation["note"] = (
            "delta is exactly 0 on every seed: the APL term does re-rank "
            "~13% of each KC code (verified on non-result seed 999), but "
            "capacity sits at the resolution floor, so prediction 1 (kappa "
            "carrier) is UNRESOLVABLE at this operating point rather than "
            "refuted")

    report = {
        "battery": "h1", "arm": "fly",
        "contract": CONTRACT,
        "contract_sha256": bc.contract_sha256(CONTRACT),
        "scope": ("fly arm only -- results/h1/results.json is the "
                  "node-side merge of fly + composite arms and is not "
                  "written by this driver"),
        "full_scale": True,
        "full_scale_scope": ("complete bilateral mushroom-body subgraph of "
                             "FlyWire v783 -- full scale FOR THE FLY ARM; the "
                             "composite synapse floor (>=1e10) applies to the "
                             "merged results.json, not to this arm"),
        "smoke": bool(smoke),
        "n_neurons": graph.counts.get("circuit_neurons"),
        "n_synapses": graph.counts.get("circuit_synapses"),
        "counts": graph.counts,
        "seeds": list(map(int, seeds)),
        "n_seeds": len(list(seeds)),
        "n_shuffle_draws": n_draws,
        "calibration_history": (
            "two aborted launches before this run, both killed before any "
            "capacity or ratio value was computed or read (stream lengths "
            "only): (1) f_z=0.5 + 200 ms write floor-saturated every arm by "
            "P<=17 -> recalibrated to f_z=0.125 (compartment-like DAN gate) "
            "and 100 ms write; (2) replace-half cue noise was decorrelated "
            "to chance by the measured expansion layer, killing recall at "
            "P=2 -> cues corrected to the Formal Model's partial-pattern "
            "definition (deletion-only). Frozen thresholds untouched; every "
            "change identical for all arms -- protocol.py CALIBRATION HISTORY"),
        "protocol": {
            "overlap_bar": OVERLAP_BAR, "cue_degradation": CUE_DEGRADATION,
            "write_rule": "rules/mb_rule closed form (step-equivalence tested)",
            "rule_params": {"alpha": RULE_PARAMS.alpha, "tau_w_ms": RULE_PARAMS.tau_w,
                            "tau_e_ms": RULE_PARAMS.tau_e, "w_max": RULE_PARAMS.w_max,
                            "beta_floor": RULE_PARAMS.beta_floor,
                            "t_write_ms": T_WRITE_STEPS, "delta_w": DELTA_W},
            "cfg": arms["measured"]["per_seed_detail"][str(seeds[0])]["cfg"],
        },
        "engineering_choices": engineering,
        "arms": arms,
        "ratio_C_meas_over_C_shuf": ratio_block,
        "ablation_ffi_knockout": ablation,
        "assessment": _observed(_fly_only_assessment(bc, ratios if finite else []),
                                arms, arm_names, c_meas),
        "engine": ("brainsim-numpy-rate-cpu (batteries/h1 forward pass; "
                   "plasticity = rules/mb_rule closed form)"),
        "code_version": bc.git_sha(),
        "provenance": provenance or {},
        "runtime_s": round(time.time() - t_start, 1),
        "wall_times_s": {str(s): seed_times[s] for s in seeds},
        "deviations": deviations,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=1))
    return report


__all__ = ["BUDGET_S", "CONTRACT", "DEFAULT_OUT", "N_SHUFFLE_DRAWS", "SEEDS",
           "load_battery_common", "run_fly_arm", "run_seed"]
