"""Partial outputs, merge, and final results assembly.

Verdict logic is NEVER implemented here: shifts and noise bands are computed
from arm results and handed to the FROZEN verdicts.h2_verdict; the CI
comes from the frozen verdicts.bootstrap_ci; the final file is written
by the frozen verdicts.write_results. Only the merge path may write
results/h2/results.json, and only for scale == "full".

Shift metric (pre-specified, config.PROVENANCE_NOTES): per seed and (k, p),
shift = ln((C_red * T_red) / (C_uni * T_uni)) with T = promoted-item T_half
(censored values enter as lower bounds -> conservative) and floors C >= 1,
T >= 1e-3 guarding degenerate cells (flagged). noise band = half-width of
the bootstrap 95% CI over per-seed shifts.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict
from pathlib import Path

from brainsim.batteries import verdicts
from brainsim.paths import RESULTS

import numpy as np

from .config import PROVENANCE_NOTES, H2Config
from .arms import build_work_list, work_key
from .measures import pareto_front

CONTRACT_NAME = "battery-h2-permanence-frontier.md"
FINAL_RESULTS = RESULTS / "h2" / "results.json"
MIN_FULL_SYNAPSES = 1_000_000_000  # frozen validate.py floor


def battery_common():
    """The frozen verdict module (brainsim.batteries.verdicts)."""
    return verdicts


def kp_key(k: int, p: float) -> str:
    return f"k={int(k)},p={p:g}"


# -- partials ----------------------------------------------------------------------


def partial_path(out_dir: Path, shard_i: int, shard_n: int) -> Path:
    return Path(out_dir) / f"partial_{shard_i}_of_{shard_n}.json"


def make_partial(cfg: H2Config, seeds: list[int], shard_i: int, shard_n: int,
                 results: dict[str, dict], gates: dict, wall_s: float) -> dict:
    bc = battery_common()
    return {
        "battery": "h2",
        "shard": [int(shard_i), int(shard_n)],
        "scale": cfg.scale,
        "engine": cfg.engine,
        "config": asdict(cfg),
        "config_hash": cfg.config_hash(),
        "contract_sha256": bc.contract_sha256(CONTRACT_NAME),
        "code_version": bc.git_sha(),
        "seeds": [int(s) for s in seeds],
        "twin_gate": gates,  # per-seed gate for every twin THIS shard built
        "results": results,
        "wall_s": wall_s,
    }


def write_partial(partial: dict, out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = partial_path(out_dir, *partial["shard"])
    dest.write_text(json.dumps(partial, indent=1))
    return dest


# -- merge --------------------------------------------------------------------------


class MergeError(RuntimeError):
    pass


def _verify_partials(partials: list[dict]) -> None:
    if not partials:
        raise MergeError("no partials to merge")
    n = partials[0]["shard"][1]
    idxs = sorted(p["shard"][0] for p in partials)
    if idxs != list(range(n)):
        raise MergeError(f"incomplete shard set: have {idxs}, need 0..{n - 1}")
    for field in ("config_hash", "contract_sha256", "scale", "engine", "seeds"):
        vals = {json.dumps(p[field], sort_keys=True) for p in partials}
        if len(vals) != 1:
            raise MergeError(f"partials disagree on {field}: {vals}")
    bc = battery_common()
    if partials[0]["contract_sha256"] != bc.contract_sha256(CONTRACT_NAME):
        raise MergeError("partials attest a different frozen contract than the "
                         "one on disk; refusing to merge")


def _collect(partials: list[dict]) -> tuple[H2Config, list[int], dict, dict]:
    cfg_d = dict(partials[0]["config"])
    for key in ("k_grid", "p_grid"):
        cfg_d[key] = tuple(cfg_d[key])
    cfg = H2Config(**cfg_d)
    seeds = list(partials[0]["seeds"])
    results: dict[str, dict] = {}
    gates: dict = {}
    for p in partials:
        for key, val in p["results"].items():
            if key in results:
                raise MergeError(f"duplicate work item across partials: {key}")
            results[key] = val
        for seed, gate in (p.get("twin_gate") or {}).items():
            if seed in gates and gates[seed]["checks"] != gate["checks"]:
                raise MergeError(f"twin gate mismatch across shards for seed {seed}")
            gates[seed] = gate
    expected = {work_key(*w) for w in build_work_list(cfg, seeds)}
    missing = expected - set(results)
    if missing:
        raise MergeError(f"{len(missing)} work items missing, e.g. {sorted(missing)[:4]}")
    for seed in seeds:
        if str(seed) not in gates:
            raise MergeError(f"no twin gate recorded for seed {seed}")
        if not gates[str(seed)]["passed"]:
            raise MergeError(f"twin gate FAILED for seed {seed}; contract forbids arms")
    return cfg, seeds, results, gates


def _get(results: dict, arm: str, k: int, p: float, seed: int) -> dict:
    return results[work_key(arm, k, p, seed)]


def analyze(cfg: H2Config, seeds: list[int], results: dict) -> dict:
    bc = battery_common()
    shift_by_kp: dict[str, float] = {}
    band_by_kp: dict[str, float] = {}
    per_seed_shifts: dict[str, list[float]] = {}
    flags: list[str] = []
    for p in cfg.p_grid:
        for k in cfg.k_grid:
            shifts = []
            for seed in seeds:
                red = _get(results, "redundancy", k, p, seed)
                uni = _get(results, "uniform", k, p, seed)
                c_r, c_u = max(red["C"], 1.0), max(uni["C"], 1.0)
                t_r = max(red["t_half_promoted"], 1e-3)
                t_u = max(uni["t_half_promoted"], 1e-3)
                if red["C"] < 1.0 or uni["C"] < 1.0:
                    flags.append(f"degenerate C floored at {kp_key(k, p)} seed {seed}")
                if red["t_half_censored"] or uni["t_half_censored"]:
                    flags.append(f"censored T_half at {kp_key(k, p)} seed {seed}"
                                 " (shift uses lower bound)")
                shifts.append(math.log((c_r * t_r) / (c_u * t_u)))
            key = kp_key(k, p)
            per_seed_shifts[key] = shifts
            shift_by_kp[key] = float(np.mean(shifts))
            lo, hi = bc.bootstrap_ci(shifts)
            band_by_kp[key] = float((hi - lo) / 2.0)
    verdict = bc.h2_verdict(shift_by_kp, band_by_kp)

    points = []
    for p in cfg.p_grid:
        cs = [_get(results, "cascade_only", 0, p, s) for s in seeds]
        points.append({"label": f"cascade_only p={p:g}", "arm": "cascade_only",
                       "k": 0, "p": p,
                       "C": float(np.mean([r["C"] for r in cs])),
                       "T_half": float(np.mean([r["t_half_promoted"] for r in cs]))})
        for k in cfg.k_grid:
            for arm in ("redundancy", "uniform", "replay", "combined"):
                rs = [_get(results, arm, k, p, s) for s in seeds]
                points.append({
                    "label": f"{arm} {kp_key(k, p)}", "arm": arm, "k": k, "p": p,
                    "C": float(np.mean([r["C"] for r in rs])),
                    "T_half": float(np.mean([r["t_half_promoted"] for r in rs])),
                    "censored_any": any(r["t_half_censored"] for r in rs),
                })
    predicted = {
        "snr_gain_sqrt_k": {str(k): math.sqrt(k) for k in cfg.k_grid},
        "crossover_p_star": {str(k): 1.0 / k for k in cfg.k_grid},
        "sigma_per_step": cfg.sigma,
        "note": ("Formal Model: k independent-turnover contacts give sqrt(k) "
                 "protected-item SNR; capacity loss ~ p*k; crossover p* ~ 1/k. "
                 "Plot predicted vs measured from these fields + points."),
    }
    return {
        "shift_by_kp": shift_by_kp,
        "noise_band_by_kp": band_by_kp,
        "per_seed_shifts": per_seed_shifts,
        "verdict": verdict,
        "frontier_points": points,
        "pareto_front": pareto_front([p for p in points if not p.get("censored_any")]),
        "predicted": predicted,
        "analysis_flags": sorted(set(flags)),
    }


def merge_partials(partials: list[dict], write_final: bool) -> dict:
    _verify_partials(partials)
    cfg, seeds, results, gates = _collect(partials)
    analysis = analyze(cfg, seeds, results)
    n_syn = min(int(g["n_contacts"]) for g in gates.values())
    merged = {
        "scale": cfg.scale, "engine": cfg.engine, "config": asdict(cfg),
        "config_hash": cfg.config_hash(), "seeds": seeds,
        "n_neurons": cfg.n, "n_synapses": n_syn,
        "twin_gate": gates, "results": results, **analysis,
        "notes": list(PROVENANCE_NOTES),
    }
    if write_final:
        if cfg.scale != "full":
            raise MergeError("refusing to write results.json from a non-full "
                             f"scale run (scale={cfg.scale}); smoke is print-only")
        if n_syn < MIN_FULL_SYNAPSES:
            raise MergeError(
                f"full-scale twin has {n_syn:.3e} substrate synapses, below the "
                f"frozen {MIN_FULL_SYNAPSES:.0e} floor; refusing to attest full_scale")
        bc = battery_common()
        dest = bc.write_results(
            "h2", CONTRACT_NAME, n_neurons=cfg.n, n_synapses=n_syn, seeds=seeds,
            verdict=analysis["verdict"], arms=results, engine=cfg.engine,
            extra={
                "twin_gate": gates, "shift_by_kp": analysis["shift_by_kp"],
                "noise_band_by_kp": analysis["noise_band_by_kp"],
                "per_seed_shifts": analysis["per_seed_shifts"],
                "frontier_points": analysis["frontier_points"],
                "pareto_front": analysis["pareto_front"],
                "predicted": analysis["predicted"],
                "analysis_flags": analysis["analysis_flags"],
                "config": asdict(cfg), "config_hash": cfg.config_hash(),
                "notes": list(PROVENANCE_NOTES),
                "n_synapses_definition": (
                    "per-arm twin substrate contact count (min over seeds); "
                    "arms run as concurrent single-GPU processes, each at this scale"),
            })
        merged["results_path"] = str(dest)
    return merged


__all__ = ["battery_common", "make_partial", "write_partial", "partial_path",
           "merge_partials", "analyze", "MergeError", "kp_key",
           "CONTRACT_NAME", "FINAL_RESULTS", "MIN_FULL_SYNAPSES"]
