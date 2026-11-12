"""Work list, partials, analysis and the node-side H1 merge.

Verdict logic is NEVER implemented here: the per-seed ratios of BOTH arms
go to the FROZEN verdicts.h1_verdict; CIs come from bootstrap_ci; the
final file is written by verdicts.write_results. Only merge_h1 may
write results/h1/results.json, and only when (a) every composite
partial is a full-scale run covering the canonical work list exactly once,
(b) the fly report attests the same frozen contract at full scale with its
contractual >= 5 seeds, (c) the composite arm meets the frozen >= 1e10
synapse floor with >= 3 seeds, and (d) every per-seed ratio on both arms is
finite (a degenerate ratio aborts the merge loudly instead of flowing into
a verdict).

Sharding: partials slice the (arm-instance x seed) work list round-robin
(--arm-shard I/N, zero-based) and/or filter by execution class ('sharded'
composite-graph instances vs the 'packed' single-device TF instances); the
merge verifies the union covers the canonical list exactly once (H3
report idiom).
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from brainsim.batteries import verdicts
from brainsim.batteries.h1.composite_analysis import (RATIO_DEF, MergeError,
                                                      analyze, work_key)
from brainsim.batteries.h1.composite_arm import (CompH1Config, base_cc,
                                                 budget_rows, forecast_table,
                                                 loop_neurons)
from brainsim.batteries.h3.budget import total_synapses
from brainsim.paths import RESULTS, display

CONTRACT = "battery-h1-wiring-premium.md"
H1_DIR = RESULTS / "h1"
FLY_ARM_JSON = H1_DIR / "fly_arm.json"
FINAL_RESULTS = H1_DIR / "results.json"
MIN_FULL_SYNAPSES = 10_000_000_000  # frozen validate.py h1 floor


def battery_common():
    """The frozen verdict module (brainsim.batteries.verdicts)."""
    return verdicts


# ---- canonical work list -----------------------------------------------------

def build_work_list(cfg: CompH1Config, seeds: list[int]) -> list[dict]:
    """Frozen enumeration; shard slices are round-robin over this list.
    TF instances are single-device 'packed' items; everything else runs at
    composite scale ('sharded', 8-rank at full scale)."""
    work = []
    for seed in seeds:
        for name in cfg.instances():
            work.append({
                "name": name, "seed": int(seed),
                "exec_class": "packed" if name == "tf_attractor" else "sharded",
            })
    return work


def select_items(cfg: CompH1Config, seeds: list[int], shard_i: int,
                 shard_n: int, items_filter: str) -> list[dict]:
    work = [it for it in build_work_list(cfg, seeds)
            if items_filter == "all" or it["exec_class"] == items_filter]
    return [it for j, it in enumerate(work) if j % shard_n == shard_i]


# ---- partials ------------------------------------------------------------------

def partial_path(out_dir: Path, items_filter: str, shard_i: int,
                 shard_n: int) -> Path:
    stem = (f"composite_partial_{shard_i}_of_{shard_n}.json"
            if items_filter == "all" else
            f"composite_partial_{items_filter}_{shard_i}_of_{shard_n}.json")
    return Path(out_dir) / stem


def make_partial(cfg: CompH1Config, seeds: list[int], shard: tuple[int, int],
                 items_filter: str, results: dict, wall_s: float) -> dict:
    bc = battery_common()
    return {
        "battery": "h1", "arm": "composite",
        "shard": [int(shard[0]), int(shard[1])],
        "items_filter": items_filter, "scale": cfg.scale,
        "engine": cfg.engine, "config": asdict(cfg),
        "config_hash": cfg.config_hash(),
        "contract_sha256": bc.contract_sha256(CONTRACT),
        "code_version": bc.git_sha(), "seeds": [int(s) for s in seeds],
        "budget_rows": budget_rows(cfg, seeds[0]),
        "forecast": forecast_table(cfg, seeds[0]),
        "results": results, "wall_s": round(wall_s, 1),
    }


def write_partial(partial: dict, out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = partial_path(out_dir, partial["items_filter"], *partial["shard"])
    dest.write_text(json.dumps(partial, indent=1))
    return dest


def _collect(partials: list[dict]) -> tuple[CompH1Config, list[int], dict]:
    if not partials:
        raise MergeError("no composite partials to merge")
    for field in ("config_hash", "contract_sha256", "scale", "engine", "seeds"):
        vals = {json.dumps(p[field], sort_keys=True) for p in partials}
        if len(vals) != 1:
            raise MergeError(f"partials disagree on {field}: {vals}")
    bc = battery_common()
    if partials[0]["contract_sha256"] != bc.contract_sha256(CONTRACT):
        raise MergeError("partials attest a different frozen contract than "
                         "the one on disk; refusing to merge")
    cfg = CompH1Config(**partials[0]["config"])
    seeds = list(partials[0]["seeds"])
    results: dict = {}
    for p in partials:
        for key, val in p["results"].items():
            if key in results:
                raise MergeError(f"duplicate work item across partials: {key}")
            results[key] = val
    expected = {work_key(it) for it in build_work_list(cfg, seeds)}
    missing, extra = expected - set(results), set(results) - expected
    if missing or extra:
        raise MergeError(f"{len(missing)} missing / {len(extra)} unexpected "
                         f"work items, e.g. {sorted(missing | extra)[:4]}")
    return cfg, seeds, results


# ---- fly arm loading ------------------------------------------------------------

def load_fly_report(path: Path = FLY_ARM_JSON) -> dict:
    if not Path(path).exists():
        raise MergeError(f"fly arm report missing: {path}")
    rep = json.loads(Path(path).read_text())
    bc = battery_common()
    if rep.get("contract_sha256") != bc.contract_sha256(CONTRACT):
        raise MergeError("fly_arm.json attests a different frozen contract "
                         "than the one on disk")
    if not rep.get("full_scale") or rep.get("smoke"):
        raise MergeError("fly_arm.json is not a full-scale, non-smoke run")
    if rep.get("n_seeds", 0) < 5:
        raise MergeError("fly arm has fewer than the contractual 5 seeds")
    ratios = rep.get("ratio_C_meas_over_C_shuf", {}).get("per_seed")
    if not ratios or any(r is None or not np.isfinite(r) for r in ratios):
        raise MergeError("fly arm ratios are missing or non-finite")
    return rep


# ---- the merge -------------------------------------------------------------------

def merge_h1(partial_paths: list[str], fly_path: Path = FLY_ARM_JSON,
             write_final: bool = True, log=print) -> dict:
    partials = [json.loads(Path(p).read_text()) for p in sorted(partial_paths)]
    cfg, seeds, results = _collect(partials)
    if cfg.scale != "full":
        raise MergeError("refusing to merge non-full-scale composite "
                         f"partials (scale={cfg.scale}); smoke is print-only")
    fly = load_fly_report(fly_path)
    stats = analyze(cfg, seeds, results, fly)
    cc = base_cc(cfg, seeds[0])
    n_syn = int(total_synapses(cc))
    for p in partials:  # partial-recorded budget must match the recompute
        if p["budget_rows"]["measured"]["target"] != n_syn:
            raise MergeError("partial budget target disagrees with the "
                             "config recompute; refusing to merge")
    if n_syn < MIN_FULL_SYNAPSES:
        raise MergeError(f"composite arm has {n_syn:.3e} synapses, below the "
                         f"frozen {MIN_FULL_SYNAPSES:.0e} floor")
    if len(seeds) < 3:
        raise MergeError("the frozen contract requires >= 3 composite seeds")
    bc = battery_common()
    fly_sha = hashlib.sha256(Path(fly_path).read_bytes()).hexdigest()
    try:
        fly_src = display(Path(fly_path))
    except ValueError:
        fly_src = str(fly_path)
    arms = {
        "fly": {
            "source": fly_src,
            "source_sha256": fly_sha,
            "n_seeds": fly["n_seeds"], "seeds": fly["seeds"],
            "n_neurons": fly["n_neurons"], "n_synapses": fly["n_synapses"],
            "capacity_ratio": stats["ratio_fly"],
            "C_by_arm": {k: v["C"] for k, v in fly["arms"].items()},
            "assessment": fly["assessment"],
            "ablation_ffi_knockout": fly["ablation_ffi_knockout"],
        },
        "composite": {
            "n_seeds": len(seeds), "seeds": seeds,
            "n_neurons": cc.n_global, "n_synapses": n_syn,
            "capacity_ratio": stats["ratio_comp"],
            "C_by_arm": {k: v["C"] for k, v in stats["arms_composite"].items()},
            "per_instance": results,
            "ablation_ffi_knockout": stats["ablation_ffi_knockout"],
        },
    }
    merged = {"scale": cfg.scale, "config": asdict(cfg),
              "config_hash": cfg.config_hash(), "seeds": seeds,
              "n_synapses": n_syn, "verdict": stats["verdict"],
              "observed": stats["observed"], "arms": arms}
    if write_final:
        dest = bc.write_results(
            "h1", CONTRACT, n_neurons=cc.n_global, n_synapses=n_syn,
            seeds=seeds, verdict=stats["verdict"], arms=arms,
            engine=("fly: brainsim-numpy-rate-cpu; composite: brainsim-torch-"
                    "lif-sharded (CompositeRuntime formation loop)"),
            extra={
                "scope": ("merged H1 battery: fly arm (results/h1/"
                          "fly_arm.json) + composite formation arm; verdict "
                          "= verdicts.h1_verdict(fly per-seed ratios, "
                          "composite per-seed ratios)"),
                "n_seeds_fly": fly["n_seeds"],
                "n_seeds_composite": len(seeds),
                "n_synapses_scope": ("composite arm total (formation loop "
                                     "as configured in the composite; "
                                     "relay+storage at pass-through floors)"),
                "observed": stats["observed"],
                "budget_rows": partials[0]["budget_rows"],
                "forecast": partials[0]["forecast"],
                "composite_config": partials[0]["config"],
                "wall_s_partials": [p["wall_s"] for p in partials],
                "merged_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "deviations": sorted({d for p in partials
                                      for d in p.get("deviations", [])}),
            })
        merged["results_path"] = str(dest)
        log(f"battery h1 merged: verdict={stats['verdict']} "
            f"ratio_comp mean={stats['ratio_comp']['mean']:.4f} "
            f"n_synapses={n_syn:.3e} -> {dest}")
    return merged


__all__ = ["CONTRACT", "FINAL_RESULTS", "FLY_ARM_JSON", "MIN_FULL_SYNAPSES",
           "MergeError", "analyze", "battery_common", "build_work_list",
           "load_fly_report", "make_partial", "merge_h1", "partial_path",
           "select_items", "work_key", "write_partial"]
