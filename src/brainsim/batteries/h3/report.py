"""Work list, partials, merge, and final results assembly for battery H3.

Verdict logic is NEVER implemented here: composite_retained and
best_part_retained are computed from arm results and handed to the FROZEN
verdicts.h3_verdict; CIs come from verdicts.bootstrap_ci; the
final file is written by verdicts.write_results. Only the merge path
may write results/h3/results.json, and only for scale == "full" at the
frozen >= 1e10 composite synapse floor with >= 3 seeds.

Sharding: partials may slice the work list round-robin (--arm-shard I/N,
zero-based) AND filter by execution class (sharded composite-scale items vs
packed bag groups); the merge verifies the UNION of partials covers the
canonical work list exactly once.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from brainsim.batteries import verdicts
from brainsim.paths import RESULTS

from . import analysis, screen
from .config import HEADLINE_LAMBDA, PROVENANCE_NOTES, H3Config

CONTRACT_NAME = "battery-h3-composite.md"
FINAL_RESULTS = RESULTS / "h3" / "results.json"
MIN_FULL_SYNAPSES = 10_000_000_000  # frozen validate.py h3 floor


def battery_common():
    """The frozen verdict module (brainsim.batteries.verdicts)."""
    return verdicts


# ---- canonical work list -----------------------------------------------------

def work_key(it: dict) -> str:
    if it["kind"] == "ko":
        return f"ko|{it['name']}|seed={it['seed']}"
    key = f"arm|{it['name']}|lf={it['lf']:g}|seed={it['seed']}"
    if it.get("group") is not None:
        key += f"|g={it['group']}"
    return key


def build_work_list(cfg: H3Config, seeds: list[int]) -> list[dict]:
    """Canonical enumeration (frozen order); shard slices are round-robin
    over this list. Bag groups are 'packed' items, everything else runs at
    composite scale ('sharded')."""
    work: list[dict] = []

    def add(kind, name, lf=HEADLINE_LAMBDA, group=None, seed=None):
        work.append({"kind": kind, "name": name, "lf": float(lf),
                     "group": group, "seed": int(seed),
                     "exec_class": "packed" if name == "bag" else "sharded"})

    for seed in seeds:
        for lf in cfg.lambda_factors:
            add("arm", "composite", lf, seed=seed)
        for s in cfg.stage_alone_arms:
            add("arm", s, seed=seed)
        if cfg.run_bag:
            for g in range(cfg.bag_groups):
                add("arm", "bag", group=g, seed=seed)
        for a in cfg.ablation_arms:
            add("arm", a, seed=seed)
        for c in cfg.context_arms:
            add("arm", c, seed=seed)
        for k in screen.ko_names(cfg):
            add("ko", k, seed=seed)
    return work


# ---- partials ------------------------------------------------------------------

def partial_path(out_dir: Path, items_filter: str, shard_i: int,
                 shard_n: int) -> Path:
    stem = f"partial_{shard_i}_of_{shard_n}.json" if items_filter == "all" \
        else f"partial_{items_filter}_{shard_i}_of_{shard_n}.json"
    return Path(out_dir) / stem


def make_partial(cfg: H3Config, seeds: list[int], shard: tuple[int, int],
                 items_filter: str, results: dict, wall_s: float) -> dict:
    bc = battery_common()
    return {
        "battery": "h3", "shard": [int(shard[0]), int(shard[1])],
        "items_filter": items_filter, "scale": cfg.scale,
        "engine": cfg.engine, "config": asdict(cfg),
        "config_hash": cfg.config_hash(),
        "contract_sha256": bc.contract_sha256(CONTRACT_NAME),
        "code_version": bc.git_sha(), "seeds": [int(s) for s in seeds],
        "results": results, "wall_s": wall_s,
    }


def write_partial(partial: dict, out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = partial_path(out_dir, partial["items_filter"], *partial["shard"])
    dest.write_text(json.dumps(partial, indent=1))
    return dest


class MergeError(RuntimeError):
    pass


def _collect(partials: list[dict]) -> tuple[H3Config, list[int], dict]:
    if not partials:
        raise MergeError("no partials to merge")
    for field in ("config_hash", "contract_sha256", "scale", "engine", "seeds"):
        vals = {json.dumps(p[field], sort_keys=True) for p in partials}
        if len(vals) != 1:
            raise MergeError(f"partials disagree on {field}: {vals}")
    bc = battery_common()
    if partials[0]["contract_sha256"] != bc.contract_sha256(CONTRACT_NAME):
        raise MergeError("partials attest a different frozen contract than "
                         "the one on disk; refusing to merge")
    cfg_d = dict(partials[0]["config"])
    for key, val in cfg_d.items():
        if isinstance(val, list):
            cfg_d[key] = tuple(val)
    cfg = H3Config(**cfg_d)
    seeds = list(partials[0]["seeds"])
    results: dict = {}
    for p in partials:
        for key, val in p["results"].items():
            if key in results:
                raise MergeError(f"duplicate work item across partials: {key}")
            results[key] = val
    expected = {work_key(it) for it in build_work_list(cfg, seeds)}
    missing = expected - set(results)
    extra = set(results) - expected
    if missing or extra:
        raise MergeError(f"{len(missing)} missing / {len(extra)} unexpected "
                         f"work items, e.g. {sorted(missing | extra)[:4]}")
    return cfg, seeds, results


# ---- analysis (implementation in analysis.py; verdict stays frozen) ---------

def analyze(cfg: H3Config, seeds: list[int], results: dict) -> dict:
    return analysis.analyze(cfg, seeds, results, work_key, battery_common)


retained_at_t = analysis.retained_at_t


def merge_partials(partials: list[dict], write_final: bool) -> dict:
    cfg, seeds, results = _collect(partials)
    stats = analyze(cfg, seeds, results)
    comp_meta = [v["meta"] for v in results.values()
                 if v["meta"].get("arm") == "composite"]
    n_syn = min(int(m["budget_target"]) for m in comp_meta)
    n_neu = min(int(m.get("n_neurons", 0)) for m in comp_meta)
    merged = {"scale": cfg.scale, "engine": cfg.engine, "config": asdict(cfg),
              "config_hash": cfg.config_hash(), "seeds": seeds,
              "n_neurons": n_neu, "n_synapses": n_syn, "results": results,
              **stats, "notes": list(PROVENANCE_NOTES)}
    if write_final:
        if cfg.scale != "full":
            raise MergeError("refusing to write results.json from a non-full "
                             f"scale run (scale={cfg.scale}); smoke is print-only")
        if n_syn < MIN_FULL_SYNAPSES:
            raise MergeError(f"composite arm has {n_syn:.3e} synapses, below "
                             f"the frozen {MIN_FULL_SYNAPSES:.0e} floor")
        if len(seeds) < 3:
            raise MergeError("the frozen contract requires >= 3 seeds")
        bc = battery_common()
        dest = bc.write_results(
            "h3", CONTRACT_NAME, n_neurons=n_neu, n_synapses=n_syn,
            seeds=seeds, verdict=stats["verdict"], arms=results,
            engine=cfg.engine,
            extra={k: stats[k] for k in
                   ("composite_retained", "best_part_retained",
                    "best_part_arm", "arm_table", "lambda_overlay",
                    "ko_screen", "budget_table", "analysis_flags")} | {
                "config": asdict(cfg), "config_hash": cfg.config_hash(),
                "notes": list(PROVENANCE_NOTES)})
        merged["results_path"] = str(dest)
    return merged


__all__ = ["build_work_list", "work_key", "make_partial", "write_partial",
           "partial_path", "merge_partials", "analyze", "retained_at_t",
           "battery_common", "MergeError", "CONTRACT_NAME", "FINAL_RESULTS",
           "MIN_FULL_SYNAPSES"]
