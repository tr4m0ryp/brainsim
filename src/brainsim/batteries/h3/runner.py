"""Battery H3 runner: work-item execution, shard slicing, in-process merge
for unsharded runs, and the smoke rails.

Smoke rails (contract: no retention/capacity number may be produced at
smoke scale): a smoke run never writes results.json (merge_partials refuses
non-full scale), and the smoke console summary suppresses every measured
value -- code-path booleans and counts only.

Execution classes (driver-spec): composite-scale items ('sharded') run
under torchrun with the runtime SPMD-sharded across ranks -- rank 0 writes
the partial; bag groups ('packed') are single-process work items, one GPU
each. --arm-shard I/N (zero-based) slices whichever class this invocation
selects; the merge checks the union.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from . import arms, report, screen
from .config import H3Config, full_config, smoke_config, tiny_config
from .protocol import run_protocol

DEFAULT_FULL_OUT = report.FINAL_RESULTS.parent


def default_seeds(n: int) -> list[int]:
    return [101 + 97 * i for i in range(n)]


def parse_shard(spec: str | None) -> tuple[int, int]:
    if spec is None:
        return 0, 1
    i_s, n_s = spec.split("/")
    i, n = int(i_s), int(n_s)
    if not (0 <= i < n):
        raise ValueError(f"--arm-shard {spec}: need 0 <= I < N (zero-based)")
    return i, n


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def run_work_item(cfg: H3Config, item: dict, rank: int = 0,
                  world_size: int = 1, device: str = "cpu") -> dict:
    """Pure function of (cfg, item): every shard partition reproduces the
    identical numbers (seed mapping is SeedSequence-keyed, never layout-
    dependent)."""
    if item["kind"] == "ko":
        system, pp, meta = screen.make_ko_system(
            cfg, item["seed"], item["name"], rank=rank,
            world_size=world_size, device=device)
    else:
        system, pp, meta = arms.make_arm_system(
            cfg, item["seed"], item["name"], item["lf"], group=item["group"],
            rank=rank, world_size=world_size, device=device)
    metrics = run_protocol(system, pp, item["seed"])
    return {"key": report.work_key(item), "kind": item["kind"],
            "name": item["name"], "seed": item["seed"], "lf": item["lf"],
            "group": item["group"], "meta": meta, "metrics": metrics}


def select_items(cfg: H3Config, seeds: list[int], shard_i: int, shard_n: int,
                 items_filter: str) -> list[dict]:
    work = [it for it in report.build_work_list(cfg, seeds)
            if items_filter == "all" or it["exec_class"] == items_filter]
    return [it for j, it in enumerate(work) if j % shard_n == shard_i]


def run_shard(cfg: H3Config, seeds: list[int], shard_i: int, shard_n: int,
              items_filter: str = "all", rank: int = 0, world_size: int = 1,
              device: str = "cpu", log=print) -> dict:
    mine = select_items(cfg, seeds, shard_i, shard_n, items_filter)
    t0 = time.monotonic()
    results: dict[str, dict] = {}
    for it in mine:
        res = run_work_item(cfg, it, rank=rank, world_size=world_size,
                            device=device)
        results[res["key"]] = res
        log(f"  done {res['key']} [{len(results)}/{len(mine)} of shard "
            f"{shard_i}/{shard_n} ({items_filter})]")
    return report.make_partial(cfg, seeds, (shard_i, shard_n), items_filter,
                               results, wall_s=time.monotonic() - t0)


def _smoke_summary(merged: dict, log=print) -> None:
    """Structural report only: the frozen contract forbids producing any
    retention/capacity number at smoke scale, so measured values are
    suppressed. Budget ratios are config arithmetic, not measurements."""
    budgets = merged.get("budget_table", {})
    matched = all(abs(b["total"] - b["target"]) <= 0.01 * b["target"]
                  for b in budgets.values()) if budgets else True
    arms_seen = sorted({r["name"] for r in merged["results"].values()
                        if r["kind"] == "arm"})
    kos_seen = sorted({r["name"] for r in merged["results"].values()
                       if r["kind"] == "ko"})
    log("=" * 72)
    log("SMOKE RUN - code-path verification only. No file was written; every")
    log("measured value is suppressed (contract: no smoke-scale numbers).")
    log(f"  work items completed : {len(merged['results'])}")
    log(f"  arms exercised       : {arms_seen}")
    log(f"  screen rows          : {kos_seen}")
    log(f"  matched budgets      : within 1% = {matched} "
        f"({len(budgets)} budget-bearing items)")
    log(f"  lambda sweep points  : "
        f"{len(merged['lambda_overlay']['lambda_factors'])}")
    log(f"  KO ranking rows      : {len(merged['ko_screen']['ranking'])}")
    log(f"  verdict machinery    : ran, produced a token in "
        f"{{supported,killed,partial}} (value suppressed at smoke scale)")
    log(f"  analysis flags       : {len(merged['analysis_flags'])}")
    log("=" * 72)


def _full_summary(merged: dict, log=print) -> None:
    log(f"battery h3 [{merged['engine']}]: verdict={merged['verdict']} "
        f"composite={merged['composite_retained']:.2f} "
        f"best_part={merged['best_part_retained']:.2f} "
        f"({merged['best_part_arm']}) n_synapses={merged['n_synapses']:.3e} "
        f"seeds={merged['seeds']}")
    for name, row in merged["arm_table"].items():
        log(f"  {name}: retained_at_T mean={row['mean']:.2f} "
            f"ci95=[{row['ci95'][0]:.2f}, {row['ci95'][1]:.2f}]")
    for fu in merged["ko_screen"]["hidden_key_followups"]:
        log(f"  hidden-key #{fu['rank']}: {fu['ko']} (d={fu['distance']:.2f})")
    if merged.get("results_path"):
        log(f"  wrote {merged['results_path']}")


def run(cfg: H3Config, seeds: list[int], out_dir: str | None = None,
        arm_shard: str | None = None, items_filter: str = "all",
        rank: int = 0, world_size: int = 1, device: str = "cpu",
        log=print) -> dict:
    if world_size > 1 and items_filter != "sharded":
        if items_filter == "packed":
            raise SystemExit("packed (bag) items are single-process work; "
                             "run them outside torchrun")
        items_filter = "sharded"
        log("torchrun context: --items forced to 'sharded' "
            "(bag groups run packed, one process per GPU)")
    shard_i, shard_n = parse_shard(arm_shard)
    partial = run_shard(cfg, seeds, shard_i, shard_n, items_filter,
                        rank=rank, world_size=world_size, device=device,
                        log=log)
    if rank != 0:
        return partial
    sharded = shard_n > 1 or items_filter != "all"
    if sharded:
        if cfg.scale == "full":
            dest = report.write_partial(
                partial, Path(out_dir) if out_dir else DEFAULT_FULL_OUT)
        else:
            if not out_dir:
                raise SystemExit("sharded smoke runs need --out (smoke "
                                 "partials may not land in results/h3)")
            dest = report.write_partial(partial, Path(out_dir))
        log(f"shard {shard_i}/{shard_n} ({items_filter}) complete -> {dest}")
        return partial
    if cfg.scale == "full":  # keep the evidence even if the merge refuses
        report.write_partial(partial, Path(out_dir) if out_dir else DEFAULT_FULL_OUT)
    merged = report.merge_partials([partial], write_final=(cfg.scale == "full"))
    if cfg.scale == "full":
        _full_summary(merged, log=log)
    else:
        _smoke_summary(merged, log=log)
    return merged


def merge_files(paths: list[str], log=print) -> dict:
    partials = [json.loads(Path(p).read_text()) for p in sorted(paths)]
    scale = partials[0].get("scale") if partials else None
    merged = report.merge_partials(partials, write_final=(scale == "full"))
    if scale == "full":
        _full_summary(merged, log=log)
    else:
        _smoke_summary(merged, log=log)
    return merged


def config_for(scale: str, engine: str, **kw) -> H3Config:
    if scale == "full":
        return full_config(engine=engine, **kw)
    if scale == "smoke":
        return smoke_config(engine=engine, **kw)
    if scale == "tiny":
        return tiny_config(engine=engine, **kw)
    raise ValueError(f"unknown scale {scale!r}")


__all__ = ["run", "run_shard", "run_work_item", "select_items",
           "merge_files", "config_for", "default_seeds", "parse_shard",
           "resolve_device", "DEFAULT_FULL_OUT"]
