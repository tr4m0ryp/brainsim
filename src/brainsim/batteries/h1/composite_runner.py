"""Battery H1 composite-arm runner: work-item execution, shard slicing and
the smoke rails.

Execution classes (mirrors the H3 session pattern): composite-graph
instances (measured, FFI-KO, config-model draws, weight shuffle, ER) run
'sharded' -- under torchrun with the runtime SPMD-sharded across the node's
8 ranks, ONE instance at a time, instances sequenced; the TF attractor is a
'packed' single-device item (the forecast puts it at ~98 GB on one H200;
the graph arms also fit one device at the thinned-loop scale, an upside the
forecast table records without exploiting). --arm-shard I/N (zero-based)
slices whichever class the invocation selects.

Smoke rails (contract: no capacity number may be produced at smoke scale):
smoke never writes under results/, the capacity machinery's checkpoint log
is silenced, and the console summary reports code-path booleans and counts
only. Full-scale invocations write composite partials; results.json exists
only through composite_report.merge_h1 (fly + composite, frozen verdict).
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from brainsim.batteries.h1 import composite_controls as controls
from brainsim.batteries.h1 import composite_report as report
from brainsim.batteries.h1.composite_arm import (FFI_KO_STAGES, CompH1Config,
                                                 base_cc, loop_neurons,
                                                 run_graph_capacity)
from brainsim.batteries.h1.composite_tf import run_tf_capacity_comp
from brainsim.batteries.h3.budget import assert_matched, total_synapses
from brainsim.batteries.h3.runner import parse_shard, resolve_device

_SILENT = (lambda *a, **k: None)


def _global_synapse_count(rt) -> int:
    local = rt.net.synapse_count()
    if rt.world_size == 1:
        return int(local)
    import torch
    import torch.distributed as dist
    t = torch.tensor([local], dtype=torch.int64, device=rt.device)
    dist.all_reduce(t)
    return int(t.item())


def _free(rt) -> None:
    import torch
    del rt
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_work_item(cfg: CompH1Config, item: dict, rank: int = 0,
                  world_size: int = 1, device: str = "cpu",
                  log=print) -> dict:
    """Pure function of (cfg, item): the item/cue/eval streams are keyed by
    (H1_KEY, seed, stream), never by shard layout, so any partition of the
    work list reproduces identical protocol draws. Control-surgery draws
    are rank-keyed (documented in composite_controls)."""
    name, seed = item["name"], item["seed"]
    cc = base_cc(cfg, seed)
    target = int(total_synapses(cc))
    if name == "tf_attractor":
        if world_size != 1:
            raise RuntimeError("tf_attractor is a packed single-device work "
                               "item; run it outside torchrun")
        cap = run_tf_capacity_comp(loop_neurons(cc), 2 * (target // 2),
                                   cc.f_formation, seed, cfg,
                                   device=device, log=log)
        assert_matched("tf_attractor", cap["n_directed_entries"], target)
        meta = {"kind": "tf", "budget_total": cap["n_directed_entries"],
                "budget_target": target, "n_units": cap["n_units"]}
    else:
        from brainsim.batteries.h3.systems import H3Runtime
        from brainsim.composite.full.runtime import CompositeRuntime
        if name == "measured_ffi_ko":
            rt = H3Runtime(cc, rank=rank, world_size=world_size,
                           device=device, topf_off=FFI_KO_STAGES)
        else:
            rt = CompositeRuntime(cc, rank=rank, world_size=world_size,
                                  device=device)
        controls.apply_control(rt, name, seed)
        built = _global_synapse_count(rt)
        # matched budget: built count vs config arithmetic (self-loop drop
        # is the only slack, ~k/n) -- asserted, not assumed
        assert_matched(name, built, target)
        cap = run_graph_capacity(rt, seed, cfg, log=log)
        meta = {"kind": "graph", "budget_total": built,
                "budget_target": target, "n_neurons": cc.n_global,
                "ffi_ko_stages": list(FFI_KO_STAGES)
                if name == "measured_ffi_ko" else None}
        _free(rt)
    return {"key": report.work_key(item), "name": name, "seed": int(seed),
            "exec_class": item["exec_class"], "capacity": cap, "meta": meta}


def run_shard(cfg: CompH1Config, seeds: list[int], shard_i: int, shard_n: int,
              items_filter: str, rank: int = 0, world_size: int = 1,
              device: str = "cpu", log=print, item_log=None) -> dict:
    mine = report.select_items(cfg, seeds, shard_i, shard_n, items_filter)
    t0 = time.monotonic()
    results: dict[str, dict] = {}
    for it in mine:
        res = run_work_item(cfg, it, rank=rank, world_size=world_size,
                            device=device,
                            log=item_log if item_log is not None else log)
        results[res["key"]] = res
        log(f"  done {res['key']} [{len(results)}/{len(mine)} of shard "
            f"{shard_i}/{shard_n} ({items_filter})]")
    return report.make_partial(cfg, seeds, (shard_i, shard_n), items_filter,
                               results, wall_s=time.monotonic() - t0)


def _smoke_summary(cfg: CompH1Config, seeds: list[int], partial: dict,
                   log=print) -> None:
    """Structural report only: the frozen contract forbids producing any
    capacity value at smoke scale, so measured values are suppressed.
    Budget ratios are config arithmetic, not measurements."""
    bc = report.battery_common()
    rows = partial["budget_rows"]
    matched = all(abs(r["total"] - r["target"]) <= 0.01 * r["target"]
                  for r in rows.values())
    names = sorted({r["name"] for r in partial["results"].values()})
    swept = all("curve_P" in r["capacity"] and len(r["capacity"]["curve_P"])
                for r in partial["results"].values())
    token = bc.h1_verdict([1.0], [1.0])  # wiring probe on dummies only
    log("=" * 72)
    log("SMOKE RUN - code-path verification only. Nothing was written under")
    log("results/; every measured value is suppressed (contract: no smoke-")
    log("scale numbers).")
    log(f"  work items completed : {len(partial['results'])}")
    log(f"  instances exercised  : {names}")
    log(f"  seeds                : {seeds}")
    log(f"  matched budgets      : within 1% = {matched} ({len(rows)} rows)")
    log(f"  capacity machinery   : sweep + interpolation ran on every "
        f"instance = {swept} (values suppressed)")
    log(f"  verdict wiring       : verdicts.h1_verdict callable, "
        f"returned a token in {{supported,killed,partial}} on dummy input "
        f"= {token in {'supported', 'killed', 'partial'}}")
    log(f"  forecast (full cfg is separate): graph arm "
        f"{partial['forecast']['graph_arm_gb_per_rank_ws8']} GB/rank at ws8")
    log("=" * 72)


def run(cfg: CompH1Config, seeds: list[int], out_dir: str | None = None,
        arm_shard: str | None = None, items_filter: str = "all",
        rank: int = 0, world_size: int = 1, device: str = "cpu",
        log=print) -> dict:
    if world_size > 1 and items_filter != "sharded":
        if items_filter == "packed":
            raise SystemExit("packed (tf) items are single-process work; "
                             "run them outside torchrun")
        items_filter = "sharded"
        log("torchrun context: --items forced to 'sharded' (tf runs packed, "
            "one process per GPU)")
    shard_i, shard_n = parse_shard(arm_shard)
    rank_log = log if rank == 0 else _SILENT
    # smoke silences the capacity checkpoint log (overlap values), full logs it
    item_log = rank_log if cfg.scale == "full" else _SILENT
    partial = run_shard(cfg, seeds, shard_i, shard_n, items_filter,
                        rank=rank, world_size=world_size, device=device,
                        log=rank_log, item_log=item_log)
    if rank != 0:
        return partial
    if cfg.scale == "full":
        dest = report.write_partial(
            partial, Path(out_dir) if out_dir else report.H1_DIR)
        log(f"shard {shard_i}/{shard_n} ({items_filter}) complete -> {dest}")
        log("merge when all partials exist: scripts/batteries/battery_h1.py "
            "--merge-h1 results/h1/composite_partial_*.json")
        return partial
    sharded = shard_n > 1 or items_filter != "all"
    if sharded:
        if not out_dir:
            raise SystemExit("sharded smoke runs need --out (smoke partials "
                             "may not land in results/h1)")
        dest = report.write_partial(partial, Path(out_dir))
        log(f"shard {shard_i}/{shard_n} ({items_filter}) complete -> {dest}")
        return partial
    _smoke_summary(cfg, seeds, partial, log=log)
    return partial


def default_seeds(scale: str, n: int | None = None) -> list[int]:
    count = n if n is not None else (3 if scale == "full" else 1)
    return list(range(count))


__all__ = ["default_seeds", "parse_shard", "resolve_device", "run",
           "run_shard", "run_work_item"]
