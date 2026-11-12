"""Battery H2 runner: shard execution, in-process merge for unsharded runs,
and the smoke rails.

Smoke rails (contract: no T_half, no frontier point may be produced at smoke
scale): a smoke run never writes results.json (merge_partials refuses
non-full scale), and the smoke console summary suppresses every measured
value -- it reports code-path booleans and counts only. Full-scale numbers
are printed and written only by the full path.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from . import artifact
from .arms import RunCaches, build_work_list, run_work_item, work_key
from .config import H2Config, full_config, smoke_config, tiny_config
from .report import (FINAL_RESULTS, MergeError, make_partial, merge_partials,
                     write_partial)

DEFAULT_FULL_OUT = FINAL_RESULTS.parent


def default_seeds(n: int) -> list[int]:
    return [101 + 97 * i for i in range(n)]


def parse_shard(spec: str | None) -> tuple[int, int]:
    if spec is None:
        return 0, 1
    i_s, n_s = spec.split("/")
    i, n = int(i_s), int(n_s)
    if not (0 <= i < n):
        raise ValueError(f"--arm-shard {spec}: need 0 <= I < N")
    return i, n


def run_shard(cfg: H2Config, seeds: list[int], shard_i: int, shard_n: int,
              stats_dir: str | None = None, log=print) -> dict:
    stats = artifact.load(stats_dir) if stats_dir else artifact.load()
    caches = RunCaches(cfg, stats)
    work = build_work_list(cfg, seeds)
    mine = [w for j, w in enumerate(work) if j % shard_n == shard_i]
    t0 = time.monotonic()
    results: dict[str, dict] = {}
    for arm, k, p, seed in mine:
        results[work_key(arm, k, p, seed)] = run_work_item(caches, arm, k, p, seed)
        log(f"  done {work_key(arm, k, p, seed)} "
            f"[{len(results)}/{len(mine)} of shard {shard_i}/{shard_n}]")
    return make_partial(cfg, seeds, shard_i, shard_n, results, caches.gates(),
                        wall_s=time.monotonic() - t0)


def _smoke_summary(merged: dict, log=print) -> None:
    """Structural report only: the frozen contract forbids producing any
    T_half or frontier number at smoke scale, so all values are suppressed."""
    gates = merged["twin_gate"]
    log("=" * 72)
    log("SMOKE RUN - code-path verification only. No file was written; every")
    log("measured value is suppressed (contract: no smoke-scale numbers).")
    log(f"  work items completed : {len(merged['results'])}")
    log(f"  twin gate            : "
        f"{ {s: g['passed'] for s, g in gates.items()} }")
    log(f"  arms exercised       : "
        f"{sorted({r['arm'] for r in merged['results'].values()})}")
    log(f"  shift grid cells     : {len(merged['shift_by_kp'])}")
    log(f"  verdict machinery    : ran, produced a token in "
        f"{{supported,killed,partial}} (value suppressed at smoke scale)")
    log(f"  analysis flags       : {len(merged['analysis_flags'])}")
    log("=" * 72)


def _full_summary(merged: dict, log=print) -> None:
    log(f"battery h2 [{merged['engine']}]: verdict={merged['verdict']} "
        f"n_synapses={merged['n_synapses']:.3e} seeds={merged['seeds']}")
    for key, shift in sorted(merged["shift_by_kp"].items()):
        band = merged["noise_band_by_kp"][key]
        mark = "EXCEEDS" if shift > band else "within"
        log(f"  {key}: shift={shift:+.4f} band={band:.4f} ({mark} noise band)")
    if merged.get("results_path"):
        log(f"  wrote {merged['results_path']}")


def run(cfg: H2Config, seeds: list[int], out_dir: str | None = None,
        arm_shard: str | None = None, stats_dir: str | None = None,
        log=print) -> dict:
    shard_i, shard_n = parse_shard(arm_shard)
    sharded = shard_n > 1
    partial = run_shard(cfg, seeds, shard_i, shard_n, stats_dir, log=log)
    if sharded:
        if cfg.scale == "full":
            dest = write_partial(partial, Path(out_dir) if out_dir else DEFAULT_FULL_OUT)
        else:
            if not out_dir:
                raise SystemExit("sharded smoke runs need --out (smoke partials "
                                 "may not land in results/h2)")
            dest = write_partial(partial, Path(out_dir))
        log(f"shard {shard_i}/{shard_n} complete -> {dest}")
        return partial
    if cfg.scale == "full":  # keep the evidence even if the merge refuses
        write_partial(partial, Path(out_dir) if out_dir else DEFAULT_FULL_OUT)
    merged = merge_partials([partial], write_final=(cfg.scale == "full"))
    if cfg.scale == "full":
        _full_summary(merged, log=log)
    else:
        _smoke_summary(merged, log=log)
    return merged


def merge_files(paths: list[str], log=print) -> dict:
    partials = [json.loads(Path(p).read_text()) for p in sorted(paths)]
    scale = partials[0].get("scale") if partials else None
    merged = merge_partials(partials, write_final=(scale == "full"))
    if scale == "full":
        _full_summary(merged, log=log)
    else:
        _smoke_summary(merged, log=log)
    return merged


def config_for(scale: str, engine: str, **kw) -> H2Config:
    if scale == "full":
        return full_config(engine=engine, **kw)
    if scale == "smoke":
        return smoke_config(engine=engine, **kw)
    if scale == "tiny":
        return tiny_config(engine=engine, **kw)
    raise ValueError(f"unknown scale {scale!r}")


__all__ = ["run", "run_shard", "merge_files", "config_for", "default_seeds",
           "parse_shard", "DEFAULT_FULL_OUT"]
