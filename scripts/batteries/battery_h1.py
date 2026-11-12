#!/usr/bin/env python3
"""Battery H1 driver (wiring premium).

Fly arm (CPU):
  python3 scripts/batteries/battery_h1.py --arm fly [--seeds 0,1,2,3,4] [--out PATH]
  python3 scripts/batteries/battery_h1.py --arm fly --smoke

Composite arm (node-side; contract arm 1b, formation stage of the composite):
  # smoke: tiny formation config, 2 controls, print-only, writes nothing
  python3 scripts/batteries/battery_h1.py --arm composite --scale smoke

  # full scale, graph instances 8-rank sharded, one instance at a time:
  torchrun --nproc-per-node 8 scripts/batteries/battery_h1.py --arm composite \
      --scale full --seeds 0,1,2 --items sharded --arm-shard 0/1

  # full scale, TF instances packed one per GPU (I of N slices the class):
  CUDA_VISIBLE_DEVICES=$i python3 scripts/batteries/battery_h1.py --arm composite \
      --scale full --seeds 0,1,2 --items packed --arm-shard $i/3

  # node-side merge: fly_arm.json + composite partials -> results.json
  python3 scripts/batteries/battery_h1.py --merge-h1 \
      results/h1/composite_partial_*.json

The fly arm runs the frozen capacity protocol on the complete FlyWire v783
mushroom body and writes results/h1/fly_arm.json. The composite arm
runs the same frozen protocol on the composite's formation loop against the
same frozen control families and writes composite partials. ONLY --merge-h1
writes results/h1/results.json, with the verdict from the frozen
battery_common.h1_verdict(fly ratios, composite ratios). Smoke runs
exercise code paths only and may not produce any capacity number.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

from brainsim.paths import REPO_ROOT as ROOT


def run_fly(args) -> int:
    from brainsim.batteries.h1.fly_arm import DEFAULT_OUT, run_fly_arm
    from brainsim.batteries.h1.protocol import ProtocolCfg

    seeds = tuple(int(s) for s in args.seeds.split(","))
    if args.smoke:
        from brainsim.batteries.h1.synthetic import make_synthetic_graph
        out = args.out or Path(tempfile.gettempdir()) / "h1_smoke" / "fly_arm_smoke.json"
        cfg = ProtocolCfg(p_cap=48, eval_sample=16, write_chunk=16)
        rep = run_fly_arm(make_synthetic_graph(), out_path=out, seeds=seeds[:2],
                          cfg=cfg, smoke=True,
                          provenance={"note": "synthetic smoke graph"})
        print(f"SMOKE OK: code paths exercised, wrote {out} "
              f"(smoke={rep['smoke']}, no real-circuit numbers)")
        return 0

    if len(seeds) < 5:
        print(f"FATAL: contract requires >= 5 seeds on the fly arm, got {seeds}")
        return 1
    import subprocess
    from brainsim.batteries.h1.convert import provenance
    from brainsim.batteries.h1.mb_extract import load_mb_graph
    print("extracting MB subgraph from FlyWire v783 ...", flush=True)
    prov = provenance()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                           capture_output=True, text=True).stdout.strip()
    prov["git_dirty"] = bool(dirty)  # code_version is HEAD; flag uncommitted code
    graph = load_mb_graph()
    for line in ("neurons " + str(graph.counts["neurons"]),
                 f"circuit pairs {graph.counts['circuit_pairs']} "
                 f"synapses {graph.counts['circuit_synapses']}"):
        print("  " + line, flush=True)
    out = args.out or DEFAULT_OUT
    rep = run_fly_arm(graph, out_path=out, seeds=seeds, provenance=prov)
    print(f"fly arm complete in {rep['runtime_s']} s -> {out}")
    print("  assessment:", rep["assessment"]["fly_only_assessment"])
    rb = rep["ratio_C_meas_over_C_shuf"]
    if "mean" in rb:
        print(f"  C_meas/C_shuf mean {rb['mean']:.4f} ci95 "
              f"[{rb['ci95'][0]:.4f}, {rb['ci95'][1]:.4f}]")
    for dev in rep["deviations"]:
        print("  !!", dev)
    return 0


def run_composite(args) -> int:
    from brainsim.batteries.h1 import composite_runner as runner
    from brainsim.batteries.h1.composite_arm import (forecast_table,
                                                     full_comp_config,
                                                     smoke_comp_config)

    scale = "smoke" if args.smoke else args.scale
    seeds = ([int(s) for s in args.seeds.split(",")] if args.seeds
             else runner.default_seeds(scale))
    if scale == "full" and len(seeds) < 3:
        print("FATAL: the frozen contract requires >= 3 composite seeds at "
              "full scale")
        return 2
    over = {k: v for k, v in (("grid_ratio", args.grid_ratio),
                              ("p_cap", args.p_cap),
                              ("eval_sample", args.eval_sample)) if v}
    cfg = (full_comp_config(**over) if scale == "full"
           else smoke_comp_config(**over))
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    device = runner.resolve_device(args.device)
    if world > 1:
        import torch
        import torch.distributed as dist
        if device == "cuda":
            torch.cuda.set_device(rank % max(1, torch.cuda.device_count()))
        dist.init_process_group("nccl" if device == "cuda" else "gloo")
    try:
        if scale == "full" and rank == 0:
            fc = forecast_table(cfg, seeds[0])
            print("forecast:", {k: fc[k] for k in
                                ("graph_arm_gb_per_rank_ws8",
                                 "graph_arm_gb_single_rank",
                                 "tf_gb_single_device", "n_synapses_total")})
        runner.run(cfg, seeds, out_dir=args.out, arm_shard=args.arm_shard,
                   items_filter=args.items, rank=rank, world_size=world,
                   device=device)
    finally:
        if world > 1:
            import torch.distributed as dist
            dist.destroy_process_group()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=["fly", "composite"])
    ap.add_argument("--seeds", default=None,
                    help="comma-separated seed list (fly default 0-4, "
                         "composite full default 0-2, smoke default 0)")
    ap.add_argument("--out", type=Path, default=None,
                    help="fly: report path; composite: partial output dir")
    ap.add_argument("--smoke", action="store_true",
                    help="code-path check only (fly: synthetic tiny graph; "
                         "composite: alias for --scale smoke)")
    ap.add_argument("--scale", choices=["smoke", "full"], default="smoke",
                    help="composite arm scale (default smoke)")
    ap.add_argument("--items", choices=["all", "sharded", "packed"],
                    default="all",
                    help="composite work-item class: graph instances "
                         "(sharded, torchrun) vs tf (packed, one GPU)")
    ap.add_argument("--arm-shard", default=None, metavar="I/N",
                    help="composite: run slice I of the selected work list "
                         "(ZERO-BASED); partials name the slice")
    ap.add_argument("--device", default="auto",
                    help="composite torch device (default auto)")
    ap.add_argument("--grid-ratio", type=float, default=None,
                    help="composite sweep resolution override (engineering, "
                         "recorded; frozen thresholds unaffected)")
    ap.add_argument("--p-cap", type=int, default=None,
                    help="composite stream cap override (engineering)")
    ap.add_argument("--eval-sample", type=int, default=None,
                    help="composite eval sample override (engineering)")
    ap.add_argument("--merge-h1", nargs="+", default=None, metavar="PARTIAL",
                    help="merge composite partials with fly_arm.json -> "
                         "results/h1/results.json (frozen verdict)")
    ap.add_argument("--fly", type=Path, default=None,
                    help="fly arm report for --merge-h1 "
                         "(default results/h1/fly_arm.json)")
    args = ap.parse_args()

    try:
        if args.merge_h1:
            from brainsim.batteries.h1 import composite_report as report
            report.merge_h1(args.merge_h1,
                            fly_path=args.fly or report.FLY_ARM_JSON)
            return 0
        if args.arm is None:
            ap.error("--arm is required unless --merge-h1 is given")
        if args.arm == "fly":
            if args.seeds is None:
                args.seeds = "0,1,2,3,4"
            return run_fly(args)
        return run_composite(args)
    except SystemExit:
        raise
    except Exception as e:  # loud, typed failure for the orchestrator
        print(f"FATAL: {type(e).__name__}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
