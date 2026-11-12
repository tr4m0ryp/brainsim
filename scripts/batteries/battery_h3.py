#!/usr/bin/env python3
"""Battery H3 driver CLI (contract: battery-h3-composite.md, FROZEN).

Usage:
  # smoke: code paths only, print-only, never writes results.json
  python3 scripts/batteries/battery_h3.py --engine torch --scale smoke

  # full scale, composite-scale items 8-rank sharded (runtime handles it):
  torchrun --nproc-per-node 8 scripts/batteries/battery_h3.py \
      --engine torch --scale full --seeds 3 --items sharded --arm-shard 0/1

  # full scale, bag groups packed one per GPU:
  CUDA_VISIBLE_DEVICES=$i python3 scripts/batteries/battery_h3.py \
      --engine torch --scale full --seeds 3 --items packed --arm-shard $i/8

  # merge the partials -> frozen verdict -> results/h3/results.json
  python3 scripts/batteries/battery_h3.py --merge results/h3/partial_*.json

Determinism: the (work item, seed) -> RNG mapping is SeedSequence-keyed and
never depends on the shard layout, so any --arm-shard partition of either
item class reproduces identical numbers.
"""
from __future__ import annotations

import argparse
import os
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", choices=["torch"], default="torch",
                    help="the composite runtime is torch-only")
    ap.add_argument("--scale", choices=["smoke", "full"], default="smoke")
    ap.add_argument("--seeds", type=int, default=None,
                    help="number of seeds (full default 3, smoke default 1)")
    ap.add_argument("--out", default=None,
                    help="output dir (full default results/h3; required "
                         "for sharded smoke)")
    ap.add_argument("--arm-shard", default=None, metavar="I/N",
                    help="run only slice I of the selected work list "
                         "(ZERO-BASED) and write partial_*_I_of_N.json")
    ap.add_argument("--items", choices=["all", "sharded", "packed"],
                    default="all",
                    help="work-item class: composite-scale (sharded) vs bag "
                         "groups (packed); default all (smoke/tiny)")
    ap.add_argument("--merge", nargs="+", default=None, metavar="PARTIAL",
                    help="merge partial_*.json files; writes results.json "
                         "only when every partial is a full-scale run")
    ap.add_argument("--device", default="auto", help="torch device (default auto)")
    args = ap.parse_args()

    from brainsim.batteries.h3 import runner

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    try:
        if args.merge:
            runner.merge_files(args.merge)
            return 0
        n_seeds = args.seeds if args.seeds else (3 if args.scale == "full" else 1)
        if args.scale == "full" and n_seeds < 3:
            print("FATAL: the frozen contract requires >= 3 seeds at full scale")
            return 2
        device = runner.resolve_device(args.device)
        if world > 1:
            import torch
            import torch.distributed as dist
            if device == "cuda":
                torch.cuda.set_device(rank % max(1, torch.cuda.device_count()))
            dist.init_process_group("nccl" if device == "cuda" else "gloo")
        try:
            cfg = runner.config_for(args.scale, args.engine)
            runner.run(cfg, runner.default_seeds(n_seeds), out_dir=args.out,
                       arm_shard=args.arm_shard, items_filter=args.items,
                       rank=rank, world_size=world, device=device)
        finally:
            if world > 1:
                import torch.distributed as dist
                dist.destroy_process_group()
        return 0
    except Exception as e:  # loud, typed failure for the orchestrator
        print(f"FATAL: {type(e).__name__}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
