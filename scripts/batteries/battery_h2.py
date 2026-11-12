#!/usr/bin/env python3
"""Battery H2 driver CLI (contract: battery-h2-permanence-frontier.md, FROZEN).

Usage:
  # smoke: code paths only, print-only, never writes results.json
  python3 scripts/batteries/battery_h2.py --engine numpy --scale smoke

  # full scale, 8 concurrent single-GPU workers (one per CUDA device):
  CUDA_VISIBLE_DEVICES=$i python3 scripts/batteries/battery_h2.py \
      --engine torch --scale full --seeds 3 --arm-shard $i/8

  # merge the partials -> verdict -> results/h2/results.json
  python3 scripts/batteries/battery_h2.py --merge results/h2/partial_*.json

Determinism: the (arm, seed) -> RNG mapping never depends on the shard
layout, so 1/1 and 8/8 partitions produce identical numbers.
"""
from __future__ import annotations

import argparse
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", choices=["numpy", "torch"], default="numpy")
    ap.add_argument("--scale", choices=["smoke", "full"], default="smoke")
    ap.add_argument("--seeds", type=int, default=None,
                    help="number of seeds (full default 3, smoke default 2)")
    ap.add_argument("--out", default=None,
                    help="output dir (full default results/h2; required "
                         "for sharded smoke)")
    ap.add_argument("--arm-shard", default=None, metavar="I/N",
                    help="run only slice I of the (arm x seed) work list and "
                         "write partial_I_of_N.json")
    ap.add_argument("--merge", nargs="+", default=None, metavar="PARTIAL",
                    help="merge partial_*.json files; writes results.json only "
                         "when every partial is a full-scale run")
    ap.add_argument("--stats-dir", default=None,
                    help="override H01 stats artifact dir (default <data>/connectomes/h01/stats)")
    ap.add_argument("--device", default="auto", help="torch device (default auto)")
    ap.add_argument("--dtype", choices=["float32", "float16"], default="float32",
                    help="torch weight dtype (fp16 is the single-GPU memory plan)")
    args = ap.parse_args()

    from brainsim.batteries.h2 import runner

    try:
        if args.merge:
            runner.merge_files(args.merge)
            return 0
        n_seeds = args.seeds if args.seeds else (3 if args.scale == "full" else 2)
        if args.scale == "full" and n_seeds < 3:
            print("FATAL: the frozen contract requires >= 3 seeds at full scale")
            return 2
        cfg = runner.config_for(args.scale, args.engine,
                                torch_device=args.device, weight_dtype=args.dtype)
        runner.run(cfg, runner.default_seeds(n_seeds), out_dir=args.out,
                   arm_shard=args.arm_shard, stats_dir=args.stats_dir)
        return 0
    except Exception as e:  # loud, typed failure for the orchestrator
        print(f"FATAL: {type(e).__name__}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
