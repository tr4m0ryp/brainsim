#!/usr/bin/env python3
"""Whole-fly engine gate on the brainsim torch engine (CPU).

  bash scripts/data/fetch_shiu_model.sh       # connectome as shipped by Shiu et al.
  python scripts/gates/fly_gate_brainsim.py   # writes results/fly-gate/brainsim/gate.json

Parameter mapping, event modes and the five known divergences from Shiu et
al.'s Brian2 model are documented in brainsim.validate.whole_fly.mapping.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from brainsim.paths import RESULTS
from brainsim.validate.whole_fly import run_gate


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dt", type=float, default=1.0, help="step in ms")
    ap.add_argument("--i-ev", choices=["charge", "g"], default="g",
                    help="Poisson event current: g-dynamics 68.75 mV (passed) "
                         "or charge-equivalent 275 mV (failed dose-response)")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--note", default="", help="extra provenance appended to notes")
    ap.add_argument("--out", default=str(RESULTS / "fly-gate" / "brainsim" / "gate.json"))
    args = ap.parse_args()

    report = run_gate(dt=args.dt, mode=args.i_ev, trials=args.trials, note=args.note)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print("GATE", "PASSED" if report["gate_passed"] else "FAILED")
    for c in report.get("checks", []):
        print(f"  [{'x' if c['pass'] else ' '}] {c['name']}: {c['observed']}")
    if not report["gate_passed"]:
        print("  reason:", report.get("reason", "see checks above"))
    return 0 if report["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
