"""Battery H1 (wiring premium): fly-arm driver, controls, and protocol.

Contract: battery-h1-wiring-premium.md (FROZEN, hash in batteries/verdicts.py).
Verdict logic lives in brainsim.batteries.verdicts only.
"""
from brainsim.batteries.h1.circuit import MBCircuit
from brainsim.batteries.h1.fly_arm import run_fly_arm, run_seed
from brainsim.batteries.h1.mb_extract import MBGraph, Projection, load_mb_graph
from brainsim.batteries.h1.protocol import (ItemBank, ProtocolCfg,
                                            capacity_from_curve,
                                            run_mb_capacity)
from brainsim.batteries.h1.synthetic import make_synthetic_graph

__all__ = ["ItemBank", "MBCircuit", "MBGraph", "Projection", "ProtocolCfg",
           "capacity_from_curve", "load_mb_graph", "make_synthetic_graph",
           "run_fly_arm", "run_mb_capacity", "run_seed"]
