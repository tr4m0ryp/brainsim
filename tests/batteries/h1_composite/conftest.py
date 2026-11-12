"""Shared fixtures of the H1 composite-arm tests (tiny scale)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from brainsim.batteries.h1 import composite_controls as controls
from brainsim.batteries.h1 import composite_report as report
from brainsim.batteries.h1 import composite_runner as runner
from brainsim.batteries.h1.composite_arm import (ACTIVE_PATH_EDGES,
                                                 base_cc, budget_rows,
                                                 forecast_table,
                                                 full_comp_config,
                                                 item_pattern,
                                                 smoke_comp_config,
                                                 tiny_comp_config)
from brainsim.batteries.h1.composite_tf import TorchTF
from brainsim.batteries.h3.budget import TOL, total_synapses
from brainsim.composite.full.runtime import CompositeRuntime
from brainsim.paths import RESULTS

ROOT = Path(__file__).resolve().parents[3]
H1_DIR = RESULTS / "h1"
SEED, SILENT = 0, (lambda *a, **k: None)

# the fly arm's real per-seed ratios (results/h1/fly_arm.json): far
# below the kill band -- the anti-premium case the tests pin
FLY_RATIOS_REAL = [0.4689537566456013, 0.0, 0.0, 0.0, 0.0]


@pytest.fixture(scope="module")
def tiny_cfg():
    return tiny_comp_config()


@pytest.fixture(scope="module")
def tiny_rt(tiny_cfg):
    return CompositeRuntime(base_cc(tiny_cfg, SEED), device="cpu")


@pytest.fixture(scope="module")
def partial_all(tiny_cfg):
    """One full tiny work list, executed once (6 instances x 1 seed)."""
    return runner.run_shard(tiny_cfg, [SEED], 0, 1, "all", log=SILENT,
                            item_log=SILENT)




def _canon(obj):
    """Stable serialization with volatile wall-clock fields stripped."""
    def strip(x):
        if isinstance(x, dict):
            return {k: strip(v) for k, v in sorted(x.items())
                    if k not in ("wall_s", "code_version")}
        if isinstance(x, list):
            return [strip(v) for v in x]
        return x
    return json.dumps(strip(obj), sort_keys=True)


def _edge_state(rt, name):
    parts = rt.edges[name].parts
    return {
        "indptr": [rt.net.projections[p].indptr.clone() for p in parts],
        "indices": torch.cat([rt.net.projections[p].indices.to(torch.int64)
                              for p in parts]),
        "weights": torch.cat([rt.net.projections[p].weights.to(torch.float32)
                              for p in parts]),
        "count": rt.edges[name].n_synapses_local(rt.net),
    }


@pytest.fixture
def canon():
    return _canon


@pytest.fixture
def edge_state():
    return _edge_state
