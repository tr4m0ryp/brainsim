"""Fixed-overhead fast path for the torch engine (B1 floor attack).

The 8x H200 benchmark fit step_time = a + E_step/T with a = 4.16 ms/step;
at biological rates (0.5-5 Hz) the a-floor is ~73% of full-scale cost. The
floor is per-step work independent of event count: the forced device-to-
host raster download in step(), the TWO all_gathers of the fired exchange,
the host-side numpy drive + H2D copy, and per-step allocations (currents
zeros, ring clones). run() removes exactly those four; step() and every
existing signature are untouched, the fast path is opt-in.

Arithmetic is op-for-op identical to a step() loop -- same values added in
the same order, the same lif_step -- so fast and slow paths produce
bit-identical fp32 spike rasters on the same device (gated by
tests/brainsim/test_torch_fastpath*.py). Data-dependent sparse ops
(nonzero, repeat_interleave inside delivery) remain; eliminating those is
the CUDA-graph follow-up that the fixed-shape exchange enables, NOT done
here (needs a GPU to validate).
"""
from __future__ import annotations

from brainsim.engine.torch_core.fastpath.exchange import FastExchange
from brainsim.engine.torch_core.fastpath.runners import run_network, run_sharded

__all__ = ["FastExchange", "run_network", "run_sharded"]
