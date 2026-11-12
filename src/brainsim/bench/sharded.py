"""Multi-rank throughput bench (B1). Launched under torchrun: every rank
builds ONLY its own post-owned slice of a synthetic k-regular network,
procedurally and in chunks (a global COO table at 5e10 synapses would be
hundreds of GB; the slice arrives on device as int32 indices + fp16 weights,
~6 B/synapse per F24). Lockstep steps with the NCCL/gloo exchange, then rank
0 writes the JSON report.

Drive is drawn rank-locally (seed + rank): fine for throughput, not raster-
identical to a single-rank run; numerical equivalence is covered by the
tests, not the bench. fast=True times ShardedNetwork.run() instead of the
step() loop -- on-device drive from a per-rank torch.Generator (same seed
arithmetic, DIFFERENT bit stream than the numpy drive; throughput only),
single-collective fixed-capacity exchange (fast_cap; the default
ceil(n/world) cannot overflow), no per-step host sync.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from brainsim.engine.lif import LIFParams
from brainsim.engine.torch_core.shard import ShardedNetwork, default_backend

CHUNK_SYN = 500_000_000  # ~7 GB host transient per chunk at 13 B/syn COO


def _add_chunked_synthetic(
    net: ShardedNetwork, k: int, seed: int, fp16: bool
) -> None:
    """k random pre (uniform over the GLOBAL index space) per local post
    neuron, lognormal weights, 80/20 exc/inh -- same statistics as the
    single-rank bench topology, generated per post-chunk.
    """
    lo, hi = net.local_range
    rng = np.random.default_rng(seed + 7919 * net.rank)
    rows_per_chunk = max(1, CHUNK_SYN // k)
    c = 0
    for start in range(lo, hi, rows_per_chunk):
        stop = min(start + rows_per_chunk, hi)
        rows = stop - start
        pre = rng.integers(0, net.n_global, size=rows * k, dtype=np.int64)
        post = np.repeat(np.arange(start, stop, dtype=np.int64), k)
        weight = rng.lognormal(-1.0, 0.5, size=pre.size).astype(np.float32)
        sign = np.where(rng.random(pre.size) < 0.8, 1, -1).astype(np.int8)
        net.add_projection(f"rec{c}", pre, post, weight, sign,
                           delay_steps=1, fp16=fp16)
        c += 1


def run_sharded_bench(
    n: int, k: int, steps: int, seed: int, device: str, fp16: bool,
    init_method: str | None = None, fast: bool = False,
    fast_cap: int | None = None,
) -> dict[str, Any] | None:
    """Returns the report dict on rank 0, None elsewhere. init_method
    file://... sidesteps torchrun's TCPStore (segfaults under torch 2.2 on
    our development machine); None uses the env:// default (torchrun).
    """
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    use_cuda = device in ("auto", "cuda") and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(rank % max(1, torch.cuda.device_count()))
    backend = default_backend("cuda" if use_cuda else "cpu")
    if init_method:
        dist.init_process_group(backend, init_method=init_method,
                                rank=rank, world_size=world)
    else:
        dist.init_process_group(backend)
    try:
        net = ShardedNetwork(n, LIFParams(), rank=rank, world_size=world,
                             device=device, fast_exchange_cap=fast_cap)
        _add_chunked_synthetic(net, k, seed, fp16)
        n_local = net.n_local
        lo, hi = net.local_range

        if fast:
            gen = torch.Generator(device=net.device)
            gen.manual_seed(seed + 1 + rank)
            buf = torch.empty(n_local, dtype=torch.float32, device=net.device)
            if net.device.type == "cuda":
                torch.cuda.synchronize()
            dist.barrier()
            start = time.perf_counter()
            counts, _ = net.run(
                steps, drive=lambda i: buf.normal_(2.0, 1.0, generator=gen))
            dist.barrier()
            elapsed = time.perf_counter() - start
            local_spikes = int(counts.sum())
        else:
            drive_rng = np.random.default_rng(seed + 1 + rank)
            ext = np.zeros(n, dtype=np.float32)
            dist.barrier()
            start = time.perf_counter()
            local_spikes = 0
            for _ in range(steps):
                ext[lo:hi] = drive_rng.normal(2.0, 1.0, size=n_local)
                local_spikes += int(net.step(external_I=ext).sum())
            dist.barrier()
            elapsed = time.perf_counter() - start

        agg_device = net.device if net.device.type == "cuda" else torch.device("cpu")
        tot = torch.tensor([local_spikes, net.synapse_count()],
                           dtype=torch.int64, device=agg_device)
        dist.all_reduce(tot)
        total_spikes, total_syn = int(tot[0].item()), int(tot[1].item())
        if rank != 0:
            return None
        syn_events = total_spikes * k
        report = {
            "n": n,
            "k": k,
            "steps": steps,
            "elapsed_s": elapsed,
            "steps_per_sec": steps / elapsed if elapsed > 0 else float("inf"),
            "spikes": total_spikes,
            "synaptic_events": syn_events,
            "synaptic_events_per_sec": syn_events / elapsed if elapsed > 0 else float("inf"),
            "backend": f"torch-{net.device.type}-sharded "
                       f"({'fp16' if fp16 else 'fp32'} weights, "
                       f"{dist.get_backend()})",
            "world_size": world,
            "n_synapses": total_syn,  # all-reduced; equals n*k by construction
        }
        if fast:
            report["fast"] = True
        return report
    finally:
        dist.destroy_process_group()


def write_report(report: dict[str, Any], path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=1))


__all__ = ["run_sharded_bench", "write_report"]
