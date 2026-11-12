"""`python -m brainsim.bench --smoke`: runs a fixed tiny recurrent LIF network
for N steps and reports steps/sec and synaptic events/sec (T11 gate #1: the
throughput benchmark). Default backend is the numpy CPU path (the reference
point); --backend torch runs the same network, built from the same seed and
draw order, on the torch engine (engine.torch_core) with --device
{auto,cpu,cuda} and optional --fp16 weights. --forecast N_SYN prints the F24
memory projection per rank for 1..8 ranks and allocates nothing.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any

import numpy as np

from brainsim.engine.backend import describe_backend
from brainsim.engine.lif import LIFParams
from brainsim.engine.network import Network
from brainsim.engine.propagate import CSRProjection


def sample_connectivity(
    n: int, k: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The bench's fixed random topology: k random pre per post neuron,
    lognormal weights, 80/20 excitatory/inhibitory. ONE draw order, shared by
    both backends, so numpy and torch run the identical table.
    """
    k = max(1, min(k, n))
    pre = np.concatenate([rng.choice(n, size=k, replace=False) for _ in range(n)])
    post = np.repeat(np.arange(n), k)
    weight = rng.lognormal(-1.0, 0.5, size=pre.size)
    sign = np.where(rng.random(pre.size) < 0.8, 1, -1).astype(np.int8)
    return pre, post, weight, sign


def build_network(n: int, k: int, seed: int = 0) -> Network:
    rng = np.random.default_rng(seed)
    net = Network(seed=seed)
    net.add_population("pop", n, LIFParams())
    pre, post, weight, sign = sample_connectivity(n, k, rng)
    csr = CSRProjection.from_table(pre, post, weight, sign, n_pre=n, n_post=n, delay_steps=1)
    net.add_projection("rec", "pop", "pop", csr)
    return net


def build_torch_network(
    n: int, k: int, seed: int = 0, device: str = "auto", fp16: bool = False
):
    """Same topology and seed as build_network on the torch engine. torch is
    imported here, not at module top: the numpy path must run without it.
    """
    from brainsim.engine.torch_core.network import TorchNetwork
    from brainsim.engine.torch_core.propagate import TorchCSRProjection

    rng = np.random.default_rng(seed)
    net = TorchNetwork(seed=seed, device=device)
    net.add_population("pop", n, LIFParams())
    pre, post, weight, sign = sample_connectivity(n, k, rng)
    csr = TorchCSRProjection.from_table(
        pre, post, weight, sign, n_pre=n, n_post=n, delay_steps=1, fp16=fp16, device=net.device
    )
    net.add_projection("rec", "pop", "pop", csr)
    return net


def run_bench(
    n: int,
    k: int,
    steps: int,
    seed: int = 0,
    backend: str = "numpy",
    device: str = "auto",
    fp16: bool = False,
    fast: bool = False,
) -> dict[str, Any]:
    """fast=True (torch backend only) times TorchNetwork.run(): the on-device
    loop with drive drawn in place per step from a torch.Generator seeded
    like the numpy stream (seed+1) -- a DIFFERENT bit stream than the numpy
    drive, fine for throughput; raster equivalence is covered by the tests,
    which pass explicit identical drive tensors instead.
    """
    if fast and backend != "torch":
        raise ValueError("fast=True applies to backend='torch' only")
    if backend == "torch":
        net = build_torch_network(n, k, seed, device=device, fp16=fp16)
        backend_desc = f"torch-{net.device.type} ({'fp16' if fp16 else 'fp32'} weights)"
    else:
        net = build_network(n, k, seed)
        backend_desc = describe_backend()

    if fast:
        import torch

        gen = torch.Generator(device=net.device)
        gen.manual_seed(seed + 1)
        buf = torch.empty(n, dtype=torch.float32, device=net.device)
        if net.device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        counts, _ = net.run(steps, drive=lambda i: buf.normal_(2.0, 1.0, generator=gen))
        elapsed = time.perf_counter() - start
        total_spikes = int(counts.sum())
    else:
        drive_rng = np.random.default_rng(seed + 1)
        start = time.perf_counter()
        total_spikes = 0
        for _ in range(steps):
            external = {"pop": drive_rng.normal(2.0, 1.0, size=n)}
            spikes = net.step(external_I=external)
            total_spikes += int(spikes["pop"].sum())
        elapsed = time.perf_counter() - start

    synaptic_events = total_spikes * k
    report = {
        "n": n,
        "k": k,
        "steps": steps,
        "elapsed_s": elapsed,
        "steps_per_sec": steps / elapsed if elapsed > 0 else float("inf"),
        "spikes": total_spikes,
        "synaptic_events": synaptic_events,
        "synaptic_events_per_sec": synaptic_events / elapsed if elapsed > 0 else float("inf"),
        "backend": backend_desc,
    }
    if fast:
        report["fast"] = True
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m brainsim.bench")
    parser.add_argument("--smoke", action="store_true", help="tiny fixed-size run for CI")
    parser.add_argument("--size", type=int, default=None, help="population size (overrides --smoke default)")
    parser.add_argument("--k", type=int, default=20, help="recurrent in-degree")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--backend", choices=("numpy", "torch"), default="numpy")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto",
                        help="torch backend only")
    parser.add_argument("--fp16", action="store_true", help="fp16 projection weights (torch backend only)")
    parser.add_argument("--fast", action="store_true",
                        help="torch backend only: run() fast path -- on-device loop, "
                        "single-collective fixed-capacity exchange, on-device drive "
                        "(drive stream differs numerically from the numpy path)")
    parser.add_argument("--fast-cap", type=int, default=None, metavar="N",
                        help="with --fast, per-rank fired capacity of the exchange "
                        "buffers (default ceil(n/world), which cannot overflow); "
                        "size to the expected max fired per rank per step -- "
                        "overflow raises, never truncates")
    parser.add_argument("--forecast", type=float, default=None, metavar="N_SYN",
                        help="print projected GB/rank for 1..8 ranks at N_SYN synapses "
                        "(neurons = --size if given, else N_SYN/--k); allocates nothing")
    parser.add_argument("--json", type=str, default=None, metavar="PATH",
                        help="also write the report dict as JSON")
    parser.add_argument("--init", type=str, default=None, metavar="URL",
                        help="dist init method for multi-rank (e.g. file:///tmp/rdzv); "
                        "default env:// (torchrun)")
    args = parser.parse_args(argv)

    if args.backend == "numpy" and (args.fp16 or args.device != "auto"):
        parser.error("--device/--fp16 apply to --backend torch")
    if args.fast and args.backend != "torch":
        parser.error("--fast applies to --backend torch")
    if args.fast_cap is not None and not args.fast:
        parser.error("--fast-cap applies with --fast")

    if args.forecast is not None:
        # torch-free by construction: torch_core.memory is pure accounting.
        from brainsim.engine.torch_core import memory as tc_memory

        n_syn = int(args.forecast)
        n_neurons = args.size if args.size is not None else max(1, n_syn // args.k)
        print(tc_memory.forecast_table(n_syn, n_neurons))
        return 0

    n = args.size if args.size is not None else (200 if args.smoke else 1000)
    steps = 50 if (args.smoke and args.size is None) else args.steps

    if args.backend == "torch" and int(os.environ.get("WORLD_SIZE", "1")) > 1:
        # torchrun path: rank-local procedural build, NCCL/gloo exchange.
        from brainsim.bench.sharded import run_sharded_bench, write_report

        report = run_sharded_bench(n, args.k, steps, args.seed,
                                   device=args.device, fp16=args.fp16,
                                   init_method=args.init, fast=args.fast,
                                   fast_cap=args.fast_cap)
        if report is None:  # non-zero ranks
            return 0
        if args.json:
            write_report(report, args.json)
        print(f"backend: {report['backend']} world={report['world_size']}")
        print(f"n={report['n']} k={report['k']} steps={report['steps']} "
              f"elapsed={report['elapsed_s']:.4f}s")
        print(f"steps/sec: {report['steps_per_sec']:.2f}")
        print(f"synaptic events/sec: {report['synaptic_events_per_sec']:.2f}")
        return 0

    report = run_bench(n, args.k, steps, args.seed, backend=args.backend,
                       device=args.device, fp16=args.fp16, fast=args.fast)
    if args.json:
        from brainsim.bench.sharded import write_report as _wr
        _wr(report, args.json)
    print(f"backend: {report['backend']}")
    print(f"n={report['n']} k={report['k']} steps={report['steps']} elapsed={report['elapsed_s']:.4f}s")
    print(f"steps/sec: {report['steps_per_sec']:.2f}")
    print(f"synaptic events/sec: {report['synaptic_events_per_sec']:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
