"""On-device multi-step loops for ShardedNetwork and TorchNetwork.

Per step these do the SAME arithmetic as step(), in the same order --
zeroed current buffer += each projection's due ring slot (in projection
insertion order) += drive, in-place I add, the same lif_step, the same
nonzero -> exchange -> deliver_fired sequence -- so rasters are
bit-identical to a step() loop with the same drive values (in-place adds
round exactly like their out-of-place forms). What they do NOT do per
step: download spikes to host, convert to numpy, append to spike_buffer,
allocate current buffers, or clone ring rows. Spike counts accumulate in a
device int64 tensor and cross the bus once at the end; with
record='raster' the full bool raster crosses once too.

Both return (counts, raster): counts is an int64 numpy array of per-step
spike totals (LOCAL for the sharded net -- all_reduce for global totals,
as the bench does; summed over populations for TorchNetwork), raster is
None unless record='raster', then a (steps, n_local) bool array for the
sharded net or {pop_name: (steps, n) bool array} for TorchNetwork.
"""
from __future__ import annotations

import torch

from brainsim.engine.torch_core.fastpath.drive import as_local_drive, as_pop_drive
from brainsim.engine.torch_core.lif_step import step as lif_step


def _norm_steps(steps) -> int:
    steps = int(steps)
    if steps < 0:
        raise ValueError(f"steps must be >= 0, got {steps}")
    return steps


def _norm_record(record) -> str | None:
    if record not in (None, "raster"):
        raise ValueError(f"record must be None or 'raster', got {record!r}")
    return record


def run_sharded(net, steps, drive=None, record=None):
    """Fast loop for ShardedNetwork; exchange via net's FastExchange."""
    steps, record = _norm_steps(steps), _norm_record(record)
    device, n_local, lo = net.device, net.n_local, net.lo
    drive_fn = as_local_drive(drive, n_local, device)
    exchange = net._ensure_fast_exchange()
    projs = list(net.projections.values())
    state, params = net.state, net.params

    counts = torch.zeros(steps, dtype=torch.int64, device=device)
    raster = (
        torch.empty((steps, n_local), dtype=torch.bool, device=device)
        if record == "raster"
        else None
    )
    currents = torch.zeros(n_local, dtype=torch.float32, device=device)

    for i in range(steps):
        currents.zero_()
        for proj in projs:
            proj.accumulate_current_into(currents)
        if drive_fn is not None:
            currents += drive_fn(i)

        state.I += currents
        spikes = lif_step(state, params)

        fired = torch.nonzero(spikes).flatten() + lo
        fired = exchange.exchange(fired, net.t)
        for proj in projs:
            proj.deliver_fired(fired)

        counts[i] = spikes.sum()
        if raster is not None:
            raster[i] = spikes
        net.t += 1

    return counts.cpu().numpy(), (None if raster is None else raster.cpu().numpy())


def run_network(net, steps, drive=None, record=None):
    """Fast loop for TorchNetwork. Does NOT populate net.spike_buffer (that
    per-step host copy is part of the overhead run() removes)."""
    steps, record = _norm_steps(steps), _norm_record(record)
    if not net.populations:
        raise ValueError("run() needs at least one population")
    device = net.device
    pops = list(net.populations.items())
    projs = list(net.projections.values())
    drive_fn = as_pop_drive(drive, net)

    counts = torch.zeros(steps, dtype=torch.int64, device=device)
    rasters = (
        {
            name: torch.empty((steps, pop.n), dtype=torch.bool, device=device)
            for name, pop in pops
        }
        if record == "raster"
        else None
    )
    currents = {
        name: torch.zeros(pop.n, dtype=torch.float32, device=device)
        for name, pop in pops
    }

    for i in range(steps):
        for buf in currents.values():
            buf.zero_()
        for proj in projs:
            proj.csr.accumulate_current_into(currents[proj.post])
        if drive_fn is not None:
            for name, extra in drive_fn(i).items():
                currents[name] += extra

        for name, pop in pops:
            pop.state.I += currents[name]
        spikes = {name: lif_step(pop.state, pop.params) for name, pop in pops}
        for proj in projs:
            proj.csr.deliver(spikes[proj.pre])

        total = None
        for name, _ in pops:
            s = spikes[name].sum()
            total = s if total is None else total + s
        counts[i] = total
        if rasters is not None:
            for name, _ in pops:
                rasters[name][i] = spikes[name]
        net.t += 1

    counts_np = counts.cpu().numpy()
    if rasters is None:
        return counts_np, None
    return counts_np, {name: r.cpu().numpy() for name, r in rasters.items()}


__all__ = ["run_network", "run_sharded"]
