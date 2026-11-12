"""Snapshot metrics for CompositeRuntime: per-rank local values plus
all-reduced global values, so battery drivers can merge per-rank reports by
construction ('global' entries are IDENTICAL on every rank)."""
from __future__ import annotations

import torch
import torch.distributed as dist

from brainsim.rules.torch_rules import cascade as t_cascade


def _all_reduced(rt, values: list[float]) -> list[float]:
    t = torch.tensor(values, dtype=torch.float64,
                     device=rt.device if rt.device.type == "cuda" else "cpu")
    if rt.world_size > 1:
        dist.all_reduce(t)
    return [float(v) for v in t]


def snapshot(rt) -> dict:
    """Metrics dict for battery drivers. 'global' entries are all-reduced
    and therefore IDENTICAL on every rank; 'local' entries are per-rank."""
    edges: dict[str, dict] = {}
    reduced: list[float] = []
    order: list[tuple[str, str]] = []
    for name, es in rt.edges.items():
        n_local = es.n_synapses_local(rt.net)
        w_local = sum(
            float(rt.net.projections[p].weights.to(torch.float32).abs().sum())
            for p in es.parts
        )
        edges[name] = {"n_syn_local": n_local, "w_abs_sum_local": w_local}
        reduced += [float(n_local), w_local]
        order += [(name, "n_syn_global"), (name, "w_abs_sum_global")]

    casc = rt.edges["store_rec"]
    d_sum = d_n = 0.0
    for p in casc.parts:
        d = t_cascade.depth(casc.cascade[p], rt.config.cascade)
        d_sum += float(d.sum())
        d_n += d.numel()
    reduced += [d_sum, d_n]

    red = _all_reduced(rt, reduced)
    for (name, key), v in zip(order, red):
        edges[name][key] = int(round(v)) if key.startswith("n_") else v
    depth_sum, depth_n = red[-2], red[-1]

    return {
        "seed": rt.config.seed,
        "world_size": rt.world_size,
        "rank": rt.rank,
        "n_global": rt.config.n_global,
        "local_range": list(rt.net.local_range),
        "n_items": len(rt.items),
        "steps_total": rt.steps_total,
        "edges": edges,
        "cascade_mean_depth_global": depth_sum / max(1.0, depth_n),
        "n_synapses_global": int(sum(e["n_syn_global"] for e in edges.values())),
        "forecast_per_rank_bytes": rt.forecast["per_rank_total"],
    }


__all__ = ["snapshot"]
