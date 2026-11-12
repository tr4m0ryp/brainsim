"""Per-synapse index recovery for plasticity rules on TorchCSRProjection (T5).

The torch rules mirror the numpy rule API: flat per-synapse gathers through
pre_idx/post_idx vectors. A CSR projection stores synapses grouped by
presynaptic id (stable within a row), so these helpers reconstruct the COO
index vectors once per projection; callers cache them (wiring is static
during a run, only weights and traces move).

The same helpers serve TorchNetwork projections and ShardedNetwork local
slices without change: a sharded slice is a TorchCSRProjection whose n_pre
is the GLOBAL neuron count and whose post ids are LOCAL to the rank, so
`synapse_pre_ids` comes out global (matching the fired-id spike exchange)
and `synapse_post_ids` local (matching the rank's own trace vectors).
"""
from __future__ import annotations

import torch

from brainsim.engine.torch_core.propagate import TorchCSRProjection


def synapse_pre_ids(proj: TorchCSRProjection) -> torch.Tensor:
    """int64 presynaptic id of every synapse, in weight-array order."""
    counts = proj.indptr[1:] - proj.indptr[:-1]
    return torch.repeat_interleave(
        torch.arange(proj.n_pre, dtype=torch.int64, device=proj.indptr.device), counts
    )


def synapse_post_ids(proj: TorchCSRProjection) -> torch.Tensor:
    """int64 postsynaptic id of every synapse, in weight-array order."""
    return proj.indices.to(torch.int64)


__all__ = ["synapse_pre_ids", "synapse_post_ids"]
