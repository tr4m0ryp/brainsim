"""Torch engine core (T5): device-agnostic sparse LIF on PyTorch, numerically
matched to the numpy reference engine (engine.lif / engine.propagate /
engine.network), with fp16-weight CSR projections (F24) and post-owned
neuron-range sharding over torch.distributed (F25).

torch is imported ONLY inside this package, and lazily even here (PEP 562
module __getattr__): `torch_core.memory` is pure byte accounting with no
torch import at all, so the bench --forecast mode works on boxes without
torch and the rest of brainsim never pays the import.
"""
from __future__ import annotations

import importlib

_EXPORTS = {
    "TorchLIFState": ".lif_step",
    "TorchCSRProjection": ".propagate",
    "TorchNetwork": ".network",
    "TorchPopulation": ".network",
    "resolve_device": ".network",
    "ShardedNetwork": ".shard",
    "partition_ranges": ".shard",
    "FastExchange": ".fastpath",
}


def __getattr__(name: str):
    if name in _EXPORTS:
        module = importlib.import_module(_EXPORTS[name], __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = sorted(_EXPORTS)
