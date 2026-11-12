"""Drive normalization for the fast path: device tensors only, validated.

The fast path never copies drive across devices or dtypes -- an implicit
as_tensor + H2D per step is floor contributor #3, exactly what run()
exists to remove -- so every drive tensor must already be fp32 on the
net's device with the exact expected shape, and anything else raises.
Accepted specs:

- ShardedNetwork.run: None | (n_local,) tensor reused per step | callable
  (step_idx) -> such a tensor. NOTE: this is the LOCAL slice, unlike
  step(external_I=...) which takes the GLOBAL vector and slices it; no
  rank should materialize global drive in the fast path.
- TorchNetwork.run: None | tensor (single-population nets only) |
  {pop_name: tensor} | callable(step_idx) -> tensor or dict. Additions
  follow the dict's iteration order, mirroring step(external_I=...).

Throughput benches generate drive in place per step (e.g.
buf.normal_(generator=...) inside the callable); that stream differs
numerically from the numpy-path drive, which is fine for benches --
equivalence tests pass explicit identical tensors instead.
"""
from __future__ import annotations

from typing import Callable

import torch


def _check_tensor(t, n: int, device: torch.device, what: str) -> torch.Tensor:
    if not torch.is_tensor(t):
        raise TypeError(f"{what}: expected a torch.Tensor, got {type(t).__name__}")
    if t.device.type != device.type:
        raise ValueError(
            f"{what}: tensor on {t.device.type!r} but the net is on "
            f"{device.type!r}; the fast path never copies drive across "
            "devices, allocate it on the net's device"
        )
    if t.dtype != torch.float32:
        raise ValueError(f"{what}: dtype {t.dtype}, expected torch.float32")
    if t.shape != (n,):
        raise ValueError(f"{what}: shape {tuple(t.shape)}, expected ({n},)")
    return t


def as_local_drive(
    drive, n_local: int, device: torch.device
) -> Callable[[int], torch.Tensor] | None:
    """ShardedNetwork.run spec -> callable(step_idx) -> LOCAL fp32 tensor."""
    if drive is None:
        return None
    if callable(drive) and not torch.is_tensor(drive):
        return lambda i: _check_tensor(drive(i), n_local, device, f"drive({i})")
    t = _check_tensor(drive, n_local, device, "drive")
    return lambda i: t


def _as_pop_dict(d, populations: dict, device: torch.device, label: str) -> dict:
    if torch.is_tensor(d):
        if len(populations) != 1:
            raise ValueError(
                f"{label}: a bare tensor drive needs exactly one population, "
                f"this net has {len(populations)}; pass a dict instead"
            )
        ((name, pop),) = populations.items()
        return {name: _check_tensor(d, pop.n, device, label)}
    if not isinstance(d, dict):
        raise TypeError(
            f"{label}: expected a torch.Tensor or a dict of them, got {type(d).__name__}"
        )
    out = {}
    for name, t in d.items():
        if name not in populations:
            raise ValueError(f"{label}: unknown population {name!r}")
        out[name] = _check_tensor(t, populations[name].n, device, f"{label}[{name!r}]")
    return out


def as_pop_drive(drive, net) -> Callable[[int], dict] | None:
    """TorchNetwork.run spec -> callable(step_idx) -> {pop_name: tensor}."""
    if drive is None:
        return None
    populations, device = net.populations, net.device
    if callable(drive) and not torch.is_tensor(drive):
        return lambda i: _as_pop_dict(drive(i), populations, device, f"drive({i})")
    const = _as_pop_dict(drive, populations, device, "drive")
    return lambda i: const


__all__ = ["as_local_drive", "as_pop_drive"]
