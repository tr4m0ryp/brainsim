"""Torch backend: same duck API as NumpyOps; the full-scale vehicle.
Cascade ops delegate to brainsim.rules.torch_rules.cascade (the mirror of the
numpy rule; this module is imported lazily by engines.get_ops, so importing
the battery package never pulls torch)."""
from __future__ import annotations

import numpy as np
import torch

from brainsim.rules.cascade import CascadeParams
from brainsim.rules.torch_rules import cascade as tcascade

_DTYPES = {"f4": torch.float32, "f2": torch.float16, "b1": torch.bool,
           "i8": torch.int64, "i4": torch.int32, "u1": torch.uint8}


class TorchOps:
    name = "torch"

    def __init__(self, seed: int, cascade_params: CascadeParams,
                 device: str = "auto", weight_dtype: str = "float32") -> None:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.gen = torch.Generator(device=self.device)
        self.gen.manual_seed(int(np.random.SeedSequence([seed, 0xD11]).generate_state(1)[0]))
        self.cascade_params = cascade_params
        self.wdtype = torch.float16 if weight_dtype == "float16" else torch.float32

    # -- transfer -------------------------------------------------------------
    def from_numpy(self, a: np.ndarray, dtype=None) -> torch.Tensor:
        # copy=True is load-bearing: on CPU, from_numpy + .to(device) can
        # alias the host array, and backend state must NEVER mutate the frozen
        # twin substrate (the ledger-invariance tests caught exactly this).
        t = torch.from_numpy(np.ascontiguousarray(a)).to(self.device, copy=True)
        if dtype is not None:
            t = t.to(_DTYPES[dtype] if isinstance(dtype, str) else dtype)
        elif t.dtype == torch.float32:
            t = t.to(self.wdtype)
        return t

    def to_numpy(self, t) -> np.ndarray:
        if isinstance(t, torch.Tensor):
            return t.detach().to("cpu").float().numpy() if t.is_floating_point() \
                else t.detach().to("cpu").numpy()
        return np.asarray(t)

    # -- construction ----------------------------------------------------------
    def zeros(self, shape, dtype="f4"):
        d = _DTYPES[dtype]
        if d == torch.float32:
            d = self.wdtype
        return torch.zeros(shape, dtype=d, device=self.device)

    def full(self, shape, value, dtype="f4"):
        d = _DTYPES[dtype]
        if d == torch.float32:
            d = self.wdtype
        return torch.full((shape,) if isinstance(shape, int) else shape, value,
                          dtype=d, device=self.device)

    def arange(self, n):
        return torch.arange(n, dtype=torch.int64, device=self.device)

    def concat(self, parts):
        return torch.cat(list(parts))

    # -- randomness --------------------------------------------------------------
    def randn(self, n):
        return torch.randn(n, generator=self.gen, dtype=self.wdtype, device=self.device)

    # -- indexing / arithmetic ------------------------------------------------------
    def as_index(self, x):
        return x.long()

    def take(self, x, idx):
        return x[idx.long() if isinstance(idx, torch.Tensor) else idx]

    def put(self, x, idx, values) -> None:
        x[idx.long()] = values

    def add_at(self, target, idx, values) -> None:
        target.index_add_(0, idx.long(), values.to(target.dtype))

    def bincount(self, idx, minlength):
        return torch.bincount(idx.long(), minlength=minlength)

    def cumsum(self, x):
        return torch.cumsum(x, dim=0)

    def repeat_interleave(self, x, repeats):
        return torch.repeat_interleave(x, repeats)

    def topk_idx(self, x, k):
        if k >= x.numel():
            return torch.arange(x.numel(), device=self.device)
        return torch.topk(x.float(), k).indices

    def exp(self, x):
        return torch.exp(x)

    def clip_min_(self, x, lo) -> None:
        x.clamp_(min=lo)

    def sum(self, x) -> float:
        return float(torch.sum(x.double()))

    def bool_mask(self, n, idx):
        m = torch.zeros(n, dtype=torch.bool, device=self.device)
        m[idx.long() if isinstance(idx, torch.Tensor) else idx] = True
        return m

    # -- cascade (torch_rules mirror on subsets) --------------------------------------
    def cascade_init(self, n_contacts):
        return torch.full((n_contacts,), self.cascade_params.n_levels - 1,
                          dtype=tcascade.STATE_DTYPE, device=self.device)

    def cascade_push_subset(self, state_subset):
        return tcascade.push(state_subset, self.cascade_params, self.gen)

    def cascade_pull_subset(self, state_subset):
        return tcascade.pull(state_subset, self.cascade_params, self.gen)

    def cascade_deepen_subset(self, state_subset, steps):
        return tcascade.deepen(state_subset, self.cascade_params, steps)

    def cascade_resistance(self, state):
        d = tcascade.depth(state, self.cascade_params).to(self.wdtype)
        return d / float(self.cascade_params.n_levels)


__all__ = ["TorchOps"]
