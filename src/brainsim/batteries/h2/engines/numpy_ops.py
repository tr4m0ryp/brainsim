"""Reference numpy backend for the H2 protocol (works now, everywhere).
Cascade ops delegate to the canonical brainsim.rules.cascade on gathered
subsets (semantically identical to masked full-array calls, but drawing
|subset| uniforms instead of nnz -- required at 1e9 contacts)."""
from __future__ import annotations

import numpy as np

from brainsim.rules import cascade


class NumpyOps:
    name = "numpy"

    def __init__(self, seed: int, cascade_params: cascade.CascadeParams) -> None:
        self.rng = np.random.default_rng(np.random.SeedSequence([seed, 0xD11]))
        self.cascade_params = cascade_params

    # -- transfer -------------------------------------------------------------
    def from_numpy(self, a: np.ndarray, dtype=None) -> np.ndarray:
        out = np.array(a, copy=True)
        return out.astype(dtype) if dtype is not None else out

    def to_numpy(self, t) -> np.ndarray:
        return np.asarray(t)

    # -- construction ----------------------------------------------------------
    def zeros(self, shape, dtype="f4"):
        return np.zeros(shape, dtype=np.dtype(dtype))

    def full(self, shape, value, dtype="f4"):
        return np.full(shape, value, dtype=np.dtype(dtype))

    def arange(self, n):
        return np.arange(n, dtype=np.int64)

    def concat(self, parts):
        return np.concatenate(parts)

    # -- randomness (backend stream: turnover + cascade only) -------------------
    def randn(self, n) -> np.ndarray:
        return self.rng.standard_normal(n, dtype=np.float32)

    # -- indexing / arithmetic ---------------------------------------------------
    def as_index(self, x):
        return x

    def take(self, x, idx):
        return x[idx]

    def put(self, x, idx, values) -> None:
        x[idx] = values

    def add_at(self, target, idx, values) -> None:
        np.add.at(target, idx, values)

    def bincount(self, idx, minlength):
        return np.bincount(idx, minlength=minlength)

    def cumsum(self, x):
        return np.cumsum(x)

    def repeat_interleave(self, x, repeats):
        return np.repeat(x, repeats)

    def topk_idx(self, x, k):
        if k >= x.size:
            return np.arange(x.size)
        return np.argpartition(-x, k - 1)[:k]

    def exp(self, x):
        return np.exp(x)

    def clip_min_(self, x, lo) -> None:
        np.maximum(x, lo, out=x)

    def sum(self, x) -> float:
        return float(np.sum(x, dtype=np.float64))

    def bool_mask(self, n, idx):
        m = np.zeros(n, dtype=bool)
        m[idx] = True
        return m

    # -- cascade (canonical numpy rules on subsets) -------------------------------
    def cascade_init(self, n_contacts) -> np.ndarray:
        return np.full(n_contacts, self.cascade_params.n_levels - 1, dtype=np.uint8)

    def cascade_push_subset(self, state_subset):
        return cascade.push(state_subset, self.cascade_params, self.rng)

    def cascade_pull_subset(self, state_subset):
        return cascade.pull(state_subset, self.cascade_params, self.rng)

    def cascade_deepen_subset(self, state_subset, steps):
        return cascade.deepen(state_subset, self.cascade_params, steps)

    def cascade_resistance(self, state) -> np.ndarray:
        """depth/n_levels in [1/n, 1]: the leak/interference protection factor."""
        d = cascade.depth(state, self.cascade_params).astype(np.float32)
        return d / np.float32(self.cascade_params.n_levels)


__all__ = ["NumpyOps"]
