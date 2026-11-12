"""Arm 5: naive-but-hard Tsodyks-Feigelman sparse attractor at matched
synapse count.

Resource parity (recorded): N units = the MB circuit's dynamic neurons
(PN + KC + MBON + APL); directed synaptic entries = the circuit's total
anatomical synapse count (PN->KC + KC->MBON + KC<->APL summed syn_count).
The attractor graph is an undirected ER graph of directed_entries // 2
unordered pairs, each contributing a symmetric pair of entries.

Rule: streamed Tsodyks-Feigelman covariance learning on the sparse mask,
J_ij = sum_mu (xi_i - f)(xi_j - f) for edges only, assembled per checkpoint
from the streamed sufficient statistics n11 (co-activation counts per edge)
and c_i (per-unit activation counts):
    J_ij = n11_ij - f (c_i + c_j) + P f^2.
Recall: 50%-degraded cue (PARTIAL pattern, deletion-only -- the same
Formal-Model cue definition as the MB arms), 15 synchronous k-WTA
iterations (coding level fixed at f, ties break by index); overlap =
|y_final & xi| / (f N) on the layer itself (the readout IS the layer for an
attractor stage). Capacity uses the same curve machinery and 0.9 bar as
every other arm.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

from brainsim.batteries.h1.protocol import (ProtocolCfg, capacity_from_curve,
                                            checkpoint_grid, rng_for)

STREAM_TF_GRAPH, STREAM_TF_PATTERNS, STREAM_TF_CUES, STREAM_TF_EVAL = 30, 31, 32, 33
TF_ITERATIONS = 15


def sample_unique_pairs(rng: np.random.Generator, n_units: int,
                        count: int) -> tuple[np.ndarray, np.ndarray]:
    """`count` unique unordered pairs (i < j), deterministic in rng."""
    max_pairs = n_units * (n_units - 1) // 2
    if count > max_pairs:
        raise ValueError("requested more pairs than the graph can hold")
    chosen: list[np.ndarray] = []
    seen: set[int] = set()
    need = count
    while need > 0:
        i = rng.integers(0, n_units, size=int(need * 1.4) + 8)
        j = rng.integers(0, n_units, size=i.size)
        lo, hi = np.minimum(i, j), np.maximum(i, j)
        ok = lo != hi
        keys = lo[ok].astype(np.int64) * n_units + hi[ok]
        take: list[int] = []
        for k in keys.tolist():
            if k not in seen:
                seen.add(k)
                take.append(k)
                if len(seen) == count:
                    break
        if take:
            chosen.append(np.asarray(take, dtype=np.int64))
        need = count - len(seen)
    keys = np.concatenate(chosen)
    return (keys // n_units).astype(np.int32), (keys % n_units).astype(np.int32)


class TFAttractor:
    """Sparse TF network with streamed sufficient statistics."""

    def __init__(self, n_units: int, directed_entries: int, f: float,
                 rng: np.random.Generator):
        self.n = n_units
        self.f = f
        self.k = max(1, int(round(f * n_units)))
        lo, hi = sample_unique_pairs(rng, n_units, directed_entries // 2)
        rows = np.concatenate([lo, hi])
        cols = np.concatenate([hi, lo])
        order = np.lexsort((cols, rows))
        self.rows, self.cols = rows[order], cols[order]
        self.indptr = np.zeros(n_units + 1, dtype=np.int64)
        np.add.at(self.indptr, self.rows + 1, 1)
        np.cumsum(self.indptr, out=self.indptr)
        self.n11 = np.zeros(self.rows.size, dtype=np.int64)
        self.c = np.zeros(n_units, dtype=np.int64)
        self.p_stored = 0

    @property
    def n_directed_entries(self) -> int:
        return int(self.rows.size)

    def store(self, active_idx: np.ndarray) -> None:
        member = np.zeros(self.n, dtype=bool)
        member[active_idx] = True
        self.c[active_idx] += 1
        for i in active_idx.tolist():
            s, e = self.indptr[i], self.indptr[i + 1]
            hits = member[self.cols[s:e]]
            self.n11[s:e] += hits
        self.p_stored += 1

    def assemble(self) -> csr_matrix:
        data = (self.n11.astype(np.float64)
                - self.f * (self.c[self.rows] + self.c[self.cols])
                + self.p_stored * self.f * self.f)
        return csr_matrix((data, self.cols, self.indptr), shape=(self.n, self.n))

    def recall(self, cues: np.ndarray, j: csr_matrix) -> np.ndarray:
        """(batch, n) 0/1 cues -> (batch, n) boolean fixed-point estimates."""
        y = cues.astype(np.float64)
        for _ in range(TF_ITERATIONS):
            h = (j @ y.T).T
            order = np.argsort(-h, axis=1, kind="stable")[:, :self.k]
            y = np.zeros_like(y)
            np.put_along_axis(y, order, 1.0, axis=1)
        return y.astype(bool)


def run_tf_capacity(n_units: int, directed_entries: int, seed: int,
                    cfg: ProtocolCfg) -> dict:
    net = TFAttractor(n_units, directed_entries, cfg.f_kc,
                      rng_for(seed, STREAM_TF_GRAPH))
    k = net.k
    rp = rng_for(seed, STREAM_TF_PATTERNS)
    items = np.stack([rp.choice(n_units, size=k, replace=False)
                      for _ in range(cfg.p_cap)]).astype(np.int32)
    n_keep = k - int(round(0.5 * k))
    ps, overlaps = [], []
    below = 0
    for p_next in checkpoint_grid(cfg):
        for mu in range(net.p_stored, p_next):
            net.store(items[mu])
        j = net.assemble()
        re = rng_for(seed, STREAM_TF_EVAL, p_next)
        sel = (np.arange(p_next, dtype=np.int64) if p_next <= cfg.eval_sample
               else np.sort(re.choice(p_next, size=cfg.eval_sample,
                                      replace=False)).astype(np.int64))
        cues = np.zeros((sel.size, n_units), dtype=np.float64)
        for r, mu in enumerate(sel.tolist()):
            rc = rng_for(seed, STREAM_TF_CUES, mu)
            keep = rc.choice(items[mu], size=n_keep, replace=False)
            cues[r, keep] = 1.0
        y = net.recall(cues, j)
        hit = np.array([y[r, items[mu]].sum() / k
                        for r, mu in enumerate(sel.tolist())])
        ps.append(p_next)
        overlaps.append(float(hit.mean()))
        below = below + 1 if overlaps[-1] < cfg.early_stop_bar else 0
        if below >= cfg.early_stop_consecutive:
            break
    out = capacity_from_curve(ps, overlaps)
    out.update({"curve_P": ps, "curve_overlap": overlaps,
                "n_units": n_units, "n_directed_entries": net.n_directed_entries,
                "iterations": TF_ITERATIONS})
    return out


__all__ = ["TFAttractor", "TF_ITERATIONS", "run_tf_capacity",
           "sample_unique_pairs"]
