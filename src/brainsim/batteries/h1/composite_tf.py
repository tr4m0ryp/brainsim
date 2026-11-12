"""Arm 5 at composite scale: Tsodyks-Feigelman sparse attractor at matched
synapse count, torch single-device (packed execution class).

Resource parity (fly tf_arm rule, restated for the composite arm): N units
= the arm's loop-stage neurons (input+separation+formation+output as
configured); directed entries = the composite arm's total synapse count
rounded to pairs. f = the formation coding level (the stage under test's
sparseness, mirroring the fly's f_kc).

Semantics are tf_arm.py's, kept bit-compatible where the math is shared:
streamed TF sufficient statistics J_ij = n11_ij - f (c_i + c_j) + P f^2 on
a fixed sparse mask, deletion-only 50% cues, 15 synchronous k-WTA
iterations with stable index tie-break, capacity through the same 0.9-bar
curve machinery. Differences, documented: (a) pairs are drawn i.i.d. WITH
replacement (the composite build's own convention; ~2% duplicate pairs act
as double contacts; the directed-entry count is matched exactly), where the
fly sampled unique pairs -- a set of 6e9 keys does not exist; (b) n11 is
int16 (p_cap < 32768 asserted) with an int32 fallback probed at
construction, and J values are materialized per checkpoint into a reusable
fp16 (cuda) / fp32 (cpu) buffer, so full scale fits one H200:
1.22e10 x (4 B col + 2 B n11 + 2 B values) ~ 98 GB.

All randomness is keyed via h1.protocol.rng_for streams 60-63 (host numpy,
device-independent).
"""
from __future__ import annotations

import time

import numpy as np
import torch

from brainsim.batteries.h1.protocol import (OVERLAP_BAR, capacity_from_curve,
                                            checkpoint_grid, rng_for)

STREAM_TF_GRAPH, STREAM_TF_PATTERNS, STREAM_TF_CUES, STREAM_TF_EVAL = 60, 61, 62, 63
TF_ITERATIONS = 15          # tf_arm parity
PAIR_CHUNK = 1 << 25        # pairs generated per keyed chunk
FLAT_BLOCK = 1 << 27        # max gathered entries per store/values block


def _n11_dtype(device: torch.device) -> torch.dtype:
    """int16 primary; probe index_add_ support and fall back to int32."""
    try:
        t = torch.zeros(4, dtype=torch.int16, device=device)
        t.index_add_(0, torch.tensor([1, 2], device=device),
                     torch.ones(2, dtype=torch.int16, device=device))
        return torch.int16
    except RuntimeError:
        return torch.int32


class TorchTF:
    """Sparse TF network with streamed sufficient statistics on one device."""

    def __init__(self, n_units: int, directed_entries: int, f: float,
                 seed: int, device: str = "cpu"):
        self.n = int(n_units)
        self.f = float(f)
        self.k = max(1, int(round(f * n_units)))
        self.seed = int(seed)
        self.device = torch.device(device)
        self.p_stored = 0
        n_pairs = directed_entries // 2
        self.n_pairs = n_pairs

        counts = torch.zeros(self.n, dtype=torch.int64, device=self.device)
        for c0, i, j in self._pair_chunks(n_pairs):
            rows = torch.cat([i, j])
            counts.index_add_(0, rows, torch.ones_like(rows))
        self.indptr = torch.zeros(self.n + 1, dtype=torch.int64, device=self.device)
        self.indptr[1:] = torch.cumsum(counts, 0)
        e = int(self.indptr[-1])
        self.col = torch.empty(e, dtype=torch.int32, device=self.device)
        fill = self.indptr[:-1].clone()
        for c0, i, j in self._pair_chunks(n_pairs):
            rows = torch.cat([i, j])
            cols = torch.cat([j, i])
            rows_s, order = torch.sort(rows, stable=True)
            cols_s = cols[order]
            new = torch.ones_like(rows_s, dtype=torch.bool)
            new[1:] = rows_s[1:] != rows_s[:-1]
            run_id = torch.cumsum(new.to(torch.int64), 0) - 1
            run_start = torch.nonzero(new).flatten()
            within = torch.arange(rows_s.numel(), device=self.device) - run_start[run_id]
            self.col[fill[rows_s] + within] = cols_s.to(torch.int32)
            fill.index_add_(0, rows_s, torch.ones_like(rows_s))
        dt = _n11_dtype(self.device)
        self.n11 = torch.zeros(e, dtype=dt, device=self.device)
        self.c = torch.zeros(self.n, dtype=torch.int32, device=self.device)
        self._vals: torch.Tensor | None = None

    def _pair_chunks(self, n_pairs: int):
        """Deterministic keyed (i, j) chunks, i != j (unordered pair drawn
        as an ordered pair with replacement; duplicates are double
        contacts, see module docstring)."""
        done = 0
        chunk_id = 0
        while done < n_pairs:
            m = min(PAIR_CHUNK, n_pairs - done)
            rng = rng_for(self.seed, STREAM_TF_GRAPH, chunk_id)
            i = rng.integers(0, self.n, size=m, dtype=np.int64)
            j = rng.integers(0, self.n, size=m, dtype=np.int64)
            bad = i == j
            while bad.any():
                j[bad] = rng.integers(0, self.n, size=int(bad.sum()), dtype=np.int64)
                bad = i == j
            yield (chunk_id,
                   torch.as_tensor(i, device=self.device),
                   torch.as_tensor(j, device=self.device))
            done += m
            chunk_id += 1

    @property
    def n_directed_entries(self) -> int:
        return int(self.col.shape[0])

    def _flat_ranges(self, active: torch.Tensor):
        """Blocks of flat CSR entry indices covering the active rows, each
        block bounded by FLAT_BLOCK gathered entries."""
        starts = self.indptr[active]
        lens = self.indptr[active + 1] - starts
        cum = torch.cumsum(lens, 0)
        lo = 0
        while lo < active.numel():
            budget = (cum - (cum[lo - 1] if lo else 0)) <= FLAT_BLOCK
            hi = max(lo + 1, int(budget.nonzero().flatten().max().item() + 1)) \
                if budget.any() else lo + 1
            s, ln = starts[lo:hi], lens[lo:hi]
            total = int(ln.sum())
            off = torch.cumsum(ln, 0) - ln
            flat = (torch.arange(total, device=self.device)
                    - torch.repeat_interleave(off, ln)
                    + torch.repeat_interleave(s, ln))
            yield flat
            lo = hi

    def store(self, active_idx: np.ndarray) -> None:
        active = torch.as_tensor(np.sort(active_idx), dtype=torch.int64,
                                 device=self.device)
        member = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        member[active] = True
        for flat in self._flat_ranges(active):
            hits = member[self.col[flat].to(torch.int64)]
            sel = flat[hits]
            self.n11.index_add_(
                0, sel, torch.ones(sel.numel(), dtype=self.n11.dtype,
                                   device=self.device))
        self.c[active] += 1
        self.p_stored += 1

    def _values(self) -> torch.Tensor:
        """J values for the current (n11, c, P), materialized blockwise into
        the reusable buffer: J = n11 - f (c_row + c_col) + P f^2."""
        vdt = getattr(self, "_vdt", None) or (
            torch.float16 if self.device.type == "cuda" else torch.float32)
        self._vdt = vdt
        if self._vals is None or self._vals.dtype != vdt:
            self._vals = torch.empty(self.n_directed_entries, dtype=vdt,
                                     device=self.device)
        pf2 = self.p_stored * self.f * self.f
        r0 = 0
        while r0 < self.n:
            r1 = int(torch.searchsorted(
                self.indptr, self.indptr[r0] + FLAT_BLOCK).item())
            r1 = min(max(r1, r0 + 1), self.n)
            s, e = int(self.indptr[r0]), int(self.indptr[r1])
            cnt = self.indptr[r0 + 1:r1 + 1] - self.indptr[r0:r1]
            rows = torch.repeat_interleave(
                torch.arange(r0, r1, device=self.device), cnt)
            v = (self.n11[s:e].to(torch.float32)
                 - self.f * (self.c[rows].to(torch.float32)
                             + self.c[self.col[s:e].to(torch.int64)].to(torch.float32))
                 + pf2)
            self._vals[s:e] = v.to(vdt)
            r0 = r1
        return self._vals

    def recall(self, cues: np.ndarray) -> np.ndarray:
        """(batch, n) 0/1 cues -> (batch, n) boolean fixed-point estimates
        (15 synchronous k-WTA iterations, stable index tie-break). If the
        device's sparse CSR matmul rejects fp16, the value buffer falls
        back to fp32 once (recorded via the buffer dtype)."""
        vals = self._values()
        j = torch.sparse_csr_tensor(self.indptr, self.col, vals,
                                    size=(self.n, self.n))
        y = torch.as_tensor(cues.T, dtype=vals.dtype, device=self.device)
        for _ in range(TF_ITERATIONS):
            try:
                h = torch.matmul(j, y).to(torch.float32)
            except RuntimeError:
                if vals.dtype != torch.float16:
                    raise
                self._vdt = torch.float32
                vals = self._values()
                j = torch.sparse_csr_tensor(self.indptr, self.col, vals,
                                            size=(self.n, self.n))
                y = y.to(torch.float32)
                h = torch.matmul(j, y).to(torch.float32)
            order = torch.argsort(-h, dim=0, stable=True)[: self.k]
            y = torch.zeros_like(y)
            y.scatter_(0, order, 1.0)
        return y.T.to(torch.bool).cpu().numpy()


def run_tf_capacity_comp(n_units: int, directed_entries: int, f: float,
                         seed: int, cfg, device: str = "cpu",
                         log=print) -> dict:
    """The frozen capacity sweep on the TF control (same grid, cue, eval,
    early-stop and interpolation machinery as every other arm)."""
    if cfg.p_cap >= 32768:
        raise ValueError("p_cap >= 32768 overflows the int16 n11 counters")
    t0 = time.time()
    net = TorchTF(n_units, directed_entries, f, seed, device=device)
    k = net.k
    n_keep = k - int(round(0.5 * k))
    ps: list[int] = []
    overlaps: list[float] = []
    below = 0
    for p_next in checkpoint_grid(cfg):
        for mu in range(net.p_stored, p_next):
            net.store(rng_for(seed, STREAM_TF_PATTERNS, mu)
                      .choice(n_units, size=k, replace=False))
        re = rng_for(seed, STREAM_TF_EVAL, p_next)
        sel = (np.arange(p_next, dtype=np.int64) if p_next <= cfg.eval_sample
               else np.sort(re.choice(p_next, size=cfg.eval_sample,
                                      replace=False)).astype(np.int64))
        cues = np.zeros((sel.size, n_units), dtype=np.float32)
        items = {}
        for r, mu in enumerate(sel.tolist()):
            items[mu] = rng_for(seed, STREAM_TF_PATTERNS, mu).choice(
                n_units, size=k, replace=False)
            keep = rng_for(seed, STREAM_TF_CUES, mu).choice(
                items[mu], size=n_keep, replace=False)
            cues[r, keep] = 1.0
        y = net.recall(cues)
        hit = np.array([y[r, items[mu]].sum() / k
                        for r, mu in enumerate(sel.tolist())])
        ps.append(p_next)
        overlaps.append(float(hit.mean()))
        log(f"    P={p_next}: overlap={overlaps[-1]:.4f} "
            f"({time.time() - t0:.0f}s)", flush=True)
        below = below + 1 if overlaps[-1] < cfg.early_stop_bar else 0
        if below >= cfg.early_stop_consecutive:
            break
    out = capacity_from_curve(ps, overlaps, bar=OVERLAP_BAR)
    out.update({"curve_P": ps, "curve_overlap": overlaps,
                "n_units": int(n_units),
                "n_directed_entries": net.n_directed_entries,
                "iterations": TF_ITERATIONS,
                "n11_dtype": str(net.n11.dtype).replace("torch.", ""),
                "n_eval": int(min(ps[-1] if ps else 0, cfg.eval_sample)),
                "wall_s": round(time.time() - t0, 1)})
    return out


__all__ = ["TF_ITERATIONS", "TorchTF", "run_tf_capacity_comp"]
