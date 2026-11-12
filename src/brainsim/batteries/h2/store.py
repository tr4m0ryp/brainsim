"""Contact store: flat per-contact arrays on the backend plus a host-built
CSR (sorted-by-pre order) for row gathers. Redundancy installs append
pending segments; segments join the CSR at the next rebuild() (documented
visibility lag; between rebuilds they still receive turnover/leak via
parts()). CSR construction is host numpy in both engines, so structure is
engine-invariant by construction.
"""
from __future__ import annotations

import numpy as np

from .twin import Twin


class ContactStore:
    def __init__(self, ops, twin: Twin) -> None:
        self.ops = ops
        self.n = twin.n
        self.substrate_nnz = twin.n_contacts
        # Host structure mirrors (frozen substrate + appended overlay).
        self.pre_host = twin.pre
        self.post_host = twin.post
        self.w0_sub_host = twin.w0  # frozen substrate weights: budget pricing
        self._pending_host: list[tuple[np.ndarray, np.ndarray]] = []
        # Backend state.
        self.post_b = ops.from_numpy(twin.post, dtype="i4")
        self.w = ops.from_numpy(twin.w0)
        self.w0 = ops.from_numpy(twin.w0)
        self.state = ops.cascade_init(twin.n_contacts)
        self._pending_b: list[dict] = []
        self.indptr: np.ndarray | None = None
        self._order_b = None
        self.rebuild()

    # -- size ------------------------------------------------------------------
    @property
    def n_contacts(self) -> int:
        return int(self.pre_host.size) + sum(p[0].size for p in self._pending_host)

    # -- install ------------------------------------------------------------------
    def append_block(self, pre: np.ndarray, post: np.ndarray, w: np.ndarray,
                     deep_state_steps: int) -> None:
        """Append installed contacts (host arrays). They enter elementwise
        dynamics immediately and row gathers at the next rebuild()."""
        ops = self.ops
        if pre.size == 0:
            return
        self._pending_host.append((pre.astype(np.int32), post.astype(np.int32)))
        wb = ops.from_numpy(w.astype(np.float32))
        state = ops.cascade_deepen_subset(ops.cascade_init(pre.size), deep_state_steps)
        self._pending_b.append({
            "post": ops.from_numpy(post.astype(np.int32), dtype="i4"),
            "w": wb, "w0": ops.from_numpy(w.astype(np.float32)), "state": state,
        })

    def rebuild(self) -> None:
        """Consolidate pending segments and rebuild the host CSR."""
        ops = self.ops
        if self._pending_host:
            pres = [self.pre_host] + [p for p, _ in self._pending_host]
            posts = [self.post_host] + [q for _, q in self._pending_host]
            self.pre_host = np.concatenate(pres)
            self.post_host = np.concatenate(posts)
            self.post_b = ops.concat([self.post_b] + [s["post"] for s in self._pending_b])
            self.w = ops.concat([self.w] + [s["w"] for s in self._pending_b])
            self.w0 = ops.concat([self.w0] + [s["w0"] for s in self._pending_b])
            self.state = ops.concat([self.state] + [s["state"] for s in self._pending_b])
            self._pending_host, self._pending_b = [], []
        order = np.argsort(self.pre_host, kind="stable")
        counts = np.bincount(self.pre_host, minlength=self.n)
        self.indptr = np.zeros(self.n + 1, dtype=np.int64)
        np.cumsum(counts, out=self.indptr[1:])
        self._order_b = self.ops.from_numpy(order.astype(np.int64), dtype="i8")

    # -- gathers ---------------------------------------------------------------------
    def row_contacts(self, rows: np.ndarray):
        """Backend indices of every consolidated contact whose pre is in
        `rows` (host int array). Ragged gather via the CSR order."""
        ops = self.ops
        starts = self.indptr[rows]
        lens = self.indptr[rows + 1] - starts
        total = int(lens.sum())
        if total == 0:
            return ops.from_numpy(np.empty(0, dtype=np.int64), dtype="i8")
        starts_b = ops.from_numpy(starts.astype(np.int64), dtype="i8")
        lens_b = ops.from_numpy(lens.astype(np.int64), dtype="i8")
        ex = ops.cumsum(lens_b) - lens_b
        base = ops.repeat_interleave(starts_b - ex, lens_b)
        pos = base + ops.arange(total)
        return ops.take(self._order_b, pos)

    def parts(self):
        """(w, w0, state, setter) triples covering consolidated + pending
        contacts, for elementwise store dynamics."""
        yield ("main", self.w, self.w0, self.state)
        for i, seg in enumerate(self._pending_b):
            yield (f"pending{i}", seg["w"], seg["w0"], seg["state"])

    def set_part_state(self, name: str, state) -> None:
        if name == "main":
            self.state = state
        else:
            self._pending_b[int(name.removeprefix("pending"))]["state"] = state

    def recall_drive(self, cue_rows: np.ndarray):
        """Post-side drive from a cue: sum of consolidated contact weights out
        of the cue's rows. Parallel contacts of a connection sum naturally,
        which is exactly the redundancy motif's readout pooling."""
        ops = self.ops
        idx = self.row_contacts(cue_rows)
        drive = ops.zeros(self.n, dtype="f4")
        if len(idx) > 0:
            posts = ops.take(self.post_b, idx)
            ops.add_at(drive, ops.as_index(posts), ops.take(self.w, idx))
        return drive


__all__ = ["ContactStore"]
