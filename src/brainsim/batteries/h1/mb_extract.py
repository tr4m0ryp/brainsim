"""MB associative subgraph extraction (battery H1, fly arm).

Builds the measured circuit from the npy cache: PN->KC expansion, KC->MBON
readout (the plastic projection under rules/mb_rule), and the KC<->APL
inhibition loop. DAN edges are counted and reported but carry no dynamics:
the dopaminergic gate enters the write rule as the exogenous per-item
signal, exactly as in rules/mb_rule.step(dan_signal=...).

Everything is deterministic: classes sort by root_id, edges sort by
(pre, post) in class-local index space.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

CIRCUIT_PROJECTIONS = ("PN->KC", "KC->MBON", "KC->APL", "APL->KC")


@dataclass
class Projection:
    """Directed weighted bipartite edge list in class-local indices."""
    pre: np.ndarray   # int32, index into the presynaptic class
    post: np.ndarray  # int32, index into the postsynaptic class
    w: np.ndarray     # float64 summed anatomical synapse count per pair
    n_pre: int
    n_post: int

    def sort(self) -> "Projection":
        order = np.lexsort((self.post, self.pre))
        return Projection(self.pre[order], self.post[order], self.w[order],
                          self.n_pre, self.n_post)

    def dense(self) -> np.ndarray:
        m = np.zeros((self.n_pre, self.n_post), dtype=np.float64)
        np.add.at(m, (self.pre, self.post), self.w)
        return m

    @property
    def n_pairs(self) -> int:
        return int(self.pre.size)

    @property
    def n_synapses(self) -> int:
        return int(self.w.sum())


@dataclass
class MBGraph:
    """The measured MB circuit plus per-class bookkeeping."""
    roots: dict[str, np.ndarray]          # class -> sorted root_ids (int64)
    pn_kc: Projection
    kc_mbon: Projection
    kc_apl: Projection
    apl_kc: Projection
    counts: dict = field(default_factory=dict)  # full per-class-pair census

    @property
    def n_pn(self) -> int:
        return int(self.roots["PN"].size)

    @property
    def n_kc(self) -> int:
        return int(self.roots["KC"].size)

    @property
    def n_mbon(self) -> int:
        return int(self.roots["MBON"].size)

    def circuit_neurons(self) -> int:
        """Neurons carrying dynamics: PN + KC + MBON + APL."""
        return self.n_pn + self.n_kc + self.n_mbon + int(self.roots["APL"].size)

    def circuit_synapses(self) -> int:
        return sum(p.n_synapses for p in
                   (self.pn_kc, self.kc_mbon, self.kc_apl, self.apl_kc))

    def circuit_pairs(self) -> int:
        return sum(p.n_pairs for p in
                   (self.pn_kc, self.kc_mbon, self.kc_apl, self.apl_kc))


def _projection(pre_root, post_root, cnt, pre_index: dict, post_index: dict,
                n_pre: int, n_post: int) -> Projection:
    sel = np.fromiter((r in pre_index for r in pre_root.tolist()),
                      dtype=bool, count=pre_root.size)
    sel &= np.fromiter((r in post_index for r in post_root.tolist()),
                       dtype=bool, count=post_root.size)
    pr, po, w = pre_root[sel], post_root[sel], cnt[sel]
    pre_idx = np.fromiter((pre_index[r] for r in pr.tolist()), dtype=np.int32,
                          count=pr.size)
    post_idx = np.fromiter((post_index[r] for r in po.tolist()), dtype=np.int32,
                           count=po.size)
    return Projection(pre_idx, post_idx, w.astype(np.float64), n_pre, n_post).sort()


def build_graph(pre_root: np.ndarray, post_root: np.ndarray, cnt: np.ndarray,
                class_roots: dict[str, list[int] | np.ndarray]) -> MBGraph:
    """Assemble the MBGraph from unique-pair edge arrays + class root lists."""
    roots = {c: np.asarray(sorted(map(int, ids)), dtype=np.int64)
             for c, ids in class_roots.items()}
    index = {c: {int(r): i for i, r in enumerate(ids)} for c, ids in roots.items()}
    cls_of: dict[int, str] = {}
    for c, ids in roots.items():
        for r in ids.tolist():
            cls_of[r] = c

    # Full census over every class pair (reported, not all simulated).
    census: dict[str, dict[str, int]] = {}
    pre_cls = [cls_of.get(r) for r in pre_root.tolist()]
    post_cls = [cls_of.get(r) for r in post_root.tolist()]
    for a, b, w in zip(pre_cls, post_cls, cnt.tolist()):
        if a is None or b is None:
            continue
        row = census.setdefault(f"{a}->{b}", {"pairs": 0, "synapses": 0})
        row["pairs"] += 1
        row["synapses"] += int(w)

    def proj(a: str, b: str) -> Projection:
        return _projection(pre_root, post_root, cnt, index[a], index[b],
                           roots[a].size, roots[b].size)

    counts = {
        "neurons": {c: int(v.size) for c, v in roots.items()},
        "class_pairs": dict(sorted(census.items())),
        "circuit_projections": list(CIRCUIT_PROJECTIONS),
    }
    g = MBGraph(roots=roots, pn_kc=proj("PN", "KC"), kc_mbon=proj("KC", "MBON"),
                kc_apl=proj("KC", "APL"), apl_kc=proj("APL", "KC"), counts=counts)
    counts["circuit_neurons"] = g.circuit_neurons()
    counts["circuit_pairs"] = g.circuit_pairs()
    counts["circuit_synapses"] = g.circuit_synapses()
    counts["subgraph_synapses_all_pairs"] = int(
        sum(r["synapses"] for r in census.values()))
    counts["subgraph_pairs_all_pairs"] = int(
        sum(r["pairs"] for r in census.values()))
    return g


def load_mb_graph() -> MBGraph:
    """Load the real measured MB circuit (builds the npy cache on first use)."""
    import json

    from brainsim.batteries.h1.convert import ensure_cache
    cache = ensure_cache()
    pre = np.load(cache / "pre.npy")
    post = np.load(cache / "post.npy")
    cnt = np.load(cache / "count.npy")
    class_roots = json.loads((cache / "class_roots.json").read_text())
    return build_graph(pre, post, cnt, class_roots)


__all__ = ["CIRCUIT_PROJECTIONS", "MBGraph", "Projection", "build_graph",
           "load_mb_graph"]
