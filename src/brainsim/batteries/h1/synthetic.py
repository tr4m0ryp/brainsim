"""Synthetic tiny MB graphs: used by the test suite and by --smoke runs.

Never used for science. Smoke runs exercise every code path on these graphs
so that, per the frozen contract, no real-circuit capacity number can be
produced at smoke scale.
"""
from __future__ import annotations

import numpy as np

from brainsim.batteries.h1.mb_extract import MBGraph, build_graph


def make_synthetic_graph(seed: int = 0, n_pn: int = 24, n_kc: int = 120,
                         n_mbon: int = 10, n_dan: int = 6) -> MBGraph:
    """Small random MB-shaped graph with fake root ids (deterministic)."""
    rng = np.random.default_rng([0x51A17, seed])
    base = 10_000
    roots = {
        "PN": list(range(base, base + n_pn)),
        "KC": list(range(base + 1000, base + 1000 + n_kc)),
        "MBON": list(range(base + 3000, base + 3000 + n_mbon)),
        "DAN": list(range(base + 4000, base + 4000 + n_dan)),
        "APL": [base + 5000, base + 5001],
    }
    pre, post, cnt = [], [], []

    def connect(a_ids, b_ids, k_per_post):
        for b in b_ids:
            srcs = rng.choice(len(a_ids), size=min(k_per_post, len(a_ids)),
                              replace=False)
            for s in srcs:
                pre.append(a_ids[s])
                post.append(b)
                cnt.append(int(rng.integers(1, 12)))

    connect(roots["PN"], roots["KC"], 5)       # expansion claws
    connect(roots["KC"], roots["MBON"], 40)    # readout convergence
    for kc in roots["KC"]:                     # APL loop touches every KC
        apl = roots["APL"][kc % 2]
        pre.extend([kc, apl])
        post.extend([apl, kc])
        cnt.extend([int(rng.integers(1, 6)), int(rng.integers(1, 6))])
    connect(roots["KC"], roots["DAN"], 3)      # census-only edges
    return build_graph(np.asarray(pre, dtype=np.int64),
                       np.asarray(post, dtype=np.int64),
                       np.asarray(cnt, dtype=np.int64), roots)


__all__ = ["make_synthetic_graph"]
