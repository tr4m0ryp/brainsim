"""One-time conversion of the v783 connections feather into an MB-subgraph
npy cache (gitignored, regenerable), following the fly-gate pattern: the
arrow read runs in a subprocess of the current interpreter (pyarrow is an
optional dependency, `pip install brainsim[flywire]`), and the science stack
loads plain .npy afterwards.

The Zenodo proofread_connections table carries one row per
(pre, post, NEUROPIL); the worker aggregates to unique (pre, post) pairs,
summing syn_count, and filters to edges whose BOTH endpoints are in the MB
node union (KC | MBON | PN | DAN | APL).
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

from brainsim.paths import DATA, display

from brainsim.batteries.h1.annotations import (ANNOTATIONS_TSV, ensure_annotations,
                                               sha256_of)

CONNECTIONS = DATA / "connectomes" / "flywire" / "proofread_connections_783.feather"
CACHE = DATA / "external" / "h1_mb_npy"

_WORKER = r"""
import csv, json, os, sys
import numpy as np
import pyarrow.feather as pf

ann, con, dst = sys.argv[1], sys.argv[2], sys.argv[3]
classes = {}
with open(ann) as f:
    for row in csv.DictReader(f, delimiter="\t"):
        rid = int(row["root_id"])
        cc, ct = row["cell_class"], row.get("cell_type") or ""
        hb = row.get("hemibrain_type") or ""
        if cc == "Kenyon_Cell": classes[rid] = "KC"
        elif ct.startswith("MBON") or hb.startswith("MBON"): classes[rid] = "MBON"
        elif cc == "ALPN": classes[rid] = "PN"
        elif cc == "DAN": classes[rid] = "DAN"
        elif ct == "APL" or hb == "APL": classes[rid] = "APL"

t = pf.read_table(con, columns=["pre_pt_root_id", "post_pt_root_id", "syn_count"],
                  memory_map=True)
pre = t.column(0).to_numpy(); post = t.column(1).to_numpy()
cnt = t.column(2).to_numpy().astype(np.int64)
ids = np.fromiter(classes.keys(), dtype=np.int64, count=len(classes))
mask = np.isin(pre, ids) & np.isin(post, ids)
pre, post, cnt = pre[mask], post[mask], cnt[mask]
raw_rows = int(pre.size)

# Aggregate (pre, post, neuropil) rows to unique (pre, post) pairs.
order = np.lexsort((post, pre))
pre, post, cnt = pre[order], post[order], cnt[order]
new = np.empty(pre.size, dtype=bool); new[0] = True
new[1:] = (pre[1:] != pre[:-1]) | (post[1:] != post[:-1])
grp = np.cumsum(new) - 1
agg = np.zeros(int(grp[-1]) + 1, dtype=np.int64)
np.add.at(agg, grp, cnt)
pre_u, post_u = pre[new], post[new]

os.makedirs(dst, exist_ok=True)
np.save(os.path.join(dst, "pre.npy"), pre_u)
np.save(os.path.join(dst, "post.npy"), post_u)
np.save(os.path.join(dst, "count.npy"), agg)
class_roots = {}
for rid, c in classes.items():
    class_roots.setdefault(c, []).append(rid)
for c in class_roots:
    class_roots[c].sort()
json.dump(class_roots, open(os.path.join(dst, "class_roots.json"), "w"))
meta = {"raw_mb_rows": raw_rows, "unique_pairs": int(pre_u.size),
        "total_synapses": int(agg.sum()),
        "src_size": os.path.getsize(con), "src_mtime": os.path.getmtime(con),
        "ann_size": os.path.getsize(ann)}
json.dump(meta, open(os.path.join(dst, "meta.json"), "w"))
print("h1 cache:", raw_rows, "rows ->", pre_u.size, "pairs,", int(agg.sum()), "synapses")
"""


def ensure_cache(force: bool = False) -> Path:
    """Build (or reuse) the MB npy cache; returns the cache directory."""
    ensure_annotations()
    meta_p = CACHE / "meta.json"
    if meta_p.exists() and not force:
        meta = json.loads(meta_p.read_text())
        src = CONNECTIONS.stat()
        if (meta.get("src_size") == src.st_size
                and abs(meta.get("src_mtime", 0) - src.st_mtime) < 1e-3
                and meta.get("ann_size") == ANNOTATIONS_TSV.stat().st_size):
            return CACHE
    if importlib.util.find_spec("pyarrow") is None:
        raise RuntimeError("h1 npy cache missing and pyarrow not installed "
                           "(pip install 'brainsim[flywire]')")
    if not CONNECTIONS.exists():
        raise RuntimeError(f"connections table missing: {CONNECTIONS}")
    subprocess.run([sys.executable, "-c", _WORKER, str(ANNOTATIONS_TSV),
                    str(CONNECTIONS), str(CACHE)], check=True)
    return CACHE


def provenance() -> dict:
    """Checksums and sources for the result JSON."""
    return {
        "connections": {"path": display(CONNECTIONS),
                        "sha256": sha256_of(CONNECTIONS)},
        "annotations": {"path": display(ANNOTATIONS_TSV),
                        "sha256": sha256_of(ANNOTATIONS_TSV)},
    }


__all__ = ["CACHE", "CONNECTIONS", "ensure_cache", "provenance"]
