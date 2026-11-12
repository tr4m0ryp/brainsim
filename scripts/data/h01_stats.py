#!/usr/bin/env python3
"""Stream H01 c3 synapse export -> storage-side statistics.

Source: H01 release (Shapson-Coe et al. 2024, Science), c3 synapse export in
the public bucket gs://h01-release, licence CC BY 4.0.

Never stores the 159 GB export: each JSON shard is downloaded, parsed,
aggregated, and deleted. Outputs under <data>/connectomes/h01/stats/:
  pairs.txt.gz        one "pre post" line per synapse (for external sort)
  histograms.json     size/type/target/confidence histograms + counters
  connection_stats.json  (after `finalize`) multi-contact distribution

Usage:
  python3 scripts/data/h01_stats.py stream [--workers 3] [--limit N]
  python3 scripts/data/h01_stats.py finalize
"""
from __future__ import annotations

import gzip
import json
import math
import subprocess
import sys
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from brainsim.paths import DATA

OUT = DATA / "connectomes" / "h01" / "stats"
BUCKET = "https://storage.googleapis.com/h01-release"
PREFIX = "data/20210729/c3/synapses/exported/"
LIST_URL = (
    "https://storage.googleapis.com/storage/v1/b/h01-release/o"
    f"?prefix={PREFIX}&fields=items(name,size),nextPageToken&maxResults=1000"
)


def list_json_shards() -> list[str]:
    with urllib.request.urlopen(LIST_URL, timeout=60) as r:
        items = json.load(r).get("items", [])
    return sorted(i["name"] for i in items if i["name"].endswith(".json"))


def bbox_volume_nm3(bb: dict) -> float:
    s = bb["size"]
    # H01 voxels are 8x8x33 nm at c3 export resolution.
    return int(s["x"]) * 8.0 * int(s["y"]) * 8.0 * int(s["z"]) * 33.0


def process_shard(name: str, hist: dict) -> int:
    """Writes pairs to a per-shard gzip: no shared file handles, kill-safe."""
    url = f"{BUCKET}/{name}"
    shard_id = name.rsplit("export", 1)[-1].replace(".json", "")
    tmp = OUT / "pairs" / f"shard_{shard_id}.txt.gz.tmp"
    final = OUT / "pairs" / f"shard_{shard_id}.txt.gz"
    if final.exists():
        return 0
    tmp.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    pairs_out = gzip.open(tmp, "wt", compresslevel=3)
    with urllib.request.urlopen(url, timeout=600) as r:
        for raw in r:
            try:
                syn = json.loads(raw)
            except json.JSONDecodeError:
                continue
            pre = syn["pre_synaptic_site"].get("neuron_id")
            post = syn["post_synaptic_partner"].get("neuron_id")
            if not pre or not post:
                continue
            n += 1
            pairs_out.write(f"{pre} {post}\n")
            t = syn.get("type", "?")
            hist["type_counts"][t] += 1
            tgt = syn["post_synaptic_partner"].get("class_label", "?")
            hist["target_counts"][tgt] += 1
            conf = syn.get("confidence", 0.0)
            hist["conf_deciles"][min(9, int(conf * 10))] += 1
            vol = bbox_volume_nm3(syn["bounding_box"])
            if vol > 0:
                b = min(59, max(0, int((math.log10(vol) - 4.0) * 10)))
                hist["logvol_bins"][f"{t}:{b}"] += 1
    pairs_out.close()
    tmp.rename(final)
    return n


def stream(workers: int, limit: int | None) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lock = OUT / "stream.lock"
    if lock.exists():
        print(f"another stream holds {lock}; refusing to double-run")
        raise SystemExit(3)
    lock.write_text(str(__import__("os").getpid()))
    try:
        shards = list_json_shards()
        if limit:
            shards = shards[:limit]
        pair_dir = OUT / "pairs"
        pair_dir.mkdir(exist_ok=True)
        todo = [
            s for s in shards
            if not (pair_dir / (
                "shard_" + s.rsplit("export", 1)[-1].replace(".json", "")
                + ".txt.gz")).exists()
        ]
        print(f"{len(shards)} shards, {len(todo)} to do", flush=True)
        hist = {
            "type_counts": Counter(), "target_counts": Counter(),
            "conf_deciles": Counter(), "logvol_bins": Counter(), "synapses": 0,
        }
        hf = OUT / "histograms.json"
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for name, count in zip(todo, ex.map(
                lambda s: process_shard(s, hist), todo
            )):
                hist["synapses"] += count
                hf.write_text(json.dumps(
                    {k: dict(v) if isinstance(v, Counter) else v
                     for k, v in hist.items()}, indent=0))
                print(f"done {name} (+{count}, total {hist['synapses']})",
                      flush=True)
        print(f"STREAM COMPLETE: {hist['synapses']} synapses", flush=True)
    finally:
        lock.unlink(missing_ok=True)


def finalize() -> None:
    pair_dir = OUT / "pairs"
    counts = OUT / "pair_counts.txt"
    n_shards = len(list(pair_dir.glob("shard_*.txt.gz")))
    print(f"finalizing over {n_shards} shard files", flush=True)
    # External sort keeps memory flat regardless of pair cardinality.
    subprocess.run(
        f"zcat {pair_dir}/shard_*.txt.gz | LC_ALL=C sort -S 8G --parallel=3 "
        f"| uniq -c | awk '{{print $1}}' | LC_ALL=C sort -n | uniq -c > {counts}",
        shell=True, check=True,
    )
    dist = {}
    with counts.open() as f:
        for line in f:
            n_pairs, k = line.split()
            dist[int(k)] = int(n_pairs)
    total_pairs = sum(dist.values())
    total_syn = sum(k * v for k, v in dist.items())
    tail = {k: v for k, v in dist.items() if k >= 10}
    out = {
        "unique_connections": total_pairs,
        "total_synapses_in_pairs": total_syn,
        "multi_contact_distribution": dist,
        "k_ge_10_tail": tail,
        "max_k": max(dist),
        "mean_k": total_syn / total_pairs,
    }
    (OUT / "connection_stats.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in
                      ("unique_connections", "max_k", "mean_k")}, indent=1))


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in {"stream", "finalize"}:
        print(__doc__)
        return 2
    if sys.argv[1] == "finalize":
        finalize()
        return 0
    workers = 3
    limit = None
    if "--workers" in sys.argv:
        workers = int(sys.argv[sys.argv.index("--workers") + 1])
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])
    stream(workers, limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
