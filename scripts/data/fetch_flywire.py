#!/usr/bin/env python3
"""Download the FlyWire v783 inputs of the mushroom-body capacity battery.

  python scripts/data/fetch_flywire.py      # ~0.9 GB into <data>/connectomes/flywire/

Files and pinned SHA-256 (the hashes recorded in results/h1/fly_arm.json):
  proofread_connections_783.feather  Zenodo record 10676866 (FlyWire v783.0,
                                     CC BY 4.0 on Zenodo; flywire.ai states
                                     CC BY-NC 4.0 for its public release)
  neuron_annotations_v2.1.0.tsv      flyconnectome/flywire_annotations tag
                                     v2.1.0 (no licence file in that repository;
                                     downloaded for local use, never redistributed)

A hash mismatch deletes the download and exits non-zero.
"""
from __future__ import annotations

import hashlib
import sys
import urllib.request

from brainsim.batteries.h1.annotations import ANNOTATIONS_TSV, ANNOTATIONS_URL
from brainsim.batteries.h1.convert import CONNECTIONS

FILES = [
    (CONNECTIONS,
     "https://zenodo.org/records/10676866/files/proofread_connections_783.feather?download=1",
     "24f960ae3e7d4f8cd30db3b62e99fb5179cc3d1e76d8c155bfb441e9737d3faf"),
    (ANNOTATIONS_TSV, ANNOTATIONS_URL,
     "30be6c73975a70c56d930e27911f36455d3886e15abf383b78edd2a5d679e0b6"),
]


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    for dest, url, want in FILES:
        if dest.exists() and sha256(dest) == want:
            print(f"ok      {dest.name}")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.name + ".part")
        print(f"fetch   {dest.name} <- {url}", flush=True)
        urllib.request.urlretrieve(url, part)  # noqa: S310 (pinned https URL)
        got = sha256(part)
        if got != want:
            part.unlink()
            print(f"FATAL: {dest.name} sha256 {got} != pinned {want}")
            return 1
        part.rename(dest)
        print(f"ok      {dest.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
