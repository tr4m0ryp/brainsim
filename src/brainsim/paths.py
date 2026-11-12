"""Where brainsim looks for connectome data and writes results.

Both roots default to directories next to the package source tree
(`<repo>/data`, `<repo>/results`) and can be moved with environment
variables, read once at import time:

  BRAINSIM_DATA     root of the downloaded datasets (connectomes/, external/)
  BRAINSIM_RESULTS  root of the battery and gate result files

Paths written INTO result files go through `display`, which records them
relative to these roots so no machine-specific prefix lands in a result.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.path.abspath(os.environ.get("BRAINSIM_DATA", REPO_ROOT / "data")))
RESULTS = Path(os.path.abspath(os.environ.get("BRAINSIM_RESULTS", REPO_ROOT / "results")))


def display(path: Path) -> str:
    """Root-relative form of `path` for result files ("data/...", "results/...")."""
    p = Path(os.path.abspath(path))  # symlinked downloads keep their data/ name
    for prefix, root in (("data", DATA), ("results", RESULTS)):
        if p == root or root in p.parents:
            return str(Path(prefix) / p.relative_to(root))
    if REPO_ROOT in p.parents:
        return str(p.relative_to(REPO_ROOT))
    raise ValueError(f"{path} is outside BRAINSIM_DATA, BRAINSIM_RESULTS and the repository")


__all__ = ["DATA", "REPO_ROOT", "RESULTS", "display"]
