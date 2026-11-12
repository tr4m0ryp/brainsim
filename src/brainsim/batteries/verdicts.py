"""Shared battery infrastructure: results contract and FROZEN verdicts.

The verdict functions below mechanically encode the pre-registered kill
criteria of the three battery contracts, written and committed before any
result-bearing run. Battery drivers may not implement their own verdict
logic; they call these. Changing a threshold here after a battery has
produced data breaks the pre-registration and must be reported with the
result (git history is the witness).

The contract documents themselves are not part of this repository; each
one is identified by the SHA-256 it had when it was frozen, and every
results file embeds that hash.

Results contract: every battery writes `<RESULTS>/<key>/results.json`
embedding the contract hash, full_scale, n_synapses, n_seeds >= 3, and a
verdict in {supported, killed, partial}.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from brainsim.paths import REPO_ROOT, RESULTS

# SHA-256 of each contract document as frozen before any result-bearing run.
FROZEN_CONTRACTS = {
    "battery-h1-wiring-premium.md":
        "3829b418b9271c1f87c2eae69a9eb82aec846d608746886b95d67038b21284a8",
    "battery-h2-permanence-frontier.md":
        "4ab0e6327261d8a4623b6d0d17ae15847ae1abca129268e02b9e6928ab11bf74",
    "battery-h3-composite.md":
        "7460fc5c93f0fc35468eb8fb133ffcffb57756096b82f86f80ca99277003e4bc",
}


def contract_sha256(contract_name: str) -> str:
    """Frozen hash of a battery contract; unknown names raise KeyError."""
    return FROZEN_CONTRACTS[contract_name]


def git_sha() -> str:
    r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                       capture_output=True, text=True)
    return r.stdout.strip()[:12] if r.returncode == 0 else "unknown"


def bootstrap_ci(values: Sequence[float], n_boot: int = 10_000,
                 alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI over seeds."""
    v = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n_boot, v.size), replace=True).mean(axis=1)
    return (float(np.quantile(means, alpha / 2)),
            float(np.quantile(means, 1 - alpha / 2)))


# ---- frozen verdicts -------------------------------------------------------

def h1_verdict(ratio_fly: Sequence[float], ratio_comp: Sequence[float]) -> str:
    """Battery H1 (frozen): claim C_meas/C_shuf > 1.10 at precision 0.9.
    Kill: ratio in [0.95, 1.05] across seeds on BOTH fly and composite arms.
    Between kill band and claim bar: partial.
    """
    def in_kill_band(rs: Sequence[float]) -> bool:
        return all(0.95 <= r <= 1.05 for r in rs)

    def clears_bar(rs: Sequence[float]) -> bool:
        return float(np.mean(rs)) > 1.10

    if in_kill_band(ratio_fly) and in_kill_band(ratio_comp):
        return "killed"
    if clears_bar(ratio_fly) or clears_bar(ratio_comp):
        return "supported"
    return "partial"


def h2_verdict(shift_by_kp: dict[str, float],
               noise_band_by_kp: dict[str, float]) -> str:
    """Battery H2 (frozen): kill iff the redundancy-motif frontier shift is
    <= the bootstrap noise band at EVERY (k, p) tested. Supported iff the
    shift exceeds the band at at least one (k, p); the frontier plot carries
    the size. No grid extension after data.
    """
    keys = set(shift_by_kp)
    if not keys or keys != set(noise_band_by_kp):
        raise ValueError("shift and noise-band grids differ")
    exceeds = [shift_by_kp[k] > noise_band_by_kp[k] for k in sorted(keys)]
    return "supported" if any(exceeds) else "killed"


def h3_verdict(composite_retained: float,
               best_part_retained: float) -> str:
    """Battery H3 (frozen): claim bar composite >= 1.25 x best single stage;
    kill iff composite <= 1.0 x best part; between: partial ("helps but
    below the pre-registered bar").
    """
    if best_part_retained <= 0:
        raise ValueError("best_part_retained must be positive")
    ratio = composite_retained / best_part_retained
    if ratio <= 1.0:
        return "killed"
    if ratio >= 1.25:
        return "supported"
    return "partial"


# ---- results writer --------------------------------------------------------

def write_results(key: str, contract_name: str, *, n_neurons: int,
                  n_synapses: int, seeds: Sequence[int], verdict: str,
                  arms: dict[str, Any], engine: str,
                  extra: dict[str, Any] | None = None) -> Path:
    if verdict not in {"supported", "killed", "partial"}:
        raise ValueError(f"invalid verdict {verdict!r}")
    out = {
        "battery": key,
        "contract": contract_name,
        "contract_sha256": contract_sha256(contract_name),
        "full_scale": True,
        "n_neurons": int(n_neurons),
        "n_synapses": int(n_synapses),
        "n_seeds": len(list(seeds)),
        "seeds": list(map(int, seeds)),
        "verdict": verdict,
        "arms": arms,
        "engine": engine,
        "code_version": git_sha(),
    }
    if extra:
        out.update(extra)
    dest = RESULTS / key / "results.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1))
    return dest


__all__ = ["FROZEN_CONTRACTS", "bootstrap_ci", "contract_sha256", "git_sha",
           "h1_verdict", "h2_verdict", "h3_verdict", "write_results"]
