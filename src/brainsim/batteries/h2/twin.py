"""H01 statistical twin (T8) at battery scale, built on HOST numpy so the
substrate is bit-identical across engines, then handed to the backend.

Contact-count pmf and strength CV come from the F28 artifact (artifact.py);
k_in and n are config (k_in is a declared fallback, see PROVENANCE_NOTES).
Weights are lognormal quantised to the 26-state Bartol ceiling via the
canonical circuits.builders.quantize_to_states.

Twin gate (contract: "Twin validated on held-out statistics BEFORE any arm
runs"): generation-fidelity tolerances on {multi-contact rate, mean_k,
contact-pmf KS} plus a shape KS of the quantised weight sample against the
FULL artifact histogram (only 2 moments were fitted, so the shape test has
held-out degrees of freedom). Every process that builds a twin re-checks the
gate and refuses to run arms on a failing twin; the authoritative gate block
is emitted once (shard 0).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from brainsim.circuits.builders import quantize_to_states

from .artifact import H01Stats
from .config import H2Config

# Engineering tolerances (documented; the gate is pre-registered code).
TOL_RATE_REL = 0.10
TOL_MEANK_REL = 0.02
TOL_CONTACT_KS = 0.02
TOL_STRENGTH_KS = 0.08  # absorbs 26-state quantisation steps (~1/26) on top of
#                          lognormal-vs-histogram mismatch; fails if the artifact
#                          shape is not lognormal-like.
GATE_STRENGTH_SAMPLE = 1_000_000


@dataclass
class Twin:
    n: int
    pre: np.ndarray        # int32, per contact (host, frozen)
    post: np.ndarray       # int32, per contact (host, frozen)
    w0: np.ndarray         # float32 substrate weights (host, frozen)
    conn_counts: np.ndarray  # contacts per connection (host, build stat)
    gate: dict

    @property
    def n_contacts(self) -> int:
        return int(self.pre.size)

    @property
    def n_connections(self) -> int:
        return int(self.conn_counts.size)


def _sample_connections(n: int, k_in: int, rng: np.random.Generator):
    """k_in draws per post with duplicate (pre,post) pairs and self-loops
    removed by a global key sort. Realised in-degree is k_in minus the
    collision/self-loop loss (recorded; gate checks self-consistency)."""
    post = np.repeat(np.arange(n, dtype=np.int64), k_in)
    pre = rng.integers(0, n, size=n * k_in, dtype=np.int64)
    key = pre * n + post
    key = key[pre != post]
    key.sort(kind="quicksort")
    keep = np.ones(key.size, dtype=bool)
    keep[1:] = key[1:] != key[:-1]
    key = key[keep]
    return (key // n).astype(np.int32), (key % n).astype(np.int32)


def build_twin(stats: H01Stats, cfg: H2Config, seed: int) -> Twin:
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x7719]))
    pre_u, post_u = _sample_connections(cfg.n, cfg.k_in, rng)
    n_conn = pre_u.size
    counts = rng.choice(stats.contact_ks, size=n_conn, p=stats.contact_pmf)
    pre = np.repeat(pre_u, counts)
    post = np.repeat(post_u, counts)
    raw_w = rng.lognormal(stats.mu_ln, stats.sigma_ln, size=pre.size)
    w0 = quantize_to_states(raw_w, cfg.n_states).astype(np.float32)
    twin = Twin(n=cfg.n, pre=pre, post=post, w0=w0,
                conn_counts=counts.astype(np.int32), gate={})
    twin.gate = run_gate(twin, stats, cfg, rng)
    return twin


def _ks_discrete(ks: np.ndarray, pmf_ref: np.ndarray, counts_obs: np.ndarray) -> float:
    obs = np.zeros(pmf_ref.size)
    hist = np.bincount(counts_obs, minlength=int(ks.max()) + 1).astype(np.float64)
    for i, k in enumerate(ks):
        obs[i] = hist[k] if k < hist.size else 0.0
    obs /= obs.sum()
    return float(np.max(np.abs(np.cumsum(obs) - np.cumsum(pmf_ref))))


def _ks_strength(w: np.ndarray, stats: H01Stats, rng: np.random.Generator) -> float:
    """Sup CDF distance between the twin's standardised ln-weights and the
    artifact's standardised ln-volume histogram (shape-only comparison)."""
    take = min(GATE_STRENGTH_SAMPLE, w.size)
    sample = w[rng.choice(w.size, size=take, replace=False)].astype(np.float64)
    lw = np.log(np.maximum(sample, 1e-30))
    z = np.sort((lw - lw.mean()) / max(lw.std(), 1e-12))
    z_edges, ref_cdf = stats.standardized_strength_cdf()
    # Evaluate both CDFs at the reference edges (twin CDF by searchsorted).
    twin_cdf = np.searchsorted(z, z_edges[1:], side="right") / z.size
    return float(np.max(np.abs(twin_cdf - ref_cdf)))


def run_gate(twin: Twin, stats: H01Stats, cfg: H2Config,
             rng: np.random.Generator | None = None) -> dict:
    rng = rng or np.random.default_rng(0)
    counts = twin.conn_counts
    rate = float(np.mean(counts >= 2))
    mean_k = float(counts.mean())
    checks = {
        "multi_contact_rate": {
            "twin": rate, "artifact": stats.multi_contact_rate,
            "tol_rel": TOL_RATE_REL,
            "pass": abs(rate - stats.multi_contact_rate)
                    <= TOL_RATE_REL * stats.multi_contact_rate,
        },
        "mean_k": {
            "twin": mean_k, "artifact": stats.mean_k, "tol_rel": TOL_MEANK_REL,
            "pass": abs(mean_k - stats.mean_k) <= TOL_MEANK_REL * stats.mean_k,
        },
        "contact_pmf_ks": {
            "value": _ks_discrete(stats.contact_ks, stats.contact_pmf, counts),
            "tol": TOL_CONTACT_KS,
        },
        "strength_shape_ks": {
            "value": _ks_strength(twin.w0, stats, rng),
            "tol": TOL_STRENGTH_KS,
        },
        "realized_mean_in_degree": {
            # Expected distinct non-self pre after k_in with-replacement draws;
            # the check is self-consistency of the sampler, not an artifact
            # statistic (the artifact has no census; see PROVENANCE_NOTES).
            "twin": twin.n_connections / twin.n,
            "expected": (cfg.n - 1) * (1.0 - (1.0 - 1.0 / cfg.n) ** cfg.k_in),
            "configured_k_in": cfg.k_in, "tol_rel": TOL_MEANK_REL,
        },
    }
    checks["contact_pmf_ks"]["pass"] = (
        checks["contact_pmf_ks"]["value"] <= TOL_CONTACT_KS)
    checks["strength_shape_ks"]["pass"] = (
        checks["strength_shape_ks"]["value"] <= TOL_STRENGTH_KS)
    d_in = checks["realized_mean_in_degree"]
    d_in["pass"] = abs(d_in["twin"] - d_in["expected"]) <= TOL_MEANK_REL * d_in["expected"]
    passed = all(c["pass"] for c in checks.values())
    return {
        "passed": bool(passed),
        "checks": checks,
        "n_connections": twin.n_connections,
        "n_contacts": twin.n_contacts,
        "note": ("contact/rate/mean_k are generation-fidelity checks; "
                 "strength_shape_ks is the held-out-shape check (2-moment fit "
                 "vs full histogram). Motif/reciprocity held-outs are not "
                 "derivable from the aggregate artifact (flagged)."),
    }


class TwinGateError(RuntimeError):
    """Raised when arms are requested on a twin whose gate did not pass."""


def require_gate(twin: Twin) -> None:
    if not twin.gate.get("passed"):
        raise TwinGateError(
            "twin gate FAILED; the frozen contract forbids running arms on an "
            f"unvalidated twin: {twin.gate.get('checks')}"
        )


__all__ = ["Twin", "build_twin", "run_gate", "require_gate", "TwinGateError"]
