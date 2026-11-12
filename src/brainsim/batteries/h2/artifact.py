"""H01 statistics artifact loader (F28): the twin's parameters come from
data/connectomes/h01/stats/{connection_stats.json,histograms.json}, never
from invented constants. Fallbacks for parameters the artifact lacks are
declared in config.PROVENANCE_NOTES and echoed into results notes.

logvol bin semantics reproduce scripts/data/h01_stats.py exactly:
b = clip(int((log10(vol_nm3) - 4.0) * 10), 0, 59), so a bin's representative
log10 volume is 4.0 + (b + 0.5) / 10. Type "2" is the excitatory synapse
class in the H01 c3 export; the storage twin is excitatory (T2), so the
strength prior is fitted on type-2 mass only (documented assumption).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from brainsim.paths import DATA

import numpy as np

DEFAULT_STATS_DIR = DATA / "connectomes" / "h01" / "stats"
EXCITATORY_TYPE = "2"


@dataclass
class H01Stats:
    """Twin-facing view of the artifact: trimmed contact pmf + strength prior."""

    contact_ks: np.ndarray          # int, 1..tail_trim_k
    contact_pmf: np.ndarray         # float, sums to 1 over contact_ks
    multi_contact_rate: float       # P(k >= 2) after trim
    mean_k: float                   # mean contacts/connection after trim
    sigma_ln: float                 # lognormal shape (natural log) of strength
    mu_ln: float                    # mean-1 normalisation: -sigma_ln^2/2
    strength_log_edges: np.ndarray  # ln-volume bin edges (reference CDF support)
    strength_cdf: np.ndarray        # empirical CDF at strength_log_edges[1:]
    provenance: dict = field(default_factory=dict)

    def standardized_strength_cdf(self) -> tuple[np.ndarray, np.ndarray]:
        """Reference CDF on standardised ln-strength (location/scale-free):
        the twin is compared to the artifact in shape only, because the twin's
        weights are mean-1 normalised (scale is not load-bearing)."""
        centers = 0.5 * (self.strength_log_edges[:-1] + self.strength_log_edges[1:])
        pmf = np.diff(np.concatenate([[0.0], self.strength_cdf]))
        mean = float(np.sum(centers * pmf))
        std = float(math.sqrt(np.sum((centers - mean) ** 2 * pmf)))
        z_edges = (self.strength_log_edges - mean) / std
        return z_edges, self.strength_cdf


def _trim_contact_distribution(dist: dict, trim_k: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """Trim to the paper-validated k<=trim_k regime (F28); record what was cut."""
    items = sorted((int(k), float(v)) for k, v in dist.items())
    ks = np.array([k for k, _ in items], dtype=np.int64)
    counts = np.array([v for _, v in items], dtype=np.float64)
    keep = ks <= trim_k
    cut_pairs = float(counts[~keep].sum())
    total = float(counts.sum())
    ks_t, counts_t = ks[keep], counts[keep]
    pmf = counts_t / counts_t.sum()
    info = {
        "trim_k": trim_k,
        "pairs_cut": cut_pairs,
        "pairs_cut_fraction": cut_pairs / total,
        "max_k_raw": int(ks.max()),
    }
    return ks_t, pmf, info


def _fit_strength_from_logvol(logvol_bins: dict) -> tuple[float, np.ndarray, np.ndarray, dict]:
    """Fit lognormal sigma (natural log) to the excitatory log-volume mass.
    Only TWO moments are fitted; the twin gate later tests the FULL shape
    against this histogram, which is the held-out degrees-of-freedom check."""
    per_bin: dict[int, float] = {}
    total_all = 0.0
    total_exc = 0.0
    for key, cnt in logvol_bins.items():
        t, b = key.split(":")
        total_all += cnt
        if t == EXCITATORY_TYPE:
            per_bin[int(b)] = per_bin.get(int(b), 0.0) + float(cnt)
            total_exc += cnt
    bins = np.array(sorted(per_bin), dtype=np.int64)
    mass = np.array([per_bin[b] for b in bins], dtype=np.float64)
    centers_log10 = 4.0 + (bins + 0.5) / 10.0
    w = mass / mass.sum()
    mean10 = float(np.sum(centers_log10 * w))
    std10 = float(math.sqrt(np.sum((centers_log10 - mean10) ** 2 * w)))
    sigma_ln = std10 * math.log(10.0)
    edges_ln = (4.0 + np.concatenate([bins, [bins[-1] + 1]]) / 10.0) * math.log(10.0)
    cdf = np.cumsum(w)
    info = {
        "excitatory_fraction_of_synapses": total_exc / total_all,
        "fit_mean_log10_vol": mean10,
        "fit_std_log10_vol": std10,
    }
    return sigma_ln, edges_ln, cdf, info


def from_dicts(connection_stats: dict, histograms: dict, trim_k: int = 50,
               source: str = "in-memory") -> H01Stats:
    ks, pmf, trim_info = _trim_contact_distribution(
        connection_stats["multi_contact_distribution"], trim_k
    )
    sigma_ln, edges_ln, cdf, fit_info = _fit_strength_from_logvol(histograms["logvol_bins"])
    mean_k = float(np.sum(ks * pmf))
    return H01Stats(
        contact_ks=ks,
        contact_pmf=pmf,
        multi_contact_rate=float(pmf[ks >= 2].sum()),
        mean_k=mean_k,
        sigma_ln=sigma_ln,
        mu_ln=-0.5 * sigma_ln * sigma_ln,
        strength_log_edges=edges_ln,
        strength_cdf=cdf,
        provenance={
            "source": source,
            "unique_connections": connection_stats.get("unique_connections"),
            "total_synapses_in_pairs": connection_stats.get("total_synapses_in_pairs"),
            "mean_k_raw": connection_stats.get("mean_k"),
            "max_k_raw": connection_stats.get("max_k"),
            **trim_info,
            **fit_info,
        },
    )


def load(stats_dir: str | Path = DEFAULT_STATS_DIR, trim_k: int = 50) -> H01Stats:
    stats_dir = Path(stats_dir)
    conn = json.loads((stats_dir / "connection_stats.json").read_text())
    hist = json.loads((stats_dir / "histograms.json").read_text())
    return from_dicts(conn, hist, trim_k=trim_k, source=str(stats_dir))


__all__ = ["H01Stats", "from_dicts", "load", "DEFAULT_STATS_DIR"]
