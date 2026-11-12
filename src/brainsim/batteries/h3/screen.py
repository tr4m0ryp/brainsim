"""Hidden-keys screen (contract: Bacik-style exhaustive single knockouts +
sampled pair knockouts, GP-outlier ranking). Single KOs enumerate the
circuit builders' cell-type labels -- granule, pyramidal, storage,
interneuron (realized in the full runtime as the top-f FF-inhibition
abstraction, per composite config docstring) -- plus the structural motifs
(multi-contact tail, FF-inhibition path, and the composite's remaining
pathway motifs). Ranking: degradation vector over {capacity, separation,
completion, retention} vs the ko:none baseline, Mahalanobis distance from
the KO degradation cloud (sklearn-free: numpy cov + pinv). Top-K rows are
the hidden-key candidates; each is emitted with its targeted follow-up
ablation spec (to be pre-registered before running).
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .config import SEED_ROOT, H3Config, base_composite
from .arms import derived_seed, headline_lambda_pred, _sigma_patch
from .protocol import screen_protocol
from .systems import RunFlags, SingleSystem, build_runtime


@dataclass(frozen=True)
class KOSpec:
    name: str
    patch: tuple = ()   # ((config_field, value), ...)
    ops: tuple = ()     # systems.OPS names applied post-build
    topf: tuple = ()    # stages with the top-f readout knocked out


# Enumeration order is FROZEN (pair sampling and ranking are keyed to it).
SINGLE_KOS: tuple[KOSpec, ...] = (
    KOSpec("ct:granule", ops=("zero:sep_in",)),
    KOSpec("ct:pyramidal", ops=("freeze_btsp",)),
    KOSpec("ct:storage", ops=("freeze_store",)),
    KOSpec("ct:interneuron",
           topf=("separation", "formation", "output", "relay", "storage")),
    KOSpec("motif:ff_inhibition", topf=("separation",)),
    KOSpec("motif:multi_contact_tail", patch=(("multi_contact_frac", 0.0),)),
    KOSpec("motif:mossy_detonator", ops=("zero:form_in",)),
    KOSpec("motif:direct_path", ops=("zero:form_direct",)),
    KOSpec("motif:out_path", ops=("zero:out_in",)),
    KOSpec("motif:relay_gate", ops=("neutral_mb",)),
    KOSpec("motif:store_path", ops=("zero:store_in",)),
)
_BY_NAME = {k.name: k for k in SINGLE_KOS}
METRICS = ("capacity", "separation", "completion", "retention")


def single_names(cfg: H3Config) -> tuple[str, ...]:
    if not cfg.ko_singles:
        return tuple(k.name for k in SINGLE_KOS)
    unknown = set(cfg.ko_singles) - set(_BY_NAME)
    if unknown:
        raise ValueError(f"unknown KO names {sorted(unknown)}")
    return tuple(cfg.ko_singles)


def pair_names(cfg: H3Config) -> tuple[str, ...]:
    """Sampled pair KOs; budget <= 2x singles (driver-spec), sampled once
    per battery (same pairs for every seed), seeded and layout-independent."""
    singles = single_names(cfg)
    cands = [(a, b) for i, a in enumerate(singles) for b in singles[i + 1:]]
    budget = cfg.ko_pairs_override if cfg.ko_pairs_override >= 0 else \
        cfg.ko_pair_budget_factor * len(singles)
    budget = min(budget, len(cands))
    if budget <= 0:
        return ()
    rng = np.random.default_rng(np.random.SeedSequence([SEED_ROOT, 7]))
    picked = rng.choice(len(cands), size=budget, replace=False)
    return tuple(f"pair:{cands[int(i)][0]}+{cands[int(i)][1]}"
                 for i in sorted(picked))


def ko_names(cfg: H3Config) -> tuple[str, ...]:
    return ("none",) + single_names(cfg) + pair_names(cfg)


def _resolve(name: str) -> list[KOSpec]:
    if name == "none":
        return []
    if name.startswith("pair:"):
        a, b = name[len("pair:"):].split("+")
        return [_BY_NAME[a], _BY_NAME[b]]
    return [_BY_NAME[name]]


def make_ko_system(cfg: H3Config, seed: int, name: str, rank: int = 0,
                   world_size: int = 1, device: str = "cpu"):
    """KO run = the composite (union readout, headline lambda) on the LEAN
    screen protocol, minus the knocked-out component(s). Config patches are
    applied pre-build; ops/topf act on the built runtime."""
    specs = _resolve(name)
    cc = _sigma_patch(base_composite(cfg, seed).with_seed(derived_seed(seed)))
    patch = {f: v for spec in specs for f, v in spec.patch}
    if patch:
        cc = replace(cc, **patch)
    flags = RunFlags(ops=tuple(o for s in specs for o in s.ops),
                     topf_off=tuple({t for s in specs for t in s.topf}))
    rt = build_runtime(cc, flags, rank=rank, world_size=world_size,
                       device=device)
    lam = headline_lambda_pred(cfg, seed)
    pp = screen_protocol(cfg, lam)
    meta = {"ko": name, "kind": "ko", "lambda_rate": lam,
            "budget_total": None, "budget_target": None}
    return SingleSystem(rt, "union", flags), pp, meta


# ---- analysis (merge-side; consumes protocol count metrics) -----------------

def screen_scalars(m: dict) -> dict:
    """{capacity, separation, completion, retention} from one run's counts.
    capacity = retained fraction after the full interference grid (pre-
    turnover); completion = mean partial-cue overlap at L=0; separation =
    mean morph-pair margin; retention = retained fraction at the final
    turnover probe."""
    def frac(row, num="retained"):
        return row[num] / row["n"] if row["n"] else 0.0

    return {
        "capacity": frac(m["epochs"][-1]),
        "separation": (m["separation"]["margin_sum"] / m["separation"]["n"]
                       if m["separation"]["n"] else 0.0),
        "completion": (m["epochs"][0]["score_sum"] / m["epochs"][0]["n"]
                       if m["epochs"][0]["n"] else 0.0),
        "retention": frac(m["retention"][-1]),
    }


def degradation_vector(base: dict, ko: dict, eps: float = 1e-9) -> list[float]:
    return [(base[k] - ko[k]) / max(abs(base[k]), eps) for k in METRICS]


def mahalanobis_ranking(vectors: dict[str, list[float]]) -> list[dict]:
    """GP-outlier score = Mahalanobis distance of each KO's degradation
    vector from the KO cloud (numpy cov + pinv per driver-spec). Descending
    distance; ties broken by name for determinism."""
    names = sorted(vectors)
    x = np.asarray([vectors[n] for n in names], dtype=np.float64)
    if len(names) < 2:
        dists = np.zeros(len(names))
    else:
        mu = x.mean(axis=0)
        cov = np.cov(x, rowvar=False)
        cov = np.atleast_2d(cov)
        pinv = np.linalg.pinv(cov)
        d = x - mu
        dists = np.sqrt(np.maximum(0.0, np.einsum("ij,jk,ik->i", d, pinv, d)))
    rows = [{"ko": n, "distance": float(dd),
             "degradation": {k: float(v) for k, v in zip(METRICS, vectors[n])}}
            for n, dd in zip(names, dists)]
    return sorted(rows, key=lambda r: (-r["distance"], r["ko"]))


def followup_list(ranking: list[dict], top_k: int) -> list[dict]:
    """Top-K hidden-key candidates, each with its targeted follow-up
    ablation spec (contract: 'each earns a targeted follow-up ablation with
    matched controls'). Emitted for pre-registration; NOT run here."""
    return [{
        "rank": i + 1,
        "ko": row["ko"],
        "distance": row["distance"],
        "followup": (f"targeted ablation of {row['ko']} on the full "
                     "composite with a matched-budget control arm; "
                     "pre-register before running"),
    } for i, row in enumerate(ranking[:top_k])]


__all__ = ["KOSpec", "SINGLE_KOS", "METRICS", "single_names", "pair_names",
           "ko_names", "make_ko_system", "screen_scalars",
           "degradation_vector", "mahalanobis_ranking", "followup_list"]
