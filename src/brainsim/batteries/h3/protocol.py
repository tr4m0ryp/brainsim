"""H3 episodic protocol (contract Protocol section): timestamped item stream
with salience-gated protection, morphed-pair separation probes, partial-cue
completion probes, an interference load sweep, then log-spaced retention
probes under accelerated turnover with lambda-rate replay transfer.

Accelerated-turnover mapping is IDENTICAL to battery H2's (driver-spec: one
compression constant across batteries): 1 storage step == 1 biological day,
per-day volatility magnitude imported from H2Config.sigma, never redefined
here; probe_times is imported from H2 so retention grids match by
construction. Metrics are emitted as COUNTS (n / retained / sums) so bag
groups merge by summation; fractions exist only at analysis time.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from brainsim.batteries.h2.config import H2Config
from brainsim.batteries.h2.config import PROVENANCE_NOTES as _H2_NOTES
from brainsim.batteries.h2.protocol import probe_times
from brainsim.composite.full import make_cue
from brainsim.composite.full.config import CompositeConfig

from .config import C_TSODYKS, SEED_ROOT, H3Config

# ONE compression constant across batteries (driver-spec). The magnitude is
# H2's field; the day mapping is attested by H2's own provenance note.
TURNOVER_SIGMA_PER_DAY = float(H2Config.sigma)
_H2_DAY_NOTE = "1 store step == 1 biological day"


def shared_turnover_constant() -> float:
    """Return H2's per-day turnover volatility after verifying H2 still
    documents the 1-step==1-day mapping; raises if the shared constant's
    provenance ever drifts (cross-battery consistency check)."""
    if not any(_H2_DAY_NOTE in n for n in _H2_NOTES):
        raise RuntimeError(
            "battery H2 no longer documents the accelerated-turnover day "
            "mapping; H3 shares that constant and refuses to run without it")
    return TURNOVER_SIGMA_PER_DAY


@dataclass(frozen=True)
class ProtoParams:
    n_input: int
    n_items: int
    n_pairs: int
    morph_keep: tuple
    salience_lo: float
    salience_hi: float
    theta_protect: float
    drop_fraction: float
    threshold: float
    m_probe: int
    grid: tuple          # cumulative interference loads, grid[0] == 0
    days: int
    per_day: int
    lambda_rate: float   # replay events per stored item per day


def arm_protocol(cfg: H3Config, lambda_rate: float) -> ProtoParams:
    from .config import base_composite
    return ProtoParams(
        n_input=base_composite(cfg, 0).n_input, n_items=cfg.n_items,
        n_pairs=cfg.n_morph_pairs, morph_keep=cfg.morph_keep,
        salience_lo=cfg.salience_lo, salience_hi=cfg.salience_hi,
        theta_protect=cfg.theta_protect, drop_fraction=cfg.drop_fraction,
        threshold=cfg.overlap_threshold, m_probe=cfg.m_probe,
        grid=cfg.interference_grid, days=cfg.retention_days,
        per_day=cfg.interference_per_day, lambda_rate=lambda_rate)


def screen_protocol(cfg: H3Config, lambda_rate: float) -> ProtoParams:
    """Lean variant for the KO screen (same code path, smaller constants;
    the ko:none baseline runs THIS protocol so degradation is like-for-like)."""
    from .config import base_composite
    return ProtoParams(
        n_input=base_composite(cfg, 0).n_input, n_items=cfg.screen_n_items,
        n_pairs=cfg.screen_pairs, morph_keep=cfg.morph_keep,
        salience_lo=cfg.salience_lo, salience_hi=cfg.salience_hi,
        theta_protect=cfg.theta_protect, drop_fraction=cfg.drop_fraction,
        threshold=cfg.overlap_threshold, m_probe=cfg.screen_m_probe,
        grid=cfg.screen_grid, days=cfg.screen_days,
        per_day=cfg.interference_per_day, lambda_rate=lambda_rate)


# ---- deterministic stream (layout-independent SeedSequence keys) -----------

def _rng(*key: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([SEED_ROOT, *map(int, key)]))


def item_pattern(pp: ProtoParams, seed: int, i: int) -> np.ndarray:
    """Bipolar pattern for stream item i. The last 2*n_pairs items form the
    morphed pairs: member B keeps `keep` of member A's coordinates and
    resamples the rest (expected input overlap ~ keep)."""
    n_base = pp.n_items - 2 * pp.n_pairs
    if i < n_base or (i - n_base) % 2 == 0:
        return _rng(1, seed, i).choice(np.array([-1.0, 1.0]), size=pp.n_input)
    a = item_pattern(pp, seed, i - 1)
    p = (i - n_base) // 2
    keep = pp.morph_keep[p % len(pp.morph_keep)]
    rng = _rng(3, seed, p)
    b = a.copy()
    swap = rng.random(pp.n_input) >= keep
    b[swap] = rng.choice(np.array([-1.0, 1.0]), size=int(swap.sum()))
    return b


def item_salience(pp: ProtoParams, seed: int, i: int) -> tuple[float, bool]:
    s = float(_rng(2, seed, i).uniform(pp.salience_lo, pp.salience_hi))
    return s, s >= pp.theta_protect


def interference_pattern(pp: ProtoParams, seed: int, j: int) -> np.ndarray:
    return _rng(4, seed, j).choice(np.array([-1.0, 1.0]), size=pp.n_input)


def pairs_of(pp: ProtoParams) -> list[tuple[int, int]]:
    n_base = pp.n_items - 2 * pp.n_pairs
    return [(n_base + 2 * p, n_base + 2 * p + 1) for p in range(pp.n_pairs)]


def lambda_star_pred(cc: CompositeConfig, pp: ProtoParams) -> float:
    """Formal-model #5: lambda* ~ r_f / ln(P_fast), with r_f predicted from
    the interference rate over the Tsodyks-Feigelman formation capacity
    (formal-model #2 scaling, C_TSODYKS prefactor flagged engineering)."""
    f = cc.f_formation
    c_form = C_TSODYKS * cc.k_form_rec / (f * math.log(1.0 / f))
    r_f = pp.per_day / max(c_form, 1e-9)
    return r_f / math.log(max(pp.n_items, 3))


# ---- probes ----------------------------------------------------------------

def _probe(system, pp: ProtoParams, seed: int, day: int, load: int) -> dict:
    pick = _rng(5, seed, day, load)
    n_sample = min(pp.m_probe, pp.n_items)
    sample = sorted(pick.choice(pp.n_items, size=n_sample, replace=False))
    out = {"day": day, "L": load, "n": 0, "retained": 0, "score_sum": 0.0,
           "fast_retained": 0, "slow_retained": 0, "fast_sum": 0.0,
           "slow_sum": 0.0}
    for i in sample:
        if not system.owns(i):
            continue
        cue = make_cue(item_pattern(pp, seed, i), pp.drop_fraction,
                       _rng(5, seed, day, load, i))
        sc = system.recall(i, cue)
        out["n"] += 1
        out["score_sum"] += sc["score"]
        out["fast_sum"] += sc["fast"]
        out["slow_sum"] += sc["slow"]
        out["retained"] += int(sc["score"] >= pp.threshold)
        out["fast_retained"] += int(sc["fast"] >= pp.threshold)
        out["slow_retained"] += int(sc["slow"] >= pp.threshold)
    return out


def _separation_probe(system, pp: ProtoParams, seed: int) -> dict:
    out = {"n": 0, "margin_sum": 0.0}
    for a, b in pairs_of(pp):
        for x, y in ((a, b), (b, a)):
            if not system.owns(x):
                continue
            cue = make_cue(item_pattern(pp, seed, x), pp.drop_fraction,
                           _rng(8, seed, x))
            self_ov, cross_ov = system.recall_pair(x, y, cue)
            out["n"] += 1
            out["margin_sum"] += self_ov - cross_ov
    return out


# ---- the protocol -----------------------------------------------------------

def run_protocol(system, pp: ProtoParams, seed: int) -> dict:
    """Episodic write -> L=0 probes (completion + separation) -> interference
    load sweep -> retention days under turnover, continual interference and
    lambda-rate replay transfer, probed at H2's log-spaced times. Returns
    count-based metrics (summable across bag groups)."""
    for i in range(pp.n_items):
        sal, prot = item_salience(pp, seed, i)
        system.write_item(i, item_pattern(pp, seed, i), sal, prot)

    epochs = [_probe(system, pp, seed, 0, 0)]
    separation = _separation_probe(system, pp, seed)

    j = 0
    for load in pp.grid[1:]:
        while j < load:
            system.write_item(pp.n_items + j,
                              interference_pattern(pp, seed, j),
                              pp.salience_lo, False)
            j += 1
        epochs.append(_probe(system, pp, seed, 0, load))

    ts = set(probe_times(pp.days)) if pp.days > 0 else {0}
    retention = [dict(epochs[-1], day=0)]
    carry, events_total = 0.0, 0
    for day in range(1, pp.days + 1):
        carry += pp.lambda_rate * pp.n_items
        budget = int(carry)
        carry -= budget
        events_total += budget
        system.day(budget)
        for _ in range(pp.per_day):
            system.write_item(pp.n_items + j,
                              interference_pattern(pp, seed, j),
                              pp.salience_lo, False)
            j += 1
        if day in ts:
            retention.append(_probe(system, pp, seed, day, pp.grid[-1]))

    return {
        "n_items": pp.n_items,
        "epochs": epochs,
        "separation": separation,
        "retention": retention,
        "lambda_rate": pp.lambda_rate,
        "replay_events_total": events_total,
        "interference_total": int(j),
        "turnover_sigma_per_day": shared_turnover_constant(),
    }


__all__ = ["ProtoParams", "arm_protocol", "screen_protocol", "run_protocol",
           "item_pattern", "item_salience", "interference_pattern",
           "pairs_of", "lambda_star_pred", "probe_times",
           "shared_turnover_constant", "TURNOVER_SIGMA_PER_DAY"]
