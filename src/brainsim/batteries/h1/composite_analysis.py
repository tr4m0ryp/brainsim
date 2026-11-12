"""H1 loop-arm analysis: both arms' ratio blocks, handed to the FROZEN
verdicts.h1_verdict, plus the work-item key and merge error shared with
composite_report (which re-exports all of them)."""
from __future__ import annotations

import numpy as np

from brainsim.batteries import verdicts
from brainsim.batteries.h1.composite_arm import CompH1Config

RATIO_DEF = "C_measured / mean(C over config-model draws), per seed"


class MergeError(RuntimeError):
    pass


def battery_common():
    """The frozen verdict module (brainsim.batteries.verdicts)."""
    return verdicts


def work_key(it: dict) -> str:
    return f"comp|{it['name']}|seed={it['seed']}"


def _summary(bc, values: list[float]) -> dict:
    lo, hi = bc.bootstrap_ci(values)
    return {"per_seed": values, "mean": float(np.mean(values)),
            "ci95": [lo, hi]}


def _cohens_d(a: list[float], b: list[float]) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    n1, n2 = a.size, b.size
    pooled = ((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2)
    return float((a.mean() - b.mean()) / np.sqrt(max(pooled, 1e-30)))


def analyze(cfg: CompH1Config, seeds: list[int], results: dict,
            fly_report: dict) -> dict:
    """Assemble both arms' ratio blocks and hand them to the FROZEN
    h1_verdict. Raises MergeError on any non-finite composite ratio (no
    verdict may be minted from a degenerate ratio)."""
    bc = battery_common()

    def c_of(name: str, seed: int) -> float:
        return float(results[work_key({"name": name, "seed": seed})]
                     ["capacity"]["C"])

    draws = [f"config:{d}" for d in range(cfg.n_shuffle_draws)]
    instances = list(cfg.instances())
    per_arm: dict[str, dict] = {}
    for name in instances:
        cs = [c_of(name, s) for s in seeds]
        per_arm[name] = {"C": _summary(bc, cs),
                         "n_zero_capacity_seeds": int(sum(c == 0.0 for c in cs))}
    c_meas = [c_of("measured", s) for s in seeds]
    c_shuf = [float(np.mean([c_of(d, s) for d in draws])) for s in seeds]
    ratios = [m / sh if sh > 0 else float("inf")
              for m, sh in zip(c_meas, c_shuf)]
    if not all(np.isfinite(r) for r in ratios):
        raise MergeError(
            f"composite per-seed ratios are non-finite ({ratios}): the "
            "config-model draws hit zero capacity; refusing to mint a "
            "verdict from a degenerate ratio")
    fly_ratios = [float(r) for r in
                  fly_report["ratio_C_meas_over_C_shuf"]["per_seed"]]
    verdict = bc.h1_verdict(fly_ratios, ratios)

    ratio_block = {"per_seed": ratios, "definition": RATIO_DEF,
                   "mean": float(np.mean(ratios))}
    ratio_block["ci95"] = list(bc.bootstrap_ci(ratios))
    ratio_block["effect_size_cohens_d"] = _cohens_d(c_meas, c_shuf)

    c_ko = [c_of("measured_ffi_ko", s) for s in seeds]
    delta = [m - k for m, k in zip(c_meas, c_ko)]
    ablation = {
        "knockout": ("feedforward-inhibition abstraction (top-f readout) "
                     "removed on the front-window stages -- the H3-"
                     "established composite realization; coding level not "
                     "holdable under this abstraction (the cap IS the "
                     "mechanism), documented deviation from the fly gamma=0 "
                     "semantics"),
        "C_measured": _summary(bc, c_meas), "C_knockout": _summary(bc, c_ko),
        "delta_C": _summary(bc, delta),
        "resolvable": bool(any(d != 0.0 for d in delta)),
    }
    means = {n: per_arm[n]["C"]["mean"] for n in instances}
    order = sorted(means, key=means.get)
    observed = ("composite arm capacity ordering (ascending means): "
                + " < ".join(f"{n}={means[n]:.2f}" for n in order))
    controls = [v for n, v in means.items()
                if n not in ("measured", "measured_ffi_ko")]
    if controls and means["measured"] < min(controls):
        observed += ("; the composite formation wiring UNDERPERFORMS every "
                     "matched control on this protocol (anti-premium); the "
                     "formal verdict is h1_verdict's, not this note's")
    return {"verdict": verdict, "arms_composite": per_arm,
            "ratio_comp": ratio_block,
            "ratio_fly": fly_report["ratio_C_meas_over_C_shuf"],
            "ablation_ffi_knockout": ablation, "observed": observed}


__all__ = ["MergeError", "RATIO_DEF", "analyze", "work_key"]
