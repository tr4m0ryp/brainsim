"""Capacity and retention estimators (pure host numpy, no backend state).

C: interpolated in (ln L, precision) between the last ramp point at or above
the threshold and the first below it; C=0 when even the first load fails;
C = last load (flagged capped) when the ramp never crossed.

T_half: first log-spaced probe time where promoted-item survival falls to
half of S(0), log-interpolated between bracketing probes; censored at t_max
is flagged and downstream uses the value as a lower bound (conservative for
the support claim, never inflating a shift).
"""
from __future__ import annotations

import math


def capacity_from_ramp(loads: list[int], precisions: list[float],
                       threshold: float) -> dict:
    ok = [i for i, p in enumerate(precisions) if p >= threshold]
    if not ok:
        return {"C": 0.0, "capacity_capped": False, "degenerate": True}
    last_ok = ok[-1]
    if last_ok == len(loads) - 1:  # never crossed: ramp ended by cap/guard
        return {"C": float(loads[-1]), "capacity_capped": True, "degenerate": False}
    l_g, l_b = loads[last_ok], loads[last_ok + 1]
    p_g, p_b = precisions[last_ok], precisions[last_ok + 1]
    frac = (p_g - threshold) / max(p_g - p_b, 1e-12)
    c = math.exp(math.log(l_g) + frac * (math.log(l_b) - math.log(l_g)))
    return {"C": float(c), "capacity_capped": False, "degenerate": False}


def t_half_from_survival(times: list[int], survival: list[float],
                         t_max: int) -> dict:
    s0 = survival[0]
    if s0 <= 0.0:
        return {"t_half": 0.0, "censored": False, "s0": s0, "degenerate": True}
    target = 0.5 * s0
    for i in range(1, len(times)):
        if survival[i] <= target:
            t_prev, t_cur = times[i - 1], times[i]
            s_prev, s_cur = survival[i - 1], survival[i]
            frac = (s_prev - target) / max(s_prev - s_cur, 1e-12)
            if t_prev == 0:
                t_half = frac * t_cur
            else:
                t_half = math.exp(math.log(t_prev)
                                  + frac * (math.log(t_cur) - math.log(t_prev)))
            return {"t_half": float(max(t_half, 1e-3)), "censored": False,
                    "s0": s0, "degenerate": False}
    return {"t_half": float(t_max), "censored": True, "s0": s0, "degenerate": False}


def pareto_front(points: list[dict]) -> list[dict]:
    """Non-dominated subset of {label, C, T_half} maximising both axes."""
    front = []
    for a in points:
        dominated = any(
            (b["C"] >= a["C"] and b["T_half"] >= a["T_half"]
             and (b["C"] > a["C"] or b["T_half"] > a["T_half"]))
            for b in points if b is not a
        )
        if not dominated:
            front.append(a)
    return sorted(front, key=lambda x: -x["C"])


__all__ = ["capacity_from_ramp", "t_half_from_survival", "pareto_front"]
