"""Merge-side analysis: per-arm retained-at-T tables (CIs from the frozen
verdicts.bootstrap_ci), the lambda predicted-vs-measured overlay, the
KO degradation vectors + Mahalanobis ranking, and the budget table. The
verdict itself comes ONLY from verdicts.h3_verdict, called here with
composite_retained and best_part_retained (best single-stage arm, per the
frozen contract's kill criterion)."""
from __future__ import annotations

import json
import math

import numpy as np

from . import screen
from .config import HEADLINE_LAMBDA, H3Config


def arm_metrics(cfg, results, work_key, name, lf, seed) -> dict:
    """One logical arm run's metrics; bag groups are summed count-wise."""
    if name != "bag":
        return results[work_key({"kind": "arm", "name": name, "lf": lf,
                                 "group": None, "seed": seed})]["metrics"]
    parts = [results[work_key({"kind": "arm", "name": "bag", "lf": lf,
                               "group": g, "seed": seed})]["metrics"]
             for g in range(cfg.bag_groups)]
    out = json.loads(json.dumps(parts[0]))
    count_keys = ("n", "retained", "score_sum", "fast_retained",
                  "slow_retained", "fast_sum", "slow_sum", "margin_sum")
    for other in parts[1:]:
        for sec in ("epochs", "retention"):
            for row, orow in zip(out[sec], other[sec]):
                for k in count_keys:
                    if k in row:
                        row[k] += orow[k]
        for k in ("n", "margin_sum"):
            out["separation"][k] += other["separation"][k]
    return out


def retained_at_t(m: dict) -> float:
    row = m["retention"][-1]
    return (row["retained"] / row["n"]) * m["n_items"] if row["n"] else 0.0


def _fit_rate(days: list[float], surv: list[float]) -> float | None:
    pts = [(d, s) for d, s in zip(days, surv) if s > 0]
    if len(pts) < 2 or pts[0][1] <= 0:
        return None
    x = np.array([p[0] for p in pts])
    y = np.log(np.array([p[1] for p in pts]))
    if np.allclose(x, x[0]):
        return None
    return float(-np.polyfit(x, y, 1)[0])


def _survival(cfg, results, work_key, name, lf, seeds, field):
    days, curves = None, []
    for seed in seeds:
        m = arm_metrics(cfg, results, work_key, name, lf, seed)
        rows = m["retention"]
        days = [r["day"] for r in rows]
        curves.append([r[field] / r["n"] if r["n"] else 0.0 for r in rows])
    return days, list(np.mean(np.array(curves), axis=0))


def analyze(cfg: H3Config, seeds: list[int], results: dict,
            work_key, battery_common) -> dict:
    bc = battery_common()
    flags: list[str] = []

    def retained_list(name, lf=HEADLINE_LAMBDA):
        return [retained_at_t(arm_metrics(cfg, results, work_key, name, lf, s))
                for s in seeds]

    arm_table: dict[str, dict] = {}
    arm_names = (["composite"] + list(cfg.stage_alone_arms)
                 + (["bag"] if cfg.run_bag else [])
                 + list(cfg.ablation_arms) + list(cfg.context_arms))
    for name in arm_names:
        vals = retained_list(name)
        lo, hi = bc.bootstrap_ci(vals) if len(vals) > 1 else (vals[0], vals[0])
        arm_table[name] = {"retained_at_T": vals,
                           "mean": float(np.mean(vals)), "ci95": [lo, hi]}

    composite_retained = arm_table["composite"]["mean"]
    part_means = {s: arm_table[s]["mean"] for s in cfg.stage_alone_arms}
    best_part_name = max(part_means, key=lambda s: (part_means[s], s))
    best_part = part_means[best_part_name]
    if best_part <= 0.0:
        flags.append("degenerate: best single-stage arm retained 0 items; "
                     "floored at 1e-9 for the frozen verdict ratio")
        best_part = 1e-9
    verdict = bc.h3_verdict(composite_retained, best_part)

    overlay = {"lambda_factors": list(cfg.lambda_factors),
               "retained_by_factor": {f"{lf:g}": float(np.mean(retained_list(
                   "composite", lf))) for lf in cfg.lambda_factors}}
    meta0 = results[work_key({"kind": "arm", "name": "composite",
                              "lf": HEADLINE_LAMBDA, "group": None,
                              "seed": seeds[0]})]["meta"]
    overlay["lambda_star_pred"] = meta0.get("lambda_star_pred")
    days, s_fast = _survival(cfg, results, work_key, "composite",
                             HEADLINE_LAMBDA, seeds, "fast_retained")
    overlay["survival_fast"] = {"day": days, "s": s_fast}
    r_f = _fit_rate(days, s_fast)
    r_s = None
    if "replay_off" in cfg.ablation_arms:
        days_s, s_slow = _survival(cfg, results, work_key, "replay_off",
                                   HEADLINE_LAMBDA, seeds, "slow_retained")
        overlay["survival_slow_lambda0"] = {"day": days_s, "s": s_slow}
        r_s = _fit_rate(days_s, s_slow)
    overlay.update({"r_f_measured": r_f, "r_s_measured": r_s})
    if r_f is not None and r_s is not None and abs(r_f - r_s) > 1e-12:
        pred, p_items, t_end = {}, cfg.n_items, float(days[-1])
        lam0 = overlay["lambda_star_pred"] or 0.0
        for lf in cfg.lambda_factors:
            lam = lf * lam0
            n_pred = p_items * math.exp(-r_f * t_end) + lam * p_items * (
                math.exp(-r_s * t_end) - math.exp(-r_f * t_end)) / (r_f - r_s)
            pred[f"{lf:g}"] = min(float(p_items), n_pred)
        overlay["predicted_retained_by_factor"] = pred
    else:
        flags.append("lambda overlay: r_f/r_s fit unavailable "
                     "(flat or empty survival); predicted curve omitted")

    base_scalars = {s: screen.screen_scalars(
        results[work_key({"kind": "ko", "name": "none", "seed": s})]["metrics"])
        for s in seeds}
    vectors: dict[str, list[float]] = {}
    for name in screen.ko_names(cfg):
        if name == "none":
            continue
        vecs = [screen.degradation_vector(
            base_scalars[s], screen.screen_scalars(
                results[work_key({"kind": "ko", "name": name,
                                  "seed": s})]["metrics"])) for s in seeds]
        vectors[name] = list(np.mean(np.array(vecs), axis=0))
    ranking = screen.mahalanobis_ranking(vectors)
    ko_screen = {"baseline": {str(s): base_scalars[s] for s in seeds},
                 "degradation_vectors": vectors, "ranking": ranking,
                 "hidden_key_followups": screen.followup_list(
                     ranking, cfg.top_k_hidden)}

    budget_table = {k: {"total": v["meta"].get("budget_total"),
                        "target": v["meta"].get("budget_target")}
                    for k, v in sorted(results.items())
                    if v["meta"].get("budget_total") is not None}
    return {"verdict": verdict, "composite_retained": composite_retained,
            "best_part_retained": best_part, "best_part_arm": best_part_name,
            "arm_table": arm_table, "lambda_overlay": overlay,
            "ko_screen": ko_screen, "budget_table": budget_table,
            "analysis_flags": sorted(set(flags))}


__all__ = ["analyze", "arm_metrics", "retained_at_t"]
