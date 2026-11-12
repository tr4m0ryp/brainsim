"""Matched-budget arithmetic (driver-spec: 'the matched-resources rule is
TOTAL synapse count equal within 1%'). Stage-alone arms keep the composite's
CompositeConfig shape with pass-through stages at thin floors and the stage
under test scaled to absorb the budget: n and the dominant in-degree scale
preserving the composite's k/n ratio, then the dominant k is trimmed to land
the total inside the 1% envelope. Every constructor ASSERTS the match; the
assertion is the mechanism, not a comment.
"""
from __future__ import annotations

import math
from dataclasses import replace

from brainsim.composite.full.build import edge_specs, edge_synapse_count
from brainsim.composite.full.config import CompositeConfig

from .config import H3Config, base_composite

TOL = 0.01  # driver-spec: within 1%

# stage -> (n field, in-edge k fields) for thinning; input is NEVER thinned
# (patterns must be identical across arms).
_STAGE_FIELDS = {
    "separation": ("n_separation", ("k_sep_in",)),
    "formation": ("n_formation", ("k_form_in", "k_form_direct", "k_form_rec")),
    "output": ("n_output", ("k_out_in",)),
    "relay": ("n_relay", ("k_relay_in",)),
    "storage": ("n_storage", ("k_store_in", "k_store_rec")),
}


class BudgetError(AssertionError):
    pass


def total_synapses(cc: CompositeConfig) -> int:
    return sum(edge_synapse_count(cc, e) for e in edge_specs(cc))


def assert_matched(name: str, total: int, target: int, tol: float = TOL) -> None:
    if abs(total - target) > tol * target:
        raise BudgetError(
            f"{name}: total {total:.4e} vs target {target:.4e} "
            f"(off {abs(total - target) / target:.2%} > {tol:.0%})")


def thin_stages(cc: CompositeConfig, stages: tuple[str, ...],
                thin_n: int, thin_k: int) -> CompositeConfig:
    patch: dict = {}
    for s in stages:
        n_field, k_fields = _STAGE_FIELDS[s]
        patch[n_field] = thin_n
        for kf in k_fields:
            patch[kf] = thin_k
    return replace(cc, **patch)


def _trim(cc: CompositeConfig, name: str, k_field: str, target: int,
          arm: str) -> CompositeConfig:
    """Adjust `k_field` by the integer that closes the residual (the edge's
    count is linear in k at fixed n, multi-contact factor included), then
    assert the 1% match."""
    spec = {e.name: e for e in edge_specs(cc)}[name]
    per_k = edge_synapse_count(cc, spec) / max(1, spec.k)  # syn per unit k
    resid = target - total_synapses(cc)
    k_new = max(1, spec.k + int(round(resid / max(per_k, 1e-9))))
    cc = replace(cc, **{k_field: k_new})
    assert_matched(arm, total_synapses(cc), target)
    return cc


def stage_alone_config(cfg: H3Config, seed: int, stage: str) -> CompositeConfig:
    """CompositeConfig for '<stage>_alone' at the composite's total budget."""
    base = base_composite(cfg, seed)
    target = total_synapses(base)
    others = tuple(s for s in _STAGE_FIELDS if s != stage)
    cc = thin_stages(base, others, cfg.thin_n, cfg.thin_k)

    n_field, _ = _STAGE_FIELDS[stage]
    n0 = getattr(base, n_field)
    if stage in ("formation", "storage"):
        k_field = "k_form_rec" if stage == "formation" else "k_store_rec"
        edge = "form_rec" if stage == "formation" else "store_rec"
        r = getattr(base, k_field) / n0
        # in-paths keep the composite's k (the stage's own identity)
        if stage == "formation":
            cc = replace(cc, k_form_in=base.k_form_in,
                         k_form_direct=base.k_form_direct)
            a = base.k_form_in + base.k_form_direct
            mc = 1.0
        else:
            cc = replace(cc, k_store_in=base.k_store_in)
            a = base.k_store_in
            mc = 1.0 + base.multi_contact_frac * (
                (2 + base.multi_contact_tail) / 2 - 1)
        rest = total_synapses(replace(cc, **{n_field: 0, k_field: 1}))
        s_rem = max(1.0, target - rest)
        n = int(round((-a + math.sqrt(a * a + 4 * r * mc * s_rem)) / (2 * r * mc)))
        cc = replace(cc, **{n_field: max(4, n),
                            k_field: max(2, int(round(r * n)))})
    else:
        k_field = {"separation": "k_sep_in", "relay": "k_relay_in"}[stage]
        edge = {"separation": "sep_in", "relay": "relay_in"}[stage]
        r = getattr(base, k_field) / n0
        rest = total_synapses(replace(cc, **{n_field: 0, k_field: 1}))
        s_rem = max(1.0, target - rest)
        n = int(round(math.sqrt(s_rem / r)))
        cc = replace(cc, **{n_field: max(4, n),
                            k_field: max(1, int(round(r * n)))})
    return _trim(cc, edge, k_field, target, f"{stage}_alone")


def bag_layout(cfg: H3Config, seed: int) -> tuple[int, int]:
    """(N modules, per-module synapse budget). driver-spec formula: N chosen
    so per-attractor size = composite formation's per-item synapse load."""
    base = base_composite(cfg, seed)
    target = total_synapses(base)
    form = {e.name: e for e in edge_specs(base)}["form_rec"]
    per_item = edge_synapse_count(base, form) / max(1, cfg.n_items)
    n_modules = max(1, int(round(target / per_item)))
    return n_modules, target


def bag_module_config(cfg: H3Config, seed: int, module_id: int,
                      n_modules: int, target_total: int) -> CompositeConfig:
    """One independent formation-copy module (no separation/relay/storage
    function: those stages sit at bag-thin floors; module memory is its
    formation attractor). Budget = total/N per module."""
    base = base_composite(cfg, seed)
    s_mod = target_total / n_modules
    cc = thin_stages(base, tuple(_STAGE_FIELDS), cfg.bag_thin_n, cfg.bag_thin_k)
    r = base.k_form_rec / base.n_formation
    a = 2 * cfg.bag_thin_k  # form_in + form_direct at bag-thin k
    rest = total_synapses(replace(cc, n_formation=0, k_form_rec=1))
    if rest > 0.5 * s_mod:  # fail loudly instead of silently overshooting
        raise BudgetError(
            f"bag module thin floors ({rest} syn) exceed half the module "
            f"budget ({s_mod:.0f} syn); shrink bag_thin_n/bag_thin_k")
    s_rem = max(1.0, s_mod - rest)
    n = int(round((-a + math.sqrt(a * a + 4 * r * s_rem)) / (2 * r)))
    cc = replace(cc, n_formation=max(4, n),
                 k_form_rec=max(2, int(round(r * n))),
                 seed=seed * 1_000_003 + module_id)
    # per-module trim so the SUM over modules lands inside 1% of the target
    spec = {e.name: e for e in edge_specs(cc)}["form_rec"]
    per_k = edge_synapse_count(cc, spec) / max(1, spec.k)
    resid = s_mod - total_synapses(cc)
    k_new = max(2, spec.k + int(round(resid / max(per_k, 1e-9))))
    return replace(cc, k_form_rec=k_new)


def relay_removed_config(cfg: H3Config, seed: int) -> CompositeConfig:
    """Ablation: no thin-relay mechanism -- relay widened to a transparent
    output-sized stage, MB gate off (arms.py zeroes mb writes); k_relay_in
    raised as far as the 1% budget envelope allows (PROVENANCE_NOTES)."""
    base = base_composite(cfg, seed)
    target = total_synapses(base)
    old = base.n_relay * base.k_relay_in
    k_new = max(1, int((0.009 * target + old) // base.n_output))
    cc = replace(base, n_relay=base.n_output, f_relay=base.f_output,
                 k_relay_in=k_new)
    assert_matched("relay_removed", total_synapses(cc), target)
    return cc


def _bisect_scale(make_cfg, target: int, s_lo: float = 0.05,
                  s_hi: float = 64.0) -> CompositeConfig:
    for _ in range(60):
        s = 0.5 * (s_lo + s_hi)
        if total_synapses(make_cfg(s)) < target:
            s_lo = s
        else:
            s_hi = s
    return make_cfg(0.5 * (s_lo + s_hi))


def context_config(cfg: H3Config, seed: int, which: str) -> CompositeConfig:
    """Matched-budget reimplementations (contract arm 5), statistics-level:
    ctx_ecdgca3 = EC->DG->CA3 front half (separation+formation scaled, rest
    thin; Nat Comput Sci 2021 separation model); ctx_kassab = DG->CA3->CA1
    loop (separation+formation+output scaled, no relay/storage function;
    Kassab & Alexandre)."""
    base = base_composite(cfg, seed)
    target = total_synapses(base)
    keep = ("separation", "formation") if which == "ctx_ecdgca3" else \
        ("separation", "formation", "output")
    others = tuple(s for s in _STAGE_FIELDS if s not in keep)

    def make(s: float) -> CompositeConfig:
        cc = thin_stages(base, others, cfg.thin_n, cfg.thin_k)
        patch = {
            "n_separation": max(4, int(round(base.n_separation * s))),
            "k_sep_in": max(1, int(round(base.k_sep_in * s))),
            "n_formation": max(4, int(round(base.n_formation * s))),
            "k_form_rec": max(2, int(round(base.k_form_rec * s))),
        }
        if "output" in keep:
            patch["n_output"] = max(4, int(round(base.n_output * s)))
            patch["k_out_in"] = max(1, int(round(base.k_out_in * s)))
        return replace(cc, **patch)

    cc = _bisect_scale(make, target)
    return _trim(cc, "form_rec", "k_form_rec", target, which)


def budget_table(cfg: H3Config, seed: int) -> dict:
    """Per-arm totals + targets (emitted into results; also what tests
    assert). Composite is its own target by construction."""
    base = base_composite(cfg, seed)
    target = total_synapses(base)
    rows = {"composite": {"total": target, "target": target}}
    for stage in _STAGE_FIELDS:
        arm = f"{stage}_alone"
        if arm in cfg.stage_alone_arms:
            rows[arm] = {"total": total_synapses(
                stage_alone_config(cfg, seed, stage)), "target": target}
    if cfg.run_bag:
        n_mod, tgt = bag_layout(cfg, seed)
        tot = sum(total_synapses(bag_module_config(cfg, seed, m, n_mod, tgt))
                  for m in range(n_mod))
        assert_matched("bag", tot, tgt)
        rows["bag"] = {"total": tot, "target": tgt, "n_modules": n_mod}
    for ctx in cfg.context_arms:
        rows[ctx] = {"total": total_synapses(context_config(cfg, seed, ctx)),
                     "target": target}
    if "relay_removed" in cfg.ablation_arms:
        rows["relay_removed"] = {"total": total_synapses(
            relay_removed_config(cfg, seed)), "target": target}
    for name, row in rows.items():
        row["ratio"] = row["total"] / row["target"]
    return rows


__all__ = ["total_synapses", "assert_matched", "thin_stages",
           "stage_alone_config", "bag_layout", "bag_module_config",
           "relay_removed_config", "context_config", "budget_table",
           "BudgetError", "TOL"]
