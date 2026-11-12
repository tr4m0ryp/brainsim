"""H2 protocol: salience-tagged item writes, continual interference,
multiplicative turnover with cascade-gated leak, log-spaced retention probes.

Store-step model per contact (w weight, w0 frozen setpoint, s cascade state,
rho = depth(s)/n_levels):
  1. w *= exp(sigma*z - sigma^2/2), z ~ N(0,1) iid per contact. Activity-
     independent turnover (Dvorkin & Ziv 2016): NOT reduced by cascade depth;
     k independent contacts pool it, which is the formal model's sqrt(k).
  2. w += theta_leak*(1-rho)*(w0 - w). Mean-reverting decay of deposits;
     cascade depth protects (Fusi resistance), substrate stays stationary.
Interference/item deposits are cascade-gated by (1-rho); replay deposits are
not (consolidation pathway, F4/F11) -- see PROVENANCE_NOTES.

Randomness discipline (shard-layout independent, engine-invariant streams):
  item/interference codes  host rng keyed (seed, p, phase)
  probe cues + samples     host rng keyed (seed, p, phase, t or L)
  replay scheduling        host rng keyed (seed, p, k, arm)
  turnover + cascade       backend rng keyed (seed, p, k, arm, phase)
"""
from __future__ import annotations

import numpy as np

from .config import SEED_ROOT, H2Config
from .measures import capacity_from_ramp, t_half_from_survival
from .motif import ArmBehavior
from .store import ContactStore
from .twin import Twin

ARM_CODES = {"redundancy": 1, "uniform": 2, "cascade_only": 3, "replay": 4, "combined": 5}
PHASE_CAP, PHASE_RET = 1, 2


def p_code(p: float) -> int:
    return int(round(p * 1000))


def host_rng(*key: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([SEED_ROOT, *map(int, key)]))


def backend_seed(seed: int, p: float, k: int, arm: str, phase: int) -> int:
    ss = np.random.SeedSequence(
        [SEED_ROOT, 4, seed, p_code(p), k, ARM_CODES[arm], phase])
    return int(ss.generate_state(1)[0])


class H2System:
    def __init__(self, ops, cfg: H2Config, twin: Twin, behavior: ArmBehavior,
                 stream_rng: np.random.Generator) -> None:
        self.ops = ops
        self.cfg = cfg
        self.store = ContactStore(ops, twin)
        self.behavior = behavior
        self.stream_rng = stream_rng
        self.a = max(1, int(round(cfg.f * cfg.n)))
        self.w_mean_sub = float(twin.w0.mean(dtype=np.float64))
        self.eta = cfg.eta_rel * self.w_mean_sub
        self.eta_dep = cfg.eta_dep_frac * self.eta
        self.codes: list[np.ndarray] = []
        self.uniform_pending = 0.0
        self.resource_capped = False
        self._replay_touch_cache: dict[int, object] = {}

    # -- codes -------------------------------------------------------------------
    def new_code(self) -> np.ndarray:
        return np.sort(self.stream_rng.choice(self.cfg.n, size=self.a,
                                              replace=False)).astype(np.int64)

    # -- write --------------------------------------------------------------------
    def write_item(self, code: np.ndarray, promote: bool, record: bool = True) -> int:
        ops, cfg = self.ops, self.cfg
        item_id = len(self.codes) if record else -1
        idx = self.store.row_contacts(code)
        if len(idx) > 0:
            posts = ops.take(self.store.post_b, idx)
            active = ops.bool_mask(cfg.n, ops.from_numpy(code, dtype="i8"))
            memb = ops.take(active, ops.as_index(posts))
            t_pot, t_dep = idx[memb], idx[~memb]
            if len(t_pot) > 0:
                st = ops.take(self.store.state, t_pot)
                gate = 1.0 - ops.cascade_resistance(st)
                ops.put(self.store.w, t_pot,
                        ops.take(self.store.w, t_pot) + self.eta * gate)
                st = ops.cascade_push_subset(st)
                if promote:  # salience gate: deepen in EVERY arm (contract)
                    st = ops.cascade_deepen_subset(st, cfg.cascade_n_levels)
                ops.put(self.store.state, t_pot, st)
            if len(t_dep) > 0:  # heterosynaptic depression, cascade-gated
                st_d = ops.take(self.store.state, t_dep)
                gate_d = 1.0 - ops.cascade_resistance(st_d)
                wd = ops.take(self.store.w, t_dep) - self.eta_dep * gate_d
                ops.clip_min_(wd, 0.0)
                ops.put(self.store.w, t_dep, wd)
                ops.put(self.store.state, t_dep, ops.cascade_pull_subset(st_d))
            if promote:
                h = ops.to_numpy(t_pot).astype(np.int64)
                touched_sub = h[h < self.store.substrate_nnz]
                self.behavior.on_promote(self, item_id, code, touched_sub)
        if record:
            self.codes.append(code)
        return item_id

    # -- replay (consolidation pathway: deposits NOT cascade-gated) ----------------
    def replay_deposit(self, item_id: int, lr_multiplier: float, cap: float) -> float:
        ops = self.ops
        idx = self._replay_touch_cache.get(item_id)
        if idx is None:
            code = self.codes[item_id]
            all_idx = self.store.row_contacts(code)
            posts = ops.take(self.store.post_b, all_idx)
            active = ops.bool_mask(self.cfg.n, ops.from_numpy(code, dtype="i8"))
            idx = all_idx[ops.take(active, ops.as_index(posts))]
            self._replay_touch_cache[item_id] = idx
        n_t = int(len(idx))
        if n_t == 0 or cap <= 0.0:
            return 0.0
        dw = lr_multiplier * self.eta
        total = n_t * dw
        scale = min(1.0, cap / total)
        ops.put(self.store.w, idx, ops.take(self.store.w, idx) + dw * scale)
        return total * scale

    # -- store dynamics --------------------------------------------------------------
    def flush_uniform(self) -> None:
        if self.uniform_pending <= 0.0:
            return
        add = self.uniform_pending / self.store.n_contacts
        for _, w, _, _ in self.store.parts():
            w += add
        self.uniform_pending = 0.0

    def store_step(self) -> None:
        cfg, ops = self.cfg, self.ops
        self.flush_uniform()
        half_var = 0.5 * cfg.sigma * cfg.sigma
        for name, w, w0, state in self.store.parts():
            z = ops.randn(len(w))
            w *= ops.exp(cfg.sigma * z - half_var)
            gate = 1.0 - ops.cascade_resistance(state)
            w += cfg.theta_leak * gate * (w0 - w)

    # -- probes -----------------------------------------------------------------------
    def probe_overlap(self, item_id: int, probe_rng: np.random.Generator) -> float:
        """Formal-model RECALL: clamp the cue, relax; discretised as
        recall_iters rounds of (drive from clamped active set -> top-a).
        Overlap is scored on the final completed set alone."""
        code = self.codes[item_id]
        n_cue = max(1, int(round(self.cfg.cue_frac * self.a)))
        cue = np.sort(probe_rng.choice(code, size=n_cue, replace=False))
        active = cue
        top = cue
        for _ in range(max(1, self.cfg.recall_iters)):
            drive = self.store.recall_drive(active)
            top = self.ops.to_numpy(self.ops.topk_idx(drive, self.a)).astype(np.int64)
            active = np.union1d(top, cue)  # clamp: cue units stay active
        return float(np.intersect1d(top, code).size / self.a)


# -- phases ------------------------------------------------------------------------


def run_capacity(system: H2System, seed: int, p: float) -> dict:
    """Sequential load ramp; precision probed per batch; no turnover between
    writes (the load itself is the interference axis; retention carries
    turnover). C = interpolated load where precision crosses the threshold."""
    cfg = system.cfg
    pe = int(round(1.0 / p))
    loads, precisions = [], []
    L = 0
    while True:
        target = cfg.ramp_l0 if L == 0 else max(L + 1, int(round(L * cfg.ramp_ratio)))
        target = min(target, cfg.n)
        for i in range(L, target):
            system.write_item(system.new_code(), promote=(i % pe == 0))
        L = target
        system.store.rebuild()
        system._replay_touch_cache.clear()
        system.flush_uniform()
        prng = host_rng(2, seed, p_code(p), PHASE_CAP, L)
        sample = prng.choice(L, size=min(cfg.m_probe, L), replace=False)
        overlaps = [system.probe_overlap(int(i), prng) for i in sample]
        precision = float(np.mean([o >= cfg.overlap_threshold for o in overlaps]))
        loads.append(L)
        precisions.append(precision)
        if precision < cfg.precision_threshold or L >= cfg.n or system.resource_capped:
            break
    out = capacity_from_ramp(loads, precisions, cfg.precision_threshold)
    out.update({"loads": loads, "precisions": precisions,
                "capacity_ledger": system.behavior.ledger.as_dict(),
                "resource_capped": system.resource_capped})
    return out


def probe_times(t_max: int) -> list[int]:
    ts = [0]
    t = 1
    while t < t_max:
        ts.append(t)
        t *= 2
    ts.append(t_max)
    return ts


def run_retention(system: H2System, seed: int, p: float, l_ref: int) -> dict:
    cfg = system.cfg
    pe = int(round(1.0 / p))
    rebuild_every = max(1, l_ref // 8)
    for i in range(l_ref):
        system.write_item(system.new_code(), promote=(i % pe == 0))
        if (i + 1) % rebuild_every == 0:
            system.store.rebuild()
    system.store.rebuild()
    system._replay_touch_cache.clear()
    prom = [i for i in range(l_ref) if i % pe == 0]
    unprom = [i for i in range(l_ref) if i % pe != 0]
    ts = probe_times(cfg.t_max)
    curves: dict[str, list] = {"t": [], "survival_promoted": [],
                               "survival_unpromoted": [], "mean_overlap_promoted": []}

    def probe(t: int) -> None:
        system.flush_uniform()
        prng = host_rng(2, seed, p_code(p), PHASE_RET, t)
        ps = prom if len(prom) <= cfg.m_probe else \
            sorted(prng.choice(prom, size=cfg.m_probe, replace=False))
        us = unprom if len(unprom) <= cfg.m_probe else \
            sorted(prng.choice(unprom, size=cfg.m_probe, replace=False))
        po = [system.probe_overlap(int(i), prng) for i in ps]
        uo = [system.probe_overlap(int(i), prng) for i in us]
        curves["t"].append(t)
        curves["survival_promoted"].append(
            float(np.mean([o >= cfg.overlap_threshold for o in po])))
        curves["survival_unpromoted"].append(
            float(np.mean([o >= cfg.overlap_threshold for o in uo])) if us else 0.0)
        curves["mean_overlap_promoted"].append(float(np.mean(po)))

    probe(0)
    for t in range(1, cfg.t_max + 1):
        system.store_step()
        for _ in range(cfg.interference_per_step):
            system.write_item(system.new_code(), promote=False, record=False)
        system.behavior.on_store_step(system, t)
        if t in ts:
            probe(t)
    half = t_half_from_survival(curves["t"], curves["survival_promoted"], cfg.t_max)
    half_u = t_half_from_survival(curves["t"], curves["survival_unpromoted"], cfg.t_max)
    return {
        "l_ref": l_ref, "n_promoted": len(prom), "curves": curves,
        "t_half_promoted": half["t_half"], "t_half_censored": half["censored"],
        "t_half_unpromoted": half_u["t_half"],
        "t_half_unpromoted_censored": half_u["censored"],
        "s0_promoted": half["s0"],
        "retention_ledger": system.behavior.ledger.as_dict(),
        "n_contacts_final": system.store.n_contacts,
        "resource_capped": system.resource_capped,
    }


__all__ = ["H2System", "run_capacity", "run_retention", "probe_times",
           "host_rng", "backend_seed", "p_code", "ARM_CODES", "PHASE_CAP", "PHASE_RET"]
