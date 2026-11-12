"""Arm behaviors: how each contract arm spends the protection budget R.

Pricing rule (identical in every arm, engine-invariant): a promoted item's
budget is sum over its touched connections of max(0, k - k0) * mean FROZEN
substrate contact weight of that connection. The redundancy arm realises the
budget as installed contacts at exactly those weights, so its realised spend
equals the priced schedule to float precision; uniform/replay/combined spend
the same schedule through their own mechanisms. The cascade-only arm prices
nothing and spends nothing (contract arm 3).
"""
from __future__ import annotations

import numpy as np

from brainsim.rules import replay as replay_rules

from .store import ContactStore


class Ledger:
    def __init__(self) -> None:
        self.allowance = 0.0
        self.spent = 0.0

    def accrue(self, amount: float) -> None:
        self.allowance += float(amount)

    def spend(self, amount: float) -> None:
        self.spent += float(amount)

    def as_dict(self) -> dict:
        return {"allowance": self.allowance, "spent": self.spent}


def connection_deficits(store: ContactStore, touched_sub_host: np.ndarray, k: int):
    """Group an item's touched SUBSTRATE contacts into connections; return
    (conn_pre, conn_post, deficit=k-k0 clipped, wbar substrate mean weight).
    Every contact of a touched connection is touched (same endpoints), so the
    within-touched count IS k0."""
    pre = store.pre_host[touched_sub_host].astype(np.int64)
    post = store.post_host[touched_sub_host].astype(np.int64)
    key = pre * store.n + post
    order = np.argsort(key, kind="stable")
    key_s = key[order]
    w_s = store.w0_sub_host[touched_sub_host][order].astype(np.float64)
    boundary = np.ones(key_s.size, dtype=bool)
    boundary[1:] = key_s[1:] != key_s[:-1]
    group = np.cumsum(boundary) - 1
    k0 = np.bincount(group)
    wsum = np.bincount(group, weights=w_s)
    wbar = wsum / k0
    conn_key = key_s[boundary]
    deficit = np.maximum(0, k - k0)
    return (conn_key // store.n).astype(np.int32), (conn_key % store.n).astype(np.int32), \
        deficit.astype(np.int64), wbar


def priced_cost(deficit: np.ndarray, wbar: np.ndarray) -> float:
    return float(np.sum(deficit * wbar))


class ArmBehavior:
    """Base: prices the schedule but spends nothing (cascade-only)."""
    name = "cascade_only"
    k: int | None = None

    def __init__(self, cfg, deep_steps: int) -> None:
        self.cfg = cfg
        self.deep_steps = deep_steps
        self.ledger = Ledger()

    def on_promote(self, system, item_id: int, code: np.ndarray,
                   touched_sub_host: np.ndarray) -> None:
        pass

    def on_store_step(self, system, t: int) -> None:
        pass


class RedundancyArm(ArmBehavior):
    name = "redundancy"

    def __init__(self, cfg, deep_steps: int, k: int) -> None:
        super().__init__(cfg, deep_steps)
        self.k = k

    def _install(self, system, touched_sub_host, k) -> float:
        cp, cq, deficit, wbar = connection_deficits(system.store, touched_sub_host, k)
        cost = priced_cost(deficit, wbar)
        need = deficit > 0
        pre_new = np.repeat(cp[need], deficit[need])
        post_new = np.repeat(cq[need], deficit[need])
        w_new = np.repeat(wbar[need], deficit[need]).astype(np.float32)
        if system.store.n_contacts + pre_new.size > self.cfg.max_contacts:
            system.resource_capped = True
            return 0.0
        system.store.append_block(pre_new, post_new, w_new, self.deep_steps)
        return cost

    def on_promote(self, system, item_id, code, touched_sub_host) -> None:
        _, _, deficit, wbar = connection_deficits(system.store, touched_sub_host, self.k)
        self.ledger.accrue(priced_cost(deficit, wbar))
        self.ledger.spend(self._install(system, touched_sub_host, self.k))


class UniformArm(ArmBehavior):
    """Matched-R control: the identical priced schedule, spread uniformly over
    every contact (structure-free strengthening)."""
    name = "uniform"

    def __init__(self, cfg, deep_steps: int, k: int) -> None:
        super().__init__(cfg, deep_steps)
        self.k = k

    def on_promote(self, system, item_id, code, touched_sub_host) -> None:
        _, _, deficit, wbar = connection_deficits(system.store, touched_sub_host, self.k)
        cost = priced_cost(deficit, wbar)
        self.ledger.accrue(cost)
        self.ledger.spend(cost)
        system.uniform_pending += cost  # flushed as +cost/nnz on every contact


class ReplayArm(ArmBehavior):
    """Same R spent on prioritised replay (rehearsal instead of structure).
    Replay deposits bypass cascade gating (consolidation pathway, F4/F11);
    a spend controller reaches the allowance by replay_spend_horizon * t_max,
    scaling the final event so realised spend equals the schedule exactly."""
    name = "replay"
    max_events_per_step = 64

    def __init__(self, cfg, deep_steps: int, k: int, rng: np.random.Generator) -> None:
        super().__init__(cfg, deep_steps)
        self.k = k
        self.scheduler = replay_rules.ReplayScheduler(
            replay_rules.ReplayParams(budget=1, lr_multiplier=cfg.replay_lr), rng)

    def on_promote(self, system, item_id, code, touched_sub_host) -> None:
        _, _, deficit, wbar = connection_deficits(system.store, touched_sub_host, self.k)
        self.ledger.accrue(priced_cost(deficit, wbar))
        self.scheduler.register(item_id, salience=1.0)

    def on_store_step(self, system, t: int) -> None:
        horizon = max(1.0, self.cfg.replay_spend_horizon * self.cfg.t_max)
        target = self.ledger.allowance * min(1.0, t / horizon)
        events = 0
        while self.ledger.spent < target - 1e-12 and events < self.max_events_per_step:
            ev = self.scheduler.run_offline_phase(budget=1)
            if not ev:
                break
            spent = system.replay_deposit(ev[0].item_id, ev[0].lr_multiplier,
                                          cap=target - self.ledger.spent)
            if spent <= 0.0:
                break
            self.ledger.spend(spent)
            events += 1


class CombinedArm(ArmBehavior):
    """Redundancy at k' = 1 + round_half_up(alpha*(k-1)) plus replay funded by
    the remainder of the SAME priced schedule (matched total budget)."""
    name = "combined"

    def __init__(self, cfg, deep_steps: int, k: int, rng: np.random.Generator) -> None:
        super().__init__(cfg, deep_steps)
        self.k = k
        self.k_struct = 1 + int(np.floor(cfg.combined_alpha * (k - 1) + 0.5))
        self._red = RedundancyArm(cfg, deep_steps, self.k_struct)
        self._rep = ReplayArm(cfg, deep_steps, k, rng)

    def on_promote(self, system, item_id, code, touched_sub_host) -> None:
        _, _, deficit_full, wbar = connection_deficits(system.store, touched_sub_host, self.k)
        total = priced_cost(deficit_full, wbar)
        self.ledger.accrue(total)
        spent_struct = self._red._install(system, touched_sub_host, self.k_struct)
        self.ledger.spend(spent_struct)
        self._rep.scheduler.register(item_id, salience=1.0)
        self._rep.ledger.accrue(total - spent_struct)

    def on_store_step(self, system, t: int) -> None:
        before = self._rep.ledger.spent
        self._rep.on_store_step(system, t)
        self.ledger.spend(self._rep.ledger.spent - before)


def make_behavior(arm: str, cfg, deep_steps: int, k: int | None,
                  arm_rng: np.random.Generator) -> ArmBehavior:
    if arm == "cascade_only":
        return ArmBehavior(cfg, deep_steps)
    if arm == "redundancy":
        return RedundancyArm(cfg, deep_steps, k)
    if arm == "uniform":
        return UniformArm(cfg, deep_steps, k)
    if arm == "replay":
        return ReplayArm(cfg, deep_steps, k, arm_rng)
    if arm == "combined":
        return CombinedArm(cfg, deep_steps, k, arm_rng)
    raise ValueError(f"unknown arm {arm!r}")


__all__ = ["Ledger", "ArmBehavior", "RedundancyArm", "UniformArm", "ReplayArm",
           "CombinedArm", "make_behavior", "connection_deficits", "priced_cost"]
