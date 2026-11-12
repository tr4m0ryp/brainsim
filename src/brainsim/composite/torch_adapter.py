"""Torch composite adapter: the SAME smoke-scale write/recall/storage cycle
as composite.pipeline, with state hosted on TorchNetwork CSR projections and
updated by the torch rules. pipeline.py stays the numpy reference, untouched;
PipelineConfig and builder seeds are shared, so both engines run one schedule.

Correspondence to composite.pipeline, stated once:
- dense masked matrices become CSR projections; (W * mask) @ x becomes a
  gather / index_add_ matvec over the projection's synapses.
- multi-contact duplicate rows in the storage table are deduplicated at
  build time: the dense pipeline collapses them into a single (post, pre)
  mask cell, so one synapse per pair is the equal-semantics translation.
- pattern/cue/turnover randomness comes from numpy Generators seeded exactly
  like run_smoke_cycle, so the cycle is deterministic; per-synapse draw
  shapes differ from the dense (n, n) draws, so values are not bitwise the
  numpy run's (the gate is the protection mechanism, not the RNG stream).
- LIF populations are declared on the TorchNetwork for structural parity;
  the smoke cycle relaxes codes through the projections exactly like
  pipeline.py (the fully spiking composite is composite/full).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from brainsim.circuits import builders
from brainsim.composite.pipeline import PipelineConfig, random_pattern
from brainsim.engine.lif import LIFParams
from brainsim.engine.torch_core.network import TorchNetwork
from brainsim.engine.torch_core.propagate import TorchCSRProjection
from brainsim.rules.allocation import AllocationParams
from brainsim.rules.cascade import CascadeParams
from brainsim.rules.replay import ReplayParams
from brainsim.rules.torch_rules import (
    TorchAllocationState,
    TorchReplayScheduler,
    allocation as t_allocation,
    cascade as t_cascade,
    synapse_post_ids,
    synapse_pre_ids,
)


def _overlap(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = float(torch.linalg.norm(a) * torch.linalg.norm(b))
    if denom < 1e-12:
        return 0.0
    return float(torch.dot(a, b)) / denom


def _sparsify(x: torch.Tensor, f: float) -> torch.Tensor:
    """Top f-fraction of |x| as a bipolar code (pipeline._sparsify semantics;
    zero-sign ties inside the selected set break positive)."""
    n_keep = max(1, int(round(f * x.numel())))
    idx = torch.topk(x.abs(), n_keep).indices
    signs = torch.sign(x.index_select(0, idx))
    signs[signs == 0.0] = 1.0
    out = torch.zeros_like(x)
    out[idx] = signs
    return out


def _unique_pairs(table) -> tuple[np.ndarray, np.ndarray]:
    pairs = np.unique(
        np.stack([np.asarray(table.pre_id), np.asarray(table.post_id)], axis=1), axis=0
    )
    return pairs[:, 0], pairs[:, 1]


@dataclass
class TorchItemRecord:
    pattern: torch.Tensor
    code: torch.Tensor
    protected: bool


class TorchCompositePipeline:
    def __init__(self, config: PipelineConfig, device: str = "cpu") -> None:
        self.config = config
        c = config
        self.rng = np.random.default_rng(c.seed)

        sep_sizes, tbl_sep = builders.make_separation(
            c.n_in, c.n_exp, f=c.f, inhibition=True, seed=c.seed
        )
        form_sizes, tbl_form = builders.make_formation(c.n_exp, c.recurrent_k, seed=c.seed + 1)
        out_sizes, _ = builders.make_output(c.n_output, seed=c.seed + 2)
        relay_sizes, _ = builders.make_relay(c.n_relay)
        store_sizes, tbl_store = builders.make_storage(
            c.n_exp, (-1.0, 0.5), n_states=26,
            multi_contact_tail=c.storage_multi_contact_tail, seed=c.seed + 4,
        )

        self.net = TorchNetwork(seed=c.seed, device=device)
        self.device = self.net.device
        lif = LIFParams()
        for name, n in {**sep_sizes, **form_sizes, **out_sizes, **relay_sizes, **store_sizes}.items():
            self.net.add_population(name, n, lif)

        keep = (
            (tbl_sep.pre_id < c.n_in)
            & (tbl_sep.post_id >= c.n_in)
            & (tbl_sep.post_id < c.n_in + c.n_exp)
        )
        sep_csr = TorchCSRProjection.from_table(
            tbl_sep.pre_id[keep], tbl_sep.post_id[keep] - c.n_in, tbl_sep.weight[keep],
            tbl_sep.sign[keep], n_pre=c.n_in, n_post=c.n_exp, device=self.device,
        )
        self.net.add_projection("sep_ff", "separation_in", "separation_exp", sep_csr)
        self.net.add_projection("form_rec", "formation", "formation", self._plastic_csr(tbl_form, c.n_exp))
        self.net.add_projection("store_rec", "storage", "storage", self._plastic_csr(tbl_store, c.n_exp))
        self._idx = {
            name: (synapse_pre_ids(p.csr), synapse_post_ids(p.csr))
            for name, p in self.net.projections.items()
        }

        self.cascade_params = CascadeParams()
        n_store_syn = self.net.projections["store_rec"].csr.n_synapses
        self.cascade_state = torch.full((n_store_syn,), self.cascade_params.n_levels - 1,
                                        dtype=t_cascade.STATE_DTYPE, device=self.device)

        self.alloc_params = AllocationParams()
        self.alloc_state = TorchAllocationState.zeros(c.n_exp, device=self.device)

        self.replay_params = ReplayParams()
        gen = torch.Generator()
        gen.manual_seed(c.seed)
        self.replay_scheduler = TorchReplayScheduler(self.replay_params, generator=gen)

        self.items: dict[int, TorchItemRecord] = {}
        self._next_id = 0

    def _plastic_csr(self, table, n: int) -> TorchCSRProjection:
        pre, post = _unique_pairs(table)  # builders drop self-loops already
        zeros = np.zeros(pre.size)
        return TorchCSRProjection.from_table(
            pre, post, zeros, np.ones(pre.size), n_pre=n, n_post=n, device=self.device
        )

    def _matvec(self, name: str, x: torch.Tensor) -> torch.Tensor:
        proj = self.net.projections[name].csr
        pre_ids, post_ids = self._idx[name]
        contrib = proj.weights.to(torch.float32) * x.index_select(0, pre_ids)
        out = torch.zeros(proj.n_post, dtype=torch.float32, device=self.device)
        out.index_add_(0, post_ids, contrib)
        return out

    def encode_separation(self, pattern: torch.Tensor) -> torch.Tensor:
        return _sparsify(self._matvec("sep_ff", pattern), self.config.f)

    def _hebbian_write(self, name: str, code: torch.Tensor, eta: float) -> None:
        proj = self.net.projections[name].csr
        pre_ids, post_ids = self._idx[name]
        w = proj.weights.to(torch.float32)
        delta = code.index_select(0, pre_ids) * code.index_select(0, post_ids)
        proj.weights.copy_(
            torch.clamp(w + eta * delta, -self.config.w_max, self.config.w_max)
        )

    def _relax(self, name: str, x: torch.Tensor, iters: int = 1) -> torch.Tensor:
        for _ in range(iters):
            x = _sparsify(self._matvec(name, x), self.config.f)
        return x

    def make_cue(
        self, pattern: torch.Tensor, drop_fraction: float, rng: np.random.Generator | None = None
    ) -> torch.Tensor:
        rng = rng or self.rng
        cue = pattern.to(torch.float32).clone()
        n_drop = int(round(drop_fraction * pattern.numel()))
        idx = rng.choice(pattern.numel(), size=n_drop, replace=False)
        cue[torch.as_tensor(idx, dtype=torch.int64, device=self.device)] = 0.0
        return cue

    def _advance_cascade_on_write(self, code: torch.Tensor, protect: bool) -> None:
        pre_ids, post_ids = self._idx["store_rec"]
        active = code != 0.0
        touched = active.index_select(0, pre_ids) & active.index_select(0, post_ids)
        u = torch.from_numpy(self.rng.random(touched.numel())).to(self.device)
        self.cascade_state = t_cascade.push(
            self.cascade_state, self.cascade_params, u, mask=touched
        )
        if protect:
            deepened = t_cascade.deepen(
                self.cascade_state, self.cascade_params, steps=self.cascade_params.n_levels
            )
            self.cascade_state = torch.where(touched, deepened, self.cascade_state)

    def write_item(self, pattern: torch.Tensor, salience: float, protect: bool = False) -> int:
        item_id = self._next_id
        self._next_id += 1

        code = self.encode_separation(pattern)
        self._hebbian_write("form_rec", code, self.config.eta_form)

        t_allocation.decay(self.alloc_state, self.alloc_params, dt_seconds=1.0)
        active_idx = torch.nonzero(code).flatten()
        self.alloc_state.excitability.index_add_(
            0, active_idx,
            torch.full((active_idx.numel(),), self.alloc_params.jump, device=self.device),
        )

        self._hebbian_write("store_rec", code, self.config.eta_store)
        self._advance_cascade_on_write(code, protect)

        self.items[item_id] = TorchItemRecord(
            pattern=pattern.clone(), code=code.clone(), protected=protect
        )
        self.replay_scheduler.register(item_id, salience)
        return item_id

    def recall(self, cue: torch.Tensor, reference_code: torch.Tensor | None = None) -> float:
        partial_code = self.encode_separation(cue)
        completed = self._relax("form_rec", partial_code)
        completed = self._relax("store_rec", completed)
        reference = reference_code if reference_code is not None else self.encode_separation(cue)
        return _overlap(completed, reference)

    def run_storage_phase(
        self, noise_sigma: float, steps: int, rng: np.random.Generator | None = None
    ) -> None:
        """pipeline.run_storage_phase on per-synapse tensors: leak-dominated
        mean-reverting turnover on touched synapses, scaled down by cascade
        depth resistance, so deep-pushed (protected) synapses stay immune."""
        rng = rng or self.rng
        proj = self.net.projections["store_rec"].csr
        touched = proj.weights.to(torch.float32) != 0.0
        resistance = (
            t_cascade.depth(self.cascade_state, self.cascade_params).to(torch.float32)
            / self.cascade_params.n_levels
        )
        decay_factor = 1.0 - self.config.leak_rate * (1.0 - resistance)
        for _ in range(steps):
            w = proj.weights.to(torch.float32)
            w = torch.where(touched, w * decay_factor, w)
            noise = torch.from_numpy(rng.normal(0.0, 1.0, size=w.numel())).to(
                dtype=torch.float32, device=self.device
            ) * noise_sigma * (1.0 - resistance)
            w = torch.where(touched, w + noise, w)
            proj.weights.copy_(torch.clamp(w, -self.config.w_max, self.config.w_max))


def run_torch_smoke_cycle(seed: int = 0, device: str = "cpu") -> dict:
    """run_smoke_cycle's exact schedule (same PipelineConfig, same seeds, same
    item count, drop fraction, and 60-step leak-dominated storage phase) on
    the torch engine. Code check, never science."""
    config = PipelineConfig(seed=seed)
    pipeline = TorchCompositePipeline(config, device=device)
    rng = np.random.default_rng(seed + 1000)
    dev = pipeline.device

    n_items = 5
    item_ids: list[int] = []
    for i in range(n_items):
        pattern = torch.as_tensor(
            random_pattern(rng, config.n_in), dtype=torch.float32, device=dev
        )
        salience = float(rng.uniform(0.5, 2.0))
        item_ids.append(pipeline.write_item(pattern, salience, protect=(i == 0)))

    def cued_overlap(item_id: int) -> float:
        record = pipeline.items[item_id]
        cue = pipeline.make_cue(record.pattern, 0.5, rng)
        return pipeline.recall(cue, reference_code=record.code)

    overlaps = [cued_overlap(i) for i in item_ids]

    control_overlaps = []
    for _ in range(2):
        unwritten = torch.as_tensor(
            random_pattern(rng, config.n_in), dtype=torch.float32, device=dev
        )
        cue = pipeline.make_cue(unwritten, 0.5, rng)
        control_overlaps.append(
            pipeline.recall(cue, reference_code=pipeline.encode_separation(unwritten))
        )

    protected_id, unprotected_id = item_ids[0], item_ids[1]
    protected_before = cued_overlap(protected_id)
    unprotected_before = cued_overlap(unprotected_id)

    pipeline.run_storage_phase(noise_sigma=0.05, steps=60, rng=rng)

    protected_after = cued_overlap(protected_id)
    unprotected_after = cued_overlap(unprotected_id)

    return {
        "cycle_completed": True,
        "overlaps": overlaps,
        "control_overlaps": control_overlaps,
        "n_items": n_items,
        "protected_overlap_before": protected_before,
        "unprotected_overlap_before": unprotected_before,
        "protected_overlap_after": protected_after,
        "unprotected_overlap_after": unprotected_after,
    }


__all__ = ["TorchCompositePipeline", "TorchItemRecord", "run_torch_smoke_cycle"]
