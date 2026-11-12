"""CompositeRuntime: the spiking, sharded, plasticity-attached end-to-end
memory loop (H1 composite arm / H3 battery substrate). Scaled sibling of
composite.torch_adapter: same phase semantics as composite.pipeline, but
codes are read off REAL spike counts flowing through the staged LIF network
(ShardedNetwork, post-owned slices, NCCL/gloo exchange), and plasticity is
applied by the T4 torch rules on the plastic subsets.

Execution model (documented deviations from a free-running simulation):
- Phases run stage-clamped spiking windows mirroring composite.pipeline's
  phase structure: the source stage's code cells are driven, the target
  stage integrates through the measured-statistics CSR wiring, and the
  target code is the top-f of cells that actually FIRED, ranked by
  integrated synaptic current (the top-f readout is the feedforward-
  inhibition abstraction, per config). Recurrent retrieval readouts
  (pipeline's relax) are iterated synaptic images over the live plastic
  weights (codes.recurrent_image_code); run_window(inhibit=...) keeps a
  free-running mean-field-inhibition attractor mode for battery arms.
- Between windows, LIF state and delay rings are reset (deterministic; like
  the numpy pipeline, weights are the only carrier of memory across phases).
  Ring reset touches TorchCSRProjection._ring directly: the engine exposes
  no public flush, and zeroing pending deliveries is the entire operation.
- SPMD contract: every rank calls every phase method with IDENTICAL
  arguments (patterns, salience, seeds). Wiring generation is layout-
  invariant (build.py), window counts/currents are all-reduced, and
  allocation uses the global top-k, so codes and metrics agree across ranks;
  the only rank-keyed randomness is cascade/turnover draws (values, not
  mechanisms, differ across world sizes).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.distributed as dist

from brainsim.composite.full import phases
from brainsim.composite.full.build import add_edge_parts, check_fit, edge_specs
from brainsim.composite.full.config import CompositeConfig
from brainsim.engine.torch_core.shard import ShardedNetwork
from brainsim.rules.torch_rules import (
    TorchAllocationState,
    TorchBTSPState,
    TorchMBRuleState,
    TorchReplayScheduler,
    synapse_post_ids,
    synapse_pre_ids,
)


class EdgeState:
    """Per-part runtime state of one logical edge: projection handles plus
    cached int64 COO index vectors (rules gather through them) and the rule
    state bound to each part."""

    def __init__(self, spec, parts: list[str], net: ShardedNetwork, cfg: CompositeConfig):
        self.spec = spec
        self.parts = parts
        self.pre_idx: dict[str, torch.Tensor] = {}
        self.post_idx: dict[str, torch.Tensor] = {}
        self.btsp: dict[str, TorchBTSPState] = {}
        self.mb: dict[str, TorchMBRuleState] = {}
        self.cascade: dict[str, torch.Tensor] = {}
        for name in parts:
            proj = net.projections[name]
            if spec.plastic:
                self.pre_idx[name] = synapse_pre_ids(proj)
                self.post_idx[name] = synapse_post_ids(proj)
            if spec.plastic == "btsp":
                self.btsp[name] = TorchBTSPState.for_projection(proj)
            elif spec.plastic == "mb":
                self.mb[name] = TorchMBRuleState.for_projection(proj)
            elif spec.plastic == "cascade":
                self.cascade[name] = torch.full(
                    (proj.n_synapses,), cfg.cascade.n_levels - 1,
                    dtype=torch.int8, device=proj.device,
                )

    def n_synapses_local(self, net: ShardedNetwork) -> int:
        return sum(net.projections[p].n_synapses for p in self.parts)


class ItemRecord:
    def __init__(self, pattern: np.ndarray, codes: dict[str, np.ndarray],
                 salience: float, protected: bool):
        self.pattern = pattern
        self.codes = codes
        self.salience = salience
        self.protected = protected


class CompositeRuntime:
    def __init__(self, config: CompositeConfig, rank: int = 0,
                 world_size: int = 1, device: str = "cpu") -> None:
        self.config = config
        self.rank, self.world_size = rank, world_size
        self.forecast = check_fit(config, world_size, device)
        self.net = ShardedNetwork(config.n_global, config.lif, rank=rank,
                                  world_size=world_size, device=device)
        self.device = self.net.device
        self.ranges = config.stage_ranges()
        self.edges: dict[str, EdgeState] = {}
        for edge_id, spec in enumerate(edge_specs(config)):
            fp16 = config.fp16_plastic if spec.plastic else config.fp16_static
            parts = add_edge_parts(config, self.net, spec, edge_id, fp16)
            self.edges[spec.name] = EdgeState(spec, parts, self.net, config)

        flo, fhi = self.ranges["formation"]
        a, b = max(flo, self.net.lo), min(fhi, self.net.hi)
        if a > b:  # rank owns no formation cells: empty slice at flo, so
            a = b = flo  # allocation's collective sees a 0-length range
        self.form_local = (a, b)
        self.alloc_state = TorchAllocationState.zeros(b - a, device=self.device)

        gen = torch.Generator()
        gen.manual_seed(config.seed)  # control plane: identical on all ranks
        self.replay = TorchReplayScheduler(config.replay, generator=gen)
        self.rank_gen = torch.Generator(device=self.device)
        self.rank_gen.manual_seed(config.seed * 1_000_003 + rank * 7_919 + 11)

        self.btsp_params = config.btsp_params()
        self.mb_params = config.mb_params()
        self.store_unit = config.store_w_unit()
        self.items: dict[int, ItemRecord] = {}
        self._next_id = 0
        self._drive = torch.zeros(config.n_global, dtype=torch.float32, device=self.device)
        self._drive_dyn = torch.zeros_like(self._drive)
        self.steps_total = 0

    # ---- window machinery ----------------------------------------------------
    def encoding_mode(self, edge_names: tuple[str, ...]):
        """Context manager: scale the named edges' weights by rho_encoding
        for the duration (Hasselmo encoding mode, see config). rho is a
        power of two, so mul/div restores weights bit-identically in fp16
        and fp32."""
        rt = self

        class _Mode:
            def __enter__(self):
                for name in edge_names:
                    for p in rt.edges[name].parts:
                        rt.net.projections[p].weights.mul_(rt.config.rho_encoding)

            def __exit__(self, *exc):
                for name in edge_names:
                    for p in rt.edges[name].parts:
                        rt.net.projections[p].weights.div_(rt.config.rho_encoding)
                return False

        return _Mode()

    def _reset_dynamics(self) -> None:
        st = self.net.state
        st.v.fill_(self.config.lif.v_rest)
        st.I.zero_()
        st.refractory_counter.zero_()
        for proj in self.net.projections.values():
            ring = getattr(proj, "_ring", None)
            if ring is not None:
                ring.zero_()

    def run_window(self, drive: torch.Tensor, steps: int,
                   collect: tuple[str, ...], drive_steps: int | None = None,
                   count_from: int = 0, inhibit: tuple[str, ...] = (),
                   ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """Reset dynamics, step `steps` times driving with `drive` for the
        first `drive_steps` steps, and from step `count_from` on accumulate
        spike counts AND integrated synaptic current (the fine-grained
        rank signal; spike counts over a short window are 0-3-quantized).
        `inhibit` names stages under the mean-field interneuron pool
        (config.inh_current_at_target), engaged only in the FREE phase
        (t >= drive_steps): the drive phase ignites, the free phase
        competes. One small all_reduce per inhibited step when sharded.
        Returns {stage: (GLOBAL int64 counts, GLOBAL fp32 current
        integral)}, all-reduced so every rank sees identical tensors."""
        cfg = self.config
        self._reset_dynamics()
        drive_steps = steps if drive_steps is None else drive_steps
        counts = torch.zeros(self.net.n_local, dtype=torch.int64, device=self.device)
        i_sum = torch.zeros(self.net.n_local, dtype=torch.float32, device=self.device)
        zero = torch.zeros_like(drive)
        decay = float(np.exp(-cfg.lif.dt / cfg.lif.tau_syn))
        inh_A = {s: 0.0 for s in inhibit}
        a_target = {
            s: max(1e-9, cfg.code_size(s) * cfg.assumed_rate * cfg.tau_accum)
            for s in inhibit
        }
        for t in range(steps):
            ext = drive if t < drive_steps else zero
            inh_on = inhibit and t >= drive_steps + cfg.inh_grace_steps
            if inh_on:
                dyn = self._drive_dyn
                dyn.copy_(ext)
                for s in inhibit:
                    plo, phi = self.ranges[s]
                    dyn[plo:phi] -= max(
                        cfg.inh_floor,
                        cfg.inh_current_at_target * inh_A[s] / a_target[s],
                    )
                ext = dyn
            spikes = self.net.step(external_I=ext)
            if inh_on:
                fired = torch.zeros(len(inhibit), dtype=torch.float32, device=self.device)
                sp_t = torch.from_numpy(spikes).to(self.device)
                for j, s in enumerate(inhibit):
                    plo, phi = self.ranges[s]
                    olo, ohi = max(plo, self.net.lo), min(phi, self.net.hi)
                    if olo < ohi:
                        fired[j] = sp_t[olo - self.net.lo: ohi - self.net.lo].sum()
                if self.world_size > 1:
                    dist.all_reduce(fired)
                for j, s in enumerate(inhibit):
                    inh_A[s] = inh_A[s] * decay + float(fired[j])
            if t >= count_from:
                counts += torch.from_numpy(spikes).to(self.device)
                i_sum += self.net.state.I
        self.steps_total += steps
        out: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for stage in collect:
            plo, phi = self.ranges[stage]
            g_c = torch.zeros(phi - plo, dtype=torch.int64, device=self.device)
            g_i = torch.zeros(phi - plo, dtype=torch.float32, device=self.device)
            olo, ohi = max(plo, self.net.lo), min(phi, self.net.hi)
            if olo < ohi:
                g_c[olo - plo: ohi - plo] = counts[olo - self.net.lo: ohi - self.net.lo]
                g_i[olo - plo: ohi - plo] = i_sum[olo - self.net.lo: ohi - self.net.lo]
            if self.world_size > 1:
                dist.all_reduce(g_c)
                dist.all_reduce(g_i)
            out[stage] = (g_c, g_i)
        return out

    def stage_global(self, stage: str, local_values: torch.Tensor) -> torch.Tensor:
        """Map a rank-local per-neuron vector onto the stage's GLOBAL index
        range and all-reduce, so every rank sees the identical stage vector."""
        plo, phi = self.ranges[stage]
        g = torch.zeros(phi - plo, dtype=local_values.dtype, device=self.device)
        olo, ohi = max(plo, self.net.lo), min(phi, self.net.hi)
        if olo < ohi:
            g[olo - plo: ohi - plo] = local_values[olo - self.net.lo: ohi - self.net.lo]
        if self.world_size > 1:
            dist.all_reduce(g)
        return g

    def code_from_window(self, stage: str,
                         window: tuple[torch.Tensor, torch.Tensor]) -> np.ndarray:
        """Top-f cells among those that actually FIRED (an all-silent stage
        yields an EMPTY code, never tie-break junk), ranked by integrated
        synaptic current -- the fine-grained, spike-delivered charge signal;
        raw counts are too quantized over a 30-step window to carry ranks.
        Returns sorted GLOBAL ids (int64 numpy), identical on every rank."""
        counts, i_sum = window
        neg_inf = torch.tensor(float("-inf"), dtype=torch.float64, device=counts.device)
        score = torch.where(counts > 0, i_sum.to(torch.float64), neg_inf)
        k = min(self.config.code_size(stage), score.shape[0])
        top = torch.topk(score, k)
        keep = torch.isfinite(top.values)
        local_ids = top.indices[keep].cpu().numpy()
        return np.sort(local_ids) + self.ranges[stage][0]

    def drive_for(self, ids: np.ndarray | torch.Tensor, gain: float,
                  norm_ref: int | None = None) -> torch.Tensor:
        """Constant-current drive on `ids`. With config.normalize_drive and a
        `norm_ref` (the write-time expected active count), the per-cell gain
        is scaled by norm_ref/len(ids), capped at norm_cap -- divisive gain
        control, so a degraded cue delivers write-comparable total charge."""
        d = self._drive
        d.zero_()
        idx = torch.as_tensor(np.asarray(ids), dtype=torch.int64, device=self.device)
        if idx.numel():
            if norm_ref is not None and self.config.normalize_drive:
                gain *= min(self.config.norm_cap, norm_ref / idx.numel())
            d[idx] = gain
        return d

    # ---- phases (implementations in phases.py) -------------------------------
    def write_phase(self, items) -> list[int]:
        return phases.write_phase(self, items)

    def recall_phase(self, item_ids, drop_fraction: float = 0.5,
                     rng: np.random.Generator | None = None) -> list[dict]:
        return phases.recall_phase(self, item_ids, drop_fraction, rng)

    def recall_codes(self, pattern: np.ndarray) -> dict[str, np.ndarray]:
        return phases.recall_codes(self, pattern)

    def encode_codes(self, pattern: np.ndarray) -> dict[str, np.ndarray]:
        return phases.encode_codes(self, pattern)

    def storage_phase(self, steps: int, noise_sigma: float | None = None,
                      replay_budget: int | None = None) -> dict:
        return phases.storage_phase(self, steps, noise_sigma, replay_budget)

    # ---- metrics ---------------------------------------------------------------
    def snapshot(self) -> dict:
        from brainsim.composite.full import metrics

        return metrics.snapshot(self)


__all__ = ["CompositeRuntime", "ItemRecord", "EdgeState"]
