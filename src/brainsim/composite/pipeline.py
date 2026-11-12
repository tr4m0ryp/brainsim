"""Wire separation -> formation -> output -> relay -> storage into one
composite write/recall/store smoke cycle (Architecture, Build Plan A5).

Simplification, stated once here rather than scattered as inline comments:
this smoke-scale composite relaxes sparse-coded attractor patterns through
the FIXED wiring skeletons that circuits.builders produces (only weights on
existing synapses move, per the Formal Model's W_plastic subset E), instead
of driving full per-millisecond LIF spiking across all five stages.
engine.lif/engine.propagate/engine.network are validated independently by
their own unit tests; the fully spiking composite is composite/full (Build Plan
A5/A6 split names the smoke cycle, not full engine integration, as the A5
deliverable). Output/relay are content-preserving pass-throughs at this
scale -- BTSP's one-shot binding and the relay's cross-stage sensitivity
battery (T10) are exercised at rule level (rules/btsp.py) and get non-trivial
inter-stage transforms once stage sizes stop matching at full scale.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from brainsim.circuits import builders
from brainsim.rules import allocation, cascade, replay
from brainsim.schema.tables import ConnectivityTable


def random_pattern(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.choice(np.array([-1.0, 1.0]), size=n)


def _overlap(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def _sparsify(x: np.ndarray, f: float) -> np.ndarray:
    """Keep the top f-fraction of |x| as a bipolar code, zero the rest. Ties
    at exactly 0 within the selected set break positive (they were selected
    on magnitude; never-selected entries must stay exactly 0).
    """
    n_keep = max(1, int(round(f * x.size)))
    idx = np.argpartition(-np.abs(x), n_keep - 1)[:n_keep]
    signs = np.sign(x[idx])
    signs[signs == 0.0] = 1.0
    out = np.zeros_like(x)
    out[idx] = signs
    return out


def _mask_from_table(table: ConnectivityTable, n: int) -> np.ndarray:
    mask = np.zeros((n, n), dtype=bool)
    mask[table.post_id, table.pre_id] = True
    return mask


def _feedforward_dense(table: ConnectivityTable, n_in: int, n_exp: int) -> np.ndarray:
    keep = (table.pre_id < n_in) & (table.post_id >= n_in) & (table.post_id < n_in + n_exp)
    W = np.zeros((n_exp, n_in))
    W[table.post_id[keep] - n_in, table.pre_id[keep]] = table.weight[keep] * table.sign[keep]
    return W


@dataclass
class PipelineConfig:
    n_in: int = 200
    n_exp: int = 300  # also formation/storage dimensionality at this smoke scale
    f: float = 0.1  # coding level (higher than the 0.03 design default so tiny D stays robust)
    recurrent_k: int = 30
    storage_multi_contact_tail: int = 5
    n_output: int = 150
    n_relay: int = 150
    eta_form: float = 0.5
    eta_store: float = 0.3
    w_max: float = 3.0
    # Leak-dominated turnover (leak >> noise): erosion must be mean-reverting,
    # or accumulated noise random-walks every touched synapse to +-w_max and
    # buries the protected item's preserved weights at recall time.
    leak_rate: float = 0.05
    seed: int = 0


@dataclass
class ItemRecord:
    pattern: np.ndarray
    code: np.ndarray
    protected: bool


class CompositePipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        c = config
        self.rng = np.random.default_rng(c.seed)

        _, tbl_sep = builders.make_separation(c.n_in, c.n_exp, f=c.f, inhibition=True, seed=c.seed)
        _, tbl_form = builders.make_formation(c.n_exp, c.recurrent_k, seed=c.seed + 1)
        self.output_pop_sizes, _ = builders.make_output(c.n_output, seed=c.seed + 2)
        self.relay_pop_sizes, _ = builders.make_relay(c.n_relay)
        _, tbl_store = builders.make_storage(
            c.n_exp, (-1.0, 0.5), n_states=26, multi_contact_tail=c.storage_multi_contact_tail, seed=c.seed + 4
        )

        self.W_sep = _feedforward_dense(tbl_sep, c.n_in, c.n_exp)
        self.form_mask = _mask_from_table(tbl_form, c.n_exp)
        self.store_mask = _mask_from_table(tbl_store, c.n_exp)
        self.W_form = np.zeros((c.n_exp, c.n_exp))
        self.W_store = np.zeros((c.n_exp, c.n_exp))

        self.cascade_params = cascade.CascadeParams()
        self.cascade_state = np.full((c.n_exp, c.n_exp), self.cascade_params.n_levels - 1, dtype=np.uint8)

        self.alloc_params = allocation.AllocationParams()
        self.alloc_state = allocation.AllocationState.zeros(c.n_exp)

        self.replay_params = replay.ReplayParams()
        self.replay_scheduler = replay.ReplayScheduler(self.replay_params, self.rng)

        self.items: dict[int, ItemRecord] = {}
        self._next_id = 0

    def encode_separation(self, pattern: np.ndarray) -> np.ndarray:
        drive = self.W_sep @ pattern
        return _sparsify(drive, self.config.f)

    def _hebbian_write(self, W: np.ndarray, mask: np.ndarray, code: np.ndarray, eta: float) -> None:
        delta = np.outer(code, code) * mask
        np.fill_diagonal(delta, 0.0)
        W += eta * delta
        np.clip(W, -self.config.w_max, self.config.w_max, out=W)

    def _relax(self, W: np.ndarray, mask: np.ndarray, x: np.ndarray, iters: int = 1) -> np.ndarray:
        """A single sparsified linear readout per stage. The masked Hebbian
        matrices here are sparse and asymmetric (unlike a dense symmetric
        Hopfield network), so iterating this map does not contract toward a
        stable fixed point -- it drifts. One pass per stage is the
        associative-readout analogue used at this smoke scale; iterative
        settling dynamics belong to the full spiking runtime (composite/full).
        """
        for _ in range(iters):
            x = _sparsify((W * mask) @ x, self.config.f)
        return x

    def make_cue(self, pattern: np.ndarray, drop_fraction: float, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or self.rng
        cue = pattern.astype(np.float64).copy()
        n_drop = int(round(drop_fraction * pattern.size))
        idx = rng.choice(pattern.size, size=n_drop, replace=False)
        cue[idx] = 0.0
        return cue

    def _advance_cascade_on_write(self, code: np.ndarray, protect: bool) -> None:
        active = code != 0.0
        touched = np.outer(active, active) & self.store_mask
        self.cascade_state = cascade.push(self.cascade_state, self.cascade_params, self.rng, mask=touched)
        if protect:
            deepened = cascade.deepen(self.cascade_state, self.cascade_params, steps=self.cascade_params.n_levels)
            self.cascade_state = np.where(touched, deepened, self.cascade_state)

    def write_item(self, pattern: np.ndarray, salience: float, protect: bool = False) -> int:
        item_id = self._next_id
        self._next_id += 1

        code = self.encode_separation(pattern)
        self._hebbian_write(self.W_form, self.form_mask, code, self.config.eta_form)

        allocation.decay(self.alloc_state, self.alloc_params, dt_seconds=1.0)
        active_idx = np.nonzero(code)[0]
        self.alloc_state.excitability[active_idx] += self.alloc_params.jump

        self._hebbian_write(self.W_store, self.store_mask, code, self.config.eta_store)
        self._advance_cascade_on_write(code, protect)

        self.items[item_id] = ItemRecord(pattern=pattern.copy(), code=code.copy(), protected=protect)
        self.replay_scheduler.register(item_id, salience)
        return item_id

    def recall(self, cue: np.ndarray, reference_code: np.ndarray | None = None) -> float:
        partial_code = self.encode_separation(cue)
        completed = self._relax(self.W_form, self.form_mask, partial_code)
        completed = self._relax(self.W_store, self.store_mask, completed)
        reference = reference_code if reference_code is not None else self.encode_separation(cue)
        return _overlap(completed, reference)

    def run_storage_phase(self, noise_sigma: float, steps: int, rng: np.random.Generator | None = None) -> None:
        """Turnover noise (Formal Model STORE op) applied only to synapses a
        write has actually touched -- never-written synapses carry no weight
        to turn over. Noise/leak on a touched synapse is scaled down by its
        cascade-state resistance (depth/n_levels), so the H01 redundancy
        motif's cascade deep-push (write_item(..., protect=True)) makes those
        synapses immune while everything else erodes.
        """
        rng = rng or self.rng
        touched = self.store_mask & (self.W_store != 0.0)
        depth = cascade.depth(self.cascade_state, self.cascade_params).astype(np.float64)
        resistance = depth / self.cascade_params.n_levels  # deeper cascade state -> more resistant
        for _ in range(steps):
            decay_factor = 1.0 - self.config.leak_rate * (1.0 - resistance)
            self.W_store = np.where(touched, self.W_store * decay_factor, self.W_store)
            noise = rng.normal(0.0, 1.0, size=self.W_store.shape) * noise_sigma * (1.0 - resistance)
            self.W_store = np.where(touched, self.W_store + noise, self.W_store)
            np.clip(self.W_store, -self.config.w_max, self.config.w_max, out=self.W_store)


def run_smoke_cycle(seed: int = 0) -> dict:
    config = PipelineConfig(seed=seed)
    pipeline = CompositePipeline(config)
    rng = np.random.default_rng(seed + 1000)

    n_items = 5
    item_ids: list[int] = []
    for i in range(n_items):
        pattern = random_pattern(rng, config.n_in)
        salience = float(rng.uniform(0.5, 2.0))
        item_ids.append(pipeline.write_item(pattern, salience, protect=(i == 0)))

    def cued_overlap(item_id: int) -> float:
        record = pipeline.items[item_id]
        cue = pipeline.make_cue(record.pattern, 0.5, rng)
        return pipeline.recall(cue, reference_code=record.code)

    overlaps = [cued_overlap(i) for i in item_ids]

    control_overlaps = []
    for _ in range(2):
        unwritten = random_pattern(rng, config.n_in)
        cue = pipeline.make_cue(unwritten, 0.5, rng)
        control_overlaps.append(pipeline.recall(cue, reference_code=pipeline.encode_separation(unwritten)))

    protected_id, unprotected_id = item_ids[0], item_ids[1]
    protected_before = cued_overlap(protected_id)
    unprotected_before = cued_overlap(unprotected_id)

    # 60 leak-dominated steps shrink unprotected weights to ~7% while cascade
    # protection holds its synapses at 100%: the deterministic mechanism must
    # dominate finite-size recall noise at this 300-unit smoke scale, or the
    # gate tests the RNG instead of the code path.
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


__all__ = ["PipelineConfig", "CompositePipeline", "ItemRecord", "random_pattern", "run_smoke_cycle"]
