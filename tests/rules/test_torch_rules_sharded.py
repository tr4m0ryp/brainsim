"""The rules' sharding contract (F25), checked without torch.distributed:
a ShardedNetwork rank sees exactly (a) its post-owned CSR slice with GLOBAL
pre ids and LOCAL post ids and (b) the global fired list from the spike
exchange. A rule stepped independently on two such slices must reproduce
the unsharded run synapse-for-synapse, because every per-synapse update
reads only pre traces (global) and post traces (local slice). Also pins the
for_projection binding: state.W aliases proj.weights, so stepping the rule
mutates the projection in place.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from brainsim.engine.torch_core.propagate import TorchCSRProjection
from brainsim.rules.stdp import STDPParams
from brainsim.rules.torch_rules import btsp as t_btsp
from brainsim.rules.torch_rules import stdp as t_stdp
from brainsim.rules.torch_rules import synapse_post_ids, synapse_pre_ids

N, SPLIT, STEPS = 40, 23, 50  # uneven post-owned split [0,23) / [23,40)


def _global_table(seed: int):
    rng = np.random.default_rng(seed)
    pre = rng.integers(0, N, 400)
    post = rng.integers(0, N, 400)
    pairs = np.unique(np.stack([pre, post], axis=1), axis=0)  # unique keys for matching
    n_syn = pairs.shape[0]
    return pairs[:, 0], pairs[:, 1], np.full(n_syn, 0.5), np.ones(n_syn, dtype=np.int8)


def _projection(pre, post, w, s, lo, hi):
    keep = (post >= lo) & (post < hi)  # exactly ShardedNetwork.add_projection's slice rule
    return TorchCSRProjection.from_table(
        pre[keep], post[keep] - lo, w[keep], s[keep], n_pre=N, n_post=hi - lo, device="cpu"
    )


def test_stdp_on_post_owned_slices_matches_unsharded():
    pre, post, w, s = _global_table(7)
    params = STDPParams()
    full = _projection(pre, post, w, s, 0, N)
    slices = [(_projection(pre, post, w, s, lo, hi), lo) for lo, hi in ((0, SPLIT), (SPLIT, N))]

    full_state = t_stdp.TorchSTDPState.for_projection(full)
    slice_states = [t_stdp.TorchSTDPState.for_projection(p) for p, _ in slices]

    full_idx = (synapse_pre_ids(full), synapse_post_ids(full))
    slice_idx = [(synapse_pre_ids(p), synapse_post_ids(p)) for p, _ in slices]

    rng = np.random.default_rng(8)
    for _ in range(STEPS):
        pre_sp = torch.as_tensor(rng.random(N) < 0.2)  # global fired list, as exchanged
        post_sp = torch.as_tensor(rng.random(N) < 0.2)
        t_stdp.step(full_state, params, pre_sp, post_sp, *full_idx)
        for (proj, lo), state, idx in zip(slices, slice_states, slice_idx):
            t_stdp.step(state, params, pre_sp, post_sp[lo : lo + proj.n_post], *idx)

    # Match synapses across layouts by their unique (pre, global post) key.
    def _keyed(proj, lo):
        key = synapse_pre_ids(proj).numpy() * N + (synapse_post_ids(proj).numpy() + lo)
        order = np.argsort(key)
        return key[order], proj.weights.numpy()[order]

    full_keys, full_w = _keyed(full, 0)
    part_keys, part_w = zip(*[_keyed(p, lo) for p, lo in slices])
    merged = np.concatenate(part_keys)
    order = np.argsort(merged)
    assert np.array_equal(full_keys, merged[order])
    assert np.array_equal(full_w, np.concatenate(part_w)[order])  # bitwise: same fp32 ops


def test_for_projection_binds_weights_in_place():
    pre, post, w, s = _global_table(9)
    proj = _projection(pre, post, w, s, 0, N)
    state = t_btsp.TorchBTSPState.for_projection(proj)
    assert state.W is proj.weights
    before = proj.weights.clone()
    idx = (synapse_pre_ids(proj), synapse_post_ids(proj))
    params = t_btsp.BTSPParams(k_plus=5.0)
    t_btsp.step(state, params, torch.ones(N, dtype=torch.bool), torch.ones(N, dtype=torch.bool), *idx)
    assert state.W is proj.weights  # still aliased after the update
    assert not torch.equal(before, proj.weights)  # projection mutated in place
