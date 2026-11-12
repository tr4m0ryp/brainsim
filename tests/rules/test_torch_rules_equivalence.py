"""numpy rules vs torch rules on identical tiny histories (fp32, cpu).

Deterministic rules (BTSP / STDP / MB) are compared directly: same spike
history in, max |W_np - W_torch| < 1e-5 out (numpy integrates in fp64, torch
in fp32; short histories keep the systematic per-step rounding drift orders
below the gate). Stochastic rules are compared under identical random
inputs via twin-seeded numpy Generators: the numpy rule consumes its own
stream, the torch rule receives the mirror stream as tensors. Cascade states
must then match EXACTLY (integer states, float64 probability table on both
sides); allocation must pick identical indices. Replay sampling is
distribution-equivalent by design (torch.multinomial is a different stream
from Generator.choice), so its check is statistical plus exact degenerate
paths.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from brainsim.rules import allocation as np_alloc
from brainsim.rules import btsp as np_btsp
from brainsim.rules import cascade as np_cascade
from brainsim.rules import mb_rule as np_mb
from brainsim.rules import replay as np_replay
from brainsim.rules import stdp as np_stdp
from brainsim.rules.torch_rules import allocation as t_alloc
from brainsim.rules.torch_rules import btsp as t_btsp
from brainsim.rules.torch_rules import cascade as t_cascade
from brainsim.rules.torch_rules import mb_rule as t_mb
from brainsim.rules.torch_rules import replay as t_replay
from brainsim.rules.torch_rules import stdp as t_stdp

TOL = 1e-5
N_PRE, N_POST, N_SYN, STEPS = 7, 5, 24, 80


def _wiring(rng):
    pre_idx = rng.integers(0, N_PRE, N_SYN)
    post_idx = rng.integers(0, N_POST, N_SYN)
    return pre_idx, post_idx, torch.as_tensor(pre_idx), torch.as_tensor(post_idx)


def test_btsp_weights_match_numpy():
    rng = np.random.default_rng(0)
    pre_idx, post_idx, pre_t, post_t = _wiring(rng)
    params = np_btsp.BTSPParams(k_plus=5.0, k_minus=0.5)
    ns = np_btsp.BTSPState.zeros(N_PRE, N_POST, N_SYN, w_init=0.4)
    ts = t_btsp.TorchBTSPState.zeros(N_PRE, N_POST, N_SYN, w_init=0.4)
    for _ in range(STEPS):
        pre = rng.random(N_PRE) < 0.15
        plateau = rng.random(N_POST) < 0.08
        np_btsp.step(ns, params, pre, plateau, pre_idx, post_idx)
        t_btsp.step(ts, params, torch.as_tensor(pre), torch.as_tensor(plateau), pre_t, post_t)
    assert float(np.abs(ns.W - ts.W.numpy()).max()) < TOL


def test_stdp_weights_match_numpy():
    rng = np.random.default_rng(1)
    pre_idx, post_idx, pre_t, post_t = _wiring(rng)
    params = np_stdp.STDPParams()
    ns = np_stdp.STDPState.zeros(N_PRE, N_POST, N_SYN, w_init=0.5)
    ts = t_stdp.TorchSTDPState.zeros(N_PRE, N_POST, N_SYN, w_init=0.5)
    for _ in range(STEPS):
        pre = rng.random(N_PRE) < 0.2
        post = rng.random(N_POST) < 0.2
        np_stdp.step(ns, params, pre, post, pre_idx, post_idx)
        t_stdp.step(ts, params, torch.as_tensor(pre), torch.as_tensor(post), pre_t, post_t)
    assert float(np.abs(ns.W - ts.W.numpy()).max()) < TOL


def test_mb_rule_weights_match_numpy():
    rng = np.random.default_rng(2)
    pre_idx, _, pre_t, _ = _wiring(rng)
    params = np_mb.MBRuleParams()
    ns = np_mb.MBRuleState.zeros(N_PRE, N_SYN)
    ts = t_mb.TorchMBRuleState.zeros(N_PRE, N_SYN)
    for _ in range(STEPS):
        kc = rng.random(N_PRE) < 0.25
        dan = rng.random(N_SYN)
        np_mb.step(ns, params, kc, dan, pre_idx)
        t_mb.step(ts, params, torch.as_tensor(kc), torch.as_tensor(dan), pre_t)
    assert float(np.abs(ns.W - ts.W.numpy()).max()) < TOL


@pytest.mark.parametrize(
    "params",
    [np_cascade.CascadeParams(), np_cascade.CascadeParams(n_levels=6, q1=0.7, x=0.6)],
)
def test_cascade_states_match_numpy_exactly(params):
    n = 500
    rng_hist = np.random.default_rng(3)
    rng_np = np.random.default_rng(4)
    rng_mirror = np.random.default_rng(4)  # twin stream: what np push/pull consumes
    s_np = rng_hist.integers(0, 2 * params.n_levels, n).astype(np.uint8)
    s_t = torch.as_tensor(s_np.astype(np.int8))

    p_np = np_cascade.transition_prob(s_np, params)
    p_t = t_cascade.transition_prob(s_t, params)
    assert np.array_equal(p_np, p_t.numpy())  # bitwise, float64 table on both sides

    for _ in range(30):
        mask = rng_hist.random(n) < 0.5
        s_np = np_cascade.push(s_np, params, rng_np, mask=mask)
        u = rng_mirror.random(s_np.shape)
        s_t = t_cascade.push(s_t, params, torch.from_numpy(u), mask=torch.from_numpy(mask))
        assert np.array_equal(s_np.astype(np.int64), s_t.numpy().astype(np.int64))

        mask = rng_hist.random(n) < 0.5
        s_np = np_cascade.pull(s_np, params, rng_np, mask=mask)
        u = rng_mirror.random(s_np.shape)
        s_t = t_cascade.pull(s_t, params, torch.from_numpy(u), mask=torch.from_numpy(mask))
        assert np.array_equal(s_np.astype(np.int64), s_t.numpy().astype(np.int64))

    for steps in (1, 3, params.n_levels):
        d_np = np_cascade.deepen(s_np, params, steps=steps)
        d_t = t_cascade.deepen(s_t, params, steps=steps)
        assert np.array_equal(d_np.astype(np.int64), d_t.numpy().astype(np.int64))
    assert np.array_equal(np_cascade.depth(s_np, params), t_cascade.depth(s_t, params).numpy())
    assert np.array_equal(
        np_cascade.effective_weight(s_np, params),
        t_cascade.effective_weight(s_t, params).numpy(),
    )


def test_allocation_matches_numpy():
    params = np_alloc.AllocationParams(tau=100.0, jump=1.0, noise_std=0.1, k=8)
    n = 150
    ns = np_alloc.AllocationState.zeros(n)
    ts = t_alloc.TorchAllocationState.zeros(n)
    rng_np = np.random.default_rng(5)
    rng_mirror = np.random.default_rng(5)  # twin stream: what np allocate consumes
    for _ in range(4):
        idx_np = np_alloc.allocate(ns, params, rng_np)
        noise = rng_mirror.normal(0.0, params.noise_std, size=n)
        idx_t = t_alloc.allocate(ts, params, torch.from_numpy(noise))
        assert np.array_equal(idx_np, idx_t.numpy())  # same winners, same descending order
        np_alloc.decay(ns, params, dt_seconds=3.0)
        t_alloc.decay(ts, params, dt_seconds=3.0)
    assert float(np.abs(ns.excitability - ts.excitability.numpy()).max()) < TOL


def test_replay_distribution_matches_numpy():
    params = np_replay.ReplayParams(budget=20000, lr_multiplier=0.05)
    np_sched = np_replay.ReplayScheduler(params, np.random.default_rng(0))
    gen = torch.Generator()
    gen.manual_seed(0)
    t_sched = t_replay.TorchReplayScheduler(params, generator=gen)
    saliences = {0: 1.0, 1: 2.0, 2: 4.0, 3: 8.0}
    for item_id, s in saliences.items():
        np_sched.register(item_id, s)
        t_sched.register(item_id, s)
    assert np_sched.items() == t_sched.items()

    counts_np = {i: 0 for i in saliences}
    counts_t = {i: 0 for i in saliences}
    for e in np_sched.run_offline_phase():
        counts_np[e.item_id] += 1
    for e in t_sched.run_offline_phase():
        assert e.lr_multiplier == params.lr_multiplier
        counts_t[e.item_id] += 1
    total = sum(saliences.values())
    for item_id, s in saliences.items():
        assert abs(counts_t[item_id] / params.budget - s / total) < 0.02
        assert abs(counts_t[item_id] - counts_np[item_id]) / params.budget < 0.03


def test_replay_degenerate_paths_match_numpy():
    gen = torch.Generator()
    gen.manual_seed(1)
    t_sched = t_replay.TorchReplayScheduler(np_replay.ReplayParams(budget=4000), generator=gen)
    assert t_sched.run_offline_phase() == []  # empty scheduler, exact numpy behaviour

    with pytest.raises(ValueError):
        t_sched.register(0, -1.0)

    t_sched.register(0, 0.0)  # zero total salience -> uniform sampling
    t_sched.register(1, 0.0)
    events = t_sched.run_offline_phase()
    frac0 = sum(1 for e in events if e.item_id == 0) / len(events)
    assert abs(frac0 - 0.5) < 0.05

    t_sched.unregister(0)
    assert all(e.item_id == 1 for e in t_sched.run_offline_phase())
