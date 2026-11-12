"""fp16 storage sanity per torch rule (F24): weights may be STORED in fp16,
but every update is computed in fp32 and cast once on write, so dynamics
must stay finite and preserve each rule's direction. Cascade has no float
weight; its analogue is the int8 state contract (dtype and level bounds).
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from brainsim.rules.allocation import AllocationParams
from brainsim.rules.btsp import BTSPParams
from brainsim.rules.cascade import CascadeParams
from brainsim.rules.mb_rule import MBRuleParams
from brainsim.rules.replay import ReplayParams
from brainsim.rules.stdp import STDPParams
from brainsim.rules.torch_rules import allocation as t_alloc
from brainsim.rules.torch_rules import btsp as t_btsp
from brainsim.rules.torch_rules import cascade as t_cascade
from brainsim.rules.torch_rules import mb_rule as t_mb
from brainsim.rules.torch_rules import replay as t_replay
from brainsim.rules.torch_rules import stdp as t_stdp

PRE = torch.tensor([0], dtype=torch.int64)
POST = torch.tensor([0], dtype=torch.int64)


def test_btsp_fp16_finite_and_potentiates():
    params = BTSPParams(k_plus=5.0, k_minus=0.1)
    state = t_btsp.TorchBTSPState.zeros(1, 1, 1, w_init=0.3, w_dtype=torch.float16)
    for t in range(6):
        pre = torch.tensor([t == 0])
        plateau = torch.tensor([t == 2])  # pre then plateau inside the ET window
        t_btsp.step(state, params, pre, plateau, PRE, POST)
    assert state.W.dtype == torch.float16
    assert torch.isfinite(state.W).all()
    assert float(state.W[0]) > 0.3


def test_stdp_fp16_finite_and_direction_preserved():
    params = STDPParams()

    def run(pre_step: int, post_step: int) -> float:
        state = t_stdp.TorchSTDPState.zeros(1, 1, 1, w_init=0.5, w_dtype=torch.float16)
        for t in range(12):
            pre = torch.tensor([t == pre_step])
            post = torch.tensor([t == post_step])
            t_stdp.step(state, params, pre, post, PRE, POST)
        assert state.W.dtype == torch.float16
        assert torch.isfinite(state.W).all()
        return float(state.W[0]) - 0.5

    assert run(pre_step=2, post_step=5) > 0  # pre-before-post potentiates
    assert run(pre_step=5, post_step=2) < 0  # post-before-pre depresses


def test_mb_rule_fp16_finite_and_depresses():
    params = MBRuleParams()
    state = t_mb.TorchMBRuleState.zeros(1, 1, w_dtype=torch.float16)
    for _ in range(20):
        t_mb.step(state, params, torch.tensor([True]), 1.0, PRE)
    assert state.W.dtype == torch.float16
    assert torch.isfinite(state.W).all()
    assert params.beta_floor <= float(state.W[0]) < 0.05


def test_cascade_int8_contract_holds():
    params = CascadeParams()
    gen = torch.Generator()
    gen.manual_seed(0)
    state = torch.full((256,), params.n_levels - 1, dtype=t_cascade.STATE_DTYPE)
    for _ in range(20):
        state = t_cascade.push(state, params, gen)
        state = t_cascade.pull(state, params, gen)
        assert state.dtype == t_cascade.STATE_DTYPE
        assert int(state.min()) >= 0 and int(state.max()) <= 2 * params.n_levels - 1
    deep = t_cascade.deepen(state, params, steps=params.n_levels)
    assert torch.equal(
        t_cascade.depth(deep, params),
        torch.full((256,), params.n_levels, dtype=torch.int64),
    )
    with pytest.raises(ValueError):
        t_cascade.deepen(state, CascadeParams(n_levels=80), steps=1)  # 2n > 128


def test_allocation_fp16_finite_and_boosts_winners():
    params = AllocationParams(tau=100.0, jump=1.0, noise_std=0.1, k=5)
    state = t_alloc.TorchAllocationState.zeros(64, dtype=torch.float16)
    gen = torch.Generator()
    gen.manual_seed(0)
    idx = t_alloc.allocate(state, params, gen)
    assert state.excitability.dtype == torch.float16
    assert torch.isfinite(state.excitability).all()
    assert torch.all(state.excitability[idx].to(torch.float32) >= params.jump - 1e-3)
    before = state.excitability.to(torch.float32).clone()
    t_alloc.decay(state, params, dt_seconds=10.0)
    after = state.excitability.to(torch.float32)
    assert torch.all(after[idx] < before[idx])


def test_replay_fp16_probs_finite_and_ordered():
    gen = torch.Generator()
    gen.manual_seed(0)
    sched = t_replay.TorchReplayScheduler(
        ReplayParams(budget=8000), generator=gen, dtype=torch.float16
    )
    saliences = {0: 0.5, 1: 2.0, 2: 8.0}
    for item_id, s in saliences.items():
        sched.register(item_id, s)
    counts = {i: 0 for i in saliences}
    for e in sched.run_offline_phase():
        counts[e.item_id] += 1
    assert sum(counts.values()) == 8000
    assert counts[0] < counts[1] < counts[2]  # direction preserved through fp16 probs
