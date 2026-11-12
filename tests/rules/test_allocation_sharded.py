"""allocate_global: the 2-rank sharded selection must reproduce the
world_size=1 selection bitwise (same pre-drawn noise, layout-independent),
and boosts must land only on the owning rank."""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
import torch

from brainsim.rules.allocation import AllocationParams
from brainsim.rules.torch_rules.allocation import TorchAllocationState
from brainsim.rules.torch_rules.allocation_sharded import allocate_global

N, K = 64, 7


def _inputs(seed: int = 0):
    rng = np.random.default_rng(seed)
    exc = rng.normal(0.0, 1.0, size=N).astype(np.float32)
    noise = rng.normal(0.0, 0.3, size=N).astype(np.float32)
    return exc, noise


def _single_rank(exc: np.ndarray, noise: np.ndarray):
    st = TorchAllocationState(excitability=torch.tensor(exc.copy()))
    params = AllocationParams(k=K)
    winners = allocate_global(st, params, torch.tensor(noise), (0, N), N)
    return winners.tolist(), st.excitability.clone()


def test_world1_matches_local_topk():
    exc, noise = _inputs()
    winners, after = _single_rank(exc, noise)
    scores = exc + noise
    expect = np.argsort(-scores, kind="stable")[:K].tolist()
    assert winners == expect
    boosted = np.flatnonzero(after.numpy() != exc)
    assert sorted(boosted.tolist()) == sorted(winners)


def _rank_entry(rank: int, world: int, rdzv: str, exc, noise, out):
    import torch.distributed as dist

    os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    dist.init_process_group("gloo", init_method=f"file://{rdzv}",
                            rank=rank, world_size=world)
    try:
        lo, hi = (0, N // 2) if rank == 0 else (N // 2, N)
        st = TorchAllocationState(excitability=torch.tensor(exc[lo:hi].copy()))
        params = AllocationParams(k=K)
        winners = allocate_global(st, params, torch.tensor(noise[lo:hi]),
                                  (lo, hi), N)
        out[rank] = (winners.tolist(), st.excitability.numpy().copy(), lo, hi)
    finally:
        dist.destroy_process_group()


def test_two_rank_matches_single_rank():
    exc, noise = _inputs()
    expect_winners, expect_after = _single_rank(exc, noise)

    ctx = torch.multiprocessing.get_context("spawn")
    with ctx.Manager() as mgr:
        out = mgr.dict()
        rdzv = tempfile.mktemp(prefix="alloc-rdzv-")
        procs = [ctx.Process(target=_rank_entry, args=(r, 2, rdzv, exc, noise, out))
                 for r in range(2)]
        try:
            for p in procs:
                p.start()
            for p in procs:
                p.join(120)
        except OSError as e:  # sandbox without spawn/socket support
            pytest.skip(f"mp.spawn unavailable: {e}")
        if any(p.exitcode != 0 for p in procs):
            pytest.fail(f"rank process failed: {[p.exitcode for p in procs]}")
        res = dict(out)

    for rank in (0, 1):
        winners, after, lo, hi = res[rank]
        assert winners == expect_winners, f"rank {rank} winner mismatch"
        np.testing.assert_array_equal(after, expect_after.numpy()[lo:hi])
