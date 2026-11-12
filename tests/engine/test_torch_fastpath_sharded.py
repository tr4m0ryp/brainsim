"""2-rank gloo: fast exchange raster == slow exchange raster == world1 raster.

Same conventions as test_torch_shard (mp.spawn, file:// init, gloo over lo,
sandbox skip-guard, dyadic weights so fp32 sums are exact regardless of
accumulation grouping). Each worker runs BOTH paths in one process group --
first the step() loop (two all_gathers per step), then run() with the
single-collective FastExchange -- on identically built nets, and the parent
stitches both against the world_size=1 reference. The fast net's capacity
is set to the TIGHT bound (max fired per rank per step, computed from the
world1 raster), so the test also pins the cap == max boundary: full buffers,
no overflow.
"""
import os
from datetime import timedelta

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import torch.distributed as dist
import torch.multiprocessing as mp

from brainsim.engine.lif import LIFParams
from brainsim.engine.torch_core.shard import ShardedNetwork, partition_ranges

N, K, STEPS, WORLD = 64, 6, 120, 2

pytestmark = pytest.mark.skipif(
    not (dist.is_available() and dist.is_gloo_available()),
    reason="torch.distributed gloo backend unavailable",
)

_SANDBOX_HINTS = ("socket", "address", "connect", "bind", "network", "unreachable", "no route")


def _dyadic_table(seed: int):
    rng = np.random.default_rng(seed)
    pre = np.concatenate([rng.choice(N, size=K, replace=False) for _ in range(N)])
    post = np.repeat(np.arange(N), K)
    weight = rng.integers(1, 32, size=pre.size) * 0.125
    sign = np.where(rng.random(pre.size) < 0.8, 1, -1).astype(np.int8)
    return pre, post, weight, sign


def _drives() -> list[np.ndarray]:
    rng = np.random.default_rng(99)
    return [rng.integers(0, 17, size=N) * 0.25 for _ in range(STEPS)]


def _build(rank: int, world_size: int, cap: int | None = None) -> ShardedNetwork:
    net = ShardedNetwork(N, LIFParams(), rank=rank, world_size=world_size,
                         device="cpu", fast_exchange_cap=cap)
    pre, post, w, s = _dyadic_table(21)
    net.add_projection("rec", pre, post, w, s, delay_steps=1)
    pre, post, w, s = _dyadic_table(22)
    net.add_projection("slow", pre, post, w, s, delay_steps=3)
    return net


def _worker(rank: int, out_dir: str, cap: int) -> None:
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    dist.init_process_group(
        "gloo",
        init_method=f"file://{out_dir}/pg",
        rank=rank,
        world_size=WORLD,
        timeout=timedelta(seconds=120),
    )
    try:
        drives = _drives()
        # Slow path: step() loop, two all_gathers per step.
        net_slow = _build(rank, WORLD)
        slow = np.asarray([net_slow.step(external_I=d) for d in drives])

        # Fast path: run() with the single-collective exchange at the TIGHT
        # cap, drive passed as the LOCAL fp32 slice per step.
        net_fast = _build(rank, WORLD, cap=cap)
        lo, hi = net_fast.local_range
        local = torch.as_tensor(
            np.stack(drives)[:, lo:hi], dtype=torch.float32)
        counts, fast = net_fast.run(
            STEPS, drive=lambda i: local[i], record="raster")

        np.save(os.path.join(out_dir, f"slow_{rank}.npy"), slow)
        np.save(os.path.join(out_dir, f"fast_{rank}.npy"), fast)
        np.save(os.path.join(out_dir, f"counts_{rank}.npy"), counts)
    finally:
        dist.destroy_process_group()


def test_fast_exchange_matches_slow_and_world1(tmp_path):
    assert not dist.is_initialized()
    net1 = _build(0, 1)
    single = np.asarray([net1.step(external_I=d) for d in _drives()])
    assert single.sum() > 0

    # Tight uniform capacity: the max fired per rank per step across ranks.
    cap = max(
        int(single[:, lo:hi].sum(axis=1).max())
        for lo, hi in partition_ranges(N, WORLD)
    )
    assert cap >= 1

    try:
        mp.spawn(_worker, args=(str(tmp_path), cap), nprocs=WORLD, join=True)
    except Exception as exc:
        msg = str(exc).lower()
        if any(hint in msg for hint in _SANDBOX_HINTS):
            pytest.skip(f"distributed spawn unavailable in this sandbox: {exc}")
        raise

    stitched_slow = np.zeros_like(single)
    stitched_fast = np.zeros_like(single)
    for rank, (lo, hi) in enumerate(partition_ranges(N, WORLD)):
        stitched_slow[:, lo:hi] = np.load(tmp_path / f"slow_{rank}.npy")
        fast_r = np.load(tmp_path / f"fast_{rank}.npy")
        stitched_fast[:, lo:hi] = fast_r
        counts_r = np.load(tmp_path / f"counts_{rank}.npy")
        assert np.array_equal(counts_r, fast_r.sum(axis=1))  # LOCAL counts

    assert np.array_equal(single, stitched_slow)
    assert np.array_equal(single, stitched_fast)
