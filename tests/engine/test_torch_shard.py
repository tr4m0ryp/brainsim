"""2-rank gloo sharding vs world_size=1: exact raster equality (F25).

Exactness is by construction, not tolerance: post-owned sharding puts EVERY
synapse of a post neuron on its owner, so each I[post] sums the identical
contribution set; dyadic weights (multiples of 1/8) make those fp32 sums
exact regardless of accumulation order; and rank-ordered all_gather
concatenation reproduces the ascending fired order of the single-rank run.
The 2-process run uses torch.multiprocessing.spawn + gloo with file:// init
(no TCP rendezvous); if the sandbox forbids the transport sockets, the test
skips on the OS error rather than masking real failures.
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

# Substrings of transport errors seen when a sandbox forbids gloo's local
# sockets; anything else (assertion failures included) propagates.
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


def _build(rank: int, world_size: int) -> ShardedNetwork:
    net = ShardedNetwork(N, LIFParams(), rank=rank, world_size=world_size, device="cpu")
    pre, post, w, s = _dyadic_table(21)
    net.add_projection("rec", pre, post, w, s, delay_steps=1)
    pre, post, w, s = _dyadic_table(22)
    net.add_projection("slow", pre, post, w, s, delay_steps=3)
    return net


def _run(net: ShardedNetwork) -> np.ndarray:
    return np.asarray([net.step(external_I=d) for d in _drives()])


def _worker(rank: int, out_dir: str) -> None:
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    dist.init_process_group(
        "gloo",
        init_method=f"file://{out_dir}/pg",
        rank=rank,
        world_size=WORLD,
        timeout=timedelta(seconds=120),
    )
    try:
        raster = _run(_build(rank, WORLD))
        np.save(os.path.join(out_dir, f"raster_{rank}.npy"), raster)
    finally:
        dist.destroy_process_group()


def test_partition_ranges_cover_disjointly():
    for n, ws in ((64, 2), (65, 2), (10, 3), (7, 7)):
        ranges = partition_ranges(n, ws)
        assert ranges[0][0] == 0 and ranges[-1][1] == n
        assert all(a[1] == b[0] for a, b in zip(ranges, ranges[1:]))


def test_two_rank_raster_equals_single_rank(tmp_path):
    # world_size=1 must run with NO torch.distributed init at all.
    assert not dist.is_initialized()
    single = _run(_build(0, 1))
    assert single.sum() > 0  # a silent network would pass equality vacuously

    try:
        mp.spawn(_worker, args=(str(tmp_path),), nprocs=WORLD, join=True)
    except Exception as exc:
        msg = str(exc).lower()
        if any(hint in msg for hint in _SANDBOX_HINTS):
            pytest.skip(f"distributed spawn unavailable in this sandbox: {exc}")
        raise

    stitched = np.zeros_like(single)
    for rank, (lo, hi) in enumerate(partition_ranges(N, WORLD)):
        stitched[:, lo:hi] = np.load(tmp_path / f"raster_{rank}.npy")
    assert np.array_equal(single, stitched)
