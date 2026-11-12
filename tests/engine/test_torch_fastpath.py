"""Fast path (engine.torch_core.fastpath) vs step(): bit-identical rasters.

Unlike the numpy-vs-torch equivalence tests, which must engineer margins
around fp64-vs-fp32 drift, fast-vs-slow here is the SAME engine on the same
device executing the same fp32 ops in the same order (in-place adds round
exactly like their out-of-place forms), so equality must hold bit-exact for
ARBITRARY weights and drives -- lognormal weights and gaussian drives, no
dyadic lattice needed. Also: capacity overflow raises, malformed drives
raise (the fast path never silently copies across device/dtype/shape), and
the bench --fast CLI emits the same JSON fields plus fast:true.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from brainsim.bench.__main__ import build_torch_network, run_bench
from brainsim.engine.lif import LIFParams
from brainsim.engine.torch_core.network import TorchNetwork
from brainsim.engine.torch_core.propagate import TorchCSRProjection
from brainsim.engine.torch_core.shard import ShardedNetwork

REPO_ROOT = Path(__file__).resolve().parents[2]
N, K, STEPS = 120, 10, 220  # raster-equality gate runs 200+ steps


def _drives(seed: int, n: int = N, steps: int = STEPS) -> np.ndarray:
    return np.random.default_rng(seed).normal(2.0, 1.0, size=(steps, n)).astype(np.float32)


def _build_sharded_world1() -> ShardedNetwork:
    net = ShardedNetwork(N, LIFParams(), device="cpu")
    rng = np.random.default_rng(11)
    pre = rng.integers(0, N, size=N * K)
    post = np.repeat(np.arange(N), K)
    weight = rng.lognormal(-1.0, 0.5, size=pre.size).astype(np.float32)
    sign = np.where(rng.random(pre.size) < 0.8, 1, -1).astype(np.int8)
    net.add_projection("rec", pre, post, weight, sign, delay_steps=1)
    net.add_projection("slow", pre[::-1].copy(), post, weight, sign, delay_steps=3)
    return net


def test_network_run_matches_step_loop_bitexact():
    ds = _drives(seed=7)
    net_slow = build_torch_network(N, K, seed=3, device="cpu")
    raster_slow = np.asarray(
        [net_slow.step(external_I={"pop": d})["pop"].copy() for d in ds]
    )
    assert raster_slow.sum() > 0  # a silent net would pass equality vacuously

    net_fast = build_torch_network(N, K, seed=3, device="cpu")
    counts, rasters = net_fast.run(
        STEPS, drive=lambda i: torch.from_numpy(ds[i]), record="raster"
    )
    assert np.array_equal(raster_slow, rasters["pop"])
    assert np.array_equal(counts, rasters["pop"].sum(axis=1))
    assert net_fast.t == STEPS
    # run() must NOT pay the per-step host copy into spike_buffer.
    assert len(net_fast.spike_buffer) == 0


def test_network_run_multi_population_dict_drive():
    def build() -> TorchNetwork:
        rng = np.random.default_rng(5)
        net = TorchNetwork(seed=0, device="cpu")
        net.add_population("A", 48, LIFParams())
        net.add_population("B", 32, LIFParams())
        pre = rng.integers(0, 48, size=48 * 6)
        post_a = rng.integers(0, 48, size=pre.size)
        post_b = rng.integers(0, 32, size=pre.size)
        w = rng.lognormal(-1.0, 0.5, size=pre.size).astype(np.float32)
        s = np.where(rng.random(pre.size) < 0.8, 1, -1).astype(np.int8)
        net.add_projection("aa", "A", "A", TorchCSRProjection.from_table(
            pre, post_a, w, s, n_pre=48, n_post=48, delay_steps=1, device="cpu"))
        # Strong feedforward weights so B actually relays (all-excitatory,
        # x8): equality would be vacuous over a silent population.
        net.add_projection("ab", "A", "B", TorchCSRProjection.from_table(
            pre, post_b, w * 8.0, np.ones_like(s), n_pre=48, n_post=32,
            delay_steps=2, device="cpu"))
        return net

    ds = _drives(seed=9, n=48)
    slow = build()
    raster_slow = {"A": [], "B": []}
    for d in ds:
        out = slow.step(external_I={"A": d})
        raster_slow["A"].append(out["A"].copy())
        raster_slow["B"].append(out["B"].copy())

    fast = build()
    counts, rasters = fast.run(
        STEPS, drive=lambda i: {"A": torch.from_numpy(ds[i])}, record="raster"
    )
    assert np.asarray(raster_slow["B"]).sum() > 0  # the relay actually fires
    for name in ("A", "B"):
        assert np.array_equal(np.asarray(raster_slow[name]), rasters[name])
    assert np.array_equal(counts, rasters["A"].sum(axis=1) + rasters["B"].sum(axis=1))


def test_sharded_world1_run_matches_step_loop_bitexact():
    ds = _drives(seed=13)
    net_slow = _build_sharded_world1()
    raster_slow = np.asarray([net_slow.step(external_I=d) for d in ds])
    assert raster_slow.sum() > 0

    net_fast = _build_sharded_world1()
    counts, raster_fast = net_fast.run(
        STEPS, drive=lambda i: torch.from_numpy(ds[i]), record="raster"
    )
    assert np.array_equal(raster_slow, raster_fast)
    assert np.array_equal(counts, raster_fast.sum(axis=1))


def test_sharded_run_constant_drive_and_no_record():
    const = np.full(N, 2.5, dtype=np.float32)
    net_slow = _build_sharded_world1()
    slow_counts = np.asarray([int(net_slow.step(external_I=const).sum()) for _ in range(80)])
    assert slow_counts.sum() > 0

    net_fast = _build_sharded_world1()
    counts, raster = net_fast.run(80, drive=torch.from_numpy(const))
    assert raster is None
    assert np.array_equal(counts, slow_counts)


def test_fast_exchange_cap_overflow_raises():
    # cap=1 but the strong constant drive fires every neuron: must RAISE,
    # never truncate. world_size=1 enforces the same capacity contract.
    net = ShardedNetwork(32, LIFParams(), device="cpu", fast_exchange_cap=1)
    with pytest.raises(RuntimeError, match="fast_exchange_cap overflow"):
        net.run(3, drive=torch.full((32,), 200.0))


def test_fast_exchange_cap_validation():
    with pytest.raises(ValueError, match="fast_exchange_cap"):
        ShardedNetwork(32, LIFParams(), device="cpu", fast_exchange_cap=0)
    # Default cap is ceil(n_global/world_size): uniform across ranks and
    # >= n_local, so it can never overflow.
    assert ShardedNetwork(33, LIFParams(), device="cpu").fast_exchange_cap == 33


def test_run_rejects_malformed_drives():
    net = _build_sharded_world1()
    with pytest.raises(ValueError, match="shape"):
        net.run(2, drive=torch.zeros(N + 1))
    with pytest.raises(ValueError, match="dtype"):
        net.run(2, drive=torch.zeros(N, dtype=torch.float64))
    with pytest.raises(TypeError, match="torch.Tensor"):
        net.run(2, drive=np.zeros(N, dtype=np.float32))
    with pytest.raises(ValueError, match="record"):
        net.run(2, record="counts")

    tnet = TorchNetwork(seed=0, device="cpu")
    tnet.add_population("A", 8, LIFParams())
    tnet.add_population("B", 8, LIFParams())
    with pytest.raises(ValueError, match="exactly one population"):
        tnet.run(2, drive=torch.zeros(8))
    with pytest.raises(ValueError, match="unknown population"):
        tnet.run(2, drive={"C": torch.zeros(8)})


def test_run_bench_fast_report():
    fast = run_bench(n=100, k=10, steps=20, seed=0, backend="torch",
                     device="cpu", fast=True)
    slow = run_bench(n=100, k=10, steps=20, seed=0, backend="torch",
                     device="cpu")
    assert fast["fast"] is True
    assert fast["steps_per_sec"] > 0
    # JSON contract: identical fields plus "fast".
    assert set(fast) == set(slow) | {"fast"}
    with pytest.raises(ValueError, match="backend"):
        run_bench(n=50, k=5, steps=5, seed=0, backend="numpy", fast=True)


def test_bench_fast_smoke_cli(tmp_path):
    out = tmp_path / "fast.json"
    result = subprocess.run(
        [sys.executable, "-m", "brainsim.bench", "--smoke", "--backend", "torch",
         "--device", "cpu", "--fast", "--json", str(out)],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "steps/sec" in result.stdout
    report = json.loads(out.read_text())
    assert report["fast"] is True

    # --fast is torch-only: the numpy backend must be refused at the parser.
    bad = subprocess.run(
        [sys.executable, "-m", "brainsim.bench", "--smoke", "--fast"],
        cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=120,
    )
    assert bad.returncode != 0
    assert "--fast" in bad.stderr
