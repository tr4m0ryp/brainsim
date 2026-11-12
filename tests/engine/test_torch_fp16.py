"""fp16 projection weights (F24): fp16 is storage-only, accumulation stays
fp32, so the dynamics must stay finite and land close to the fp32 run --
weight quantization error is ~5e-4 relative, far under the 5% spike-count
band asserted here on the bench network.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from brainsim.bench.__main__ import build_torch_network

N, K, STEPS = 1000, 20, 100


def _total_spikes(net, drives) -> int:
    return sum(int(net.step(external_I={"pop": d})["pop"].sum()) for d in drives)


def test_fp16_weights_finite_and_within_5pct_of_fp32():
    drive_rng = np.random.default_rng(1)
    drives = [drive_rng.normal(2.0, 1.0, size=N) for _ in range(STEPS)]

    net32 = build_torch_network(N, K, seed=0, device="cpu", fp16=False)
    net16 = build_torch_network(N, K, seed=0, device="cpu", fp16=True)
    assert net16.projections["rec"].csr.weights.dtype == torch.float16
    assert net32.projections["rec"].csr.weights.dtype == torch.float32

    total32 = _total_spikes(net32, drives)
    total16 = _total_spikes(net16, drives)

    state = net16.populations["pop"].state
    assert torch.isfinite(state.v).all()
    assert torch.isfinite(state.I).all()

    assert total32 > 0
    assert abs(total16 - total32) / total32 <= 0.05, (total16, total32)
