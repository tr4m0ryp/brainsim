"""F24 byte-budget accounting: the static fp16+int32 layout must sit at
6 B/synapse and every realistic layout under the 16 B/synapse requirement;
the forecast (bench --forecast, torch-free) must agree with bytes measured
off live projection tensors.
"""
import numpy as np
import pytest

from brainsim.engine.torch_core import memory as tc_memory


def test_static_fp16_layout_within_f24_budget():
    assert tc_memory.synapse_bytes(fp16=True) == 6
    assert tc_memory.synapse_bytes(fp16=False) == 8
    # Full plastic add-ons (uint8 cascade + fp16 trace) stay inside 16 B too.
    assert tc_memory.synapse_bytes(fp16=True, cascade=True, trace=True) == 9
    assert tc_memory.synapse_bytes(fp16=True, cascade=True, trace=True) <= 16


def test_forecast_composite_scale_fits_f24_envelope():
    # F24 anchor: 5e10 synapses at <=16 B/syn = 800 GB inside 1,128 GB (8x H200).
    n_syn, n_neurons = 50_000_000_000, 5_000_000
    for ranks in range(1, 9):
        f = tc_memory.forecast_rank_bytes(n_syn, n_neurons, ranks, fp16=True)
        assert f["bytes_per_synapse"] <= 16.0, (ranks, f)
    per_rank = tc_memory.forecast_rank_bytes(n_syn, n_neurons, 8, fp16=True)
    assert per_rank["total"] * 8 <= 1128e9
    assert per_rank["total"] <= 141e9  # one H200


def test_forecast_table_renders_all_ranks():
    table = tc_memory.forecast_table(1_000_000, 10_000)
    assert "GB/rank(fp16)" in table
    assert len(table.splitlines()) == 2 + 8  # header pair + ranks 1..8


def test_measured_projection_bytes_match_accounting():
    torch = pytest.importorskip("torch")
    from brainsim.engine.torch_core.propagate import TorchCSRProjection

    n, k, delay = 40, 10, 2
    rng = np.random.default_rng(0)
    pre = np.repeat(np.arange(n), k)
    post = np.concatenate([rng.choice(n, size=k, replace=False) for _ in range(n)])
    weight = rng.lognormal(-1.0, 0.5, size=pre.size)
    sign = np.where(rng.random(pre.size) < 0.8, 1, -1).astype(np.int8)
    proj = TorchCSRProjection.from_table(
        pre, post, weight, sign, n_pre=n, n_post=n, delay_steps=delay, fp16=True
    )

    measured = tc_memory.measured_projection_bytes(proj)
    predicted = tc_memory.projection_bytes(
        n_pre=n, n_post=n, n_syn=n * k, fp16=True, delay_steps=delay
    )
    assert measured == predicted
    assert measured["synapse_arrays"] / proj.n_synapses == 6.0
    assert measured["total"] / proj.n_synapses <= 16.0
