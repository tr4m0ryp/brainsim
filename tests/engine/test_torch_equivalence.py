"""numpy engine vs torch-cpu engine: exact agreement, by construction.

Bit-exact membrane equality is impossible: the numpy engine integrates in
float64, the torch engine in float32 (fp32 state is a design decision, F24),
and the exp() decay factors are not exactly representable, so v trajectories
drift apart by ~1e-4 mV. Raster equality therefore has to be ENGINEERED, in
two layers:

1. Current level: dyadic weights (multiples of 1/8, exact in fp16/fp32/fp64)
   make every delivered synaptic sum exact in all precisions, so projection
   outputs are compared with np.array_equal, zero tolerance. This is where
   sign folding, CSR construction, and the delay ring are pinned.
2. Raster level: a pulse-relay network whose every threshold decision is a
   coarse jump (>= ~2.5 mV from v_th, verified by an in-test margin check
   at >= 1.0 mV), never a hover. A drive that lets neurons drift near v_th
   puts decisions within fp32-vs-fp64 drift of the threshold and raster
   equality genuinely fails; structural margins are the fix, not tolerances.
   Spike rasters then match exactly over 120 steps, and per-step integer
   spike counts with them.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from brainsim.engine.lif import LIFParams
from brainsim.engine.network import Network
from brainsim.engine.propagate import CSRProjection
from brainsim.engine.torch_core.network import TorchNetwork
from brainsim.engine.torch_core.propagate import TorchCSRProjection

# Pulse-relay layout: A [0,24) gets an external pulse every PERIOD steps and
# spikes on the pulse step; B1 [24,40) relays it one step later over a strong
# delay-1 projection; B2 [40,52) three steps later over a strong delay-3
# projection; C [52,64) receives weak mixed-sign input from both projections
# and must stay silent. tau_syn=1.0 kills residual currents between pulses,
# so no neuron ever drifts near threshold.
N, STEPS, PERIOD = 64, 120, 8
A, B1, B2, C = range(0, 24), range(24, 40), range(40, 52), range(52, 64)
STRONG_W, FAN_IN = 24.0, 8
PARAMS = LIFParams(tau_syn=1.0)
MARGIN_FLOOR = 1.0  # mV; structural margins are ~2.5 mV, drift is ~1e-4 mV


def _relay_table(rng, strong_post, weak_post):
    """A -> strong_post at weight 24 (spike-deciding), A -> weak_post at
    dyadic mixed-sign weights (|sum| <= 31, never spike-deciding)."""
    pre, post, weight, sign = [], [], [], []
    for p in strong_post:
        pre.append(rng.choice(np.asarray(A), size=FAN_IN, replace=False))
        post.append(np.full(FAN_IN, p))
        weight.append(np.full(FAN_IN, STRONG_W))
        sign.append(np.ones(FAN_IN, dtype=np.int8))
    for p in weak_post:
        pre.append(rng.choice(np.asarray(A), size=FAN_IN, replace=False))
        post.append(np.full(FAN_IN, p))
        weight.append(rng.integers(1, 32, size=FAN_IN) * 0.125)  # 0.125 .. 3.875
        sign.append(np.where(rng.random(FAN_IN) < 0.5, 1, -1).astype(np.int8))
    return (np.concatenate(pre), np.concatenate(post),
            np.concatenate(weight), np.concatenate(sign))


def _tables(seed: int):
    rng = np.random.default_rng(seed)
    fast = _relay_table(rng, B1, rng.permutation(np.asarray(C))[:6])
    slow = _relay_table(rng, B2, rng.permutation(np.asarray(C))[6:])
    return fast, slow


def _drives(seed: int) -> list[np.ndarray]:
    """Pulse amplitude 192..200 on a 0.25 lattice for A, zero elsewhere."""
    rng = np.random.default_rng(seed)
    drives = []
    for t in range(STEPS):
        d = np.zeros(N)
        if t % PERIOD == 0:
            d[list(A)] = 192.0 + rng.integers(0, 33, size=len(A)) * 0.25
        drives.append(d)
    return drives


def _build(engine: str, seed: int):
    fast, slow = _tables(seed)
    if engine == "numpy":
        net = Network(seed=0)
        projs = {1: CSRProjection.from_table(*fast, n_pre=N, n_post=N, delay_steps=1),
                 3: CSRProjection.from_table(*slow, n_pre=N, n_post=N, delay_steps=3)}
    else:
        net = TorchNetwork(seed=0, device="cpu")
        projs = {1: TorchCSRProjection.from_table(
                     *fast, n_pre=N, n_post=N, delay_steps=1, device="cpu"),
                 3: TorchCSRProjection.from_table(
                     *slow, n_pre=N, n_post=N, delay_steps=3, device="cpu")}
    net.add_population("pop", N, PARAMS)
    net.add_projection("fast", "pop", "pop", projs[1])
    net.add_projection("slow", "pop", "pop", projs[3])
    return net


def _run(net, drives) -> np.ndarray:
    return np.asarray([net.step(external_I={"pop": d})["pop"].copy() for d in drives])


def _run_numpy_with_margins(drives):
    """Raster plus per-step two-sided decision margins min |v_decision - v_th|
    over active neurons. Decision values are recovered from exact pre-step
    copies of (v, refractory) and post-step I (undone by one decay factor,
    ~1 ulp): the same fp64 expression engine.lif.step evaluates, covering
    spiking AND non-spiking neurons.
    """
    net = _build("numpy", seed=17)
    pop = net.populations["pop"]
    decay = np.exp(-PARAMS.dt / PARAMS.tau_syn)
    raster, margins = [], []
    for d in drives:
        v0 = pop.state.v.copy()
        active = pop.state.refractory_counter <= 0
        raster.append(net.step(external_I={"pop": d})["pop"].copy())
        I_tot = pop.state.I / decay
        decision = v0 + PARAMS.dt * (-(v0 - PARAMS.v_rest) + PARAMS.R * I_tot) / PARAMS.tau_mem
        if active.any():
            margins.append(np.abs(decision[active] - PARAMS.v_th).min())
    return np.asarray(raster), np.asarray(margins)


def test_projection_currents_exactly_equal_to_numpy():
    """Sign folding, stable CSR build, ring, and fp16/fp32 storage, pinned at
    the current level: dyadic weights make every sum exact, so numpy fp64 and
    torch outputs must agree to the bit."""
    rng = np.random.default_rng(3)
    n_pre, n_post = 40, 32
    pre = np.repeat(np.arange(n_pre), 5)
    post = rng.integers(0, n_post, size=pre.size)
    weight = rng.integers(1, 64, size=pre.size) * 0.125
    sign = np.where(rng.random(pre.size) < 0.5, 1, -1).astype(np.int8)
    for delay in (1, 3):
        for fp16 in (False, True):
            np_proj = CSRProjection.from_table(
                pre, post, weight, sign, n_pre=n_pre, n_post=n_post, delay_steps=delay)
            t_proj = TorchCSRProjection.from_table(
                pre, post, weight, sign, n_pre=n_pre, n_post=n_post,
                delay_steps=delay, fp16=fp16, device="cpu")
            for _ in range(12):
                fired = rng.random(n_pre) < 0.3
                np_proj.deliver(fired)
                t_proj.deliver(torch.as_tensor(fired))
                assert np.array_equal(
                    np_proj.pop_current(), t_proj.pop_current().numpy().astype(np.float64))


def test_numpy_and_torch_rasters_identical():
    drives = _drives(seed=13)
    raster_np, margins = _run_numpy_with_margins(drives)
    raster_t = _run(_build("torch", seed=17), drives)

    # Design self-check: if a change ever lets a decision value hover near
    # v_th, fail HERE with the reason, not with a raster mismatch downstream.
    assert margins.min() > MARGIN_FLOOR, (
        f"decision margin {margins.min():.2e} mV under {MARGIN_FLOOR} mV: network no "
        "longer avoids threshold-edge ambiguity, raster equality is not guaranteed")

    # The relay structure itself: A on pulse steps, B1 one step later (delay
    # 1), B2 three steps later (delay 3), C silent. Pins ring timing to the
    # step clock and makes the equality below non-vacuous.
    pulses = np.arange(0, STEPS, PERIOD)
    assert raster_np[:, list(A)].sum() == pulses.size * len(A)
    assert all(raster_np[t, list(A)].all() for t in pulses)
    assert raster_np[:, list(B1)].sum() == pulses.size * len(B1)
    assert all(raster_np[t + 1, list(B1)].all() for t in pulses)
    assert raster_np[:, list(B2)].sum() == pulses.size * len(B2)
    assert all(raster_np[t + 3, list(B2)].all() for t in pulses)
    assert raster_np[:, list(C)].sum() == 0

    assert np.array_equal(raster_np, raster_t)
    # Per-step integer spike counts, zero tolerance (implied by the above,
    # asserted explicitly as the T5 acceptance criterion).
    assert np.array_equal(raster_np.sum(axis=1), raster_t.sum(axis=1))


def test_torch_engine_deterministic_across_runs():
    drives = _drives(seed=13)
    assert np.array_equal(
        _run(_build("torch", seed=17), drives), _run(_build("torch", seed=17), drives))
