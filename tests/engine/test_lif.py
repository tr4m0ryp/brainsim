import numpy as np

from brainsim.engine.lif import LIFParams, LIFState, step


def test_subthreshold_no_spike():
    params = LIFParams(dt=1.0)
    state = LIFState.zeros(3, params)
    state.I[:] = 0.0
    spikes = step(state, params)
    assert not spikes.any()
    assert np.allclose(state.v, params.v_rest)


def test_threshold_and_reset():
    params = LIFParams(dt=1.0, R=1.0, tau_mem=10.0)
    state = LIFState.zeros(1, params)
    state.I[:] = 1000.0  # strong drive, must cross threshold quickly
    spiked = False
    for _ in range(20):
        spikes = step(state, params)
        if spikes[0]:
            spiked = True
            break
    assert spiked
    assert state.v[0] == params.v_reset


def test_refractory_blocks_integration():
    params = LIFParams(dt=1.0, refractory=2.2, tau_mem=1.0)
    state = LIFState.zeros(1, params)
    state.I[:] = 1000.0  # strong drive forces an immediate spike
    spikes = step(state, params)
    assert spikes[0]
    assert state.refractory_counter[0] == params.refractory_steps

    v_during_refractory = []
    for _ in range(params.refractory_steps):
        state.I[:] = 1000.0  # even under strong drive, refractory neuron must not integrate
        step(state, params)
        v_during_refractory.append(state.v[0])
    assert all(v == params.v_reset for v in v_during_refractory)


def test_deterministic_seeded_run():
    def run(seed: int) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        params = LIFParams(dt=1.0)
        state = LIFState.zeros(20, params)
        spike_trace, v_trace = [], []
        for _ in range(50):
            state.I = state.I + rng.normal(5.0, 2.0, size=20)
            spikes = step(state, params)
            spike_trace.append(spikes.copy())
            v_trace.append(state.v.copy())
        return np.array(spike_trace), np.array(v_trace)

    spikes_a, v_a = run(seed=42)
    spikes_b, v_b = run(seed=42)
    spikes_c, v_c = run(seed=7)
    assert np.array_equal(spikes_a, spikes_b)
    assert np.array_equal(v_a, v_b)
    assert not np.allclose(v_a, v_c)


def test_synaptic_current_decays():
    params = LIFParams(dt=1.0, tau_syn=5.0)
    state = LIFState.zeros(1, params)
    state.I[:] = 10.0
    step(state, params)
    assert state.I[0] < 10.0
    assert state.I[0] == 10.0 * np.exp(-params.dt / params.tau_syn)
