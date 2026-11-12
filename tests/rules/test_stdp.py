import numpy as np

from brainsim.rules.stdp import STDPParams, STDPState, step


def _run_pair(pre_spike_step: int, post_spike_step: int, n_steps: int, params: STDPParams) -> float:
    state = STDPState.zeros(n_pre=1, n_post=1, n_syn=1, w_init=0.5)
    pre_idx = np.array([0])
    post_idx = np.array([0])
    w0 = state.W.copy()
    for t in range(n_steps):
        pre_spikes = np.array([t == pre_spike_step])
        post_spikes = np.array([t == post_spike_step])
        step(state, params, pre_spikes, post_spikes, pre_idx, post_idx)
    return float(state.W[0] - w0[0])


def test_pre_before_post_potentiates():
    params = STDPParams(dt=1.0)
    dw = _run_pair(pre_spike_step=5, post_spike_step=10, n_steps=20, params=params)
    assert dw > 0


def test_post_before_pre_depresses():
    params = STDPParams(dt=1.0)
    dw = _run_pair(pre_spike_step=10, post_spike_step=5, n_steps=20, params=params)
    assert dw < 0


def test_magnitude_decays_with_delta_t():
    params = STDPParams(dt=1.0)
    dw_close = _run_pair(pre_spike_step=9, post_spike_step=10, n_steps=20, params=params)
    dw_far = _run_pair(pre_spike_step=1, post_spike_step=10, n_steps=20, params=params)
    assert dw_close > dw_far > 0

    dw_close_dep = _run_pair(pre_spike_step=10, post_spike_step=9, n_steps=20, params=params)
    dw_far_dep = _run_pair(pre_spike_step=10, post_spike_step=1, n_steps=20, params=params)
    assert dw_close_dep < dw_far_dep < 0


def test_weights_clip_to_bounds():
    params = STDPParams(dt=1.0, a_plus=10.0, w_max=1.0, w_min=0.0)
    state = STDPState.zeros(n_pre=1, n_post=1, n_syn=1, w_init=0.9)
    pre_idx = np.array([0])
    post_idx = np.array([0])
    for t in range(5):
        pre_spikes = np.array([t == 0])
        post_spikes = np.array([t == 1])
        step(state, params, pre_spikes, post_spikes, pre_idx, post_idx)
    assert state.W[0] <= params.w_max
