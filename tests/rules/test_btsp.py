import numpy as np

from brainsim.rules.btsp import BTSPParams, BTSPState, step


def _run(n_steps: int, pre_spike_steps: dict, plateau_step: int | None, params: BTSPParams):
    state = BTSPState.zeros(n_pre=2, n_post=1, n_syn=2, w_init=0.5)
    pre_idx = np.array([0, 1])
    post_idx = np.array([0, 0])
    w0 = state.W.copy()
    for t in range(n_steps):
        pre_spikes = np.array([t in pre_spike_steps.get(0, []), t in pre_spike_steps.get(1, [])])
        plateau = np.array([t == plateau_step])
        step(state, params, pre_spikes, plateau, pre_idx, post_idx)
    return state.W - w0


def test_no_plateau_no_change():
    params = BTSPParams(dt=1.0)
    dw = _run(n_steps=50, pre_spike_steps={0: [10], 1: [20]}, plateau_step=None, params=params)
    assert np.allclose(dw, 0.0, atol=1e-9)


def test_plateau_potentiates_recently_active_input_more():
    params = BTSPParams(dt=1.0, k_plus=5.0, k_minus=0.1)
    # synapse 0's presynaptic neuron fires recently before the plateau (t=990),
    # synapse 1's fires much earlier (t=0). Plateau at t=1000.
    dw = _run(n_steps=1001, pre_spike_steps={0: [990], 1: [0]}, plateau_step=1000, params=params)
    assert dw[0] > 0
    assert dw[0] > dw[1] >= 0


def test_weights_stay_bounded():
    params = BTSPParams(dt=1.0, k_plus=50.0, k_minus=50.0, w_max=1.0)
    state = BTSPState.zeros(n_pre=1, n_post=1, n_syn=1, w_init=0.5)
    pre_idx = np.array([0])
    post_idx = np.array([0])
    for t in range(200):
        pre_spikes = np.array([t % 3 == 0])
        plateau = np.array([t % 5 == 0])
        step(state, params, pre_spikes, plateau, pre_idx, post_idx)
        assert 0.0 <= state.W[0] <= params.w_max
