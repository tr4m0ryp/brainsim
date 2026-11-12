import numpy as np

from brainsim.rules.allocation import AllocationParams, AllocationState, allocate, decay


def test_recently_allocated_neurons_win_next_allocation():
    params = AllocationParams(tau=100.0, jump=10.0, noise_std=0.1, k=10)
    n = 200
    rng = np.random.default_rng(0)
    state = AllocationState.zeros(n)

    first = set(allocate(state, params, rng).tolist())
    decay(state, params, dt_seconds=1.0)  # short elapsed time relative to tau
    second = set(allocate(state, params, rng).tolist())

    overlap = len(first & second)
    assert overlap == params.k  # jump (10) dwarfs noise_std (0.1): same neurons win again


def test_far_apart_events_dont_overlap_much():
    params = AllocationParams(tau=100.0, jump=10.0, noise_std=0.1, k=10)
    n = 200
    rng = np.random.default_rng(1)
    state = AllocationState.zeros(n)

    first = set(allocate(state, params, rng).tolist())
    decay(state, params, dt_seconds=10.0 * params.tau)  # far beyond tau: boost decays away
    second = set(allocate(state, params, rng).tolist())

    overlap = len(first & second)
    assert overlap <= 2  # noise-dominated regime, chance overlap only


def test_decay_reduces_excitability_monotonically():
    params = AllocationParams(tau=50.0)
    state = AllocationState.zeros(5)
    state.excitability[:] = 1.0
    prev = state.excitability.copy()
    for _ in range(5):
        decay(state, params, dt_seconds=10.0)
        assert np.all(state.excitability < prev)
        prev = state.excitability.copy()
