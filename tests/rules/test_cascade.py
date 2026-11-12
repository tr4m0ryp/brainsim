import numpy as np

from brainsim.rules import cascade


def test_state_dtype_stays_uint8():
    params = cascade.CascadeParams()
    rng = np.random.default_rng(0)
    state = np.full(100, params.n_levels - 1, dtype=np.uint8)
    for _ in range(20):
        state = cascade.push(state, params, rng)
        assert state.dtype == np.uint8
        state = cascade.pull(state, params, rng)
        assert state.dtype == np.uint8


def test_deep_states_flip_less_under_noise_than_shallow():
    params = cascade.CascadeParams(n_levels=8, q1=0.5, x=0.5)
    n = 2000
    rng = np.random.default_rng(1)

    shallow = np.full(n, params.n_levels - 1, dtype=np.uint8)  # depth=1, depressed branch
    deep = np.zeros(n, dtype=np.uint8)  # depth=n_levels, depressed branch (max protection)

    shallow_start = cascade.effective_weight(shallow, params).copy()
    deep_start = cascade.effective_weight(deep, params).copy()

    for _ in range(30):
        direction = rng.integers(0, 2, size=n)  # 0 -> pull, 1 -> push (symmetric noise)
        push_mask = direction == 1
        pull_mask = ~push_mask
        shallow = cascade.push(shallow, params, rng, mask=push_mask)
        shallow = cascade.pull(shallow, params, rng, mask=pull_mask)
        deep = cascade.push(deep, params, rng, mask=push_mask)
        deep = cascade.pull(deep, params, rng, mask=pull_mask)

    shallow_flips = np.mean(cascade.effective_weight(shallow, params) != shallow_start)
    deep_flips = np.mean(cascade.effective_weight(deep, params) != deep_start)
    assert deep_flips < shallow_flips


def test_deepen_is_deterministic_and_protective():
    params = cascade.CascadeParams()
    state = np.full(50, params.n_levels - 1, dtype=np.uint8)  # shallow depressed
    deepened = cascade.deepen(state, params, steps=params.n_levels)
    assert np.all(cascade.depth(deepened, params) == params.n_levels)
    assert np.all(cascade.effective_weight(deepened, params) == 0)


def test_distribution_converges():
    params = cascade.CascadeParams(n_levels=8, q1=0.7, x=0.6)
    n = 5000
    rng = np.random.default_rng(2)
    state = np.zeros(n, dtype=np.uint8)

    def run(steps: int) -> float:
        s = state.copy()
        r = np.random.default_rng(2)
        for _ in range(steps):
            direction = r.integers(0, 2, size=n)
            s = cascade.push(s, params, r, mask=direction == 1)
            s = cascade.pull(s, params, r, mask=direction == 0)
        return float(cascade.effective_weight(s, params).mean())

    mean_100 = run(100)
    mean_200 = run(200)
    assert abs(mean_200 - mean_100) < 0.05
