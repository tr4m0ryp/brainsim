import numpy as np

from brainsim.engine.propagate import CSRProjection


def test_known_tiny_graph_currents_and_signs():
    # pre 0 -> post 0 (w=2, +), pre 0 -> post 1 (w=1, -), pre 1 -> post 1 (w=3, +)
    pre_id = np.array([0, 0, 1])
    post_id = np.array([0, 1, 1])
    weight = np.array([2.0, 1.0, 3.0])
    sign = np.array([1, -1, 1])
    csr = CSRProjection.from_table(pre_id, post_id, weight, sign, n_pre=2, n_post=2, delay_steps=1)

    csr.deliver(np.array([True, False]))
    current = csr.pop_current()
    assert np.allclose(current, [2.0, -1.0])

    csr.deliver(np.array([False, True]))
    current = csr.pop_current()
    assert np.allclose(current, [0.0, 3.0])


def test_no_spikes_delivers_nothing():
    pre_id = np.array([0, 1])
    post_id = np.array([0, 0])
    weight = np.array([1.0, 1.0])
    sign = np.array([1, 1])
    csr = CSRProjection.from_table(pre_id, post_id, weight, sign, n_pre=2, n_post=1, delay_steps=1)
    csr.deliver(np.array([False, False]))
    assert np.allclose(csr.pop_current(), [0.0])


def test_delay_steps_ring_buffer():
    pre_id = np.array([0])
    post_id = np.array([0])
    weight = np.array([5.0])
    sign = np.array([1])
    csr = CSRProjection.from_table(pre_id, post_id, weight, sign, n_pre=1, n_post=1, delay_steps=3)

    csr.deliver(np.array([True]))
    # not due yet at pop 0, 1
    assert np.allclose(csr.pop_current(), [0.0])
    assert np.allclose(csr.pop_current(), [0.0])
    # due on the 3rd pop
    assert np.allclose(csr.pop_current(), [5.0])
    # and gone afterward
    assert np.allclose(csr.pop_current(), [0.0])


def test_multiple_spiking_presynaptic_neurons_accumulate():
    pre_id = np.array([0, 1, 2])
    post_id = np.array([0, 0, 0])
    weight = np.array([1.0, 2.0, 4.0])
    sign = np.array([1, 1, -1])
    csr = CSRProjection.from_table(pre_id, post_id, weight, sign, n_pre=3, n_post=1, delay_steps=1)
    csr.deliver(np.array([True, True, True]))
    assert np.allclose(csr.pop_current(), [1.0 + 2.0 - 4.0])
