import numpy as np

from brainsim.rules.replay import ReplayParams, ReplayScheduler


def test_higher_salience_replayed_proportionally_more():
    rng = np.random.default_rng(0)
    params = ReplayParams(budget=20000, lr_multiplier=0.05)
    scheduler = ReplayScheduler(params, rng)
    saliences = {0: 1.0, 1: 2.0, 2: 4.0, 3: 8.0}
    for item_id, s in saliences.items():
        scheduler.register(item_id, s)

    events = scheduler.run_offline_phase()
    counts = {item_id: 0 for item_id in saliences}
    for e in events:
        counts[e.item_id] += 1
        assert e.lr_multiplier == params.lr_multiplier

    total = sum(saliences.values())
    for item_id, s in saliences.items():
        expected_frac = s / total
        observed_frac = counts[item_id] / params.budget
        assert abs(observed_frac - expected_frac) < 0.02

    # strictly monotonic with salience given the large budget
    ordered = sorted(saliences, key=lambda i: saliences[i])
    assert counts[ordered[0]] < counts[ordered[1]] < counts[ordered[2]] < counts[ordered[3]]


def test_empty_scheduler_returns_no_events():
    rng = np.random.default_rng(0)
    scheduler = ReplayScheduler(ReplayParams(budget=5), rng)
    assert scheduler.run_offline_phase() == []


def test_unregister_removes_item():
    rng = np.random.default_rng(0)
    scheduler = ReplayScheduler(ReplayParams(budget=100), rng)
    scheduler.register(0, 1.0)
    scheduler.register(1, 1.0)
    scheduler.unregister(0)
    events = scheduler.run_offline_phase()
    assert all(e.item_id == 1 for e in events)
