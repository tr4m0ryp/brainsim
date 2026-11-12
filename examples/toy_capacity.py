#!/usr/bin/env python3
"""Toy end-to-end run on a SYNTHETIC mushroom-body-shaped graph (CPU, < 1 min).

No connectome data is needed and no number printed here is a result about any
animal: the graph is random, with fake neuron ids.

  python examples/toy_capacity.py

Part 1, spiking engine: the PN -> KC -> MBON projections of the graph run as
leaky integrate-and-fire populations on the numpy engine under Bernoulli PN
drive, and the dopamine-gated mushroom-body rule (rules/mb_rule) depresses
the KC -> MBON weights of the MBONs that receive a dopamine signal.

Part 2, capacity protocol: the same capacity measurement the fly arm ran on
FlyWire v783 (batteries/h1), at toy size: stream random odor-like patterns,
recall from half-deleted cues, report C = max P with mean overlap >= 0.9 on
the graph and on its degree-preserving shuffle, and their ratio.
"""
from __future__ import annotations

import time

import numpy as np

from brainsim.batteries.h1 import controls
from brainsim.batteries.h1.circuit import MBCircuit
from brainsim.batteries.h1.protocol import ItemBank, ProtocolCfg, rng_for, run_mb_capacity
from brainsim.batteries.h1.mb_extract import build_graph
from brainsim.engine.lif import LIFParams
from brainsim.engine.network import Network
from brainsim.engine.propagate import CSRProjection
from brainsim.rules import mb_rule

W_UNIT = 1.5          # mV of synaptic current per anatomical synapse
DRIVE_P = 0.2         # PN spike probability per 1 ms step while driven
STEPS = 300


def make_toy_graph(seed: int = 0, n_pn: int = 150, n_kc: int = 1500,
                   n_mbon: int = 32, kc_per_mbon: float = 0.4):
    """Random MB-shaped graph with fake ids: 6 PN claws per KC, each MBON
    reading a random 40% of KCs, one APL feedback loop over all KCs."""
    rng = np.random.default_rng(seed)
    roots = {"PN": list(range(10_000, 10_000 + n_pn)),
             "KC": list(range(20_000, 20_000 + n_kc)),
             "MBON": list(range(40_000, 40_000 + n_mbon)),
             "DAN": [50_000], "APL": [60_000, 60_001]}
    pre, post, cnt = [], [], []
    for kc in roots["KC"]:
        for pn in rng.choice(roots["PN"], size=6, replace=False):
            pre.append(pn); post.append(kc); cnt.append(int(rng.integers(1, 12)))
        apl = roots["APL"][kc % 2]
        pre += [kc, apl]; post += [apl, kc]; cnt += [int(rng.integers(1, 6))] * 2
    for mbon in roots["MBON"]:
        for kc in rng.choice(roots["KC"], size=int(kc_per_mbon * n_kc), replace=False):
            pre.append(kc); post.append(mbon); cnt.append(int(rng.integers(1, 8)))
    return build_graph(np.asarray(pre, dtype=np.int64), np.asarray(post, dtype=np.int64),
                       np.asarray(cnt, dtype=np.int64), roots)


def spiking_pass(g) -> None:
    net = Network(seed=0)
    params = LIFParams()
    for name, n in (("PN", g.n_pn), ("KC", g.n_kc), ("MBON", g.n_mbon)):
        net.add_population(name, n, params)
    for name, proj, pre, post in (("pn_kc", g.pn_kc, "PN", "KC"),
                                  ("kc_mbon", g.kc_mbon, "KC", "MBON")):
        csr = CSRProjection.from_table(proj.pre, proj.post, proj.w * W_UNIT,
                                       np.ones(proj.n_pairs), proj.n_pre,
                                       proj.n_post, delay_steps=2)
        net.add_projection(name, pre, post, csr)

    rule_p = mb_rule.MBRuleParams()
    kc_mbon = net.projections["kc_mbon"].csr
    pre_idx = np.repeat(np.arange(kc_mbon.n_pre), np.diff(kc_mbon.indptr))
    rule = mb_rule.MBRuleState.zeros(kc_mbon.n_pre, kc_mbon.n_synapses)
    dan_on = np.isin(kc_mbon.indices, np.arange(g.n_mbon // 4))  # gated MBONs
    rng = np.random.default_rng(1)
    odor = rng.choice(g.n_pn, size=max(1, g.n_pn // 4), replace=False)
    counts = {"PN": 0, "KC": 0, "MBON": 0}
    for _ in range(STEPS):
        drive = np.zeros(g.n_pn)
        drive[odor[rng.random(odor.size) < DRIVE_P]] = 40.0
        spikes = net.step({"PN": drive})
        for k in counts:
            counts[k] += int(spikes[k].sum())
        mb_rule.step(rule, rule_p, spikes["KC"], dan_on.astype(float), pre_idx)
    print(f"spiking pass: {net.neuron_count()} LIF neurons, "
          f"{net.synapse_count()} connections, {STEPS} steps of 1 ms")
    print("  spikes per population:", counts)
    gated = rule.W[dan_on]
    print(f"  mb_rule: gated KC->MBON weight mean {gated.mean():.4f} "
          f"(w_max {rule_p.w_max}), ungated unchanged at {rule.W[~dan_on].mean():.4f}")


def capacity_pass(g, seeds=(0, 1, 2), n_draws=3) -> None:
    cfg = ProtocolCfg(p_cap=512, eval_sample=64, write_chunk=128)
    print(f"capacity pass: f_pn {cfg.f_pn}, f_kc {cfg.f_kc}, f_z {cfg.f_z}, "
          f"overlap bar 0.9, cues 50% deleted, seeds {list(seeds)}")
    ratios = []
    for seed in seeds:
        bank = ItemBank.generate(seed, cfg, g.n_pn, g.n_mbon)
        c_graph = run_mb_capacity(MBCircuit.from_graph(g, cfg.f_kc, 1.0),
                                  bank, seed, cfg)["C"]
        c_shuf = np.mean([run_mb_capacity(
            MBCircuit.from_graph(controls.config_shuffle(
                g, rng_for(seed, 10, d))[0], cfg.f_kc, 1.0), bank, seed, cfg)["C"]
            for d in range(n_draws)])
        ratio = c_graph / c_shuf if c_shuf > 0 else float("nan")
        ratios.append(ratio)
        print(f"  seed {seed}: C(graph) {c_graph:.2f}, "
              f"C(degree-preserving shuffle, mean of {n_draws}) {c_shuf:.2f}, "
              f"ratio {ratio:.3f}")
    print(f"  mean ratio over seeds {np.nanmean(ratios):.3f} (synthetic graph, "
          "not a measurement of any animal)")


def main() -> None:
    t0 = time.time()
    g = make_toy_graph()
    print(f"synthetic graph: {g.n_pn} PN, {g.n_kc} KC, {g.n_mbon} MBON, 2 APL")
    spiking_pass(g)
    capacity_pass(g)
    print(f"done in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
