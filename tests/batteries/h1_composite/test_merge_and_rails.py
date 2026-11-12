"""H1 composite-arm tests, part 2: shard/merge coverage, verdict wiring
including the anti-premium case, the TF control, smoke rails."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from brainsim.batteries.h1 import composite_controls as controls
from brainsim.batteries.h1 import composite_report as report
from brainsim.batteries.h1 import composite_runner as runner
from brainsim.batteries.h1.composite_arm import (ACTIVE_PATH_EDGES,
                                                 base_cc, budget_rows,
                                                 forecast_table,
                                                 full_comp_config,
                                                 item_pattern,
                                                 smoke_comp_config,
                                                 tiny_comp_config)
from brainsim.batteries.h1.composite_tf import TorchTF
from brainsim.batteries.h3.budget import TOL, total_synapses
from brainsim.composite.full.runtime import CompositeRuntime
from brainsim.paths import RESULTS

ROOT = Path(__file__).resolve().parents[3]
H1_DIR = RESULTS / "h1"
SEED, SILENT = 0, (lambda *a, **k: None)

# the fly arm's real per-seed ratios (results/h1/fly_arm.json): far
# below the kill band -- the anti-premium case the tests pin
FLY_RATIOS_REAL = [0.4689537566456013, 0.0, 0.0, 0.0, 0.0]


# ---- sharding / merge coverage (h2 amendment rule 5 idiom) --------------------

def test_shard_partition_reproduces_the_unsharded_results(tiny_cfg, partial_all, canon):
    pa = runner.run_shard(tiny_cfg, [SEED], 0, 2, "all", log=SILENT,
                          item_log=SILENT)
    pb = runner.run_shard(tiny_cfg, [SEED], 1, 2, "all", log=SILENT,
                          item_log=SILENT)
    merged = {**pa["results"], **pb["results"]}
    assert canon(merged) == canon(partial_all["results"])
    expected = {report.work_key(it)
                for it in report.build_work_list(tiny_cfg, [SEED])}
    assert set(merged) == expected


def test_exec_class_filters_partition_the_work_list(tiny_cfg):
    work = report.build_work_list(tiny_cfg, [SEED])
    sharded = report.select_items(tiny_cfg, [SEED], 0, 1, "sharded")
    packed = report.select_items(tiny_cfg, [SEED], 0, 1, "packed")
    assert len(sharded) + len(packed) == len(work)
    assert all(it["name"] == "tf_attractor" for it in packed)
    assert len(packed) == 1


def test_collect_refuses_incomplete_duplicate_or_mismatched(tiny_cfg, partial_all):
    keys = sorted(partial_all["results"])
    half = {k: partial_all["results"][k] for k in keys[: len(keys) // 2]}
    incomplete = report.make_partial(tiny_cfg, [SEED], (0, 1), "all", half, 0.0)
    with pytest.raises(report.MergeError):
        report._collect([incomplete])
    with pytest.raises(report.MergeError):  # duplicates
        report._collect([partial_all, incomplete])
    tampered = json.loads(json.dumps(partial_all))
    tampered["config_hash"] = "deadbeef"
    with pytest.raises(report.MergeError):
        report._collect([partial_all, tampered])


def test_merge_h1_refuses_non_full_scale(tiny_cfg, partial_all, tmp_path):
    dest = report.write_partial(json.loads(json.dumps(partial_all)), tmp_path)
    before = sorted(p.name for p in H1_DIR.glob("*"))
    with pytest.raises(report.MergeError, match="full"):
        report.merge_h1([str(dest)], log=SILENT)
    assert sorted(p.name for p in H1_DIR.glob("*")) == before


# ---- verdict wiring (FROZEN verdicts.h1_verdict only) -------------------

def _fake_results(cfg, seeds, c_by_instance):
    out = {}
    for it in report.build_work_list(cfg, seeds):
        c = float(c_by_instance.get(it["name"], 1.0))
        out[report.work_key(it)] = {
            "key": report.work_key(it), "name": it["name"], "seed": it["seed"],
            "exec_class": it["exec_class"], "capacity": {"C": c}, "meta": {}}
    return out


def _fly_stub(ratios):
    return {"ratio_C_meas_over_C_shuf": {"per_seed": list(ratios)}}


def test_verdict_wiring_anti_premium_is_partial_not_killed(tiny_cfg):
    """BOTH arms far BELOW the kill band: the frozen band logic does not
    kill (kill requires [0.95, 1.05] on both) and nothing clears the 1.10
    bar -> partial. Frozen semantics asserted, not re-litigated."""
    bc = report.battery_common()
    seeds = [0, 1, 2]
    res = _fake_results(tiny_cfg, seeds, {"measured": 0.5, "config:0": 2.0})
    stats = report.analyze(tiny_cfg, seeds, res, _fly_stub(FLY_RATIOS_REAL))
    assert stats["ratio_comp"]["per_seed"] == [0.25, 0.25, 0.25]
    assert stats["verdict"] == "partial"
    assert stats["verdict"] == bc.h1_verdict(FLY_RATIOS_REAL, [0.25] * 3)
    assert bc.h1_verdict(FLY_RATIOS_REAL, [0.25] * 3) != "killed"
    assert "anti-premium" in stats["observed"]


def test_verdict_wiring_killed_and_supported(tiny_cfg):
    bc = report.battery_common()
    seeds = [0, 1, 2]
    in_band = [1.0, 1.01, 0.99, 1.02, 0.98]
    res = _fake_results(tiny_cfg, seeds, {"measured": 2.0, "config:0": 2.0})
    stats = report.analyze(tiny_cfg, seeds, res, _fly_stub(in_band))
    assert stats["verdict"] == "killed" == bc.h1_verdict(in_band, [1.0] * 3)
    res = _fake_results(tiny_cfg, seeds, {"measured": 2.4, "config:0": 2.0})
    stats = report.analyze(tiny_cfg, seeds, res, _fly_stub(FLY_RATIOS_REAL))
    assert stats["verdict"] == "supported" == bc.h1_verdict(
        FLY_RATIOS_REAL, [1.2] * 3)


def test_degenerate_composite_ratio_aborts_instead_of_minting_a_verdict(tiny_cfg):
    res = _fake_results(tiny_cfg, [0], {"measured": 1.0, "config:0": 0.0})
    with pytest.raises(report.MergeError, match="degenerate"):
        report.analyze(tiny_cfg, [0], res, _fly_stub(FLY_RATIOS_REAL))


def test_fly_report_loader_validates_the_real_artifact(tmp_path):
    if not report.FLY_ARM_JSON.exists():
        pytest.skip("fly arm artifact not present")
    rep = report.load_fly_report()
    assert rep["n_seeds"] >= 5
    assert all(np.isfinite(r) for r in
               rep["ratio_C_meas_over_C_shuf"]["per_seed"])
    bad = json.loads(report.FLY_ARM_JSON.read_text())
    bad["contract_sha256"] = "0" * 64
    p = tmp_path / "fly_bad.json"
    p.write_text(json.dumps(bad))
    with pytest.raises(report.MergeError, match="contract"):
        report.load_fly_report(p)


# ---- TF control: streamed stats match the TF definition -----------------------

def test_tf_streamed_stats_match_batch_definition():
    net = TorchTF(40, 200, 0.1, seed=9, device="cpu")
    from brainsim.batteries.h1.composite_arm import CompH1Config  # noqa: F401
    from brainsim.batteries.h1.protocol import rng_for
    pats = [rng_for(9, 61, m).choice(40, size=4, replace=False)
            for m in range(6)]
    for p in pats:
        net.store(p)
    vals = net._values().to(torch.float64).numpy()
    from brainsim.rules.torch_rules import synapse_pre_ids
    rows = np.repeat(np.arange(40), (net.indptr[1:] - net.indptr[:-1]).numpy())
    cols = net.col.numpy()
    x = np.zeros((6, 40))
    for r, p in enumerate(pats):
        x[r, p] = 1.0
    dense = (x - 0.1).T @ (x - 0.1)
    assert np.allclose(vals, dense[rows, cols], atol=1e-6)
    net2 = TorchTF(40, 200, 0.1, seed=9, device="cpu")
    assert torch.equal(net.col, net2.col) and torch.equal(net.indptr, net2.indptr)
    assert net.n_directed_entries == 200


def test_tf_work_item_matches_budget(tiny_cfg, partial_all):
    tf = partial_all["results"][
        report.work_key({"name": "tf_attractor", "seed": SEED})]
    target = int(total_synapses(base_cc(tiny_cfg, SEED)))
    assert abs(tf["capacity"]["n_directed_entries"] - target) <= TOL * target
    assert tf["capacity"]["iterations"] == 15
    assert tf["meta"]["kind"] == "tf"


def test_tf_rejects_p_cap_beyond_int16_counters(tiny_cfg):
    from dataclasses import replace
    from brainsim.batteries.h1.composite_tf import run_tf_capacity_comp
    bad = replace(tiny_cfg, p_cap=40000)
    with pytest.raises(ValueError, match="32768"):
        run_tf_capacity_comp(40, 200, 0.1, 0, bad)


# ---- smoke rails ---------------------------------------------------------------

def test_smoke_cli_exits_zero_prints_no_values_and_writes_nothing():
    files_before = sorted(p.name for p in H1_DIR.glob("*")) if H1_DIR.exists() else []
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts/batteries/battery_h1.py"),
         "--arm", "composite", "--scale", "smoke"],
        capture_output=True, text=True, cwd=ROOT, timeout=900)
    assert proc.returncode == 0, proc.stderr
    assert "SMOKE RUN" in proc.stdout
    assert "suppressed" in proc.stdout
    assert "overlap=" not in proc.stdout  # capacity values never printed
    files_after = sorted(p.name for p in H1_DIR.glob("*")) if H1_DIR.exists() else []
    assert files_before == files_after
    assert not (H1_DIR / "results.json").exists() or files_before


def test_sharded_smoke_partials_need_out_dir(tmp_path):
    cfg = smoke_comp_config()
    with pytest.raises(SystemExit):
        runner.run(cfg, [SEED], out_dir=None, arm_shard="3/4", log=SILENT)
    runner.run(cfg, [SEED], out_dir=str(tmp_path), arm_shard="3/4", log=SILENT)
    assert (tmp_path / "composite_partial_3_of_4.json").exists()


def test_smoke_and_full_instance_scopes(tiny_cfg):
    assert len(smoke_comp_config().instances()) == 4   # measured+KO+2 controls
    full = full_comp_config().instances()
    assert len(full) == 10
    assert sum(n.startswith("config:") for n in full) == 5  # contract draws
    assert {"weight_shuffle", "erdos_renyi", "tf_attractor"} <= set(full)
