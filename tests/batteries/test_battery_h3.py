"""Battery H3 driver tests (tiny scale: structure, determinism, matched
budgets, sharding, verdict wiring, smoke rails -- never magnitudes)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
torch = pytest.importorskip("torch")  # noqa: E402 (gates the imports below)
from brainsim.batteries.h2.config import H2Config
from brainsim.batteries.h3 import arms, budget, protocol, report, runner, screen
from brainsim.batteries.h3.config import full_config, tiny_config
from brainsim.composite.full import phases
from brainsim.composite.full.build import forecast
from brainsim.paths import RESULTS

ROOT = Path(__file__).resolve().parents[2]
RESULTS_JSON = RESULTS / "h3" / "results.json"
SEED, SILENT = 101, (lambda *a, **k: None)


@pytest.fixture(scope="module")
def tiny_cfg():
    return tiny_config(ko_singles=("ct:granule",), ko_pairs_override=0)


@pytest.fixture(scope="module")
def merged_all(tiny_cfg):
    """One full tiny battery (1/1) + its 2-shard partition, merged."""
    p_all = runner.run_shard(tiny_cfg, [SEED], 0, 1, "all", log=SILENT)
    pa = runner.run_shard(tiny_cfg, [SEED], 0, 2, "all", log=SILENT)
    pb = runner.run_shard(tiny_cfg, [SEED], 1, 2, "all", log=SILENT)
    return (report.merge_partials([p_all], write_final=False),
            report.merge_partials([pa, pb], write_final=False), p_all)


# -- matched-budget invariants (driver-spec: assertions, not comments) ---------

def test_full_scale_budgets_within_one_percent():
    tbl = budget.budget_table(full_config(), SEED)
    assert tbl["composite"]["target"] >= 1e10  # frozen validate.py floor
    for name, row in tbl.items():
        assert abs(row["ratio"] - 1.0) <= budget.TOL, (name, row)
    assert tbl["bag"]["n_modules"] > 1


def test_tiny_budgets_within_one_percent(tiny_cfg):
    for name, row in budget.budget_table(tiny_cfg, SEED).items():
        assert abs(row["ratio"] - 1.0) <= budget.TOL, (name, row)


def test_budget_assertion_fires():
    with pytest.raises(budget.BudgetError):
        budget.assert_matched("x", 102, 100)  # 2% off


def test_full_scale_arms_fit_the_node():
    """Every composite-scale arm must fit 8 x 141 GB (they run sharded;
    stage-alone arms do NOT fit one device at full scale, measured here)."""
    cfg = full_config()
    for name, (kind, _) in arms.ARMS.items():
        if kind == "bag":
            continue
        cc = arms.arm_config(cfg, SEED, name)
        assert forecast(cc, 8)["per_rank_total"] <= 141e9, name
    one_gpu = forecast(arms.arm_config(cfg, SEED, "formation_alone"), 1)
    assert one_gpu["per_rank_total"] > 141e9  # packed is bag-only at full


def test_full_scale_bag_group_fits_one_gpu():
    cfg = full_config()
    n_mod, tgt = budget.bag_layout(cfg, SEED)
    per_group = 0
    for m in range(n_mod):
        if m % cfg.bag_groups == 0:
            mc = budget.bag_module_config(cfg, SEED, m, n_mod, tgt)
            per_group += forecast(mc, 1)["per_rank_total"]
    assert per_group <= 141e9


# -- shared accelerated-turnover constant (driver-spec: ONE across batteries) --

def test_turnover_constant_imported_from_h2(tiny_cfg):
    assert protocol.TURNOVER_SIGMA_PER_DAY == H2Config.sigma
    assert protocol.shared_turnover_constant() == H2Config.sigma
    cc = arms.arm_config(tiny_cfg, SEED, "composite")
    assert cc.turnover_noise_sigma == H2Config.sigma


# -- seed/layout independence (h2 amendment rule 5 idiom) ----------------------

def _canon(m):
    keep = {k: v for k, v in m.items() if k != "config"}
    return json.dumps(keep, sort_keys=True, default=str)


def test_shard_merge_equals_unsharded(merged_all, tiny_cfg):
    m_all, m_sh, _ = merged_all
    assert _canon(m_all) == _canon(m_sh)
    expected = {report.work_key(it)
                for it in report.build_work_list(tiny_cfg, [SEED])}
    assert set(m_all["results"]) == expected
    assert m_all["verdict"] in {"supported", "killed", "partial"}


def test_class_filter_partition_merges_identically(merged_all, tiny_cfg):
    """Splitting the SAME results by exec class (sharded vs packed, the
    full-scale invocation layout) must merge to the identical analysis."""
    m_all, _, p_all = merged_all
    classes = {report.work_key(it): it["exec_class"]
               for it in report.build_work_list(tiny_cfg, [SEED])}
    parts = []
    for f in ("sharded", "packed"):
        sub = {k: v for k, v in p_all["results"].items() if classes[k] == f}
        parts.append(report.make_partial(tiny_cfg, [SEED], (0, 1), f, sub, 0.0))
    assert _canon(report.merge_partials(parts, write_final=False)) == _canon(m_all)


def test_merge_refuses_incomplete_duplicate_or_mismatched(merged_all, tiny_cfg):
    _, _, p_all = merged_all
    keys = sorted(p_all["results"])
    half = {k: p_all["results"][k] for k in keys[: len(keys) // 2]}
    incomplete = report.make_partial(tiny_cfg, [SEED], (0, 1), "all", half, 0.0)
    with pytest.raises(report.MergeError):
        report.merge_partials([incomplete], write_final=False)
    with pytest.raises(report.MergeError):  # duplicate work items
        report.merge_partials([p_all, incomplete], write_final=False)
    tampered = json.loads(json.dumps(p_all))
    tampered["config_hash"] = "deadbeef"
    with pytest.raises(report.MergeError):
        report.merge_partials([p_all, tampered], write_final=False)


def test_merge_never_writes_final_below_full_scale(merged_all):
    _, _, p_all = merged_all
    before = RESULTS_JSON.read_bytes() if RESULTS_JSON.exists() else None
    with pytest.raises(report.MergeError):
        report.merge_partials([p_all], write_final=True)  # scale == tiny
    after = RESULTS_JSON.read_bytes() if RESULTS_JSON.exists() else None
    assert before == after


# -- screen ---------------------------------------------------------------------

def test_screen_determinism(tiny_cfg):
    it = {"kind": "ko", "name": "ct:granule", "lf": 1.0, "group": None,
          "seed": SEED}
    a = runner.run_work_item(tiny_cfg, it)
    b = runner.run_work_item(tiny_cfg, it)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_pair_sampling_budget_and_determinism():
    cfg = full_config()
    singles = screen.single_names(cfg)
    pairs = screen.pair_names(cfg)
    assert pairs == screen.pair_names(cfg)  # seeded, layout-independent
    assert len(pairs) == min(2 * len(singles),
                             len(singles) * (len(singles) - 1) // 2)
    assert len(set(pairs)) == len(pairs)
    assert len(singles) == len(screen.SINGLE_KOS) == 11


def test_ko_ops_freeze_their_targets(tiny_cfg):
    pp_pat = protocol.item_pattern(protocol.screen_protocol(tiny_cfg, 0.0),
                                   SEED, 0)
    for name, edge in (("ct:pyramidal", "form_rec"), ("ct:storage", "store_rec"),
                       ("ct:granule", "sep_in")):
        system, _, _ = screen.make_ko_system(tiny_cfg, SEED, name)
        system.write_item(0, pp_pat, 1.0, True)
        w = sum(float(system.rt.net.projections[p].weights.abs().sum())
                for p in system.rt.edges[edge].parts)
        assert w == 0.0, (name, edge, w)
    system, _, _ = screen.make_ko_system(tiny_cfg, SEED, "motif:multi_contact_tail")
    assert system.rt.config.multi_contact_frac == 0.0


def test_mahalanobis_ranking_is_deterministic_and_ranked():
    vecs = {"a": [0.0, 0.0, 0.0, 0.0], "b": [0.1, 0.0, 0.1, 0.0],
            "c": [5.0, 4.0, 3.0, 2.0]}
    r1 = screen.mahalanobis_ranking(vecs)
    assert [x["ko"] for x in r1] == [x["ko"] for x in screen.mahalanobis_ranking(vecs)]
    assert r1[0]["distance"] >= r1[-1]["distance"]
    assert len(screen.followup_list(r1, 2)) == 2


# -- verdict wiring (FROZEN verdicts.h3_verdict only) ---------------------

def _fake_results(cfg, seeds, retained):
    """Minimal metrics: arm -> retained count at T (out of n_items)."""
    out = {}
    for it in report.build_work_list(cfg, seeds):
        n = cfg.n_items if it["kind"] == "arm" else cfg.screen_n_items
        r = retained.get(it["name"], 0)
        row = {"day": 0, "L": 0, "n": n, "retained": r, "score_sum": float(r),
               "fast_retained": r, "slow_retained": 0, "fast_sum": float(r),
               "slow_sum": 0.0}
        m = {"n_items": n, "epochs": [dict(row), dict(row)],
             "separation": {"n": 2, "margin_sum": 1.0},
             "retention": [dict(row), dict(row, day=cfg.retention_days)],
             "lambda_rate": 0.0, "replay_events_total": 0,
             "interference_total": 0, "turnover_sigma_per_day": 0.03}
        meta = {"arm": it["name"], "kind": it["kind"], "budget_total": 100,
                "budget_target": 100, "lambda_star_pred": 0.1, "n_neurons": 10}
        out[report.work_key(it)] = {
            "key": report.work_key(it), "kind": it["kind"], "name": it["name"],
            "seed": it["seed"], "lf": it["lf"], "group": it["group"],
            "meta": meta, "metrics": m}
    return out


def test_verdict_wiring_killed_partial_supported(tiny_cfg):
    cfg = tiny_config(run_bag=False, ko_singles=("ct:granule",),
                      ko_pairs_override=0, n_items=100, screen_n_items=100)
    seeds = [1, 2, 3]
    bc = report.battery_common()
    for comp, part, want in ((40, 50, "killed"), (50, 50, "killed"),
                             (55, 50, "partial"), (80, 50, "supported")):
        res = _fake_results(cfg, seeds, {"composite": comp,
                                         "formation_alone": part})
        a = report.analyze(cfg, seeds, res)
        assert a["verdict"] == want, (comp, part, a["verdict"])
        assert a["verdict"] == bc.h3_verdict(a["composite_retained"],
                                             a["best_part_retained"])
    assert bc.h3_verdict(100.0, 79.9) == "supported"  # 1.25x bar frozen


# -- risky arm code paths --------------------------------------------------------

def test_stdp_swap_writes_and_restores(tiny_cfg):
    orig = phases._btsp_write
    system, pp, _ = arms.make_arm_system(tiny_cfg, SEED, "btsp_to_stdp", 1.0)
    system.write_item(0, protocol.item_pattern(pp, SEED, 0), 1.0, False)
    w = sum(float(system.rt.net.projections[p].weights.abs().sum())
            for p in system.rt.edges["form_rec"].parts)
    assert w > 0.0  # the swap really writes (incremental, not one-shot)
    assert phases._btsp_write is orig  # scoped patch restored


def test_poisson_boundary_arm_runs(tiny_cfg):
    system, pp, _ = arms.make_arm_system(tiny_cfg, SEED, "boundary_poisson", 1.0)
    system.write_item(0, protocol.item_pattern(pp, SEED, 0), 1.0, False)
    codes = system.rt.items[0].codes
    assert all(c.size > 0 for c in codes.values())


def test_allocation_off_runs_with_zero_k(tiny_cfg):
    system, pp, _ = arms.make_arm_system(tiny_cfg, SEED, "allocation_off", 1.0)
    assert system.rt.config.alloc_k == 0
    system.write_item(0, protocol.item_pattern(pp, SEED, 0), 1.0, False)
    assert 0 in system.rt.items


def test_all_full_arm_configs_construct():
    cfg = full_config()
    for name, (kind, readout) in arms.ARMS.items():
        if kind == "bag":
            continue
        cc = arms.arm_config(cfg, SEED, name)
        assert cc.n_global > 0
        assert readout in ("union", "separation", "formation", "output",
                           "relay", "storage")
    assert len(cfg.ablation_arms) == 10  # the 10 frozen ablations


# -- smoke CLI rails ---------------------------------------------------------------

def test_smoke_cli_exits_zero_and_writes_nothing(tmp_path):
    """Real CLI, print-only: exit 0, results/h3 untouched, every
    measured value suppressed. Patched to tiny for suite speed (same code
    paths and rails as smoke; scale != 'full' is the guarded branch)."""
    h3_dir = RESULTS_JSON.parent
    files_before = sorted(p.name for p in h3_dir.glob("*")) if h3_dir.exists() else []
    patch = (
        "import brainsim.batteries.h3.runner as r, brainsim.batteries.h3.config as c;"
        "r.config_for = lambda scale, engine, **kw: c.tiny_config("
        "ko_singles=('ct:granule',), ko_pairs_override=0, run_bag=False);"
        "import runpy, sys; sys.argv = ['battery_h3.py', '--engine', 'torch',"
        " '--scale', 'smoke', '--seeds', '1'];"
        f"runpy.run_path(r'{ROOT}/scripts/batteries/battery_h3.py', run_name='__main__')"
    )
    proc = subprocess.run([sys.executable, "-c", patch], capture_output=True,
                          text=True, cwd=ROOT, timeout=900)
    assert proc.returncode == 0, proc.stderr
    assert "SMOKE RUN" in proc.stdout
    assert "suppressed" in proc.stdout
    files_after = sorted(p.name for p in h3_dir.glob("*")) if h3_dir.exists() else []
    assert files_before == files_after
    assert not RESULTS_JSON.exists() or files_before  # no new results.json


def test_sharded_smoke_partials_need_out_dir(tiny_cfg, tmp_path):
    with pytest.raises(SystemExit):
        runner.run(tiny_cfg, [SEED], out_dir=None, arm_shard="0/9", log=SILENT)
    runner.run(tiny_cfg, [SEED], out_dir=str(tmp_path), arm_shard="8/9",
               log=SILENT)
    assert (tmp_path / "partial_8_of_9.json").exists()
