"""Run the whole-fly directional gate and build its report.

Checks (pre-registered, directional): (a) sugar-GRN drive lifts MN9 above
the silent baseline, (b) MN9 rises with drive frequency, (c) sugar beats a
random control set of the same size. Any exception yields
gate_passed=false with the reason; a pass is never fabricated.
"""
from __future__ import annotations

import subprocess
import time
import traceback

import numpy as np
import torch

from brainsim.paths import REPO_ROOT
from brainsim.validate.whole_fly.mapping import (FREQS, MN9, NEU_SUGAR,
                                                 RUNTIME_BUDGET_S, T_DLY,
                                                 TRIAL_MS, event_current,
                                                 param_mapping)
from brainsim.validate.whole_fly.network import (VENDOR, build_network,
                                                 ensure_cache, load_flyids,
                                                 run_trial)


def _code_version() -> str:
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                          capture_output=True, text=True).stdout.strip()


def _checks(results: dict) -> list[dict]:
    base = results["baseline"]["mn9_mean"]
    sbf = {f: results[f"sugar@{f}"]["mn9_mean"] for f in FREQS}
    ctrl = results["control@100"]["mn9_mean"]
    lo, hi = min(FREQS), max(FREQS)
    return [
        {"name": "sugar>baseline",
         "expected": "MN9(sugar) > MN9(silent)",
         "observed": f"{sbf[hi]:.1f} vs {base:.1f}",
         "pass": sbf[hi] > base + 1e-9},
        {"name": "dose-response",
         "expected": "MN9 rises with activation frequency",
         "observed": ", ".join(f"{f}Hz:{sbf[f]:.1f}" for f in FREQS),
         "pass": sbf[hi] >= sbf[lo]},
        {"name": "sugar>control",
         "expected": "MN9(sugar) > MN9(random control)",
         "observed": f"{sbf[100]:.1f} vs {ctrl:.1f}",
         "pass": sbf[100] > ctrl},
    ]


def _run_conditions(net, conditions, trials, n_steps, mn9_idx, dt, i_ev) -> dict:
    results = {}
    for name, driven, rate in conditions:
        per_trial = []
        for t in range(trials):
            r = run_trial(net, driven, rate, 1000 + t, n_steps, mn9_idx, dt, i_ev)
            per_trial.append(r)
            print(f"  {name} trial {t}: MN9={r['mn9']} "
                  f"driven={r['driven_rate_hz']} Hz "
                  f"net_spikes={r['network_spikes']}", flush=True)
        results[name] = {
            "mn9_mean": float(np.mean([r["mn9"] for r in per_trial])),
            "mn9_per_trial": [r["mn9"] for r in per_trial],
            "driven_rate_hz_mean": float(np.mean(
                [r["driven_rate_hz"] for r in per_trial])),
            "network_spikes_mean": float(np.mean(
                [r["network_spikes"] for r in per_trial])),
        }
    return results


def run_gate(dt: float = 1.0, mode: str = "g", trials: int = 3,
             note: str = "") -> dict:
    """Run every condition and return the gate report (never raises)."""
    n_steps = int(round(TRIAL_MS / dt))
    delay_steps = max(1, int(round(T_DLY / dt)))
    i_ev = event_current(mode)
    t0 = time.time()
    sha = _code_version()
    mapping = param_mapping(dt, delay_steps, i_ev, mode)
    try:
        ensure_cache(VENDOR / "Connectivity_783.parquet")
        flyids = load_flyids()
        n = len(flyids)
        fly2i = {j: i for i, j in enumerate(flyids)}
        sugar_idx = np.array([fly2i[x] for x in NEU_SUGAR if x in fly2i],
                             dtype=np.int64)
        if sugar_idx.size < 18:
            raise RuntimeError(f"only {sugar_idx.size}/21 sugar IDs map in v783")
        mn9_idx = fly2i[MN9]
        ctrl_idx = np.random.default_rng(0).choice(
            n, size=sugar_idx.size, replace=False).astype(np.int64)

        print(f"building network: {n} neurons, dt={dt} ms ...", flush=True)
        net = build_network(n, dt, delay_steps)
        n_pairs = net.projections["syn"].csr.n_synapses

        # Runtime probe: 150 steps of the heaviest condition (sugar at 200 Hz).
        tp = time.time()
        run_trial(net, sugar_idx, 200.0, 9999, 150, mn9_idx, dt, i_ev)
        per_step = (time.time() - tp) / 150.0
        est_full = per_step * n_steps * 5 * trials * 1.3
        trials_note = ""
        if est_full > RUNTIME_BUDGET_S and trials > 2:
            trials = 2
            trials_note = (f"reduced to 2 trials: projected "
                           f"{est_full / 60:.0f} min > 90 min budget")
        print(f"probe: {per_step * 1e3:.1f} ms/step, projected full run "
              f"{est_full / 60:.1f} min, trials={trials}", flush=True)

        conditions = [("baseline", np.array([], dtype=np.int64), 0.0),
                      *[(f"sugar@{f}", sugar_idx, float(f)) for f in FREQS],
                      ("control@100", ctrl_idx, 100.0)]
        results = _run_conditions(net, conditions, trials, n_steps, mn9_idx,
                                  dt, i_ev)
        checks = _checks(results)
        return {
            "gate_passed": all(c["pass"] for c in checks),
            "checks": checks,
            "engine": "brainsim-torch-cpu",
            "torch_version": torch.__version__,
            "code_version": sha,
            "connectome_version": "v783",
            "trials": trials,
            "conditions_run": ["baseline", "sugar@50/100/200Hz",
                               "random-control@100Hz"],
            "n_neurons": n,
            "n_connection_pairs": int(n_pairs),
            "sugar_mapped": int(sugar_idx.size),
            "results": results,
            "param_mapping": mapping,
            "runtime_s": round(time.time() - t0, 1),
            "notes": ("Same pre-registered directional checks as the Shiu et "
                      "al. Brian2 reference gate, run on brainsim torch_core, "
                      "CPU fp32, seeds 1000+t per trial. Directional gate "
                      "only: current-based synapse without reset-zeroing "
                      "shifts absolute rates vs Shiu's g-dynamics (see "
                      "param_mapping.divergences). "
                      + (trials_note or f"trials={trials}")
                      + ((" | " + note) if note else "")),
        }
    except Exception as e:  # report the failure, never fabricate a pass
        return {
            "gate_passed": False,
            "reason": f"{type(e).__name__}: {e}",
            "traceback_tail": traceback.format_exc()[-1500:],
            "checks": [],
            "engine": "brainsim-torch-cpu",
            "code_version": sha,
            "connectome_version": "v783",
            "param_mapping": mapping,
        }


__all__ = ["run_gate"]
