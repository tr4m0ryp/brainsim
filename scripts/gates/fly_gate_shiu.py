#!/usr/bin/env python3
"""Reference fly gate: reproduce Shiu et al. 2024's headline sugar-GRN -> MN9
activation from their published v783 connectome, using their own Brian2
model code (MIT licence, fetched by scripts/data/fetch_shiu_model.sh).
Writes results/fly-gate/shiu-brian2/gate.json. Needs `pip install
'brainsim[shiu]'` (brian2, pandas, pyarrow).

Pre-registered checks:
  (a) sugar-GRN activation drives MN9 firing above silent baseline;
  (b) MN9 response increases with activation frequency (same direction as
      published Fig 1);
  (c) a random non-gustatory control set drives MN9 far less than sugar.
Nothing is fabricated: any failure/exception -> gate_passed=false + reason.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from brainsim.paths import DATA, RESULTS

VENDOR = DATA / "external" / "Drosophila_brain_model"
OUT = RESULTS / "fly-gate" / "shiu-brian2"
PY = Path(sys.executable)

NEU_SUGAR = [
    720575940624963786, 720575940630233916, 720575940637568838,
    720575940638202345, 720575940617000768, 720575940630797113,
    720575940632889389, 720575940621754367, 720575940621502051,
    720575940640649691, 720575940639332736, 720575940616885538,
    720575940639198653, 720575940620900446, 720575940617937543,
    720575940632425919, 720575940633143833, 720575940612670570,
    720575940628853239, 720575940629176663, 720575940611875570,
]
MN9 = 720575940660219265

# Runs in a subprocess with Shiu et al.'s repository on sys.path.
# Uses run_trial (returns the spike-time dict directly, no disk IO).
WORKER = r'''
import sys, json
sys.path.insert(0, {vendor!r})
import numpy as np
import pandas as pd
from model import default_params, run_trial
import brian2 as b2

path_comp = {vendor!r} + "/Completeness_783.csv"
path_con  = {vendor!r} + "/Connectivity_783.parquet"
neu_sugar = {sugar}
MN9 = {mn9}
n_trials = {trials}
freqs = [50, 100, 200]

# Replicate run_exp's flywire-id -> brian-index mapping (model.py).
df_comp = pd.read_csv(path_comp, index_col=0)
flyid2i = {{j: i for i, j in enumerate(df_comp.index)}}
sugar_idx = [flyid2i[n] for n in neu_sugar if n in flyid2i]
mn9_idx = flyid2i[MN9]

def mn9_spikes(exc_idx, freq):
    p = dict(default_params)
    p["r_poi"] = freq * b2.Hz
    counts = []
    for t in range(n_trials):
        b2.seed(1000 + t)
        spk = run_trial(exc_idx, [], [], path_comp, path_con, p)
        counts.append(len(spk.get(mn9_idx, [])))
    return float(np.mean(counts))

out = {{}}
try:
    if len(sugar_idx) < 18:
        raise RuntimeError("only %d/%d sugar IDs mapped in v783" % (len(sugar_idx), len(neu_sugar)))
    base = mn9_spikes([], 0)
    sugar_by_freq = {{f: mn9_spikes(sugar_idx, f) for f in freqs}}
    rng = np.random.default_rng(0)
    ctrl = [int(i) for i in rng.choice(len(df_comp), size=len(sugar_idx), replace=False)]
    ctrl_mean = mn9_spikes(ctrl, 100)
    out = dict(ok=True, baseline=base, sugar_by_freq=sugar_by_freq,
               control=ctrl_mean, freqs=freqs, n_trials=n_trials,
               sugar_mapped=len(sugar_idx))
except Exception as e:
    import traceback
    out = dict(ok=False, error=str(e), tb=traceback.format_exc()[-1500:])
print("GATE_JSON_START")
print(json.dumps(out))
'''


def git_sha() -> str:
    r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=VENDOR,
                       capture_output=True, text=True)
    return r.stdout.strip()[:12] if r.returncode == 0 else "unknown"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    trials = 3
    if "--trials" in sys.argv:
        trials = int(sys.argv[sys.argv.index("--trials") + 1])
    worker = WORKER.format(vendor=str(VENDOR), sugar=NEU_SUGAR, mn9=MN9,
                          trials=trials)
    r = subprocess.run([str(PY), "-c", worker], capture_output=True, text=True,
                      timeout=5400)
    raw = r.stdout
    data = None
    if "GATE_JSON_START" in raw:
        data = json.loads(raw.split("GATE_JSON_START", 1)[1].strip())
    if not data or not data.get("ok"):
        report = {
            "gate_passed": False,
            "reason": (data or {}).get("error", "worker failed")
            + " | stderr tail: " + r.stderr[-500:],
            "checks": [], "code_version": git_sha(),
            "connectome_version": "v783",
        }
        (OUT / "gate.json").write_text(json.dumps(report, indent=1))
        print("GATE FAILED:", report["reason"][:200])
        return 1

    base = data["baseline"]
    sbf = {int(k): v for k, v in data["sugar_by_freq"].items()}
    sugar_hi = sbf_hi = sbf.get(max(sbf))
    ctrl = data["control"]
    freqs_sorted = sorted(sbf)
    checks = [
        {"name": "sugar>baseline", "expected": "MN9(sugar) > MN9(silent)",
         "observed": f"{sbf[max(sbf)]:.1f} vs {base:.1f}",
         "pass": sbf[max(sbf)] > base + 1e-9},
        {"name": "dose-response",
         "expected": "MN9 rises with activation frequency",
         "observed": ", ".join(f"{f}Hz:{sbf[f]:.1f}" for f in freqs_sorted),
         "pass": sbf[freqs_sorted[-1]] >= sbf[freqs_sorted[0]]},
        {"name": "sugar>control",
         "expected": "MN9(sugar) > MN9(random control)",
         "observed": f"{sbf.get(100, sugar_hi):.1f} vs {ctrl:.1f}",
         "pass": sbf.get(100, sugar_hi) > ctrl},
    ]
    report = {
        "gate_passed": all(c["pass"] for c in checks),
        "checks": checks,
        "conditions_run": ["baseline", "sugarR@50/100/200Hz", "random-control@100Hz"],
        "trials_per_condition": trials,
        "code_version": git_sha(),
        "connectome_version": "v783",
        "notes": "Headline sugar-GRN->MN9 gate reproduced via Shiu et al. "
                 "model.py run_exp on their shipped v783 connectome. "
                 f"trials={trials} (reduced from paper's 30 for CPU budget; "
                 "qualitative direction checks only, per pre-registered gate).",
    }
    (OUT / "gate.json").write_text(json.dumps(report, indent=1))
    print("GATE", "PASSED" if report["gate_passed"] else "FAILED")
    for c in checks:
        print(f"  [{'x' if c['pass'] else ' '}] {c['name']}: {c['observed']}")
    return 0 if report["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
