"""Whole-fly network for the engine gate: data cache, build, reset, trial.

The connectome comes from Shiu et al.'s model repository (Connectivity_783.
parquet and Completeness_783.csv, FlyWire v783), fetched into
<data>/external/Drosophila_brain_model by scripts/data/fetch_shiu_model.sh.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from brainsim.engine.lif import LIFParams
from brainsim.engine.torch_core.network import TorchNetwork
from brainsim.engine.torch_core.propagate import TorchCSRProjection
from brainsim.paths import DATA
from brainsim.validate.whole_fly.mapping import (TAU_MEM, TAU_SYN, T_RFC,
                                                 V_REST, V_TH, W_SYN)

VENDOR = DATA / "external" / "Drosophila_brain_model"
CACHE = DATA / "external" / "fly_v783_npy"

CONVERTER = r"""
import json, os, sys
import numpy as np
import pyarrow.parquet as pq
src, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=True)
t = pq.ParquetFile(src).read(columns=[
    "Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"])
np.save(os.path.join(dst, "pre.npy"), t.column(0).to_numpy().astype(np.int32))
np.save(os.path.join(dst, "post.npy"), t.column(1).to_numpy().astype(np.int32))
np.save(os.path.join(dst, "count.npy"), t.column(2).to_numpy().astype(np.float32))
meta = {"rows": t.num_rows, "src_size": os.path.getsize(src),
        "src_mtime": os.path.getmtime(src)}
json.dump(meta, open(os.path.join(dst, "meta.json"), "w"))
print("converted", t.num_rows, "rows")
"""


def ensure_cache(src: Path) -> None:
    meta_p = CACHE / "meta.json"
    if meta_p.exists():
        meta = json.loads(meta_p.read_text())
        if (meta.get("src_size") == src.stat().st_size
                and abs(meta.get("src_mtime", 0) - src.stat().st_mtime) < 1e-3):
            return
    if importlib.util.find_spec("pyarrow") is None:
        raise RuntimeError("npy cache missing and pyarrow not installed "
                           "(pip install 'brainsim[flywire]')")
    if not src.exists():
        raise RuntimeError(f"{src} missing: run scripts/data/fetch_shiu_model.sh")
    subprocess.run([sys.executable, "-c", CONVERTER, str(src), str(CACHE)],
                   check=True)


def load_flyids() -> list[int]:
    with open(VENDOR / "Completeness_783.csv") as f:
        r = csv.reader(f)
        next(r)
        return [int(row[0]) for row in r]


def build_network(n: int, dt: float, delay_steps: int) -> TorchNetwork:
    pre = np.load(CACHE / "pre.npy")
    post = np.load(CACHE / "post.npy")
    cnt = np.load(CACHE / "count.npy")  # signed synapse count per pair
    params = LIFParams(v_rest=V_REST, v_th=V_TH, v_reset=V_REST,
                       tau_mem=TAU_MEM, tau_syn=TAU_SYN, refractory=T_RFC,
                       R=1.0, dt=dt)
    net = TorchNetwork(seed=0, spike_buffer_size=1, device="cpu")
    net.add_population("fly", n, params)
    csr = TorchCSRProjection.from_table(
        pre_id=pre, post_id=post, weight=cnt * np.float32(W_SYN),
        sign=np.float32(1.0), n_pre=n, n_post=n, delay_steps=delay_steps,
        fp16=False, device="cpu")
    net.add_projection("syn", "fly", "fly", csr)
    return net


def reset(net: TorchNetwork) -> None:
    pop = net.populations["fly"]
    pop.state.v.fill_(V_REST)
    pop.state.I.zero_()
    pop.state.refractory_counter.zero_()
    csr = net.projections["syn"].csr
    csr._ring.zero_()
    csr._ring_ptr = 0
    net.t = 0
    net.spike_buffer.clear()


def run_trial(net: TorchNetwork, driven: np.ndarray, rate_hz: float,
              seed: int, n_steps: int, mn9_idx: int, dt: float,
              i_ev: float) -> dict:
    reset(net)
    rng = np.random.default_rng(seed)
    p = rate_hz * dt / 1000.0
    n = net.populations["fly"].n
    ext = np.zeros(n, dtype=np.float32)
    mn9 = spikes_total = spikes_driven = 0
    for _ in range(n_steps):
        if driven.size and p > 0.0:
            ev = driven[rng.random(driven.size) < p]
            if ev.size:
                ext[ev] = i_ev
            s = net.step({"fly": ext})["fly"]
            if ev.size:
                ext[ev] = 0.0
        else:
            s = net.step()["fly"]
        mn9 += int(s[mn9_idx])
        spikes_total += int(s.sum())
        if driven.size:
            spikes_driven += int(s[driven].sum())
    driven_hz = (spikes_driven / driven.size / (n_steps * dt / 1000.0)
                 if driven.size else 0.0)
    return {"mn9": mn9, "network_spikes": spikes_total,
            "driven_rate_hz": round(driven_hz, 1)}


__all__ = ["CACHE", "VENDOR", "build_network", "ensure_cache", "load_flyids",
           "reset", "run_trial"]
