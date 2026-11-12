"""Runtime wrappers for H3 arms and KOs, all EXTERNAL to the read-only
composite runtime: the H3Runtime subclass carries the two manipulations
without a public hook (top-f readout KO, input-boundary Poisson surrogate)
and OPS holds the named knockout operations applied to a built runtime.
BindingSwap lives in binding.py, the System adapters in adapters.py; both
are re-exported here as the package's runtime-wrapper surface.

SPMD: every manipulation is host-deterministic and identical across ranks
(shared-seed host rng for the Poisson boundary; param replacement and edge
zeroing act on each rank's own parts of the same logical edges).
"""
from __future__ import annotations

from dataclasses import replace as dc_replace

import numpy as np
import torch

from brainsim.composite.full.runtime import CompositeRuntime

from .adapters import BagSystem, SingleSystem
from .binding import BindingSwap
from .flags import RunFlags

BOUNDARY_SPIKE_P = 0.2  # engineering: Bernoulli spike prob/step, rate-matched


class H3Runtime(CompositeRuntime):
    """CompositeRuntime + the two manipulations that need a hook the public
    API does not expose: top-f readout KO (interneuron abstraction, per
    config docstring) and the input-boundary Poisson drive surrogate."""

    def __init__(self, config, rank=0, world_size=1, device="cpu",
                 topf_off: tuple = (), poisson: bool = False):
        super().__init__(config, rank=rank, world_size=world_size, device=device)
        self._topf_off = frozenset(topf_off)
        self._poisson_p = BOUNDARY_SPIKE_P if poisson else None
        # shared-seed host rng: identical Bernoulli stream on every rank
        self._pois_rng = np.random.default_rng([config.seed, 0xB0A])

    def code_from_window(self, stage, window):
        if stage not in self._topf_off:
            return super().code_from_window(stage, window)
        counts, _ = window  # KO of the FF-inhibition abstraction: no top-f
        fired = torch.nonzero(counts > 0).flatten().cpu().numpy()
        return np.sort(fired) + self.ranges[stage][0]

    def run_window(self, drive, steps, collect, drive_steps=None,
                   count_from=0, inhibit=()):
        if self._poisson_p is None:
            return super().run_window(drive, steps, collect, drive_steps,
                                      count_from, inhibit)
        assert not inhibit, "poisson boundary arm never uses inhibit mode"
        p = self._poisson_p
        self._reset_dynamics()
        drive_steps = steps if drive_steps is None else drive_steps
        counts = torch.zeros(self.net.n_local, dtype=torch.int64, device=self.device)
        i_sum = torch.zeros(self.net.n_local, dtype=torch.float32, device=self.device)
        in_lo, in_hi = self.ranges["input"]
        idx = torch.nonzero(drive, as_tuple=False).flatten()
        idx = idx[(idx >= in_lo) & (idx < in_hi)]  # only the input BOUNDARY
        vals = drive[idx] / p  # rate-matched: mean drive preserved
        rest = drive.clone()
        rest[idx] = 0.0  # non-boundary components stay correlation-preserving
        zero = torch.zeros_like(drive)
        buf = torch.zeros_like(drive)
        for t in range(steps):
            if t < drive_steps:
                buf.copy_(rest)
                if idx.numel():
                    keep = torch.as_tensor(
                        self._pois_rng.random(idx.shape[0]) < p,
                        device=self.device)
                    buf[idx[keep]] = vals[keep]
                ext = buf
            else:
                ext = zero
            spikes = self.net.step(external_I=ext)
            if t >= count_from:
                counts += torch.from_numpy(spikes).to(self.device)
                i_sum += self.net.state.I
        self.steps_total += steps
        out = {}
        for stage in collect:  # same global mapping as the parent window
            plo, phi = self.ranges[stage]
            g_c = torch.zeros(phi - plo, dtype=torch.int64, device=self.device)
            g_i = torch.zeros(phi - plo, dtype=torch.float32, device=self.device)
            olo, ohi = max(plo, self.net.lo), min(phi, self.net.hi)
            if olo < ohi:
                g_c[olo - plo: ohi - plo] = counts[olo - self.net.lo: ohi - self.net.lo]
                g_i[olo - plo: ohi - plo] = i_sum[olo - self.net.lo: ohi - self.net.lo]
            if self.world_size > 1:
                import torch.distributed as dist
                dist.all_reduce(g_c)
                dist.all_reduce(g_i)
            out[stage] = (g_c, g_i)
        return out


# ---- KO ops (applied to a BUILT runtime; zero-init plastic edges stay zero
# once their rule ceiling is zeroed: btsp/mb clamp to [0, w_max]) ------------

def _zero_edges(rt, *names):
    for name in names:
        for p in rt.edges[name].parts:
            rt.net.projections[p].weights.zero_()


OPS = {
    "zero:sep_in": lambda rt: _zero_edges(rt, "sep_in"),
    "zero:form_in": lambda rt: _zero_edges(rt, "form_in"),
    "zero:form_direct": lambda rt: _zero_edges(rt, "form_direct"),
    "zero:out_in": lambda rt: _zero_edges(rt, "out_in"),
    "zero:store_in": lambda rt: _zero_edges(rt, "store_in"),
    "freeze_btsp": lambda rt: setattr(
        rt, "btsp_params", dc_replace(rt.btsp_params, w_max=0.0)),
    "freeze_store": lambda rt: setattr(rt, "store_unit", 0.0),
    "neutral_mb": lambda rt: (
        _zero_edges(rt, "relay_in"),
        setattr(rt, "mb_params",
                dc_replace(rt.mb_params, w_max=0.0, beta_floor=0.0)))[0],
}


def build_runtime(cc, flags: RunFlags, rank=0, world_size=1, device="cpu"):
    rt = H3Runtime(cc, rank=rank, world_size=world_size, device=device,
                   topf_off=flags.topf_off, poisson=flags.poisson)
    for op in flags.ops:
        OPS[op](rt)
    return rt



__all__ = ["RunFlags", "H3Runtime", "BindingSwap", "SingleSystem",
           "BagSystem", "build_runtime", "OPS", "BOUNDARY_SPIKE_P"]
