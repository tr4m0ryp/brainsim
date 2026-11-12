"""Scoped BTSP->STDP binding swap (ablation arm). The runtime exposes no
binding hook; the least-drift realization is a scoped override of
phases._btsp_write, restored on exit -- never a re-implementation of the
write phase."""
from __future__ import annotations

import numpy as np
import torch

from brainsim.composite.full import phases
from brainsim.composite.full.codes import bool_at
from brainsim.rules.stdp import STDPParams
from brainsim.rules.torch_rules import TorchSTDPState, stdp as t_stdp


class BindingSwap:
    """Scoped BTSP->STDP swap for the write path. phases.write_phase calls
    the module-level _btsp_write; the runtime exposes no binding hook, so the
    least-drift realization is a scoped module-attribute override (restored
    in __exit__), never a re-implementation of the write phase."""

    def __init__(self, rt):
        self.rt = rt
        w_max = rt.config.btsp_w_max()  # matched ceiling; the RULE differs
        # Bi & Poo kernel; amplitudes scale with the ceiling exactly as the
        # STDPParams defaults scale with w_max=1 (engineering, flagged).
        self.params = STDPParams(a_plus=0.01 * w_max, a_minus=0.012 * w_max,
                                 w_min=0.0, w_max=w_max, dt=1.0)
        self.states: dict[str, TorchSTDPState] = {}

    def _write(self, rt, engram: np.ndarray) -> None:
        """One write = a causal pairing episode at 1 ms steps: a pre volley
        builds the pre trace, then a post volley reads it (pre-before-post).
        Simultaneous volleys would hand every pairing to Bi&Poo's
        depression-dominant branch and freeze W at 0 (measured); the split
        is the engineering discretization of a causal co-activation write,
        flagged. Incremental-vs-one-shot is the mechanism contrast."""
        cfg = rt.config
        es = rt.edges["form_rec"]
        lo, hi = rt.net.local_range
        pre = bool_at(cfg.n_global, engram, rt.device)
        post = bool_at(rt.net.n_local,
                       engram[(engram >= lo) & (engram < hi)] - lo, rt.device)
        no_pre, no_post = torch.zeros_like(pre), torch.zeros_like(post)
        half = max(1, cfg.btsp_write_steps // 2)
        for p in es.parts:
            st = self.states.get(p)
            if st is None:
                st = self.states[p] = TorchSTDPState.for_projection(
                    rt.net.projections[p])
            st.trace_pre.zero_()   # 5 s inter-item rest: traces are exactly
            st.trace_post.zero_()  # dead at Bi&Poo taus; write = one episode
            for _ in range(half):
                t_stdp.step(st, self.params, pre, no_post,
                            es.pre_idx[p], es.post_idx[p])
            for _ in range(half):
                t_stdp.step(st, self.params, no_pre, post,
                            es.pre_idx[p], es.post_idx[p])

    def __enter__(self):
        self._orig = phases._btsp_write
        phases._btsp_write = self._write
        return self

    def __exit__(self, *exc):
        phases._btsp_write = self._orig
        return False


__all__ = ["BindingSwap"]
