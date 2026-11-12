"""Dopamine-gated eligibility rule, mushroom-body form (F1, Jiang &
Litwin-Kumar 2021, PLOS CB 10.1371/journal.pcbi.1009205).

    dw_ij/dt = alpha * (e_KC * e_DAN) / tau_W,  tau_W = 5 s, w_max = 0.05

Kenyon-cell (KC, presynaptic) activity leaves an eligibility trace e_KC; a
coincident dopaminergic (DAN) signal e_DAN drains the synapse toward a floor
(`beta_floor`, default 0) rather than potentiating it -- this is the
depression-dominant readout pathway used to shape valence at the MB output
(the sign is a documented modelling choice; the design doc gives the
magnitude equation only). `beta_floor` keeps a residual, non-erasable weight
if configured above 0.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MBRuleParams:
    alpha: float = 1.0
    tau_w: float = 5000.0  # ms (5 s), Jiang & Litwin-Kumar 2021
    tau_e: float = 200.0  # ms, KC eligibility trace decay (not pinned by F1; free param)
    w_max: float = 0.05  # Jiang & Litwin-Kumar 2021
    beta_floor: float = 0.0  # optional non-zero floor, F1 "optional beta floor"
    dt: float = 1.0  # ms


@dataclass
class MBRuleState:
    e_kc: np.ndarray  # per presynaptic (KC) neuron
    W: np.ndarray  # per synapse, initialised near w_max

    @classmethod
    def zeros(cls, n_pre: int, n_syn: int, w_init: float | None = None) -> "MBRuleState":
        params_w_max = 0.05 if w_init is None else w_init
        return cls(
            e_kc=np.zeros(n_pre, dtype=np.float64),
            W=np.full(n_syn, params_w_max, dtype=np.float64),
        )


def step(
    state: MBRuleState,
    params: MBRuleParams,
    kc_spikes: np.ndarray,
    dan_signal: np.ndarray,
    pre_idx: np.ndarray,
) -> np.ndarray:
    """Advance the KC eligibility trace and depress W by one dt in place.

    `dan_signal` is per-synapse (or broadcastable), representing the
    dopaminergic drive gating each synapse's depression this step.
    """
    state.e_kc = state.e_kc * np.exp(-params.dt / params.tau_e)
    state.e_kc = np.where(kc_spikes, 1.0, state.e_kc)

    dw = params.alpha * (state.e_kc[pre_idx] * dan_signal) / params.tau_w * params.dt
    state.W = np.clip(state.W - dw, params.beta_floor, params.w_max)
    return state.W


__all__ = ["MBRuleParams", "MBRuleState", "step"]
