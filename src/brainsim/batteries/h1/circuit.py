"""MB circuit dynamics for the capacity protocol.

Forward pass (rate reduction of the feedforward MB, engineering choice
recorded in the results JSON -- full LIF spiking at capacity-sweep load does
not fit the CPU budget and the frozen contract pins the RULE, not the neuron
model):
  h = x_PN @ W(PN->KC)                        anatomical counts as drive
  y0 = top-K(h),  K = round(f_kc * n_KC)      global sparseness (KC-WTA)
  a  = y0 @ W(KC->APL)                        measured APL loop, per APL unit
  inh_j = sum_apl a_apl * W(APL->KC)[j]
  h' = h - g * inh, g: mean(g*inh) = gamma * mean(h) per pattern
  y  = top-K(h')                              inhibition RE-RANKS at fixed K
The feedforward-inhibition knockout (Formal Model prediction 1, the kappa
carrier) sets gamma = 0: same coding level, no measured-inhibition
re-ranking. Ties in top-K break by index (deterministic).

Write: rules/mb_rule closed form. During a T_write presentation the KC
eligibility e is pinned at 1 by ongoing spikes and the DAN gate is 1 on
synapses onto the item's target MBONs, so iterating mb_rule.step gives
exactly dw = alpha * n_steps * dt / tau_w on those synapses (clipped to
[beta_floor, w_max]). test_battery_h1 proves the equivalence against
mb_rule.step; the protocol applies the closed form.

Recall readout: MBON response m = y_cue @ (M * W); suppression relative to
the naive circuit s_i = 1 - m_i / m0_i with m0 = w_max * (y_cue @ M);
retrieved target = top-|z| MBONs by suppression.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from brainsim.batteries.h1.mb_extract import MBGraph
from brainsim.rules.mb_rule import MBRuleParams

RULE_PARAMS = MBRuleParams()          # frozen defaults (F1 parameters)
T_WRITE_STEPS = 100                   # 100 ms presentation at dt=1 ms
DELTA_W = (RULE_PARAMS.alpha * RULE_PARAMS.dt * T_WRITE_STEPS
           / RULE_PARAMS.tau_w)       # = 0.02 = 0.4 * w_max per pairing
# (calibrated 200 -> 100 ms together with f_z, see protocol.py CALIBRATION
# HISTORY: chosen before any capacity value existed, identical for all arms)


def closed_form_delta(n_steps: int = T_WRITE_STEPS,
                      params: MBRuleParams = RULE_PARAMS) -> float:
    """dw for a synapse with e=1 and dan=1 held for n_steps (unclipped)."""
    return params.alpha * params.dt * n_steps / params.tau_w


def topk_rows(h: np.ndarray, k: int) -> np.ndarray:
    """Deterministic per-row top-k -> boolean matrix. Ties break by index."""
    # stable argsort on -h: equal values keep ascending index order.
    order = np.argsort(-h, axis=1, kind="stable")[:, :k]
    out = np.zeros(h.shape, dtype=bool)
    np.put_along_axis(out, order, True, axis=1)
    return out


@dataclass
class MBCircuit:
    """Batched rate-level MB circuit over a (possibly surrogate) MBGraph."""
    W_pk: np.ndarray    # (n_pn, n_kc) drive counts
    M_km: np.ndarray    # (n_kc, n_mbon) anatomical multiplicities (0 = no synapse)
    w_k2a: np.ndarray   # (n_kc, n_apl) KC->APL counts
    w_a2k: np.ndarray   # (n_kc, n_apl) APL->KC counts
    f_kc: float
    gamma_apl: float    # 1.0 measured inhibition; 0.0 = knockout ablation

    @classmethod
    def from_graph(cls, g: MBGraph, f_kc: float, gamma_apl: float) -> "MBCircuit":
        return cls(
            W_pk=g.pn_kc.dense(),
            M_km=g.kc_mbon.dense(),
            w_k2a=g.kc_apl.dense(),
            w_a2k=g.apl_kc.dense().T,   # stored (n_kc, n_apl)
            f_kc=f_kc, gamma_apl=gamma_apl)

    @property
    def n_kc(self) -> int:
        return self.W_pk.shape[1]

    @property
    def n_mbon(self) -> int:
        return self.M_km.shape[1]

    @property
    def k_kc(self) -> int:
        return max(1, int(round(self.f_kc * self.n_kc)))

    def kc_activate(self, x_pn: np.ndarray) -> np.ndarray:
        """(batch, n_pn) 0/1 -> (batch, n_kc) boolean KC codes."""
        h = x_pn.astype(np.float64) @ self.W_pk
        if self.gamma_apl == 0.0 or self.w_k2a.size == 0:
            return topk_rows(h, self.k_kc)
        y0 = topk_rows(h, self.k_kc)
        a = y0.astype(np.float64) @ self.w_k2a          # (batch, n_apl)
        inh = a @ self.w_a2k.T                          # (batch, n_kc)
        mean_inh = inh.mean(axis=1, keepdims=True)
        mean_h = h.mean(axis=1, keepdims=True)
        gain = np.where(mean_inh > 0.0,
                        self.gamma_apl * mean_h / np.maximum(mean_inh, 1e-30),
                        0.0)
        return topk_rows(h - gain * inh, self.k_kc)

    def init_plastic(self) -> np.ndarray:
        """Per-connection plastic weight, w_max where a measured synapse exists."""
        return np.where(self.M_km > 0.0, RULE_PARAMS.w_max, 0.0)

    def write(self, W_plastic: np.ndarray, y_kc: np.ndarray,
              z_mbon_idx: np.ndarray) -> None:
        """One streamed item: depress measured synapses (active KC x DAN-gated
        MBON) by the closed-form mb_rule amount, in place."""
        rows = np.nonzero(y_kc)[0]
        blk = W_plastic[np.ix_(rows, z_mbon_idx)]
        np.clip(blk - DELTA_W, RULE_PARAMS.beta_floor, RULE_PARAMS.w_max, out=blk)
        W_plastic[np.ix_(rows, z_mbon_idx)] = blk

    def readout_suppression(self, y_kc: np.ndarray,
                            W_plastic: np.ndarray) -> np.ndarray:
        """(batch, n_kc) bool -> (batch, n_mbon) fractional suppression."""
        y = y_kc.astype(np.float64)
        m = y @ (self.M_km * W_plastic)
        m0 = RULE_PARAMS.w_max * (y @ self.M_km)
        with np.errstate(invalid="ignore", divide="ignore"):
            s = 1.0 - m / m0
        return np.where(m0 > 0.0, s, 0.0)


__all__ = ["DELTA_W", "MBCircuit", "RULE_PARAMS", "T_WRITE_STEPS",
           "closed_form_delta", "topk_rows"]
