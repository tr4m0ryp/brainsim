"""Battery H2 configuration: every constant carries a provenance tag so the
results notes can attest which numbers trace to the H01 artifact, the frozen
contract, or the Formal Model, and which are engineering choices (honesty
rail: no untraceable constants).

Provenance vocabulary: "contract" (battery-h2-permanence-frontier.md),
"artifact" (data/connectomes/h01/stats), "formal-model"
(MemBrain connectome design notes, Formal Model), "design-doc"
(T2/T8/F3/F4/F5), "engineering" (our choice, justified in the note).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

# Contract-frozen grids (battery-h2-permanence-frontier.md, Arms).
K_GRID = (2, 5, 10, 20, 50)
P_GRID = (0.01, 0.05, 0.1, 0.2)
ARM_NAMES = ("redundancy", "uniform", "cascade_only", "replay", "combined")

# Root entropy for every SeedSequence in the battery. Seed derivation is
# (purpose, seed, arm, k, p)-keyed and never depends on shard layout, so any
# arm-shard partition reproduces the identical numbers (amendment rule 5).
SEED_ROOT = 0x48325F42  # "H2_B"


@dataclass(frozen=True)
class H2Config:
    # -- scale ---------------------------------------------------------------
    scale: str = "full"          # tiny | smoke | full
    n: int = 1_000_000           # design-doc T8: twin sized to formation output ~1M
    k_in: int = 1000             # ENGINEERING FALLBACK: the H01 stats artifact has no
    #                              neuron census, so mean in-degree is not derivable
    #                              from it; 1000 is chosen so n*k_in*mean_k >= 1e9,
    #                              the frozen validate.py floor. Flagged in notes.
    f: float = 0.03              # design-doc coding level (F: vision default)
    # -- twin ----------------------------------------------------------------
    n_states: int = 26           # design-doc F5 (Bartol 26-state ceiling)
    tail_trim_k: int = 50        # artifact/F28: paper-validated multi-contact regime
    # -- cascade (F3 defaults, rules/cascade.py) ------------------------------
    cascade_n_levels: int = 8
    cascade_q1: float = 0.5
    cascade_x: float = 0.5
    # -- store dynamics -------------------------------------------------------
    sigma: float = 0.03          # formal-model mapping: Loewenstein 2011 daily
    #                              multiplicative spine volatility ~0.15/day scaled by
    #                              sqrt(A_twin/A_ctx) = sqrt(15/300) ~ 0.22 to equalise
    #                              pooled-SNR trajectories; 1 step == 1 biological day.
    theta_leak: float = 0.02     # engineering: unprotected deposit half-life ~35 steps
    #                              (~1 biological month unconsolidated trace).
    eta_rel: float = 3.0         # engineering: deposit = eta_rel * mean substrate w;
    #                              initial-recall headroom over the heavy-tailed
    #                              (sigma_ln ~ 1.1, artifact) substrate drive.
    recall_iters: int = 2        # formal-model RECALL is clamped-cue relaxation;
    #                              iteration count is the engineering discretisation.
    eta_dep_frac: float = 0.5    # engineering: heterosynaptic depression fraction.
    interference_per_step: int = 1  # contract: "continual interference writes at
    #                              fixed rate"; the rate value is an engineering choice.
    # -- protocol --------------------------------------------------------------
    cue_frac: float = 0.5        # engineering (composite-pipeline idiom, 50% cue)
    overlap_threshold: float = 0.9   # contract
    precision_threshold: float = 0.9  # contract
    t_max: int = 2048            # engineering: covers predicted uncensored T_half for
    #                              k <= 20 (formal-model sqrt(k)); k=50 may censor,
    #                              censoring is flagged, never hidden.
    ramp_l0: int = 16
    ramp_ratio: float = 1.25     # engineering: capacity grid resolution 0.22 in ln L
    m_probe: int = 128           # engineering: probe sample size (binomial se ~2.7%)
    lref_frac: float = 0.5       # engineering: retention load = 0.5 * C_cascade_only
    lref_floor: int = 4
    # -- budget arms -----------------------------------------------------------
    replay_lr: float = 0.05      # design-doc F4 (Kali-Dayan eta_ctx multiplier)
    replay_spend_horizon: float = 0.5  # engineering: spend allowance by 0.5*t_max
    combined_alpha: float = 0.5  # engineering: structural share of the combined budget
    max_contacts: int = 8_000_000_000  # engineering guard: install cap; runs that hit
    #                              it are marked resource_capped in results.
    # -- grid -------------------------------------------------------------------
    k_grid: tuple = K_GRID       # contract
    p_grid: tuple = P_GRID       # contract
    # -- engine -------------------------------------------------------------------
    engine: str = "numpy"        # numpy | torch
    torch_device: str = "auto"
    weight_dtype: str = "float32"  # torch may run fp16 (orchestrator memory plan)

    def config_hash(self) -> str:
        """Hash of every number that affects the science; shard layout, output
        paths and timings are deliberately excluded (merge invariance)."""
        payload = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


def full_config(engine: str = "numpy", **kw) -> H2Config:
    return H2Config(scale="full", engine=engine, **kw)


def smoke_config(engine: str = "numpy", **kw) -> H2Config:
    """Code-path verification only. The contract forbids producing any T_half
    or frontier point at smoke scale; the runner enforces print-only."""
    kw.setdefault("n", 2000)
    kw.setdefault("k_in", 200)   # denser than full: keeps cue->target pooling
    kw.setdefault("f", 0.1)      # nonzero at tiny n (composite-pipeline idiom)
    kw.setdefault("t_max", 64)
    kw.setdefault("ramp_l0", 8)
    kw.setdefault("m_probe", 32)
    kw.setdefault("k_grid", (5, 20))
    kw.setdefault("p_grid", (0.1,))
    return H2Config(scale="smoke", engine=engine, **kw)


def tiny_config(engine: str = "numpy", **kw) -> H2Config:
    """Test-suite scale; smaller than smoke, same code paths."""
    kw.setdefault("n", 400)
    kw.setdefault("k_in", 120)
    kw.setdefault("f", 0.15)
    kw.setdefault("t_max", 16)
    kw.setdefault("ramp_l0", 4)
    kw.setdefault("m_probe", 16)
    kw.setdefault("k_grid", (2,))
    kw.setdefault("p_grid", (0.2,))
    return H2Config(scale="tiny", engine=engine, **kw)


# Provenance table emitted verbatim into results notes (honesty rail).
PROVENANCE_NOTES = [
    "k_in: ENGINEERING FALLBACK. The F28 artifact aggregates 151.5M synapses "
    "by (pre,post) pair with no neuron census, so measured mean in-degree is "
    "unavailable; k_in=1000 at n=1e6 meets the frozen >=1e9 synapse floor.",
    "twin strength prior: lognormal sigma fitted to the artifact's type-2 "
    "(excitatory) log-volume histogram (volume as strength proxy, F5 "
    "Bartol/Loewenstein); scale normalised to mean 1 (protocol is scale-free, "
    "only the CV is load-bearing).",
    "contact-count pmf: artifact multi_contact_distribution trimmed to k<=50 "
    "per F28 (paper-validated regime; 309-outlier excluded, logged not used).",
    "sigma=0.03/step: Loewenstein 2011 multiplicative daily volatility 0.15 "
    "scaled by sqrt(A_twin/A_ctx = 15/300); 1 store step == 1 biological day "
    "== ~8.64e7 native 1 ms engine steps (accelerated-turnover compression).",
    "turnover sigma is activity-independent (Dvorkin & Ziv 2016) and is NOT "
    "reduced by cascade depth; cascade resistance gates leak and interference "
    "deposits only. Deviation from composite/pipeline.py smoke idiom (which "
    "scales noise too) is deliberate: fully cascade-immune weights would make "
    "the frontier question vacuous.",
    "shift metric: per-seed log frontier-area ratio ln((C*T)/(C_u*T_u)) vs "
    "the matched-R uniform arm; noise band = half-width of the frozen "
    "verdicts.bootstrap_ci over per-seed shifts.",
    "T_half: promoted-item class, log-interpolated survival crossing of "
    "0.5*S(0); censored at t_max is flagged and used as a lower bound.",
    "L_ref = max(4, round(0.5*C_cascade_only(seed,p))): retention load rule, "
    "fixed a priori; couples arms to the baseline capacity of the same seed.",
    "budget R(k,p,item) = sum over the item's touched connections of "
    "max(0, k - k0) * mean substrate contact weight; computed from frozen "
    "substrate weights so all arms price the identical schedule.",
    "capacity phase has no turnover steps (load itself is the interference); "
    "retention phase carries turnover + continual interference writes.",
    "twin gate: contact pmf / multi-contact rate / mean_k are generation-"
    "fidelity checks; the strength check is a 2-moment fit tested against the "
    "full artifact histogram shape (held-out shape degrees of freedom). True "
    "held-out motif statistics (reciprocity, triads) are not derivable from "
    "the aggregate artifact; flagged, not silently skipped.",
    "n_synapses in results.json = per-arm twin substrate contact count; "
    "full-scale arms run as concurrent single-GPU processes each at this "
    "scale (orchestrator amendment 2026-08-09).",
]
