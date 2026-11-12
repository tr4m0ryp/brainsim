"""CompositeConfig: the ONE source of truth for the full-scale composite
runtime (C7/C8). Every neuron count and in-degree either traces to the
design doc (the MemBrain connectome design notes, F-entry cited on the
field) or is flagged as an engineering choice there. Batteries parameterize
exclusively through this dataclass. Full-scale totals at the defaults:
~1.78M neurons (F21 census + T8 twin; C8 counts this as "order 2M"),
~2.26e10 synapses (inside C8's 2-5e10). Interneuron pools (~11%, F21) are
not explicit populations: the top-f readout + mean-field pool stand in;
explicit interneurons are a fidelity upgrade slot.

Gain policy (engineering, frozen before any science run): drive currents
and plastic weight ceilings DERIVE from the scale via one formula (target
current / (expected active in-degree * assumed rate * synaptic accumulation
factor)), so one config class yields working dynamics at smoke and full
scale; the targets (current_target_*, assumed_rate) are the frozen part.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

from brainsim.engine.lif import LIFParams
from brainsim.rules.allocation import AllocationParams
from brainsim.rules.btsp import BTSPParams
from brainsim.rules.cascade import CascadeParams
from brainsim.rules.mb_rule import MBRuleParams
from brainsim.rules.replay import ReplayParams

STAGES = ("input", "separation", "formation", "output", "relay", "storage")

@dataclass(frozen=True)
class CompositeConfig:
    # --- stage neuron counts (full-scale defaults, per-field provenance)
    n_input: int = 24_000  # EC-II MEC (F21)
    n_separation: int = 300_000  # DG granule, mid of 240k-350k (F21)
    n_formation: int = 110_000  # CA3 pyramidal (F21)
    n_output: int = 311_500  # CA1 pyr, rat model value, mouse abs. unconfirmed (F21)
    n_relay: int = 30_000  # engineering: "thin" thalamus-analog (T10), ~10% of output
    n_storage: int = 1_000_000  # H01 twin matched to formation output (T8)

    # --- in-degrees (per-field provenance)
    k_sep_in: int = 4_000  # engineering, inside F21's loop-total envelope
    # (~1.2-1.5e10 formation-loop synapses); per-cell perforant in-degree unpinned
    k_form_in: int = 50  # mossy-fiber convergence per CA3 pyr (F21)
    # Direct input->formation path (EC-II -> CA3 perforant projection):
    # the RETRIEVAL pathway of the Treves-Rolls CA3 architecture (mossy
    # detonators encode, the direct path cues retrieval). Without it, cue
    # degradation is SQUARED through the sparse mossy map and recall dies
    # (measured at smoke). In-degree engineering choice (design doc does
    # not pin it; order from Bezaire-class CA3 input budgets).
    k_form_direct: int = 4_000
    k_form_rec: int = 11_000  # CA3 recurrent synapses/cell (F21, Ishizuka)
    k_out_in: int = 30_000  # CA1 ~30k exc synapses (F21, Megias), assigned
    # wholly to formation->output as the dominant afferent (flagged)
    k_relay_in: int = 100  # engineering: thin relay coupling (T10)
    k_store_in: int = 100  # engineering: thin relay coupling (T10)
    k_store_rec: int = 10_000  # engineering: sized so the composite total
    # lands inside C8's 2-5e10; H01 supplies twin STATISTICS, not in-degree

    # --- H01 twin multi-contact tail (F20/F28)
    multi_contact_frac: float = 0.001
    multi_contact_tail: int = 50

    # --- coding levels (design default f~0.03 at scale; the smoke preset
    # raises f like composite.pipeline so tiny stages stay robust)
    f_separation: float = 0.03
    f_formation: float = 0.03
    f_output: float = 0.03
    f_relay: float = 0.03
    f_storage: float = 0.03

    # --- static weight statistics (T2: lognormal, 26-state ceiling per F5)
    weight_mu: float = -1.0
    weight_sigma: float = 0.5
    n_weight_states: int = 26
    # Homeostatic per-post row normalization on built static edges
    # (engineering choice, see build._generate_chunk): keeps the top-f
    # readout driven by WHICH pres are active instead of row-mass hubs.
    # The un-normalized variant is a battery ablation arm.
    row_normalize: bool = True

    # --- LIF core (T1, Shiu parameterisation via engine.lif defaults)
    lif: LIFParams = field(default_factory=LIFParams)

    # --- phase windows and drives (engineering; currents on the LIF scale
    # where v_th - v_rest = 7)
    write_steps: int = 30
    recall_steps: int = 30
    drive_gain: float = 4.0  # pattern current per active input neuron
    clamp_gain: float = 4.0  # current clamping a source stage's code cells
    exc_drive_gain: float = 0.5  # allocation-excitability bonus current
    # Hasselmo encoding mode (ACh suppression of recurrents during encoding;
    # adopted mechanism, engineering-scaled): write windows scale plastic
    # recurrent weights by rho so prior attractors cannot capture the new
    # code. Power of two => suppress/restore is bit-exact in fp16/fp32,
    # sized to keep rho * w_max * co-active in-degree under LIF threshold.
    rho_encoding: float = 0.0625
    # Rest interval between items: plasticity traces (BTSP et/is, MB e_kc)
    # decay by this wall time, so items are distinct memories, not one
    # episode (5 s puts BTSP cross-item coincidence at ~0.003).
    inter_item_seconds: float = 5.0
    # Allocation interval: distinct memories sit DAYS apart on the Cai-2016
    # clock (co-allocation 5 h, segregation 7 d; tau 36 h): 5 days rotates
    # the winner set per item; co-allocation batteries shrink it.
    alloc_interval_seconds: float = 5.0 * 86_400.0
    normalize_drive: bool = True  # divisive gain control on window drives
    # (pipeline._sparsify's scale invariance): per-cell drive scales by
    # write-reference/active-count, capped at norm_cap x nominal.
    norm_cap: float = 4.0
    # Recurrent-image completion iterations (pipeline's relax, iterated =
    # associative settling; one pass amplifies the seed's own-assembly
    # fraction ~3-4x, measured at smoke).
    completion_iters: int = 3
    # Retrieval evidence floor (engineering): a cell joins a completed code
    # only above this fraction of a full member's expected image current
    # (k*f*w_full); iterated completion otherwise confabulates the nearest
    # stored item for ANY cue. Sits between a control/eroded image
    # (~0.07-0.2 of full) and a true degraded cue's (~0.3-0.5), so novel
    # cues and eroded assemblies image to EMPTY.
    image_evidence_frac: float = 0.25

    # --- derived-gain targets (engineering constants; see module docstring)
    current_target_ff: float = 14.0  # feedforward stage drive target
    current_target_direct: float = 10.0  # direct path: sub-dominant to the
    # mossy detonators at write (selection stays mossy-variance-driven, so
    # encoding keeps decorrelating), correlation carrier at retrieval
    # Recurrent weight-ceiling targets (via btsp_w_max()/store_w_unit()):
    # delivered current of one full assembly at the assumed rate; sets the
    # image-readout dynamic range and, in the optional attractor mode, the
    # sustain margin (formation sized so a ~1/3-complete assembly recruits
    # through merged-state inhibition; measured at smoke).
    current_target_formation: float = 72.0
    current_target_storage: float = 40.0
    assumed_rate: float = 0.1  # spikes/step of a driven cell, calibration pt
    # Mean-field interneuron pool for run_window(inhibit=...), the OPTIONAL
    # spiking-attractor retrieval mode (default readout is the image,
    # codes.recurrent_image_code): one global unit per stage (F21's ~11%
    # census as population feedback). Current = max(floor, at_target *
    # A/A_target), A = smoothed population spike count, A_target = one
    # code-sized assembly at the assumed rate; engages grace steps after
    # drive release (referees assemblies, not the drive echo; measured).
    inh_current_at_target: float = 10.0
    inh_floor: float = 12.0
    inh_grace_steps: int = 8

    # --- plasticity (T4 rule set; params imported from the rules modules).
    # BTSP: tau_et/tau_is stay at the Milstein-2021 measurements;
    # k_plus/k_minus/dt are application engineering (one write integrates
    # steps*dt = 1 s of plateau, binding to ~95% of w_max: F2's one-shot);
    # w_max is the derived delivered-current gain (btsp_w_max()).
    btsp_write_steps: int = 10
    btsp_dt_ms: float = 100.0
    btsp_k_plus: float = 3.0
    btsp_k_minus: float = 0.1
    cascade: CascadeParams = field(default_factory=CascadeParams)  # F3 defaults
    allocation: AllocationParams = field(default_factory=AllocationParams)  # F6
    alloc_k: int = 10  # engram cells allocated per write event
    mb: MBRuleParams = field(default_factory=MBRuleParams)  # F1 alpha/tau/dt;
    # w_max/beta_floor overridden by the derived relay scale (mb_params())
    mb_write_steps: int = 5
    mb_gate_scale: float = 1.0  # salience -> dan_signal (F8 write-depth gate)
    mb_floor_frac: float = 0.4  # beta_floor as fraction of derived mb w_max
    replay: ReplayParams = field(default_factory=ReplayParams)  # F11 defaults
    replay_budget: int = 0  # events per storage phase; 0 mirrors the numpy
    # smoke cycle, which registers items but never runs an offline phase

    # --- storage write + turnover (pipeline semantics, in derived store
    # weight units; eta_store == w_max_store: one write saturates -- the
    # numpy pipeline's scale-invariant sparsify tolerates graded eta)
    eta_store: float = 3.0
    w_max_store: float = 3.0
    leak_rate: float = 0.05  # pipeline.PipelineConfig.leak_rate
    turnover_noise_sigma: float = 0.05  # pipeline smoke value, in w units

    # --- engine layout (F24)
    fp16_static: bool = False
    fp16_plastic: bool = False
    delay_steps: int = 1
    chunk_synapses: int = 250_000_000  # build-time COO chunk (host transient)
    device_capacity_bytes: int | None = None  # None: autodetect (cuda/host)
    seed: int = 0

    # ---- derived layout ------------------------------------------------------
    def stage_sizes(self) -> dict[str, int]:
        return {
            "input": self.n_input,
            "separation": self.n_separation,
            "formation": self.n_formation,
            "output": self.n_output,
            "relay": self.n_relay,
            "storage": self.n_storage,
        }

    def stage_ranges(self) -> dict[str, tuple[int, int]]:
        """Contiguous global index ranges, stage order fixed by STAGES."""
        sizes = self.stage_sizes()
        ranges, lo = {}, 0
        for name in STAGES:
            ranges[name] = (lo, lo + sizes[name])
            lo += sizes[name]
        return ranges

    @property
    def n_global(self) -> int:
        return sum(self.stage_sizes().values())

    def coding_level(self, stage: str) -> float:
        return float(getattr(self, f"f_{stage}"))

    def code_size(self, stage: str) -> int:
        return max(1, int(round(self.coding_level(stage) * self.stage_sizes()[stage])))

    # ---- derived gains -------------------------------------------------------
    @property
    def tau_accum(self) -> float:
        """Steady-state multiplier of a constant per-step current under the
        LIF synaptic decay: I_ss = c / (1 - exp(-dt/tau_syn))."""
        return 1.0 / (1.0 - math.exp(-self.lif.dt / self.lif.tau_syn))

    @property
    def lognormal_mean(self) -> float:
        return math.exp(self.weight_mu + 0.5 * self.weight_sigma**2)

    def active_fraction(self, stage: str) -> float:
        """Expected active fraction of a stage when it acts as a source:
        bipolar input patterns drive ~half the input stage; every other stage
        is clamped/read at its coding level."""
        return 0.5 if stage == "input" else self.coding_level(stage)

    def static_scale(self, k: int, pre_stage: str,
                     target: float | None = None) -> float:
        """Weight scale for a static edge: target current / (expected active
        in-degree * assumed rate * accumulation * mean lognormal draw)."""
        active_k = max(1e-9, k * self.active_fraction(pre_stage))
        target = self.current_target_ff if target is None else target
        return target / (
            active_k * self.assumed_rate * self.tau_accum * self.lognormal_mean
        )

    def btsp_w_max(self) -> float:
        asm = max(1e-9, self.k_form_rec * self.f_formation)
        return self.current_target_formation / (asm * self.assumed_rate * self.tau_accum)

    def store_w_unit(self) -> float:
        """One pipeline weight unit on the storage stage: a fully written
        synapse set (w = w_max_store units) delivers the storage target."""
        asm = max(1e-9, self.k_store_rec * self.f_storage)
        return self.current_target_storage / (
            asm * self.assumed_rate * self.tau_accum * self.w_max_store
        )

    def mb_w_max(self) -> float:
        asm = max(1e-9, self.k_relay_in * self.f_output)
        return self.current_target_ff / (asm * self.assumed_rate * self.tau_accum)

    def btsp_params(self) -> BTSPParams:
        return BTSPParams(
            k_plus=self.btsp_k_plus, k_minus=self.btsp_k_minus,
            w_max=self.btsp_w_max(), dt=self.btsp_dt_ms,
        )

    def mb_params(self) -> MBRuleParams:
        """F1 rule form with the composite-scale ceiling: w_max/beta_floor
        are the derived relay drive scale (the 0.05 KC->MBON value is the
        fly-track magnitude; engineering choice, flagged)."""
        w_max = self.mb_w_max()
        return MBRuleParams(
            alpha=self.mb.alpha, tau_w=self.mb.tau_w, tau_e=self.mb.tau_e,
            w_max=w_max, beta_floor=self.mb_floor_frac * w_max, dt=self.mb.dt,
        )

    # ---- presets ---------------------------------------------------------------
    @classmethod
    def smoke(cls, seed: int = 0) -> "CompositeConfig":
        """~2k-neuron CPU preset: code-check scale, never science. f = 0.1
        for the same reason composite.pipeline uses it at smoke scale."""
        return cls(
            n_input=200, n_separation=400, n_formation=300, n_output=200,
            n_relay=100, n_storage=800,
            k_sep_in=40, k_form_in=20, k_form_direct=40, k_form_rec=30,
            k_out_in=40, k_relay_in=30, k_store_in=30, k_store_rec=40,
            multi_contact_frac=0.01, multi_contact_tail=5,
            f_separation=0.1, f_formation=0.1, f_output=0.1, f_relay=0.1,
            f_storage=0.0625,  # code 50 of 800: keeps 5 smoke items from
            # tiling half the stage (indicator overlaps have no sign
            # cancellation; full scale has this headroom by construction)
            chunk_synapses=1_000_000,
            seed=seed,
        )

    @classmethod
    def full_scale(cls, seed: int = 0) -> "CompositeConfig":
        """Design-doc census (per-field provenance comments above); fp16
        weights per F24, fp32 accumulation everywhere (rules guarantee it)."""
        return cls(fp16_static=True, fp16_plastic=True, seed=seed)

    def with_seed(self, seed: int) -> "CompositeConfig":
        return replace(self, seed=seed)

__all__ = ["CompositeConfig", "STAGES"]
