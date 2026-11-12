"""Whole-fly engine gate: the SAME pre-registered directional checks as
scripts/gates/fly_gate_shiu.py, with the FlyWire v783 connectome running on
the brainsim torch engine (brainsim.engine.torch_core) instead of Shiu et
al. 2024's Brian2 code.

The reference gate executes Shiu's own model
(results/fly-gate/shiu-brian2/gate.json, gate_passed=true). This gate
re-certifies the same biology on brainsim: (a) sugar-GRN drive lifts MN9 above the silent
baseline, (b) MN9 rises with drive frequency, (c) sugar beats a random
control set. Checks are DIRECTIONAL by pre-registration; exact rates are not
the criterion (the engines differ in synapse handling, see divergences).

Parameter mapping (Shiu Brian2 -> brainsim LIFParams)
-----------------------------------------------------
Shiu: dv/dt = (v_0 - v + g)/t_mbr; dg/dt = -g/tau; on_pre g += w after a
1.8 ms delay; reset v=v_rst and g=0; refractory 2.2 ms with v and g frozen;
threshold v > v_th; Brian2 exact 'linear' method at dt=0.1 ms.
Ours: v += dt*(-(v - v_rest) + R*I)/tau_mem (forward Euler); I *=
exp(-dt/tau_syn) each step; ring-buffer delayed delivery adds w to I; reset
v=v_reset with I kept; refractory clamps v to v_reset while I keeps decaying
and accumulating; spike at v >= v_th.

With R = 1, our I is Shiu's g in mV: identical jump-and-exponential-decay
law, identical signed weight w = 0.275 mV x 'Excitatory x Connectivity'.

  Shiu constant          -> LIFParams        value   note
  v_0 = v_rst = -52 mV   -> v_rest, v_reset  -52.0   identical
  v_th = -45 mV          -> v_th             -45.0   identical
  t_mbr = 20 ms          -> tau_mem           20.0   identical
  tau = 5 ms             -> tau_syn            5.0   same decay law, I === g
  t_rfc = 2.2 ms         -> refractory         2.2   -> 2 steps = 2.0 ms
  t_dly = 1.8 ms         -> delay_steps          2   -> 2.0 ms at dt=1.0
  w_syn = 0.275 mV       -> CSR weight     signed count * 0.275, fp32
  Brian dt = 0.1 exact   -> dt                 1.0   Euler, 1000 steps/trial
  r_poi                  -> Bernoulli(r*dt) per driven neuron per step
  w_syn*f_poi = 68.75 mV on v -> I_EV = 275 mV on I (= 68.75*tau_mem/tau_syn)

dt = 1.0 ms over 0.6 ms: refractory and delay both land on 2 steps = 2.0 ms
(vs Shiu 2.2/1.8 ms), a trial is exactly 1000 steps, and the Euler leak
factor dt/tau_mem = 0.05 is deep inside stability; dt=0.6 maps the delay
exactly (3 steps) but distorts refractory to 2.4 ms and costs 67% more steps.

Poisson drive: Shiu's PoissonInput adds w_syn*f_poi = 68.75 mV directly to v
(instantly suprathreshold, the rest-to-threshold gap is 7 mV) and zeroes the
driven neuron's refractory. Our engine has no voltage-jump pathway, so each
Bernoulli event injects I_EV = w_syn*f_poi*tau_mem/tau_syn = 275 mV into I:
the event's total membrane depolarization integral matches Shiu's exactly
(integral (v - v_rest) dt = 68.75*tau_mem = I_EV*tau_syn = 1375 mV*ms) and
the first-step dv = dt*I_EV/tau_mem = 13.75 mV clears the 7 mV gap in one
step, mirroring Shiu's instant fire. The "g" mode is the alternative
"g-dynamics" reading (event adds 68.75 mV to I, exactly like a network
synapse of weight w_syn*f_poi), the pre-authorized fallback. The charge
mode failed the dose-response check (200 Hz tipped the network into a
runaway state; results/fly-gate/brainsim/gate-attempt-charge-iev275.json)
and the g mode passed (results/fly-gate/brainsim/gate.json), so g is the
default.

Known divergences (why the gate is directional, not rate-matched):
 1. Spike reset keeps I (Shiu zeroes g): residual drive, bursty driven
    neurons, effective GRN rate above r_poi.
 2. Refractory 2.0 ms with v clamped and I decaying/accumulating (Shiu:
    2.2 ms with v and g frozen; on_pre still accumulates g there).
 3. Driven neurons keep the 2-step refractory (Shiu sets rfc=0 for them).
 4. Bernoulli(r*dt) allows at most one event per step (Poisson allows more);
    at r*dt <= 0.2 the shortfall is <= 2 percent of events.
 5. Forward Euler at dt=1.0 ms vs Brian2 exact linear at 0.1 ms.
All five shift absolute rates; none reverses the pre-registered directions.

Data path: Connectivity_783.parquet (shipped with Shiu et al.'s model
repository, fetched by scripts/data/fetch_shiu_model.sh) is converted once,
in a subprocess with pyarrow, into int32/fp32 .npy under
<data>/external/fly_v783_npy/ (regenerable).
"""
from __future__ import annotations

# Shiu et al. 2024 constants (model.py of their repository, MIT licence).
V_REST, V_TH, TAU_MEM, TAU_SYN = -52.0, -45.0, 20.0, 5.0
T_RFC, T_DLY, W_SYN, F_POI = 2.2, 1.8, 0.275, 250.0
FREQS = (50, 100, 200)
TRIAL_MS = 1000.0
RUNTIME_BUDGET_S = 90 * 60

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


def param_mapping(dt: float, delay_steps: int, i_ev: float, mode: str) -> dict:
    """The Shiu -> brainsim parameter table recorded in every gate report."""
    return {
        "v_rest=v_reset": "-52 mV (Shiu v_0 = v_rst)",
        "v_th": "-45 mV (Shiu)",
        "tau_mem": "20 ms (Shiu t_mbr)",
        "tau_syn": "5 ms (Shiu tau; our I is Shiu's g via R=1)",
        "refractory": f"2.2 ms -> {max(1, round(T_RFC / dt))} steps at dt={dt}",
        "delay": f"1.8 ms -> delay_steps={delay_steps} ({delay_steps * dt} ms)",
        "dt": f"{dt} ms forward Euler (Shiu: Brian2 exact linear at 0.1 ms)",
        "synapse_weight": "0.275 mV x signed 'Excitatory x Connectivity', fp32",
        "poisson_event": (f"Bernoulli(r*dt) per driven neuron; event injects "
                          f"{i_ev:.2f} mV into I ({mode} interpretation; "
                          "charge = w_syn*f_poi*tau_mem/tau_syn matches Shiu's "
                          "+68.75 mV v-jump depolarization integral)"),
        "divergences": [
            "spike reset keeps I (Shiu zeroes g)",
            "refractory clamps v while I decays/accumulates (Shiu freezes v,g)",
            "driven neurons keep refractory (Shiu sets rfc=0 for them)",
            "Bernoulli(r*dt) vs Poisson (<= 2 percent event shortfall)",
            f"Euler dt={dt} vs Brian2 exact linear dt=0.1",
        ],
    }


def event_current(mode: str) -> float:
    """Poisson event current in mV: 'charge' = 275.0, 'g' = 68.75."""
    if mode == "charge":
        return W_SYN * F_POI * TAU_MEM / TAU_SYN
    if mode == "g":
        return W_SYN * F_POI
    raise ValueError(f"unknown event mode {mode!r}")


__all__ = ["F_POI", "FREQS", "MN9", "NEU_SUGAR", "RUNTIME_BUDGET_S", "TAU_MEM",
           "TAU_SYN", "TRIAL_MS", "T_DLY", "T_RFC", "V_REST", "V_TH", "W_SYN",
           "event_current", "param_mapping"]
