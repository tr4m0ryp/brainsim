"""Battery H3 configuration. Every constant carries a provenance tag
(honesty rail: no untraceable constants). Vocabulary: "contract"
(battery-h3-composite.md), "driver-spec" (driver-spec-h3.md), "formal-model"
(MemBrain connectome design notes, Formal Model), "design-doc"
(F/T entries), "h2" (shared with battery H2, imported not redefined),
"engineering" (our choice, justified in the note).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

from brainsim.composite.full.config import CompositeConfig

# Root entropy for every SeedSequence in the battery; the (work item, seed)
# -> RNG mapping never depends on shard layout (H2 amendment rule 5 idiom).
SEED_ROOT = 0x48335F42  # "H3_B"

# Contract-frozen names (battery-h3-composite.md Arms; driver-spec-h3).
STAGE_ALONE = ("separation_alone", "formation_alone", "relay_alone",
               "storage_alone")
ABLATIONS = ("btsp_to_stdp", "cascade_n1", "cascade_n8", "cascade_n16",
             "replay_off", "replay_uniform", "allocation_off",
             "relay_removed", "boundary_poisson", "row_normalize_off")
CONTEXT_ROWS = ("ctx_ecdgca3", "ctx_kassab")  # contract arm 5

# lambda sweep factors x lambda*_pred (driver-spec: "grid around the Formal
# Model's predicted lambda*"). Log-spaced with wide upper wing: the finite
# retention horizon and integer replay-event quantisation bound the useful
# range from below (engineering; recorded in PROVENANCE_NOTES). The headline
# composite row for the verdict is factor 1.0, frozen before any data.
LAMBDA_FACTORS_FULL = (0.25, 1.0, 4.0, 16.0, 64.0)
HEADLINE_LAMBDA = 1.0

# Tsodyks-Feigelman capacity prefactor for lambda*_pred (formal-model #2
# gives the SCALING P_max ~ c*k/(f ln 1/f); c itself is unmeasured before
# the run -- engineering point value, flagged; the overlay figure carries
# the measured correction).
C_TSODYKS = 0.1


@dataclass(frozen=True)
class H3Config:
    scale: str = "full"                    # tiny | smoke | full
    engine: str = "torch"                  # composite runtime is torch-only
    device: str = "auto"
    # -- episodic stream (contract Protocol) ---------------------------------
    n_items: int = 96          # engineering: probe-population size; every
    #                            item is probed, retained-at-T is a count
    n_morph_pairs: int = 8     # engineering: separation-probe pairs, counted
    #                            inside n_items (last 2*pairs items)
    morph_keep: tuple = (0.6, 0.8)  # engineering: fraction of coordinates
    #                            shared within a pair, cycled across pairs
    salience_lo: float = 0.5   # composite-pipeline idiom (U(0.5, 2.0))
    salience_hi: float = 2.0
    theta_protect: float = 1.7  # engineering: salience protection gate,
    #                            top ~20% of U(0.5,2) earn cascade deepen
    # -- probes (contract: partial-cue completion, precision >= 0.9) ---------
    drop_fraction: float = 0.5   # composite-pipeline idiom (50% cue)
    overlap_threshold: float = 0.9  # contract
    m_probe: int = 48            # engineering: probe sample per epoch
    # -- interference load sweep (contract: "interference load L swept") -----
    interference_grid: tuple = (0, 112, 224, 448)  # engineering: cumulative
    #                            interfering writes = {0, 1, 2, 4} x stream
    # -- retention under turnover (contract: log-spaced probes) --------------
    retention_days: int = 32     # engineering: 1 day per turnover step (h2)
    interference_per_day: int = 4  # engineering: continual interference rate
    # -- lambda sweep ---------------------------------------------------------
    lambda_factors: tuple = LAMBDA_FACTORS_FULL  # driver-spec
    # -- screen (lean protocol; same code path, smaller constants) -----------
    screen_n_items: int = 48
    screen_pairs: int = 4
    screen_grid: tuple = (0, 96)
    screen_days: int = 8
    screen_m_probe: int = 24
    ko_pair_budget_factor: int = 2  # driver-spec: pairs <= 2x singles
    top_k_hidden: int = 3           # engineering: follow-up list length
    # -- arm enablement (smoke shrinks these; full runs everything) ----------
    stage_alone_arms: tuple = STAGE_ALONE
    ablation_arms: tuple = ABLATIONS
    context_arms: tuple = CONTEXT_ROWS
    run_bag: bool = True
    bag_groups: int = 8     # driver-spec sharding: bag packs one group/GPU
    ko_singles: tuple = ()  # () = full catalog (screen.SINGLE_KOS)
    ko_pairs_override: int = -1  # -1 = budget rule; >=0 fixes the count
    # -- thin pass-through stage floors (driver-spec: "other stages replaced
    # by pass-through relays"); engineering sizes, budget-asserted ------------
    thin_n: int = 2000
    thin_k: int = 32
    bag_thin_n: int = 64
    bag_thin_k: int = 16

    def config_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


def full_config(engine: str = "torch", **kw) -> H3Config:
    return H3Config(scale="full", engine=engine, **kw)


def smoke_config(engine: str = "torch", **kw) -> H3Config:
    """Code-path verification only (contract: NO retention/capacity number
    may be produced at smoke scale). driver-spec smoke scope: 2 arms + 1
    ablation + mini-screen, print-only."""
    kw.setdefault("n_items", 3)
    kw.setdefault("n_morph_pairs", 1)
    kw.setdefault("morph_keep", (0.8,))
    kw.setdefault("m_probe", 5)
    kw.setdefault("interference_grid", (0, 4))
    kw.setdefault("retention_days", 2)
    kw.setdefault("interference_per_day", 1)
    kw.setdefault("lambda_factors", (1.0, 2.0))
    kw.setdefault("screen_n_items", 3)
    kw.setdefault("screen_pairs", 1)
    kw.setdefault("screen_grid", (0, 2))
    kw.setdefault("screen_days", 1)
    kw.setdefault("screen_m_probe", 3)
    kw.setdefault("stage_alone_arms", ("formation_alone",))
    kw.setdefault("ablation_arms", ("btsp_to_stdp",))
    kw.setdefault("context_arms", ())
    kw.setdefault("run_bag", False)
    kw.setdefault("ko_singles", ("ct:granule", "motif:multi_contact_tail"))
    kw.setdefault("ko_pairs_override", 1)
    kw.setdefault("thin_n", 60)
    kw.setdefault("thin_k", 12)
    return H3Config(scale="smoke", engine=engine, **kw)


def tiny_config(engine: str = "torch", **kw) -> H3Config:
    """Test-suite scale; smaller than smoke, same code paths."""
    kw.setdefault("n_items", 2)
    kw.setdefault("n_morph_pairs", 1)
    kw.setdefault("morph_keep", (0.8,))
    kw.setdefault("m_probe", 4)
    kw.setdefault("interference_grid", (0, 2))
    kw.setdefault("retention_days", 1)
    kw.setdefault("interference_per_day", 1)
    kw.setdefault("lambda_factors", (1.0,))
    kw.setdefault("screen_n_items", 2)
    kw.setdefault("screen_pairs", 1)
    kw.setdefault("screen_grid", (0, 1))
    kw.setdefault("screen_days", 1)
    kw.setdefault("screen_m_probe", 2)
    kw.setdefault("stage_alone_arms", ("formation_alone",))
    kw.setdefault("ablation_arms", ("cascade_n1",))
    kw.setdefault("context_arms", ())
    kw.setdefault("run_bag", True)
    kw.setdefault("bag_groups", 2)
    kw.setdefault("ko_singles", ("ct:granule", "motif:multi_contact_tail"))
    kw.setdefault("ko_pairs_override", 1)
    kw.setdefault("thin_n", 40)
    kw.setdefault("thin_k", 10)
    kw.setdefault("bag_thin_n", 16)
    kw.setdefault("bag_thin_k", 4)
    return H3Config(scale="tiny", engine=engine, **kw)


def base_composite(cfg: H3Config, seed: int) -> CompositeConfig:
    """The composite arm's CompositeConfig at this battery scale. full =
    CompositeConfig.full_scale (design-doc census, driver-spec); smoke/tiny
    are shrunken code-check presets (engineering, never science)."""
    if cfg.scale == "full":
        return CompositeConfig.full_scale(seed=seed)
    if cfg.scale == "smoke":
        return CompositeConfig(
            n_input=120, n_separation=240, n_formation=180, n_output=120,
            n_relay=60, n_storage=400,
            k_sep_in=24, k_form_in=12, k_form_direct=24, k_form_rec=20,
            k_out_in=24, k_relay_in=16, k_store_in=16, k_store_rec=24,
            multi_contact_frac=0.01, multi_contact_tail=4,
            f_separation=0.1, f_formation=0.1, f_output=0.1, f_relay=0.1,
            f_storage=0.1, chunk_synapses=500_000, seed=seed)
    return CompositeConfig(
        n_input=80, n_separation=160, n_formation=120, n_output=80,
        n_relay=40, n_storage=200,
        k_sep_in=20, k_form_in=10, k_form_direct=20, k_form_rec=16,
        k_out_in=20, k_relay_in=12, k_store_in=12, k_store_rec=20,
        multi_contact_frac=0.01, multi_contact_tail=3,
        f_separation=0.1, f_formation=0.1, f_output=0.1, f_relay=0.1,
        f_storage=0.1, chunk_synapses=100_000, seed=seed)


PROVENANCE_NOTES = [
    "matched-resources rule: TOTAL synapse count of every stage-alone and "
    "bag arm equals the composite arm's within 1% (driver-spec); asserted at "
    "arm-construction time and again by tests, never assumed.",
    "stage-alone redistribution: pass-through stages at thin floors "
    "(thin_n/thin_k, engineering); the stage under test scales n and its "
    "dominant in-degree preserving the composite's k/n ratio, then the "
    "dominant k is trimmed for the 1% match (driver-spec 'via the config's "
    "per-stage counts').",
    "bag-of-attractors: N = round(total_budget / (S_form_rec/P_items)) "
    "independent formation-stage copies, i.e. per-attractor synapse size = "
    "composite formation's per-item load (driver-spec formula); items and "
    "interference route round-robin item->module i mod N. Engineering "
    "choice, recorded per driver-spec.",
    "lambda semantics: lambda = expected replay-transfer events per stored "
    "item per retention day; per-day replay budget = fractional accumulator "
    "of lambda * n_registered_items (deterministic, layout-independent). "
    "lambda*_pred = r_f_pred / ln(P) (formal-model #5) with r_f_pred = "
    "interference_per_day / C_form_pred and C_form_pred = C_TSODYKS * "
    "k_form_rec / (f_formation * ln(1/f_formation)) (formal-model #2 "
    "scaling; C_TSODYKS=0.1 engineering point value). Headline verdict row "
    "= factor 1.0, frozen pre-data; other factors are the overlay figure.",
    "accelerated turnover: 1 storage_phase step == 1 biological day == "
    "~8.64e7 native 1 ms engine steps, IDENTICAL to battery H2's mapping "
    "(one compression constant across batteries, driver-spec); per-day "
    "volatility magnitude imported from H2Config.sigma (0.03). Units "
    "bridge, flagged: H2 applies it as multiplicative lognormal volatility "
    "on the twin substrate, the composite runtime as additive turnover "
    "noise in store-weight units at matched per-day magnitude.",
    "cascade_n1 ablation: with the runtime's resistance = depth/n_levels, "
    "n_levels=1 makes every touched synapse fully turnover-resistant -- the "
    "ablation removes the cascade's SELECTIVITY (protected vs unprotected "
    "become identical), not protection itself. Stated, not hidden.",
    "boundary_poisson ablation: the composite is statistics-derived, so the "
    "one synthesis boundary is the input presentation; the arm replaces the "
    "correlation-preserving constant-current presentation with a rate-"
    "matched per-step Bernoulli spike surrogate (F13 dichotomy at the "
    "composite's boundary; spike probability 0.2/step engineering).",
    "relay_removed ablation: config-only realization of 'no thin-relay "
    "mechanism' = relay widened to a transparent output-sized stage with "
    "the MB gate off (n_relay=n_output, k_relay_in raised, mb writes 0); "
    "the <=1% budget envelope is asserted. The screen's motif:relay_gate KO "
    "covers the complementary 'relay path dead' manipulation.",
    "interference items: salience fixed at salience_lo, never protected, "
    "registered in the replay pool (prioritised replay must earn its keep "
    "against them); probes sample only the P stream items.",
    "retained-at-T: an item counts as retained iff its recall overlap vs "
    "its stored code >= 0.9 (contract precision bar) at the arm's readout; "
    "composite readout = max(formation, storage) per formal-model #5 "
    "(union of fast and slow stores); stage-alone arms read their stage.",
    "n_synapses in results.json = the composite arm's total synapse count "
    "(driver-spec; validator floor 1e10).",
    "screen: single KOs enumerate builders' cell-type labels (granule, "
    "pyramidal, storage, interneuron) + structural motifs; pair KOs "
    "sampled seeded with budget <= 2x singles (driver-spec); ranking = "
    "Mahalanobis distance (numpy cov + pinv) of the degradation vector "
    "over {capacity, separation, completion, retention} from the KO cloud.",
    "alloc_k=0 (allocation_off) exercises zero-size dist collectives in "
    "allocate_global; verified under gloo, NCCL >= 2.10 required at full "
    "scale.",
]

__all__ = ["H3Config", "full_config", "smoke_config", "tiny_config",
           "base_composite", "SEED_ROOT", "STAGE_ALONE", "ABLATIONS",
           "CONTEXT_ROWS", "LAMBDA_FACTORS_FULL", "HEADLINE_LAMBDA",
           "C_TSODYKS", "PROVENANCE_NOTES"]
