"""Battery H3: composite-beats-parts + the hidden-keys screen.

Frozen contract: battery-h3-composite.md
(sha embedded in every partial and in results.json). Design decisions:
the H3 driver specification. Verdicts come ONLY
from brainsim.batteries.verdicts.h3_verdict via report.merge_partials.

Layout: config (constants + provenance), budget (matched-resources solver,
1% assertions), arms (contract arms 1-5 on CompositeConfig), flags/systems
(runtime wrappers: KO ops, readout/boundary subclass), binding (BTSP->STDP
swap), adapters (SingleSystem/BagSystem), protocol (episodic stream,
probes, lambda sweep; H2-shared turnover constant), screen (KO catalog +
Mahalanobis ranking), analysis (merge-side tables/overlay), report
(partials, merge, frozen-verdict wiring), runner (shards, smoke rails).
"""
from brainsim.batteries.h3.config import (H3Config, full_config,
                                          smoke_config, tiny_config)

__all__ = ["H3Config", "full_config", "smoke_config", "tiny_config"]
