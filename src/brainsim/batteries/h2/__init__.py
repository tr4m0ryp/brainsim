"""Battery H2: the H01 permanence motif and the retention-capacity frontier.

FROZEN contract: battery-h2-permanence-
frontier.md. Verdicts, bootstrap CIs and the results writer are the frozen
functions in brainsim.batteries.verdicts; this package only produces the
arm measurements and calls them.

Public surface: config presets, the artifact loader, twin builder + gate,
and the runner (shard / merge / smoke).
"""
from .artifact import H01Stats, load as load_artifact
from .config import H2Config, full_config, smoke_config, tiny_config
from .runner import config_for, default_seeds, merge_files, run, run_shard
from .twin import Twin, TwinGateError, build_twin, require_gate

__all__ = [
    "H01Stats", "load_artifact",
    "H2Config", "full_config", "smoke_config", "tiny_config",
    "config_for", "default_seeds", "merge_files", "run", "run_shard",
    "Twin", "TwinGateError", "build_twin", "require_gate",
]
