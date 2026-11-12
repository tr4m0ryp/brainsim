"""Full-scale composite runtime (C7/C8): the spiking, sharded,
plasticity-attached end-to-end memory loop the H1 composite arm and the H3
battery execute on. Scaled sibling of composite.torch_adapter (which stays
the smoke-scale, dense-relaxation reference next to composite.pipeline).

Public surface for battery drivers:
    CompositeConfig      one source of truth for scale + parameters
    CompositeRuntime     write_phase / recall_phase / recall_codes /
                         storage_phase / snapshot
    run_full_smoke_cycle pipeline-schedule smoke cycle (code check)
    forecast / check_fit per-rank memory projection (F24) before allocation

torch is imported by runtime/build/phases (via engine.torch_core), not by
config, so scale planning and forecasts stay importable everywhere.
"""
from brainsim.composite.full.config import STAGES, CompositeConfig

__all__ = [
    "CompositeConfig",
    "STAGES",
    "CompositeRuntime",
    "run_full_smoke_cycle",
    "forecast",
    "check_fit",
    "overlap_sets",
    "make_cue",
    "encode_codes",
    "recall_codes",
]

_LAZY = {
    "CompositeRuntime": "brainsim.composite.full.runtime",
    "run_full_smoke_cycle": "brainsim.composite.full.cycle",
    "forecast": "brainsim.composite.full.build",
    "check_fit": "brainsim.composite.full.build",
    "overlap_sets": "brainsim.composite.full.phases",
    "make_cue": "brainsim.composite.full.phases",
    "encode_codes": "brainsim.composite.full.phases",
    "recall_codes": "brainsim.composite.full.phases",
}


def __getattr__(name: str):
    if name in _LAZY:
        import importlib

        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
