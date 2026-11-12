"""Whole-fly engine gate: Shiu et al. 2024 sugar-GRN -> MN9 checks on brainsim."""
from brainsim.validate.whole_fly.gate import run_gate
from brainsim.validate.whole_fly.mapping import event_current, param_mapping

__all__ = ["event_current", "param_mapping", "run_gate"]
