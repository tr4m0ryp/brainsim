"""Execution flags a work item carries into runtime construction (the two
manipulations without a CompositeConfig knob run through these)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RunFlags:
    binding: str = "btsp"          # "btsp" | "stdp" (ablation swap)
    uniform_replay: bool = False   # flatten replay salience after writes
    poisson: bool = False          # input-boundary Poisson surrogate
    topf_off: tuple = ()           # stages with the top-f readout knocked out
    ops: tuple = ()                # named KO ops applied post-build


__all__ = ["RunFlags"]
