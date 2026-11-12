"""Sparse LIF engine core: CPU/numpy reference path (T1) plus the torch
engine package (T5).

torch is imported only by `engine.backend` (availability probe, inside a
try/except) and inside `engine.torch_core` (the torch engine itself, never
imported unless a caller asks for that backend). The reference engine (lif,
propagate, network) is pure numpy and fully testable without a GPU or torch
installed.
"""
