"""Backend seam for the H2 protocol: one numpy implementation (reference,
works everywhere) and one torch implementation (full-scale vehicle). The
protocol code is written once against this duck-typed API.

Stream randomness (item codes, cues, probe samples, promotion pattern) is
ALWAYS host numpy and engine-independent; only turnover noise and cascade
uniforms come from the backend generator, so ledgers and budgets are exactly
engine-invariant.

torch and brainsim.rules.torch_rules are imported lazily INSIDE the torch
branch only (project rule: brainsim.rules stays torch-free).
"""
from __future__ import annotations


def get_ops(engine: str, seed: int, cascade_params, device: str = "auto",
            weight_dtype: str = "float32"):
    if engine == "numpy":
        from .numpy_ops import NumpyOps

        return NumpyOps(seed, cascade_params)
    if engine == "torch":
        try:
            from .torch_ops import TorchOps
        except ImportError as e:  # torch or torch_rules missing: no stubs.
            raise SystemExit(
                "battery_h2: --engine torch requires torch and "
                f"brainsim.rules.torch_rules (import failed: {e}). "
                "Run with --engine numpy or install the torch stack."
            )
        return TorchOps(seed, cascade_params, device=device, weight_dtype=weight_dtype)
    raise ValueError(f"unknown engine {engine!r}")


__all__ = ["get_ops"]
