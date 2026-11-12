"""Optional GPU backend probe (T5). The numpy CPU path never requires torch;
this module and the `engine.torch_core` package are the only places in
brainsim allowed to import it -- here only inside a try/except so the
package stays importable and testable without it.
"""
from __future__ import annotations

try:
    import torch
except ImportError:  # pragma: no cover - exercised on machines without torch
    torch = None


def torch_available() -> bool:
    return torch is not None


def cuda_available() -> bool:
    return torch_available() and torch.cuda.is_available()


def describe_backend() -> str:
    """Human-readable backend description for bench/log output."""
    if not torch_available():
        return "numpy-cpu (torch not installed)"
    if cuda_available():
        return f"torch-cuda ({torch.cuda.get_device_name(0)})"
    return "torch-cpu (cuda unavailable)"


__all__ = ["torch_available", "cuda_available", "describe_backend"]
