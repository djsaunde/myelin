"""Helpers for optional GPU dependencies."""

from __future__ import annotations

from importlib.util import find_spec

_TORCH_AVAILABLE = find_spec("torch") is not None
_TRITON_AVAILABLE = find_spec("triton") is not None


def has_torch() -> bool:
    """Return whether PyTorch is importable in the current environment."""

    return _TORCH_AVAILABLE


def has_triton() -> bool:
    """Return whether Triton is importable in the current environment."""

    return _TRITON_AVAILABLE


def require_triton():
    """Import triton or raise an actionable error."""

    try:
        import triton
    except ModuleNotFoundError as exc:
        msg = "Triton is required for fused kernels. Install with `uv sync --extra cuda`."
        raise ModuleNotFoundError(msg) from exc
    return triton
