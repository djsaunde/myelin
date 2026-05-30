"""Baseline implementations and timing helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from myelin.functional import lif_unroll
from myelin.neurons import LIFParams, LIFState

LIFUnroll = Callable[[torch.Tensor, LIFState, LIFParams], tuple[LIFState, torch.Tensor]]


@dataclass(frozen=True)
class TimingResult:
    """Average latency measurement for a benchmarked callable."""

    label: str
    seconds: float


@dataclass(frozen=True)
class CompiledFunction:
    """Compiled callable plus the time spent constructing it."""

    fn: LIFUnroll
    construction_seconds: float


def eager_lif_unroll(
    inputs: torch.Tensor,
    initial_state: LIFState,
    params: LIFParams,
) -> tuple[LIFState, torch.Tensor]:
    """Eager PyTorch LIF reference path."""

    return lif_unroll(inputs, initial_state, params)


def compiled_lif_unroll() -> LIFUnroll:
    """Return a compiled LIF reference path.

    ``fullgraph=True`` is intentional for benchmarking. If this fails, the
    reference path is not being captured as one graph and the benchmark should
    surface that instead of silently measuring a graph-broken fallback.
    """

    return torch.compile(lif_unroll, mode="reduce-overhead", fullgraph=True)


def timed_compiled_lif_unroll() -> CompiledFunction:
    """Return a compiled LIF reference path and wrapper construction time."""

    import time

    start = time.perf_counter()
    fn = compiled_lif_unroll()
    return CompiledFunction(fn=fn, construction_seconds=time.perf_counter() - start)


def synchronize_if_needed(device: torch.device | str) -> None:
    """Synchronize CUDA work when timing GPU operations."""

    torch_device = torch.device(device)
    if torch_device.type == "cuda":
        torch.cuda.synchronize(torch_device)


def time_lif_unroll(
    label: str,
    fn: LIFUnroll,
    inputs: torch.Tensor,
    initial_state: LIFState,
    params: LIFParams,
    *,
    warmup: int,
    repeats: int,
) -> TimingResult:
    """Time an LIF unroll callable with warmup excluded from measurement."""

    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if repeats <= 0:
        raise ValueError("repeats must be positive")

    import time

    for _ in range(warmup):
        fn(inputs, initial_state, params)
    synchronize_if_needed(inputs.device)

    start = time.perf_counter()
    for _ in range(repeats):
        fn(inputs, initial_state, params)
    synchronize_if_needed(inputs.device)

    return TimingResult(label=label, seconds=(time.perf_counter() - start) / repeats)


def max_cuda_memory_allocated(device: torch.device | str) -> int | None:
    """Return peak CUDA memory in bytes, or ``None`` for non-CUDA devices."""

    torch_device = torch.device(device)
    if torch_device.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(torch_device)


def reset_cuda_peak_memory(device: torch.device | str) -> None:
    """Reset peak CUDA memory stats when available."""

    torch_device = torch.device(device)
    if torch_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(torch_device)


def compiled_available() -> bool:
    """Return whether this PyTorch build exposes ``torch.compile``."""

    compile_fn: Any = getattr(torch, "compile", None)
    return callable(compile_fn)
