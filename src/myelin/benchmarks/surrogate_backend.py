"""Benchmark surrogate LIF backend choices."""

from __future__ import annotations

import argparse
import time
import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime

import torch

from myelin._optional import has_triton
from myelin.baselines import compiled_available, synchronize_if_needed
from myelin.benchmarks.lif import format_memory, format_ms, format_speedup, gpu_name, parse_shape
from myelin.checkpointing import CheckpointSize, parse_checkpoint_size, resolve_checkpoint_size
from myelin.neurons import LIFParams, LIFState
from myelin.surrogates import SURROGATE_NAMES, SurrogateName

StepFn = Callable[[torch.Tensor, torch.Tensor, LIFParams], torch.Tensor]

DEFAULT_SHAPES = [
    (25, 64, 2048),
    (100, 64, 2048),
    (200, 64, 2048),
]

WORKLOAD_DIAGRAM = """\
```text
[T, B, F] inputs
      |
      v
[F, N] trainable weight
      |
      v
backend boundary:
    eager/compiled/Triton LIF: materialized [T, B, N] currents
    stream/Triton synapse: per-timestep or fused current computation
      |
      v
surrogate LIF recurrence
      |
      v
spike-rate style scalar loss
      |
      v
loss.backward() -> d(weight), optional d(inputs)
```"""


@dataclass(frozen=True)
class Timing:
    forward_only_seconds: float
    split_forward_seconds: float
    backward_seconds: float
    forward_backward_seconds: float


@dataclass(frozen=True)
class Memory:
    allocated_bytes: int | None
    forward_peak_bytes: int | None
    backward_peak_bytes: int | None


def make_inputs(
    timesteps: int,
    batch: int,
    features: int,
    neurons: int,
    device: str,
    input_grad: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = torch.rand((timesteps, batch, features), device=device)
    inputs.requires_grad_(input_grad)
    weight = (torch.rand((features, neurons), device=device) - 0.5) * 0.02
    weight.requires_grad_(True)
    return inputs, weight


def make_step(
    *,
    backend: str,
    surrogate: SurrogateName,
    surrogate_slope: float,
) -> StepFn:
    def step(inputs: torch.Tensor, weight: torch.Tensor, params: LIFParams) -> torch.Tensor:
        from myelin.kernels import surrogate_lif_forward

        currents = torch.matmul(inputs, weight)
        initial = LIFState(
            membrane=torch.zeros(
                currents.shape[1:],
                dtype=currents.dtype,
                device=currents.device,
            )
        )
        final_state, spikes = surrogate_lif_forward(
            currents,
            initial,
            params,
            surrogate=surrogate,
            surrogate_slope=surrogate_slope,
            hard_forward=True,
            backend=backend,  # pyright: ignore[reportArgumentType]
        )
        return spikes.mean().square() + 0.01 * final_state.membrane.square().mean()

    return step


def make_stream_synapse_step(
    *,
    surrogate: SurrogateName,
    surrogate_slope: float,
) -> StepFn:
    def step(inputs: torch.Tensor, weight: torch.Tensor, params: LIFParams) -> torch.Tensor:
        from myelin.kernels import linear_surrogate_lif_forward

        final_state, spikes = linear_surrogate_lif_forward(
            inputs,
            weight,
            None,
            params,
            surrogate=surrogate,
            surrogate_slope=surrogate_slope,
            hard_forward=True,
            backend="torch",
        )
        return spikes.mean().square() + 0.01 * final_state.membrane.square().mean()

    return step


def make_checkpoint_synapse_step(
    *,
    surrogate: SurrogateName,
    surrogate_slope: float,
    checkpoint_size: CheckpointSize,
) -> StepFn:
    def step(inputs: torch.Tensor, weight: torch.Tensor, params: LIFParams) -> torch.Tensor:
        from myelin.kernels import linear_surrogate_lif_forward

        final_state, spikes = linear_surrogate_lif_forward(
            inputs,
            weight,
            None,
            params,
            surrogate=surrogate,
            surrogate_slope=surrogate_slope,
            hard_forward=True,
            backend="torch",
            checkpoint_size=checkpoint_size,
        )
        return spikes.mean().square() + 0.01 * final_state.membrane.square().mean()

    return step


def make_triton_stream_synapse_step(
    *,
    surrogate: SurrogateName,
    surrogate_slope: float,
) -> StepFn:
    def step(inputs: torch.Tensor, weight: torch.Tensor, params: LIFParams) -> torch.Tensor:
        from myelin.kernels import linear_surrogate_lif_forward

        final_state, spikes = linear_surrogate_lif_forward(
            inputs,
            weight,
            None,
            params,
            surrogate=surrogate,
            surrogate_slope=surrogate_slope,
            hard_forward=True,
            backend="triton",
        )
        return spikes.mean().square() + 0.01 * final_state.membrane.square().mean()

    return step


def make_generated_triton_stream_synapse_step(
    *,
    surrogate: SurrogateName,
    surrogate_slope: float,
) -> StepFn:
    def step(inputs: torch.Tensor, weight: torch.Tensor, params: LIFParams) -> torch.Tensor:
        from myelin.kernels import linear_surrogate_lif_forward

        final_state, spikes = linear_surrogate_lif_forward(
            inputs,
            weight,
            None,
            params,
            surrogate=surrogate,
            surrogate_slope=surrogate_slope,
            hard_forward=True,
            backend="triton_generated",
        )
        return spikes.mean().square() + 0.01 * final_state.membrane.square().mean()

    return step


def make_generated_triton_checkpoint_synapse_step(
    *,
    surrogate: SurrogateName,
    surrogate_slope: float,
    checkpoint_size: CheckpointSize,
) -> StepFn:
    def step(inputs: torch.Tensor, weight: torch.Tensor, params: LIFParams) -> torch.Tensor:
        from myelin.kernels import linear_surrogate_lif_forward

        final_state, spikes = linear_surrogate_lif_forward(
            inputs,
            weight,
            None,
            params,
            surrogate=surrogate,
            surrogate_slope=surrogate_slope,
            hard_forward=True,
            backend="triton_generated",
            checkpoint_size=checkpoint_size,
        )
        return spikes.mean().square() + 0.01 * final_state.membrane.square().mean()

    return step


def make_triton_checkpoint_synapse_step(
    *,
    surrogate: SurrogateName,
    surrogate_slope: float,
    checkpoint_size: CheckpointSize,
) -> StepFn:
    def step(inputs: torch.Tensor, weight: torch.Tensor, params: LIFParams) -> torch.Tensor:
        from myelin.kernels import linear_surrogate_lif_forward

        final_state, spikes = linear_surrogate_lif_forward(
            inputs,
            weight,
            None,
            params,
            surrogate=surrogate,
            surrogate_slope=surrogate_slope,
            hard_forward=True,
            backend="triton",
            checkpoint_size=checkpoint_size,
        )
        return spikes.mean().square() + 0.01 * final_state.membrane.square().mean()

    return step


def compile_step(fn: StepFn, compile_mode: str) -> StepFn:
    if compile_mode == "default":
        return torch.compile(fn, fullgraph=True)
    return torch.compile(fn, mode=compile_mode, fullgraph=True)


def time_forward(
    fn: StepFn,
    inputs: torch.Tensor,
    weight: torch.Tensor,
    params: LIFParams,
    *,
    warmup: int,
    repeats: int,
) -> float:
    for _ in range(warmup):
        inputs.grad = None
        weight.grad = None
        fn(inputs, weight, params)
    synchronize_if_needed(inputs.device)

    start = time.perf_counter()
    for _ in range(repeats):
        inputs.grad = None
        weight.grad = None
        fn(inputs, weight, params)
    synchronize_if_needed(inputs.device)
    return (time.perf_counter() - start) / repeats


def time_forward_backward(
    fn: StepFn,
    inputs: torch.Tensor,
    weight: torch.Tensor,
    params: LIFParams,
    *,
    warmup: int,
    repeats: int,
) -> tuple[float, float, float]:
    for _ in range(warmup):
        inputs.grad = None
        weight.grad = None
        loss = fn(inputs, weight, params)
        loss.backward()
    synchronize_if_needed(inputs.device)

    forward_elapsed = 0.0
    backward_elapsed = 0.0
    for _ in range(repeats):
        inputs.grad = None
        weight.grad = None
        start = time.perf_counter()
        loss = fn(inputs, weight, params)
        synchronize_if_needed(inputs.device)
        forward_elapsed += time.perf_counter() - start

        start = time.perf_counter()
        loss.backward()
        synchronize_if_needed(inputs.device)
        backward_elapsed += time.perf_counter() - start

    forward_seconds = forward_elapsed / repeats
    backward_seconds = backward_elapsed / repeats
    return forward_seconds, backward_seconds, forward_seconds + backward_seconds


def time_step(
    fn: StepFn,
    inputs: torch.Tensor,
    weight: torch.Tensor,
    params: LIFParams,
    *,
    warmup: int,
    repeats: int,
) -> Timing:
    forward_only = time_forward(fn, inputs, weight, params, warmup=warmup, repeats=repeats)
    split_forward, backward, forward_backward = time_forward_backward(
        fn,
        inputs,
        weight,
        params,
        warmup=warmup,
        repeats=repeats,
    )
    return Timing(
        forward_only_seconds=forward_only,
        split_forward_seconds=split_forward,
        backward_seconds=backward,
        forward_backward_seconds=forward_backward,
    )


def memory_checkpoints(
    fn: StepFn,
    inputs: torch.Tensor,
    weight: torch.Tensor,
    params: LIFParams,
) -> Memory:
    device = inputs.device
    if device.type != "cuda":
        return Memory(None, None, None)

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    allocated = torch.cuda.memory_allocated(device)

    weight.grad = None
    inputs.grad = None
    loss = fn(inputs, weight, params)
    synchronize_if_needed(device)
    forward_peak = torch.cuda.max_memory_allocated(device)

    loss.backward()
    synchronize_if_needed(device)
    backward_peak = torch.cuda.max_memory_allocated(device)

    return Memory(allocated, forward_peak, backward_peak)


def benchmark_one(
    *,
    timesteps: int,
    batch: int,
    features: int,
    neurons: int,
    device: str,
    warmup: int,
    repeats: int,
    compile_enabled: bool,
    compile_mode: str,
    surrogate: SurrogateName,
    surrogate_slope: float,
    checkpoint_size: CheckpointSize,
    input_grad: bool,
) -> dict[str, object]:
    inputs, eager_weight = make_inputs(
        timesteps,
        batch,
        features,
        neurons,
        device,
        input_grad=input_grad,
    )
    resolved_checkpoint_size = resolve_checkpoint_size(timesteps, checkpoint_size)
    compiled_weight = eager_weight.detach().clone().requires_grad_(True)
    stream_weight = eager_weight.detach().clone().requires_grad_(True)
    checkpoint_weight = eager_weight.detach().clone().requires_grad_(True)
    triton_weight = eager_weight.detach().clone().requires_grad_(True)
    generated_triton_weight = eager_weight.detach().clone().requires_grad_(True)
    triton_stream_weight = eager_weight.detach().clone().requires_grad_(True)
    generated_triton_stream_weight = eager_weight.detach().clone().requires_grad_(True)
    triton_checkpoint_weight = eager_weight.detach().clone().requires_grad_(True)
    generated_triton_checkpoint_weight = eager_weight.detach().clone().requires_grad_(True)
    params = LIFParams()

    eager_step = make_step(
        backend="torch",
        surrogate=surrogate,
        surrogate_slope=surrogate_slope,
    )
    eager_timing = time_step(
        eager_step,
        inputs,
        eager_weight,
        params,
        warmup=warmup,
        repeats=repeats,
    )
    eager_memory = memory_checkpoints(eager_step, inputs, eager_weight, params)

    stream_step = make_stream_synapse_step(
        surrogate=surrogate,
        surrogate_slope=surrogate_slope,
    )
    stream_timing = time_step(
        stream_step,
        inputs,
        stream_weight,
        params,
        warmup=warmup,
        repeats=repeats,
    )
    stream_memory = memory_checkpoints(stream_step, inputs, stream_weight, params)

    checkpoint_step = make_checkpoint_synapse_step(
        surrogate=surrogate,
        surrogate_slope=surrogate_slope,
        checkpoint_size=resolved_checkpoint_size,
    )
    checkpoint_timing = time_step(
        checkpoint_step,
        inputs,
        checkpoint_weight,
        params,
        warmup=warmup,
        repeats=repeats,
    )
    checkpoint_memory = memory_checkpoints(checkpoint_step, inputs, checkpoint_weight, params)

    compiled_timing = None
    compiled_memory = Memory(None, None, None)
    compile_warmup_seconds = None
    compile_error = None
    if compile_enabled:
        try:
            compiled_step = compile_step(eager_step, compile_mode)
            start = time.perf_counter()
            inputs.grad = None
            compiled_weight.grad = None
            loss = compiled_step(inputs, compiled_weight, params)
            loss.backward()
            synchronize_if_needed(device)
            compile_warmup_seconds = time.perf_counter() - start
            compiled_timing = time_step(
                compiled_step,
                inputs,
                compiled_weight,
                params,
                warmup=warmup,
                repeats=repeats,
            )
            compiled_memory = memory_checkpoints(compiled_step, inputs, compiled_weight, params)
        except Exception as exc:  # noqa: BLE001 - benchmark should report compiler failures.
            compile_error = f"{type(exc).__name__}: {exc}"
        finally:
            synchronize_if_needed(device)

    triton_timing = None
    triton_memory = Memory(None, None, None)
    generated_triton_timing = None
    generated_triton_memory = Memory(None, None, None)
    triton_stream_timing = None
    triton_stream_memory = Memory(None, None, None)
    generated_triton_stream_timing = None
    generated_triton_stream_memory = Memory(None, None, None)
    triton_checkpoint_timing = None
    triton_checkpoint_memory = Memory(None, None, None)
    generated_triton_checkpoint_timing = None
    generated_triton_checkpoint_memory = Memory(None, None, None)
    triton_error = None
    generated_triton_error = None
    triton_stream_error = None
    generated_triton_stream_error = None
    triton_checkpoint_error = None
    generated_triton_checkpoint_error = None
    if torch.device(device).type == "cuda" and has_triton():
        try:
            triton_step = make_step(
                backend="triton",
                surrogate=surrogate,
                surrogate_slope=surrogate_slope,
            )
            triton_timing = time_step(
                triton_step,
                inputs,
                triton_weight,
                params,
                warmup=warmup,
                repeats=repeats,
            )
            triton_memory = memory_checkpoints(triton_step, inputs, triton_weight, params)
        except Exception as exc:  # noqa: BLE001 - benchmark should report backend failures.
            triton_error = f"{type(exc).__name__}: {exc}"
        finally:
            synchronize_if_needed(device)

        try:
            generated_triton_step = make_step(
                backend="triton_generated",
                surrogate=surrogate,
                surrogate_slope=surrogate_slope,
            )
            generated_triton_timing = time_step(
                generated_triton_step,
                inputs,
                generated_triton_weight,
                params,
                warmup=warmup,
                repeats=repeats,
            )
            generated_triton_memory = memory_checkpoints(
                generated_triton_step,
                inputs,
                generated_triton_weight,
                params,
            )
        except Exception as exc:  # noqa: BLE001 - benchmark should report backend failures.
            generated_triton_error = f"{type(exc).__name__}: {exc}"
        finally:
            synchronize_if_needed(device)

        try:
            triton_stream_step = make_triton_stream_synapse_step(
                surrogate=surrogate,
                surrogate_slope=surrogate_slope,
            )
            triton_stream_timing = time_step(
                triton_stream_step,
                inputs,
                triton_stream_weight,
                params,
                warmup=warmup,
                repeats=repeats,
            )
            triton_stream_memory = memory_checkpoints(
                triton_stream_step,
                inputs,
                triton_stream_weight,
                params,
            )
        except Exception as exc:  # noqa: BLE001 - benchmark should report backend failures.
            triton_stream_error = f"{type(exc).__name__}: {exc}"
        finally:
            synchronize_if_needed(device)

        try:
            generated_triton_stream_step = make_generated_triton_stream_synapse_step(
                surrogate=surrogate,
                surrogate_slope=surrogate_slope,
            )
            generated_triton_stream_timing = time_step(
                generated_triton_stream_step,
                inputs,
                generated_triton_stream_weight,
                params,
                warmup=warmup,
                repeats=repeats,
            )
            generated_triton_stream_memory = memory_checkpoints(
                generated_triton_stream_step,
                inputs,
                generated_triton_stream_weight,
                params,
            )
        except Exception as exc:  # noqa: BLE001 - benchmark should report backend failures.
            generated_triton_stream_error = f"{type(exc).__name__}: {exc}"
        finally:
            synchronize_if_needed(device)

        try:
            triton_checkpoint_step = make_triton_checkpoint_synapse_step(
                surrogate=surrogate,
                surrogate_slope=surrogate_slope,
                checkpoint_size=resolved_checkpoint_size,
            )
            triton_checkpoint_timing = time_step(
                triton_checkpoint_step,
                inputs,
                triton_checkpoint_weight,
                params,
                warmup=warmup,
                repeats=repeats,
            )
            triton_checkpoint_memory = memory_checkpoints(
                triton_checkpoint_step,
                inputs,
                triton_checkpoint_weight,
                params,
            )
        except Exception as exc:  # noqa: BLE001 - benchmark should report backend failures.
            triton_checkpoint_error = f"{type(exc).__name__}: {exc}"
        finally:
            synchronize_if_needed(device)

        try:
            generated_triton_checkpoint_step = make_generated_triton_checkpoint_synapse_step(
                surrogate=surrogate,
                surrogate_slope=surrogate_slope,
                checkpoint_size=resolved_checkpoint_size,
            )
            generated_triton_checkpoint_timing = time_step(
                generated_triton_checkpoint_step,
                inputs,
                generated_triton_checkpoint_weight,
                params,
                warmup=warmup,
                repeats=repeats,
            )
            generated_triton_checkpoint_memory = memory_checkpoints(
                generated_triton_checkpoint_step,
                inputs,
                generated_triton_checkpoint_weight,
                params,
            )
        except Exception as exc:  # noqa: BLE001 - benchmark should report backend failures.
            generated_triton_checkpoint_error = f"{type(exc).__name__}: {exc}"
        finally:
            synchronize_if_needed(device)

    return {
        "timesteps": timesteps,
        "batch": batch,
        "features": features,
        "neurons": neurons,
        "eager_forward_only_seconds": eager_timing.forward_only_seconds,
        "stream_forward_only_seconds": stream_timing.forward_only_seconds,
        "checkpoint_forward_only_seconds": checkpoint_timing.forward_only_seconds,
        "compiled_forward_only_seconds": None
        if compiled_timing is None
        else compiled_timing.forward_only_seconds,
        "triton_forward_only_seconds": None
        if triton_timing is None
        else triton_timing.forward_only_seconds,
        "generated_triton_forward_only_seconds": None
        if generated_triton_timing is None
        else generated_triton_timing.forward_only_seconds,
        "triton_stream_forward_only_seconds": None
        if triton_stream_timing is None
        else triton_stream_timing.forward_only_seconds,
        "generated_triton_stream_forward_only_seconds": None
        if generated_triton_stream_timing is None
        else generated_triton_stream_timing.forward_only_seconds,
        "triton_checkpoint_forward_only_seconds": None
        if triton_checkpoint_timing is None
        else triton_checkpoint_timing.forward_only_seconds,
        "generated_triton_checkpoint_forward_only_seconds": None
        if generated_triton_checkpoint_timing is None
        else generated_triton_checkpoint_timing.forward_only_seconds,
        "eager_split_forward_seconds": eager_timing.split_forward_seconds,
        "stream_split_forward_seconds": stream_timing.split_forward_seconds,
        "checkpoint_split_forward_seconds": checkpoint_timing.split_forward_seconds,
        "compiled_split_forward_seconds": None
        if compiled_timing is None
        else compiled_timing.split_forward_seconds,
        "triton_split_forward_seconds": None
        if triton_timing is None
        else triton_timing.split_forward_seconds,
        "generated_triton_split_forward_seconds": None
        if generated_triton_timing is None
        else generated_triton_timing.split_forward_seconds,
        "triton_stream_split_forward_seconds": None
        if triton_stream_timing is None
        else triton_stream_timing.split_forward_seconds,
        "generated_triton_stream_split_forward_seconds": None
        if generated_triton_stream_timing is None
        else generated_triton_stream_timing.split_forward_seconds,
        "triton_checkpoint_split_forward_seconds": None
        if triton_checkpoint_timing is None
        else triton_checkpoint_timing.split_forward_seconds,
        "generated_triton_checkpoint_split_forward_seconds": None
        if generated_triton_checkpoint_timing is None
        else generated_triton_checkpoint_timing.split_forward_seconds,
        "eager_backward_seconds": eager_timing.backward_seconds,
        "stream_backward_seconds": stream_timing.backward_seconds,
        "checkpoint_backward_seconds": checkpoint_timing.backward_seconds,
        "compiled_backward_seconds": None
        if compiled_timing is None
        else compiled_timing.backward_seconds,
        "triton_backward_seconds": None
        if triton_timing is None
        else triton_timing.backward_seconds,
        "generated_triton_backward_seconds": None
        if generated_triton_timing is None
        else generated_triton_timing.backward_seconds,
        "triton_stream_backward_seconds": None
        if triton_stream_timing is None
        else triton_stream_timing.backward_seconds,
        "generated_triton_stream_backward_seconds": None
        if generated_triton_stream_timing is None
        else generated_triton_stream_timing.backward_seconds,
        "triton_checkpoint_backward_seconds": None
        if triton_checkpoint_timing is None
        else triton_checkpoint_timing.backward_seconds,
        "generated_triton_checkpoint_backward_seconds": None
        if generated_triton_checkpoint_timing is None
        else generated_triton_checkpoint_timing.backward_seconds,
        "eager_forward_backward_seconds": eager_timing.forward_backward_seconds,
        "stream_forward_backward_seconds": stream_timing.forward_backward_seconds,
        "checkpoint_forward_backward_seconds": checkpoint_timing.forward_backward_seconds,
        "compiled_forward_backward_seconds": None
        if compiled_timing is None
        else compiled_timing.forward_backward_seconds,
        "triton_forward_backward_seconds": None
        if triton_timing is None
        else triton_timing.forward_backward_seconds,
        "generated_triton_forward_backward_seconds": None
        if generated_triton_timing is None
        else generated_triton_timing.forward_backward_seconds,
        "triton_stream_forward_backward_seconds": None
        if triton_stream_timing is None
        else triton_stream_timing.forward_backward_seconds,
        "generated_triton_stream_forward_backward_seconds": None
        if generated_triton_stream_timing is None
        else generated_triton_stream_timing.forward_backward_seconds,
        "triton_checkpoint_forward_backward_seconds": None
        if triton_checkpoint_timing is None
        else triton_checkpoint_timing.forward_backward_seconds,
        "generated_triton_checkpoint_forward_backward_seconds": None
        if generated_triton_checkpoint_timing is None
        else generated_triton_checkpoint_timing.forward_backward_seconds,
        "compiled_forward_backward_speedup": None
        if compiled_timing is None
        else eager_timing.forward_backward_seconds / compiled_timing.forward_backward_seconds,
        "stream_forward_backward_speedup": (
            eager_timing.forward_backward_seconds / stream_timing.forward_backward_seconds
        ),
        "checkpoint_forward_backward_speedup": (
            eager_timing.forward_backward_seconds / checkpoint_timing.forward_backward_seconds
        ),
        "triton_forward_backward_speedup": None
        if triton_timing is None
        else eager_timing.forward_backward_seconds / triton_timing.forward_backward_seconds,
        "generated_triton_forward_backward_speedup": None
        if generated_triton_timing is None
        else eager_timing.forward_backward_seconds
        / generated_triton_timing.forward_backward_seconds,
        "triton_stream_forward_backward_speedup": None
        if triton_stream_timing is None
        else eager_timing.forward_backward_seconds / triton_stream_timing.forward_backward_seconds,
        "generated_triton_stream_forward_backward_speedup": None
        if generated_triton_stream_timing is None
        else eager_timing.forward_backward_seconds
        / generated_triton_stream_timing.forward_backward_seconds,
        "triton_checkpoint_forward_backward_speedup": None
        if triton_checkpoint_timing is None
        else eager_timing.forward_backward_seconds
        / triton_checkpoint_timing.forward_backward_seconds,
        "generated_triton_checkpoint_forward_backward_speedup": None
        if generated_triton_checkpoint_timing is None
        else eager_timing.forward_backward_seconds
        / generated_triton_checkpoint_timing.forward_backward_seconds,
        "compile_warmup_seconds": compile_warmup_seconds,
        "compile_error": compile_error,
        "triton_error": triton_error,
        "generated_triton_error": generated_triton_error,
        "triton_stream_error": triton_stream_error,
        "generated_triton_stream_error": generated_triton_stream_error,
        "triton_checkpoint_error": triton_checkpoint_error,
        "generated_triton_checkpoint_error": generated_triton_checkpoint_error,
        "eager_forward_peak_bytes": eager_memory.forward_peak_bytes,
        "stream_forward_peak_bytes": stream_memory.forward_peak_bytes,
        "checkpoint_forward_peak_bytes": checkpoint_memory.forward_peak_bytes,
        "compiled_forward_peak_bytes": compiled_memory.forward_peak_bytes,
        "triton_forward_peak_bytes": triton_memory.forward_peak_bytes,
        "generated_triton_forward_peak_bytes": generated_triton_memory.forward_peak_bytes,
        "triton_stream_forward_peak_bytes": triton_stream_memory.forward_peak_bytes,
        "generated_triton_stream_forward_peak_bytes": (
            generated_triton_stream_memory.forward_peak_bytes
        ),
        "triton_checkpoint_forward_peak_bytes": triton_checkpoint_memory.forward_peak_bytes,
        "generated_triton_checkpoint_forward_peak_bytes": (
            generated_triton_checkpoint_memory.forward_peak_bytes
        ),
        "eager_backward_peak_bytes": eager_memory.backward_peak_bytes,
        "stream_backward_peak_bytes": stream_memory.backward_peak_bytes,
        "checkpoint_backward_peak_bytes": checkpoint_memory.backward_peak_bytes,
        "compiled_backward_peak_bytes": compiled_memory.backward_peak_bytes,
        "triton_backward_peak_bytes": triton_memory.backward_peak_bytes,
        "generated_triton_backward_peak_bytes": generated_triton_memory.backward_peak_bytes,
        "triton_stream_backward_peak_bytes": triton_stream_memory.backward_peak_bytes,
        "generated_triton_stream_backward_peak_bytes": (
            generated_triton_stream_memory.backward_peak_bytes
        ),
        "triton_checkpoint_backward_peak_bytes": triton_checkpoint_memory.backward_peak_bytes,
        "generated_triton_checkpoint_backward_peak_bytes": (
            generated_triton_checkpoint_memory.backward_peak_bytes
        ),
    }


def run_sweep(
    shapes: Iterable[tuple[int, int, int]],
    *,
    features: int,
    device: str,
    warmup: int,
    repeats: int,
    compile_enabled: bool,
    compile_mode: str,
    surrogate: SurrogateName,
    surrogate_slope: float,
    checkpoint_size: CheckpointSize,
    input_grad: bool,
) -> list[dict[str, object]]:
    results = []
    for timesteps, batch, neurons in shapes:
        results.append(
            benchmark_one(
                timesteps=timesteps,
                batch=batch,
                features=features,
                neurons=neurons,
                device=device,
                warmup=warmup,
                repeats=repeats,
                compile_enabled=compile_enabled,
                compile_mode=compile_mode,
                surrogate=surrogate,
                surrogate_slope=surrogate_slope,
                checkpoint_size=checkpoint_size,
                input_grad=input_grad,
            )
        )
    return results


def environment_metadata(
    args: argparse.Namespace,
    compile_enabled: bool,
    shapes: Iterable[tuple[int, int, int]],
) -> dict[str, object]:
    resolved_checkpoint_sizes = {
        timesteps: resolve_checkpoint_size(timesteps, args.checkpoint_size)
        for timesteps, _batch, _neurons in shapes
    }
    return {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "device": str(torch.device(args.device)),
        "gpu": gpu_name(args.device),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "features": args.features,
        "surrogate": args.surrogate,
        "surrogate_slope": args.surrogate_slope,
        "checkpoint_size": args.checkpoint_size,
        "resolved_checkpoint_sizes": resolved_checkpoint_sizes,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "compile_enabled": compile_enabled,
        "compile_mode": args.compile_mode if compile_enabled else None,
        "compile_fullgraph": compile_enabled,
        "compile_time_included": False,
        "matmul_precision": torch.get_float32_matmul_precision(),
        "dtype": "torch.float32",
        "input_grad": args.input_grad,
    }


def optional_ratio(numerator: object, denominator: object) -> float | None:
    """Return numerator / denominator for optional benchmark values."""

    if not isinstance(numerator, float) or not isinstance(denominator, float):
        return None
    return numerator / denominator


def print_markdown(results: list[dict[str, object]], metadata: dict[str, object]) -> None:
    print("# Surrogate LIF Backend Benchmark")
    print()
    print("Compile time is excluded from latency measurements.")
    print()
    print("## Workload")
    print()
    print(WORKLOAD_DIAGRAM)
    print()
    print("## Environment")
    print()
    for key, value in metadata.items():
        print(f"- `{key}`: `{value}`")
    print()
    print("## Results")
    print()
    print("### Latency")
    print()
    print(
        "| T | Batch | Features | N | Eager Fwd-only ms | Compiled Fwd-only ms | "
        "Stream Fwd-only ms | Checkpoint Fwd-only ms | Triton LIF Fwd-only ms | "
        "Triton Synapse Fwd-only ms | Triton Checkpoint Fwd-only ms | "
        "Eager Split Fwd ms | "
        "Compiled Split Fwd ms | Stream Split Fwd ms | Checkpoint Split Fwd ms | "
        "Triton Split Fwd ms | Triton Synapse Split Fwd ms | "
        "Triton Checkpoint Split Fwd ms | Eager Bwd ms | Compiled Bwd ms | "
        "Stream Bwd ms | Checkpoint Bwd ms | Triton LIF Bwd ms | "
        "Triton Synapse Bwd ms | Triton Checkpoint Bwd ms | Eager Fwd+Bwd ms | "
        "Compiled Fwd+Bwd ms | Stream Fwd+Bwd ms | Checkpoint Fwd+Bwd ms | "
        "Triton LIF Fwd+Bwd ms | Triton Synapse Fwd+Bwd ms | "
        "Triton Checkpoint Fwd+Bwd ms | "
        "Compiled Fwd+Bwd Speedup | Stream Fwd+Bwd Speedup | "
        "Checkpoint Fwd+Bwd Speedup | Triton LIF Fwd+Bwd Speedup | "
        "Triton Synapse Fwd+Bwd Speedup | Triton Checkpoint Fwd+Bwd Speedup | "
        "Compile Warmup ms | Compile Error | Triton Error | Triton Synapse Error | "
        "Triton Checkpoint Error |"
    )
    print(
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|"
    )
    for result in results:
        print(
            f"| {result['timesteps']} | {result['batch']} | {result['features']} | "
            f"{result['neurons']} | "
            f"{format_ms(result['eager_forward_only_seconds'])} | "
            f"{format_ms(result['compiled_forward_only_seconds'])} | "
            f"{format_ms(result['stream_forward_only_seconds'])} | "
            f"{format_ms(result['checkpoint_forward_only_seconds'])} | "
            f"{format_ms(result['triton_forward_only_seconds'])} | "
            f"{format_ms(result['triton_stream_forward_only_seconds'])} | "
            f"{format_ms(result['triton_checkpoint_forward_only_seconds'])} | "
            f"{format_ms(result['eager_split_forward_seconds'])} | "
            f"{format_ms(result['compiled_split_forward_seconds'])} | "
            f"{format_ms(result['stream_split_forward_seconds'])} | "
            f"{format_ms(result['checkpoint_split_forward_seconds'])} | "
            f"{format_ms(result['triton_split_forward_seconds'])} | "
            f"{format_ms(result['triton_stream_split_forward_seconds'])} | "
            f"{format_ms(result['triton_checkpoint_split_forward_seconds'])} | "
            f"{format_ms(result['eager_backward_seconds'])} | "
            f"{format_ms(result['compiled_backward_seconds'])} | "
            f"{format_ms(result['stream_backward_seconds'])} | "
            f"{format_ms(result['checkpoint_backward_seconds'])} | "
            f"{format_ms(result['triton_backward_seconds'])} | "
            f"{format_ms(result['triton_stream_backward_seconds'])} | "
            f"{format_ms(result['triton_checkpoint_backward_seconds'])} | "
            f"{format_ms(result['eager_forward_backward_seconds'])} | "
            f"{format_ms(result['compiled_forward_backward_seconds'])} | "
            f"{format_ms(result['stream_forward_backward_seconds'])} | "
            f"{format_ms(result['checkpoint_forward_backward_seconds'])} | "
            f"{format_ms(result['triton_forward_backward_seconds'])} | "
            f"{format_ms(result['triton_stream_forward_backward_seconds'])} | "
            f"{format_ms(result['triton_checkpoint_forward_backward_seconds'])} | "
            f"{format_speedup(result['compiled_forward_backward_speedup'])} | "
            f"{format_speedup(result['stream_forward_backward_speedup'])} | "
            f"{format_speedup(result['checkpoint_forward_backward_speedup'])} | "
            f"{format_speedup(result['triton_forward_backward_speedup'])} | "
            f"{format_speedup(result['triton_stream_forward_backward_speedup'])} | "
            f"{format_speedup(result['triton_checkpoint_forward_backward_speedup'])} | "
            f"{format_ms(result['compile_warmup_seconds'])} | "
            f"{result['compile_error'] or ''} | "
            f"{result['triton_error'] or ''} |"
            f" {result['triton_stream_error'] or ''} |"
            f" {result['triton_checkpoint_error'] or ''} |"
        )
    print()
    print("### Generated Triton Comparison")
    print()
    print(
        "| T | Batch | Features | N | Handwritten LIF Fwd+Bwd ms | "
        "Generated LIF Fwd+Bwd ms | Generated LIF vs Handwritten | "
        "Handwritten Synapse Fwd+Bwd ms | Generated Synapse Fwd+Bwd ms | "
        "Generated Synapse vs Handwritten | Handwritten Checkpoint Fwd+Bwd ms | "
        "Generated Checkpoint Fwd+Bwd ms | Generated Checkpoint vs Handwritten | "
        "Generated LIF Error | Generated Synapse Error | Generated Checkpoint Error |"
    )
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|")
    for result in results:
        generated_lif_speedup = optional_ratio(
            result["triton_forward_backward_seconds"],
            result["generated_triton_forward_backward_seconds"],
        )
        generated_synapse_speedup = optional_ratio(
            result["triton_stream_forward_backward_seconds"],
            result["generated_triton_stream_forward_backward_seconds"],
        )
        generated_checkpoint_speedup = optional_ratio(
            result["triton_checkpoint_forward_backward_seconds"],
            result["generated_triton_checkpoint_forward_backward_seconds"],
        )
        print(
            f"| {result['timesteps']} | {result['batch']} | {result['features']} | "
            f"{result['neurons']} | "
            f"{format_ms(result['triton_forward_backward_seconds'])} | "
            f"{format_ms(result['generated_triton_forward_backward_seconds'])} | "
            f"{format_speedup(generated_lif_speedup)} | "
            f"{format_ms(result['triton_stream_forward_backward_seconds'])} | "
            f"{format_ms(result['generated_triton_stream_forward_backward_seconds'])} | "
            f"{format_speedup(generated_synapse_speedup)} | "
            f"{format_ms(result['triton_checkpoint_forward_backward_seconds'])} | "
            f"{format_ms(result['generated_triton_checkpoint_forward_backward_seconds'])} | "
            f"{format_speedup(generated_checkpoint_speedup)} | "
            f"{result['generated_triton_error'] or ''} | "
            f"{result['generated_triton_stream_error'] or ''} | "
            f"{result['generated_triton_checkpoint_error'] or ''} |"
        )
    print()
    print("### CUDA Memory")
    print()
    print(
        "| T | Batch | Features | N | Eager Fwd Peak MB | Compiled Fwd Peak MB | "
        "Stream Fwd Peak MB | Checkpoint Fwd Peak MB | Triton LIF Fwd Peak MB | "
        "Triton Synapse Fwd Peak MB | Triton Checkpoint Fwd Peak MB | "
        "Eager Bwd Peak MB | Compiled Bwd Peak MB | Stream Bwd Peak MB | "
        "Checkpoint Bwd Peak MB | Triton LIF Bwd Peak MB | Triton Synapse Bwd Peak MB | "
        "Triton Checkpoint Bwd Peak MB |"
    )
    print(
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    for result in results:
        print(
            f"| {result['timesteps']} | {result['batch']} | {result['features']} | "
            f"{result['neurons']} | "
            f"{format_memory(result['eager_forward_peak_bytes'])} | "
            f"{format_memory(result['compiled_forward_peak_bytes'])} | "
            f"{format_memory(result['stream_forward_peak_bytes'])} | "
            f"{format_memory(result['checkpoint_forward_peak_bytes'])} | "
            f"{format_memory(result['triton_forward_peak_bytes'])} | "
            f"{format_memory(result['triton_stream_forward_peak_bytes'])} | "
            f"{format_memory(result['triton_checkpoint_forward_peak_bytes'])} | "
            f"{format_memory(result['eager_backward_peak_bytes'])} | "
            f"{format_memory(result['compiled_backward_peak_bytes'])} | "
            f"{format_memory(result['stream_backward_peak_bytes'])} | "
            f"{format_memory(result['checkpoint_backward_peak_bytes'])} | "
            f"{format_memory(result['triton_backward_peak_bytes'])} |"
            f" {format_memory(result['triton_stream_backward_peak_bytes'])} |"
            f" {format_memory(result['triton_checkpoint_backward_peak_bytes'])} |"
        )
    print()
    print("### Generated Triton CUDA Memory")
    print()
    print(
        "| T | Batch | Features | N | Handwritten LIF Bwd Peak MB | "
        "Generated LIF Bwd Peak MB | Handwritten Synapse Bwd Peak MB | "
        "Generated Synapse Bwd Peak MB | Handwritten Checkpoint Bwd Peak MB | "
        "Generated Checkpoint Bwd Peak MB |"
    )
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for result in results:
        print(
            f"| {result['timesteps']} | {result['batch']} | {result['features']} | "
            f"{result['neurons']} | "
            f"{format_memory(result['triton_backward_peak_bytes'])} | "
            f"{format_memory(result['generated_triton_backward_peak_bytes'])} | "
            f"{format_memory(result['triton_stream_backward_peak_bytes'])} | "
            f"{format_memory(result['generated_triton_stream_backward_peak_bytes'])} | "
            f"{format_memory(result['triton_checkpoint_backward_peak_bytes'])} | "
            f"{format_memory(result['generated_triton_checkpoint_backward_peak_bytes'])} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--features", type=int, default=128)
    parser.add_argument("--neurons", type=int, default=2048)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--shape", action="append", type=parse_shape, default=[])
    parser.add_argument("--format", choices=["md"], default="md")
    parser.add_argument(
        "--compile-mode",
        choices=["default", "reduce-overhead", "max-autotune"],
        default="reduce-overhead",
    )
    parser.add_argument(
        "--surrogate",
        choices=SURROGATE_NAMES,
        default="fast_sigmoid",
    )
    parser.add_argument("--surrogate-slope", type=float, default=5.0)
    parser.add_argument("--checkpoint-size", type=parse_checkpoint_size, default=25)
    parser.add_argument(
        "--input-grad",
        action="store_true",
        help="Require gradients for inputs as well as weights.",
    )
    parser.add_argument(
        "--matmul-precision",
        choices=["current", "highest", "high", "medium"],
        default="current",
    )
    args = parser.parse_args()

    original_matmul_precision = torch.get_float32_matmul_precision()
    if args.matmul_precision != "current":
        torch.set_float32_matmul_precision(args.matmul_precision)
    warnings.filterwarnings(
        "ignore",
        message="CUDA inputs detected and Triton is available.*backend='torch'.*",
        category=RuntimeWarning,
    )

    use_compile = (not args.no_compile) and compiled_available()
    shapes = args.shape or (
        DEFAULT_SHAPES if args.sweep else [(args.timesteps, args.batch, args.neurons)]
    )
    results = run_sweep(
        shapes,
        features=args.features,
        device=args.device,
        warmup=args.warmup,
        repeats=args.repeats,
        compile_enabled=use_compile,
        compile_mode=args.compile_mode,
        surrogate=args.surrogate,  # pyright: ignore[reportArgumentType]
        surrogate_slope=args.surrogate_slope,
        checkpoint_size=args.checkpoint_size,
        input_grad=args.input_grad,
    )
    print_markdown(results, environment_metadata(args, use_compile, shapes))

    if args.matmul_precision != "current":
        torch.set_float32_matmul_precision(original_matmul_precision)


if __name__ == "__main__":
    main()
