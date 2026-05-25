"""Benchmark dense spike traces vs direct spike-rate classifier readout."""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime

import torch
import torch.nn.functional as F

from spiker._optional import has_triton
from spiker.benchmarks.lif import format_memory, format_ms, gpu_name, parse_shape
from spiker.checkpointing import CheckpointSize, parse_checkpoint_size, resolve_checkpoint_size
from spiker.kernels import Backend, linear_surrogate_lif_forward, linear_surrogate_lif_rate_forward
from spiker.modules import LinearSurrogateLIFRate
from spiker.neurons import LIFParams
from spiker.surrogates import SURROGATE_NAMES, SurrogateName

StepFn = Callable[[], torch.Tensor]

DEFAULT_SHAPES = [
    (100, 128, 10),
    (100, 128, 1000),
    (100, 128, 2048),
]


@dataclass(frozen=True)
class Timing:
    forward_backward_seconds: float


@dataclass(frozen=True)
class Memory:
    backward_peak_bytes: int | None


@dataclass(frozen=True)
class OptionalResult:
    timing: Timing | None
    memory: Memory
    error: str | None = None


def synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def clear_gradients(tensors: Iterable[torch.Tensor]) -> None:
    for tensor in tensors:
        tensor.grad = None


def time_step(
    fn: StepFn,
    grad_tensors: Iterable[torch.Tensor],
    device: torch.device,
    *,
    warmup: int,
    repeats: int,
) -> Timing:
    grad_tensors = tuple(grad_tensors)
    for _ in range(warmup):
        clear_gradients(grad_tensors)
        loss = fn()
        loss.backward()
    synchronize_if_needed(device)

    elapsed = 0.0
    for _ in range(repeats):
        clear_gradients(grad_tensors)
        start = time.perf_counter()
        loss = fn()
        loss.backward()
        synchronize_if_needed(device)
        elapsed += time.perf_counter() - start

    return Timing(forward_backward_seconds=elapsed / repeats)


def memory_checkpoints(
    fn: StepFn,
    grad_tensors: Iterable[torch.Tensor],
    device: torch.device,
) -> Memory:
    grad_tensors = tuple(grad_tensors)
    if device.type != "cuda":
        return Memory(None)

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    clear_gradients(grad_tensors)
    loss = fn()
    loss.backward()
    synchronize_if_needed(device)
    return Memory(torch.cuda.max_memory_allocated(device))


def benchmark_optional(
    fn: StepFn,
    grad_tensors: Iterable[torch.Tensor],
    device: torch.device,
    *,
    warmup: int,
    repeats: int,
) -> OptionalResult:
    try:
        timing = time_step(fn, grad_tensors, device, warmup=warmup, repeats=repeats)
        memory = memory_checkpoints(fn, grad_tensors, device)
    except Exception as exc:  # noqa: BLE001 - benchmark should report backend failures.
        synchronize_if_needed(device)
        return OptionalResult(
            timing=None, memory=Memory(None), error=f"{type(exc).__name__}: {exc}"
        )
    return OptionalResult(timing=timing, memory=memory)


def make_inputs(
    timesteps: int,
    batch: int,
    features: int,
    classes: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    inputs = (torch.rand((timesteps, batch, features), device=device) < 0.1).to(torch.float32)
    weight = (torch.rand((features, classes), device=device) - 0.5) * 0.02
    weight.requires_grad_(True)
    bias = torch.zeros(classes, device=device, requires_grad=True)
    targets = torch.randint(classes, (batch,), device=device)
    return inputs, weight, bias, targets


def dense_readout_step(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    targets: torch.Tensor,
    params: LIFParams,
    *,
    surrogate: SurrogateName,
    surrogate_slope: float,
    checkpoint_size: CheckpointSize,
    backend: Backend,
) -> torch.Tensor:
    _final_state, spikes = linear_surrogate_lif_forward(
        inputs,
        weight,
        bias,
        params,
        surrogate=surrogate,
        surrogate_slope=surrogate_slope,
        backend=backend,
        checkpoint_size=checkpoint_size,
    )
    logits = spikes.mean(dim=0)
    return F.cross_entropy(logits, targets)


def rate_readout_step(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    targets: torch.Tensor,
    params: LIFParams,
    *,
    surrogate: SurrogateName,
    surrogate_slope: float,
    checkpoint_size: CheckpointSize,
    backend: Backend,
) -> torch.Tensor:
    _final_state, logits = linear_surrogate_lif_rate_forward(
        inputs,
        weight,
        bias,
        params,
        surrogate=surrogate,
        surrogate_slope=surrogate_slope,
        backend=backend,
        checkpoint_size=checkpoint_size,
        reduction="none",
    )
    return F.cross_entropy(logits, targets)


def benchmark_one(
    *,
    timesteps: int,
    batch: int,
    features: int,
    classes: int,
    device: str,
    warmup: int,
    repeats: int,
    surrogate: SurrogateName,
    surrogate_slope: float,
    checkpoint_size: CheckpointSize,
) -> dict[str, object]:
    inputs, weight, bias, targets = make_inputs(timesteps, batch, features, classes, device)
    backend: Backend = "triton" if torch.device(device).type == "cuda" else "torch"
    resolved_checkpoint_size = resolve_checkpoint_size(timesteps, checkpoint_size)
    dense_weight = weight.detach().clone().requires_grad_(True)
    dense_bias = bias.detach().clone().requires_grad_(True)
    rate_weight = weight.detach().clone().requires_grad_(True)
    rate_bias = bias.detach().clone().requires_grad_(True)
    generated_rate_weight = weight.detach().clone().requires_grad_(True)
    generated_rate_bias = bias.detach().clone().requires_grad_(True)
    params = LIFParams()
    module_rate = LinearSurrogateLIFRate(
        features,
        classes,
        params,
        surrogate_slope=surrogate_slope,
        backend=backend,
        checkpoint_size=resolved_checkpoint_size,
        reduction="none",
    ).to(device=device)
    module_rate.synapse.weight.data.copy_(weight)
    assert module_rate.synapse.bias is not None
    module_rate.synapse.bias.data.copy_(bias)
    generated_module_rate = LinearSurrogateLIFRate(
        features,
        classes,
        params,
        surrogate_slope=surrogate_slope,
        backend="triton_generated",
        checkpoint_size=resolved_checkpoint_size,
        reduction="none",
    ).to(device=device)
    generated_module_rate.synapse.weight.data.copy_(weight)
    assert generated_module_rate.synapse.bias is not None
    generated_module_rate.synapse.bias.data.copy_(bias)
    chunk = resolved_checkpoint_size

    def dense_step() -> torch.Tensor:
        return dense_readout_step(
            inputs,
            dense_weight,
            dense_bias,
            targets,
            params,
            surrogate=surrogate,
            surrogate_slope=surrogate_slope,
            checkpoint_size=chunk,
            backend=backend,
        )

    def rate_step() -> torch.Tensor:
        return rate_readout_step(
            inputs,
            rate_weight,
            rate_bias,
            targets,
            params,
            surrogate=surrogate,
            surrogate_slope=surrogate_slope,
            checkpoint_size=chunk,
            backend=backend,
        )

    def module_rate_step() -> torch.Tensor:
        logits = module_rate(inputs)
        return F.cross_entropy(logits, targets)

    def generated_rate_step() -> torch.Tensor:
        return rate_readout_step(
            inputs,
            generated_rate_weight,
            generated_rate_bias,
            targets,
            params,
            surrogate=surrogate,
            surrogate_slope=surrogate_slope,
            checkpoint_size=chunk,
            backend="triton_generated",
        )

    def generated_module_rate_step() -> torch.Tensor:
        logits = generated_module_rate(inputs)
        return F.cross_entropy(logits, targets)

    dense_timing = time_step(
        dense_step,
        (dense_weight, dense_bias),
        inputs.device,
        warmup=warmup,
        repeats=repeats,
    )
    dense_memory = memory_checkpoints(dense_step, (dense_weight, dense_bias), inputs.device)
    rate_timing = time_step(
        rate_step,
        (rate_weight, rate_bias),
        inputs.device,
        warmup=warmup,
        repeats=repeats,
    )
    rate_memory = memory_checkpoints(rate_step, (rate_weight, rate_bias), inputs.device)
    module_rate_timing = time_step(
        module_rate_step,
        tuple(module_rate.parameters()),
        inputs.device,
        warmup=warmup,
        repeats=repeats,
    )
    module_rate_memory = memory_checkpoints(
        module_rate_step,
        tuple(module_rate.parameters()),
        inputs.device,
    )
    should_run_generated = torch.device(device).type == "cuda" and has_triton()
    if should_run_generated:
        generated_rate_result = benchmark_optional(
            generated_rate_step,
            (generated_rate_weight, generated_rate_bias),
            inputs.device,
            warmup=warmup,
            repeats=repeats,
        )
        generated_module_rate_result = benchmark_optional(
            generated_module_rate_step,
            tuple(generated_module_rate.parameters()),
            inputs.device,
            warmup=warmup,
            repeats=repeats,
        )
    else:
        generated_rate_result = OptionalResult(
            timing=None,
            memory=Memory(None),
            error="requires CUDA tensors and Triton",
        )
        generated_module_rate_result = OptionalResult(
            timing=None,
            memory=Memory(None),
            error="requires CUDA tensors and Triton",
        )

    return {
        "timesteps": timesteps,
        "batch": batch,
        "features": features,
        "classes": classes,
        "dense_forward_backward_seconds": dense_timing.forward_backward_seconds,
        "rate_forward_backward_seconds": rate_timing.forward_backward_seconds,
        "module_rate_forward_backward_seconds": module_rate_timing.forward_backward_seconds,
        "generated_rate_forward_backward_seconds": None
        if generated_rate_result.timing is None
        else generated_rate_result.timing.forward_backward_seconds,
        "generated_module_rate_forward_backward_seconds": None
        if generated_module_rate_result.timing is None
        else generated_module_rate_result.timing.forward_backward_seconds,
        "dense_backward_peak_bytes": dense_memory.backward_peak_bytes,
        "rate_backward_peak_bytes": rate_memory.backward_peak_bytes,
        "module_rate_backward_peak_bytes": module_rate_memory.backward_peak_bytes,
        "generated_rate_backward_peak_bytes": generated_rate_result.memory.backward_peak_bytes,
        "generated_module_rate_backward_peak_bytes": (
            generated_module_rate_result.memory.backward_peak_bytes
        ),
        "generated_rate_error": generated_rate_result.error,
        "generated_module_rate_error": generated_module_rate_result.error,
    }


def run_sweep(
    shapes: Iterable[tuple[int, int, int]],
    *,
    features: int,
    device: str,
    warmup: int,
    repeats: int,
    surrogate: SurrogateName,
    surrogate_slope: float,
    checkpoint_size: CheckpointSize,
) -> list[dict[str, object]]:
    return [
        benchmark_one(
            timesteps=timesteps,
            batch=batch,
            features=features,
            classes=classes,
            device=device,
            warmup=warmup,
            repeats=repeats,
            surrogate=surrogate,
            surrogate_slope=surrogate_slope,
            checkpoint_size=checkpoint_size,
        )
        for timesteps, batch, classes in shapes
    ]


def print_markdown(args: argparse.Namespace, results: list[dict[str, object]]) -> None:
    print("# Rate Readout Benchmark")
    print()
    print(
        "Compares a dense `[T, B, classes]` spike readout against direct `[B, classes]` "
        "spike rates for a classifier loss."
    )
    print()
    print("## Environment")
    print()
    print(f"- `generated_utc`: `{datetime.now(UTC).isoformat(timespec='seconds')}`")
    print(f"- `device`: `{args.device}`")
    print(f"- `gpu`: `{gpu_name(args.device)}`")
    print(f"- `torch`: `{torch.__version__}`")
    print(f"- `features`: `{args.features}`")
    print(f"- `checkpoint_size`: `{args.checkpoint_size}`")
    resolved_sizes = {
        timesteps: resolve_checkpoint_size(timesteps, args.checkpoint_size)
        for timesteps, _batch, _classes in (args.shape or DEFAULT_SHAPES)
    }
    print(f"- `resolved_checkpoint_sizes`: `{resolved_sizes}`")
    print(f"- `warmup`: `{args.warmup}`")
    print(f"- `repeats`: `{args.repeats}`")
    print()
    print("## Results")
    print()
    print(
        "| T | Batch | Features | Classes | Dense Fwd+Bwd ms | Rate Fwd+Bwd ms | "
        "Module Rate Fwd+Bwd ms | Generated Rate Fwd+Bwd ms | "
        "Generated Module Rate Fwd+Bwd ms | Dense Bwd Peak MB | Rate Bwd Peak MB | "
        "Module Rate Bwd Peak MB | Generated Rate Bwd Peak MB | "
        "Generated Module Rate Bwd Peak MB | Generated Errors |"
    )
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for result in results:
        generated_errors = "; ".join(
            error
            for error in (
                result["generated_rate_error"],
                result["generated_module_rate_error"],
            )
            if isinstance(error, str)
        )
        print(
            f"| {result['timesteps']} | {result['batch']} | {result['features']} | "
            f"{result['classes']} | "
            f"{format_ms(result['dense_forward_backward_seconds'])} | "
            f"{format_ms(result['rate_forward_backward_seconds'])} | "
            f"{format_ms(result['module_rate_forward_backward_seconds'])} | "
            f"{format_ms(result['generated_rate_forward_backward_seconds'])} | "
            f"{format_ms(result['generated_module_rate_forward_backward_seconds'])} | "
            f"{format_memory(result['dense_backward_peak_bytes'])} | "
            f"{format_memory(result['rate_backward_peak_bytes'])} |"
            f" {format_memory(result['module_rate_backward_peak_bytes'])} |"
            f" {format_memory(result['generated_rate_backward_peak_bytes'])} |"
            f" {format_memory(result['generated_module_rate_backward_peak_bytes'])} |"
            f" {generated_errors} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", action="append", type=parse_shape, default=[])
    parser.add_argument("--features", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument(
        "--surrogate",
        choices=SURROGATE_NAMES,
        default="fast_sigmoid",
    )
    parser.add_argument("--surrogate-slope", type=float, default=5.0)
    parser.add_argument("--checkpoint-size", type=parse_checkpoint_size, default=25)
    args = parser.parse_args()

    results = run_sweep(
        args.shape or DEFAULT_SHAPES,
        features=args.features,
        device=args.device,
        warmup=args.warmup,
        repeats=args.repeats,
        surrogate=args.surrogate,
        surrogate_slope=args.surrogate_slope,
        checkpoint_size=args.checkpoint_size,
    )
    print_markdown(args, results)


if __name__ == "__main__":
    main()
