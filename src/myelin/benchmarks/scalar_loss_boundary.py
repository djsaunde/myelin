"""Compare scalar-loss compiled PyTorch and Triton checkpoint boundaries."""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime

import torch

from myelin._optional import has_triton
from myelin.baselines import compiled_available, synchronize_if_needed
from myelin.benchmarks.checkpoint_rate import dense_checkpoint_step, rate_checkpoint_step
from myelin.benchmarks.lif import format_memory, format_ms, gpu_name, parse_shape
from myelin.checkpointing import CheckpointSize, parse_checkpoint_size, resolve_checkpoint_size
from myelin.neurons import LIFParams
from myelin.surrogates import SURROGATE_NAMES, SurrogateName
from myelin.workloads import (
    dense_fast_surrogate_lif_spike_loss,
    dense_hard_fast_surrogate_lif_spike_loss,
    looped_fast_surrogate_lif_spike_loss,
    looped_hard_fast_surrogate_lif_spike_loss,
)

TrainStep = Callable[[torch.Tensor, torch.Tensor, LIFParams], torch.Tensor]
LossStep = Callable[[], torch.Tensor]
ManualStep = Callable[[], None]

DEFAULT_SHAPES = [(100, 64, 2048)]


@dataclass(frozen=True)
class ScalarLossResult:
    variant: str
    forward_backward_seconds: float | None
    allocated_bytes: int | None
    peak_bytes: int | None
    compile_warmup_seconds: float | None = None
    error: str | None = None

    @property
    def increment_bytes(self) -> int | None:
        if self.allocated_bytes is None or self.peak_bytes is None:
            return None
        return self.peak_bytes - self.allocated_bytes


def make_inputs(
    timesteps: int,
    batch: int,
    features: int,
    neurons: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = torch.rand((timesteps, batch, features), device=device)
    weight = (torch.rand((features, neurons), device=device) - 0.5) * 0.02
    weight.requires_grad_(True)
    return inputs, weight


def compile_train_step(fn: TrainStep, compile_mode: str) -> TrainStep:
    if compile_mode == "default":
        return torch.compile(fn, fullgraph=True)
    return torch.compile(fn, mode=compile_mode, fullgraph=True)


def clear_gradients(tensors: Iterable[torch.Tensor]) -> None:
    for tensor in tensors:
        tensor.grad = None


def time_loss_step(
    fn: LossStep,
    grad_tensors: Iterable[torch.Tensor],
    device: torch.device,
    *,
    warmup: int,
    repeats: int,
) -> float:
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
    return elapsed / repeats


def memory_loss_step(
    fn: LossStep,
    grad_tensors: Iterable[torch.Tensor],
    device: torch.device,
) -> tuple[int | None, int | None]:
    grad_tensors = tuple(grad_tensors)
    if device.type != "cuda":
        return None, None

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    allocated = torch.cuda.memory_allocated(device)
    clear_gradients(grad_tensors)
    loss = fn()
    loss.backward()
    synchronize_if_needed(device)
    return allocated, torch.cuda.max_memory_allocated(device)


def benchmark_loss_step(
    variant: str,
    fn: LossStep,
    grad_tensors: Iterable[torch.Tensor],
    device: torch.device,
    *,
    warmup: int,
    repeats: int,
    compile_warmup_seconds: float | None = None,
) -> ScalarLossResult:
    try:
        forward_backward_seconds = time_loss_step(
            fn,
            grad_tensors,
            device,
            warmup=warmup,
            repeats=repeats,
        )
        allocated_bytes, peak_bytes = memory_loss_step(fn, grad_tensors, device)
    except Exception as exc:  # noqa: BLE001 - benchmark should report backend failures.
        synchronize_if_needed(device)
        return ScalarLossResult(
            variant=variant,
            forward_backward_seconds=None,
            allocated_bytes=None,
            peak_bytes=None,
            compile_warmup_seconds=compile_warmup_seconds,
            error=f"{type(exc).__name__}: {exc}",
        )
    return ScalarLossResult(
        variant=variant,
        forward_backward_seconds=forward_backward_seconds,
        allocated_bytes=allocated_bytes,
        peak_bytes=peak_bytes,
        compile_warmup_seconds=compile_warmup_seconds,
    )


def time_manual_step(
    fn: ManualStep,
    grad_tensors: Iterable[torch.Tensor],
    device: torch.device,
    *,
    warmup: int,
    repeats: int,
) -> float:
    grad_tensors = tuple(grad_tensors)
    for _ in range(warmup):
        clear_gradients(grad_tensors)
        fn()
    synchronize_if_needed(device)

    elapsed = 0.0
    for _ in range(repeats):
        clear_gradients(grad_tensors)
        start = time.perf_counter()
        fn()
        synchronize_if_needed(device)
        elapsed += time.perf_counter() - start
    return elapsed / repeats


def memory_manual_step(
    fn: ManualStep,
    grad_tensors: Iterable[torch.Tensor],
    device: torch.device,
) -> tuple[int | None, int | None]:
    grad_tensors = tuple(grad_tensors)
    if device.type != "cuda":
        return None, None

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    allocated = torch.cuda.memory_allocated(device)
    clear_gradients(grad_tensors)
    fn()
    synchronize_if_needed(device)
    return allocated, torch.cuda.max_memory_allocated(device)


def benchmark_manual_step(
    variant: str,
    fn: ManualStep,
    grad_tensors: Iterable[torch.Tensor],
    device: torch.device,
    *,
    warmup: int,
    repeats: int,
) -> ScalarLossResult:
    try:
        forward_backward_seconds = time_manual_step(
            fn,
            grad_tensors,
            device,
            warmup=warmup,
            repeats=repeats,
        )
        allocated_bytes, peak_bytes = memory_manual_step(fn, grad_tensors, device)
    except Exception as exc:  # noqa: BLE001 - benchmark should report backend failures.
        synchronize_if_needed(device)
        return ScalarLossResult(
            variant=variant,
            forward_backward_seconds=None,
            allocated_bytes=None,
            peak_bytes=None,
            error=f"{type(exc).__name__}: {exc}",
        )
    return ScalarLossResult(
        variant=variant,
        forward_backward_seconds=forward_backward_seconds,
        allocated_bytes=allocated_bytes,
        peak_bytes=peak_bytes,
    )


def compiled_loss_step(
    variant: str,
    fn: TrainStep,
    inputs: torch.Tensor,
    weight: torch.Tensor,
    params: LIFParams,
    *,
    compile_mode: str,
    warmup: int,
    repeats: int,
) -> ScalarLossResult:
    if not compiled_available():
        return ScalarLossResult(
            variant=variant,
            forward_backward_seconds=None,
            allocated_bytes=None,
            peak_bytes=None,
            error="torch.compile is unavailable",
        )
    try:
        compiled = compile_train_step(fn, compile_mode)
        start = time.perf_counter()
        weight.grad = None
        loss = compiled(inputs, weight, params)
        loss.backward()
        synchronize_if_needed(inputs.device)
        compile_warmup_seconds = time.perf_counter() - start
    except Exception as exc:  # noqa: BLE001 - benchmark should report compiler failures.
        synchronize_if_needed(inputs.device)
        return ScalarLossResult(
            variant=variant,
            forward_backward_seconds=None,
            allocated_bytes=None,
            peak_bytes=None,
            error=f"{type(exc).__name__}: {exc}",
        )

    return benchmark_loss_step(
        variant,
        lambda: compiled(inputs, weight, params),
        (weight,),
        inputs.device,
        warmup=warmup,
        repeats=repeats,
        compile_warmup_seconds=compile_warmup_seconds,
    )


def run_one(
    *,
    timesteps: int,
    batch: int,
    features: int,
    neurons: int,
    device: str,
    warmup: int,
    repeats: int,
    compile_mode: str,
    surrogate: SurrogateName,
    surrogate_slope: float,
    checkpoint_size: CheckpointSize,
) -> list[ScalarLossResult]:
    inputs, base_weight = make_inputs(timesteps, batch, features, neurons, device)
    params = LIFParams()
    chunk = resolve_checkpoint_size(timesteps, checkpoint_size)
    results: list[ScalarLossResult] = []

    materialized_weight = base_weight.detach().clone().requires_grad_(True)
    results.append(
        compiled_loss_step(
            "torch.compile materialized soft-surrogate scalar loss",
            dense_fast_surrogate_lif_spike_loss,
            inputs,
            materialized_weight,
            params,
            compile_mode=compile_mode,
            warmup=warmup,
            repeats=repeats,
        )
    )

    looped_weight = base_weight.detach().clone().requires_grad_(True)
    results.append(
        compiled_loss_step(
            "torch.compile looped soft-surrogate scalar loss",
            looped_fast_surrogate_lif_spike_loss,
            inputs,
            looped_weight,
            params,
            compile_mode=compile_mode,
            warmup=warmup,
            repeats=repeats,
        )
    )

    hard_materialized_weight = base_weight.detach().clone().requires_grad_(True)
    results.append(
        compiled_loss_step(
            "torch.compile materialized hard-forward scalar loss",
            dense_hard_fast_surrogate_lif_spike_loss,
            inputs,
            hard_materialized_weight,
            params,
            compile_mode=compile_mode,
            warmup=warmup,
            repeats=repeats,
        )
    )

    hard_looped_weight = base_weight.detach().clone().requires_grad_(True)
    results.append(
        compiled_loss_step(
            "torch.compile looped hard-forward scalar loss",
            looped_hard_fast_surrogate_lif_spike_loss,
            inputs,
            hard_looped_weight,
            params,
            compile_mode=compile_mode,
            warmup=warmup,
            repeats=repeats,
        )
    )

    dense_weight = base_weight.detach().clone().requires_grad_(True)
    results.append(
        benchmark_loss_step(
            "Triton checkpoint dense-spike scalar loss",
            lambda: dense_checkpoint_step(
                inputs,
                dense_weight,
                params,
                surrogate=surrogate,
                surrogate_slope=surrogate_slope,
                checkpoint_size=chunk,
            ),
            (dense_weight,),
            inputs.device,
            warmup=warmup,
            repeats=repeats,
        )
    )

    rate_weight = base_weight.detach().clone().requires_grad_(True)
    results.append(
        benchmark_loss_step(
            "Triton checkpoint scalar-rate loss",
            lambda: rate_checkpoint_step(
                inputs,
                rate_weight,
                params,
                surrogate=surrogate,
                surrogate_slope=surrogate_slope,
                checkpoint_size=chunk,
            ),
            (rate_weight,),
            inputs.device,
            warmup=warmup,
            repeats=repeats,
        )
    )

    generated_rate_weight = base_weight.detach().clone().requires_grad_(True)
    if torch.device(device).type == "cuda" and has_triton():
        from myelin.kernels import linear_surrogate_lif_rate_forward
        from myelin.triton import (
            linear_surrogate_lif_checkpoint_backward_replay_weight_bias,
            linear_surrogate_lif_checkpoint_rate_forward,
        )

        def generated_rate_loss() -> torch.Tensor:
            final_state, spike_rate = linear_surrogate_lif_rate_forward(
                inputs,
                generated_rate_weight,
                None,
                params,
                surrogate=surrogate,
                surrogate_slope=surrogate_slope,
                backend="triton_generated",
                checkpoint_size=chunk,
                reduction="mean",
            )
            return spike_rate.square() + 0.01 * final_state.membrane.square().mean()

        results.append(
            benchmark_loss_step(
                "Generated Triton checkpoint scalar-rate loss",
                generated_rate_loss,
                (generated_rate_weight,),
                inputs.device,
                warmup=warmup,
                repeats=repeats,
            )
        )

        replay_weight = base_weight.detach().clone().requires_grad_(True)

        def replay_rate_step() -> None:
            final_state, spike_rates, chunk_start_membranes = (
                linear_surrogate_lif_checkpoint_rate_forward(
                    inputs,
                    replay_weight,
                    None,
                    params,
                    checkpoint_size=chunk,
                )
            )
            spike_rate = spike_rates.mean()
            grad_final = 0.02 * final_state.membrane / final_state.membrane.numel()
            grad_rate = 2.0 * spike_rate
            grad_weight, _grad_bias = linear_surrogate_lif_checkpoint_backward_replay_weight_bias(
                inputs,
                replay_weight,
                None,
                chunk_start_membranes,
                grad_final,
                None,
                params,
                surrogate=surrogate,
                surrogate_slope=surrogate_slope,
                grad_spike_rate=grad_rate,
                needs_bias_grad=False,
                checkpoint_size=chunk,
            )
            replay_weight.grad = grad_weight

        results.append(
            benchmark_manual_step(
                "Triton checkpoint scalar-rate replay-no-scratch loss",
                replay_rate_step,
                (replay_weight,),
                inputs.device,
                warmup=warmup,
                repeats=repeats,
            )
        )
    else:
        results.append(
            ScalarLossResult(
                variant="Generated Triton checkpoint scalar-rate loss",
                forward_backward_seconds=None,
                allocated_bytes=None,
                peak_bytes=None,
                error="requires CUDA tensors and Triton",
            )
        )
        results.append(
            ScalarLossResult(
                variant="Triton checkpoint scalar-rate replay-no-scratch loss",
                forward_backward_seconds=None,
                allocated_bytes=None,
                peak_bytes=None,
                error="requires CUDA tensors and Triton",
            )
        )
    return results


def run_sweep(args: argparse.Namespace) -> list[tuple[tuple[int, int, int, int], ScalarLossResult]]:
    rows: list[tuple[tuple[int, int, int, int], ScalarLossResult]] = []
    for timesteps, batch, neurons in args.shape or DEFAULT_SHAPES:
        for result in run_one(
            timesteps=timesteps,
            batch=batch,
            features=args.features,
            neurons=neurons,
            device=args.device,
            warmup=args.warmup,
            repeats=args.repeats,
            compile_mode=args.compile_mode,
            surrogate=args.surrogate,
            surrogate_slope=args.surrogate_slope,
            checkpoint_size=args.checkpoint_size,
        ):
            rows.append(((timesteps, batch, args.features, neurons), result))
    return rows


def print_markdown(
    args: argparse.Namespace,
    rows: list[tuple[tuple[int, int, int, int], ScalarLossResult]],
) -> None:
    print("# Scalar Loss Boundary Benchmark")
    print()
    print("Compares scalar-loss PyTorch graphs captured by `torch.compile` against")
    print("Triton checkpoint paths that keep spike-rate objectives out of the public")
    print("dense `[T, B, N]` spike-output contract.")
    print()
    print("Rows marked hard-forward use thresholded spikes in forward with")
    print("fast-sigmoid straight-through gradients, matching the Triton surrogate")
    print("contract more closely than the soft-forward rows.")
    print()
    print("## Environment")
    print()
    print(f"- `generated_utc`: `{datetime.now(UTC).isoformat(timespec='seconds')}`")
    print(f"- `device`: `{args.device}`")
    print(f"- `gpu`: `{gpu_name(args.device)}`")
    print(f"- `torch`: `{torch.__version__}`")
    print(f"- `cuda_available`: `{torch.cuda.is_available()}`")
    print(f"- `cuda_version`: `{torch.version.cuda}`")
    print(f"- `features`: `{args.features}`")
    print(f"- `checkpoint_size`: `{args.checkpoint_size}`")
    resolved_sizes = {
        timesteps: resolve_checkpoint_size(timesteps, args.checkpoint_size)
        for timesteps, _batch, _neurons in (args.shape or DEFAULT_SHAPES)
    }
    print(f"- `resolved_checkpoint_sizes`: `{resolved_sizes}`")
    print(f"- `surrogate`: `{args.surrogate}`")
    print(f"- `surrogate_slope`: `{args.surrogate_slope}`")
    print(f"- `compile_mode`: `{args.compile_mode}`")
    print(f"- `matmul_precision`: `{torch.get_float32_matmul_precision()}`")
    print(f"- `warmup`: `{args.warmup}`")
    print(f"- `repeats`: `{args.repeats}`")
    print()
    print("## Results")
    print()
    print(
        "| T | Batch | Features | N | Variant | Fwd+Bwd ms | Baseline Alloc MB | "
        "Peak MB | Increment MB | Compile Warmup ms | Error |"
    )
    print("|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---|")
    for shape, result in rows:
        timesteps, batch, features, neurons = shape
        print(
            f"| {timesteps} | {batch} | {features} | {neurons} | "
            f"{result.variant} | "
            f"{format_ms(result.forward_backward_seconds)} | "
            f"{format_memory(result.allocated_bytes)} | "
            f"{format_memory(result.peak_bytes)} | "
            f"{format_memory(result.increment_bytes)} | "
            f"{format_ms(result.compile_warmup_seconds)} | "
            f"{result.error or ''} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", action="append", type=parse_shape, default=[])
    parser.add_argument("--features", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune"),
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
        "--matmul-precision",
        choices=("current", "highest", "high", "medium"),
        default="current",
    )
    args = parser.parse_args()
    original_matmul_precision = torch.get_float32_matmul_precision()
    if args.matmul_precision != "current":
        torch.set_float32_matmul_precision(args.matmul_precision)
    try:
        print_markdown(args, run_sweep(args))
    finally:
        if args.matmul_precision != "current":
            torch.set_float32_matmul_precision(original_matmul_precision)


if __name__ == "__main__":
    main()
