"""Sweep regular torch.compile against Triton rate-training paths."""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime

import torch

from myelin._optional import has_triton
from myelin.baselines import compiled_available, synchronize_if_needed
from myelin.benchmarks.lif import format_memory, format_ms, gpu_name
from myelin.checkpointing import CheckpointSize, parse_checkpoint_size, resolve_checkpoint_size
from myelin.neurons import LIFParams, LIFState

DEFAULT_SHAPES = (
    (50, 64, 128, 1024),
    (100, 64, 128, 2048),
    (200, 64, 128, 2048),
)


@dataclass(frozen=True)
class SweepResult:
    timesteps: int
    batch: int
    features: int
    neurons: int
    path: str
    seconds: float | None
    peak_bytes: int | None
    compile_warmup_seconds: float | None
    error: str | None = None


LossFn = Callable[[], torch.Tensor]
StepFn = Callable[[], torch.Tensor]


def parse_shape(raw: str) -> tuple[int, int, int, int]:
    parts = raw.split(",")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("shape must be T,B,F,N")
    try:
        shape = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("shape values must be integers") from exc
    if any(value <= 0 for value in shape):
        raise argparse.ArgumentTypeError("shape values must be positive")
    return shape  # pyright: ignore[reportReturnType]


def make_tensors(
    *,
    timesteps: int,
    batch: int,
    features: int,
    neurons: int,
    device: str,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    inputs = torch.rand((timesteps, batch, features), device=device, generator=generator)
    weight = (torch.rand((features, neurons), device=device, generator=generator) - 0.5) * 0.02
    return inputs, weight


def clear_gradients(tensors: Iterable[torch.Tensor]) -> None:
    for tensor in tensors:
        tensor.grad = None


def time_step(
    fn: StepFn,
    grad_tensors: Iterable[torch.Tensor],
    *,
    device: torch.device,
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


def memory_step(
    fn: StepFn,
    grad_tensors: Iterable[torch.Tensor],
    *,
    device: torch.device,
) -> int | None:
    if device.type != "cuda":
        return None
    grad_tensors = tuple(grad_tensors)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    clear_gradients(grad_tensors)
    loss = fn()
    loss.backward()
    synchronize_if_needed(device)
    return torch.cuda.max_memory_allocated(device)


def compile_loss_fn(fn: LossFn, compile_mode: str) -> tuple[LossFn, float | None]:
    if not compiled_available():
        return fn, None
    compiled = torch.compile(fn, mode=compile_mode, fullgraph=True)
    start = time.perf_counter()
    loss = compiled()
    loss.backward()
    synchronize_if_needed(torch.device(loss.device))
    return compiled, time.perf_counter() - start


def make_regular_compile_loss(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    params: LIFParams,
) -> LossFn:
    from myelin.kernels import surrogate_lif_forward

    def loss_fn() -> torch.Tensor:
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
            surrogate="fast_sigmoid",
            surrogate_slope=5.0,
            hard_forward=True,
            backend="torch",
        )
        return spikes.mean().square() + 0.01 * final_state.membrane.square().mean()

    return loss_fn


def make_rate_loss(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    params: LIFParams,
    *,
    backend: str,
    checkpoint_size: int,
) -> LossFn:
    from myelin.kernels import linear_surrogate_lif_rate_forward

    def loss_fn() -> torch.Tensor:
        final_state, rate = linear_surrogate_lif_rate_forward(
            inputs,
            weight,
            None,
            params,
            surrogate="fast_sigmoid",
            surrogate_slope=5.0,
            hard_forward=True,
            backend=backend,  # pyright: ignore[reportArgumentType]
            checkpoint_size=checkpoint_size,
            reduction="mean",
        )
        return rate.square() + 0.01 * final_state.membrane.square().mean()

    return loss_fn


def benchmark_path(
    *,
    shape: tuple[int, int, int, int],
    path: str,
    loss_fn: LossFn,
    grad_tensors: Iterable[torch.Tensor],
    device: torch.device,
    warmup: int,
    repeats: int,
    compile_mode: str | None = None,
) -> SweepResult:
    timesteps, batch, features, neurons = shape
    try:
        compile_warmup_seconds = None
        step_fn = loss_fn
        if compile_mode is not None:
            step_fn, compile_warmup_seconds = compile_loss_fn(loss_fn, compile_mode)
        seconds = time_step(
            step_fn,
            grad_tensors,
            device=device,
            warmup=warmup,
            repeats=repeats,
        )
        peak_bytes = memory_step(step_fn, grad_tensors, device=device)
        return SweepResult(
            timesteps,
            batch,
            features,
            neurons,
            path,
            seconds,
            peak_bytes,
            compile_warmup_seconds,
        )
    except Exception as exc:  # noqa: BLE001 - benchmark should report backend failures.
        synchronize_if_needed(device)
        return SweepResult(
            timesteps,
            batch,
            features,
            neurons,
            path,
            None,
            None,
            None,
            f"{type(exc).__name__}: {exc}",
        )


def run_shape(
    shape: tuple[int, int, int, int],
    *,
    device: str,
    seed: int,
    warmup: int,
    repeats: int,
    checkpoint_size: CheckpointSize,
    compile_mode: str,
) -> list[SweepResult]:
    torch_device = torch.device(device)
    timesteps, batch, features, neurons = shape
    inputs, base_weight = make_tensors(
        timesteps=timesteps,
        batch=batch,
        features=features,
        neurons=neurons,
        device=device,
        seed=seed,
    )
    resolved_checkpoint_size = resolve_checkpoint_size(timesteps, checkpoint_size)
    params = LIFParams()
    results: list[SweepResult] = []

    regular_weight = base_weight.detach().clone().requires_grad_(True)
    results.append(
        benchmark_path(
            shape=shape,
            path="regular torch.compile scalar-loss training",
            loss_fn=make_regular_compile_loss(inputs, regular_weight, params),
            grad_tensors=(regular_weight,),
            device=torch_device,
            warmup=warmup,
            repeats=repeats,
            compile_mode=compile_mode,
        )
    )

    if torch_device.type != "cuda" or not has_triton():
        for path in (
            "existing Triton rate training",
            'torch.compile public backend="triton_compile"',
        ):
            results.append(
                SweepResult(
                    timesteps,
                    batch,
                    features,
                    neurons,
                    path,
                    None,
                    None,
                    None,
                    "CUDA and Triton are required",
                )
            )
        return results

    triton_weight = base_weight.detach().clone().requires_grad_(True)
    results.append(
        benchmark_path(
            shape=shape,
            path="existing Triton rate training",
            loss_fn=make_rate_loss(
                inputs,
                triton_weight,
                params,
                backend="triton",
                checkpoint_size=resolved_checkpoint_size,
            ),
            grad_tensors=(triton_weight,),
            device=torch_device,
            warmup=warmup,
            repeats=repeats,
        )
    )

    triton_compile_weight = base_weight.detach().clone().requires_grad_(True)
    results.append(
        benchmark_path(
            shape=shape,
            path='torch.compile public backend="triton_compile"',
            loss_fn=make_rate_loss(
                inputs,
                triton_compile_weight,
                params,
                backend="triton_compile",
                checkpoint_size=resolved_checkpoint_size,
            ),
            grad_tensors=(triton_compile_weight,),
            device=torch_device,
            warmup=warmup,
            repeats=repeats,
            compile_mode=compile_mode,
        )
    )
    return results


def run_benchmark(args: argparse.Namespace) -> list[SweepResult]:
    if args.matmul_precision is not None:
        torch.set_float32_matmul_precision(args.matmul_precision)
    shapes = tuple(args.shape or DEFAULT_SHAPES)
    results: list[SweepResult] = []
    for index, shape in enumerate(shapes):
        results.extend(
            run_shape(
                shape,
                device=args.device,
                seed=args.seed + index,
                warmup=args.warmup,
                repeats=args.repeats,
                checkpoint_size=args.checkpoint_size,
                compile_mode=args.compile_mode,
            )
        )
    return results


def _format_compile_warmup(seconds: float | None) -> str:
    return "" if seconds is None else f"{seconds * 1000:.1f}"


def print_markdown(args: argparse.Namespace, results: list[SweepResult]) -> None:
    print("# Compile/Triton Rate Training Sweep")
    print()
    print("Compares regular compiled PyTorch scalar-loss training with the existing")
    print("Triton rate path and the public compile-visible `triton_compile` backend.")
    print()
    print("## Environment")
    print()
    print(f"- `generated_utc`: `{datetime.now(UTC).isoformat(timespec='seconds')}`")
    print(f"- `device`: `{args.device}`")
    print(f"- `gpu`: `{gpu_name(args.device)}`")
    print(f"- `torch`: `{torch.__version__}`")
    print(f"- `cuda_available`: `{torch.cuda.is_available()}`")
    print(f"- `cuda_version`: `{torch.version.cuda}`")
    print(f"- `checkpoint_size`: `{args.checkpoint_size}`")
    print(f"- `compile_mode`: `{args.compile_mode}`")
    print(f"- `warmup`: `{args.warmup}`")
    print(f"- `repeats`: `{args.repeats}`")
    print()
    print("## Results")
    print()
    print("| T | B | F | N | Path | ms | Peak MB | Compile Warmup ms | Error |")
    print("|---:|---:|---:|---:|---|---:|---:|---:|---|")
    for result in results:
        print(
            f"| {result.timesteps} | "
            f"{result.batch} | "
            f"{result.features} | "
            f"{result.neurons} | "
            f"{result.path} | "
            f"{format_ms(result.seconds)} | "
            f"{format_memory(result.peak_bytes)} | "
            f"{_format_compile_warmup(result.compile_warmup_seconds)} | "
            f"{result.error or ''} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", type=parse_shape, action="append", default=[])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint-size", type=parse_checkpoint_size, default=25)
    parser.add_argument(
        "--compile-mode",
        choices=["default", "reduce-overhead", "max-autotune"],
        default="reduce-overhead",
    )
    parser.add_argument(
        "--matmul-precision",
        choices=["highest", "high", "medium"],
        default="high",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print_markdown(args, run_benchmark(args))


if __name__ == "__main__":
    main()
