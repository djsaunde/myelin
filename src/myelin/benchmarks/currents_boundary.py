"""Compare materialized and per-timestep-current PyTorch training graphs."""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

import torch

from myelin.baselines import compiled_available, synchronize_if_needed
from myelin.benchmarks.lif import format_memory, format_ms, gpu_name, parse_shape
from myelin.benchmarks.lif_bptt import memory_checkpoints, time_train_step
from myelin.neurons import LIFParams
from myelin.workloads import (
    dense_fast_surrogate_lif_spike_loss,
    looped_fast_surrogate_lif_spike_loss,
)

TrainStep = Callable[[torch.Tensor, torch.Tensor, LIFParams], torch.Tensor]

DEFAULT_SHAPES = [
    (25, 64, 2048),
    (100, 64, 2048),
]


@dataclass(frozen=True)
class BoundaryResult:
    timesteps: int
    batch: int
    features: int
    neurons: int
    workload: str
    eager_forward_backward_seconds: float | None
    compiled_forward_backward_seconds: float | None
    eager_allocated_bytes: int | None
    compiled_allocated_bytes: int | None
    eager_forward_peak_bytes: int | None
    compiled_forward_peak_bytes: int | None
    eager_backward_peak_bytes: int | None
    compiled_backward_peak_bytes: int | None
    compile_warmup_seconds: float | None
    error: str | None = None

    @property
    def speedup(self) -> float | None:
        if (
            self.eager_forward_backward_seconds is None
            or self.compiled_forward_backward_seconds is None
        ):
            return None
        return self.eager_forward_backward_seconds / self.compiled_forward_backward_seconds

    @property
    def eager_forward_increment_bytes(self) -> int | None:
        if self.eager_allocated_bytes is None or self.eager_forward_peak_bytes is None:
            return None
        return self.eager_forward_peak_bytes - self.eager_allocated_bytes

    @property
    def compiled_forward_increment_bytes(self) -> int | None:
        if self.compiled_allocated_bytes is None or self.compiled_forward_peak_bytes is None:
            return None
        return self.compiled_forward_peak_bytes - self.compiled_allocated_bytes

    @property
    def eager_forward_increment_ratio(self) -> float | None:
        increment = self.eager_forward_increment_bytes
        if increment is None:
            return None
        return increment / current_bytes(self)

    @property
    def compiled_forward_increment_ratio(self) -> float | None:
        increment = self.compiled_forward_increment_bytes
        if increment is None:
            return None
        return increment / current_bytes(self)


def compile_step(fn: TrainStep, compile_mode: str) -> TrainStep:
    if compile_mode == "default":
        return torch.compile(fn, fullgraph=True)
    return torch.compile(fn, mode=compile_mode, fullgraph=True)


def make_tensors(
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


def run_one(
    *,
    timesteps: int,
    batch: int,
    features: int,
    neurons: int,
    workload: str,
    fn: TrainStep,
    device: str,
    warmup: int,
    repeats: int,
    compile_mode: str,
) -> BoundaryResult:
    inputs, eager_weight = make_tensors(timesteps, batch, features, neurons, device)
    compiled_weight = eager_weight.detach().clone().requires_grad_(True)
    params = LIFParams()

    try:
        eager_timing = time_train_step(
            fn,
            inputs,
            eager_weight,
            params,
            warmup=warmup,
            repeats=repeats,
        )
        eager_memory = memory_checkpoints(fn, inputs, eager_weight, params)

        compiled_fn = compile_step(fn, compile_mode)
        start = time.perf_counter()
        compiled_weight.grad = None
        loss = compiled_fn(inputs, compiled_weight, params)
        loss.backward()
        synchronize_if_needed(inputs.device)
        compile_warmup_seconds = time.perf_counter() - start

        compiled_timing = time_train_step(
            compiled_fn,
            inputs,
            compiled_weight,
            params,
            warmup=warmup,
            repeats=repeats,
        )
        compiled_memory = memory_checkpoints(compiled_fn, inputs, compiled_weight, params)
    except Exception as exc:  # noqa: BLE001 - benchmark should report compiler/backend failures.
        synchronize_if_needed(torch.device(device))
        return BoundaryResult(
            timesteps=timesteps,
            batch=batch,
            features=features,
            neurons=neurons,
            workload=workload,
            eager_forward_backward_seconds=None,
            compiled_forward_backward_seconds=None,
            eager_allocated_bytes=None,
            compiled_allocated_bytes=None,
            eager_forward_peak_bytes=None,
            compiled_forward_peak_bytes=None,
            eager_backward_peak_bytes=None,
            compiled_backward_peak_bytes=None,
            compile_warmup_seconds=None,
            error=f"{type(exc).__name__}: {exc}",
        )

    return BoundaryResult(
        timesteps=timesteps,
        batch=batch,
        features=features,
        neurons=neurons,
        workload=workload,
        eager_forward_backward_seconds=eager_timing.forward_backward_seconds,
        compiled_forward_backward_seconds=compiled_timing.forward_backward_seconds,
        eager_allocated_bytes=eager_memory.allocated_bytes,
        compiled_allocated_bytes=compiled_memory.allocated_bytes,
        eager_forward_peak_bytes=eager_memory.after_forward_bytes,
        compiled_forward_peak_bytes=compiled_memory.after_forward_bytes,
        eager_backward_peak_bytes=eager_memory.after_backward_bytes,
        compiled_backward_peak_bytes=compiled_memory.after_backward_bytes,
        compile_warmup_seconds=compile_warmup_seconds,
    )


def run_benchmark(args: argparse.Namespace) -> list[BoundaryResult]:
    if not compiled_available():
        raise RuntimeError("torch.compile is unavailable in this PyTorch build")

    workloads: list[tuple[str, TrainStep]] = [
        ("Materialized currents", dense_fast_surrogate_lif_spike_loss),
        ("Per-timestep matmul", looped_fast_surrogate_lif_spike_loss),
    ]
    results: list[BoundaryResult] = []
    for timesteps, batch, neurons in args.shape:
        for workload, fn in workloads:
            results.append(
                run_one(
                    timesteps=timesteps,
                    batch=batch,
                    features=args.features,
                    neurons=neurons,
                    workload=workload,
                    fn=fn,
                    device=args.device,
                    warmup=args.warmup,
                    repeats=args.repeats,
                    compile_mode=args.compile_mode,
                )
            )
    return results


def current_bytes(result: BoundaryResult) -> int:
    return (
        result.timesteps
        * result.batch
        * result.neurons
        * torch.tensor([], dtype=torch.float32).element_size()
    )


def format_ratio(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.2f}x"


def print_markdown(args: argparse.Namespace, results: list[BoundaryResult]) -> None:
    print("# Currents Boundary Check")
    print()
    print("Compile time is excluded from latency measurements. This compares whether")
    print("the PyTorch workload explicitly materializes `[T, B, N]` currents before")
    print("the recurrent loop or computes `[B, N]` currents inside the timestep loop.")
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
    print(f"- `warmup`: `{args.warmup}`")
    print(f"- `repeats`: `{args.repeats}`")
    print(f"- `compile_mode`: `{args.compile_mode}`")
    print("- `compile_fullgraph`: `True`")
    print(f"- `matmul_precision`: `{torch.get_float32_matmul_precision()}`")
    print()
    print("## Results")
    print()
    print(
        "| T | Batch | F | N | Workload | Expected Currents MB | Eager Fwd+Bwd ms | "
        "Compiled Fwd+Bwd ms | Speedup | Eager Fwd Peak MB | Compiled Fwd Peak MB | "
        "Eager Fwd Increment MB | Compiled Fwd Increment MB | "
        "Eager Increment / Currents | Compiled Increment / Currents | "
        "Eager Bwd Peak MB | Compiled Bwd Peak MB | Compile Warmup ms | Error |"
    )
    print(
        "|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"
    )
    for result in results:
        print(
            f"| {result.timesteps} | "
            f"{result.batch} | "
            f"{result.features} | "
            f"{result.neurons} | "
            f"{result.workload} | "
            f"{format_memory(current_bytes(result))} | "
            f"{format_ms(result.eager_forward_backward_seconds)} | "
            f"{format_ms(result.compiled_forward_backward_seconds)} | "
            f"{'' if result.speedup is None else f'{result.speedup:.2f}x'} | "
            f"{format_memory(result.eager_forward_peak_bytes)} | "
            f"{format_memory(result.compiled_forward_peak_bytes)} | "
            f"{format_memory(result.eager_forward_increment_bytes)} | "
            f"{format_memory(result.compiled_forward_increment_bytes)} | "
            f"{format_ratio(result.eager_forward_increment_ratio)} | "
            f"{format_ratio(result.compiled_forward_increment_ratio)} | "
            f"{format_memory(result.eager_backward_peak_bytes)} | "
            f"{format_memory(result.compiled_backward_peak_bytes)} | "
            f"{format_ms(result.compile_warmup_seconds)} | "
            f"{result.error or ''} |"
        )
    print()
    print("## Takeaway")
    print()
    print(
        "`torch.compile` can reduce the materialized-looking PyTorch graph to a "
        "small peak allocation, so source-level `currents = matmul(...)` is not "
        "proof that `[T, B, N]` survives as a distinct runtime allocation."
    )
    print(
        "The per-timestep graph remains useful as a semantic boundary check: it "
        "forces current production inside the recurrence, while the materialized "
        "graph lets Inductor decide whether to eliminate or rematerialize the "
        "temporary."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--features", type=int, default=128)
    parser.add_argument("--shape", action="append", type=parse_shape)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--compile-mode",
        choices=["default", "reduce-overhead", "max-autotune"],
        default="reduce-overhead",
    )
    parser.add_argument(
        "--matmul-precision",
        choices=["current", "highest", "high", "medium"],
        default="high",
    )
    args = parser.parse_args()
    if args.shape is None:
        args.shape = DEFAULT_SHAPES

    original_matmul_precision = torch.get_float32_matmul_precision()
    if args.matmul_precision != "current":
        torch.set_float32_matmul_precision(args.matmul_precision)
    try:
        print_markdown(args, run_benchmark(args))
    finally:
        if args.matmul_precision != "current":
            torch.set_float32_matmul_precision(original_matmul_precision)


if __name__ == "__main__":
    main()
