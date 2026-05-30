"""Benchmark eager and compiled dense LIF forward+backward workloads."""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime

import torch

from myelin.baselines import compiled_available, reset_cuda_peak_memory, synchronize_if_needed
from myelin.benchmarks.lif import (
    DEFAULT_SWEEP_SHAPES,
    format_memory,
    format_ms,
    format_speedup,
    gpu_name,
    parse_shape,
    print_report,
)
from myelin.neurons import LIFParams
from myelin.workloads import (
    dense_fast_surrogate_lif_spike_loss,
    dense_lif_loss,
    dense_surrogate_lif_spike_loss,
    dense_triangular_surrogate_lif_spike_loss,
    looped_fast_surrogate_lif_spike_loss,
    looped_surrogate_lif_spike_loss,
)

TrainStep = Callable[[torch.Tensor, torch.Tensor, LIFParams], torch.Tensor]


@dataclass(frozen=True)
class TrainTiming:
    forward_only_seconds: float
    split_forward_seconds: float
    backward_seconds: float
    forward_backward_seconds: float


@dataclass(frozen=True)
class MemoryCheckpoints:
    allocated_bytes: int | None
    after_forward_bytes: int | None
    after_backward_bytes: int | None


@dataclass(frozen=True)
class CompiledTrainStep:
    fn: TrainStep
    construction_seconds: float


WORKLOAD_DIAGRAM = """\
```text
[T, B, F] inputs
      |
      v
[F, N] trainable weight
      |
      v
[T, B, N] input currents
      |
      v
LIF unroll across time
      |
      v
final membrane energy loss
      |
      v
loss.backward() -> d(weight)
```"""

SURROGATE_WORKLOAD_DIAGRAM = """\
```text
[T, B, F] inputs
      |
      v
[F, N] trainable weight
      |
      v
[T, B, N] input currents
      |
      v
differentiable surrogate LIF unroll across time
      |
      v
spike-rate loss over all timesteps
      |
      v
loss.backward() -> d(weight)
```"""

LOOPED_SURROGATE_WORKLOAD_DIAGRAM = """\
```text
[T, B, F] inputs
      |
      v
for each timestep:
    [B, F] input[t] x [F, N] weight -> [B, N] current[t]
    surrogate LIF step
      |
      v
spike-rate loss over all timesteps
      |
      v
loss.backward() -> d(weight)
```"""

WORKLOADS: dict[str, tuple[TrainStep, str]] = {
    "membrane": (dense_lif_loss, WORKLOAD_DIAGRAM),
    "surrogate-fast": (dense_fast_surrogate_lif_spike_loss, SURROGATE_WORKLOAD_DIAGRAM),
    "surrogate-spike": (dense_surrogate_lif_spike_loss, SURROGATE_WORKLOAD_DIAGRAM),
    "surrogate-triangular": (
        dense_triangular_surrogate_lif_spike_loss,
        SURROGATE_WORKLOAD_DIAGRAM,
    ),
    "surrogate-looped": (looped_surrogate_lif_spike_loss, LOOPED_SURROGATE_WORKLOAD_DIAGRAM),
    "surrogate-fast-looped": (
        looped_fast_surrogate_lif_spike_loss,
        LOOPED_SURROGATE_WORKLOAD_DIAGRAM,
    ),
}


def compiled_train_step(fn: TrainStep, compile_mode: str) -> TrainStep:
    """Return a compiled dense LIF training loss."""

    if compile_mode == "default":
        return torch.compile(fn, fullgraph=True)
    return torch.compile(fn, mode=compile_mode, fullgraph=True)


def timed_compiled_train_step(fn: TrainStep, compile_mode: str) -> CompiledTrainStep:
    start = time.perf_counter()
    compiled_fn = compiled_train_step(fn, compile_mode)
    return CompiledTrainStep(
        fn=compiled_fn,
        construction_seconds=time.perf_counter() - start,
    )


def make_workload_tensors(
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


def cuda_peak_memory(device: torch.device | str) -> int | None:
    if torch.device(device).type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(torch.device(device))


def time_forward(
    fn: TrainStep,
    inputs: torch.Tensor,
    weight: torch.Tensor,
    params: LIFParams,
    *,
    warmup: int,
    repeats: int,
) -> float:
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if repeats <= 0:
        raise ValueError("repeats must be positive")

    for _ in range(warmup):
        weight.grad = None
        fn(inputs, weight, params)
    synchronize_if_needed(inputs.device)

    start = time.perf_counter()
    for _ in range(repeats):
        weight.grad = None
        fn(inputs, weight, params)
    synchronize_if_needed(inputs.device)

    return (time.perf_counter() - start) / repeats


def time_forward_backward_split(
    fn: TrainStep,
    inputs: torch.Tensor,
    weight: torch.Tensor,
    params: LIFParams,
    *,
    warmup: int,
    repeats: int,
) -> tuple[float, float, float]:
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if repeats <= 0:
        raise ValueError("repeats must be positive")

    for _ in range(warmup):
        weight.grad = None
        loss = fn(inputs, weight, params)
        loss.backward()
    synchronize_if_needed(inputs.device)

    forward_elapsed = 0.0
    backward_elapsed = 0.0
    for _ in range(repeats):
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


def time_train_step(
    fn: TrainStep,
    inputs: torch.Tensor,
    weight: torch.Tensor,
    params: LIFParams,
    *,
    warmup: int,
    repeats: int,
) -> TrainTiming:
    forward_only_seconds = time_forward(
        fn,
        inputs,
        weight,
        params,
        warmup=warmup,
        repeats=repeats,
    )
    split_forward_seconds, backward_seconds, forward_backward_seconds = time_forward_backward_split(
        fn,
        inputs,
        weight,
        params,
        warmup=warmup,
        repeats=repeats,
    )
    return TrainTiming(
        forward_only_seconds=forward_only_seconds,
        split_forward_seconds=split_forward_seconds,
        backward_seconds=backward_seconds,
        forward_backward_seconds=forward_backward_seconds,
    )


def memory_checkpoints(
    fn: TrainStep,
    inputs: torch.Tensor,
    weight: torch.Tensor,
    params: LIFParams,
) -> MemoryCheckpoints:
    device = inputs.device
    if device.type != "cuda":
        return MemoryCheckpoints(None, None, None)

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    allocated_bytes = torch.cuda.memory_allocated(device)

    weight.grad = None
    loss = fn(inputs, weight, params)
    synchronize_if_needed(device)
    after_forward_bytes = torch.cuda.max_memory_allocated(device)

    loss.backward()
    synchronize_if_needed(device)
    after_backward_bytes = torch.cuda.max_memory_allocated(device)

    return MemoryCheckpoints(
        allocated_bytes=allocated_bytes,
        after_forward_bytes=after_forward_bytes,
        after_backward_bytes=after_backward_bytes,
    )


def run_bptt_benchmark(
    timesteps: int,
    batch: int,
    features: int,
    neurons: int,
    warmup: int,
    repeats: int,
    device: str,
    compile_enabled: bool,
    fn: TrainStep,
    compile_mode: str,
) -> dict[str, object]:
    inputs, eager_weight = make_workload_tensors(timesteps, batch, features, neurons, device)
    compiled_weight = eager_weight.detach().clone().requires_grad_(True)
    params = LIFParams(threshold=1.0)

    reset_cuda_peak_memory(device)
    eager_timing = time_train_step(
        fn,
        inputs,
        eager_weight,
        params,
        warmup=warmup,
        repeats=repeats,
    )
    eager_memory = memory_checkpoints(fn, inputs, eager_weight, params)

    compiled_timing = None
    compiled_memory = MemoryCheckpoints(None, None, None)
    compile_warmup_seconds = None
    compile_error = None
    if compile_enabled:
        try:
            compiled_result = timed_compiled_train_step(fn, compile_mode)
            compiled_fn = compiled_result.fn
            start = time.perf_counter()
            compile_loss = compiled_fn(inputs, compiled_weight, params)
            compile_loss.backward()
            synchronize_if_needed(device)
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
        except Exception as exc:  # noqa: BLE001 - benchmark should report compiler failures.
            compile_error = f"{type(exc).__name__}: {exc}"
        finally:
            synchronize_if_needed(device)

    forward_only_speedup = None
    split_forward_speedup = None
    backward_speedup = None
    forward_backward_speedup = None
    if compiled_timing is not None:
        forward_only_speedup = (
            eager_timing.forward_only_seconds / compiled_timing.forward_only_seconds
        )
        split_forward_speedup = (
            eager_timing.split_forward_seconds / compiled_timing.split_forward_seconds
        )
        backward_speedup = eager_timing.backward_seconds / compiled_timing.backward_seconds
        forward_backward_speedup = (
            eager_timing.forward_backward_seconds / compiled_timing.forward_backward_seconds
        )

    return {
        "device": str(torch.device(device)),
        "timesteps": timesteps,
        "batch": batch,
        "features": features,
        "neurons": neurons,
        "warmup": warmup,
        "repeats": repeats,
        "eager_forward_only_seconds": eager_timing.forward_only_seconds,
        "compiled_forward_only_seconds": None
        if compiled_timing is None
        else compiled_timing.forward_only_seconds,
        "forward_only_speedup": forward_only_speedup,
        "eager_split_forward_seconds": eager_timing.split_forward_seconds,
        "compiled_split_forward_seconds": None
        if compiled_timing is None
        else compiled_timing.split_forward_seconds,
        "split_forward_speedup": split_forward_speedup,
        "eager_backward_seconds": eager_timing.backward_seconds,
        "compiled_backward_seconds": None
        if compiled_timing is None
        else compiled_timing.backward_seconds,
        "backward_speedup": backward_speedup,
        "eager_forward_backward_seconds": eager_timing.forward_backward_seconds,
        "compiled_forward_backward_seconds": None
        if compiled_timing is None
        else compiled_timing.forward_backward_seconds,
        "forward_backward_speedup": forward_backward_speedup,
        "compile_warmup_seconds": compile_warmup_seconds,
        "compile_error": compile_error,
        "eager_allocated_bytes": eager_memory.allocated_bytes,
        "eager_after_forward_bytes": eager_memory.after_forward_bytes,
        "eager_after_backward_bytes": eager_memory.after_backward_bytes,
        "compiled_allocated_bytes": compiled_memory.allocated_bytes,
        "compiled_after_forward_bytes": compiled_memory.after_forward_bytes,
        "compiled_after_backward_bytes": compiled_memory.after_backward_bytes,
    }


def run_sweep(
    shapes: Iterable[tuple[int, int, int]],
    *,
    features: int,
    warmup: int,
    repeats: int,
    device: str,
    compile_enabled: bool,
    fn: TrainStep,
    compile_mode: str,
) -> list[dict[str, object]]:
    return [
        run_bptt_benchmark(
            timesteps=timesteps,
            batch=batch,
            features=features,
            neurons=neurons,
            warmup=warmup,
            repeats=repeats,
            device=device,
            compile_enabled=compile_enabled,
            fn=fn,
            compile_mode=compile_mode,
        )
        for timesteps, batch, neurons in shapes
    ]


def environment_metadata(
    device: str,
    features: int,
    warmup: int,
    repeats: int,
    compile_enabled: bool,
    workload: str,
    compile_mode: str,
    matmul_precision: str,
) -> dict[str, object]:
    return {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "device": str(torch.device(device)),
        "gpu": gpu_name(device),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "features": features,
        "workload": workload,
        "warmup": warmup,
        "repeats": repeats,
        "compile_enabled": compile_enabled,
        "compile_mode": compile_mode if compile_enabled else None,
        "compile_fullgraph": compile_enabled,
        "compile_time_included": False,
        "matmul_precision": matmul_precision,
        "dtype": "torch.float32",
    }


def print_csv(results: list[dict[str, object]]) -> None:
    print(
        "T,B,F,N,eager_fwd_only_ms,compiled_fwd_only_ms,fwd_only_speedup,"
        "eager_split_fwd_ms,compiled_split_fwd_ms,split_fwd_speedup,"
        "eager_bwd_ms,compiled_bwd_ms,bwd_speedup,"
        "eager_fwbw_ms,compiled_fwbw_ms,fwbw_speedup,"
        "compile_warmup_ms,"
        "eager_alloc_mb,eager_fwd_peak_mb,eager_bwd_peak_mb,"
        "compiled_alloc_mb,compiled_fwd_peak_mb,compiled_bwd_peak_mb,compile_error"
    )
    for result in results:
        print(
            f"{result['timesteps']},{result['batch']},{result['features']},"
            f"{result['neurons']},{format_ms(result['eager_forward_only_seconds'])},"
            f"{format_ms(result['compiled_forward_only_seconds'])},"
            f"{format_speedup(result['forward_only_speedup']).removesuffix('x')},"
            f"{format_ms(result['eager_split_forward_seconds'])},"
            f"{format_ms(result['compiled_split_forward_seconds'])},"
            f"{format_speedup(result['split_forward_speedup']).removesuffix('x')},"
            f"{format_ms(result['eager_backward_seconds'])},"
            f"{format_ms(result['compiled_backward_seconds'])},"
            f"{format_speedup(result['backward_speedup']).removesuffix('x')},"
            f"{format_ms(result['eager_forward_backward_seconds'])},"
            f"{format_ms(result['compiled_forward_backward_seconds'])},"
            f"{format_speedup(result['forward_backward_speedup']).removesuffix('x')},"
            f"{format_ms(result['compile_warmup_seconds'])},"
            f"{format_memory(result['eager_allocated_bytes'])},"
            f"{format_memory(result['eager_after_forward_bytes'])},"
            f"{format_memory(result['eager_after_backward_bytes'])},"
            f"{format_memory(result['compiled_allocated_bytes'])},"
            f"{format_memory(result['compiled_after_forward_bytes'])},"
            f"{format_memory(result['compiled_after_backward_bytes'])},"
            f"{result['compile_error'] or ''}"
        )


def print_markdown(
    results: list[dict[str, object]],
    metadata: dict[str, object],
    workload_diagram: str,
) -> None:
    print("# Dense LIF BPTT Baseline")
    print()
    print("Compile time is excluded from latency measurements.")
    print()
    print("## Workload")
    print()
    print(workload_diagram)
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
        "Fwd-only Speedup | Eager Split Fwd ms | Compiled Split Fwd ms | "
        "Split Fwd Speedup | Eager Bwd ms | Compiled Bwd ms | Bwd Speedup | "
        "Eager Fwd+Bwd ms | Compiled Fwd+Bwd ms | Fwd+Bwd Speedup | "
        "Compile Warmup ms | Compile Error |"
    )
    print(
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"
    )
    for result in results:
        print(
            f"| {result['timesteps']} | {result['batch']} | {result['features']} | "
            f"{result['neurons']} | {format_ms(result['eager_forward_only_seconds'])} | "
            f"{format_ms(result['compiled_forward_only_seconds'])} | "
            f"{format_speedup(result['forward_only_speedup'])} | "
            f"{format_ms(result['eager_split_forward_seconds'])} | "
            f"{format_ms(result['compiled_split_forward_seconds'])} | "
            f"{format_speedup(result['split_forward_speedup'])} | "
            f"{format_ms(result['eager_backward_seconds'])} | "
            f"{format_ms(result['compiled_backward_seconds'])} | "
            f"{format_speedup(result['backward_speedup'])} | "
            f"{format_ms(result['eager_forward_backward_seconds'])} | "
            f"{format_ms(result['compiled_forward_backward_seconds'])} | "
            f"{format_speedup(result['forward_backward_speedup'])} | "
            f"{format_ms(result['compile_warmup_seconds'])} | "
            f"{result['compile_error'] or ''} |"
        )
    print()
    print("### CUDA Memory")
    print()
    print(
        "| T | Batch | Features | N | Eager Alloc MB | Eager Fwd Peak MB | "
        "Eager Bwd Peak MB | Compiled Alloc MB | Compiled Fwd Peak MB | "
        "Compiled Bwd Peak MB |"
    )
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for result in results:
        print(
            f"| {result['timesteps']} | {result['batch']} | {result['features']} | "
            f"{result['neurons']} | {format_memory(result['eager_allocated_bytes'])} | "
            f"{format_memory(result['eager_after_forward_bytes'])} | "
            f"{format_memory(result['eager_after_backward_bytes'])} | "
            f"{format_memory(result['compiled_allocated_bytes'])} | "
            f"{format_memory(result['compiled_after_forward_bytes'])} | "
            f"{format_memory(result['compiled_after_backward_bytes'])} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--features", type=int, default=128)
    parser.add_argument("--neurons", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--shape", action="append", type=parse_shape, default=[])
    parser.add_argument("--format", choices=["kv", "csv", "md"], default="kv")
    parser.add_argument("--workload", choices=sorted(WORKLOADS), default="membrane")
    parser.add_argument(
        "--compile-mode",
        choices=["default", "reduce-overhead", "max-autotune"],
        default="reduce-overhead",
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
    active_matmul_precision = torch.get_float32_matmul_precision()

    workload_fn, workload_diagram = WORKLOADS[args.workload]
    use_compile = (not args.no_compile) and compiled_available()
    if args.sweep or args.shape:
        shapes = args.shape or DEFAULT_SWEEP_SHAPES
        results = run_sweep(
            shapes,
            features=args.features,
            warmup=args.warmup,
            repeats=args.repeats,
            device=args.device,
            compile_enabled=use_compile,
            fn=workload_fn,
            compile_mode=args.compile_mode,
        )
        if args.format == "csv":
            print_csv(results)
        elif args.format == "md":
            metadata = environment_metadata(
                args.device,
                args.features,
                args.warmup,
                args.repeats,
                use_compile,
                args.workload,
                args.compile_mode,
                active_matmul_precision,
            )
            print_markdown(results, metadata, workload_diagram)
        else:
            for index, result in enumerate(results):
                if index:
                    print()
                print_report(result)
        return

    result = run_bptt_benchmark(
        timesteps=args.timesteps,
        batch=args.batch,
        features=args.features,
        neurons=args.neurons,
        warmup=args.warmup,
        repeats=args.repeats,
        device=args.device,
        compile_enabled=use_compile,
        fn=workload_fn,
        compile_mode=args.compile_mode,
    )
    print_report(result)

    if args.matmul_precision != "current":
        torch.set_float32_matmul_precision(original_matmul_precision)


if __name__ == "__main__":
    main()
