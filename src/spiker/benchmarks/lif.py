"""Benchmark eager and compiled LIF reference paths."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from datetime import UTC, datetime

import torch

from spiker._optional import has_triton
from spiker.baselines import (
    compiled_available,
    eager_lif_unroll,
    max_cuda_memory_allocated,
    reset_cuda_peak_memory,
    synchronize_if_needed,
    time_lif_unroll,
    timed_compiled_lif_unroll,
)
from spiker.neurons import LIFParams, LIFState

DEFAULT_SWEEP_SHAPES = [
    (25, 16, 512),
    (25, 64, 2048),
    (25, 256, 8192),
    (100, 16, 512),
    (100, 64, 2048),
    (100, 256, 8192),
    (200, 16, 512),
    (200, 64, 2048),
    (200, 256, 8192),
]


def make_inputs(
    timesteps: int, batch: int, neurons: int, device: str
) -> tuple[torch.Tensor, LIFState]:
    inputs = torch.rand((timesteps, batch, neurons), device=device)
    state = LIFState(membrane=torch.zeros((batch, neurons), device=device))
    return inputs, state


def run_lif_benchmark(
    timesteps: int,
    batch: int,
    neurons: int,
    warmup: int,
    repeats: int,
    device: str,
    compile_enabled: bool,
) -> dict[str, object]:
    inputs, state = make_inputs(timesteps, batch, neurons, device)
    params = LIFParams()

    reset_cuda_peak_memory(device)
    eager = time_lif_unroll(
        "eager",
        eager_lif_unroll,
        inputs,
        state,
        params,
        warmup=warmup,
        repeats=repeats,
    )

    compiled = None
    compile_warmup_seconds = None
    compile_error = None
    if compile_enabled:
        try:
            compiled_result = timed_compiled_lif_unroll()
            compiled_fn = compiled_result.fn
            import time

            start = time.perf_counter()
            compiled_fn(inputs, state, params)
            synchronize_if_needed(device)
            compile_warmup_seconds = time.perf_counter() - start
            compiled = time_lif_unroll(
                "compiled",
                compiled_fn,
                inputs,
                state,
                params,
                warmup=warmup,
                repeats=repeats,
            )
        except Exception as exc:  # noqa: BLE001 - benchmark should report compiler failures.
            compile_error = f"{type(exc).__name__}: {exc}"
        finally:
            synchronize_if_needed(device)

    triton_result = None
    triton_error = None
    if torch.device(device).type == "cuda" and has_triton():
        try:
            from spiker.triton import lif_forward

            triton_result = time_lif_unroll(
                "triton",
                lif_forward,
                inputs,
                state,
                params,
                warmup=warmup,
                repeats=repeats,
            )
        except Exception as exc:  # noqa: BLE001 - benchmark should report kernel failures.
            triton_error = f"{type(exc).__name__}: {exc}"
        finally:
            synchronize_if_needed(device)

    peak_memory = max_cuda_memory_allocated(device)
    speedup = None
    if compiled is not None:
        speedup = eager.seconds / compiled.seconds
    triton_speedup = None
    if triton_result is not None:
        triton_speedup = eager.seconds / triton_result.seconds

    return {
        "device": str(torch.device(device)),
        "timesteps": timesteps,
        "batch": batch,
        "neurons": neurons,
        "warmup": warmup,
        "repeats": repeats,
        "eager_seconds": eager.seconds,
        "compiled_seconds": None if compiled is None else compiled.seconds,
        "compile_warmup_seconds": compile_warmup_seconds,
        "compiled_speedup": speedup,
        "compile_error": compile_error,
        "triton_seconds": None if triton_result is None else triton_result.seconds,
        "triton_speedup": triton_speedup,
        "triton_error": triton_error,
        "cuda_peak_memory_bytes": peak_memory,
    }


def parse_shape(raw: str) -> tuple[int, int, int]:
    parts = raw.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("shape must be T,B,N")
    try:
        timesteps, batch, neurons = (int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("shape values must be integers") from exc
    if timesteps <= 0 or batch <= 0 or neurons <= 0:
        raise argparse.ArgumentTypeError("shape values must be positive")
    return timesteps, batch, neurons


def gpu_name(device: str) -> str | None:
    torch_device = torch.device(device)
    if torch_device.type != "cuda":
        return None
    return torch.cuda.get_device_name(torch_device)


def format_ms(seconds: object) -> str:
    if not isinstance(seconds, float):
        return ""
    return f"{seconds * 1000:.3f}"


def format_speedup(speedup: object) -> str:
    if not isinstance(speedup, float):
        return ""
    return f"{speedup:.2f}x"


def format_memory(memory_bytes: object) -> str:
    if not isinstance(memory_bytes, int):
        return ""
    return f"{memory_bytes / (1024 * 1024):.1f}"


def environment_metadata(
    device: str, warmup: int, repeats: int, compile_enabled: bool
) -> dict[str, object]:
    return {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "device": str(torch.device(device)),
        "gpu": gpu_name(device),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "warmup": warmup,
        "repeats": repeats,
        "compile_enabled": compile_enabled,
        "compile_mode": "reduce-overhead" if compile_enabled else None,
        "compile_fullgraph": compile_enabled,
        "compile_time_included": False,
        "dtype": "torch.float32",
    }


def run_sweep(
    shapes: Iterable[tuple[int, int, int]],
    *,
    warmup: int,
    repeats: int,
    device: str,
    compile_enabled: bool,
) -> list[dict[str, object]]:
    results = []
    for timesteps, batch, neurons in shapes:
        results.append(
            run_lif_benchmark(
                timesteps=timesteps,
                batch=batch,
                neurons=neurons,
                warmup=warmup,
                repeats=repeats,
                device=device,
                compile_enabled=compile_enabled,
            )
        )
    return results


def print_report(result: dict[str, object]) -> None:
    for key, value in result.items():
        if isinstance(value, float):
            print(f"{key}={value:.6f}")
        else:
            print(f"{key}={value}")


def print_csv(results: list[dict[str, object]]) -> None:
    print(
        "T,B,N,eager_ms,compiled_ms,triton_ms,compile_warmup_ms,"
        "compiled_speedup,triton_speedup,peak_mem_mb,compile_error,triton_error"
    )
    for result in results:
        print(
            f"{result['timesteps']},{result['batch']},{result['neurons']},"
            f"{format_ms(result['eager_seconds'])},"
            f"{format_ms(result['compiled_seconds'])},"
            f"{format_ms(result['triton_seconds'])},"
            f"{format_ms(result['compile_warmup_seconds'])},"
            f"{format_speedup(result['compiled_speedup']).removesuffix('x')},"
            f"{format_speedup(result['triton_speedup']).removesuffix('x')},"
            f"{format_memory(result['cuda_peak_memory_bytes'])},"
            f"{result['compile_error'] or ''},"
            f"{result['triton_error'] or ''}"
        )


def print_markdown(results: list[dict[str, object]], metadata: dict[str, object]) -> None:
    print("# LIF Compile Baseline")
    print()
    print("Compile time is excluded from latency measurements.")
    print()
    print("## Environment")
    print()
    for key, value in metadata.items():
        print(f"- `{key}`: `{value}`")
    print()
    print("## Results")
    print()
    print(
        "| T | Batch | N | Eager ms | Compiled ms | Triton ms | Compile Warmup ms | "
        "Compiled Speedup | Triton Speedup | Peak CUDA MB | Compile Error | Triton Error |"
    )
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|")
    for result in results:
        print(
            f"| {result['timesteps']} | {result['batch']} | {result['neurons']} | "
            f"{format_ms(result['eager_seconds'])} | "
            f"{format_ms(result['compiled_seconds'])} | "
            f"{format_ms(result['triton_seconds'])} | "
            f"{format_ms(result['compile_warmup_seconds'])} | "
            f"{format_speedup(result['compiled_speedup'])} | "
            f"{format_speedup(result['triton_speedup'])} | "
            f"{format_memory(result['cuda_peak_memory_bytes'])} | "
            f"{result['compile_error'] or ''} | "
            f"{result['triton_error'] or ''} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--neurons", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--shape", action="append", type=parse_shape, default=[])
    parser.add_argument("--format", choices=["kv", "csv", "md"], default="kv")
    args = parser.parse_args()

    use_compile = (not args.no_compile) and compiled_available()
    if args.sweep or args.shape:
        shapes = args.shape or DEFAULT_SWEEP_SHAPES
        results = run_sweep(
            shapes,
            warmup=args.warmup,
            repeats=args.repeats,
            device=args.device,
            compile_enabled=use_compile,
        )
        if args.format == "csv":
            print_csv(results)
        elif args.format == "md":
            metadata = environment_metadata(args.device, args.warmup, args.repeats, use_compile)
            print_markdown(results, metadata)
        else:
            for index, result in enumerate(results):
                if index:
                    print()
                print_report(result)
        return

    result = run_lif_benchmark(
        timesteps=args.timesteps,
        batch=args.batch,
        neurons=args.neurons,
        warmup=args.warmup,
        repeats=args.repeats,
        device=args.device,
        compile_enabled=use_compile,
    )
    print_report(result)


if __name__ == "__main__":
    main()
