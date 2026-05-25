"""Benchmark handwritten vs generated surrogate LIF backward kernels."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from datetime import UTC, datetime

import torch

from spiker._optional import has_triton
from spiker.baselines import synchronize_if_needed
from spiker.benchmarks.lif import format_memory, format_ms, format_speedup, gpu_name
from spiker.neurons import LIFParams
from spiker.surrogates import SURROGATE_NAMES, SurrogateName


@dataclass(frozen=True)
class Result:
    label: str
    seconds: float | None
    peak_bytes: int | None
    speedup_vs_handwritten: float | None = None
    error: str | None = None


def make_tensors(
    timesteps: int,
    batch: int,
    neurons: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pre_reset = torch.rand((timesteps, batch, neurons), device=device) * 1.4 - 0.2
    spikes = (pre_reset >= 0.7).to(pre_reset.dtype)
    grad_final = torch.rand((batch, neurons), device=device) - 0.5
    grad_spikes = torch.rand_like(pre_reset) - 0.5
    return pre_reset, spikes, grad_final, grad_spikes


def time_backward(
    fn,
    pre_reset: torch.Tensor,
    spikes: torch.Tensor,
    grad_final: torch.Tensor,
    grad_spikes: torch.Tensor,
    params: LIFParams,
    *,
    surrogate: SurrogateName,
    surrogate_slope: float,
    block_size: int,
    warmup: int,
    repeats: int,
) -> tuple[float, int | None]:
    for _ in range(warmup):
        fn(
            pre_reset,
            spikes,
            grad_final,
            grad_spikes,
            params,
            surrogate=surrogate,
            surrogate_slope=surrogate_slope,
            block_size=block_size,
        )
    synchronize_if_needed(pre_reset.device)

    peak = None
    if pre_reset.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(pre_reset.device)
    start = time.perf_counter()
    for _ in range(repeats):
        fn(
            pre_reset,
            spikes,
            grad_final,
            grad_spikes,
            params,
            surrogate=surrogate,
            surrogate_slope=surrogate_slope,
            block_size=block_size,
        )
    synchronize_if_needed(pre_reset.device)
    seconds = (time.perf_counter() - start) / repeats
    if pre_reset.device.type == "cuda":
        peak = torch.cuda.max_memory_allocated(pre_reset.device)
    return seconds, peak


def run_one(args: argparse.Namespace) -> list[Result]:
    if torch.device(args.device).type != "cuda" or not has_triton():
        return [
            Result(
                label="handwritten Triton backward",
                seconds=None,
                peak_bytes=None,
                error="CUDA and Triton are required",
            ),
            Result(
                label="generated Triton backward",
                seconds=None,
                peak_bytes=None,
                error="CUDA and Triton are required",
            ),
        ]

    from spiker.triton import generated_lif_surrogate_backward, surrogate_lif_backward

    pre_reset, spikes, grad_final, grad_spikes = make_tensors(
        args.timesteps,
        args.batch,
        args.neurons,
        args.device,
    )
    params = LIFParams(tau_mem=args.tau_mem, threshold=args.threshold, reset=args.reset)

    results: list[Result] = []
    handwritten_seconds = None
    try:
        seconds, peak = time_backward(
            surrogate_lif_backward,
            pre_reset,
            spikes,
            grad_final,
            grad_spikes,
            params,
            surrogate=args.surrogate,
            surrogate_slope=args.surrogate_slope,
            block_size=args.block_size,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        handwritten_seconds = seconds
        results.append(Result("handwritten Triton backward", seconds, peak))
    except Exception as exc:  # noqa: BLE001 - benchmark should report kernel failures.
        results.append(
            Result("handwritten Triton backward", None, None, error=f"{type(exc).__name__}: {exc}")
        )

    try:
        seconds, peak = time_backward(
            generated_lif_surrogate_backward,
            pre_reset,
            spikes,
            grad_final,
            grad_spikes,
            params,
            surrogate=args.surrogate,
            surrogate_slope=args.surrogate_slope,
            block_size=args.block_size,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        speedup = None if handwritten_seconds is None else handwritten_seconds / seconds
        results.append(Result("generated Triton backward", seconds, peak, speedup))
    except Exception as exc:  # noqa: BLE001 - benchmark should report kernel failures.
        results.append(
            Result("generated Triton backward", None, None, error=f"{type(exc).__name__}: {exc}")
        )

    return results


def print_markdown(args: argparse.Namespace, results: list[Result]) -> None:
    print("# Generated Surrogate Backward Benchmark")
    print()
    print("## Environment")
    print()
    print(f"- `generated_utc`: `{datetime.now(UTC).isoformat(timespec='seconds')}`")
    print(f"- `device`: `{args.device}`")
    print(f"- `gpu`: `{gpu_name(args.device)}`")
    print(f"- `torch`: `{torch.__version__}`")
    print(f"- `cuda_available`: `{torch.cuda.is_available()}`")
    print(f"- `cuda_version`: `{torch.version.cuda}`")
    print(f"- `shape`: `T={args.timesteps}, B={args.batch}, N={args.neurons}`")
    print(f"- `surrogate`: `{args.surrogate}`")
    print(f"- `surrogate_slope`: `{args.surrogate_slope}`")
    print(f"- `block_size`: `{args.block_size}`")
    print(f"- `warmup`: `{args.warmup}`")
    print(f"- `repeats`: `{args.repeats}`")
    print()
    print("## Results")
    print()
    print("| Variant | Backward ms | Peak MB | Speedup vs handwritten | Error |")
    print("|---|---:|---:|---:|---|")
    for result in results:
        print(
            f"| {result.label} | "
            f"{format_ms(result.seconds)} | "
            f"{format_memory(result.peak_bytes)} | "
            f"{format_speedup(result.speedup_vs_handwritten)} | "
            f"{result.error or ''} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--neurons", type=int, default=2048)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument(
        "--surrogate",
        choices=SURROGATE_NAMES,
        default="fast_sigmoid",
    )
    parser.add_argument("--surrogate-slope", type=float, default=5.0)
    parser.add_argument("--tau-mem", type=float, default=20.0)
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--reset", type=float, default=0.0)
    args = parser.parse_args()

    print_markdown(args, run_one(args))


if __name__ == "__main__":
    main()
