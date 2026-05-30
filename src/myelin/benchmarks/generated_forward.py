"""Benchmark generated fused-time forward kernels for multiple neuron IRs."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from datetime import UTC, datetime

import torch

from myelin._optional import has_triton
from myelin.baselines import synchronize_if_needed
from myelin.benchmarks.lif import format_memory, format_ms, format_speedup, gpu_name
from myelin.functional import alif_unroll, izhikevich_unroll, lif_unroll
from myelin.kernels import alif_forward, izhikevich_forward
from myelin.neurons import (
    ALIFParams,
    ALIFState,
    IzhikevichParams,
    IzhikevichState,
    LIFParams,
    LIFState,
)
from myelin.triton import generated_lif_forward


@dataclass(frozen=True)
class Result:
    neuron: str
    torch_seconds: float | None
    generated_seconds: float | None
    peak_bytes: int | None
    speedup: float | None
    state_max_error: float | None
    spike_mismatch_rate: float | None
    error: str | None = None


def time_forward(fn, *, warmup: int, repeats: int) -> tuple[float, int | None]:
    for _ in range(warmup):
        fn()
    synchronize_if_needed("cuda")

    peak = None
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    synchronize_if_needed("cuda")
    seconds = (time.perf_counter() - start) / repeats
    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated()
    return seconds, peak


def max_tensor_error(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left - right).abs().max().item())


def spike_mismatch_rate(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left != right).to(torch.float32).mean().item())


def run_lif(args: argparse.Namespace) -> Result:
    inputs = torch.rand((args.timesteps, args.batch, args.neurons), device=args.device)
    initial = LIFState(membrane=torch.rand((args.batch, args.neurons), device=args.device) * 0.2)
    params = LIFParams(tau_mem=args.tau_mem, threshold=args.threshold, reset=args.reset)

    try:
        expected_state, expected_spikes = lif_unroll(inputs, initial, params)
        actual_state, actual_spikes = generated_lif_forward(
            inputs,
            initial,
            params,
            block_size=args.block_size,
        )
        state_max_error = max_tensor_error(actual_state.membrane, expected_state.membrane)
        mismatch_rate = spike_mismatch_rate(actual_spikes, expected_spikes)
        torch_seconds, _torch_peak = time_forward(
            lambda: lif_unroll(inputs, initial, params),
            warmup=args.warmup,
            repeats=args.repeats,
        )
        generated_seconds, peak = time_forward(
            lambda: generated_lif_forward(inputs, initial, params, block_size=args.block_size),
            warmup=args.warmup,
            repeats=args.repeats,
        )
        return Result(
            "LIF",
            torch_seconds,
            generated_seconds,
            peak,
            torch_seconds / generated_seconds,
            state_max_error,
            mismatch_rate,
        )
    except Exception as exc:  # noqa: BLE001 - benchmark should report backend failures.
        return Result("LIF", None, None, None, None, None, None, f"{type(exc).__name__}: {exc}")


def run_alif(args: argparse.Namespace) -> Result:
    inputs = torch.rand((args.timesteps, args.batch, args.neurons), device=args.device)
    initial = ALIFState(
        membrane=torch.rand((args.batch, args.neurons), device=args.device) * 0.2,
        adaptation=torch.rand((args.batch, args.neurons), device=args.device) * 0.1,
    )
    params = ALIFParams(
        tau_mem=args.tau_mem,
        tau_adaptation=args.tau_adaptation,
        threshold=args.threshold,
        reset=args.reset,
        beta=args.beta,
    )

    try:
        expected_state, expected_spikes = alif_unroll(inputs, initial, params)
        actual_state, actual_spikes = alif_forward(
            inputs,
            initial,
            params,
            backend="triton_generated",
            block_size=args.block_size,
        )
        state_max_error = max(
            max_tensor_error(actual_state.membrane, expected_state.membrane),
            max_tensor_error(actual_state.adaptation, expected_state.adaptation),
        )
        mismatch_rate = spike_mismatch_rate(actual_spikes, expected_spikes)
        torch_seconds, _torch_peak = time_forward(
            lambda: alif_unroll(inputs, initial, params),
            warmup=args.warmup,
            repeats=args.repeats,
        )
        generated_seconds, peak = time_forward(
            lambda: alif_forward(
                inputs,
                initial,
                params,
                backend="triton_generated",
                block_size=args.block_size,
            ),
            warmup=args.warmup,
            repeats=args.repeats,
        )
        return Result(
            "ALIF",
            torch_seconds,
            generated_seconds,
            peak,
            torch_seconds / generated_seconds,
            state_max_error,
            mismatch_rate,
        )
    except Exception as exc:  # noqa: BLE001 - benchmark should report backend failures.
        return Result("ALIF", None, None, None, None, None, None, f"{type(exc).__name__}: {exc}")


def run_izhikevich(args: argparse.Namespace) -> Result:
    inputs = torch.full(
        (args.timesteps, args.batch, args.neurons),
        10.0,
        device=args.device,
    )
    initial = IzhikevichState(
        voltage=torch.full((args.batch, args.neurons), -65.0, device=args.device),
        recovery=torch.full((args.batch, args.neurons), -13.0, device=args.device),
    )
    params = IzhikevichParams()

    try:
        expected_state, expected_spikes = izhikevich_unroll(inputs, initial, params)
        actual_state, actual_spikes = izhikevich_forward(
            inputs,
            initial,
            params,
            backend="triton_generated",
            block_size=args.block_size,
        )
        state_max_error = max(
            max_tensor_error(actual_state.voltage, expected_state.voltage),
            max_tensor_error(actual_state.recovery, expected_state.recovery),
        )
        mismatch_rate = spike_mismatch_rate(actual_spikes, expected_spikes)
        torch_seconds, _torch_peak = time_forward(
            lambda: izhikevich_unroll(inputs, initial, params),
            warmup=args.warmup,
            repeats=args.repeats,
        )
        generated_seconds, peak = time_forward(
            lambda: izhikevich_forward(
                inputs,
                initial,
                params,
                backend="triton_generated",
                block_size=args.block_size,
            ),
            warmup=args.warmup,
            repeats=args.repeats,
        )
        return Result(
            "Izhikevich",
            torch_seconds,
            generated_seconds,
            peak,
            torch_seconds / generated_seconds,
            state_max_error,
            mismatch_rate,
        )
    except Exception as exc:  # noqa: BLE001 - benchmark should report backend failures.
        return Result(
            "Izhikevich",
            None,
            None,
            None,
            None,
            None,
            None,
            f"{type(exc).__name__}: {exc}",
        )


def print_markdown(args: argparse.Namespace, results: list[Result]) -> None:
    print("# Generated Forward Benchmark")
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
    print(f"- `block_size`: `{args.block_size}`")
    print(f"- `warmup`: `{args.warmup}`")
    print(f"- `repeats`: `{args.repeats}`")
    print()
    print("## Results")
    print()
    print(
        "| Neuron | Torch ms | Generated ms | Speedup | Peak MB | "
        "State Max Error | Spike Mismatch Rate | Error |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---|")
    for result in results:
        state_error = "" if result.state_max_error is None else f"{result.state_max_error:.3e}"
        mismatch_rate = (
            "" if result.spike_mismatch_rate is None else f"{result.spike_mismatch_rate:.3e}"
        )
        print(
            f"| {result.neuron} | "
            f"{format_ms(result.torch_seconds)} | "
            f"{format_ms(result.generated_seconds)} | "
            f"{format_speedup(result.speedup)} | "
            f"{format_memory(result.peak_bytes)} | "
            f"{state_error} | "
            f"{mismatch_rate} | "
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
    parser.add_argument("--tau-mem", type=float, default=20.0)
    parser.add_argument("--tau-adaptation", type=float, default=100.0)
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--reset", type=float, default=0.0)
    parser.add_argument("--beta", type=float, default=1.0)
    args = parser.parse_args()

    if torch.device(args.device).type != "cuda" or not has_triton():
        results = [
            Result("LIF", None, None, None, None, None, None, "CUDA and Triton are required"),
            Result("ALIF", None, None, None, None, None, None, "CUDA and Triton are required"),
            Result(
                "Izhikevich",
                None,
                None,
                None,
                None,
                None,
                None,
                "CUDA and Triton are required",
            ),
        ]
    else:
        results = [run_lif(args), run_alif(args), run_izhikevich(args)]
    print_markdown(args, results)


if __name__ == "__main__":
    main()
