"""Benchmark handwritten vs generated fused dense-synapse surrogate LIF training."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from datetime import UTC, datetime

import torch

from spiker._optional import has_triton
from spiker.autograd import (
    generated_triton_linear_surrogate_lif_function,
    triton_linear_surrogate_lif_function,
)
from spiker.baselines import synchronize_if_needed
from spiker.benchmarks.lif import format_memory, format_ms, format_speedup, gpu_name
from spiker.neurons import LIFParams
from spiker.surrogates import SURROGATE_NAMES, SurrogateName


@dataclass(frozen=True)
class Result:
    label: str
    forward_backward_seconds: float | None
    peak_bytes: int | None
    speedup_vs_handwritten: float | None = None
    error: str | None = None


def make_tensors(
    timesteps: int,
    batch: int,
    features: int,
    neurons: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    inputs = torch.rand((timesteps, batch, features), device=device)
    weight = ((torch.rand((features, neurons), device=device) - 0.5) * 0.02).requires_grad_()
    bias = torch.zeros((neurons,), device=device, requires_grad=True)
    return inputs, weight, bias


def time_step(
    fn,
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    params: LIFParams,
    *,
    surrogate: SurrogateName,
    surrogate_slope: float,
    warmup: int,
    repeats: int,
) -> tuple[float, int | None]:
    for _ in range(warmup):
        weight.grad = None
        bias.grad = None
        final_state, spikes = fn(
            inputs,
            weight,
            bias,
            params,
            surrogate=surrogate,
            surrogate_slope=surrogate_slope,
            hard_forward=True,
        )
        loss = spikes.mean().square() + 0.01 * final_state.membrane.square().mean()
        loss.backward()
    synchronize_if_needed(inputs.device)

    peak = None
    if inputs.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(inputs.device)
    start = time.perf_counter()
    for _ in range(repeats):
        weight.grad = None
        bias.grad = None
        final_state, spikes = fn(
            inputs,
            weight,
            bias,
            params,
            surrogate=surrogate,
            surrogate_slope=surrogate_slope,
            hard_forward=True,
        )
        loss = spikes.mean().square() + 0.01 * final_state.membrane.square().mean()
        loss.backward()
    synchronize_if_needed(inputs.device)
    seconds = (time.perf_counter() - start) / repeats
    if inputs.device.type == "cuda":
        peak = torch.cuda.max_memory_allocated(inputs.device)
    return seconds, peak


def run_one(args: argparse.Namespace) -> list[Result]:
    if torch.device(args.device).type != "cuda" or not has_triton():
        return [
            Result(
                "handwritten Triton fused synapse", None, None, error="CUDA and Triton required"
            ),
            Result("generated Triton fused synapse", None, None, error="CUDA and Triton required"),
        ]

    inputs, weight, bias = make_tensors(
        args.timesteps,
        args.batch,
        args.features,
        args.neurons,
        args.device,
    )
    generated_weight = weight.detach().clone().requires_grad_()
    generated_bias = bias.detach().clone().requires_grad_()
    params = LIFParams(tau_mem=args.tau_mem, threshold=args.threshold, reset=args.reset)

    results: list[Result] = []
    handwritten_seconds = None
    try:
        seconds, peak = time_step(
            triton_linear_surrogate_lif_function,
            inputs,
            weight,
            bias,
            params,
            surrogate=args.surrogate,
            surrogate_slope=args.surrogate_slope,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        handwritten_seconds = seconds
        results.append(Result("handwritten Triton fused synapse", seconds, peak))
    except Exception as exc:  # noqa: BLE001 - benchmark should report kernel failures.
        results.append(
            Result(
                "handwritten Triton fused synapse", None, None, error=f"{type(exc).__name__}: {exc}"
            )
        )

    try:
        seconds, peak = time_step(
            generated_triton_linear_surrogate_lif_function,
            inputs,
            generated_weight,
            generated_bias,
            params,
            surrogate=args.surrogate,
            surrogate_slope=args.surrogate_slope,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        speedup = None if handwritten_seconds is None else handwritten_seconds / seconds
        results.append(Result("generated Triton fused synapse", seconds, peak, speedup))
    except Exception as exc:  # noqa: BLE001 - benchmark should report kernel failures.
        results.append(
            Result(
                "generated Triton fused synapse", None, None, error=f"{type(exc).__name__}: {exc}"
            )
        )

    return results


def print_markdown(args: argparse.Namespace, results: list[Result]) -> None:
    print("# Generated Fused Synapse Backward Benchmark")
    print()
    print("## Environment")
    print()
    print(f"- `generated_utc`: `{datetime.now(UTC).isoformat(timespec='seconds')}`")
    print(f"- `device`: `{args.device}`")
    print(f"- `gpu`: `{gpu_name(args.device)}`")
    print(f"- `torch`: `{torch.__version__}`")
    print(f"- `cuda_available`: `{torch.cuda.is_available()}`")
    print(f"- `cuda_version`: `{torch.version.cuda}`")
    print(f"- `shape`: `T={args.timesteps}, B={args.batch}, F={args.features}, N={args.neurons}`")
    print(f"- `surrogate`: `{args.surrogate}`")
    print(f"- `surrogate_slope`: `{args.surrogate_slope}`")
    print(f"- `warmup`: `{args.warmup}`")
    print(f"- `repeats`: `{args.repeats}`")
    print()
    print("## Results")
    print()
    print("| Variant | Fwd+Bwd ms | Peak MB | Speedup vs handwritten | Error |")
    print("|---|---:|---:|---:|---|")
    for result in results:
        print(
            f"| {result.label} | "
            f"{format_ms(result.forward_backward_seconds)} | "
            f"{format_memory(result.peak_bytes)} | "
            f"{format_speedup(result.speedup_vs_handwritten)} | "
            f"{result.error or ''} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--features", type=int, default=128)
    parser.add_argument("--neurons", type=int, default=2048)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
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
