"""Benchmark online eligibility rules against surrogate BPTT baselines."""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

import torch

from myelin.baselines import synchronize_if_needed
from myelin.benchmarks.lif import format_memory, format_ms, gpu_name
from myelin.dsl import SurrogateBuilder
from myelin.neurons import ALIFParams, LIFParams
from myelin.online import linear_alif_online_eligibility_grad, linear_lif_online_eligibility_grad

StepFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor], torch.Tensor]


def _fast_sigmoid_alias_surrogate_ir():
    builder = SurrogateBuilder("fast_sigmoid_alias")
    centered = builder.centered()
    return builder.build(0.5 / (1.0 + centered.abs()).square())


FAST_SIGMOID_ALIAS_SURROGATE_IR = _fast_sigmoid_alias_surrogate_ir()


@dataclass(frozen=True)
class BenchmarkResult:
    variant: str
    update_seconds: float | None
    peak_bytes: int | None
    grad_weight_norm: float | None
    error: str | None = None


def lif_online_step(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    learning_signal: torch.Tensor,
) -> torch.Tensor:
    result = linear_lif_online_eligibility_grad(
        inputs,
        weight,
        bias,
        learning_signal,
        LIFParams(tau_mem=8.0, threshold=1.0, reset=0.0),
    )
    return result.grad_weight


def alif_online_step(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    learning_signal: torch.Tensor,
) -> torch.Tensor:
    result = linear_alif_online_eligibility_grad(
        inputs,
        weight,
        bias,
        learning_signal,
        ALIFParams(tau_mem=8.0, tau_adaptation=20.0, threshold=1.0, reset=0.0, beta=0.5),
    )
    return result.grad_weight


def lif_online_custom_surrogate_step(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    learning_signal: torch.Tensor,
) -> torch.Tensor:
    result = linear_lif_online_eligibility_grad(
        inputs,
        weight,
        bias,
        learning_signal,
        LIFParams(tau_mem=8.0, threshold=1.0, reset=0.0),
        surrogate=FAST_SIGMOID_ALIAS_SURROGATE_IR,
    )
    return result.grad_weight


def alif_online_custom_surrogate_step(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    learning_signal: torch.Tensor,
) -> torch.Tensor:
    result = linear_alif_online_eligibility_grad(
        inputs,
        weight,
        bias,
        learning_signal,
        ALIFParams(tau_mem=8.0, tau_adaptation=20.0, threshold=1.0, reset=0.0, beta=0.5),
        surrogate=FAST_SIGMOID_ALIAS_SURROGATE_IR,
    )
    return result.grad_weight


def lif_bptt_step(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    learning_signal: torch.Tensor,
) -> torch.Tensor:
    params = LIFParams(tau_mem=8.0, threshold=1.0, reset=0.0)
    train_weight = weight.detach().clone().requires_grad_(True)
    train_bias = None if bias is None else bias.detach().clone().requires_grad_(True)
    membrane = inputs.new_zeros((inputs.shape[1], weight.shape[1]))
    loss = inputs.new_zeros(())
    for input_features, local_signal in zip(
        inputs.unbind(dim=0),
        learning_signal.unbind(dim=0),
        strict=True,
    ):
        current = torch.matmul(input_features, train_weight)
        if train_bias is not None:
            current = current + train_bias
        membrane = membrane * params.decay + current
        centered = 5.0 * (membrane - params.threshold)
        spike = 0.5 * (centered / (1.0 + centered.abs()) + 1.0)
        loss = loss + (spike * local_signal).sum()
        membrane = membrane * (1.0 - spike)
    loss.backward()
    if train_weight.grad is None:
        raise RuntimeError("BPTT did not produce weight gradients")
    return train_weight.grad


def alif_bptt_step(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    learning_signal: torch.Tensor,
) -> torch.Tensor:
    params = ALIFParams(tau_mem=8.0, tau_adaptation=20.0, threshold=1.0, reset=0.0, beta=0.5)
    train_weight = weight.detach().clone().requires_grad_(True)
    train_bias = None if bias is None else bias.detach().clone().requires_grad_(True)
    membrane = inputs.new_zeros((inputs.shape[1], weight.shape[1]))
    adaptation = inputs.new_zeros((inputs.shape[1], weight.shape[1]))
    loss = inputs.new_zeros(())
    for input_features, local_signal in zip(
        inputs.unbind(dim=0),
        learning_signal.unbind(dim=0),
        strict=True,
    ):
        current = torch.matmul(input_features, train_weight)
        if train_bias is not None:
            current = current + train_bias
        membrane = membrane * params.decay + current
        centered = 5.0 * (membrane - params.threshold - params.beta * adaptation)
        spike = 0.5 * (centered / (1.0 + centered.abs()) + 1.0)
        loss = loss + (spike * local_signal).sum()
        membrane = membrane * (1.0 - spike)
        adaptation = adaptation * params.adaptation_decay + spike
    loss.backward()
    if train_weight.grad is None:
        raise RuntimeError("BPTT did not produce weight gradients")
    return train_weight.grad


def make_inputs(
    *,
    timesteps: int,
    batch: int,
    features: int,
    neurons: int,
    device: str,
    bias: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
    inputs = torch.rand((timesteps, batch, features), device=device)
    weight = (torch.rand((features, neurons), device=device) - 0.5) * 0.02
    bias_tensor = torch.zeros(neurons, device=device) if bias else None
    learning_signal = torch.rand((timesteps, batch, neurons), device=device) - 0.5
    return inputs, weight, bias_tensor, learning_signal


def time_step(
    fn: StepFn,
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    learning_signal: torch.Tensor,
    *,
    warmup: int,
    repeats: int,
) -> tuple[float, torch.Tensor]:
    for _ in range(warmup):
        fn(inputs, weight, bias, learning_signal)
    synchronize_if_needed(inputs.device)

    last_grad = weight
    start = time.perf_counter()
    for _ in range(repeats):
        last_grad = fn(inputs, weight, bias, learning_signal)
    synchronize_if_needed(inputs.device)
    return (time.perf_counter() - start) / repeats, last_grad


def memory_peak(
    fn: StepFn,
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    learning_signal: torch.Tensor,
) -> int | None:
    if inputs.device.type != "cuda":
        return None
    torch.cuda.synchronize(inputs.device)
    torch.cuda.reset_peak_memory_stats(inputs.device)
    fn(inputs, weight, bias, learning_signal)
    synchronize_if_needed(inputs.device)
    return torch.cuda.max_memory_allocated(inputs.device)


def benchmark_variant(
    variant: str,
    fn: StepFn,
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    learning_signal: torch.Tensor,
    *,
    warmup: int,
    repeats: int,
) -> BenchmarkResult:
    try:
        update_seconds, grad_weight = time_step(
            fn,
            inputs,
            weight,
            bias,
            learning_signal,
            warmup=warmup,
            repeats=repeats,
        )
        peak_bytes = memory_peak(fn, inputs, weight, bias, learning_signal)
    except Exception as exc:  # noqa: BLE001 - benchmark should report failures.
        synchronize_if_needed(inputs.device)
        return BenchmarkResult(
            variant=variant,
            update_seconds=None,
            peak_bytes=None,
            grad_weight_norm=None,
            error=f"{type(exc).__name__}: {exc}",
        )
    return BenchmarkResult(
        variant=variant,
        update_seconds=update_seconds,
        peak_bytes=peak_bytes,
        grad_weight_norm=float(grad_weight.norm()),
    )


def run_benchmark(args: argparse.Namespace) -> list[BenchmarkResult]:
    torch.manual_seed(args.seed)
    inputs, weight, bias, learning_signal = make_inputs(
        timesteps=args.timesteps,
        batch=args.batch,
        features=args.features,
        neurons=args.neurons,
        device=args.device,
        bias=not args.no_bias,
    )
    variants: list[tuple[str, StepFn]] = [
        ("Online LIF eligibility", lif_online_step),
        ("Online ALIF eligibility", alif_online_step),
        ("Online LIF custom surrogate IR", lif_online_custom_surrogate_step),
        ("Online ALIF custom surrogate IR", alif_online_custom_surrogate_step),
        ("BPTT LIF surrogate", lif_bptt_step),
        ("BPTT ALIF surrogate", alif_bptt_step),
    ]
    return [
        benchmark_variant(
            label,
            fn,
            inputs,
            weight,
            bias,
            learning_signal,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        for label, fn in variants
    ]


def print_markdown(args: argparse.Namespace, results: list[BenchmarkResult]) -> None:
    print("# Online Learning Benchmark")
    print()
    print("Timing excludes warmup. Online variants compute local eligibility gradients;")
    print("BPTT variants build a PyTorch graph through the full surrogate recurrence.")
    print()
    print("## Environment")
    print()
    print(f"- `generated_utc`: `{datetime.now(UTC).isoformat(timespec='seconds')}`")
    print(f"- `device`: `{args.device}`")
    print(f"- `gpu`: `{gpu_name(args.device)}`")
    print(f"- `torch`: `{torch.__version__}`")
    print(f"- `shape`: `T={args.timesteps}, B={args.batch}, F={args.features}, N={args.neurons}`")
    print(f"- `bias`: `{not args.no_bias}`")
    print(f"- `seed`: `{args.seed}`")
    print(f"- `warmup`: `{args.warmup}`")
    print(f"- `repeats`: `{args.repeats}`")
    print()
    print("## Results")
    print()
    print("| Variant | Update ms | Peak MB | Grad Weight Norm | Error |")
    print("|---|---:|---:|---:|---|")
    for result in results:
        grad_norm = "" if result.grad_weight_norm is None else f"{result.grad_weight_norm:.6f}"
        print(
            f"| {result.variant} | "
            f"{format_ms(result.update_seconds)} | "
            f"{format_memory(result.peak_bytes)} | "
            f"{grad_norm} | "
            f"{result.error or ''} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--timesteps", type=int, default=25)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--features", type=int, default=32)
    parser.add_argument("--neurons", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-bias", action="store_true")
    args = parser.parse_args()
    print_markdown(args, run_benchmark(args))


if __name__ == "__main__":
    main()
