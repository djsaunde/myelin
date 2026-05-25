"""Benchmark custom NeuronIR through the module API."""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal, cast

import torch

from spiker import CustomNeuronCell, NeuronBuilder, TimeUnroll, evaluate_neuron_unroll
from spiker._optional import has_triton
from spiker.baselines import synchronize_if_needed
from spiker.benchmarks.lif import format_memory, format_ms, format_speedup, gpu_name
from spiker.dsl import NeuronIR, where

Variant = Literal["lif", "alif", "refractory_lif"]
VARIANTS: tuple[Variant, ...] = ("lif", "alif", "refractory_lif")


@dataclass(frozen=True)
class Result:
    variant: Variant
    backend: str
    seconds: float | None
    speedup_vs_torch: float | None
    peak_bytes: int | None
    state_max_error: float | None
    spike_mismatch_rate: float | None
    error: str | None = None


def build_custom_lif_ir() -> NeuronIR:
    builder = NeuronBuilder("benchmark_custom_lif")
    membrane = builder.state("membrane")
    current = builder.input("input_current")
    decay = builder.param("decay")
    threshold = builder.param("threshold")
    reset = builder.param("reset")

    pre_reset = membrane * decay + current
    did_spike = pre_reset.ge(threshold)
    spike = where(did_spike, 1.0, 0.0)
    return builder.build(
        next_state={"membrane": where(did_spike, reset, pre_reset)},
        outputs={"spike": spike},
    )


def build_custom_alif_ir() -> NeuronIR:
    builder = NeuronBuilder("benchmark_custom_alif")
    membrane = builder.state("membrane")
    adaptation = builder.state("adaptation")
    current = builder.input("input_current")
    decay = builder.param("decay")
    adaptation_decay = builder.param("adaptation_decay")
    threshold = builder.param("threshold")
    reset = builder.param("reset")
    beta = builder.param("beta")

    pre_reset = membrane * decay + current
    adaptive_threshold = threshold + beta * adaptation
    did_spike = pre_reset.ge(adaptive_threshold)
    spike = where(did_spike, 1.0, 0.0)
    return builder.build(
        next_state={
            "membrane": where(did_spike, reset, pre_reset),
            "adaptation": adaptation * adaptation_decay + spike,
        },
        outputs={"spike": spike},
    )


def build_custom_refractory_lif_ir() -> NeuronIR:
    builder = NeuronBuilder("benchmark_custom_refractory_lif")
    membrane = builder.state("membrane")
    refractory = builder.state("refractory")
    current = builder.input("input_current")
    decay = builder.param("decay")
    threshold = builder.param("threshold")
    reset = builder.param("reset")
    refractory_steps = builder.param("refractory_steps")

    active_refractory = refractory.ge(0.5)
    pre_reset = membrane * decay + current
    candidate_spike = pre_reset.ge(threshold)
    spike = where(active_refractory, 0.0, where(candidate_spike, 1.0, 0.0))
    did_spike = spike.ge(0.5)
    return builder.build(
        next_state={
            "membrane": where(active_refractory, reset, where(did_spike, reset, pre_reset)),
            "refractory": where(
                did_spike,
                refractory_steps,
                where(active_refractory, refractory - 1.0, 0.0),
            ),
        },
        outputs={"spike": spike},
    )


def build_variant_ir(variant: Variant) -> NeuronIR:
    if variant == "lif":
        return build_custom_lif_ir()
    if variant == "alif":
        return build_custom_alif_ir()
    if variant == "refractory_lif":
        return build_custom_refractory_lif_ir()
    raise ValueError(f"unsupported custom neuron variant: {variant}")


def params_for_variant(variant: Variant) -> dict[str, float]:
    if variant == "lif":
        return {
            "decay": 0.85,
            "threshold": 1.0,
            "reset": 0.0,
        }
    if variant == "alif":
        return {
            "decay": 0.85,
            "adaptation_decay": 0.9,
            "threshold": 1.0,
            "reset": 0.0,
            "beta": 0.5,
        }
    if variant == "refractory_lif":
        return {
            "decay": 0.85,
            "threshold": 1.0,
            "reset": 0.0,
            "refractory_steps": 2.0,
        }
    raise ValueError(f"unsupported custom neuron variant: {variant}")


def initial_state_for_variant(
    variant: Variant,
    *,
    batch: int,
    neurons: int,
    device: str,
) -> dict[str, torch.Tensor]:
    if variant == "lif":
        return {
            "membrane": torch.rand((batch, neurons), device=device) * 0.2,
        }
    if variant == "alif":
        return {
            "membrane": torch.rand((batch, neurons), device=device) * 0.2,
            "adaptation": torch.rand((batch, neurons), device=device) * 0.1,
        }
    if variant == "refractory_lif":
        return {
            "membrane": torch.rand((batch, neurons), device=device) * 0.2,
            "refractory": torch.zeros((batch, neurons), device=device),
        }
    raise ValueError(f"unsupported custom neuron variant: {variant}")


def max_state_error(
    actual: dict[str, torch.Tensor],
    expected: dict[str, torch.Tensor],
) -> float:
    return max(float((actual[name] - expected[name]).abs().max().item()) for name in expected)


def spike_mismatch_rate(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float((actual != expected).to(torch.float32).mean().item())


def time_forward(
    fn: Callable[[], object],
    *,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> tuple[float, int | None]:
    for _ in range(warmup):
        fn()
    synchronize_if_needed(device)

    peak = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    synchronize_if_needed(device)
    seconds = (time.perf_counter() - start) / repeats
    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated(device)
    return seconds, peak


def run_backend(
    *,
    variant: Variant,
    backend: str,
    ir: NeuronIR,
    params: dict[str, float],
    inputs: torch.Tensor,
    initial_state: dict[str, torch.Tensor],
    expected_state: dict[str, torch.Tensor],
    expected_spikes: torch.Tensor,
    torch_seconds: float | None,
    warmup: int,
    repeats: int,
) -> Result:
    try:
        unroll = TimeUnroll(
            CustomNeuronCell(ir, params),
            backend=backend,  # pyright: ignore[reportArgumentType]
        )
        actual_state, actual_spikes = unroll(inputs, initial_state)
        if not isinstance(actual_state, dict):
            raise TypeError("custom neuron module returned non-dict state")
        seconds, peak = time_forward(
            lambda: unroll(inputs, initial_state),
            device=inputs.device,
            warmup=warmup,
            repeats=repeats,
        )
        speedup = None if torch_seconds is None else torch_seconds / seconds
        return Result(
            variant=variant,
            backend=backend,
            seconds=seconds,
            speedup_vs_torch=speedup,
            peak_bytes=peak,
            state_max_error=max_state_error(actual_state, expected_state),
            spike_mismatch_rate=spike_mismatch_rate(actual_spikes, expected_spikes),
        )
    except Exception as exc:  # noqa: BLE001 - benchmark should report backend failures.
        return Result(
            variant=variant,
            backend=backend,
            seconds=None,
            speedup_vs_torch=None,
            peak_bytes=None,
            state_max_error=None,
            spike_mismatch_rate=None,
            error=f"{type(exc).__name__}: {exc}",
        )


def print_markdown(args: argparse.Namespace, results: list[Result]) -> None:
    print("# Custom Neuron Module Benchmark")
    print()
    print("## Environment")
    print()
    print(f"- `generated_utc`: `{datetime.now(UTC).isoformat(timespec='seconds')}`")
    print(f"- `device`: `{args.device}`")
    print(f"- `gpu`: `{gpu_name(args.device)}`")
    print(f"- `torch`: `{torch.__version__}`")
    print(f"- `cuda_available`: `{torch.cuda.is_available()}`")
    print(f"- `cuda_version`: `{torch.version.cuda}`")
    print(f"- `variant`: `{args.variant}`")
    print(f"- `shape`: `T={args.timesteps}, B={args.batch}, N={args.neurons}`")
    print(f"- `warmup`: `{args.warmup}`")
    print(f"- `repeats`: `{args.repeats}`")
    print()
    print("## Workload")
    print()
    descriptions = {
        "lif": "Custom hard-reset LIF `NeuronIR` matching the generated-backward ABI",
        "alif": "Custom two-state ALIF-style `NeuronIR`",
        "refractory_lif": "Custom refractory-LIF `NeuronIR` with counter-like state",
    }
    selected_variants = tuple(dict.fromkeys(result.variant for result in results))
    for variant in selected_variants:
        print(f"- `{variant}`: {descriptions[variant]}")
    print()
    print("Each variant is wrapped in `CustomNeuronCell` and `TimeUnroll`.")
    print()
    print("## Results")
    print()
    print(
        "| Variant | Backend | Forward ms | Speedup vs torch | Peak MB | "
        "State Max Error | Spike Mismatch Rate | Error |"
    )
    print("|---|---|---:|---:|---:|---:|---:|---|")
    for result in results:
        state_error = "" if result.state_max_error is None else f"{result.state_max_error:.3e}"
        mismatch_rate = (
            "" if result.spike_mismatch_rate is None else f"{result.spike_mismatch_rate:.3e}"
        )
        print(
            f"| {result.variant} | "
            f"{result.backend} | "
            f"{format_ms(result.seconds)} | "
            f"{format_speedup(result.speedup_vs_torch)} | "
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
    parser.add_argument("--variant", choices=(*VARIANTS, "all"), default="alif")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()

    inputs = torch.rand((args.timesteps, args.batch, args.neurons), device=args.device)

    results: list[Result] = []
    selected_variants: tuple[Variant, ...] = (
        VARIANTS if args.variant == "all" else (cast(Variant, args.variant),)
    )
    for variant in selected_variants:
        ir = build_variant_ir(variant)
        params = params_for_variant(variant)
        initial_state = initial_state_for_variant(
            variant,
            batch=args.batch,
            neurons=args.neurons,
            device=args.device,
        )
        expected_state, expected_spikes = evaluate_neuron_unroll(ir, inputs, initial_state, params)

        backends = ["torch"]
        if torch.device(args.device).type == "cuda" and has_triton():
            backends.extend(["auto", "triton_generated"])

        torch_seconds = None
        for backend in backends:
            result = run_backend(
                variant=variant,
                backend=backend,
                ir=ir,
                params=params,
                inputs=inputs,
                initial_state=initial_state,
                expected_state=expected_state,
                expected_spikes=expected_spikes,
                torch_seconds=torch_seconds,
                warmup=args.warmup,
                repeats=args.repeats,
            )
            if backend == "torch":
                torch_seconds = result.seconds
                if result.seconds is not None:
                    result = Result(
                        variant=result.variant,
                        backend=result.backend,
                        seconds=result.seconds,
                        speedup_vs_torch=1.0,
                        peak_bytes=result.peak_bytes,
                        state_max_error=result.state_max_error,
                        spike_mismatch_rate=result.spike_mismatch_rate,
                        error=result.error,
                    )
            results.append(result)

    print_markdown(args, results)


if __name__ == "__main__":
    main()
