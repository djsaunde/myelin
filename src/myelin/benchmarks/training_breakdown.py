"""Break down compiled-vs-Triton surrogate LIF training costs."""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

import torch

from myelin._optional import has_triton
from myelin.baselines import synchronize_if_needed
from myelin.benchmarks.lif import format_memory, format_ms, gpu_name
from myelin.checkpointing import parse_checkpoint_size, resolve_checkpoint_size
from myelin.neurons import LIFParams
from myelin.packing import PackedSpikes
from myelin.surrogates import SURROGATE_NAMES, SurrogateName


@dataclass(frozen=True)
class BreakdownResult:
    component: str
    path: str
    seconds: float | None
    peak_bytes: int | None
    note: str
    error: str | None = None


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


def _time_callable(
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


def _benchmark_component(
    *,
    component: str,
    path: str,
    fn: Callable[[], object],
    device: torch.device,
    warmup: int,
    repeats: int,
    note: str,
) -> BreakdownResult:
    try:
        seconds, peak = _time_callable(fn, device=device, warmup=warmup, repeats=repeats)
        return BreakdownResult(component, path, seconds, peak, note)
    except Exception as exc:  # noqa: BLE001 - benchmark should report backend failures.
        synchronize_if_needed(device)
        return BreakdownResult(component, path, None, None, note, f"{type(exc).__name__}: {exc}")


def _benchmark_matmul(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    *,
    warmup: int,
    repeats: int,
) -> BreakdownResult:
    return _benchmark_component(
        component="dense projection",
        path="torch matmul",
        fn=lambda: torch.matmul(inputs, weight),
        device=inputs.device,
        warmup=warmup,
        repeats=repeats,
        note="dense [T,B,F] x [F,N] projection only",
    )


def _benchmark_compiled_training(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    params: LIFParams,
    *,
    warmup: int,
    repeats: int,
    compile_mode: str,
    surrogate: SurrogateName,
    surrogate_slope: float,
) -> BreakdownResult:
    from myelin.benchmarks.surrogate_backend import compile_step, make_step

    step = compile_step(
        make_step(
            backend="torch",
            surrogate=surrogate,
            surrogate_slope=surrogate_slope,
        ),
        compile_mode,
    )
    compiled_weight = weight.detach().clone().requires_grad_(True)

    def run() -> None:
        compiled_weight.grad = None
        loss = step(inputs, compiled_weight, params)
        loss.backward()

    return _benchmark_component(
        component="full training",
        path="torch.compile materialized scalar loss",
        fn=run,
        device=inputs.device,
        warmup=warmup,
        repeats=repeats,
        note="whole scalar loss inside compiled graph",
    )


def _triton_forward_state(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    params: LIFParams,
    *,
    checkpoint_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from myelin.triton import linear_surrogate_lif_checkpoint_forward

    final_state, spikes, chunk_starts = linear_surrogate_lif_checkpoint_forward(
        inputs,
        weight,
        None,
        params,
        checkpoint_size=checkpoint_size,
    )
    return final_state.membrane, spikes, chunk_starts


def _triton_rate_forward_state(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    params: LIFParams,
    *,
    checkpoint_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from myelin.triton import linear_surrogate_lif_checkpoint_rate_forward

    final_state, rates, chunk_starts = linear_surrogate_lif_checkpoint_rate_forward(
        inputs,
        weight,
        None,
        params,
        checkpoint_size=checkpoint_size,
    )
    return final_state.membrane, rates, chunk_starts


def _triton_packed_forward_state(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    params: LIFParams,
    *,
    checkpoint_size: int,
) -> tuple[torch.Tensor, PackedSpikes, torch.Tensor]:
    from myelin.triton import linear_surrogate_lif_checkpoint_packed_forward

    final_state, packed_spikes, chunk_starts = linear_surrogate_lif_checkpoint_packed_forward(
        inputs,
        weight,
        None,
        params,
        checkpoint_size=checkpoint_size,
    )
    return final_state.membrane, packed_spikes, chunk_starts


def _triton_backward_call(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    chunk_starts: torch.Tensor,
    grad_final: torch.Tensor,
    grad_spikes: torch.Tensor | None,
    params: LIFParams,
    *,
    surrogate: SurrogateName,
    surrogate_slope: float,
    checkpoint_size: int,
    needs_input_grad: bool,
    needs_weight_grad: bool,
    grad_spike_rate: torch.Tensor | None = None,
) -> object:
    from myelin.triton import linear_surrogate_lif_checkpoint_backward

    return linear_surrogate_lif_checkpoint_backward(
        inputs,
        weight,
        None,
        chunk_starts,
        grad_final,
        grad_spikes,
        params,
        surrogate=surrogate,
        surrogate_slope=surrogate_slope,
        grad_spike_rate=grad_spike_rate,
        needs_input_grad=needs_input_grad,
        needs_weight_grad=needs_weight_grad,
        needs_bias_grad=False,
        checkpoint_size=checkpoint_size,
    )


def run_benchmark(args: argparse.Namespace) -> list[BreakdownResult]:
    device = torch.device(args.device)
    inputs, weight = make_tensors(
        timesteps=args.timesteps,
        batch=args.batch,
        features=args.features,
        neurons=args.neurons,
        device=args.device,
        seed=args.seed,
    )
    params = LIFParams()
    checkpoint_size = resolve_checkpoint_size(args.timesteps, args.checkpoint_size)
    results = [_benchmark_matmul(inputs, weight, warmup=args.warmup, repeats=args.repeats)]

    if args.no_compile:
        results.append(
            BreakdownResult(
                "full training",
                "torch.compile materialized scalar loss",
                None,
                None,
                "whole scalar loss inside compiled graph",
                "compile disabled",
            )
        )
    else:
        results.append(
            _benchmark_compiled_training(
                inputs,
                weight,
                params,
                warmup=args.warmup,
                repeats=args.repeats,
                compile_mode=args.compile_mode,
                surrogate=args.surrogate,
                surrogate_slope=args.surrogate_slope,
            )
        )

    if device.type != "cuda" or not has_triton():
        missing = "CUDA and Triton are required"
        for component, path, note in (
            ("checkpoint forward", "Triton dense spikes", "fused projection + LIF forward"),
            (
                "checkpoint forward",
                "Triton packed spikes",
                "same forward with direct packed spike output",
            ),
            (
                "checkpoint forward",
                "Triton rate output",
                "same forward contract without dense spikes",
            ),
            ("checkpoint backward", "Triton recurrent only", "no dweight/dinput outputs requested"),
            (
                "checkpoint backward",
                "Triton recurrent + dweight",
                "default training gradient target",
            ),
            (
                "checkpoint backward",
                "Triton recurrent + dweight + dinput",
                "input-gradient variant",
            ),
            ("checkpoint backward", "Triton rate recurrent + dweight", "rate-output backward"),
        ):
            results.append(BreakdownResult(component, path, None, None, note, missing))
        return results

    final_membrane, spikes, chunk_starts = _triton_forward_state(
        inputs,
        weight,
        params,
        checkpoint_size=checkpoint_size,
    )
    grad_final = 0.02 * final_membrane / final_membrane.numel()
    grad_spikes = torch.full_like(spikes, 2.0 / spikes.numel())
    grad_rate = torch.tensor(2.0, dtype=inputs.dtype, device=inputs.device)

    results.append(
        _benchmark_component(
            component="checkpoint forward",
            path="Triton dense spikes",
            fn=lambda: _triton_forward_state(
                inputs,
                weight,
                params,
                checkpoint_size=checkpoint_size,
            ),
            device=device,
            warmup=args.warmup,
            repeats=args.repeats,
            note="fused projection + LIF forward returning [T,B,N] spikes",
        )
    )
    results.append(
        _benchmark_component(
            component="checkpoint forward",
            path="Triton packed spikes",
            fn=lambda: _triton_packed_forward_state(
                inputs,
                weight,
                params,
                checkpoint_size=checkpoint_size,
            ),
            device=device,
            warmup=args.warmup,
            repeats=args.repeats,
            note="fused projection + LIF forward returning packed [T,B,ceil(N/32)] spikes",
        )
    )
    results.append(
        _benchmark_component(
            component="checkpoint forward",
            path="Triton rate output",
            fn=lambda: _triton_rate_forward_state(
                inputs,
                weight,
                params,
                checkpoint_size=checkpoint_size,
            ),
            device=device,
            warmup=args.warmup,
            repeats=args.repeats,
            note="fused projection + LIF forward returning [B,N] rates",
        )
    )
    results.append(
        _benchmark_component(
            component="checkpoint backward",
            path="Triton recurrent only",
            fn=lambda: _triton_backward_call(
                inputs,
                weight,
                chunk_starts,
                grad_final,
                grad_spikes,
                params,
                surrogate=args.surrogate,
                surrogate_slope=args.surrogate_slope,
                checkpoint_size=checkpoint_size,
                needs_input_grad=False,
                needs_weight_grad=False,
            ),
            device=device,
            warmup=args.warmup,
            repeats=args.repeats,
            note="proxy for reverse recurrence; kernel grid still follows training layout",
        )
    )
    results.append(
        _benchmark_component(
            component="checkpoint backward",
            path="Triton recurrent + dweight",
            fn=lambda: _triton_backward_call(
                inputs,
                weight,
                chunk_starts,
                grad_final,
                grad_spikes,
                params,
                surrogate=args.surrogate,
                surrogate_slope=args.surrogate_slope,
                checkpoint_size=checkpoint_size,
                needs_input_grad=False,
                needs_weight_grad=True,
            ),
            device=device,
            warmup=args.warmup,
            repeats=args.repeats,
            note="default training target when input gradients are not needed",
        )
    )
    results.append(
        _benchmark_component(
            component="checkpoint backward",
            path="Triton recurrent + dweight + dinput",
            fn=lambda: _triton_backward_call(
                inputs,
                weight,
                chunk_starts,
                grad_final,
                grad_spikes,
                params,
                surrogate=args.surrogate,
                surrogate_slope=args.surrogate_slope,
                checkpoint_size=checkpoint_size,
                needs_input_grad=True,
                needs_weight_grad=True,
            ),
            device=device,
            warmup=args.warmup,
            repeats=args.repeats,
            note="training target when gradients through inputs are requested",
        )
    )
    rate_final, _rates, rate_chunk_starts = _triton_rate_forward_state(
        inputs,
        weight,
        params,
        checkpoint_size=checkpoint_size,
    )
    rate_grad_final = 0.02 * rate_final / rate_final.numel()
    results.append(
        _benchmark_component(
            component="checkpoint backward",
            path="Triton rate recurrent + dweight",
            fn=lambda: _triton_backward_call(
                inputs,
                weight,
                rate_chunk_starts,
                rate_grad_final,
                None,
                params,
                surrogate=args.surrogate,
                surrogate_slope=args.surrogate_slope,
                checkpoint_size=checkpoint_size,
                needs_input_grad=False,
                needs_weight_grad=True,
                grad_spike_rate=grad_rate,
            ),
            device=device,
            warmup=args.warmup,
            repeats=args.repeats,
            note="rate-output backward avoids dense grad_spikes",
        )
    )
    return results


def print_markdown(args: argparse.Namespace, results: list[BreakdownResult]) -> None:
    print("# Training Breakdown")
    print()
    print("Breaks surrogate LIF training into projection, forward, and backward components.")
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
    print(f"- `checkpoint_size`: `{args.checkpoint_size}`")
    print(
        f"- `resolved_checkpoint_size`: "
        f"`{resolve_checkpoint_size(args.timesteps, args.checkpoint_size)}`"
    )
    print(f"- `compile_mode`: `{args.compile_mode}`")
    print(f"- `no_compile`: `{args.no_compile}`")
    print(f"- `warmup`: `{args.warmup}`")
    print(f"- `repeats`: `{args.repeats}`")
    print()
    print("## Results")
    print()
    print("| Component | Path | ms | Peak MB | Note | Error |")
    print("|---|---|---:|---:|---|---|")
    for result in results:
        print(
            f"| {result.component} | "
            f"{result.path} | "
            f"{format_ms(result.seconds)} | "
            f"{format_memory(result.peak_bytes)} | "
            f"{result.note} | "
            f"{result.error or ''} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--features", type=int, default=128)
    parser.add_argument("--neurons", type=int, default=2048)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint-size", type=parse_checkpoint_size, default=25)
    parser.add_argument(
        "--compile-mode",
        choices=["default", "reduce-overhead", "max-autotune"],
        default="reduce-overhead",
    )
    parser.add_argument(
        "--surrogate",
        choices=SURROGATE_NAMES,
        default="fast_sigmoid",
    )
    parser.add_argument("--surrogate-slope", type=float, default=5.0)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print_markdown(args, run_benchmark(args))


if __name__ == "__main__":
    main()
