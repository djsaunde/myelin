"""Benchmark checkpointed spike-rate training without dense spike outputs."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from datetime import UTC, datetime

import torch

from spiker.autograd import triton_linear_surrogate_lif_checkpoint_rate_function
from spiker.benchmarks.lif import format_memory, format_ms, gpu_name, parse_shape
from spiker.benchmarks.surrogate_backend import memory_checkpoints, time_step
from spiker.checkpointing import CheckpointSize, parse_checkpoint_size, resolve_checkpoint_size
from spiker.kernels import linear_surrogate_lif_forward
from spiker.neurons import LIFParams
from spiker.surrogates import SURROGATE_NAMES, SurrogateName

DEFAULT_SHAPES = [
    (100, 64, 2048),
    (200, 64, 2048),
    (500, 64, 2048),
]


def make_inputs(
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


def dense_checkpoint_step(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    params: LIFParams,
    *,
    surrogate: SurrogateName,
    surrogate_slope: float,
    checkpoint_size: CheckpointSize,
) -> torch.Tensor:
    final_state, spikes = linear_surrogate_lif_forward(
        inputs,
        weight,
        None,
        params,
        surrogate=surrogate,
        surrogate_slope=surrogate_slope,
        backend="triton",
        checkpoint_size=checkpoint_size,
    )
    return spikes.mean().square() + 0.01 * final_state.membrane.square().mean()


def rate_checkpoint_step(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    params: LIFParams,
    *,
    surrogate: SurrogateName,
    surrogate_slope: float,
    checkpoint_size: int,
) -> torch.Tensor:
    final_state, spike_rate = triton_linear_surrogate_lif_checkpoint_rate_function(
        inputs,
        weight,
        None,
        params,
        surrogate=surrogate,
        surrogate_slope=surrogate_slope,
        checkpoint_size=checkpoint_size,
    )
    return spike_rate.square() + 0.01 * final_state.membrane.square().mean()


def benchmark_one(
    *,
    timesteps: int,
    batch: int,
    features: int,
    neurons: int,
    device: str,
    warmup: int,
    repeats: int,
    surrogate: SurrogateName,
    surrogate_slope: float,
    checkpoint_size: CheckpointSize,
) -> dict[str, object]:
    inputs, weight = make_inputs(timesteps, batch, features, neurons, device)
    dense_weight = weight.detach().clone().requires_grad_(True)
    rate_weight = weight.detach().clone().requires_grad_(True)
    params = LIFParams()
    chunk = resolve_checkpoint_size(timesteps, checkpoint_size)

    def dense_step(
        step_inputs: torch.Tensor,
        step_weight: torch.Tensor,
        step_params: LIFParams,
    ) -> torch.Tensor:
        return dense_checkpoint_step(
            step_inputs,
            step_weight,
            step_params,
            surrogate=surrogate,
            surrogate_slope=surrogate_slope,
            checkpoint_size=chunk,
        )

    def rate_step(
        step_inputs: torch.Tensor,
        step_weight: torch.Tensor,
        step_params: LIFParams,
    ) -> torch.Tensor:
        return rate_checkpoint_step(
            step_inputs,
            step_weight,
            step_params,
            surrogate=surrogate,
            surrogate_slope=surrogate_slope,
            checkpoint_size=chunk,
        )

    dense_timing = time_step(
        dense_step, inputs, dense_weight, params, warmup=warmup, repeats=repeats
    )
    dense_memory = memory_checkpoints(dense_step, inputs, dense_weight, params)
    rate_timing = time_step(rate_step, inputs, rate_weight, params, warmup=warmup, repeats=repeats)
    rate_memory = memory_checkpoints(rate_step, inputs, rate_weight, params)

    return {
        "timesteps": timesteps,
        "batch": batch,
        "features": features,
        "neurons": neurons,
        "dense_forward_backward_seconds": dense_timing.forward_backward_seconds,
        "rate_forward_backward_seconds": rate_timing.forward_backward_seconds,
        "dense_backward_peak_bytes": dense_memory.backward_peak_bytes,
        "rate_backward_peak_bytes": rate_memory.backward_peak_bytes,
    }


def run_sweep(
    shapes: Iterable[tuple[int, int, int]],
    *,
    features: int,
    device: str,
    warmup: int,
    repeats: int,
    surrogate: SurrogateName,
    surrogate_slope: float,
    checkpoint_size: CheckpointSize,
) -> list[dict[str, object]]:
    return [
        benchmark_one(
            timesteps=timesteps,
            batch=batch,
            features=features,
            neurons=neurons,
            device=device,
            warmup=warmup,
            repeats=repeats,
            surrogate=surrogate,
            surrogate_slope=surrogate_slope,
            checkpoint_size=checkpoint_size,
        )
        for timesteps, batch, neurons in shapes
    ]


def print_markdown(args: argparse.Namespace, results: list[dict[str, object]]) -> None:
    print("# Checkpoint Spike-Rate Benchmark")
    print()
    print("Compares dense checkpointed spike output against a scalar spike-rate output.")
    print()
    print("## Environment")
    print()
    print(f"- `generated_utc`: `{datetime.now(UTC).isoformat(timespec='seconds')}`")
    print(f"- `device`: `{args.device}`")
    print(f"- `gpu`: `{gpu_name(args.device)}`")
    print(f"- `torch`: `{torch.__version__}`")
    print(f"- `features`: `{args.features}`")
    print(f"- `checkpoint_size`: `{args.checkpoint_size}`")
    resolved_sizes = {
        timesteps: resolve_checkpoint_size(timesteps, args.checkpoint_size)
        for timesteps, _batch, _neurons in (args.shape or DEFAULT_SHAPES)
    }
    print(f"- `resolved_checkpoint_sizes`: `{resolved_sizes}`")
    print(f"- `warmup`: `{args.warmup}`")
    print(f"- `repeats`: `{args.repeats}`")
    print()
    print("## Results")
    print()
    print(
        "| T | Batch | Features | N | Dense Fwd+Bwd ms | Rate Fwd+Bwd ms | "
        "Dense Bwd Peak MB | Rate Bwd Peak MB |"
    )
    print("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for result in results:
        print(
            f"| {result['timesteps']} | {result['batch']} | {result['features']} | "
            f"{result['neurons']} | "
            f"{format_ms(result['dense_forward_backward_seconds'])} | "
            f"{format_ms(result['rate_forward_backward_seconds'])} | "
            f"{format_memory(result['dense_backward_peak_bytes'])} | "
            f"{format_memory(result['rate_backward_peak_bytes'])} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", action="append", type=parse_shape, default=[])
    parser.add_argument("--features", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--surrogate",
        choices=SURROGATE_NAMES,
        default="fast_sigmoid",
    )
    parser.add_argument("--surrogate-slope", type=float, default=5.0)
    parser.add_argument("--checkpoint-size", type=parse_checkpoint_size, default=25)
    args = parser.parse_args()

    results = run_sweep(
        args.shape or DEFAULT_SHAPES,
        features=args.features,
        device=args.device,
        warmup=args.warmup,
        repeats=args.repeats,
        surrogate=args.surrogate,
        surrogate_slope=args.surrogate_slope,
        checkpoint_size=args.checkpoint_size,
    )
    print_markdown(args, results)


if __name__ == "__main__":
    main()
