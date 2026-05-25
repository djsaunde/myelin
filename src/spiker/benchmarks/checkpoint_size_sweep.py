"""Sweep checkpoint chunk sizes for Triton checkpointed LIF training."""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

import torch

from spiker.baselines import synchronize_if_needed
from spiker.benchmarks.currents_audit import (
    AuditResult,
    audit_variant,
    expected_current_bytes,
    make_triton_checkpoint_rate_step,
)
from spiker.benchmarks.lif import format_memory, format_ms, gpu_name
from spiker.benchmarks.surrogate_backend import make_inputs, make_triton_checkpoint_synapse_step
from spiker.neurons import LIFParams
from spiker.surrogates import SURROGATE_NAMES, SurrogateName

CheckpointVariant = Literal["dense", "rate", "generated_rate", "replay_rate"]
ManualStep = Callable[[], None]


@dataclass(frozen=True)
class SweepResult:
    checkpoint_size: int
    chunks: int
    variant: CheckpointVariant
    audit: AuditResult
    forward_backward_seconds_override: float | None = None

    @property
    def forward_backward_seconds(self) -> float | None:
        if self.forward_backward_seconds_override is not None:
            return self.forward_backward_seconds_override
        if self.audit.split_forward_seconds is None or self.audit.backward_seconds is None:
            return None
        return self.audit.split_forward_seconds + self.audit.backward_seconds


def parse_checkpoint_sizes(value: str) -> tuple[int, ...]:
    sizes = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not sizes:
        raise argparse.ArgumentTypeError("checkpoint size list must not be empty")
    if any(size <= 0 for size in sizes):
        raise argparse.ArgumentTypeError("checkpoint sizes must be positive")
    return sizes


def pre_reset_scratch_bytes(
    *,
    checkpoint_size: int,
    timesteps: int,
    batch: int,
    neurons: int,
    dtype: torch.dtype,
) -> int:
    chunk = min(checkpoint_size, timesteps)
    return chunk * batch * neurons * torch.tensor([], dtype=dtype).element_size()


def rate_pareto_frontier(results: list[SweepResult]) -> tuple[SweepResult, ...]:
    """Return non-dominated rate-output choices by latency and memory.

    Dense-output rows answer a different API-contract question, so this frontier
    only compares the low-memory rate variants.
    """

    candidates = [
        result
        for result in results
        if result.variant in {"rate", "generated_rate", "replay_rate"}
        and result.forward_backward_seconds is not None
        and result.audit.backward_increment_bytes is not None
        and result.audit.error is None
    ]
    frontier: list[SweepResult] = []
    for index, candidate in enumerate(candidates):
        candidate_time = candidate.forward_backward_seconds
        candidate_memory = candidate.audit.backward_increment_bytes
        if candidate_time is None or candidate_memory is None:
            continue
        dominated = False
        for other_index, other in enumerate(candidates):
            if other_index == index:
                continue
            other_time = other.forward_backward_seconds
            other_memory = other.audit.backward_increment_bytes
            if other_time is None or other_memory is None:
                continue
            no_worse = other_time <= candidate_time and other_memory <= candidate_memory
            strictly_better = other_time < candidate_time or other_memory < candidate_memory
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)
    return tuple(
        sorted(
            frontier,
            key=lambda result: (
                result.audit.backward_increment_bytes,
                result.forward_backward_seconds,
                result.variant,
                result.checkpoint_size,
            ),
        )
    )


def _clear_gradients(tensors: tuple[torch.Tensor, ...]) -> None:
    for tensor in tensors:
        tensor.grad = None


def audit_manual_variant(
    label: str,
    fn: ManualStep,
    grad_tensors: tuple[torch.Tensor, ...],
    device: torch.device,
    *,
    warmup: int,
    repeats: int,
) -> tuple[AuditResult, float | None]:
    """Audit a manual forward+backward path that does not use autograd timing helpers."""

    try:
        for _ in range(warmup):
            _clear_gradients(grad_tensors)
            fn()
        synchronize_if_needed(device)

        elapsed = 0.0
        for _ in range(repeats):
            _clear_gradients(grad_tensors)
            start = time.perf_counter()
            fn()
            synchronize_if_needed(device)
            elapsed += time.perf_counter() - start
        forward_backward_seconds = elapsed / repeats

        allocated = None
        peak = None
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            allocated = torch.cuda.memory_allocated(device)
            _clear_gradients(grad_tensors)
            fn()
            synchronize_if_needed(device)
            peak = torch.cuda.max_memory_allocated(device)
    except Exception as exc:  # noqa: BLE001 - benchmark should report backend failures.
        synchronize_if_needed(device)
        return (
            AuditResult(
                label=label,
                split_forward_seconds=None,
                backward_seconds=None,
                allocated_bytes=None,
                forward_peak_bytes=None,
                backward_peak_bytes=None,
                error=f"{type(exc).__name__}: {exc}",
            ),
            None,
        )

    return (
        AuditResult(
            label=label,
            split_forward_seconds=None,
            backward_seconds=None,
            allocated_bytes=allocated,
            forward_peak_bytes=peak,
            backward_peak_bytes=peak,
        ),
        forward_backward_seconds,
    )


def make_triton_checkpoint_replay_rate_step(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    params: LIFParams,
    *,
    surrogate: SurrogateName,
    surrogate_slope: float,
    checkpoint_size: int,
) -> ManualStep:
    """Return a manual scalar-rate step using the replay-no-scratch backward."""

    from spiker.triton import (
        linear_surrogate_lif_checkpoint_backward_replay_weight_bias,
        linear_surrogate_lif_checkpoint_rate_forward,
    )

    def step() -> None:
        final_state, spike_rates, chunk_start_membranes = (
            linear_surrogate_lif_checkpoint_rate_forward(
                inputs,
                weight,
                None,
                params,
                checkpoint_size=checkpoint_size,
            )
        )
        spike_rate = spike_rates.mean()
        grad_final = 0.02 * final_state.membrane / final_state.membrane.numel()
        grad_rate = 2.0 * spike_rate
        grad_weight, _grad_bias = linear_surrogate_lif_checkpoint_backward_replay_weight_bias(
            inputs,
            weight,
            None,
            chunk_start_membranes,
            grad_final,
            None,
            params,
            surrogate=surrogate,
            surrogate_slope=surrogate_slope,
            grad_spike_rate=grad_rate,
            needs_bias_grad=False,
            checkpoint_size=checkpoint_size,
        )
        weight.grad = grad_weight

    return step


def run_sweep(args: argparse.Namespace) -> list[SweepResult]:
    inputs, base_weight = make_inputs(
        args.timesteps,
        args.batch,
        args.features,
        args.neurons,
        args.device,
        input_grad=False,
    )
    params = LIFParams()
    surrogate: SurrogateName = args.surrogate
    results: list[SweepResult] = []

    for requested_checkpoint_size in args.checkpoint_sizes:
        checkpoint_size = min(requested_checkpoint_size, args.timesteps)
        chunks = (args.timesteps + checkpoint_size - 1) // checkpoint_size
        variants = [
            (
                "dense",
                make_triton_checkpoint_synapse_step(
                    surrogate=surrogate,
                    surrogate_slope=args.surrogate_slope,
                    checkpoint_size=checkpoint_size,
                ),
            ),
            (
                "rate",
                make_triton_checkpoint_rate_step(
                    surrogate=surrogate,
                    surrogate_slope=args.surrogate_slope,
                    checkpoint_size=checkpoint_size,
                    backend="triton",
                ),
            ),
            (
                "generated_rate",
                make_triton_checkpoint_rate_step(
                    surrogate=surrogate,
                    surrogate_slope=args.surrogate_slope,
                    checkpoint_size=checkpoint_size,
                    backend="triton_generated",
                ),
            ),
        ]
        for variant, step in variants:
            weight = base_weight.detach().clone().requires_grad_(True)
            audit = audit_variant(
                variant,
                step,
                inputs,
                weight,
                params,
                warmup=args.warmup,
                repeats=args.repeats,
            )
            results.append(
                SweepResult(
                    checkpoint_size=checkpoint_size,
                    chunks=chunks,
                    variant=variant,  # pyright: ignore[reportArgumentType]
                    audit=audit,
                )
            )
            synchronize_if_needed(inputs.device)

        replay_weight = base_weight.detach().clone().requires_grad_(True)
        replay_audit, replay_seconds = audit_manual_variant(
            "replay_rate",
            make_triton_checkpoint_replay_rate_step(
                inputs,
                replay_weight,
                params,
                surrogate=surrogate,
                surrogate_slope=args.surrogate_slope,
                checkpoint_size=checkpoint_size,
            ),
            (replay_weight,),
            inputs.device,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        results.append(
            SweepResult(
                checkpoint_size=checkpoint_size,
                chunks=chunks,
                variant="replay_rate",
                audit=replay_audit,
                forward_backward_seconds_override=replay_seconds,
            )
        )
        synchronize_if_needed(inputs.device)

    return results


def print_markdown(args: argparse.Namespace, results: list[SweepResult]) -> None:
    current_bytes = expected_current_bytes(
        timesteps=args.timesteps,
        batch=args.batch,
        neurons=args.neurons,
        dtype=torch.float32,
    )
    print("# Checkpoint Size Sweep")
    print()
    print("Sweeps Triton checkpoint chunk size for dense and rate-output LIF training.")
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
    print(f"- `checkpoint_sizes`: `{args.checkpoint_sizes}`")
    print(f"- `expected_currents_mb`: `{format_memory(current_bytes)}`")
    print(f"- `warmup`: `{args.warmup}`")
    print(f"- `repeats`: `{args.repeats}`")
    print()
    print("## Results")
    print()
    print(
        "| Checkpoint | Chunks | Variant | Fwd ms | Bwd ms | Fwd+Bwd ms | "
        "Expected Scratch MB | Fwd Increment MB | Bwd Increment MB | Error |"
    )
    print("|---:|---:|---|---:|---:|---:|---:|---:|---:|---|")
    for result in results:
        scratch_bytes = pre_reset_scratch_bytes(
            checkpoint_size=result.checkpoint_size,
            timesteps=args.timesteps,
            batch=args.batch,
            neurons=args.neurons,
            dtype=torch.float32,
        )
        print(
            f"| {result.checkpoint_size} | "
            f"{result.chunks} | "
            f"{result.variant} | "
            f"{format_ms(result.audit.split_forward_seconds)} | "
            f"{format_ms(result.audit.backward_seconds)} | "
            f"{format_ms(result.forward_backward_seconds)} | "
            f"{format_memory(scratch_bytes)} | "
            f"{format_memory(result.audit.forward_increment_bytes)} | "
            f"{format_memory(result.audit.backward_increment_bytes)} | "
            f"{result.audit.error or ''} |"
        )
    print()
    frontier = rate_pareto_frontier(results)
    if frontier:
        print("## Rate Pareto Frontier")
        print()
        print(
            "Non-dominated rate-output choices when minimizing fwd+bwd latency "
            "and backward peak-memory increment."
        )
        print()
        print("| Checkpoint | Chunks | Variant | Fwd+Bwd ms | Bwd Increment MB |")
        print("|---:|---:|---|---:|---:|")
        for result in frontier:
            print(
                f"| {result.checkpoint_size} | "
                f"{result.chunks} | "
                f"{result.variant} | "
                f"{format_ms(result.forward_backward_seconds)} | "
                f"{format_memory(result.audit.backward_increment_bytes)} |"
            )
        print()
    print("## Takeaway")
    print()
    print(
        "The replay-rate variant avoids the chunk-sized pre-reset scratch by "
        "replaying chunk prefixes inside the backward kernel. It gives a lower "
        "memory floor, but its latency grows quickly with checkpoint size."
    )
    print()
    print(
        "Use the regular `rate` rows for the default speed/memory tradeoff and "
        "`replay_rate` as a memory-floor experiment."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--features", type=int, default=128)
    parser.add_argument("--neurons", type=int, default=2048)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--checkpoint-sizes", type=parse_checkpoint_sizes, default=(5, 10, 25, 50))
    parser.add_argument(
        "--surrogate",
        choices=SURROGATE_NAMES,
        default="fast_sigmoid",
    )
    parser.add_argument("--surrogate-slope", type=float, default=5.0)
    parser.add_argument(
        "--matmul-precision",
        choices=["current", "highest", "high", "medium"],
        default="current",
    )
    args = parser.parse_args()

    original_matmul_precision = torch.get_float32_matmul_precision()
    if args.matmul_precision != "current":
        torch.set_float32_matmul_precision(args.matmul_precision)
    try:
        print_markdown(args, run_sweep(args))
    finally:
        if args.matmul_precision != "current":
            torch.set_float32_matmul_precision(original_matmul_precision)


if __name__ == "__main__":
    main()
