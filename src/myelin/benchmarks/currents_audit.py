"""Audit whether backend paths appear to materialize dense current tensors."""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

import torch

from myelin._optional import has_triton
from myelin.baselines import compiled_available, synchronize_if_needed
from myelin.benchmarks.lif import format_memory, format_ms, gpu_name
from myelin.benchmarks.surrogate_backend import (
    StepFn,
    compile_step,
    make_inputs,
    make_step,
    make_stream_synapse_step,
    make_triton_checkpoint_synapse_step,
    make_triton_stream_synapse_step,
    time_forward_backward,
)
from myelin.checkpointing import CheckpointSize, parse_checkpoint_size, resolve_checkpoint_size
from myelin.neurons import LIFParams
from myelin.surrogates import SURROGATE_NAMES, SurrogateName

RateBackend = Literal["triton", "triton_generated"]


@dataclass(frozen=True)
class AuditResult:
    label: str
    split_forward_seconds: float | None
    backward_seconds: float | None
    allocated_bytes: int | None
    forward_peak_bytes: int | None
    backward_peak_bytes: int | None
    error: str | None = None

    @property
    def forward_increment_bytes(self) -> int | None:
        if self.allocated_bytes is None or self.forward_peak_bytes is None:
            return None
        return self.forward_peak_bytes - self.allocated_bytes

    @property
    def backward_increment_bytes(self) -> int | None:
        if self.allocated_bytes is None or self.backward_peak_bytes is None:
            return None
        return self.backward_peak_bytes - self.allocated_bytes

    def forward_extra_over_dense_output_bytes(self, dense_output_bytes: int) -> int | None:
        increment = self.forward_increment_bytes
        if increment is None:
            return None
        return max(0, increment - dense_output_bytes)

    def backward_extra_over_dense_output_bytes(self, dense_output_bytes: int) -> int | None:
        increment = self.backward_increment_bytes
        if increment is None:
            return None
        return max(0, increment - dense_output_bytes)


def measure_memory(
    fn: StepFn,
    inputs: torch.Tensor,
    weight: torch.Tensor,
    params: LIFParams,
) -> tuple[int | None, int | None, int | None]:
    device = inputs.device
    if device.type != "cuda":
        return None, None, None

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    allocated = torch.cuda.memory_allocated(device)

    weight.grad = None
    loss = fn(inputs, weight, params)
    synchronize_if_needed(device)
    forward_peak = torch.cuda.max_memory_allocated(device)

    loss.backward()
    synchronize_if_needed(device)
    backward_peak = torch.cuda.max_memory_allocated(device)
    return allocated, forward_peak, backward_peak


def make_triton_checkpoint_rate_step(
    *,
    surrogate: SurrogateName,
    surrogate_slope: float,
    checkpoint_size: CheckpointSize,
    backend: RateBackend = "triton",
) -> StepFn:
    """Return a scalar-loss step that avoids dense spike output materialization."""

    def step(inputs: torch.Tensor, weight: torch.Tensor, params: LIFParams) -> torch.Tensor:
        from myelin.kernels import linear_surrogate_lif_rate_forward

        final_state, spike_rate = linear_surrogate_lif_rate_forward(
            inputs,
            weight,
            None,
            params,
            surrogate=surrogate,
            surrogate_slope=surrogate_slope,
            hard_forward=True,
            backend=backend,
            checkpoint_size=checkpoint_size,
            reduction="mean",
        )
        return spike_rate.square() + 0.01 * final_state.membrane.square().mean()

    return step


def audit_variant(
    label: str,
    fn: StepFn,
    inputs: torch.Tensor,
    weight: torch.Tensor,
    params: LIFParams,
    *,
    warmup: int,
    repeats: int,
) -> AuditResult:
    try:
        split_forward_seconds, backward_seconds, _ = time_forward_backward(
            fn,
            inputs,
            weight,
            params,
            warmup=warmup,
            repeats=repeats,
        )
        allocated, forward_peak, backward_peak = measure_memory(fn, inputs, weight, params)
    except Exception as exc:  # noqa: BLE001 - benchmark should report backend failures.
        synchronize_if_needed(inputs.device)
        return AuditResult(
            label=label,
            split_forward_seconds=None,
            backward_seconds=None,
            allocated_bytes=None,
            forward_peak_bytes=None,
            backward_peak_bytes=None,
            error=f"{type(exc).__name__}: {exc}",
        )

    return AuditResult(
        label=label,
        split_forward_seconds=split_forward_seconds,
        backward_seconds=backward_seconds,
        allocated_bytes=allocated,
        forward_peak_bytes=forward_peak,
        backward_peak_bytes=backward_peak,
    )


def expected_current_bytes(
    *,
    timesteps: int,
    batch: int,
    neurons: int,
    dtype: torch.dtype,
) -> int:
    return timesteps * batch * neurons * torch.tensor([], dtype=dtype).element_size()


def expected_chunk_start_bytes(
    *,
    timesteps: int,
    batch: int,
    neurons: int,
    checkpoint_size: CheckpointSize,
    dtype: torch.dtype,
) -> int:
    chunk_size = resolve_checkpoint_size(timesteps, checkpoint_size)
    num_chunks = (timesteps + chunk_size - 1) // chunk_size
    return num_chunks * batch * neurons * torch.tensor([], dtype=dtype).element_size()


def expected_checkpoint_scratch_bytes(
    *,
    timesteps: int,
    batch: int,
    neurons: int,
    checkpoint_size: CheckpointSize,
    dtype: torch.dtype,
) -> int:
    chunk_size = resolve_checkpoint_size(timesteps, checkpoint_size)
    return chunk_size * batch * neurons * torch.tensor([], dtype=dtype).element_size()


def exposes_dense_spike_output(label: str) -> bool:
    """Return whether this audit variant exposes dense `[T, B, N]` spikes."""

    return label in {
        "Eager materialized currents",
        "PyTorch streamed custom autograd",
        "Triton fused synapse full trace",
        "Triton checkpoint recompute",
    }


def run_audit(args: argparse.Namespace) -> list[AuditResult]:
    inputs, eager_weight = make_inputs(
        args.timesteps,
        args.batch,
        args.features,
        args.neurons,
        args.device,
        input_grad=False,
    )
    params = LIFParams()
    surrogate: SurrogateName = args.surrogate
    resolved_checkpoint_size = resolve_checkpoint_size(args.timesteps, args.checkpoint_size)

    variants: list[tuple[str, StepFn, torch.Tensor]] = [
        (
            "Eager materialized currents",
            make_step(
                backend="torch",
                surrogate=surrogate,
                surrogate_slope=args.surrogate_slope,
            ),
            eager_weight,
        ),
        (
            "PyTorch streamed custom autograd",
            make_stream_synapse_step(
                surrogate=surrogate,
                surrogate_slope=args.surrogate_slope,
            ),
            eager_weight.detach().clone().requires_grad_(True),
        ),
    ]

    if compiled_available() and not args.no_compile:
        compiled_step = compile_step(variants[0][1], args.compile_mode)
        variants.insert(
            1,
            (
                "torch.compile materialized graph",
                compiled_step,
                eager_weight.detach().clone().requires_grad_(True),
            ),
        )

    device = torch.device(args.device)
    if device.type == "cuda" and has_triton():
        variants.extend(
            [
                (
                    "Triton fused synapse full trace",
                    make_triton_stream_synapse_step(
                        surrogate=surrogate,
                        surrogate_slope=args.surrogate_slope,
                    ),
                    eager_weight.detach().clone().requires_grad_(True),
                ),
                (
                    "Triton checkpoint recompute",
                    make_triton_checkpoint_synapse_step(
                        surrogate=surrogate,
                        surrogate_slope=args.surrogate_slope,
                        checkpoint_size=resolved_checkpoint_size,
                    ),
                    eager_weight.detach().clone().requires_grad_(True),
                ),
                (
                    "Triton checkpoint rate output",
                    make_triton_checkpoint_rate_step(
                        surrogate=surrogate,
                        surrogate_slope=args.surrogate_slope,
                        checkpoint_size=resolved_checkpoint_size,
                        backend="triton",
                    ),
                    eager_weight.detach().clone().requires_grad_(True),
                ),
                (
                    "Generated Triton checkpoint rate output",
                    make_triton_checkpoint_rate_step(
                        surrogate=surrogate,
                        surrogate_slope=args.surrogate_slope,
                        checkpoint_size=resolved_checkpoint_size,
                        backend="triton_generated",
                    ),
                    eager_weight.detach().clone().requires_grad_(True),
                ),
            ]
        )

    results = []
    for label, fn, weight in variants:
        results.append(
            audit_variant(
                label,
                fn,
                inputs,
                weight,
                params,
                warmup=args.warmup,
                repeats=args.repeats,
            )
        )
    return results


def print_markdown(args: argparse.Namespace, results: list[AuditResult]) -> None:
    current_bytes = expected_current_bytes(
        timesteps=args.timesteps,
        batch=args.batch,
        neurons=args.neurons,
        dtype=torch.float32,
    )
    chunk_start_bytes = expected_chunk_start_bytes(
        timesteps=args.timesteps,
        batch=args.batch,
        neurons=args.neurons,
        checkpoint_size=args.checkpoint_size,
        dtype=torch.float32,
    )
    checkpoint_scratch_bytes = expected_checkpoint_scratch_bytes(
        timesteps=args.timesteps,
        batch=args.batch,
        neurons=args.neurons,
        checkpoint_size=args.checkpoint_size,
        dtype=torch.float32,
    )
    print("# Currents Materialization Audit")
    print()
    print("The compiled PyTorch workload returns a scalar loss from inside the compiled")
    print("graph. Dense-output Triton variants return `[T, B, N]` spikes to Python")
    print("before the scalar loss consumes them, so they have a dense-output allocation")
    print("lower bound that the compiled scalar-loss graph does not expose.")
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
    print(f"- `expected_currents_bytes`: `{current_bytes}`")
    print(f"- `expected_currents_mb`: `{format_memory(current_bytes)}`")
    print(f"- `expected_dense_spike_output_bytes`: `{current_bytes}`")
    print(f"- `expected_dense_spike_output_mb`: `{format_memory(current_bytes)}`")
    print(f"- `expected_chunk_start_bytes`: `{chunk_start_bytes}`")
    print(f"- `expected_chunk_start_mb`: `{format_memory(chunk_start_bytes)}`")
    print(f"- `expected_checkpoint_scratch_bytes`: `{checkpoint_scratch_bytes}`")
    print(f"- `expected_checkpoint_scratch_mb`: `{format_memory(checkpoint_scratch_bytes)}`")
    print(f"- `checkpoint_size`: `{args.checkpoint_size}`")
    print(
        f"- `resolved_checkpoint_size`: "
        f"`{resolve_checkpoint_size(args.timesteps, args.checkpoint_size)}`"
    )
    print(f"- `compile_mode`: `{args.compile_mode if not args.no_compile else None}`")
    print()
    print("## Results")
    print()
    print(
        "| Variant | Dense Spike Output | Fwd ms | Bwd ms | Baseline Alloc MB | "
        "Fwd Peak MB | Fwd Increment MB | Fwd Increment / Currents | Bwd Peak MB | "
        "Bwd Increment MB | Fwd Extra Over Dense Output MB | "
        "Bwd Extra Over Dense Output MB | Error |"
    )
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for result in results:
        fwd_increment = result.forward_increment_bytes
        bwd_increment = result.backward_increment_bytes
        ratio = None if fwd_increment is None else fwd_increment / current_bytes
        ratio_text = "" if ratio is None else f"{ratio:.2f}x"
        dense_output = exposes_dense_spike_output(result.label)
        fwd_extra = (
            result.forward_extra_over_dense_output_bytes(current_bytes) if dense_output else None
        )
        bwd_extra = (
            result.backward_extra_over_dense_output_bytes(current_bytes) if dense_output else None
        )
        print(
            f"| {result.label} | "
            f"{'yes' if dense_output else 'no'} | "
            f"{format_ms(result.split_forward_seconds)} | "
            f"{format_ms(result.backward_seconds)} | "
            f"{format_memory(result.allocated_bytes)} | "
            f"{format_memory(result.forward_peak_bytes)} | "
            f"{format_memory(fwd_increment)} | "
            f"{ratio_text} | "
            f"{format_memory(result.backward_peak_bytes)} | "
            f"{format_memory(bwd_increment)} | "
            f"{format_memory(fwd_extra)} | "
            f"{format_memory(bwd_extra)} | "
            f"{result.error or ''} |"
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
    parser.add_argument("--no-compile", action="store_true")
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
    parser.add_argument("--checkpoint-size", type=parse_checkpoint_size, default=25)
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
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=(
                    "CUDA inputs detected and Triton is available, but .* is running "
                    "with backend='torch'.*"
                ),
                category=RuntimeWarning,
            )
            print_markdown(args, run_audit(args))
    finally:
        if args.matmul_precision != "current":
            torch.set_float32_matmul_precision(original_matmul_precision)


if __name__ == "__main__":
    main()
