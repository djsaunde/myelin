"""Canonical compiled-vs-Triton performance frontier report."""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from datetime import UTC, datetime

import torch

from spiker.baselines import compiled_available
from spiker.benchmarks.currents_audit import (
    AuditResult,
    expected_current_bytes,
    exposes_dense_spike_output,
    run_audit,
)
from spiker.benchmarks.lif import (
    format_memory,
    format_speedup,
    gpu_name,
    run_lif_benchmark,
)
from spiker.checkpointing import parse_checkpoint_size, resolve_checkpoint_size


@dataclass(frozen=True)
class FrontierRow:
    contract: str
    variant: str
    dense_spike_output: bool | None
    forward_ms: float | None
    backward_ms: float | None
    forward_backward_ms: float | None
    peak_increment_bytes: int | None
    speedup_vs_compile: float | None
    compile_warmup_ms: float | None
    note: str
    error: str | None = None


def _seconds_to_ms(seconds: float | None) -> float | None:
    return None if seconds is None else seconds * 1000.0


def _frontier_ms(value: float | None) -> str:
    return "" if value is None else f"{value:.3f}"


def _frontier_memory(memory_bytes: int | None) -> str:
    return "" if memory_bytes is None else format_memory(memory_bytes)


def _audit_total(result: AuditResult) -> float | None:
    if result.split_forward_seconds is None or result.backward_seconds is None:
        return None
    return result.split_forward_seconds + result.backward_seconds


def _audit_increment(result: AuditResult) -> int | None:
    return result.backward_increment_bytes


def _speedup_vs_compile(value: float | None, compile_value: float | None) -> float | None:
    if value is None or compile_value is None:
        return None
    return compile_value / value


def _audit_by_label(results: list[AuditResult]) -> dict[str, AuditResult]:
    return {result.label: result for result in results}


def run_benchmark(args: argparse.Namespace) -> list[FrontierRow]:
    compile_enabled = (not args.no_compile) and compiled_available()
    rows: list[FrontierRow] = []

    lif_forward = run_lif_benchmark(
        timesteps=args.timesteps,
        batch=args.batch,
        neurons=args.neurons,
        warmup=args.warmup,
        repeats=args.repeats,
        device=args.device,
        compile_enabled=compile_enabled,
    )
    compiled_forward = lif_forward["compiled_seconds"]
    triton_forward = lif_forward["triton_seconds"]
    rows.append(
        FrontierRow(
            contract="hard LIF forward",
            variant="torch.compile",
            dense_spike_output=True,
            forward_ms=_seconds_to_ms(
                compiled_forward if isinstance(compiled_forward, float) else None
            ),
            backward_ms=None,
            forward_backward_ms=None,
            peak_increment_bytes=None,
            speedup_vs_compile=1.0 if isinstance(compiled_forward, float) else None,
            compile_warmup_ms=_seconds_to_ms(
                lif_forward["compile_warmup_seconds"]
                if isinstance(lif_forward["compile_warmup_seconds"], float)
                else None
            ),
            note="forward-only baseline; compile time excluded",
            error=(
                lif_forward["compile_error"]
                if isinstance(lif_forward["compile_error"], str)
                else None
            ),
        )
    )
    rows.append(
        FrontierRow(
            contract="hard LIF forward",
            variant="Triton fused-time",
            dense_spike_output=True,
            forward_ms=_seconds_to_ms(
                triton_forward if isinstance(triton_forward, float) else None
            ),
            backward_ms=None,
            forward_backward_ms=None,
            peak_increment_bytes=None,
            speedup_vs_compile=_speedup_vs_compile(
                triton_forward if isinstance(triton_forward, float) else None,
                compiled_forward if isinstance(compiled_forward, float) else None,
            ),
            compile_warmup_ms=None,
            note="single fused-time forward kernel",
            error=(
                lif_forward["triton_error"]
                if isinstance(lif_forward["triton_error"], str)
                else None
            ),
        )
    )

    audit_args = argparse.Namespace(
        timesteps=args.timesteps,
        batch=args.batch,
        features=args.features,
        neurons=args.neurons,
        device=args.device,
        warmup=args.warmup,
        repeats=args.repeats,
        no_compile=args.no_compile,
        compile_mode=args.compile_mode,
        surrogate=args.surrogate,
        surrogate_slope=args.surrogate_slope,
        checkpoint_size=args.checkpoint_size,
        matmul_precision="current",
    )
    audit_results = _audit_by_label(run_audit(audit_args))
    compiled_training = audit_results.get("torch.compile materialized graph")
    compiled_training_total = None if compiled_training is None else _audit_total(compiled_training)

    selected_training = (
        (
            "dense-output training",
            "torch.compile materialized graph",
            "whole scalar loss captured by Inductor",
        ),
        (
            "dense-output training",
            "Triton checkpoint recompute",
            "returns dense spikes; checkpointed backward",
        ),
        (
            "rate/scalar training",
            "Triton checkpoint rate output",
            "avoids dense spike output; handwritten backward",
        ),
        (
            "rate/scalar training",
            "Generated Triton checkpoint rate output",
            "avoids dense spike output; generated backward chunk",
        ),
    )
    for contract, label, note in selected_training:
        result = audit_results.get(label)
        if result is None:
            rows.append(
                FrontierRow(
                    contract=contract,
                    variant=label,
                    dense_spike_output=None,
                    forward_ms=None,
                    backward_ms=None,
                    forward_backward_ms=None,
                    peak_increment_bytes=None,
                    speedup_vs_compile=None,
                    compile_warmup_ms=None,
                    note=note,
                    error="variant was not produced",
                )
            )
            continue
        total = _audit_total(result)
        rows.append(
            FrontierRow(
                contract=contract,
                variant=label,
                dense_spike_output=exposes_dense_spike_output(label),
                forward_ms=_seconds_to_ms(result.split_forward_seconds),
                backward_ms=_seconds_to_ms(result.backward_seconds),
                forward_backward_ms=_seconds_to_ms(total),
                peak_increment_bytes=_audit_increment(result),
                speedup_vs_compile=(
                    1.0
                    if label == "torch.compile materialized graph" and total is not None
                    else _speedup_vs_compile(total, compiled_training_total)
                ),
                compile_warmup_ms=None,
                note=note,
                error=result.error,
            )
        )

    return rows


def print_markdown(args: argparse.Namespace, rows: list[FrontierRow]) -> None:
    current_bytes = expected_current_bytes(
        timesteps=args.timesteps,
        batch=args.batch,
        neurons=args.neurons,
        dtype=torch.float32,
    )
    print("# Performance Frontier")
    print()
    print("Canonical compiled-vs-Triton comparison for equal or explicit output contracts.")
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
    print(f"- `dense_spike_or_current_mb`: `{format_memory(current_bytes)}`")
    print(f"- `checkpoint_size`: `{args.checkpoint_size}`")
    print(
        f"- `resolved_checkpoint_size`: "
        f"`{resolve_checkpoint_size(args.timesteps, args.checkpoint_size)}`"
    )
    print(f"- `compile_mode`: `{None if args.no_compile else args.compile_mode}`")
    print(f"- `warmup`: `{args.warmup}`")
    print(f"- `repeats`: `{args.repeats}`")
    print()
    print("## Frontier")
    print()
    print(
        "| Contract | Variant | Dense Spike Output | Fwd ms | Bwd ms | Fwd+Bwd ms | "
        "Bwd Increment MB | Speedup vs Compile | Compile Warmup ms | Note | Error |"
    )
    print("|---|---|---|---:|---:|---:|---:|---:|---:|---|---|")
    for row in rows:
        dense_output = (
            "" if row.dense_spike_output is None else "yes" if row.dense_spike_output else "no"
        )
        print(
            f"| {row.contract} | "
            f"{row.variant} | "
            f"{dense_output} | "
            f"{_frontier_ms(row.forward_ms)} | "
            f"{_frontier_ms(row.backward_ms)} | "
            f"{_frontier_ms(row.forward_backward_ms)} | "
            f"{_frontier_memory(row.peak_increment_bytes)} | "
            f"{format_speedup(row.speedup_vs_compile)} | "
            f"{_frontier_ms(row.compile_warmup_ms)} | "
            f"{row.note} | "
            f"{row.error or ''} |"
        )
    print()
    print("## Interpretation")
    print()
    print("- Forward-only fused-time Triton is the clean launch-overhead win.")
    print("- Dense-output training is a close fight because `torch.compile` sees the whole loss.")
    print(
        "- Triton should win by changing the contract: rate/scalar outputs, "
        "packing, and sparse comms."
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
        choices=("sigmoid", "fast_sigmoid", "atan", "triangular", "superspike", "multi_gaussian"),
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
            print_markdown(args, run_benchmark(args))
    finally:
        if args.matmul_precision != "current":
            torch.set_float32_matmul_precision(original_matmul_precision)


if __name__ == "__main__":
    main()
