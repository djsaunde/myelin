"""Compare compiled myelin examples against eager snnTorch across timesteps."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime

from myelin.benchmarks.mnist_compare import Result, parse_variants, run_variant


@dataclass(frozen=True)
class TimestepSetting:
    timesteps: int


DEFAULT_TIMESTEPS: tuple[TimestepSetting, ...] = (
    TimestepSetting(10),
    TimestepSetting(25),
    TimestepSetting(50),
)


@dataclass(frozen=True)
class MatrixRow:
    setting: TimestepSetting
    result: Result


def result_status(result: Result) -> str:
    if result.returncode != 0:
        return f"failed:{result.returncode}"
    if result.final_loss is None or result.final_accuracy is None:
        return "missing_metrics"
    return "ok"


def format_optional(value: float | None, precision: int = 4) -> str:
    if value is None:
        return ""
    return f"{value:.{precision}f}"


def speedup(numerator: float | None, denominator: float | None) -> str:
    if numerator is None or denominator is None or numerator <= 0:
        return ""
    return f"{numerator / denominator:.2f}x"


def run_matrix(args: argparse.Namespace) -> list[MatrixRow]:
    variants = parse_variants(args.variant or ["rate", "conv", "snntorch_dense", "snntorch_conv"])
    settings = args.timesteps or list(DEFAULT_TIMESTEPS)
    rows: list[MatrixRow] = []
    for setting in settings:
        setting_args = argparse.Namespace(**vars(args))
        setting_args.timesteps = setting.timesteps
        for variant in variants:
            rows.append(MatrixRow(setting=setting, result=run_variant(setting_args, variant)))
    return rows


def print_markdown(args: argparse.Namespace, rows: list[MatrixRow]) -> None:
    print("# myelin vs snnTorch Matrix")
    print()
    print("Compares compiled myelin examples against eager snnTorch examples across timesteps.")
    print()
    print("## Environment")
    print()
    print(f"- `generated_utc`: `{datetime.now(UTC).isoformat(timespec='seconds')}`")
    print(f"- `device`: `{args.device}`")
    print(f"- `encoding`: `{args.encoding}`")
    print(f"- `batch`: `{args.batch}`")
    print(f"- `hidden`: `{args.hidden}`")
    print(f"- `epochs`: `{args.epochs}`")
    print(f"- `train_limit`: `{args.train_limit}`")
    print(f"- `test_limit`: `{args.test_limit}`")
    print(f"- `compile_myelin_only`: `{args.compile_myelin_only}`")
    print(f"- `matmul_precision`: `{args.matmul_precision}`")
    print()
    print("## Results")
    print()
    print(
        "| T | Variant | Final Test Loss | Final Test Acc | Total s | Peak CUDA MB | "
        "Avg Step ms | Steady Step ms | Compiled | Status |"
    )
    print("|---:|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for row in rows:
        result = row.result
        print(
            f"| {row.setting.timesteps} | "
            f"{result.name} | "
            f"{format_optional(result.final_loss, 6)} | "
            f"{format_optional(result.final_accuracy, 4)} | "
            f"{format_optional(result.total_seconds, 3)} | "
            f"{format_optional(result.peak_cuda_memory_mb, 1)} | "
            f"{format_optional(result.average_step_ms, 3)} | "
            f"{format_optional(result.steady_state_step_ms, 3)} | "
            f"{result.compiled} | "
            f"{result_status(result)} |"
        )

    print()
    print("## Derived Speedups")
    print()
    print("| T | Comparison | Steady Step Speedup | Peak Memory Ratio |")
    print("|---:|---|---:|---:|")
    by_timestep: dict[int, dict[str, Result]] = {}
    for row in rows:
        by_timestep.setdefault(row.setting.timesteps, {})[row.result.name] = row.result
    for timesteps, results in by_timestep.items():
        comparisons = (
            ("rate vs snntorch_dense", results.get("rate"), results.get("snntorch_dense")),
            ("conv vs snntorch_conv", results.get("conv"), results.get("snntorch_conv")),
        )
        for label, myelin_result, snntorch_result in comparisons:
            if myelin_result is None or snntorch_result is None:
                continue
            step_speedup = speedup(
                snntorch_result.steady_state_step_ms,
                myelin_result.steady_state_step_ms,
            )
            memory_ratio = speedup(
                snntorch_result.peak_cuda_memory_mb,
                myelin_result.peak_cuda_memory_mb,
            )
            print(f"| {timesteps} | {label} | {step_speedup} | {memory_ratio} |")

    failures = [row for row in rows if result_status(row.result) != "ok"]
    if failures:
        print()
        print("## Failures")
        print()
        print("| T | Variant | Tail |")
        print("|---:|---|---|")
        for row in failures:
            print(f"| {row.setting.timesteps} | {row.result.name} | {row.result.error_tail} |")


def parse_timesteps(value: str) -> TimestepSetting:
    try:
        timesteps = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("timesteps must be an integer") from exc
    if timesteps <= 0:
        raise argparse.ArgumentTypeError("timesteps must be positive")
    return TimestepSetting(timesteps)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", action="append", default=[])
    parser.add_argument("--timesteps", type=parse_timesteps, action="append", default=[])
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--encoding", choices=("repeat", "poisson", "latency"), default="poisson")
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--grad-clip", type=float, default=0.1)
    parser.add_argument("--rate-checkpoint-size", default="balanced")
    parser.add_argument("--surrogate-slope", type=float, default=5.0)
    parser.add_argument("--snntorch-beta", type=float, default=0.95)
    parser.add_argument(
        "--matmul-precision",
        choices=("highest", "high", "medium"),
        default="highest",
    )
    parser.add_argument("--smooth-forward", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--compile-myelin-only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--conv-synapse-init", choices=("myelin", "fan_in"), default="fan_in")
    parser.add_argument("--log-every", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--train-limit", type=int, default=1024)
    parser.add_argument("--test-limit", type=int, default=1024)
    args = parser.parse_args()

    rows = run_matrix(args)
    print_markdown(args, rows)


if __name__ == "__main__":
    main()
