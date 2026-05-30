"""Run a small quality/performance matrix for MNIST rate-readout variants."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime

from myelin.benchmarks.mnist_compare import Result, parse_variants, run_variant
from myelin.checkpointing import parse_checkpoint_size


@dataclass(frozen=True)
class MatrixSetting:
    name: str
    timesteps: int
    hidden: int


DEFAULT_SETTINGS: tuple[MatrixSetting, ...] = (
    MatrixSetting("t10_h128", timesteps=10, hidden=128),
    MatrixSetting("t25_h128", timesteps=25, hidden=128),
    MatrixSetting("t10_h256", timesteps=10, hidden=256),
)


@dataclass(frozen=True)
class MatrixRow:
    setting: MatrixSetting
    result: Result


def parse_setting(value: str) -> MatrixSetting:
    parts = value.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("setting must be NAME,TIMESTEPS,HIDDEN")
    name, timesteps_text, hidden_text = parts
    try:
        timesteps = int(timesteps_text)
        hidden = int(hidden_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("timesteps and hidden must be integers") from exc
    if timesteps <= 0 or hidden <= 0:
        raise argparse.ArgumentTypeError("timesteps and hidden must be positive")
    return MatrixSetting(name=name, timesteps=timesteps, hidden=hidden)


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


def run_matrix(args: argparse.Namespace) -> list[MatrixRow]:
    variants = parse_variants(args.variant or ["rate", "rate_triton_compile"])
    rows: list[MatrixRow] = []
    for setting in args.setting:
        setting_args = argparse.Namespace(**vars(args))
        setting_args.timesteps = setting.timesteps
        setting_args.hidden = setting.hidden
        for variant in variants:
            rows.append(MatrixRow(setting=setting, result=run_variant(setting_args, variant)))
    return rows


def print_markdown(args: argparse.Namespace, rows: list[MatrixRow]) -> None:
    print("# MNIST Rate-Readout Matrix")
    print()
    print("Compares rate-readout variants across matched MNIST training settings.")
    print()
    print("## Environment")
    print()
    print(f"- `generated_utc`: `{datetime.now(UTC).isoformat(timespec='seconds')}`")
    print(f"- `device`: `{args.device}`")
    print(f"- `encoding`: `{args.encoding}`")
    print(f"- `batch`: `{args.batch}`")
    print(f"- `epochs`: `{args.epochs}`")
    print(f"- `grad_clip`: `{args.grad_clip}`")
    print(f"- `rate_checkpoint_size`: `{args.rate_checkpoint_size}`")
    print(f"- `train_limit`: `{args.train_limit}`")
    print(f"- `test_limit`: `{args.test_limit}`")
    print(f"- `compile_myelin_only`: `{args.compile_myelin_only}`")
    print(f"- `matmul_precision`: `{args.matmul_precision}`")
    print()
    print("## Results")
    print()
    print(
        "| Setting | T | Hidden | Variant | Final Test Loss | Final Test Acc | Total s | "
        "Peak CUDA MB | Avg Step ms | Steady Step ms | Compiled | Status |"
    )
    print("|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for row in rows:
        result = row.result
        print(
            f"| {row.setting.name} | "
            f"{row.setting.timesteps} | "
            f"{row.setting.hidden} | "
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

    failures = [row for row in rows if result_status(row.result) != "ok"]
    if failures:
        print()
        print("## Failures")
        print()
        print("| Setting | Variant | Tail |")
        print("|---|---|---|")
        for row in failures:
            print(f"| {row.setting.name} | {row.result.name} | {row.result.error_tail} |")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", action="append", default=[])
    parser.add_argument("--setting", type=parse_setting, action="append", default=[])
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--encoding", choices=("repeat", "poisson", "latency"), default="poisson")
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--grad-clip", type=float, default=0.1)
    parser.add_argument("--rate-checkpoint-size", type=parse_checkpoint_size, default="balanced")
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
    parser.add_argument("--conv-synapse-init", choices=("myelin", "fan_in"))
    parser.add_argument("--log-every", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--train-limit", type=int, default=4096)
    parser.add_argument("--test-limit", type=int, default=2048)
    args = parser.parse_args()
    if not args.setting:
        args.setting = list(DEFAULT_SETTINGS)

    rows = run_matrix(args)
    print_markdown(args, rows)


if __name__ == "__main__":
    main()
