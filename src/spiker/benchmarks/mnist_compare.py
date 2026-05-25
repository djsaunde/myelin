"""Run comparable MNIST example trainings and summarize final metrics."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from spiker.checkpointing import parse_checkpoint_size

METRIC_RE = re.compile(r"^(?P<name>[a-z_]+)=(?P<value>[-+0-9.eE]+)$")


@dataclass(frozen=True)
class Variant:
    name: str
    script: str
    extra_args: tuple[str, ...] = ()
    is_snntorch: bool = False
    uses_rate_readout: bool = False
    supports_compile: bool = True


@dataclass(frozen=True)
class Result:
    name: str
    command: str
    compiled: bool
    returncode: int
    final_loss: float | None
    final_accuracy: float | None
    total_seconds: float | None
    peak_cuda_memory_mb: float | None
    average_step_ms: float | None
    steady_state_step_ms: float | None
    error_tail: str


VARIANTS = {
    "dense": Variant("dense", "train_mnist.py"),
    "rate": Variant("rate", "train_mnist_rate.py", uses_rate_readout=True),
    "rate_generated": Variant(
        "rate_generated",
        "train_mnist_rate.py",
        extra_args=("--backend", "triton_generated"),
        uses_rate_readout=True,
    ),
    "rate_triton_compile": Variant(
        "rate_triton_compile",
        "train_mnist_rate.py",
        extra_args=("--backend", "triton_compile"),
        uses_rate_readout=True,
    ),
    "conv": Variant("conv", "train_mnist_conv.py"),
    "snntorch_dense": Variant(
        "snntorch_dense",
        "train_mnist_snntorch.py",
        extra_args=("--model", "dense"),
        is_snntorch=True,
    ),
    "snntorch_conv": Variant(
        "snntorch_conv",
        "train_mnist_snntorch.py",
        extra_args=("--model", "conv"),
        is_snntorch=True,
    ),
}

COMPILE_POLICY_SCRIPTS = frozenset({"train_mnist.py", "train_mnist_rate.py", "train_mnist_conv.py"})


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def parse_metrics(stdout: str) -> dict[str, float]:
    metrics = {}
    for line in stdout.splitlines():
        match = METRIC_RE.match(line.strip())
        if match is not None:
            metrics[match.group("name")] = float(match.group("value"))
    return metrics


def error_tail(stderr: str, stdout: str) -> str:
    lines = (stderr or stdout).splitlines()
    return " ".join(lines[-3:])


def run_variant(args: argparse.Namespace, variant: Variant) -> Result:
    root = repo_root()
    command = [
        sys.executable,
        str(root / "examples" / variant.script),
        "--data-dir",
        args.data_dir,
        "--device",
        args.device,
        "--timesteps",
        str(args.timesteps),
        "--encoding",
        args.encoding,
        "--batch",
        str(args.batch),
        "--hidden",
        str(args.hidden),
        "--epochs",
        str(args.epochs),
        "--lr",
        str(args.lr),
        "--surrogate-slope",
        str(args.surrogate_slope),
        "--matmul-precision",
        args.matmul_precision,
        "--log-every",
        str(args.log_every),
        "--eval-every",
        str(args.eval_every),
        "--eval-batches",
        str(args.eval_batches),
        "--train-limit",
        str(args.train_limit),
        "--test-limit",
        str(args.test_limit),
        *variant.extra_args,
    ]
    if args.grad_clip is not None:
        command.extend(["--grad-clip", str(args.grad_clip)])
    if variant.uses_rate_readout:
        command.extend(["--checkpoint-size", str(args.rate_checkpoint_size)])
    if variant.is_snntorch:
        command.extend(["--beta", str(args.snntorch_beta)])
    if variant.script == "train_mnist_conv.py" and args.conv_synapse_init is not None:
        command.extend(["--synapse-init", args.conv_synapse_init])
    compile_variant = (
        args.compile or (args.compile_spiker_only and not variant.is_snntorch)
    ) and variant.supports_compile
    if compile_variant:
        if variant.script in COMPILE_POLICY_SCRIPTS:
            command.extend(["--compile", "on"])
        else:
            command.append("--compile")
    elif variant.script in COMPILE_POLICY_SCRIPTS:
        command.extend(["--compile", "off"])
    if args.smooth_forward:
        command.append("--smooth-forward")

    completed = subprocess.run(
        command,
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    metrics = parse_metrics(completed.stdout)
    return Result(
        name=variant.name,
        command=" ".join(command),
        compiled=compile_variant,
        returncode=completed.returncode,
        final_loss=metrics.get("final_test_loss"),
        final_accuracy=metrics.get("final_test_accuracy"),
        total_seconds=metrics.get("total_training_seconds"),
        peak_cuda_memory_mb=metrics.get("peak_cuda_memory_mb"),
        average_step_ms=metrics.get("average_step_ms"),
        steady_state_step_ms=metrics.get("steady_state_average_step_ms"),
        error_tail=error_tail(completed.stderr, completed.stdout),
    )


def format_optional(value: float | None, precision: int = 4) -> str:
    if value is None:
        return ""
    return f"{value:.{precision}f}"


def print_markdown(args: argparse.Namespace, results: list[Result]) -> None:
    print("# MNIST Example Comparison")
    print()
    print("Runs the repo MNIST examples with matched training knobs.")
    print()
    print("## Environment")
    print()
    print(f"- `generated_utc`: `{datetime.now(UTC).isoformat(timespec='seconds')}`")
    print(f"- `device`: `{args.device}`")
    print(f"- `timesteps`: `{args.timesteps}`")
    print(f"- `encoding`: `{args.encoding}`")
    print(f"- `batch`: `{args.batch}`")
    print(f"- `hidden`: `{args.hidden}`")
    print(f"- `epochs`: `{args.epochs}`")
    print(f"- `grad_clip`: `{args.grad_clip}`")
    print(f"- `rate_checkpoint_size`: `{args.rate_checkpoint_size}`")
    print(f"- `train_limit`: `{args.train_limit}`")
    print(f"- `test_limit`: `{args.test_limit}`")
    print(f"- `compile`: `{args.compile}`")
    print(f"- `compile_spiker_only`: `{args.compile_spiker_only}`")
    print(f"- `conv_synapse_init`: `{args.conv_synapse_init}`")
    print(f"- `matmul_precision`: `{args.matmul_precision}`")
    print(f"- `snntorch_beta`: `{args.snntorch_beta}`")
    print()
    print("## Results")
    print()
    print(
        "| Variant | Final Test Loss | Final Test Acc | Total s | Peak CUDA MB | "
        "Avg Step ms | Steady Step ms | Compiled | Status |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for result in results:
        status = result_status(result)
        print(
            f"| {result.name} | "
            f"{format_optional(result.final_loss, 6)} | "
            f"{format_optional(result.final_accuracy, 4)} | "
            f"{format_optional(result.total_seconds, 3)} | "
            f"{format_optional(result.peak_cuda_memory_mb, 1)} | "
            f"{format_optional(result.average_step_ms, 3)} | "
            f"{format_optional(result.steady_state_step_ms, 3)} | "
            f"{result.compiled} | "
            f"{status} |"
        )
    if any(not result.compiled for result in results):
        print()
        print("## Compile Notes")
        print()
        print(
            "The `Compiled` column is per variant. Variants that are not currently "
            "`torch.compile` compatible still run eager even when `--compile` or "
            "`--compile-spiker-only` is requested."
        )

    failures = [result for result in results if result_status(result) != "ok"]
    if failures:
        print()
        print("## Failures")
        print()
        print("| Variant | Tail |")
        print("|---|---|")
        for result in failures:
            print(f"| {result.name} | {result.error_tail} |")


def parse_variants(names: list[str]) -> list[Variant]:
    variants = []
    for name in names:
        if name not in VARIANTS:
            choices = ", ".join(sorted(VARIANTS))
            raise ValueError(f"unsupported variant {name!r}; choices: {choices}")
        variants.append(VARIANTS[name])
    return variants


def result_status(result: Result) -> str:
    if result.returncode != 0:
        return f"failed:{result.returncode}"
    if result.final_loss is None or result.final_accuracy is None:
        return "missing_metrics"
    return "ok"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", action="append", default=[])
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--timesteps", type=int, default=10)
    parser.add_argument("--encoding", choices=("repeat", "poisson", "latency"), default="poisson")
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--grad-clip", type=float)
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
    parser.add_argument("--compile-spiker-only", action="store_true")
    parser.add_argument("--conv-synapse-init", choices=("spiker", "fan_in"))
    parser.add_argument("--log-every", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--train-limit", type=int, default=1024)
    parser.add_argument("--test-limit", type=int, default=1024)
    args = parser.parse_args()

    variants = parse_variants(args.variant or ["dense", "rate", "conv"])
    results = [run_variant(args, variant) for variant in variants]
    print_markdown(args, results)


if __name__ == "__main__":
    main()
