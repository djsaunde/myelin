"""Run reproducible benchmark suites and write markdown artifacts."""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import torch

from spiker._optional import has_triton

Preset = str


@dataclass(frozen=True)
class BenchmarkSpec:
    """One benchmark command and its artifact stem."""

    name: str
    module: str
    args: tuple[str, ...]
    artifact_stem: str
    device_override: str | None = None
    inject_device: bool = True


@dataclass(frozen=True)
class BenchmarkRun:
    """Resolved benchmark command and output path."""

    spec: BenchmarkSpec
    command: tuple[str, ...]
    output_path: Path


SMOKE_SPECS: tuple[BenchmarkSpec, ...] = (
    BenchmarkSpec(
        name="headline",
        module="spiker.benchmarks.headline",
        args=(),
        artifact_stem="headline",
        inject_device=False,
    ),
    BenchmarkSpec(
        name="generated_forward",
        module="spiker.benchmarks.generated_forward",
        args=(
            "--timesteps",
            "16",
            "--batch",
            "8",
            "--neurons",
            "64",
            "--warmup",
            "1",
            "--repeats",
            "1",
        ),
        artifact_stem="generated_forward_smoke",
    ),
    BenchmarkSpec(
        name="packed_forward",
        module="spiker.benchmarks.packed_forward",
        args=("--shape", "16,8,64", "--warmup", "1", "--repeats", "1"),
        artifact_stem="packed_forward_smoke",
    ),
    BenchmarkSpec(
        name="rate_readout",
        module="spiker.benchmarks.rate_readout",
        args=(
            "--shape",
            "16,8,10",
            "--features",
            "16",
            "--checkpoint-size",
            "4",
            "--matmul-precision",
            "high",
            "--warmup",
            "1",
            "--repeats",
            "1",
        ),
        artifact_stem="rate_readout_smoke",
    ),
    BenchmarkSpec(
        name="scalar_loss_boundary",
        module="spiker.benchmarks.scalar_loss_boundary",
        args=(
            "--shape",
            "16,8,64",
            "--features",
            "16",
            "--checkpoint-size",
            "4",
            "--warmup",
            "1",
            "--repeats",
            "1",
        ),
        artifact_stem="scalar_loss_boundary_smoke",
    ),
    BenchmarkSpec(
        name="performance_frontier",
        module="spiker.benchmarks.performance_frontier",
        args=(
            "--timesteps",
            "16",
            "--batch",
            "8",
            "--features",
            "16",
            "--neurons",
            "64",
            "--checkpoint-size",
            "4",
            "--warmup",
            "1",
            "--repeats",
            "1",
        ),
        artifact_stem="performance_frontier_smoke",
    ),
    BenchmarkSpec(
        name="training_breakdown",
        module="spiker.benchmarks.training_breakdown",
        args=(
            "--timesteps",
            "16",
            "--batch",
            "8",
            "--features",
            "16",
            "--neurons",
            "64",
            "--checkpoint-size",
            "4",
            "--warmup",
            "1",
            "--repeats",
            "1",
            "--no-compile",
        ),
        artifact_stem="training_breakdown_smoke",
    ),
    BenchmarkSpec(
        name="compile_triton_boundary",
        module="spiker.benchmarks.compile_triton_boundary",
        args=(
            "--timesteps",
            "16",
            "--batch",
            "8",
            "--features",
            "16",
            "--neurons",
            "64",
            "--checkpoint-size",
            "4",
            "--warmup",
            "1",
            "--repeats",
            "1",
        ),
        artifact_stem="compile_triton_boundary_smoke",
    ),
    BenchmarkSpec(
        name="compile_triton_sweep",
        module="spiker.benchmarks.compile_triton_sweep",
        args=(
            "--shape",
            "16,8,16,64",
            "--checkpoint-size",
            "4",
            "--warmup",
            "1",
            "--repeats",
            "1",
        ),
        artifact_stem="compile_triton_sweep_smoke",
    ),
    BenchmarkSpec(
        name="custom_neuron_module",
        module="spiker.benchmarks.custom_neuron_module",
        args=(
            "--variant",
            "all",
            "--timesteps",
            "16",
            "--batch",
            "8",
            "--neurons",
            "64",
            "--warmup",
            "1",
            "--repeats",
            "1",
        ),
        artifact_stem="custom_neuron_module_smoke",
    ),
    BenchmarkSpec(
        name="custom_surrogate_training",
        module="spiker.benchmarks.custom_surrogate_training",
        args=(
            "--timesteps",
            "16",
            "--batch",
            "8",
            "--neurons",
            "64",
            "--warmup",
            "1",
            "--repeats",
            "1",
        ),
        artifact_stem="custom_surrogate_training_smoke",
    ),
    BenchmarkSpec(
        name="online_learning",
        module="spiker.benchmarks.online_learning",
        args=(
            "--timesteps",
            "4",
            "--batch",
            "2",
            "--features",
            "3",
            "--neurons",
            "5",
            "--warmup",
            "1",
            "--repeats",
            "1",
            "--seed",
            "0",
        ),
        artifact_stem="online_learning_smoke",
    ),
    BenchmarkSpec(
        name="hardware_export",
        module="spiker.benchmarks.hardware_export",
        args=(
            "--in-features",
            "3",
            "--out-features",
            "5",
            "--max-inputs-per-core",
            "2",
            "--max-outputs-per-core",
            "3",
        ),
        artifact_stem="hardware_export_smoke",
        inject_device=False,
    ),
    BenchmarkSpec(
        name="distributed_collectives_gloo",
        module="spiker.benchmarks.distributed_collectives",
        args=(
            "--backend",
            "gloo",
            "--world-size",
            "2",
            "--timesteps",
            "4",
            "--batch",
            "2",
            "--neurons",
            "35",
            "--warmup",
            "0",
            "--repeats",
            "1",
            "--timeout",
            "30",
        ),
        artifact_stem="distributed_collectives_gloo_smoke",
        device_override="cpu",
    ),
    BenchmarkSpec(
        name="distributed_collectives_nccl",
        module="spiker.benchmarks.distributed_collectives",
        args=(
            "--backend",
            "nccl",
            "--world-size",
            "1",
            "--timesteps",
            "16",
            "--batch",
            "8",
            "--neurons",
            "64",
            "--warmup",
            "1",
            "--repeats",
            "2",
            "--timeout",
            "60",
        ),
        artifact_stem="distributed_collectives_nccl_smoke",
    ),
)

CORE_SPECS: tuple[BenchmarkSpec, ...] = (
    BenchmarkSpec(
        name="headline",
        module="spiker.benchmarks.headline",
        args=(),
        artifact_stem="headline",
        inject_device=False,
    ),
    BenchmarkSpec(
        name="generated_forward",
        module="spiker.benchmarks.generated_forward",
        args=(
            "--timesteps",
            "100",
            "--batch",
            "64",
            "--neurons",
            "2048",
            "--warmup",
            "5",
            "--repeats",
            "20",
        ),
        artifact_stem="generated_forward",
    ),
    BenchmarkSpec(
        name="packed_forward",
        module="spiker.benchmarks.packed_forward",
        args=("--warmup", "5", "--repeats", "20", "--shape", "100,64,2048"),
        artifact_stem="packed_forward",
    ),
    BenchmarkSpec(
        name="rate_readout",
        module="spiker.benchmarks.rate_readout",
        args=(
            "--shape",
            "100,128,10",
            "--shape",
            "100,128,1000",
            "--shape",
            "100,128,2048",
            "--features",
            "128",
            "--checkpoint-size",
            "25",
            "--matmul-precision",
            "high",
            "--warmup",
            "3",
            "--repeats",
            "10",
        ),
        artifact_stem="rate_readout",
    ),
    BenchmarkSpec(
        name="scalar_loss_boundary",
        module="spiker.benchmarks.scalar_loss_boundary",
        args=(
            "--shape",
            "100,64,2048",
            "--features",
            "128",
            "--checkpoint-size",
            "25",
            "--warmup",
            "3",
            "--repeats",
            "10",
        ),
        artifact_stem="scalar_loss_boundary",
    ),
    BenchmarkSpec(
        name="performance_frontier",
        module="spiker.benchmarks.performance_frontier",
        args=(
            "--timesteps",
            "100",
            "--batch",
            "64",
            "--features",
            "128",
            "--neurons",
            "2048",
            "--checkpoint-size",
            "25",
            "--matmul-precision",
            "high",
            "--warmup",
            "3",
            "--repeats",
            "10",
        ),
        artifact_stem="performance_frontier",
    ),
    BenchmarkSpec(
        name="training_breakdown",
        module="spiker.benchmarks.training_breakdown",
        args=(
            "--timesteps",
            "100",
            "--batch",
            "64",
            "--features",
            "128",
            "--neurons",
            "2048",
            "--checkpoint-size",
            "25",
            "--warmup",
            "3",
            "--repeats",
            "10",
        ),
        artifact_stem="training_breakdown",
    ),
    BenchmarkSpec(
        name="compile_triton_boundary",
        module="spiker.benchmarks.compile_triton_boundary",
        args=(
            "--timesteps",
            "100",
            "--batch",
            "64",
            "--features",
            "128",
            "--neurons",
            "2048",
            "--checkpoint-size",
            "25",
            "--warmup",
            "3",
            "--repeats",
            "10",
        ),
        artifact_stem="compile_triton_boundary",
    ),
    BenchmarkSpec(
        name="compile_triton_sweep",
        module="spiker.benchmarks.compile_triton_sweep",
        args=(
            "--checkpoint-size",
            "25",
            "--warmup",
            "3",
            "--repeats",
            "10",
        ),
        artifact_stem="compile_triton_sweep",
    ),
    BenchmarkSpec(
        name="mnist_rate_matrix",
        module="spiker.benchmarks.mnist_rate_matrix",
        args=(
            "--epochs",
            "2",
            "--train-limit",
            "4096",
            "--test-limit",
            "2048",
            "--eval-batches",
            "4",
            "--log-every",
            "1000",
            "--eval-every",
            "1000",
            "--matmul-precision",
            "highest",
            "--grad-clip",
            "0.1",
        ),
        artifact_stem="mnist_rate_matrix",
    ),
    BenchmarkSpec(
        name="custom_neuron_module",
        module="spiker.benchmarks.custom_neuron_module",
        args=(
            "--variant",
            "all",
            "--timesteps",
            "100",
            "--batch",
            "64",
            "--neurons",
            "2048",
            "--warmup",
            "5",
            "--repeats",
            "20",
        ),
        artifact_stem="custom_neuron_module",
    ),
    BenchmarkSpec(
        name="surrogate_backend",
        module="spiker.benchmarks.surrogate_backend",
        args=(
            "--timesteps",
            "100",
            "--batch",
            "64",
            "--features",
            "128",
            "--neurons",
            "2048",
            "--checkpoint-size",
            "25",
            "--warmup",
            "3",
            "--repeats",
            "10",
        ),
        artifact_stem="surrogate_backend",
    ),
    BenchmarkSpec(
        name="custom_surrogate_training",
        module="spiker.benchmarks.custom_surrogate_training",
        args=(
            "--timesteps",
            "100",
            "--batch",
            "64",
            "--neurons",
            "2048",
            "--warmup",
            "5",
            "--repeats",
            "20",
        ),
        artifact_stem="custom_surrogate_training",
    ),
    BenchmarkSpec(
        name="online_learning",
        module="spiker.benchmarks.online_learning",
        args=(
            "--timesteps",
            "100",
            "--batch",
            "64",
            "--features",
            "128",
            "--neurons",
            "512",
            "--warmup",
            "3",
            "--repeats",
            "10",
            "--seed",
            "0",
        ),
        artifact_stem="online_learning",
    ),
    BenchmarkSpec(
        name="hardware_export",
        module="spiker.benchmarks.hardware_export",
        args=(
            "--in-features",
            "3",
            "--out-features",
            "5",
            "--max-inputs-per-core",
            "2",
            "--max-outputs-per-core",
            "3",
        ),
        artifact_stem="hardware_export",
        inject_device=False,
    ),
    BenchmarkSpec(
        name="distributed_collectives_gloo",
        module="spiker.benchmarks.distributed_collectives",
        args=(
            "--backend",
            "gloo",
            "--world-size",
            "2",
            "--timesteps",
            "100",
            "--batch",
            "64",
            "--neurons",
            "2048",
            "--warmup",
            "3",
            "--repeats",
            "10",
            "--timeout",
            "90",
        ),
        artifact_stem="distributed_collectives_gloo",
        device_override="cpu",
    ),
    BenchmarkSpec(
        name="distributed_collectives_nccl",
        module="spiker.benchmarks.distributed_collectives",
        args=(
            "--backend",
            "nccl",
            "--world-size",
            "1",
            "--timesteps",
            "100",
            "--batch",
            "64",
            "--neurons",
            "2048",
            "--warmup",
            "3",
            "--repeats",
            "10",
            "--timeout",
            "60",
        ),
        artifact_stem="distributed_collectives_nccl",
    ),
)

PRESETS: dict[Preset, tuple[BenchmarkSpec, ...]] = {
    "smoke": SMOKE_SPECS,
    "core": CORE_SPECS,
}


def make_runs(
    specs: Sequence[BenchmarkSpec],
    *,
    device: str,
    output_dir: Path,
    suffix: str,
) -> tuple[BenchmarkRun, ...]:
    """Resolve benchmark specs into subprocess commands and artifact paths."""

    runs: list[BenchmarkRun] = []
    for spec in specs:
        command_device = spec.device_override or device
        device_args = ("--device", command_device) if spec.inject_device else ()
        command = (sys.executable, "-m", spec.module, *device_args, *spec.args)
        output_path = output_dir / f"{spec.artifact_stem}_{suffix}.md"
        runs.append(BenchmarkRun(spec=spec, command=command, output_path=output_path))
    return tuple(runs)


def run_benchmark(run: BenchmarkRun) -> None:
    """Run one benchmark command and write stdout to its output artifact."""

    run.output_path.parent.mkdir(parents=True, exist_ok=True)
    started = datetime.now(UTC).isoformat(timespec="seconds")
    completed = subprocess.run(
        run.command,
        check=False,
        capture_output=True,
        text=True,
    )
    finished = datetime.now(UTC).isoformat(timespec="seconds")
    header = "\n".join(
        [
            f"<!-- benchmark_runner_name: {run.spec.name} -->",
            f"<!-- benchmark_runner_started: {started} -->",
            f"<!-- benchmark_runner_finished: {finished} -->",
            f"<!-- benchmark_runner_command: {' '.join(run.command)} -->",
            "",
        ]
    )
    body = completed.stdout
    if completed.returncode != 0:
        body += "\n\n## Runner Error\n\n"
        body += f"- `returncode`: `{completed.returncode}`\n"
        if completed.stderr:
            body += "\n```text\n" + completed.stderr + "\n```\n"
    run.output_path.write_text(header + body)
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(
            completed.returncode,
            run.command,
            output=completed.stdout,
            stderr=completed.stderr,
        )


def print_run_table(runs: Sequence[BenchmarkRun]) -> None:
    """Print a markdown table of planned benchmark commands."""

    print("| Benchmark | Artifact | Command |")
    print("|---|---|---|")
    for run in runs:
        print(f"| {run.spec.name} | `{run.output_path}` | `{' '.join(run.command)}` |")


def _validate_cuda_requirement(device: str, *, require_cuda: bool) -> None:
    if not require_cuda:
        return
    if torch.device(device).type != "cuda":
        raise RuntimeError("--require-cuda requires a CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if not has_triton():
        raise RuntimeError("Triton is not installed; run `uv sync --extra cuda`")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=sorted(PRESETS), default="smoke")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/results"))
    parser.add_argument("--suffix", default="local")
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()

    specs = PRESETS[args.preset]
    if args.only:
        requested = set(args.only)
        specs = tuple(spec for spec in specs if spec.name in requested)
        missing = requested - {spec.name for spec in specs}
        if missing:
            choices = ", ".join(spec.name for spec in PRESETS[args.preset])
            raise ValueError(f"unknown benchmark(s): {sorted(missing)}; choices: {choices}")

    _validate_cuda_requirement(args.device, require_cuda=args.require_cuda)
    runs = make_runs(
        specs,
        device=args.device,
        output_dir=args.output_dir,
        suffix=args.suffix,
    )
    print_run_table(runs)
    if args.dry_run:
        return

    for run in runs:
        print(f"\n## Running {run.spec.name}", flush=True)
        run_benchmark(run)
        print(f"wrote {run.output_path}", flush=True)


if __name__ == "__main__":
    main()
