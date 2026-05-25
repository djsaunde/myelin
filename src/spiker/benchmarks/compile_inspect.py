"""Inspect ``torch.compile`` graphs and generated Inductor code for SNN workloads."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import torch

from spiker.benchmarks.lif import gpu_name
from spiker.neurons import LIFParams
from spiker.workloads import (
    dense_fast_surrogate_lif_spike_loss,
    looped_fast_surrogate_lif_spike_loss,
)

WorkloadName = Literal["materialized", "looped"]
TrainStep = Callable[[torch.Tensor, torch.Tensor, LIFParams], torch.Tensor]


@dataclass(frozen=True)
class CompileInspection:
    workload: WorkloadName
    debug_dir: Path
    graph_count: int | None
    graph_break_count: int | None
    op_count: int | None
    generated_files: tuple[Path, ...]
    output_code_files: tuple[Path, ...]
    output_code_stats: dict[str, int]
    elapsed_seconds: float


def workload_fn(name: WorkloadName) -> TrainStep:
    if name == "materialized":
        return dense_fast_surrogate_lif_spike_loss
    if name == "looped":
        return looped_fast_surrogate_lif_spike_loss
    raise ValueError(f"unsupported workload: {name}")


def make_tensors(
    *,
    timesteps: int,
    batch: int,
    features: int,
    neurons: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = torch.rand((timesteps, batch, features), device=device)
    weight = ((torch.rand((features, neurons), device=device) - 0.5) * 0.02).requires_grad_()
    return inputs, weight


def run_inspection(args: argparse.Namespace) -> CompileInspection:
    import torch._dynamo as dynamo
    import torch._inductor.config as inductor_config
    import torch.compiler.config as compiler_config

    debug_dir = Path(args.output_dir).resolve()
    if args.clean and debug_dir.exists():
        shutil.rmtree(debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)

    os.environ["TORCH_COMPILE_DEBUG"] = "1"
    compiler_config.force_disable_caches = args.force_disable_caches
    inductor_config.trace.enabled = True
    inductor_config.trace.debug_dir = str(debug_dir)
    dynamo.reset()

    fn = workload_fn(args.workload)
    inputs, weight = make_tensors(
        timesteps=args.timesteps,
        batch=args.batch,
        features=args.features,
        neurons=args.neurons,
        device=args.device,
    )
    params = LIFParams()

    explain = dynamo.explain(fn)(inputs, weight, params)
    compiled = torch.compile(fn, mode=args.compile_mode, fullgraph=True)

    start = time.perf_counter()
    weight.grad = None
    loss = compiled(inputs, weight, params)
    loss.backward()
    if inputs.device.type == "cuda":
        torch.cuda.synchronize(inputs.device)
    elapsed_seconds = time.perf_counter() - start

    generated_files = tuple(sorted(path for path in debug_dir.rglob("*") if path.is_file()))
    output_code_files = tuple(path for path in generated_files if path.name == "output_code.py")
    return CompileInspection(
        workload=args.workload,
        debug_dir=debug_dir,
        graph_count=getattr(explain, "graph_count", None),
        graph_break_count=getattr(explain, "graph_break_count", None),
        op_count=getattr(explain, "op_count", None),
        generated_files=generated_files,
        output_code_files=output_code_files,
        output_code_stats=summarize_output_code(output_code_files),
        elapsed_seconds=elapsed_seconds,
    )


def summarize_output_code(paths: tuple[Path, ...]) -> dict[str, int]:
    text = "\n".join(path.read_text() for path in paths)
    reuse_assignments = re.findall(r"\bbuf\d+\s*=\s*buf\d+\s*;\s*del\s+buf\d+\b", text)
    return {
        "bytes": len(text.encode()),
        "lines": text.count("\n") + (1 if text else 0),
        "cpp_fused_kernels": text.count("cpp_fused_"),
        "triton_jit_kernels": text.count("@triton_heuristics"),
        "extern_kernels_mm": text.count("extern_kernels.mm"),
        "extern_kernels_bmm": text.count("extern_kernels.bmm"),
        "empty_strided_allocs": text.count("empty_strided_cuda"),
        "reinterpret_tensor_calls": text.count("reinterpret_tensor("),
        "explicit_del_calls": len(re.findall(r"\bdel\s+buf\d+\b", text)),
        "reuse_assignments": len(reuse_assignments),
        "buf_mentions": text.count("buf"),
    }


def print_markdown(args: argparse.Namespace, inspection: CompileInspection) -> None:
    print("# Torch Compile Inspection")
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
    print(f"- `workload`: `{inspection.workload}`")
    print(f"- `compile_mode`: `{args.compile_mode}`")
    print(f"- `force_disable_caches`: `{args.force_disable_caches}`")
    print(f"- `debug_dir`: `{inspection.debug_dir}`")
    print()
    print("## Dynamo")
    print()
    print("| Metric | Value |")
    print("|---|---:|")
    print(f"| graph_count | {inspection.graph_count} |")
    print(f"| graph_break_count | {inspection.graph_break_count} |")
    print(f"| op_count | {inspection.op_count} |")
    print(f"| compile_first_run_ms | {inspection.elapsed_seconds * 1000:.3f} |")
    print()
    print("## Inductor Artifacts")
    print()
    print("| Metric | Value |")
    print("|---|---:|")
    print(f"| files | {len(inspection.generated_files)} |")
    print(f"| output_code.py files | {len(inspection.output_code_files)} |")
    for key, value in inspection.output_code_stats.items():
        print(f"| {key} | {value} |")
    print()
    print("## Output Code Files")
    print()
    print("| File |")
    print("|---|")
    for path in inspection.output_code_files:
        print(f"| `{path}` |")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", choices=("materialized", "looped"), default="materialized")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--timesteps", type=int, default=16)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--features", type=int, default=16)
    parser.add_argument("--neurons", type=int, default=32)
    parser.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        default="reduce-overhead",
    )
    parser.add_argument("--output-dir", default="benchmarks/compile_debug")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument(
        "--allow-compile-cache",
        action="store_false",
        dest="force_disable_caches",
        help="Allow existing compiler caches; faster, but may skip fresh debug artifacts.",
    )
    parser.set_defaults(force_disable_caches=True)
    args = parser.parse_args()
    print_markdown(args, run_inspection(args))


if __name__ == "__main__":
    main()
