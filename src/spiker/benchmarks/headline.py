"""Summarize canonical benchmark artifacts into one headline report."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

SCALAR_RE = re.compile(r"^(?P<name>[a-z_]+)=(?P<value>[-+0-9.eE]+)$")


@dataclass(frozen=True)
class HeadlineRow:
    area: str
    headline: str
    artifact: str
    status: str = "ok"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def parse_scalar_metrics(text: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for line in text.splitlines():
        match = SCALAR_RE.match(line.strip())
        if match is not None:
            metrics[match.group("name")] = float(match.group("value"))
    return metrics


def parse_markdown_table(text: str, header_prefix: str) -> list[dict[str, str]]:
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if not line.startswith(header_prefix):
            continue
        if index + 1 >= len(lines):
            return []
        headers = split_table_row(line)
        rows: list[dict[str, str]] = []
        for row_line in lines[index + 2 :]:
            if not row_line.startswith("|"):
                break
            cells = split_table_row(row_line)
            if len(cells) != len(headers):
                break
            rows.append(dict(zip(headers, cells, strict=True)))
        return rows
    return []


def split_table_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def read_artifact(results_dir: Path, artifact: str) -> str:
    return (results_dir / artifact).read_text()


def format_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def format_metric(value: float, precision: int = 3) -> str:
    return f"{value:.{precision}f}"


def summarize_quality_artifact(
    results_dir: Path,
    *,
    area: str,
    artifact: str,
    extra: str,
) -> HeadlineRow:
    try:
        metrics = parse_scalar_metrics(read_artifact(results_dir, artifact))
        accuracy = metrics["final_test_accuracy"]
        loss = metrics["final_test_loss"]
        memory = metrics["peak_cuda_memory_mb"]
        steady_ms = metrics.get("steady_state_average_step_ms")
    except (FileNotFoundError, KeyError, ValueError) as exc:
        return HeadlineRow(
            area,
            f"missing or invalid artifact: {type(exc).__name__}",
            artifact,
            "missing",
        )

    steady_text = "" if steady_ms is None else f", steady {format_metric(steady_ms)} ms"
    headline = (
        f"{format_percent(accuracy)} accuracy, loss {format_metric(loss, 6)}, "
        f"peak {format_metric(memory, 1)} MB{steady_text}; {extra}"
    )
    return HeadlineRow(area, headline, artifact)


def summarize_mnist_rate_matrix(results_dir: Path) -> HeadlineRow:
    artifact = "mnist_rate_matrix_5090.md"
    try:
        rows = parse_markdown_table(read_artifact(results_dir, artifact), "| Setting |")
        t25_rate = next(
            row for row in rows if row["Setting"] == "t25_h128" and row["Variant"] == "rate"
        )
        t25_triton = next(
            row
            for row in rows
            if row["Setting"] == "t25_h128" and row["Variant"] == "rate_triton_compile"
        )
    except (FileNotFoundError, KeyError, StopIteration) as exc:
        return HeadlineRow(
            "Rate Matrix",
            f"missing or invalid artifact: {type(exc).__name__}",
            artifact,
            "missing",
        )

    headline = (
        "T=25 memory: regular rate "
        f"{t25_rate['Peak CUDA MB']} MB vs triton_compile {t25_triton['Peak CUDA MB']} MB; "
        f"accuracy {t25_rate['Final Test Acc']} vs {t25_triton['Final Test Acc']}"
    )
    return HeadlineRow("Rate Matrix", headline, artifact)


def summarize_performance_frontier(results_dir: Path) -> HeadlineRow:
    artifact = "performance_frontier_5090.md"
    try:
        rows = parse_markdown_table(read_artifact(results_dir, artifact), "| Contract |")
        forward = next(row for row in rows if row["Variant"] == "Triton fused-time")
        rate = next(row for row in rows if row["Variant"] == "Triton checkpoint rate output")
        compiled = next(row for row in rows if row["Variant"] == "torch.compile materialized graph")
    except (FileNotFoundError, KeyError, StopIteration) as exc:
        return HeadlineRow(
            "Performance Frontier",
            f"missing or invalid artifact: {type(exc).__name__}",
            artifact,
            "missing",
        )

    headline = (
        f"forward Triton {forward['Speedup vs Compile']} vs compile; "
        f"training compile {compiled['Fwd+Bwd ms']} ms vs rate Triton {rate['Fwd+Bwd ms']} ms"
    )
    return HeadlineRow("Performance Frontier", headline, artifact)


def summarize_compile_boundary(results_dir: Path) -> HeadlineRow:
    artifact = "compile_triton_boundary_5090.md"
    try:
        rows = parse_markdown_table(read_artifact(results_dir, artifact), "| Path |")
        public = next(
            row
            for row in rows
            if row["Path"] == "torch.compile(public triton_compile rate training)"
        )
        public_bias = next(
            row
            for row in rows
            if row["Path"] == "torch.compile(public triton_compile rate training + bias)"
        )
    except (FileNotFoundError, KeyError, StopIteration) as exc:
        return HeadlineRow(
            "Compile/Triton Boundary",
            f"missing or invalid artifact: {type(exc).__name__}",
            artifact,
            "missing",
        )

    headline = (
        f"public triton_compile {public['ms']} ms / {public['Peak MB']} MB; "
        f"with bias {public_bias['ms']} ms / {public_bias['Peak MB']} MB"
    )
    return HeadlineRow("Compile/Triton Boundary", headline, artifact)


def summarize_snntorch_matrix(results_dir: Path) -> HeadlineRow:
    artifact = "snntorch_matrix_5090.md"
    try:
        rows = parse_markdown_table(read_artifact(results_dir, artifact), "| T | Comparison |")
        rate_t50 = next(
            row
            for row in rows
            if row["T"] == "50" and row["Comparison"] == "rate vs snntorch_dense"
        )
        conv_t50 = next(
            row for row in rows if row["T"] == "50" and row["Comparison"] == "conv vs snntorch_conv"
        )
    except (FileNotFoundError, KeyError, StopIteration) as exc:
        return HeadlineRow(
            "spiker vs snnTorch",
            f"missing or invalid artifact: {type(exc).__name__}",
            artifact,
            "missing",
        )

    headline = (
        f"T=50 steady-step speedups: rate {rate_t50['Steady Step Speedup']}, "
        f"conv {conv_t50['Steady Step Speedup']}; "
        f"rate memory ratio {rate_t50['Peak Memory Ratio']}"
    )
    return HeadlineRow("spiker vs snnTorch", headline, artifact)


def collect_headlines(results_dir: Path) -> list[HeadlineRow]:
    return [
        summarize_quality_artifact(
            results_dir,
            area="Rate MLP Quality",
            artifact="mnist_rate_mlp_dropout_label_smoothing_5090.md",
            extra="larger regularized rate-readout MLP",
        ),
        summarize_quality_artifact(
            results_dir,
            area="Conv Quality",
            artifact="mnist_conv_dropout_label_smoothing_5090.md",
            extra="regularized conv SNN",
        ),
        summarize_mnist_rate_matrix(results_dir),
        summarize_snntorch_matrix(results_dir),
        summarize_performance_frontier(results_dir),
        summarize_compile_boundary(results_dir),
    ]


def print_markdown(rows: list[HeadlineRow], *, results_dir: Path) -> None:
    print("# Headline Benchmark Summary")
    print()
    print(f"Summarizes saved benchmark artifacts from `{results_dir}`.")
    print()
    print("| Area | Headline | Artifact | Status |")
    print("|---|---|---|---|")
    for row in rows:
        print(f"| {row.area} | {row.headline} | `{row.artifact}` | {row.status} |")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=repo_root() / "benchmarks" / "results",
    )
    args = parser.parse_args()

    rows = collect_headlines(args.results_dir)
    print_markdown(rows, results_dir=args.results_dir)


if __name__ == "__main__":
    main()
