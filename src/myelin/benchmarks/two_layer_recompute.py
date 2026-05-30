"""Benchmark composed two-layer rate training against whole-model recompute."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from datetime import UTC, datetime

import torch
from torch import nn

from myelin.checkpointing import parse_checkpoint_size, resolve_checkpoint_size
from myelin.kernels import (
    two_layer_surrogate_lif_rate_packed_hidden_forward,
    two_layer_surrogate_lif_rate_recompute_forward,
)
from myelin.modules import (
    LinearSurrogateLIF,
    LinearSurrogateLIFRate,
    LinearSynapse,
    fast_sigmoid_surrogate,
)
from myelin.neurons import LIFParams


@dataclass(frozen=True)
class Shape:
    timesteps: int
    batch: int
    features: int
    hidden: int
    classes: int


@dataclass(frozen=True)
class BenchRow:
    shape: Shape
    variant: str
    ms: float | None
    peak_mb: float | None
    error: str = ""


class ComposedRateModel(nn.Module):
    def __init__(self, shape: Shape, checkpoint_size: int) -> None:
        super().__init__()
        params = LIFParams()
        self.hidden = LinearSurrogateLIF(
            shape.features,
            shape.hidden,
            params,
            surrogate=fast_sigmoid_surrogate,
            surrogate_slope=5.0,
            backend="triton",
            stream_synapse=True,
            checkpoint_size=checkpoint_size,
        )
        self.output = LinearSurrogateLIFRate(
            shape.hidden,
            shape.classes,
            params,
            surrogate=fast_sigmoid_surrogate,
            surrogate_slope=5.0,
            backend="triton",
            checkpoint_size=checkpoint_size,
            reduction="none",
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.output(self.hidden(inputs))


class PackedHiddenRateModel(nn.Module):
    def __init__(self, shape: Shape, checkpoint_size: int) -> None:
        super().__init__()
        self.hidden = LinearSynapse(shape.features, shape.hidden)
        self.output = LinearSynapse(shape.hidden, shape.classes)
        self.params = LIFParams()
        self.checkpoint_size = checkpoint_size

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return two_layer_surrogate_lif_rate_packed_hidden_forward(
            inputs,
            self.hidden.weight,
            self.hidden.bias,
            self.output.weight,
            self.output.bias,
            self.params,
            surrogate="fast_sigmoid",
            surrogate_slope=5.0,
            checkpoint_size=self.checkpoint_size,
        )


class RecomputeRateModel(nn.Module):
    def __init__(self, shape: Shape, checkpoint_size: int) -> None:
        super().__init__()
        self.hidden = LinearSynapse(shape.features, shape.hidden)
        self.output = LinearSynapse(shape.hidden, shape.classes)
        self.params = LIFParams()
        self.checkpoint_size = checkpoint_size

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return two_layer_surrogate_lif_rate_recompute_forward(
            inputs,
            self.hidden.weight,
            self.hidden.bias,
            self.output.weight,
            self.output.bias,
            self.params,
            surrogate="fast_sigmoid",
            surrogate_slope=5.0,
            checkpoint_size=self.checkpoint_size,
        )


def parse_shape(value: str) -> Shape:
    parts = value.split(",")
    if len(parts) != 5:
        raise argparse.ArgumentTypeError("shape must be T,B,F,H,C")
    try:
        timesteps, batch, features, hidden, classes = (int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("shape values must be integers") from exc
    if min(timesteps, batch, features, hidden, classes) <= 0:
        raise argparse.ArgumentTypeError("shape values must be positive")
    return Shape(timesteps, batch, features, hidden, classes)


def sync(device: str) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(torch.device(device))


def clone_recompute_from_composed(
    composed: ComposedRateModel,
    shape: Shape,
    checkpoint_size: int,
) -> RecomputeRateModel:
    recompute = RecomputeRateModel(shape, checkpoint_size)
    recompute.hidden.weight.data.copy_(composed.hidden.synapse.weight)
    recompute.output.weight.data.copy_(composed.output.synapse.weight)
    assert composed.hidden.synapse.bias is not None
    assert composed.output.synapse.bias is not None
    assert recompute.hidden.bias is not None
    assert recompute.output.bias is not None
    recompute.hidden.bias.data.copy_(composed.hidden.synapse.bias)
    recompute.output.bias.data.copy_(composed.output.synapse.bias)
    return recompute


def clone_packed_hidden_from_composed(
    composed: ComposedRateModel,
    shape: Shape,
    checkpoint_size: int,
) -> PackedHiddenRateModel:
    packed = PackedHiddenRateModel(shape, checkpoint_size)
    packed.hidden.weight.data.copy_(composed.hidden.synapse.weight)
    packed.output.weight.data.copy_(composed.output.synapse.weight)
    assert composed.hidden.synapse.bias is not None
    assert composed.output.synapse.bias is not None
    assert packed.hidden.bias is not None
    assert packed.output.bias is not None
    packed.hidden.bias.data.copy_(composed.hidden.synapse.bias)
    packed.output.bias.data.copy_(composed.output.synapse.bias)
    return packed


def measure_step(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    device: str,
    warmup: int,
    repeats: int,
) -> tuple[float, float]:
    loss_fn = nn.CrossEntropyLoss()
    params = tuple(model.parameters())
    for _ in range(warmup):
        for parameter in params:
            parameter.grad = None
        loss = loss_fn(model(inputs), targets)
        loss.backward()
    sync(device)
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(torch.device(device))
    timings = []
    for _ in range(repeats):
        for parameter in params:
            parameter.grad = None
        sync(device)
        start = time.perf_counter()
        loss = loss_fn(model(inputs), targets)
        loss.backward()
        sync(device)
        timings.append(time.perf_counter() - start)
    peak_mb = 0.0
    if torch.device(device).type == "cuda":
        peak_mb = torch.cuda.max_memory_allocated(torch.device(device)) / (1024 * 1024)
    return sum(timings) / len(timings) * 1000, peak_mb


def run_benchmark(args: argparse.Namespace) -> list[BenchRow]:
    if torch.device(args.device).type != "cuda":
        return [BenchRow(shape, "composed", None, None, "CUDA is required") for shape in args.shape]

    torch.manual_seed(args.seed)
    rows: list[BenchRow] = []
    for shape in args.shape:
        checkpoint_size = resolve_checkpoint_size(shape.timesteps, args.checkpoint_size)
        inputs = torch.rand(
            (shape.timesteps, shape.batch, shape.features),
            device=args.device,
        )
        targets = torch.randint(0, shape.classes, (shape.batch,), device=args.device)
        composed = ComposedRateModel(shape, checkpoint_size).to(device=args.device)
        recompute = clone_recompute_from_composed(composed, shape, checkpoint_size).to(
            device=args.device
        )
        packed_hidden = clone_packed_hidden_from_composed(
            composed,
            shape,
            checkpoint_size,
        ).to(device=args.device)
        for variant, model in (
            ("composed", composed),
            ("packed_hidden", packed_hidden),
            ("recompute", recompute),
        ):
            try:
                ms, peak_mb = measure_step(
                    model,
                    inputs,
                    targets,
                    device=args.device,
                    warmup=args.warmup,
                    repeats=args.repeats,
                )
                rows.append(BenchRow(shape, variant, ms, peak_mb))
            except Exception as exc:  # pragma: no cover - benchmark diagnostics
                rows.append(BenchRow(shape, variant, None, None, f"{type(exc).__name__}: {exc}"))
    return rows


def format_optional(value: float | None, precision: int = 3) -> str:
    if value is None:
        return ""
    return f"{value:.{precision}f}"


def summarize_result(rows: list[BenchRow]) -> str:
    complete_rows = [row for row in rows if row.ms is not None and row.peak_mb is not None]
    if len(complete_rows) != len(rows):
        return "One or more variants failed; inspect the Error column before drawing conclusions."

    by_shape: dict[Shape, dict[str, BenchRow]] = {}
    for row in complete_rows:
        by_shape.setdefault(row.shape, {})[row.variant] = row

    slower_shapes = 0
    lower_memory_shapes = 0
    packed_faster_shapes = 0
    packed_lower_memory_shapes = 0
    for variants in by_shape.values():
        composed = variants.get("composed")
        recompute = variants.get("recompute")
        if composed is None or recompute is None:
            continue
        assert recompute.ms is not None
        assert composed.ms is not None
        assert recompute.peak_mb is not None
        assert composed.peak_mb is not None
        slower_shapes += int(recompute.ms > composed.ms)
        lower_memory_shapes += int(recompute.peak_mb < composed.peak_mb)
        packed = variants.get("packed_hidden")
        if packed is not None:
            assert packed.ms is not None
            assert packed.peak_mb is not None
            packed_faster_shapes += int(packed.ms < composed.ms)
            packed_lower_memory_shapes += int(packed.peak_mb < composed.peak_mb)

    if slower_shapes == len(by_shape) and lower_memory_shapes == 0:
        return (
            "Whole-model PyTorch recompute was slower at every tested shape and did not "
            "reduce peak memory. Packed-hidden boundary storage was faster than the "
            f"composed path on {packed_faster_shapes}/{len(by_shape)} shapes, but lower-memory "
            f"on {packed_lower_memory_shapes}/{len(by_shape)} shapes. This says packed storage "
            "alone is not enough for these shapes; the remaining lever is "
            "avoiding dense unpack/backward scratch."
        )
    return (
        f"Whole-model recompute was slower on {slower_shapes}/{len(by_shape)} shapes "
        f"and lower-memory on {lower_memory_shapes}/{len(by_shape)} shapes."
    )


def print_markdown(args: argparse.Namespace, rows: list[BenchRow]) -> None:
    print("# Two-Layer Rate Recompute Benchmark")
    print()
    print(
        "Compares the composed two-layer rate model with packed-hidden boundary storage "
        "and a whole-model recompute path."
    )
    print()
    print("## Environment")
    print()
    print(f"- `generated_utc`: `{datetime.now(UTC).isoformat(timespec='seconds')}`")
    print(f"- `device`: `{args.device}`")
    print(f"- `checkpoint_size`: `{args.checkpoint_size}`")
    print(f"- `warmup`: `{args.warmup}`")
    print(f"- `repeats`: `{args.repeats}`")
    print()
    print("## Results")
    print()
    print("| T | B | F | H | C | Variant | ms | Peak MB | Error |")
    print("|---:|---:|---:|---:|---:|---|---:|---:|---|")
    for row in rows:
        print(
            f"| {row.shape.timesteps} | {row.shape.batch} | {row.shape.features} | "
            f"{row.shape.hidden} | {row.shape.classes} | {row.variant} | "
            f"{format_optional(row.ms)} | {format_optional(row.peak_mb, 1)} | {row.error} |"
        )
    print()
    print("## Interpretation")
    print()
    print(summarize_result(rows))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--shape",
        type=parse_shape,
        action="append",
        default=None,
    )
    parser.add_argument("--checkpoint-size", type=parse_checkpoint_size, default="balanced")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.shape is None:
        args.shape = [
            Shape(10, 128, 784, 128, 10),
            Shape(25, 128, 784, 128, 10),
            Shape(50, 128, 784, 128, 10),
        ]

    rows = run_benchmark(args)
    print_markdown(args, rows)


if __name__ == "__main__":
    main()
