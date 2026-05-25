"""Benchmark SpikeGPT-style language-model training paths."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import cast

import torch
from torch import nn

from spiker.baselines import (
    max_cuda_memory_allocated,
    reset_cuda_peak_memory,
    synchronize_if_needed,
)
from spiker.benchmarks.lif import format_memory, format_ms, gpu_name
from spiker.language import SpikeGPTConfig, SpikeLanguageModel


@dataclass(frozen=True)
class TrainingResult:
    path: str
    seconds: float | None
    peak_bytes: int | None
    tokens_per_second: float | None
    loss: float | None
    error: str | None = None


def _make_batch(
    args: argparse.Namespace, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = torch.randint(
        low=0,
        high=args.vocab_size,
        size=(args.batch, args.context_length + 1),
        device=device,
    )
    return tokens[:, :-1], tokens[:, 1:]


def _make_model(args: argparse.Namespace, device: torch.device) -> SpikeLanguageModel:
    config = SpikeGPTConfig(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        n_layer=args.layers,
        n_embd=args.embedding,
        dropout=args.dropout,
        lif_threshold=args.lif_threshold,
        spike_embedding=not args.dense_embedding,
    )
    return SpikeLanguageModel(config).to(device=device)


def _train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    optimizer.zero_grad(set_to_none=True)
    result = model(inputs, targets)
    if not isinstance(result, tuple):
        raise RuntimeError("SpikeGPT training benchmark expected (loss, logits)")
    loss, _logits = result
    loss.backward()
    optimizer.step()
    return loss.detach()


def _benchmark_path(
    args: argparse.Namespace,
    *,
    path: str,
    compile_model: bool,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
) -> TrainingResult:
    try:
        torch.manual_seed(args.seed)
        raw_model = _make_model(args, device)
        model: nn.Module = raw_model
        if compile_model:
            model = cast(
                nn.Module,
                torch.compile(raw_model, mode="reduce-overhead", fullgraph=True),
            )
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

        last_loss = None
        for _ in range(args.warmup):
            last_loss = _train_step(model, optimizer, inputs, targets)
        synchronize_if_needed(device)
        reset_cuda_peak_memory(device)

        start = time.perf_counter()
        for _ in range(args.repeats):
            last_loss = _train_step(model, optimizer, inputs, targets)
        synchronize_if_needed(device)
        seconds = (time.perf_counter() - start) / args.repeats
        tokens_per_second = inputs.numel() / seconds
        return TrainingResult(
            path=path,
            seconds=seconds,
            peak_bytes=max_cuda_memory_allocated(device),
            tokens_per_second=tokens_per_second,
            loss=None if last_loss is None else float(last_loss),
        )
    except Exception as exc:  # noqa: BLE001 - benchmark should report compile/runtime failures.
        synchronize_if_needed(device)
        return TrainingResult(
            path=path,
            seconds=None,
            peak_bytes=None,
            tokens_per_second=None,
            loss=None,
            error=f"{type(exc).__name__}: {exc}",
        )


def run_benchmark(args: argparse.Namespace) -> list[TrainingResult]:
    if args.batch <= 0:
        raise ValueError("--batch must be positive")
    if args.context_length <= 0:
        raise ValueError("--context-length must be positive")
    if args.layers <= 0:
        raise ValueError("--layers must be positive")
    if args.embedding <= 0:
        raise ValueError("--embedding must be positive")
    if args.vocab_size <= 0:
        raise ValueError("--vocab-size must be positive")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")

    device = torch.device(args.device)
    torch.set_float32_matmul_precision(args.matmul_precision)
    torch.manual_seed(args.seed)
    inputs, targets = _make_batch(args, device)
    rows = [
        _benchmark_path(
            args,
            path="eager",
            compile_model=False,
            inputs=inputs,
            targets=targets,
            device=device,
        )
    ]
    if args.compile:
        rows.append(
            _benchmark_path(
                args,
                path="torch_compile_fullgraph",
                compile_model=True,
                inputs=inputs,
                targets=targets,
                device=device,
            )
        )
    return rows


def print_markdown(args: argparse.Namespace, rows: list[TrainingResult]) -> None:
    print("# SpikeGPT Training Benchmark")
    print()
    print(f"Generated: {datetime.now(UTC).isoformat()}")
    print(f"Device: {args.device} ({gpu_name(args.device)})")
    print(
        "Shape: "
        f"batch={args.batch}, context_length={args.context_length}, layers={args.layers}, "
        f"embedding={args.embedding}, vocab_size={args.vocab_size}"
    )
    print(
        f"Warmup: {args.warmup}; repeats: {args.repeats}; seed: {args.seed}; "
        f"compile={args.compile}; matmul_precision={args.matmul_precision}"
    )
    print()
    print("| Path | Step time | Tokens/s | Peak memory | Loss | Error |")
    print("|---|---:|---:|---:|---:|---|")
    for row in rows:
        tokens_per_second = "" if row.tokens_per_second is None else f"{row.tokens_per_second:.1f}"
        loss = "" if row.loss is None else f"{row.loss:.6f}"
        print(
            f"| {row.path} | {format_ms(row.seconds)} | {tokens_per_second} | "
            f"{format_memory(row.peak_bytes)} | {loss} | {row.error or ''} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--embedding", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lif-threshold", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument(
        "--dense-embedding",
        action="store_true",
        help="use dense token embeddings instead of hard surrogate binary embeddings",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--matmul-precision", choices=("highest", "high", "medium"), default="high")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    print_markdown(args, run_benchmark(args))


if __name__ == "__main__":
    main()
