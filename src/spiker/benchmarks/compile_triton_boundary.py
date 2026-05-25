"""Measure a compile-visible Triton forward boundary."""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

import torch

from spiker._optional import has_triton
from spiker.baselines import synchronize_if_needed
from spiker.benchmarks.lif import format_memory, format_ms, gpu_name
from spiker.checkpointing import parse_checkpoint_size, resolve_checkpoint_size
from spiker.neurons import LIFParams


@dataclass(frozen=True)
class BoundaryResult:
    path: str
    seconds: float | None
    peak_bytes: int | None
    graph_count: int | None
    graph_breaks: int | None
    note: str
    error: str | None = None


def make_tensors(
    *,
    timesteps: int,
    batch: int,
    features: int,
    neurons: int,
    device: str,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    inputs = torch.rand((timesteps, batch, features), device=device, generator=generator)
    weight = (torch.rand((features, neurons), device=device, generator=generator) - 0.5) * 0.02
    target_rates = torch.full((batch, neurons), 0.05, dtype=inputs.dtype, device=inputs.device)
    return inputs, weight, target_rates


def _time_callable(
    fn: Callable[[], object],
    *,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> tuple[float, int | None]:
    for _ in range(warmup):
        fn()
    synchronize_if_needed(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    synchronize_if_needed(device)
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    return (time.perf_counter() - start) / repeats, peak


def _graph_stats(fn: Callable[[], torch.Tensor]) -> tuple[int | None, int | None]:
    try:
        explanation = torch._dynamo.explain(fn)()
    except Exception:
        return None, None
    return int(explanation.graph_count), int(explanation.graph_break_count)


def _run_result(
    *,
    path: str,
    fn: Callable[[], torch.Tensor],
    device: torch.device,
    warmup: int,
    repeats: int,
    note: str,
    graph_stats: bool,
) -> BoundaryResult:
    try:
        seconds, peak = _time_callable(fn, device=device, warmup=warmup, repeats=repeats)
        graph_count, graph_breaks = _graph_stats(fn) if graph_stats else (None, None)
        return BoundaryResult(path, seconds, peak, graph_count, graph_breaks, note)
    except Exception as exc:  # noqa: BLE001 - benchmark should report backend failures.
        synchronize_if_needed(device)
        return BoundaryResult(path, None, None, None, None, note, f"{type(exc).__name__}: {exc}")


def make_raw_triton_loss(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    target_rates: torch.Tensor,
    params: LIFParams,
    *,
    checkpoint_size: int,
) -> Callable[[], torch.Tensor]:
    from spiker.triton import linear_surrogate_lif_checkpoint_rate_forward

    def run() -> torch.Tensor:
        final_state, rates, _chunk_starts = linear_surrogate_lif_checkpoint_rate_forward(
            inputs,
            weight,
            None,
            params,
            checkpoint_size=checkpoint_size,
        )
        return (rates - target_rates).pow(2).mean() + 0.01 * final_state.membrane.pow(2).mean()

    return run


def make_compile_visible_loss(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    target_rates: torch.Tensor,
    params: LIFParams,
    *,
    checkpoint_size: int,
) -> Callable[[], torch.Tensor]:
    from spiker.triton import linear_lif_checkpoint_rate_forward_no_bias_op

    def run() -> torch.Tensor:
        final_membrane, rates, _chunk_starts = linear_lif_checkpoint_rate_forward_no_bias_op(
            inputs,
            weight,
            params.decay,
            params.threshold,
            params.reset,
            5.0,
            checkpoint_size,
            16,
            32,
            32,
        )
        return (rates - target_rates).pow(2).mean() + 0.01 * final_membrane.pow(2).mean()

    return run


def make_existing_triton_training_step(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    target_rates: torch.Tensor,
    params: LIFParams,
    *,
    checkpoint_size: int,
) -> Callable[[], torch.Tensor]:
    from spiker.kernels import linear_surrogate_lif_rate_forward

    def run() -> torch.Tensor:
        weight.grad = None
        final_state, rates = linear_surrogate_lif_rate_forward(
            inputs,
            weight,
            None,
            params,
            surrogate="fast_sigmoid",
            surrogate_slope=5.0,
            backend="triton",
            checkpoint_size=checkpoint_size,
            reduction="none",
        )
        loss = (rates - target_rates).pow(2).mean()
        loss = loss + 0.01 * final_state.membrane.pow(2).mean()
        loss.backward()
        return loss

    return run


def make_compile_visible_training_step(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    target_rates: torch.Tensor,
    params: LIFParams,
    *,
    checkpoint_size: int,
    compile_mode: str | None = None,
) -> Callable[[], torch.Tensor]:
    loss_fn = make_compile_visible_loss(
        inputs,
        weight,
        target_rates,
        params,
        checkpoint_size=checkpoint_size,
    )
    if compile_mode is not None:
        loss_fn = torch.compile(loss_fn, mode=compile_mode, fullgraph=True)

    def run() -> torch.Tensor:
        weight.grad = None
        loss = loss_fn()
        loss.backward()
        return loss

    return run


def make_public_triton_compile_training_step(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    target_rates: torch.Tensor,
    params: LIFParams,
    *,
    checkpoint_size: int,
    compile_mode: str | None = None,
) -> Callable[[], torch.Tensor]:
    from spiker.kernels import linear_surrogate_lif_rate_forward

    def loss_fn() -> torch.Tensor:
        final_state, rates = linear_surrogate_lif_rate_forward(
            inputs,
            weight,
            bias,
            params,
            surrogate="fast_sigmoid",
            surrogate_slope=5.0,
            backend="triton_compile",
            checkpoint_size=checkpoint_size,
            reduction="none",
        )
        loss = (rates - target_rates).pow(2).mean()
        return loss + 0.01 * final_state.membrane.pow(2).mean()

    compiled_loss_fn = (
        torch.compile(loss_fn, mode=compile_mode, fullgraph=True)
        if compile_mode is not None
        else loss_fn
    )

    def run() -> torch.Tensor:
        weight.grad = None
        if bias is not None:
            bias.grad = None
        loss = compiled_loss_fn()
        loss.backward()
        return loss

    return run


def run_benchmark(args: argparse.Namespace) -> list[BoundaryResult]:
    device = torch.device(args.device)
    if device.type != "cuda" or not has_triton():
        return [
            BoundaryResult(
                "compile-visible Triton op",
                None,
                None,
                None,
                None,
                "requires CUDA and Triton",
                "CUDA and Triton are required",
            )
        ]

    if args.matmul_precision is not None:
        torch.set_float32_matmul_precision(args.matmul_precision)

    inputs, weight, target_rates = make_tensors(
        timesteps=args.timesteps,
        batch=args.batch,
        features=args.features,
        neurons=args.neurons,
        device=args.device,
        seed=args.seed,
    )
    params = LIFParams()
    checkpoint_size = resolve_checkpoint_size(args.timesteps, args.checkpoint_size)

    raw_triton_loss = make_raw_triton_loss(
        inputs,
        weight,
        target_rates,
        params,
        checkpoint_size=checkpoint_size,
    )
    compile_visible_loss = make_compile_visible_loss(
        inputs,
        weight,
        target_rates,
        params,
        checkpoint_size=checkpoint_size,
    )
    compiled_visible_loss = torch.compile(
        compile_visible_loss,
        mode=args.compile_mode,
        fullgraph=True,
    )
    existing_training_weight = weight.detach().clone().requires_grad_(True)
    visible_training_weight = weight.detach().clone().requires_grad_(True)
    compiled_visible_training_weight = weight.detach().clone().requires_grad_(True)
    public_compiled_training_weight = weight.detach().clone().requires_grad_(True)
    public_compiled_training_bias_weight = weight.detach().clone().requires_grad_(True)
    public_compile_bias = (
        (torch.rand((args.neurons,), dtype=weight.dtype, device=weight.device) - 0.5) * 0.01
    ).requires_grad_(True)
    existing_training = make_existing_triton_training_step(
        inputs,
        existing_training_weight,
        target_rates,
        params,
        checkpoint_size=checkpoint_size,
    )
    visible_training = make_compile_visible_training_step(
        inputs,
        visible_training_weight,
        target_rates,
        params,
        checkpoint_size=checkpoint_size,
    )
    compiled_visible_training = make_compile_visible_training_step(
        inputs,
        compiled_visible_training_weight,
        target_rates,
        params,
        checkpoint_size=checkpoint_size,
        compile_mode=args.compile_mode,
    )
    public_compiled_training = make_public_triton_compile_training_step(
        inputs,
        public_compiled_training_weight,
        None,
        target_rates,
        params,
        checkpoint_size=checkpoint_size,
        compile_mode=args.compile_mode,
    )
    public_compiled_training_bias = make_public_triton_compile_training_step(
        inputs,
        public_compiled_training_bias_weight,
        public_compile_bias,
        target_rates,
        params,
        checkpoint_size=checkpoint_size,
        compile_mode=args.compile_mode,
    )

    return [
        _run_result(
            path="raw Python Triton wrapper + eager loss",
            fn=raw_triton_loss,
            device=device,
            warmup=args.warmup,
            repeats=args.repeats,
            note="existing forward wrapper; downstream loss is outside torch.compile",
            graph_stats=False,
        ),
        _run_result(
            path="triton_op wrapper + eager loss",
            fn=compile_visible_loss,
            device=device,
            warmup=args.warmup,
            repeats=args.repeats,
            note="same kernel exposed through torch.library.triton_op",
            graph_stats=True,
        ),
        _run_result(
            path="torch.compile(triton_op wrapper + loss)",
            fn=compiled_visible_loss,
            device=device,
            warmup=args.warmup,
            repeats=args.repeats,
            note="tests whether Inductor can capture Triton launch plus downstream loss",
            graph_stats=False,
        ),
        _run_result(
            path="existing Triton custom autograd rate training",
            fn=existing_training,
            device=device,
            warmup=args.warmup,
            repeats=args.repeats,
            note="current public rate-training path, including loss.backward()",
            graph_stats=False,
        ),
        _run_result(
            path="triton_op registered-autograd rate training",
            fn=visible_training,
            device=device,
            warmup=args.warmup,
            repeats=args.repeats,
            note="compile-visible forward with registered custom-op backward",
            graph_stats=False,
        ),
        _run_result(
            path="torch.compile(triton_op registered-autograd rate training)",
            fn=compiled_visible_training,
            device=device,
            warmup=args.warmup,
            repeats=args.repeats,
            note="compiled forward/loss with registered custom-op backward",
            graph_stats=False,
        ),
        _run_result(
            path="torch.compile(public triton_compile rate training)",
            fn=public_compiled_training,
            device=device,
            warmup=args.warmup,
            repeats=args.repeats,
            note="same compile-visible path through public backend='triton_compile'",
            graph_stats=False,
        ),
        _run_result(
            path="torch.compile(public triton_compile rate training + bias)",
            fn=public_compiled_training_bias,
            device=device,
            warmup=args.warmup,
            repeats=args.repeats,
            note="same public backend with bias gradients enabled",
            graph_stats=False,
        ),
    ]


def print_markdown(args: argparse.Namespace, results: list[BoundaryResult]) -> None:
    print("# Compile-Visible Triton Boundary")
    print()
    print("Measures an experimental `torch.library.triton_op` wrapper around the")
    print("checkpointed linear LIF rate-forward kernel.")
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
    print(f"- `checkpoint_size`: `{args.checkpoint_size}`")
    print(
        f"- `resolved_checkpoint_size`: "
        f"`{resolve_checkpoint_size(args.timesteps, args.checkpoint_size)}`"
    )
    print(f"- `compile_mode`: `{args.compile_mode}`")
    print(f"- `warmup`: `{args.warmup}`")
    print(f"- `repeats`: `{args.repeats}`")
    print()
    print("## Results")
    print()
    print("| Path | ms | Peak MB | Graph Count | Graph Breaks | Note | Error |")
    print("|---|---:|---:|---:|---:|---|---|")
    for result in results:
        graph_count = "" if result.graph_count is None else str(result.graph_count)
        graph_breaks = "" if result.graph_breaks is None else str(result.graph_breaks)
        print(
            f"| {result.path} | "
            f"{format_ms(result.seconds)} | "
            f"{format_memory(result.peak_bytes)} | "
            f"{graph_count} | "
            f"{graph_breaks} | "
            f"{result.note} | "
            f"{result.error or ''} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--features", type=int, default=128)
    parser.add_argument("--neurons", type=int, default=2048)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint-size", type=parse_checkpoint_size, default=25)
    parser.add_argument(
        "--compile-mode",
        choices=["default", "reduce-overhead", "max-autotune"],
        default="reduce-overhead",
    )
    parser.add_argument(
        "--matmul-precision",
        choices=["highest", "high", "medium"],
        default="high",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print_markdown(args, run_benchmark(args))


if __name__ == "__main__":
    main()
