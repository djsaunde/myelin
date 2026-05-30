"""Benchmark dense vs directly bitpacked Triton LIF forward."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from datetime import UTC, datetime

import torch

from myelin._optional import has_triton
from myelin.baselines import synchronize_if_needed
from myelin.benchmarks.lif import format_memory, format_ms, format_speedup, gpu_name, parse_shape
from myelin.neurons import LIFParams, LIFState
from myelin.packing import packed_spike_bytes

DEFAULT_SHAPES = [
    (25, 64, 2048),
    (100, 64, 2048),
    (200, 64, 2048),
]


@dataclass(frozen=True)
class Result:
    timesteps: int
    batch: int
    neurons: int
    dense_seconds: float | None
    dense_pack_seconds: float | None
    packed_seconds: float | None
    dense_spike_bytes: int | None
    packed_spike_bytes: int | None
    speedup_vs_dense: float | None
    speedup_vs_dense_pack: float | None
    round_trip_ok: bool | None
    error: str | None = None


def time_call(fn, *, warmup: int, repeats: int, device: torch.device) -> float:
    for _ in range(warmup):
        fn()
    synchronize_if_needed(device)

    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    synchronize_if_needed(device)
    return (time.perf_counter() - start) / repeats


def run_one(
    timesteps: int,
    batch: int,
    neurons: int,
    device: str,
    *,
    warmup: int,
    repeats: int,
) -> Result:
    if torch.device(device).type != "cuda" or not has_triton():
        return Result(
            timesteps,
            batch,
            neurons,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            "CUDA and Triton are required",
        )

    from myelin.kernels import lif_forward_packed_spikes
    from myelin.triton import lif_forward, pack_spikes_triton, unpack_spikes_triton

    inputs = torch.rand((timesteps, batch, neurons), device=device)
    initial = LIFState(membrane=torch.rand((batch, neurons), device=device) * 0.2)
    params = LIFParams()

    try:
        dense_state, dense_spikes = lif_forward(inputs, initial, params)
        packed_state, packed_spikes = lif_forward_packed_spikes(
            inputs,
            initial,
            params,
            backend="triton",
        )
        unpacked = unpack_spikes_triton(packed_spikes, dtype=inputs.dtype)
        round_trip_ok = bool(
            torch.equal(unpacked, dense_spikes)
            and torch.allclose(packed_state.membrane, dense_state.membrane)
        )

        dense_seconds = time_call(
            lambda: lif_forward(inputs, initial, params),
            warmup=warmup,
            repeats=repeats,
            device=inputs.device,
        )
        dense_pack_seconds = time_call(
            lambda: pack_spikes_triton(lif_forward(inputs, initial, params)[1]),
            warmup=warmup,
            repeats=repeats,
            device=inputs.device,
        )
        packed_seconds = time_call(
            lambda: lif_forward_packed_spikes(inputs, initial, params, backend="triton"),
            warmup=warmup,
            repeats=repeats,
            device=inputs.device,
        )
        dense_bytes = dense_spikes.numel() * dense_spikes.element_size()
        packed_bytes = packed_spike_bytes(packed_spikes)
        return Result(
            timesteps,
            batch,
            neurons,
            dense_seconds,
            dense_pack_seconds,
            packed_seconds,
            dense_bytes,
            packed_bytes,
            dense_seconds / packed_seconds,
            dense_pack_seconds / packed_seconds,
            round_trip_ok,
        )
    except Exception as exc:  # noqa: BLE001 - benchmark should report backend failures.
        return Result(
            timesteps,
            batch,
            neurons,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            f"{type(exc).__name__}: {exc}",
        )


def print_markdown(args: argparse.Namespace, results: list[Result]) -> None:
    print("# Packed LIF Forward Benchmark")
    print()
    print("## Environment")
    print()
    print(f"- `generated_utc`: `{datetime.now(UTC).isoformat(timespec='seconds')}`")
    print(f"- `device`: `{args.device}`")
    print(f"- `gpu`: `{gpu_name(args.device)}`")
    print(f"- `torch`: `{torch.__version__}`")
    print(f"- `cuda_available`: `{torch.cuda.is_available()}`")
    print(f"- `cuda_version`: `{torch.version.cuda}`")
    print(f"- `warmup`: `{args.warmup}`")
    print(f"- `repeats`: `{args.repeats}`")
    print()
    print("## Results")
    print()
    print(
        "| T | Batch | N | Dense ms | Dense+Pack ms | Direct Packed ms | "
        "Packed vs Dense | Packed vs Dense+Pack | Dense Spikes MB | Packed Spikes MB | "
        "Round Trip | Error |"
    )
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|")
    for result in results:
        print(
            f"| {result.timesteps} | {result.batch} | {result.neurons} | "
            f"{format_ms(result.dense_seconds)} | "
            f"{format_ms(result.dense_pack_seconds)} | "
            f"{format_ms(result.packed_seconds)} | "
            f"{format_speedup(result.speedup_vs_dense)} | "
            f"{format_speedup(result.speedup_vs_dense_pack)} | "
            f"{format_memory(result.dense_spike_bytes)} | "
            f"{format_memory(result.packed_spike_bytes)} | "
            f"{'' if result.round_trip_ok is None else result.round_trip_ok} | "
            f"{result.error or ''} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--shape", action="append", type=parse_shape, default=[])
    args = parser.parse_args()

    if args.warmup < 0 or args.repeats <= 0:
        raise ValueError("warmup must be non-negative and repeats must be positive")
    shapes = args.shape or DEFAULT_SHAPES
    results = [
        run_one(t, b, n, args.device, warmup=args.warmup, repeats=args.repeats)
        for t, b, n in shapes
    ]
    print_markdown(args, results)


if __name__ == "__main__":
    main()
