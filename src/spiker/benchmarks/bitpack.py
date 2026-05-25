"""Benchmark bitpacked spike storage footprint."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from datetime import UTC, datetime

import torch

from spiker._optional import has_triton
from spiker.baselines import synchronize_if_needed
from spiker.benchmarks.lif import format_memory, format_ms, gpu_name, parse_shape
from spiker.packing import (
    dense_spike_bytes,
    pack_spikes,
    packed_spike_bytes,
    packed_spike_count,
    spike_compression_ratio,
    unpack_spikes,
)

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
    dense_bytes: int
    packed_bytes: int
    compression_ratio: float
    packed_shape: tuple[int, ...]
    torch_pack_seconds: float
    triton_pack_seconds: float | None
    packed_count_backend: str
    packed_count_seconds: float
    unpack_sum_seconds: float
    per_neuron_count_seconds: float
    unpack_per_neuron_sum_seconds: float
    spike_count_ok: bool
    per_neuron_count_ok: bool
    round_trip_ok: bool
    triton_round_trip_ok: bool | None


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
    rate: float,
    *,
    warmup: int,
    repeats: int,
) -> Result:
    spikes = (torch.rand((timesteps, batch, neurons), device=device) < rate).to(torch.float32)
    packed = pack_spikes(spikes)
    unpacked = unpack_spikes(packed, dtype=spikes.dtype)
    torch_pack_seconds = time_call(
        lambda: pack_spikes(spikes),
        warmup=warmup,
        repeats=repeats,
        device=spikes.device,
    )
    expected_count = spikes.sum().to(torch.int64)
    packed_count = packed_spike_count(packed)
    packed_count_seconds = time_call(
        lambda: packed_spike_count(packed),
        warmup=warmup,
        repeats=repeats,
        device=spikes.device,
    )
    leading_dims = tuple(range(spikes.ndim - 1))
    expected_per_neuron_count = spikes.sum(dim=leading_dims).to(torch.int64)
    per_neuron_count = packed_spike_count(packed, dim=leading_dims)
    per_neuron_count_seconds = time_call(
        lambda: packed_spike_count(packed, dim=leading_dims),
        warmup=warmup,
        repeats=repeats,
        device=spikes.device,
    )
    unpack_sum_seconds = time_call(
        lambda: unpack_spikes(packed, dtype=spikes.dtype).sum(),
        warmup=warmup,
        repeats=repeats,
        device=spikes.device,
    )
    unpack_per_neuron_sum_seconds = time_call(
        lambda: unpack_spikes(packed, dtype=spikes.dtype).sum(dim=leading_dims),
        warmup=warmup,
        repeats=repeats,
        device=spikes.device,
    )
    triton_pack_seconds = None
    triton_round_trip_ok = None
    packed_count_backend = "torch tensor bit ops"
    if spikes.is_cuda and has_triton():
        from spiker.triton import pack_spikes_triton, unpack_spikes_triton

        packed_count_backend = "Triton auto"
        triton_packed = pack_spikes_triton(spikes)
        triton_unpacked = unpack_spikes_triton(triton_packed, dtype=spikes.dtype)
        triton_round_trip_ok = bool(torch.equal(triton_unpacked, spikes))
        triton_pack_seconds = time_call(
            lambda: pack_spikes_triton(spikes),
            warmup=warmup,
            repeats=repeats,
            device=spikes.device,
        )
    return Result(
        timesteps=timesteps,
        batch=batch,
        neurons=neurons,
        dense_bytes=dense_spike_bytes(spikes),
        packed_bytes=packed_spike_bytes(packed),
        compression_ratio=spike_compression_ratio(spikes, packed),
        packed_shape=packed.packed_shape,
        torch_pack_seconds=torch_pack_seconds,
        triton_pack_seconds=triton_pack_seconds,
        packed_count_backend=packed_count_backend,
        packed_count_seconds=packed_count_seconds,
        unpack_sum_seconds=unpack_sum_seconds,
        per_neuron_count_seconds=per_neuron_count_seconds,
        unpack_per_neuron_sum_seconds=unpack_per_neuron_sum_seconds,
        spike_count_ok=bool(torch.equal(packed_count.cpu(), expected_count.cpu())),
        per_neuron_count_ok=bool(
            torch.equal(per_neuron_count.cpu(), expected_per_neuron_count.cpu())
        ),
        round_trip_ok=bool(torch.equal(unpacked, spikes)),
        triton_round_trip_ok=triton_round_trip_ok,
    )


def print_markdown(args: argparse.Namespace, results: list[Result]) -> None:
    print("# Bitpacked Spike Storage Benchmark")
    print()
    print("## Environment")
    print()
    print(f"- `generated_utc`: `{datetime.now(UTC).isoformat(timespec='seconds')}`")
    print(f"- `device`: `{args.device}`")
    print(f"- `gpu`: `{gpu_name(args.device)}`")
    print(f"- `torch`: `{torch.__version__}`")
    print(f"- `cuda_available`: `{torch.cuda.is_available()}`")
    print(f"- `cuda_version`: `{torch.version.cuda}`")
    print(f"- `rate`: `{args.rate}`")
    print(f"- `warmup`: `{args.warmup}`")
    print(f"- `repeats`: `{args.repeats}`")
    print()
    print("## Results")
    print()
    print(
        "| T | Batch | N | Dense MB | Packed MB | Compression | Packed Shape | "
        "Torch Pack ms | Triton Pack ms | Count Backend | Packed Count ms | Unpack+Sum ms | "
        "Per-Neuron Count ms | Unpack+Per-Neuron Sum ms | Count OK | Per-Neuron OK | "
        "Round Trip | Triton Round Trip |"
    )
    print("|---:|---:|---:|---:|---:|---:|---|---:|---:|---|---:|---:|---:|---:|---|---|---|---|")
    for result in results:
        print(
            f"| {result.timesteps} | {result.batch} | {result.neurons} | "
            f"{format_memory(result.dense_bytes)} | "
            f"{format_memory(result.packed_bytes)} | "
            f"{result.compression_ratio:.2f}x | "
            f"`{result.packed_shape}` | "
            f"{format_ms(result.torch_pack_seconds)} | "
            f"{format_ms(result.triton_pack_seconds)} | "
            f"{result.packed_count_backend} | "
            f"{format_ms(result.packed_count_seconds)} | "
            f"{format_ms(result.unpack_sum_seconds)} | "
            f"{format_ms(result.per_neuron_count_seconds)} | "
            f"{format_ms(result.unpack_per_neuron_sum_seconds)} | "
            f"{result.spike_count_ok} | "
            f"{result.per_neuron_count_ok} | "
            f"{result.round_trip_ok} | "
            f"{'' if result.triton_round_trip_ok is None else result.triton_round_trip_ok} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rate", type=float, default=0.05)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--shape", action="append", type=parse_shape, default=[])
    args = parser.parse_args()

    if not 0.0 <= args.rate <= 1.0:
        raise ValueError("rate must be between 0 and 1")
    if args.warmup < 0 or args.repeats <= 0:
        raise ValueError("warmup must be non-negative and repeats must be positive")
    shapes = args.shape or DEFAULT_SHAPES
    results = [
        run_one(t, b, n, args.device, args.rate, warmup=args.warmup, repeats=args.repeats)
        for t, b, n in shapes
    ]
    print_markdown(args, results)


if __name__ == "__main__":
    main()
