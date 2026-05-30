"""Benchmark packed spike collectives with a local process group."""

from __future__ import annotations

import argparse
import multiprocessing
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from queue import Empty
from tempfile import TemporaryDirectory

import torch
import torch.distributed as dist

from myelin.distributed import (
    packed_spike_all_gather,
    packed_spike_count_all_reduce,
    packed_spike_rate_all_reduce,
)
from myelin.packing import (
    PackedSpikes,
    dense_spike_bytes,
    pack_spikes,
    packed_spike_bytes,
    unpack_spikes,
)


@dataclass(frozen=True)
class RankResult:
    rank: int
    device: str
    dense_all_gather_seconds: float
    packed_all_gather_seconds: float
    dense_count_all_reduce_seconds: float
    packed_count_all_reduce_seconds: float
    packed_rate_all_reduce_seconds: float
    dense_payload_bytes: int
    packed_payload_bytes: int
    count_shape: tuple[int, ...]
    rate_shape: tuple[int, ...]
    packed_all_gather_matches_dense: bool
    count_max_error: int
    rate_max_error: float


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    if device.type == "cuda" and dist.get_backend() == "nccl":
        device_index = device.index if device.index is not None else torch.cuda.current_device()
        dist.barrier(device_ids=[device_index])
    else:
        dist.barrier()
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_call(
    fn: Callable[[], object],
    *,
    warmup: int,
    repeats: int,
    device: torch.device,
) -> float:
    for _ in range(warmup):
        fn()
    synchronize(device)
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    synchronize(device)
    return (time.perf_counter() - start) / repeats


def resolve_worker_device(args: argparse.Namespace, rank: int) -> torch.device:
    requested = torch.device(args.device)
    if requested.type != "cuda":
        return requested
    if not torch.cuda.is_available():
        raise RuntimeError("--device cuda requires CUDA")
    device_count = torch.cuda.device_count()
    if device_count == 0:
        raise RuntimeError("--device cuda requires at least one CUDA device")
    device = requested if requested.index is not None else torch.device("cuda", rank % device_count)
    torch.cuda.set_device(device)
    return device


def worker(
    rank: int,
    world_size: int,
    init_method: str,
    args: argparse.Namespace,
    queue: multiprocessing.Queue,
) -> None:
    try:
        device = resolve_worker_device(args, rank)
        dist.init_process_group(
            backend=args.backend,
            init_method=init_method,
            rank=rank,
            world_size=world_size,
        )
        generator = torch.Generator(device=device)
        generator.manual_seed(args.seed + rank)
        spikes = (
            torch.rand(
                (args.timesteps, args.batch, args.neurons),
                generator=generator,
                device=device,
            )
            < args.spike_probability
        ).to(torch.float32)
        packed = pack_spikes(spikes)

        def dense_all_gather() -> list[torch.Tensor]:
            gathered = [torch.empty_like(spikes) for _ in range(world_size)]
            dist.all_gather(gathered, spikes)
            return gathered

        def packed_all_gather() -> list[PackedSpikes]:
            return packed_spike_all_gather(packed)

        def dense_count_all_reduce() -> torch.Tensor:
            counts = spikes.sum(dim=-1).to(torch.int64).contiguous()
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)
            return counts

        def packed_count_all_reduce() -> torch.Tensor:
            return packed_spike_count_all_reduce(packed, dim=-1)

        def packed_rate_all_reduce() -> torch.Tensor:
            return packed_spike_rate_all_reduce(packed, dim=-1)

        dense_seconds = time_call(
            dense_all_gather,
            warmup=args.warmup,
            repeats=args.repeats,
            device=device,
        )
        packed_seconds = time_call(
            packed_all_gather,
            warmup=args.warmup,
            repeats=args.repeats,
            device=device,
        )
        dense_count_seconds = time_call(
            dense_count_all_reduce,
            warmup=args.warmup,
            repeats=args.repeats,
            device=device,
        )
        count = packed_count_all_reduce()
        rate = packed_rate_all_reduce()
        gathered_dense = dense_all_gather()
        gathered_packed = packed_all_gather()
        packed_all_gather_matches_dense = all(
            torch.equal(unpack_spikes(actual_packed, dtype=spikes.dtype), actual_dense)
            for actual_packed, actual_dense in zip(
                gathered_packed,
                gathered_dense,
                strict=True,
            )
        )
        expected_count = spikes.sum(dim=-1).to(torch.int64).contiguous()
        dist.all_reduce(expected_count, op=dist.ReduceOp.SUM)
        expected_rate = expected_count.to(torch.float32) / (args.neurons * world_size)
        count_max_error = int((count - expected_count).abs().max().item())
        rate_max_error = float((rate - expected_rate).abs().max().item())
        count_seconds = time_call(
            packed_count_all_reduce,
            warmup=args.warmup,
            repeats=args.repeats,
            device=device,
        )
        rate_seconds = time_call(
            packed_rate_all_reduce,
            warmup=args.warmup,
            repeats=args.repeats,
            device=device,
        )

        queue.put(
            (
                rank,
                None,
                RankResult(
                    rank=rank,
                    device=str(device),
                    dense_all_gather_seconds=dense_seconds,
                    packed_all_gather_seconds=packed_seconds,
                    dense_count_all_reduce_seconds=dense_count_seconds,
                    packed_count_all_reduce_seconds=count_seconds,
                    packed_rate_all_reduce_seconds=rate_seconds,
                    dense_payload_bytes=dense_spike_bytes(spikes),
                    packed_payload_bytes=packed_spike_bytes(packed),
                    count_shape=tuple(count.shape),
                    rate_shape=tuple(rate.shape),
                    packed_all_gather_matches_dense=packed_all_gather_matches_dense,
                    count_max_error=count_max_error,
                    rate_max_error=rate_max_error,
                ),
            )
        )
    except Exception as exc:  # noqa: BLE001 - child process should report failures.
        queue.put((rank, f"{type(exc).__name__}: {exc}", None))
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def run_benchmark(args: argparse.Namespace) -> list[RankResult]:
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available")
    if args.backend == "gloo" and not dist.is_gloo_available():
        raise RuntimeError("torch.distributed gloo backend is not available")
    if args.backend == "nccl":
        if not dist.is_nccl_available():
            raise RuntimeError("torch.distributed nccl backend is not available")
        if torch.device(args.device).type != "cuda":
            raise RuntimeError("--backend nccl requires --device cuda")
        if torch.cuda.device_count() < args.world_size and torch.device(args.device).index is None:
            raise RuntimeError(
                "--backend nccl with implicit CUDA devices requires at least one device per rank; "
                f"got {torch.cuda.device_count()} devices for world_size={args.world_size}"
            )

    with TemporaryDirectory() as tmpdir:
        init_method = f"file://{Path(tmpdir) / 'distributed_collectives_init'}"
        context = multiprocessing.get_context("spawn")
        queue = context.Queue()
        processes = [
            context.Process(
                target=worker,
                args=(rank, args.world_size, init_method, args, queue),
            )
            for rank in range(args.world_size)
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=args.timeout)

        failures = []
        for process in processes:
            if process.is_alive():
                process.terminate()
                failures.append(f"rank process {process.pid} timed out")
            elif process.exitcode != 0:
                failures.append(f"rank process {process.pid} exited with {process.exitcode}")

        results = []
        for _ in range(args.world_size):
            try:
                rank, error, result = queue.get(timeout=1)
            except Empty:
                failures.append("timed out waiting for a rank result")
                break
            if error is not None:
                failures.append(f"rank {rank}: {error}")
            elif isinstance(result, RankResult):
                results.append(result)
        if len(results) != args.world_size:
            failures.append(f"expected {args.world_size} rank results, got {len(results)}")
        if failures:
            raise RuntimeError("; ".join(failures))
        return sorted(results, key=lambda item: item.rank)


def format_ms(seconds: float) -> str:
    return f"{seconds * 1000.0:.3f}"


def format_mb(bytes_value: int) -> str:
    return f"{bytes_value / 1024 / 1024:.1f}"


def print_markdown(args: argparse.Namespace, results: list[RankResult]) -> None:
    print("# Distributed Packed Collectives Benchmark")
    print()
    print("Local multi-process benchmark using `torch.distributed` and packed spike helpers.")
    print()
    print("## Environment")
    print()
    print(f"- `generated_utc`: `{datetime.now(UTC).isoformat(timespec='seconds')}`")
    print(f"- `backend`: `{args.backend}`")
    print(f"- `device`: `{args.device}`")
    print(f"- `world_size`: `{args.world_size}`")
    print(f"- `shape`: `T={args.timesteps}, B={args.batch}, N={args.neurons}`")
    print(f"- `spike_probability`: `{args.spike_probability}`")
    print(f"- `seed`: `{args.seed}`")
    print(f"- `warmup`: `{args.warmup}`")
    print(f"- `repeats`: `{args.repeats}`")
    print()
    print("## Results")
    print()
    print(
        "| Rank | Dense AllGather ms | Packed AllGather ms | Dense Count AllReduce ms | "
        "Packed Count AllReduce ms | Packed Rate AllReduce ms | Dense Payload MB | "
        "Packed Payload MB | Compression | "
        "Device | Count Shape | Rate Shape | Packed Gather OK | Count Max Error | "
        "Rate Max Error |"
    )
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---:|---:|---:|")
    for result in results:
        compression = result.dense_payload_bytes / result.packed_payload_bytes
        print(
            f"| {result.rank} | "
            f"{format_ms(result.dense_all_gather_seconds)} | "
            f"{format_ms(result.packed_all_gather_seconds)} | "
            f"{format_ms(result.dense_count_all_reduce_seconds)} | "
            f"{format_ms(result.packed_count_all_reduce_seconds)} | "
            f"{format_ms(result.packed_rate_all_reduce_seconds)} | "
            f"{format_mb(result.dense_payload_bytes)} | "
            f"{format_mb(result.packed_payload_bytes)} | "
            f"{compression:.2f}x | "
            f"`{result.device}` | "
            f"`{result.count_shape}` | "
            f"`{result.rate_shape}` | "
            f"{result.packed_all_gather_matches_dense} | "
            f"{result.count_max_error} | "
            f"{result.rate_max_error:.3e} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="gloo")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--neurons", type=int, default=2048)
    parser.add_argument("--spike-probability", type=float, default=0.05)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()
    if args.world_size <= 0:
        raise ValueError("world_size must be positive")
    if args.timesteps <= 0 or args.batch <= 0 or args.neurons <= 0:
        raise ValueError("shape values must be positive")
    if not 0.0 <= args.spike_probability <= 1.0:
        raise ValueError("spike_probability must be in [0, 1]")
    if args.warmup < 0 or args.repeats <= 0:
        raise ValueError("warmup must be non-negative and repeats must be positive")
    print_markdown(args, run_benchmark(args))


if __name__ == "__main__":
    main()
