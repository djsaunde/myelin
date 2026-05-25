from __future__ import annotations

import multiprocessing
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.distributed as torch_dist

from spiker.distributed import (
    distributed_available_and_initialized,
    fsdp_available,
    packed_spike_all_gather,
    packed_spike_count_all_reduce,
    packed_spike_rate_all_reduce,
    wrap_fsdp_if_initialized,
)
from spiker.packing import PackedSpikes, pack_spikes

pytestmark = pytest.mark.extended


def _run_packed_collective_worker(
    rank: int,
    world_size: int,
    init_method: str,
    queue: multiprocessing.Queue,
) -> None:
    torch_dist.init_process_group(
        backend="gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
    )
    try:
        spikes = torch.zeros((2, 1, 35), dtype=torch.float32)
        spikes[..., rank] = 1.0
        if rank == 1:
            spikes[1, 0, 2] = 1.0
        packed = pack_spikes(spikes)

        gathered = packed_spike_all_gather(packed)
        gathered_dense = [item.data for item in gathered]
        expected_packed = []
        for expected_rank in range(world_size):
            expected_spikes = torch.zeros_like(spikes)
            expected_spikes[..., expected_rank] = 1.0
            if expected_rank == 1:
                expected_spikes[1, 0, 2] = 1.0
            expected_packed.append(pack_spikes(expected_spikes).data)

        counts = packed_spike_count_all_reduce(packed, dim=-1)
        expected_counts = torch.zeros((2, 1), dtype=torch.int64)
        for item in _expected_rank_spikes():
            expected_counts = expected_counts + item.sum(dim=-1).to(torch.int64)
        rates = packed_spike_rate_all_reduce(packed, dim=-1)
        expected_rates = expected_counts.to(torch.float32) / (spikes.shape[-1] * world_size)

        ok = (
            all(
                torch.equal(actual, expected)
                for actual, expected in zip(gathered_dense, expected_packed, strict=True)
            )
            and torch.equal(counts, expected_counts)
            and torch.allclose(rates, expected_rates)
        )
        queue.put((rank, ok, ""))
    except Exception as exc:  # noqa: BLE001 - child process should report failures.
        queue.put((rank, False, f"{type(exc).__name__}: {exc}"))
    finally:
        torch_dist.destroy_process_group()


def _expected_rank_spikes() -> list[torch.Tensor]:
    expected: list[torch.Tensor] = []
    for rank in range(2):
        spikes = torch.zeros((2, 1, 35), dtype=torch.float32)
        spikes[..., rank] = 1.0
        if rank == 1:
            spikes[1, 0, 2] = 1.0
        expected.append(spikes)
    return expected


class FakeReduceOp:
    SUM = "sum"


class FakeDist:
    ReduceOp = FakeReduceOp

    def __init__(self, *, initialized: bool = True, world_size: int = 2) -> None:
        self.initialized = initialized
        self.world_size = world_size
        self.last_reduce_op: object | None = None

    def is_available(self) -> bool:
        return True

    def is_initialized(self) -> bool:
        return self.initialized

    def get_world_size(self, group: Any | None = None) -> int:
        return self.world_size

    def all_gather(
        self,
        output_tensors: list[torch.Tensor],
        input_tensor: torch.Tensor,
        group: Any | None = None,
    ) -> None:
        for rank, output in enumerate(output_tensors):
            output.copy_(input_tensor + rank)

    def all_reduce(
        self,
        tensor: torch.Tensor,
        op: object | None = None,
        group: Any | None = None,
    ) -> None:
        self.last_reduce_op = op
        tensor.mul_(self.world_size)


class FakeFSDP(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, **kwargs: Any) -> None:
        super().__init__()
        self.module = module
        self.kwargs = kwargs


def test_distributed_available_and_initialized_uses_torch_distributed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import spiker.distributed as distributed

    monkeypatch.setattr(distributed, "dist", FakeDist(initialized=True))
    assert distributed_available_and_initialized()

    monkeypatch.setattr(distributed, "dist", FakeDist(initialized=False))
    assert not distributed_available_and_initialized()


def test_fsdp_available_uses_import_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    import spiker.distributed as distributed

    monkeypatch.setattr(distributed, "_fsdp_cls", lambda: FakeFSDP)
    assert fsdp_available()

    def missing_fsdp() -> type[torch.nn.Module]:
        raise ImportError("missing")

    monkeypatch.setattr(distributed, "_fsdp_cls", missing_fsdp)
    assert not fsdp_available()


def test_wrap_fsdp_if_initialized_returns_module_when_not_initialized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import spiker.distributed as distributed

    monkeypatch.setattr(distributed, "dist", FakeDist(initialized=False))
    module = torch.nn.Linear(2, 3)

    assert wrap_fsdp_if_initialized(module) is module
    with pytest.raises(RuntimeError, match="not initialized"):
        wrap_fsdp_if_initialized(module, require_initialized=True)


def test_wrap_fsdp_if_initialized_wraps_with_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import spiker.distributed as distributed

    monkeypatch.setattr(distributed, "dist", FakeDist(initialized=True))
    monkeypatch.setattr(distributed, "_fsdp_cls", lambda: FakeFSDP)
    module = torch.nn.Linear(2, 3)
    group = object()

    wrapped = wrap_fsdp_if_initialized(module, group=group, sync_module_states=False)

    assert isinstance(wrapped, FakeFSDP)
    assert wrapped.module is module
    assert wrapped.kwargs["use_orig_params"] is True
    assert wrapped.kwargs["process_group"] is group
    assert wrapped.kwargs["sync_module_states"] is False


def test_packed_spike_all_gather_preserves_packed_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import spiker.distributed as distributed

    monkeypatch.setattr(distributed, "dist", FakeDist(world_size=2))
    spikes = torch.zeros((2, 3, 35), dtype=torch.float32)
    spikes[..., 0] = 1.0
    packed = pack_spikes(spikes)

    gathered = packed_spike_all_gather(packed)

    assert len(gathered) == 2
    assert gathered[0].original_shape == packed.original_shape
    assert gathered[1].original_shape == packed.original_shape
    assert torch.equal(gathered[0].data, packed.data)
    assert torch.equal(gathered[1].data, packed.data + 1)


def test_packed_spike_count_all_reduce_sums_counts_across_ranks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import spiker.distributed as distributed

    fake_dist = FakeDist(world_size=3)
    monkeypatch.setattr(distributed, "dist", fake_dist)
    spikes = torch.zeros((2, 3, 35), dtype=torch.float32)
    spikes[0, :, 0] = 1.0
    spikes[1, :, 1] = 1.0
    packed = pack_spikes(spikes)

    actual = packed_spike_count_all_reduce(packed, dim=-1)

    assert torch.equal(actual, spikes.sum(dim=-1).to(torch.int64) * 3)
    assert fake_dist.last_reduce_op == FakeReduceOp.SUM


def test_packed_spike_rate_all_reduce_averages_equal_shape_ranks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import spiker.distributed as distributed

    monkeypatch.setattr(distributed, "dist", FakeDist(world_size=4))
    spikes = torch.zeros((2, 3, 35), dtype=torch.float32)
    spikes[..., :7] = 1.0
    packed = pack_spikes(spikes)

    actual = packed_spike_rate_all_reduce(packed, dim=-1)

    assert torch.allclose(actual, spikes.mean(dim=-1))


def test_packed_collectives_require_initialized_distributed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import spiker.distributed as distributed

    monkeypatch.setattr(distributed, "dist", FakeDist(initialized=False))
    packed = PackedSpikes(data=torch.zeros((1, 1), dtype=torch.int32), original_shape=(1, 32))

    with pytest.raises(RuntimeError, match="not initialized"):
        packed_spike_all_gather(packed)


def test_public_distributed_helpers_are_exported() -> None:
    import spiker

    assert spiker.distributed_available_and_initialized
    assert spiker.fsdp_available
    assert spiker.wrap_fsdp_if_initialized
    assert spiker.packed_spike_all_gather
    assert spiker.packed_spike_count_all_reduce
    assert spiker.packed_spike_rate_all_reduce


def test_packed_collectives_work_with_gloo_process_group(tmp_path: Path) -> None:
    if not torch_dist.is_available():
        pytest.skip("torch.distributed is not available")
    if not torch_dist.is_gloo_available():
        pytest.skip("torch.distributed gloo backend is not available")

    world_size = 2
    init_path = tmp_path / "packed_collectives_init"
    init_method = f"file://{init_path}"
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    processes = [
        context.Process(
            target=_run_packed_collective_worker,
            args=(rank, world_size, init_method, queue),
        )
        for rank in range(world_size)
    ]

    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=20)

    failures: list[str] = []
    for process in processes:
        if process.is_alive():
            process.terminate()
            failures.append(f"rank process {process.pid} timed out")
        elif process.exitcode != 0:
            failures.append(f"rank process {process.pid} exited with {process.exitcode}")

    results = [queue.get(timeout=1) for _ in range(queue.qsize())]
    failures.extend(message for _rank, ok, message in results if not ok)
    if len(results) != world_size:
        failures.append(f"expected {world_size} child results, got {len(results)}")

    assert not failures
