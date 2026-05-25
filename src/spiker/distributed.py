"""Distributed helpers for packed spike tensors."""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
from torch import nn

from spiker.packing import (
    CountDim,
    PackedSpikes,
    _normalize_count_dims,
    _validate_packed_spikes,
    packed_spike_count,
)


def distributed_available_and_initialized() -> bool:
    """Return whether ``torch.distributed`` can run collectives now."""

    return dist.is_available() and dist.is_initialized()


def fsdp_available() -> bool:
    """Return whether PyTorch FSDP can be imported in this environment."""

    try:
        _fsdp_cls()
    except (ImportError, AttributeError):
        return False
    return True


def wrap_fsdp_if_initialized(
    module: nn.Module,
    *,
    group: Any | None = None,
    require_initialized: bool = False,
    **fsdp_kwargs: Any,
) -> nn.Module:
    """Wrap ``module`` with FSDP when distributed is initialized.

    In single-process runs this returns the original module by default. Set
    ``require_initialized=True`` when silently falling back would hide a launch
    configuration error.
    """

    if not distributed_available_and_initialized():
        if require_initialized:
            _require_distributed_initialized()
        return module

    fsdp_cls = _fsdp_cls()
    kwargs = dict(fsdp_kwargs)
    kwargs.setdefault("use_orig_params", True)
    if group is not None:
        kwargs.setdefault("process_group", group)
    return fsdp_cls(module, **kwargs)


def packed_spike_all_gather(
    packed: PackedSpikes,
    *,
    group: Any | None = None,
) -> list[PackedSpikes]:
    """All-gather packed spike payloads across ranks.

    The returned list has one ``PackedSpikes`` object per rank. All ranks are
    expected to contribute the same packed shape and original dense shape.
    """

    _validate_packed_spikes(packed)
    _require_distributed_initialized()
    world_size = dist.get_world_size(group=group)
    local_data = packed.data.contiguous()
    gathered = [torch.empty_like(local_data) for _ in range(world_size)]
    dist.all_gather(gathered, local_data, group=group)
    return [PackedSpikes(data=data, original_shape=packed.original_shape) for data in gathered]


def packed_spike_count_all_reduce(
    packed: PackedSpikes,
    dim: CountDim = None,
    *,
    group: Any | None = None,
) -> torch.Tensor:
    """Return packed spike counts summed across all ranks."""

    _require_distributed_initialized()
    counts = _packed_spike_count_for_collective(packed, dim).contiguous()
    dist.all_reduce(counts, op=dist.ReduceOp.SUM, group=group)
    return counts


def packed_spike_rate_all_reduce(
    packed: PackedSpikes,
    dim: CountDim = None,
    *,
    group: Any | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return packed spike rates averaged across equally shaped local ranks."""

    counts = packed_spike_count_all_reduce(packed, dim=dim, group=group)
    world_size = dist.get_world_size(group=group)
    denominator = _count_denominator(packed, dim) * world_size
    return counts.to(dtype) / denominator


def _require_distributed_initialized() -> None:
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available")
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed is not initialized")


def _fsdp_cls() -> type[nn.Module]:
    from torch.distributed.fsdp import FullyShardedDataParallel

    return FullyShardedDataParallel


def _count_denominator(packed: PackedSpikes, dim: CountDim) -> int:
    if dim is None:
        return packed.original_numel

    dims = _normalize_count_dims(dim, ndim=len(packed.original_shape))
    denominator = 1
    for index in dims:
        denominator *= packed.original_shape[index]
    return denominator


def _packed_spike_count_for_collective(packed: PackedSpikes, dim: CountDim) -> torch.Tensor:
    return packed_spike_count(packed, dim=dim)
