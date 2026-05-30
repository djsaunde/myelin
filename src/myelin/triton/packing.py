"""Triton kernels for bitpacked spike tensors."""

from __future__ import annotations

import torch

from myelin._optional import require_triton
from myelin.packing import (
    CountDim,
    PackedSpikes,
    _normalize_count_dims,
    _validate_packed_spikes,
    packed_last_dim_size,
)

triton = require_triton()
import triton.language as tl  # noqa: E402


@triton.jit
def _pack_spikes_kernel(
    spikes_ptr,
    packed_ptr,
    rows: tl.constexpr,
    neurons: tl.constexpr,
    packed_neurons: tl.constexpr,
    block_words: tl.constexpr,
):
    program_id = tl.program_id(0)
    word_offsets = program_id * block_words + tl.arange(0, block_words)
    mask_words = word_offsets < rows * packed_neurons
    row_offsets = word_offsets // packed_neurons
    packed_offsets = word_offsets - row_offsets * packed_neurons
    bit_offsets = tl.arange(0, 32)
    neuron_offsets = packed_offsets[:, None] * 32 + bit_offsets[None, :]
    spike_offsets = row_offsets[:, None] * neurons + neuron_offsets
    bit_mask = mask_words[:, None] & (neuron_offsets < neurons)
    values = tl.load(spikes_ptr + spike_offsets, mask=bit_mask, other=0.0) != 0.0
    weights = (1 << bit_offsets).to(tl.int64)
    packed = tl.sum(values.to(tl.int64) * weights[None, :], axis=1).to(tl.int32)
    tl.store(packed_ptr + word_offsets, packed, mask=mask_words)


@triton.jit
def _unpack_spikes_kernel(
    packed_ptr,
    spikes_ptr,
    total_elements: tl.constexpr,
    neurons: tl.constexpr,
    packed_neurons: tl.constexpr,
    block_size: tl.constexpr,
):
    program_id = tl.program_id(0)
    offsets = program_id * block_size + tl.arange(0, block_size)
    mask = offsets < total_elements
    row_offsets = offsets // neurons
    neuron_offsets = offsets - row_offsets * neurons
    packed_offsets = neuron_offsets // 32
    bit_offsets = neuron_offsets - packed_offsets * 32
    packed = tl.load(
        packed_ptr + row_offsets * packed_neurons + packed_offsets,
        mask=mask,
        other=0,
    ).to(tl.int64)
    bits = ((packed >> bit_offsets) & 1).to(tl.float32)
    tl.store(spikes_ptr + offsets, bits, mask=mask)


@triton.jit
def _packed_spike_row_counts_kernel(
    packed_ptr,
    counts_ptr,
    rows: tl.constexpr,
    neurons: tl.constexpr,
    packed_neurons: tl.constexpr,
    block_words: tl.constexpr,
):
    row = tl.program_id(0)
    word_offsets = tl.arange(0, block_words)
    bit_offsets = tl.arange(0, 32)
    word_mask = word_offsets < packed_neurons
    words = tl.load(
        packed_ptr + row * packed_neurons + word_offsets,
        mask=(row < rows) & word_mask,
        other=0,
    ).to(tl.int64)
    neuron_offsets = word_offsets[:, None] * 32 + bit_offsets[None, :]
    bit_mask = word_mask[:, None] & (neuron_offsets < neurons)
    bits = ((words[:, None] >> bit_offsets[None, :]) & 1).to(tl.int64)
    count = tl.sum(tl.where(bit_mask, bits, 0))
    tl.store(counts_ptr + row, count, mask=row < rows)


def _next_power_of_2(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


def pack_spikes_triton(spikes: torch.Tensor, *, block_words: int = 128) -> PackedSpikes:
    """Pack CUDA spikes into int32 words along the last dimension using Triton."""

    if spikes.ndim == 0:
        raise ValueError("spikes must have at least one dimension")
    if spikes.shape[-1] == 0:
        raise ValueError("cannot pack an empty last dimension")
    if not spikes.is_cuda:
        raise ValueError("Triton spike packing requires CUDA inputs")
    if block_words <= 0 or block_words & (block_words - 1):
        raise ValueError("block_words must be a positive power of two")

    contiguous_spikes = spikes.contiguous()
    original_shape = tuple(contiguous_spikes.shape)
    neurons = original_shape[-1]
    packed_neurons = packed_last_dim_size(neurons)
    rows = contiguous_spikes.numel() // neurons
    packed = torch.empty(
        (*original_shape[:-1], packed_neurons),
        dtype=torch.int32,
        device=contiguous_spikes.device,
    )
    grid = (triton.cdiv(rows * packed_neurons, block_words),)
    _pack_spikes_kernel[grid](
        contiguous_spikes,
        packed,
        rows,  # pyright: ignore[reportArgumentType]
        neurons,  # pyright: ignore[reportArgumentType]
        packed_neurons,  # pyright: ignore[reportArgumentType]
        block_words,  # pyright: ignore[reportArgumentType]
    )
    return PackedSpikes(data=packed, original_shape=original_shape)


def unpack_spikes_triton(
    packed: PackedSpikes,
    *,
    dtype: torch.dtype = torch.float32,
    block_size: int = 256,
) -> torch.Tensor:
    """Unpack CUDA int32 spike words using Triton."""

    if packed.data.dtype != torch.int32:
        raise ValueError("packed.data must have dtype torch.int32")
    if not packed.data.is_cuda:
        raise ValueError("Triton spike unpacking requires CUDA inputs")
    if len(packed.original_shape) == 0:
        raise ValueError("packed.original_shape must have at least one dimension")
    if packed.original_shape[-1] == 0:
        raise ValueError("cannot unpack an empty last dimension")
    if block_size <= 0 or block_size & (block_size - 1):
        raise ValueError("block_size must be a positive power of two")

    neurons = packed.original_shape[-1]
    packed_neurons = packed_last_dim_size(neurons)
    expected_shape = (*packed.original_shape[:-1], packed_neurons)
    if tuple(packed.data.shape) != expected_shape:
        raise ValueError(
            "packed.data shape does not match packed.original_shape; "
            f"got {tuple(packed.data.shape)} and expected {expected_shape}"
        )

    if dtype not in {torch.float16, torch.float32, torch.float64, torch.bfloat16}:
        raise ValueError("Triton unpack currently supports floating output dtypes")
    output = torch.empty(packed.original_shape, dtype=dtype, device=packed.data.device)
    total_elements = output.numel()
    grid = (triton.cdiv(total_elements, block_size),)
    _unpack_spikes_kernel[grid](
        packed.data.contiguous(),
        output,
        total_elements,  # pyright: ignore[reportArgumentType]
        neurons,  # pyright: ignore[reportArgumentType]
        packed_neurons,  # pyright: ignore[reportArgumentType]
        block_size,  # pyright: ignore[reportArgumentType]
    )
    return output


def packed_spike_counts_triton(
    packed: PackedSpikes,
    dim: CountDim = -1,
    *,
    block_words: int | None = None,
) -> torch.Tensor:
    """Return packed spike counts using a CUDA Triton row-count kernel."""

    _validate_packed_spikes(packed)
    if not packed.data.is_cuda:
        raise ValueError("Triton spike counting requires CUDA inputs")

    packed_neurons = packed.data.shape[-1]
    actual_block_words = _next_power_of_2(packed_neurons) if block_words is None else block_words
    if actual_block_words < packed_neurons:
        raise ValueError("block_words must be at least the packed last dimension")
    if actual_block_words <= 0 or actual_block_words & (actual_block_words - 1):
        raise ValueError("block_words must be a positive power of two")

    rows = packed.data.numel() // packed_neurons
    row_counts = torch.empty(
        packed.original_shape[:-1],
        dtype=torch.int64,
        device=packed.data.device,
    )
    _packed_spike_row_counts_kernel[(rows,)](
        packed.data.contiguous(),
        row_counts.reshape(-1),
        rows,  # pyright: ignore[reportArgumentType]
        packed.original_shape[-1],  # pyright: ignore[reportArgumentType]
        packed_neurons,  # pyright: ignore[reportArgumentType]
        actual_block_words,  # pyright: ignore[reportArgumentType]
    )

    if dim is None:
        return row_counts.sum()

    dims = _normalize_count_dims(dim, ndim=len(packed.original_shape))
    last_dim = len(packed.original_shape) - 1
    if last_dim not in dims:
        raise ValueError(
            "packed_spike_counts_triton requires reducing the packed/original last dimension"
        )

    remaining_dims = tuple(index for index in dims if index != last_dim)
    if not remaining_dims:
        return row_counts
    return row_counts.sum(dim=remaining_dims)
