"""Bitpacked spike tensor utilities."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import TypeAlias

import torch

from spiker._optional import has_triton

BITS_PER_WORD = 32
CountDim: TypeAlias = int | Sequence[int] | None


@dataclass(frozen=True)
class PackedSpikes:
    """Spike tensor packed into signed int32 words along the last dimension."""

    data: torch.Tensor
    original_shape: tuple[int, ...]

    @property
    def packed_shape(self) -> tuple[int, ...]:
        return tuple(self.data.shape)

    @property
    def original_numel(self) -> int:
        numel = 1
        for size in self.original_shape:
            numel *= size
        return numel


def packed_last_dim_size(size: int) -> int:
    """Return the number of int32 words required to pack a last dimension."""

    if size < 0:
        raise ValueError("size must be non-negative")
    return (size + BITS_PER_WORD - 1) // BITS_PER_WORD


def pack_spikes(spikes: torch.Tensor) -> PackedSpikes:
    """Pack binary spikes into int32 words along the last dimension.

    Inputs may be bool, integer, or floating tensors. Nonzero values are treated
    as spikes. The returned tensor has shape ``spikes.shape[:-1] +
    (ceil(N / 32),)`` and dtype ``torch.int32``.
    """

    if spikes.ndim == 0:
        raise ValueError("spikes must have at least one dimension")
    if spikes.shape[-1] == 0:
        raise ValueError("cannot pack an empty last dimension")

    original_shape = tuple(spikes.shape)
    packed_n = packed_last_dim_size(spikes.shape[-1])
    padded_n = packed_n * BITS_PER_WORD
    binary = (spikes != 0).to(torch.int64)
    if padded_n != spikes.shape[-1]:
        pad_shape = (*spikes.shape[:-1], padded_n - spikes.shape[-1])
        padding = torch.zeros(pad_shape, dtype=torch.int64, device=spikes.device)
        binary = torch.cat((binary, padding), dim=-1)

    words = binary.reshape(*spikes.shape[:-1], packed_n, BITS_PER_WORD)
    bit_offsets = torch.arange(BITS_PER_WORD, dtype=torch.int64, device=spikes.device)
    weights = (1 << bit_offsets).reshape((1,) * (words.ndim - 1) + (BITS_PER_WORD,))
    packed = torch.sum(words * weights, dim=-1).to(torch.int32)
    return PackedSpikes(data=packed, original_shape=original_shape)


def unpack_spikes(packed: PackedSpikes, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Unpack a ``PackedSpikes`` object back to its original shape."""

    _validate_packed_spikes(packed)
    bit_offsets = torch.arange(BITS_PER_WORD, dtype=torch.int64, device=packed.data.device)
    unpacked = ((packed.data.to(torch.int64).unsqueeze(-1) >> bit_offsets) & 1).to(dtype)
    padded = unpacked.reshape(*packed.original_shape[:-1], -1)
    return padded[..., : packed.original_shape[-1]]


def packed_spike_count(
    packed: PackedSpikes,
    dim: CountDim = None,
) -> torch.Tensor:
    """Return active spike counts with dense-like reduction semantics.

    ``dim=None`` returns the total count. When ``dim`` includes the original
    last dimension, that reduction is performed from packed words directly and
    returns counts over the remaining original dimensions. Reductions that keep
    the original last dimension require unpacking and are supported as a
    convenience path.
    """

    if dim is None:
        return packed_spike_counts(packed, dim=None)

    dims = _normalize_count_dims(dim, ndim=len(packed.original_shape))
    if len(packed.original_shape) - 1 not in dims:
        if dims == tuple(range(len(packed.original_shape) - 1)):
            return _packed_spike_counts_over_leading_dims(packed)
        dense = unpack_spikes(packed, dtype=torch.float32)
        return dense.sum(dim=tuple(dims)).to(torch.int64)

    return packed_spike_counts(packed, dim=dims)


def packed_spike_counts(
    packed: PackedSpikes,
    dim: CountDim = -1,
) -> torch.Tensor:
    """Return active spike counts from packed words.

    ``dim`` must include the original last dimension unless it is ``None``.
    This keeps the helper on the fast packed path: for `[T, B, N]` spikes,
    ``dim=-1`` returns `[T, B]`, ``dim=(0, -1)`` returns `[B]`, and
    ``dim=None`` returns a scalar total.
    """

    if packed.data.is_cuda and has_triton():
        from spiker.triton.packing import packed_spike_counts_triton

        return packed_spike_counts_triton(packed, dim=dim)

    row_counts = _packed_spike_row_counts(packed)
    if dim is None:
        return row_counts.sum()

    dims = _normalize_count_dims(dim, ndim=len(packed.original_shape))
    last_dim = len(packed.original_shape) - 1
    if last_dim not in dims:
        raise ValueError(
            "packed_spike_counts requires reducing the packed/original last dimension; "
            "use packed_spike_count for dense-like reductions that keep that dimension"
        )

    remaining_dims = tuple(index for index in dims if index != last_dim)
    if not remaining_dims:
        return row_counts
    return row_counts.sum(dim=remaining_dims)


def packed_spike_rate(
    packed: PackedSpikes,
    dim: CountDim = None,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return mean spike activity for a packed spike tensor."""

    counts = packed_spike_count(packed, dim=dim)
    dims = (
        tuple(range(len(packed.original_shape)))
        if dim is None
        else _normalize_count_dims(dim, ndim=len(packed.original_shape))
    )
    denominator = 1
    for index in dims:
        denominator *= packed.original_shape[index]
    return counts.to(dtype) / denominator


def _packed_spike_row_counts(packed: PackedSpikes) -> torch.Tensor:
    """Return active spike counts for each row before the packed dimension."""

    _validate_packed_spikes(packed)
    if packed.data.device.type == "cpu":
        return _packed_spike_row_counts_cpu(packed)

    words = packed.data.to(torch.int64) & ((1 << BITS_PER_WORD) - 1)
    remainder = packed.original_shape[-1] % BITS_PER_WORD
    if remainder != 0:
        full_word_mask = (1 << BITS_PER_WORD) - 1
        valid_last_word_mask = (1 << remainder) - 1
        masks = torch.full(
            (words.shape[-1],),
            full_word_mask,
            dtype=torch.int64,
            device=words.device,
        )
        masks[-1] = valid_last_word_mask
        words = words & masks.reshape((1,) * (words.ndim - 1) + (words.shape[-1],))

    bit_offsets = torch.arange(BITS_PER_WORD, dtype=torch.int64, device=words.device)
    bit_counts = ((words.unsqueeze(-1) >> bit_offsets) & 1).sum(dim=-1, dtype=torch.int64)
    return bit_counts.sum(dim=-1)


def _packed_spike_row_counts_cpu(packed: PackedSpikes) -> torch.Tensor:
    words = packed.data.contiguous()
    remainder = packed.original_shape[-1] % BITS_PER_WORD
    if remainder != 0:
        masks = torch.full(
            (words.shape[-1],),
            -1,
            dtype=torch.int32,
            device=words.device,
        )
        masks[-1] = (1 << remainder) - 1
        words = words & masks.reshape((1,) * (words.ndim - 1) + (words.shape[-1],))

    byte_values = words.contiguous().view(torch.uint8).reshape(*words.shape, 4)
    byte_counts = _byte_popcount_lut()[byte_values.to(torch.long)]
    return byte_counts.sum(dim=(-1, -2), dtype=torch.int64)


def _packed_spike_counts_over_leading_dims(packed: PackedSpikes) -> torch.Tensor:
    _validate_packed_spikes(packed)
    words = packed.data.to(torch.int64) & ((1 << BITS_PER_WORD) - 1)
    bit_offsets = torch.arange(BITS_PER_WORD, dtype=torch.int64, device=words.device)
    bits = (words.unsqueeze(-1) >> bit_offsets) & 1
    leading_dims = tuple(range(len(packed.original_shape) - 1))
    counts = bits.sum(dim=leading_dims, dtype=torch.int64) if leading_dims else bits.to(torch.int64)
    return counts.reshape(-1)[: packed.original_shape[-1]]


@lru_cache(maxsize=1)
def _byte_popcount_lut() -> torch.Tensor:
    return torch.tensor([int(value).bit_count() for value in range(256)], dtype=torch.int64)


def _normalize_count_dims(dim: int | Sequence[int], *, ndim: int) -> tuple[int, ...]:
    raw_dims = (dim,) if isinstance(dim, int) else tuple(dim)
    normalized: list[int] = []
    for raw_dim in raw_dims:
        actual_dim = raw_dim + ndim if raw_dim < 0 else raw_dim
        if actual_dim < 0 or actual_dim >= ndim:
            raise IndexError(f"dimension out of range: {raw_dim}")
        if actual_dim in normalized:
            raise ValueError(f"duplicate reduction dimension: {raw_dim}")
        normalized.append(actual_dim)
    return tuple(normalized)


def _validate_packed_spikes(packed: PackedSpikes) -> None:
    if packed.data.dtype != torch.int32:
        raise ValueError("packed.data must have dtype torch.int32")
    if len(packed.original_shape) == 0:
        raise ValueError("packed.original_shape must have at least one dimension")
    if packed.original_shape[-1] <= 0:
        raise ValueError("packed.original_shape must have non-empty last dimension")
    expected_packed_shape = (
        *packed.original_shape[:-1],
        packed_last_dim_size(packed.original_shape[-1]),
    )
    if tuple(packed.data.shape) != expected_packed_shape:
        raise ValueError(
            "packed.data shape does not match packed.original_shape; "
            f"got {tuple(packed.data.shape)} and expected {expected_packed_shape}"
        )


def dense_spike_bytes(spikes: torch.Tensor) -> int:
    """Return storage bytes for a dense spike tensor."""

    return spikes.numel() * spikes.element_size()


def packed_spike_bytes(packed: PackedSpikes) -> int:
    """Return storage bytes for a packed spike tensor."""

    return packed.data.numel() * packed.data.element_size()


def spike_compression_ratio(spikes: torch.Tensor, packed: PackedSpikes) -> float:
    """Return dense bytes divided by packed bytes."""

    packed_bytes = packed_spike_bytes(packed)
    if packed_bytes == 0:
        raise ValueError("packed representation has zero bytes")
    return dense_spike_bytes(spikes) / packed_bytes
