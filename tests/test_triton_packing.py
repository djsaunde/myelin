from __future__ import annotations

import pytest
import torch

from spiker._optional import has_triton
from spiker.packing import (
    PackedSpikes,
    pack_spikes,
    packed_spike_count,
    packed_spike_counts,
    packed_spike_rate,
    unpack_spikes,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not has_triton(),
    reason="Triton packing tests require CUDA and Triton",
)


@pytest.mark.parametrize("last_dim", [1, 31, 32, 33, 65])
def test_triton_pack_spikes_matches_torch_pack(last_dim: int) -> None:
    from spiker.triton import pack_spikes_triton

    torch.manual_seed(last_dim)
    spikes = (torch.rand((5, 3, last_dim), device="cuda") > 0.7).to(torch.float32)

    expected = pack_spikes(spikes)
    actual = pack_spikes_triton(spikes, block_words=16)

    assert actual.original_shape == tuple(spikes.shape)
    assert actual.data.dtype == torch.int32
    assert torch.equal(actual.data, expected.data)


@pytest.mark.parametrize("last_dim", [1, 31, 32, 33, 65])
def test_triton_unpack_spikes_matches_torch_unpack(last_dim: int) -> None:
    from spiker.triton import pack_spikes_triton, unpack_spikes_triton

    torch.manual_seed(100 + last_dim)
    spikes = (torch.rand((5, 3, last_dim), device="cuda") > 0.7).to(torch.float32)
    packed = pack_spikes_triton(spikes, block_words=16)

    expected = unpack_spikes(packed, dtype=torch.float32)
    actual = unpack_spikes_triton(packed, dtype=torch.float32, block_size=64)

    assert torch.equal(actual, expected)
    assert torch.equal(actual, spikes)


def test_triton_pack_spikes_handles_high_bit() -> None:
    from spiker.triton import pack_spikes_triton, unpack_spikes_triton

    spikes = torch.zeros((1, 32), device="cuda")
    spikes[0, 31] = 1.0

    packed = pack_spikes_triton(spikes, block_words=16)
    unpacked = unpack_spikes_triton(packed)

    assert packed.data.item() == -(1 << 31)
    assert torch.equal(unpacked, spikes)


@pytest.mark.parametrize("last_dim", [1, 31, 32, 33, 2048])
def test_triton_packed_spike_counts_match_torch_counts(last_dim: int) -> None:
    from spiker.triton import pack_spikes_triton, packed_spike_counts_triton

    torch.manual_seed(200 + last_dim)
    spikes = (torch.rand((5, 3, last_dim), device="cuda") > 0.7).to(torch.float32)
    packed = pack_spikes_triton(spikes, block_words=128)

    assert torch.equal(packed_spike_counts_triton(packed), spikes.sum(dim=-1).to(torch.int64))
    assert torch.equal(
        packed_spike_counts_triton(packed, dim=None),
        spikes.sum().to(torch.int64),
    )
    assert torch.equal(
        packed_spike_counts_triton(packed, dim=(0, -1)),
        spikes.sum(dim=(0, 2)).to(torch.int64),
    )


def test_public_packed_spike_counts_and_rates_match_dense_on_cuda() -> None:
    from spiker.triton import pack_spikes_triton

    torch.manual_seed(266)
    spikes = (torch.rand((5, 3, 65), device="cuda") > 0.7).to(torch.float32)
    packed = pack_spikes_triton(spikes, block_words=128)

    assert torch.equal(packed_spike_counts(packed), spikes.sum(dim=-1).to(torch.int64))
    assert torch.equal(packed_spike_counts(packed, dim=None), spikes.sum().to(torch.int64))
    assert torch.equal(
        packed_spike_count(packed, dim=(0, -1)), spikes.sum(dim=(0, 2)).to(torch.int64)
    )
    assert torch.equal(
        packed_spike_count(packed, dim=(0, 1)),
        spikes.sum(dim=(0, 1)).to(torch.int64),
    )
    assert torch.allclose(packed_spike_rate(packed), spikes.mean())
    assert torch.allclose(packed_spike_rate(packed, dim=-1), spikes.mean(dim=-1))
    assert torch.allclose(packed_spike_rate(packed, dim=(0, 1)), spikes.mean(dim=(0, 1)))


def test_triton_packed_spike_counts_rejects_dims_that_keep_packed_axis() -> None:
    from spiker.triton import pack_spikes_triton, packed_spike_counts_triton

    spikes = torch.zeros((2, 3, 35), device="cuda")
    packed = pack_spikes_triton(spikes)

    with pytest.raises(ValueError, match="requires reducing"):
        packed_spike_counts_triton(packed, dim=0)


def test_triton_unpack_rejects_shape_mismatch() -> None:
    from spiker.triton import unpack_spikes_triton

    packed = PackedSpikes(
        data=torch.zeros((2, 2), dtype=torch.int32, device="cuda"),
        original_shape=(2, 65),
    )

    with pytest.raises(ValueError, match="shape"):
        unpack_spikes_triton(packed)
