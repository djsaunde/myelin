from __future__ import annotations

import pytest
import torch

from myelin.packing import (
    BITS_PER_WORD,
    PackedSpikes,
    dense_spike_bytes,
    pack_spikes,
    packed_last_dim_size,
    packed_spike_bytes,
    packed_spike_count,
    packed_spike_counts,
    packed_spike_rate,
    spike_compression_ratio,
    unpack_spikes,
)


def test_lif_forward_packed_spikes_torch_backend_matches_dense_reference() -> None:
    from myelin.functional import lif_unroll
    from myelin.kernels import lif_forward_packed_spikes
    from myelin.neurons import LIFParams, LIFState

    torch.manual_seed(0)
    inputs = torch.rand((5, 2, 35))
    initial = LIFState(membrane=torch.rand((2, 35)) * 0.2)
    params = LIFParams(tau_mem=10.0, threshold=1.0, reset=-0.25)

    expected_state, expected_spikes = lif_unroll(inputs, initial, params)
    actual_state, packed_spikes = lif_forward_packed_spikes(
        inputs,
        initial,
        params,
        backend="torch",
    )

    assert torch.allclose(actual_state.membrane, expected_state.membrane)
    assert torch.equal(unpack_spikes(packed_spikes, dtype=inputs.dtype), expected_spikes)


@pytest.mark.parametrize("last_dim", [1, 31, 32, 33, 65])
def test_pack_unpack_spikes_round_trips_float_inputs(last_dim: int) -> None:
    torch.manual_seed(last_dim)
    spikes = (torch.rand((3, 2, last_dim)) > 0.7).to(torch.float32)

    packed = pack_spikes(spikes)
    unpacked = unpack_spikes(packed, dtype=spikes.dtype)

    assert packed.data.dtype == torch.int32
    assert packed.original_shape == tuple(spikes.shape)
    assert packed.data.shape == (3, 2, packed_last_dim_size(last_dim))
    assert torch.equal(unpacked, spikes)


def test_pack_spikes_treats_nonzero_values_as_spikes() -> None:
    spikes = torch.tensor([[[0.0, 2.0, -3.0, 0.0]]])

    packed = pack_spikes(spikes)
    unpacked = unpack_spikes(packed, dtype=torch.float32)

    assert torch.equal(unpacked, torch.tensor([[[0.0, 1.0, 1.0, 0.0]]]))


def test_pack_spikes_sets_high_bit() -> None:
    spikes = torch.zeros((1, BITS_PER_WORD))
    spikes[0, -1] = 1.0

    packed = pack_spikes(spikes)
    unpacked = unpack_spikes(packed)

    assert packed.data.item() == -(1 << 31)
    assert torch.equal(unpacked, spikes)


def test_pack_spikes_reports_memory_savings() -> None:
    spikes = torch.zeros((100, 64, 2048), dtype=torch.float32)
    packed = pack_spikes(spikes)

    assert dense_spike_bytes(spikes) == 100 * 64 * 2048 * 4
    assert packed_spike_bytes(packed) == 100 * 64 * 64 * 4
    assert spike_compression_ratio(spikes, packed) == 32.0


def test_packed_spike_count_and_rate_match_dense_spikes() -> None:
    spikes = torch.zeros((3, 2, 35), dtype=torch.float32)
    spikes[0, 0, 0] = 1.0
    spikes[0, 0, 31] = 1.0
    spikes[0, 0, 34] = 1.0
    spikes[2, 1, 10] = 1.0
    packed = pack_spikes(spikes)

    count = packed_spike_count(packed)
    rate = packed_spike_rate(packed)

    assert count.dtype == torch.int64
    assert count.device == packed.data.device
    assert count.item() == int(spikes.sum().item())
    assert torch.allclose(rate, spikes.mean())


def test_packed_spike_count_supports_dense_like_dim_reductions() -> None:
    spikes = torch.zeros((3, 2, 35), dtype=torch.float32)
    spikes[0, 0, 0] = 1.0
    spikes[0, 1, 31] = 1.0
    spikes[1, 0, 34] = 1.0
    spikes[2, 1, 10] = 1.0
    packed = pack_spikes(spikes)

    assert torch.equal(packed_spike_count(packed, dim=0), spikes.sum(dim=0).to(torch.int64))
    assert torch.equal(packed_spike_count(packed, dim=1), spikes.sum(dim=1).to(torch.int64))
    assert torch.equal(packed_spike_count(packed, dim=-1), spikes.sum(dim=-1).to(torch.int64))
    assert torch.equal(
        packed_spike_count(packed, dim=(0, 1)),
        spikes.sum(dim=(0, 1)).to(torch.int64),
    )
    assert torch.equal(
        packed_spike_count(packed, dim=(0, -1)),
        spikes.sum(dim=(0, 2)).to(torch.int64),
    )


def test_packed_spike_counts_match_dense_row_counts() -> None:
    spikes = torch.zeros((3, 2, 35), dtype=torch.float32)
    spikes[0, 0, 0] = 1.0
    spikes[0, 0, 31] = 1.0
    spikes[0, 0, 34] = 1.0
    spikes[1, 1, 5] = 1.0
    spikes[2, 0, 10] = 1.0
    spikes[2, 0, 11] = 1.0
    packed = pack_spikes(spikes)

    counts = packed_spike_counts(packed)

    assert counts.dtype == torch.int64
    assert counts.shape == spikes.shape[:-1]
    assert torch.equal(counts, spikes.sum(dim=-1).to(torch.int64))


def test_packed_spike_counts_support_fast_dim_reductions() -> None:
    spikes = torch.zeros((3, 2, 35), dtype=torch.float32)
    spikes[0, 0, 0] = 1.0
    spikes[0, 1, 31] = 1.0
    spikes[1, 0, 34] = 1.0
    spikes[2, 1, 10] = 1.0
    packed = pack_spikes(spikes)

    assert torch.equal(packed_spike_counts(packed, dim=None), spikes.sum().to(torch.int64))
    assert torch.equal(packed_spike_counts(packed, dim=-1), spikes.sum(dim=-1).to(torch.int64))
    assert torch.equal(
        packed_spike_counts(packed, dim=(0, -1)),
        spikes.sum(dim=(0, 2)).to(torch.int64),
    )
    assert torch.equal(
        packed_spike_counts(packed, dim=(1, -1)),
        spikes.sum(dim=(1, 2)).to(torch.int64),
    )


def test_packed_spike_rate_supports_dim_reductions() -> None:
    spikes = torch.zeros((3, 2, 35), dtype=torch.float32)
    spikes[0, 0, 0] = 1.0
    spikes[0, 1, 31] = 1.0
    spikes[1, 0, 34] = 1.0
    spikes[2, 1, 10] = 1.0
    packed = pack_spikes(spikes)

    assert torch.allclose(packed_spike_rate(packed), spikes.mean())
    assert torch.allclose(packed_spike_rate(packed, dim=-1), spikes.mean(dim=-1))
    assert torch.allclose(
        packed_spike_rate(packed, dim=(0, -1)),
        spikes.mean(dim=(0, 2)),
    )


def test_packed_spike_counts_rejects_dims_that_keep_packed_axis() -> None:
    spikes = torch.zeros((3, 2, 35), dtype=torch.float32)
    packed = pack_spikes(spikes)

    with pytest.raises(ValueError, match="last dimension"):
        packed_spike_counts(packed, dim=0)


def test_packed_spike_helpers_reject_duplicate_dims() -> None:
    spikes = torch.zeros((3, 2, 35), dtype=torch.float32)
    packed = pack_spikes(spikes)

    with pytest.raises(ValueError, match="duplicate"):
        packed_spike_count(packed, dim=(0, -3))


def test_packed_spike_count_ignores_padding_bits() -> None:
    packed = PackedSpikes(
        data=torch.tensor([[[-1, -1]]], dtype=torch.int32),
        original_shape=(1, 1, 35),
    )

    assert torch.equal(packed_spike_counts(packed), torch.tensor([[35]]))
    assert packed_spike_count(packed).item() == 35
    assert torch.allclose(packed_spike_rate(packed), torch.tensor(1.0))


def test_unpack_spikes_rejects_shape_mismatch() -> None:
    packed = PackedSpikes(data=torch.zeros((2, 2), dtype=torch.int32), original_shape=(2, 65))

    with pytest.raises(ValueError, match="shape"):
        unpack_spikes(packed)


def test_pack_spikes_rejects_scalar_and_empty_last_dim() -> None:
    with pytest.raises(ValueError, match="at least one dimension"):
        pack_spikes(torch.tensor(1.0))
    with pytest.raises(ValueError, match="empty last dimension"):
        pack_spikes(torch.empty((2, 0)))


def test_packed_helpers_reject_empty_original_last_dim() -> None:
    packed = PackedSpikes(data=torch.zeros((2, 0), dtype=torch.int32), original_shape=(2, 0))

    with pytest.raises(ValueError, match="non-empty last dimension"):
        packed_spike_count(packed)
