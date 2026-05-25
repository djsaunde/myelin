from __future__ import annotations

import torch

from spiker.neurons import LIFParams
from spiker.workloads import (
    dense_fast_surrogate_lif_spike_loss,
    dense_hard_fast_surrogate_lif_spike_loss,
    dense_lif_loss,
    dense_surrogate_lif_spike_loss,
    dense_triangular_surrogate_lif_spike_loss,
    looped_fast_surrogate_lif_spike_loss,
    looped_hard_fast_surrogate_lif_spike_loss,
    looped_surrogate_lif_spike_loss,
)


def test_dense_lif_loss_backpropagates_to_weight() -> None:
    inputs = torch.rand((4, 2, 3))
    weight = torch.rand((3, 5), requires_grad=True) * 0.02
    weight.retain_grad()

    loss = dense_lif_loss(inputs, weight, LIFParams())
    loss.backward()

    assert loss.ndim == 0
    assert weight.grad is not None
    assert weight.grad.shape == weight.shape


def test_dense_surrogate_lif_spike_loss_backpropagates_to_weight() -> None:
    inputs = torch.rand((4, 2, 3))
    weight = torch.rand((3, 5), requires_grad=True) * 0.02
    weight.retain_grad()

    loss = dense_surrogate_lif_spike_loss(inputs, weight, LIFParams())
    loss.backward()

    assert loss.ndim == 0
    assert weight.grad is not None
    assert weight.grad.shape == weight.shape


def test_surrogate_variants_backpropagate_to_weight() -> None:
    inputs = torch.rand((4, 2, 3))
    params = LIFParams()

    for workload in [
        dense_fast_surrogate_lif_spike_loss,
        dense_hard_fast_surrogate_lif_spike_loss,
        dense_triangular_surrogate_lif_spike_loss,
        looped_fast_surrogate_lif_spike_loss,
        looped_hard_fast_surrogate_lif_spike_loss,
        looped_surrogate_lif_spike_loss,
    ]:
        weight = torch.rand((3, 5), requires_grad=True) * 0.02
        weight.retain_grad()

        loss = workload(inputs, weight, params)
        loss.backward()

        assert loss.ndim == 0
        assert weight.grad is not None
        assert weight.grad.shape == weight.shape


def test_looped_fast_surrogate_matches_materialized_fast_surrogate() -> None:
    torch.manual_seed(0)
    inputs = torch.rand((4, 2, 3))
    materialized_weight = (torch.rand((3, 5)) * 0.02).requires_grad_(True)
    looped_weight = materialized_weight.detach().clone().requires_grad_(True)
    params = LIFParams()

    materialized_loss = dense_fast_surrogate_lif_spike_loss(
        inputs,
        materialized_weight,
        params,
        surrogate_slope=5.0,
    )
    looped_loss = looped_fast_surrogate_lif_spike_loss(
        inputs,
        looped_weight,
        params,
        surrogate_slope=5.0,
    )
    materialized_loss.backward()
    looped_loss.backward()

    assert torch.allclose(looped_loss, materialized_loss)
    assert materialized_weight.grad is not None
    assert looped_weight.grad is not None
    assert torch.allclose(looped_weight.grad, materialized_weight.grad)


def test_looped_hard_fast_surrogate_matches_materialized_hard_fast_surrogate() -> None:
    torch.manual_seed(1)
    inputs = torch.rand((4, 2, 3))
    materialized_weight = (torch.rand((3, 5)) * 0.4).requires_grad_(True)
    looped_weight = materialized_weight.detach().clone().requires_grad_(True)
    params = LIFParams(tau_mem=4.0, threshold=0.2, reset=-0.1)

    materialized_loss = dense_hard_fast_surrogate_lif_spike_loss(
        inputs,
        materialized_weight,
        params,
        surrogate_slope=5.0,
    )
    looped_loss = looped_hard_fast_surrogate_lif_spike_loss(
        inputs,
        looped_weight,
        params,
        surrogate_slope=5.0,
    )
    materialized_loss.backward()
    looped_loss.backward()

    assert torch.allclose(looped_loss, materialized_loss)
    assert materialized_weight.grad is not None
    assert looped_weight.grad is not None
    assert torch.allclose(looped_weight.grad, materialized_weight.grad)
