from __future__ import annotations

import pytest
import torch

from spiker._optional import has_triton
from spiker.neurons import LIFParams

pytestmark = pytest.mark.extended


@pytest.mark.skipif(
    not torch.cuda.is_available() or not has_triton(),
    reason="compile-visible Triton op requires CUDA and Triton",
)
def test_compile_visible_rate_forward_matches_existing_triton_forward() -> None:
    from spiker.triton import (
        linear_lif_checkpoint_rate_forward_no_bias_op,
        linear_surrogate_lif_checkpoint_rate_forward,
    )

    inputs = torch.rand((8, 4, 16), device="cuda")
    weight = (torch.rand((16, 32), device="cuda") - 0.5) * 0.02
    params = LIFParams()

    expected_state, expected_rates, expected_chunks = linear_surrogate_lif_checkpoint_rate_forward(
        inputs,
        weight,
        None,
        params,
        checkpoint_size=4,
    )
    final_membrane, rates, chunks = linear_lif_checkpoint_rate_forward_no_bias_op(
        inputs,
        weight,
        params.decay,
        params.threshold,
        params.reset,
        5.0,
        4,
        16,
        32,
        32,
    )

    assert torch.allclose(final_membrane, expected_state.membrane)
    assert torch.allclose(rates, expected_rates)
    assert torch.allclose(chunks, expected_chunks)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not has_triton(),
    reason="compile-visible Triton op requires CUDA and Triton",
)
def test_compile_visible_rate_forward_can_be_torch_compiled() -> None:
    from spiker.triton import linear_lif_checkpoint_rate_forward_no_bias_op

    inputs = torch.rand((8, 4, 16), device="cuda")
    weight = (torch.rand((16, 32), device="cuda") - 0.5) * 0.02
    target = torch.full((4, 32), 0.05, device="cuda")
    params = LIFParams()

    def loss_fn() -> torch.Tensor:
        final_membrane, rates, _chunks = linear_lif_checkpoint_rate_forward_no_bias_op(
            inputs,
            weight,
            params.decay,
            params.threshold,
            params.reset,
            5.0,
            4,
            16,
            32,
            32,
        )
        return (rates - target).pow(2).mean() + 0.01 * final_membrane.pow(2).mean()

    expected = loss_fn()
    compiled = torch.compile(loss_fn, mode="reduce-overhead", fullgraph=True)
    actual = compiled()
    assert torch.allclose(actual, expected)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not has_triton(),
    reason="compile-visible Triton op requires CUDA and Triton",
)
def test_compile_visible_rate_forward_backward_matches_existing_triton_rate_path() -> None:
    from spiker.kernels import linear_surrogate_lif_rate_forward
    from spiker.triton import linear_lif_checkpoint_rate_forward_no_bias_op

    inputs = torch.rand((8, 4, 16), device="cuda")
    weight = ((torch.rand((16, 32), device="cuda") - 0.5) * 0.02).requires_grad_(True)
    reference_weight = weight.detach().clone().requires_grad_(True)
    target = torch.full((4, 32), 0.05, device="cuda")
    params = LIFParams()

    final_membrane, rates, _chunks = linear_lif_checkpoint_rate_forward_no_bias_op(
        inputs,
        weight,
        params.decay,
        params.threshold,
        params.reset,
        5.0,
        4,
        16,
        32,
        32,
    )
    loss = (rates - target).pow(2).mean() + 0.01 * final_membrane.pow(2).mean()
    loss.backward()

    reference_state, reference_rates = linear_surrogate_lif_rate_forward(
        inputs,
        reference_weight,
        None,
        params,
        surrogate="fast_sigmoid",
        surrogate_slope=5.0,
        backend="triton",
        checkpoint_size=4,
        reduction="none",
    )
    reference_loss = (reference_rates - target).pow(2).mean()
    reference_loss = reference_loss + 0.01 * reference_state.membrane.pow(2).mean()
    reference_loss.backward()

    assert weight.grad is not None
    assert reference_weight.grad is not None
    assert torch.allclose(weight.grad, reference_weight.grad)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not has_triton(),
    reason="compile-visible Triton op requires CUDA and Triton",
)
def test_compile_visible_rate_forward_backward_can_be_torch_compiled() -> None:
    from spiker.triton import linear_lif_checkpoint_rate_forward_no_bias_op

    inputs = torch.rand((8, 4, 16), device="cuda")
    weight = ((torch.rand((16, 32), device="cuda") - 0.5) * 0.02).requires_grad_(True)
    reference_weight = weight.detach().clone().requires_grad_(True)
    target = torch.full((4, 32), 0.05, device="cuda")
    params = LIFParams()

    def loss_fn(active_weight: torch.Tensor) -> torch.Tensor:
        final_membrane, rates, _chunks = linear_lif_checkpoint_rate_forward_no_bias_op(
            inputs,
            active_weight,
            params.decay,
            params.threshold,
            params.reset,
            5.0,
            4,
            16,
            32,
            32,
        )
        return (rates - target).pow(2).mean() + 0.01 * final_membrane.pow(2).mean()

    reference_loss = loss_fn(reference_weight)
    reference_loss.backward()

    compiled_loss_fn = torch.compile(loss_fn, mode="reduce-overhead", fullgraph=True)
    loss = compiled_loss_fn(weight)
    loss.backward()

    assert weight.grad is not None
    assert reference_weight.grad is not None
    assert torch.allclose(weight.grad, reference_weight.grad)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not has_triton(),
    reason="compile-visible Triton op requires CUDA and Triton",
)
def test_compile_visible_rate_forward_bias_backward_can_be_torch_compiled() -> None:
    from spiker.triton import linear_lif_checkpoint_rate_forward_bias_op

    inputs = torch.rand((8, 4, 16), device="cuda")
    weight = ((torch.rand((16, 32), device="cuda") - 0.5) * 0.02).requires_grad_(True)
    bias = ((torch.rand((32,), device="cuda") - 0.5) * 0.01).requires_grad_(True)
    reference_weight = weight.detach().clone().requires_grad_(True)
    reference_bias = bias.detach().clone().requires_grad_(True)
    target = torch.full((4, 32), 0.05, device="cuda")
    params = LIFParams()

    def loss_fn(active_weight: torch.Tensor, active_bias: torch.Tensor) -> torch.Tensor:
        final_membrane, rates, _chunks = linear_lif_checkpoint_rate_forward_bias_op(
            inputs,
            active_weight,
            active_bias,
            params.decay,
            params.threshold,
            params.reset,
            5.0,
            4,
            16,
            32,
            32,
        )
        return (rates - target).pow(2).mean() + 0.01 * final_membrane.pow(2).mean()

    reference_loss = loss_fn(reference_weight, reference_bias)
    reference_loss.backward()

    compiled_loss_fn = torch.compile(loss_fn, mode="reduce-overhead", fullgraph=True)
    loss = compiled_loss_fn(weight, bias)
    loss.backward()

    assert weight.grad is not None
    assert bias.grad is not None
    assert reference_weight.grad is not None
    assert reference_bias.grad is not None
    assert torch.allclose(weight.grad, reference_weight.grad)
    assert torch.allclose(bias.grad, reference_bias.grad)
