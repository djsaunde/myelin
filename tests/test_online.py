from __future__ import annotations

import pytest
import torch

from myelin.dsl import SurrogateBuilder
from myelin.neurons import ALIFParams, LIFParams
from myelin.online import linear_alif_online_eligibility_grad, linear_lif_online_eligibility_grad
from myelin.surrogates import fast_sigmoid_surrogate

pytestmark = pytest.mark.extended


def test_online_lif_eligibility_grad_matches_autograd_when_no_resets_occur() -> None:
    torch.manual_seed(0)
    inputs = torch.rand((5, 3, 4)) * 0.2
    weight = ((torch.rand((4, 6)) - 0.5) * 0.02).requires_grad_()
    bias = (torch.rand((6,)) * 0.01).requires_grad_()
    learning_signal = torch.rand((5, 3, 6)) - 0.5
    params = LIFParams(tau_mem=8.0, threshold=10.0, reset=0.0)
    surrogate_slope = 5.0

    result = linear_lif_online_eligibility_grad(
        inputs,
        weight.detach(),
        bias.detach(),
        learning_signal,
        params,
        surrogate="fast_sigmoid",
        surrogate_slope=surrogate_slope,
    )

    membrane = inputs.new_zeros((inputs.shape[1], weight.shape[1]))
    smooth_spikes = []
    for input_features in inputs.unbind(dim=0):
        membrane = membrane * params.decay + torch.matmul(input_features, weight) + bias
        smooth_spikes.append(
            fast_sigmoid_surrogate(surrogate_slope * (membrane - params.threshold))
        )
    loss = (torch.stack(smooth_spikes, dim=0) * learning_signal).sum()
    loss.backward()

    assert torch.equal(result.spikes, torch.zeros_like(result.spikes))
    assert weight.grad is not None
    assert bias.grad is not None
    assert torch.allclose(result.grad_weight, weight.grad, atol=1e-6, rtol=1e-5)
    assert result.grad_bias is not None
    assert torch.allclose(result.grad_bias, bias.grad, atol=1e-6, rtol=1e-5)


def test_online_lif_eligibility_grad_returns_hard_spikes_and_final_state() -> None:
    inputs = torch.ones((3, 2, 4))
    weight = torch.full((4, 5), 0.4)
    bias = torch.zeros(5)
    learning_signal = torch.ones((3, 2, 5))
    params = LIFParams(tau_mem=8.0, threshold=1.0, reset=-0.25)

    result = linear_lif_online_eligibility_grad(inputs, weight, bias, learning_signal, params)

    assert result.spikes.shape == (3, 2, 5)
    assert torch.equal(result.spikes[0], torch.ones((2, 5)))
    assert result.final_state.membrane.shape == (2, 5)
    assert result.grad_weight.shape == weight.shape
    assert result.grad_bias is not None
    assert result.grad_bias.shape == bias.shape


def test_online_lif_eligibility_grad_supports_biasless_weights() -> None:
    inputs = torch.rand((3, 2, 4))
    weight = torch.rand((4, 5))
    learning_signal = torch.rand((3, 2, 5))
    params = LIFParams()

    result = linear_lif_online_eligibility_grad(inputs, weight, None, learning_signal, params)

    assert result.grad_bias is None
    assert result.grad_weight.shape == weight.shape


def test_online_lif_eligibility_grad_accepts_custom_surrogate_ir() -> None:
    torch.manual_seed(7)
    inputs = torch.rand((4, 2, 3))
    weight = (torch.rand((3, 5)) - 0.5) * 0.1
    bias = torch.rand((5,)) * 0.01
    learning_signal = torch.rand((4, 2, 5)) - 0.5
    params = LIFParams(tau_mem=8.0, threshold=0.75, reset=-0.1)
    builder = SurrogateBuilder("wide_fast")
    centered = builder.centered()
    width = builder.param("width")
    custom_surrogate = builder.build(0.5 / (1.0 + (centered / width).abs()).square())

    actual = linear_lif_online_eligibility_grad(
        inputs,
        weight,
        bias,
        learning_signal,
        params,
        surrogate=custom_surrogate,
        surrogate_slope=5.0,
        surrogate_params={"width": 1.0},
    )
    expected = linear_lif_online_eligibility_grad(
        inputs,
        weight,
        bias,
        learning_signal,
        params,
        surrogate="fast_sigmoid",
        surrogate_slope=5.0,
    )

    assert torch.equal(actual.spikes, expected.spikes)
    assert torch.allclose(actual.final_state.membrane, expected.final_state.membrane)
    assert torch.allclose(actual.grad_weight, expected.grad_weight)
    assert actual.grad_bias is not None
    assert expected.grad_bias is not None
    assert torch.allclose(actual.grad_bias, expected.grad_bias)


def test_online_lif_eligibility_grad_rejects_shape_mismatch() -> None:
    inputs = torch.rand((3, 2, 4))
    weight = torch.rand((4, 5))
    learning_signal = torch.rand((3, 2, 6))

    with pytest.raises(ValueError, match="learning_signal"):
        linear_lif_online_eligibility_grad(inputs, weight, None, learning_signal, LIFParams())

    with pytest.raises(ValueError, match="inputs.shape"):
        linear_lif_online_eligibility_grad(
            inputs,
            torch.rand((3, 5)),
            None,
            torch.rand((3, 2, 5)),
            LIFParams(),
        )


def test_online_lif_eligibility_grad_rejects_missing_custom_surrogate_params() -> None:
    inputs = torch.rand((3, 2, 4))
    weight = torch.rand((4, 5))
    learning_signal = torch.rand((3, 2, 5))
    builder = SurrogateBuilder("wide_fast")
    centered = builder.centered()
    custom_surrogate = builder.build(centered.abs())

    with pytest.raises(ValueError, match="surrogate_params"):
        linear_lif_online_eligibility_grad(
            inputs,
            weight,
            None,
            learning_signal,
            LIFParams(),
            surrogate="fast_sigmoid",
            surrogate_params={"unused": 1.0},
        )

    width = builder.param("width")
    parameterized_surrogate = builder.build(centered.abs() / width)
    with pytest.raises(ValueError, match="missing"):
        linear_lif_online_eligibility_grad(
            inputs,
            weight,
            None,
            learning_signal,
            LIFParams(),
            surrogate=parameterized_surrogate,
        )

    result = linear_lif_online_eligibility_grad(
        inputs,
        weight,
        None,
        learning_signal,
        LIFParams(),
        surrogate=custom_surrogate,
    )
    assert result.grad_weight.shape == weight.shape


def test_online_alif_eligibility_grad_matches_lif_when_beta_is_zero() -> None:
    torch.manual_seed(3)
    inputs = torch.rand((5, 3, 4)) * 0.2
    weight = (torch.rand((4, 6)) - 0.5) * 0.02
    bias = torch.rand((6,)) * 0.01
    learning_signal = torch.rand((5, 3, 6)) - 0.5
    lif_params = LIFParams(tau_mem=8.0, threshold=10.0, reset=0.0)
    alif_params = ALIFParams(
        tau_mem=8.0,
        tau_adaptation=5.0,
        threshold=10.0,
        reset=0.0,
        beta=0.0,
    )

    lif = linear_lif_online_eligibility_grad(
        inputs,
        weight,
        bias,
        learning_signal,
        lif_params,
        surrogate="fast_sigmoid",
        surrogate_slope=5.0,
    )
    alif = linear_alif_online_eligibility_grad(
        inputs,
        weight,
        bias,
        learning_signal,
        alif_params,
        surrogate="fast_sigmoid",
        surrogate_slope=5.0,
    )

    assert torch.equal(alif.spikes, lif.spikes)
    assert torch.allclose(alif.final_state.membrane, lif.final_state.membrane)
    assert torch.equal(alif.final_state.adaptation, torch.zeros_like(alif.final_state.adaptation))
    assert torch.allclose(alif.grad_weight, lif.grad_weight)
    assert alif.grad_bias is not None
    assert lif.grad_bias is not None
    assert torch.allclose(alif.grad_bias, lif.grad_bias)


def test_online_alif_eligibility_grad_returns_adaptive_state_and_gradients() -> None:
    inputs = torch.ones((3, 2, 4))
    weight = torch.full((4, 5), 0.4)
    bias = torch.zeros(5)
    learning_signal = torch.ones((3, 2, 5))
    params = ALIFParams(tau_mem=8.0, tau_adaptation=4.0, threshold=1.0, reset=-0.25, beta=0.5)

    result = linear_alif_online_eligibility_grad(inputs, weight, bias, learning_signal, params)

    assert result.spikes.shape == (3, 2, 5)
    assert result.final_state.membrane.shape == (2, 5)
    assert result.final_state.adaptation.shape == (2, 5)
    assert torch.count_nonzero(result.final_state.adaptation) > 0
    assert result.grad_weight.shape == weight.shape
    assert result.grad_bias is not None
    assert result.grad_bias.shape == bias.shape


def test_online_alif_eligibility_grad_matches_dense_adaptation_trace_reference() -> None:
    torch.manual_seed(6)
    inputs = torch.rand((5, 2, 3))
    weight = (torch.rand((3, 4)) - 0.5) * 0.2
    bias = torch.rand((4,)) * 0.01
    learning_signal = torch.rand((5, 2, 4)) - 0.5
    params = ALIFParams(tau_mem=8.0, tau_adaptation=5.0, threshold=0.6, reset=-0.1, beta=0.35)

    actual = linear_alif_online_eligibility_grad(
        inputs,
        weight,
        bias,
        learning_signal,
        params,
        surrogate="fast_sigmoid",
        surrogate_slope=5.0,
    )
    expected = _linear_alif_online_dense_adaptation_reference(
        inputs,
        weight,
        bias,
        learning_signal,
        params,
    )

    assert torch.equal(actual.spikes, expected.spikes)
    assert torch.allclose(actual.final_state.membrane, expected.final_state.membrane)
    assert torch.allclose(actual.final_state.adaptation, expected.final_state.adaptation)
    assert torch.allclose(actual.grad_weight, expected.grad_weight)
    assert actual.grad_bias is not None
    assert expected.grad_bias is not None
    assert torch.allclose(actual.grad_bias, expected.grad_bias)


def test_online_alif_eligibility_grad_supports_biasless_weights() -> None:
    inputs = torch.rand((3, 2, 4))
    weight = torch.rand((4, 5))
    learning_signal = torch.rand((3, 2, 5))
    params = ALIFParams()

    result = linear_alif_online_eligibility_grad(inputs, weight, None, learning_signal, params)

    assert result.grad_bias is None
    assert result.grad_weight.shape == weight.shape


def test_online_alif_eligibility_grad_accepts_custom_surrogate_ir() -> None:
    torch.manual_seed(8)
    inputs = torch.rand((4, 2, 3))
    weight = (torch.rand((3, 5)) - 0.5) * 0.1
    learning_signal = torch.rand((4, 2, 5)) - 0.5
    params = ALIFParams(tau_mem=8.0, tau_adaptation=5.0, threshold=0.75, reset=-0.1, beta=0.25)
    builder = SurrogateBuilder("fast_alias")
    centered = builder.centered()
    custom_surrogate = builder.build(0.5 / (1.0 + centered.abs()).square())

    actual = linear_alif_online_eligibility_grad(
        inputs,
        weight,
        None,
        learning_signal,
        params,
        surrogate=custom_surrogate,
        surrogate_slope=5.0,
    )
    expected = linear_alif_online_eligibility_grad(
        inputs,
        weight,
        None,
        learning_signal,
        params,
        surrogate="fast_sigmoid",
        surrogate_slope=5.0,
    )

    assert torch.equal(actual.spikes, expected.spikes)
    assert torch.allclose(actual.final_state.membrane, expected.final_state.membrane)
    assert torch.allclose(actual.final_state.adaptation, expected.final_state.adaptation)
    assert torch.allclose(actual.grad_weight, expected.grad_weight)
    assert actual.grad_bias is None


def test_linear_online_alif_module_matches_functional_helper() -> None:
    from myelin.modules import LinearOnlineALIF

    torch.manual_seed(4)
    inputs = torch.rand((5, 3, 4))
    learning_signal = torch.rand((5, 3, 6)) - 0.5
    params = ALIFParams(tau_mem=8.0, tau_adaptation=5.0, threshold=0.7, reset=-0.25, beta=0.25)
    module = LinearOnlineALIF(
        4,
        6,
        params,
        surrogate=fast_sigmoid_surrogate,
        surrogate_slope=5.0,
    )

    actual = module(inputs, learning_signal)
    expected = linear_alif_online_eligibility_grad(
        inputs,
        module.synapse.weight,
        module.synapse.bias,
        learning_signal,
        params,
        surrogate="fast_sigmoid",
        surrogate_slope=5.0,
    )

    assert torch.equal(actual.spikes, expected.spikes)
    assert torch.allclose(actual.final_state.membrane, expected.final_state.membrane)
    assert torch.allclose(actual.final_state.adaptation, expected.final_state.adaptation)
    assert torch.allclose(actual.grad_weight, expected.grad_weight)
    assert actual.grad_bias is not None
    assert expected.grad_bias is not None
    assert torch.allclose(actual.grad_bias, expected.grad_bias)


def test_linear_online_alif_step_online_updates_parameters() -> None:
    from myelin.modules import LinearOnlineALIF

    torch.manual_seed(5)
    inputs = torch.rand((4, 2, 3))
    learning_signal = torch.rand((4, 2, 5)) - 0.5
    module = LinearOnlineALIF(3, 5)
    weight_before = module.synapse.weight.detach().clone()
    assert module.synapse.bias is not None
    bias_before = module.synapse.bias.detach().clone()

    result = module.step_online(inputs, learning_signal, lr=0.1)

    assert torch.allclose(module.synapse.weight, weight_before - 0.1 * result.grad_weight)
    assert result.grad_bias is not None
    assert module.synapse.bias is not None
    assert torch.allclose(module.synapse.bias, bias_before - 0.1 * result.grad_bias)
    with pytest.raises(ValueError, match="lr"):
        module.step_online(inputs, learning_signal, lr=-0.1)


def test_linear_online_lif_module_matches_functional_helper() -> None:
    from myelin.modules import LinearOnlineLIF

    torch.manual_seed(1)
    inputs = torch.rand((5, 3, 4))
    learning_signal = torch.rand((5, 3, 6)) - 0.5
    params = LIFParams(tau_mem=8.0, threshold=0.7, reset=-0.25)
    module = LinearOnlineLIF(
        4,
        6,
        params,
        surrogate=fast_sigmoid_surrogate,
        surrogate_slope=5.0,
    )

    actual = module(inputs, learning_signal)
    expected = linear_lif_online_eligibility_grad(
        inputs,
        module.synapse.weight,
        module.synapse.bias,
        learning_signal,
        params,
        surrogate="fast_sigmoid",
        surrogate_slope=5.0,
    )

    assert torch.equal(actual.spikes, expected.spikes)
    assert torch.allclose(actual.final_state.membrane, expected.final_state.membrane)
    assert torch.allclose(actual.grad_weight, expected.grad_weight)
    assert actual.grad_bias is not None
    assert expected.grad_bias is not None
    assert torch.allclose(actual.grad_bias, expected.grad_bias)


def test_linear_online_lif_module_accepts_custom_surrogate_ir() -> None:
    from myelin.modules import LinearOnlineLIF

    torch.manual_seed(9)
    inputs = torch.rand((5, 3, 4))
    learning_signal = torch.rand((5, 3, 6)) - 0.5
    params = LIFParams(tau_mem=8.0, threshold=0.7, reset=-0.25)
    builder = SurrogateBuilder("wide_fast")
    centered = builder.centered()
    width = builder.param("width")
    custom_surrogate = builder.build(0.5 / (1.0 + (centered / width).abs()).square())
    module = LinearOnlineLIF(
        4,
        6,
        params,
        surrogate=custom_surrogate,
        surrogate_slope=5.0,
        surrogate_params={"width": 1.0},
    )

    actual = module(inputs, learning_signal)
    expected = linear_lif_online_eligibility_grad(
        inputs,
        module.synapse.weight,
        module.synapse.bias,
        learning_signal,
        params,
        surrogate=custom_surrogate,
        surrogate_slope=5.0,
        surrogate_params={"width": 1.0},
    )

    assert torch.equal(actual.spikes, expected.spikes)
    assert torch.allclose(actual.grad_weight, expected.grad_weight)
    assert actual.grad_bias is not None
    assert expected.grad_bias is not None
    assert torch.allclose(actual.grad_bias, expected.grad_bias)


def test_linear_online_alif_module_accepts_custom_surrogate_ir() -> None:
    from myelin.modules import LinearOnlineALIF

    torch.manual_seed(10)
    inputs = torch.rand((5, 3, 4))
    learning_signal = torch.rand((5, 3, 6)) - 0.5
    params = ALIFParams(tau_mem=8.0, tau_adaptation=5.0, threshold=0.7, reset=-0.25, beta=0.25)
    builder = SurrogateBuilder("fast_alias")
    centered = builder.centered()
    custom_surrogate = builder.build(0.5 / (1.0 + centered.abs()).square())
    module = LinearOnlineALIF(
        4,
        6,
        params,
        surrogate=custom_surrogate,
        surrogate_slope=5.0,
    )

    actual = module(inputs, learning_signal)
    expected = linear_alif_online_eligibility_grad(
        inputs,
        module.synapse.weight,
        module.synapse.bias,
        learning_signal,
        params,
        surrogate=custom_surrogate,
        surrogate_slope=5.0,
    )

    assert torch.equal(actual.spikes, expected.spikes)
    assert torch.allclose(actual.final_state.adaptation, expected.final_state.adaptation)
    assert torch.allclose(actual.grad_weight, expected.grad_weight)
    assert actual.grad_bias is not None
    assert expected.grad_bias is not None
    assert torch.allclose(actual.grad_bias, expected.grad_bias)


def test_linear_online_lif_step_online_updates_parameters() -> None:
    from myelin.modules import LinearOnlineLIF

    torch.manual_seed(2)
    inputs = torch.rand((4, 2, 3))
    learning_signal = torch.rand((4, 2, 5)) - 0.5
    module = LinearOnlineLIF(3, 5)
    weight_before = module.synapse.weight.detach().clone()
    assert module.synapse.bias is not None
    bias_before = module.synapse.bias.detach().clone()

    result = module.step_online(inputs, learning_signal, lr=0.1)

    assert torch.allclose(module.synapse.weight, weight_before - 0.1 * result.grad_weight)
    assert result.grad_bias is not None
    assert module.synapse.bias is not None
    assert torch.allclose(module.synapse.bias, bias_before - 0.1 * result.grad_bias)


def test_linear_online_lif_supports_biasless_layers_and_rejects_negative_lr() -> None:
    from myelin.modules import LinearOnlineLIF

    inputs = torch.rand((4, 2, 3))
    learning_signal = torch.rand((4, 2, 5))
    module = LinearOnlineLIF(3, 5, bias=False)

    result = module(inputs, learning_signal)

    assert module.synapse.bias is None
    assert result.grad_bias is None
    with pytest.raises(ValueError, match="lr"):
        module.step_online(inputs, learning_signal, lr=-0.1)


def test_online_helpers_are_exported() -> None:
    import myelin

    assert myelin.OnlineALIFGrad
    assert myelin.OnlineLIFGrad
    assert myelin.LinearOnlineALIF
    assert myelin.LinearOnlineLIF
    assert myelin.linear_alif_online_eligibility_grad
    assert myelin.linear_lif_online_eligibility_grad


def _linear_alif_online_dense_adaptation_reference(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    learning_signal: torch.Tensor,
    params: ALIFParams,
):
    from myelin.neurons import ALIFState
    from myelin.online import OnlineALIFGrad
    from myelin.surrogates import surrogate_derivative

    _timesteps, batch, features = inputs.shape
    neurons = weight.shape[1]
    membrane = inputs.new_zeros((batch, neurons))
    adaptation = inputs.new_zeros((batch, neurons))
    membrane_eligibility = inputs.new_zeros((batch, features))
    adaptation_eligibility = inputs.new_zeros((batch, features, neurons))
    bias_membrane_eligibility = inputs.new_zeros((batch,))
    bias_adaptation_eligibility = inputs.new_zeros((batch, neurons))
    grad_weight = torch.zeros_like(weight)
    grad_bias = torch.zeros_like(bias) if bias is not None else None
    spike_values: list[torch.Tensor] = []

    for input_features, local_signal in zip(
        inputs.unbind(dim=0),
        learning_signal.unbind(dim=0),
        strict=True,
    ):
        current = torch.matmul(input_features, weight)
        if bias is not None:
            current = current + bias
        pre_reset = membrane * params.decay + current
        adaptive_threshold = params.threshold + params.beta * adaptation
        centered = 5.0 * (pre_reset - adaptive_threshold)
        d_spike_d_membrane = 5.0 * surrogate_derivative(centered, "fast_sigmoid")
        spike = (pre_reset >= adaptive_threshold).to(inputs.dtype)

        membrane_eligibility = membrane_eligibility * params.decay + input_features
        bias_membrane_eligibility = bias_membrane_eligibility * params.decay + 1.0
        centered_eligibility = (
            membrane_eligibility[:, :, None] - params.beta * adaptation_eligibility
        )
        bias_centered_eligibility = (
            bias_membrane_eligibility[:, None] - params.beta * bias_adaptation_eligibility
        )
        local_factor = local_signal * d_spike_d_membrane
        grad_weight = grad_weight + torch.einsum("bfn,bn->fn", centered_eligibility, local_factor)
        if grad_bias is not None:
            grad_bias = grad_bias + (local_factor * bias_centered_eligibility).sum(dim=0)

        adaptation_eligibility = (
            adaptation_eligibility * params.adaptation_decay
            + d_spike_d_membrane[:, None, :] * centered_eligibility
        )
        bias_adaptation_eligibility = (
            bias_adaptation_eligibility * params.adaptation_decay
            + d_spike_d_membrane * bias_centered_eligibility
        )
        membrane = torch.where(spike.bool(), pre_reset.new_tensor(params.reset), pre_reset)
        adaptation = adaptation * params.adaptation_decay + spike
        spike_values.append(spike)

    return OnlineALIFGrad(
        final_state=ALIFState(membrane=membrane, adaptation=adaptation),
        spikes=torch.stack(spike_values, dim=0),
        grad_weight=grad_weight,
        grad_bias=grad_bias,
    )
