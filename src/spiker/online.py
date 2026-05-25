"""Online learning reference rules."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import torch

from spiker.codegen import compile_surrogate_derivative_ir
from spiker.dsl import NeuronIR, validate_surrogate_derivative_ir
from spiker.neurons import ALIFParams, ALIFState, LIFParams, LIFState
from spiker.surrogates import SurrogateName, surrogate_derivative

OnlineSurrogate = SurrogateName | NeuronIR


@dataclass(frozen=True)
class OnlineLIFGrad:
    """Result of an online dense-synapse LIF eligibility update."""

    final_state: LIFState
    spikes: torch.Tensor
    grad_weight: torch.Tensor
    grad_bias: torch.Tensor | None


@dataclass(frozen=True)
class OnlineALIFGrad:
    """Result of an online dense-synapse ALIF eligibility update."""

    final_state: ALIFState
    spikes: torch.Tensor
    grad_weight: torch.Tensor
    grad_bias: torch.Tensor | None


def linear_lif_online_eligibility_grad(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    learning_signal: torch.Tensor,
    params: LIFParams,
    *,
    surrogate: OnlineSurrogate = "fast_sigmoid",
    surrogate_slope: float = 5.0,
    surrogate_params: Mapping[str, float] | None = None,
) -> OnlineLIFGrad:
    """Compute a local online eligibility-gradient estimate for dense LIF.

    ``learning_signal[t, b, n]`` is interpreted as ``dL/d spike[t, b, n]``.
    The eligibility trace tracks ``d membrane / d weight`` online:

    ```text
    e[t] = decay * e[t - 1] + input[t]
    grad_w += learning_signal[t] * d_spike_d_membrane[t] * e[t]
    ```

    This is an e-prop/OSTL-style reference path, not full BPTT: it intentionally
    ignores gradients through reset and through future recurrent state.
    """

    _validate_online_inputs(inputs, weight, bias, learning_signal)
    _timesteps, batch, features = inputs.shape
    neurons = weight.shape[1]
    membrane = inputs.new_zeros((batch, neurons))
    eligibility = inputs.new_zeros((batch, features))
    bias_eligibility = inputs.new_zeros((batch,))
    grad_weight = torch.zeros_like(weight)
    grad_bias = torch.zeros_like(bias) if bias is not None else None
    spike_values: list[torch.Tensor] = []
    surrogate_derivative_fn = _resolve_online_surrogate_derivative(surrogate, surrogate_params)

    for input_features, local_signal in zip(
        inputs.unbind(dim=0),
        learning_signal.unbind(dim=0),
        strict=True,
    ):
        current = torch.matmul(input_features, weight)
        if bias is not None:
            current = current + bias
        pre_reset = membrane * params.decay + current
        spike = (pre_reset >= params.threshold).to(inputs.dtype)
        centered = surrogate_slope * (pre_reset - params.threshold)
        d_spike_d_membrane = surrogate_slope * surrogate_derivative_fn(centered)
        local_factor = local_signal * d_spike_d_membrane

        eligibility = eligibility * params.decay + input_features
        bias_eligibility = bias_eligibility * params.decay + 1.0
        grad_weight = grad_weight + torch.einsum("bf,bn->fn", eligibility, local_factor)
        if grad_bias is not None:
            grad_bias = grad_bias + (local_factor * bias_eligibility[:, None]).sum(dim=0)

        membrane = torch.where(spike.bool(), pre_reset.new_tensor(params.reset), pre_reset)
        spike_values.append(spike)

    return OnlineLIFGrad(
        final_state=LIFState(membrane=membrane),
        spikes=torch.stack(spike_values, dim=0),
        grad_weight=grad_weight,
        grad_bias=grad_bias,
    )


def linear_alif_online_eligibility_grad(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    learning_signal: torch.Tensor,
    params: ALIFParams,
    *,
    surrogate: OnlineSurrogate = "fast_sigmoid",
    surrogate_slope: float = 5.0,
    surrogate_params: Mapping[str, float] | None = None,
) -> OnlineALIFGrad:
    """Compute a local online eligibility-gradient estimate for dense ALIF.

    The ALIF trace extends the LIF eligibility rule with an adaptation trace.
    The local spike derivative uses the effective centered membrane
    ``pre_reset - beta * adaptation``. Like the LIF online rule, this is an
    e-prop/OSTL-style estimate and intentionally ignores gradients through the
    hard reset path.
    """

    _validate_online_inputs(inputs, weight, bias, learning_signal)
    _timesteps, batch, features = inputs.shape
    neurons = weight.shape[1]
    membrane = inputs.new_zeros((batch, neurons))
    adaptation = inputs.new_zeros((batch, neurons))
    membrane_eligibility = inputs.new_zeros((batch, features))
    centered_eligibility = inputs.new_zeros((batch, features, neurons))
    previous_d_spike_d_membrane = inputs.new_zeros((batch, neurons))
    bias_membrane_eligibility = inputs.new_zeros((batch,))
    bias_centered_eligibility = inputs.new_zeros((batch, neurons))
    grad_weight = torch.zeros_like(weight)
    grad_bias = torch.zeros_like(bias) if bias is not None else None
    spike_values: list[torch.Tensor] = []
    surrogate_derivative_fn = _resolve_online_surrogate_derivative(surrogate, surrogate_params)

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
        centered = surrogate_slope * (pre_reset - adaptive_threshold)
        d_spike_d_center = surrogate_derivative_fn(centered)
        d_spike_d_membrane = surrogate_slope * d_spike_d_center
        spike = (pre_reset >= adaptive_threshold).to(inputs.dtype)

        previous_membrane_eligibility = membrane_eligibility
        centered_eligibility = (
            previous_membrane_eligibility[:, :, None] * (params.decay - params.adaptation_decay)
            + centered_eligibility
            * (params.adaptation_decay - params.beta * previous_d_spike_d_membrane[:, None, :])
            + input_features[:, :, None]
        )
        membrane_eligibility = previous_membrane_eligibility * params.decay + input_features

        previous_bias_membrane_eligibility = bias_membrane_eligibility
        bias_centered_eligibility = (
            previous_bias_membrane_eligibility[:, None] * (params.decay - params.adaptation_decay)
            + bias_centered_eligibility
            * (params.adaptation_decay - params.beta * previous_d_spike_d_membrane)
            + 1.0
        )
        bias_membrane_eligibility = previous_bias_membrane_eligibility * params.decay + 1.0
        local_factor = local_signal * d_spike_d_membrane
        grad_weight = grad_weight + torch.einsum(
            "bfn,bn->fn",
            centered_eligibility,
            local_factor,
        )
        if grad_bias is not None:
            grad_bias = grad_bias + (local_factor * bias_centered_eligibility).sum(dim=0)

        previous_d_spike_d_membrane = d_spike_d_membrane

        membrane = torch.where(spike.bool(), pre_reset.new_tensor(params.reset), pre_reset)
        adaptation = adaptation * params.adaptation_decay + spike
        spike_values.append(spike)

    return OnlineALIFGrad(
        final_state=ALIFState(membrane=membrane, adaptation=adaptation),
        spikes=torch.stack(spike_values, dim=0),
        grad_weight=grad_weight,
        grad_bias=grad_bias,
    )


def _validate_online_inputs(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    learning_signal: torch.Tensor,
) -> None:
    if inputs.ndim != 3:
        raise ValueError(f"inputs must be shaped [T, B, F]; got {tuple(inputs.shape)}")
    if weight.ndim != 2:
        raise ValueError(f"weight must be shaped [F, N]; got {tuple(weight.shape)}")
    if inputs.shape[2] != weight.shape[0]:
        raise ValueError("inputs.shape[2] must match weight.shape[0]")
    expected_signal_shape = (inputs.shape[0], inputs.shape[1], weight.shape[1])
    if learning_signal.shape != expected_signal_shape:
        raise ValueError(
            "learning_signal must be shaped [T, B, N]; "
            f"got {tuple(learning_signal.shape)} and expected {expected_signal_shape}"
        )
    if weight.device != inputs.device or weight.dtype != inputs.dtype:
        raise ValueError("weight must have the same device and dtype as inputs")
    if learning_signal.device != inputs.device or learning_signal.dtype != inputs.dtype:
        raise ValueError("learning_signal must have the same device and dtype as inputs")
    if bias is not None:
        if bias.shape != (weight.shape[1],):
            raise ValueError("bias must be shaped [N]")
        if bias.device != inputs.device or bias.dtype != inputs.dtype:
            raise ValueError("bias must have the same device and dtype as inputs")


def _resolve_online_surrogate_derivative(
    surrogate: OnlineSurrogate,
    surrogate_params: Mapping[str, float] | None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    if isinstance(surrogate, str):
        if surrogate_params:
            raise ValueError("surrogate_params are only supported for custom surrogate IRs")
        return lambda centered: surrogate_derivative(centered, surrogate)

    validate_surrogate_derivative_ir(surrogate)
    params = {} if surrogate_params is None else dict(surrogate_params)
    missing_params = set(surrogate.params) - set(params)
    if missing_params:
        raise ValueError(f"surrogate_params is missing values: {sorted(missing_params)}")
    compiled = compile_surrogate_derivative_ir(surrogate)
    return lambda centered: compiled(centered, params)
