"""Small trainable workloads used by benchmarks."""

from __future__ import annotations

import torch

from myelin.functional import lif_unroll
from myelin.losses import spike_rate_loss
from myelin.neurons import LIFParams, LIFState


def dense_lif_loss(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    params: LIFParams,
) -> torch.Tensor:
    """Project dense inputs through a LIF population and return a scalar loss.

    Workload:

    ``[T, B, F] inputs -> [F, N] weight -> [T, B, N] currents -> LIF -> loss``

    The loss uses final membrane energy so gradients flow through the recurrent
    membrane dynamics without requiring a surrogate spike gradient yet.
    """

    currents = torch.matmul(inputs, weight)
    initial_state = LIFState(
        membrane=torch.zeros(
            currents.shape[1:],
            dtype=currents.dtype,
            device=currents.device,
        )
    )
    final_state, _spikes = lif_unroll(currents, initial_state, params)
    return final_state.membrane.square().mean()


def dense_surrogate_lif_spike_loss(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    params: LIFParams,
    *,
    surrogate_slope: float = 10.0,
    target_rate: float = 0.1,
) -> torch.Tensor:
    """Train through a differentiable spike-rate surrogate.

    This keeps the same dense projection as ``dense_lif_loss`` but makes the
    training objective depend on spike-like outputs at every timestep.
    """

    currents = torch.matmul(inputs, weight)
    membrane = torch.zeros(
        currents.shape[1:],
        dtype=currents.dtype,
        device=currents.device,
    )
    spike_sum = membrane.new_zeros(())

    for input_current in currents.unbind(dim=0):
        membrane = membrane * params.decay + input_current
        spike = torch.sigmoid(surrogate_slope * (membrane - params.threshold))
        spike_sum = spike_sum + spike.mean()
        membrane = membrane * (1.0 - spike)

    spike_rate = spike_sum / currents.shape[0]
    return spike_rate_loss(spike_rate, target_rate)


def dense_fast_surrogate_lif_spike_loss(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    params: LIFParams,
    *,
    surrogate_slope: float = 10.0,
    target_rate: float = 0.1,
) -> torch.Tensor:
    """Use a cheaper fast-sigmoid surrogate spike rate loss."""

    currents = torch.matmul(inputs, weight)
    membrane = torch.zeros(
        currents.shape[1:],
        dtype=currents.dtype,
        device=currents.device,
    )
    spike_sum = membrane.new_zeros(())

    for input_current in currents.unbind(dim=0):
        membrane = membrane * params.decay + input_current
        centered = surrogate_slope * (membrane - params.threshold)
        spike = 0.5 * (centered / (1.0 + centered.abs()) + 1.0)
        spike_sum = spike_sum + spike.mean()
        membrane = membrane * (1.0 - spike)

    spike_rate = spike_sum / currents.shape[0]
    return spike_rate_loss(spike_rate, target_rate)


def dense_hard_fast_surrogate_lif_spike_loss(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    params: LIFParams,
    *,
    surrogate_slope: float = 5.0,
    target_rate: float = 0.1,
) -> torch.Tensor:
    """Use hard-forward spikes with fast-sigmoid straight-through gradients."""

    currents = torch.matmul(inputs, weight)
    membrane = torch.zeros(
        currents.shape[1:],
        dtype=currents.dtype,
        device=currents.device,
    )
    spike_sum = membrane.new_zeros(())

    for input_current in currents.unbind(dim=0):
        membrane = membrane * params.decay + input_current
        centered = surrogate_slope * (membrane - params.threshold)
        soft_spike = 0.5 * (centered / (1.0 + centered.abs()) + 1.0)
        hard_spike = (membrane >= params.threshold).to(currents.dtype)
        spike = hard_spike + soft_spike - soft_spike.detach()
        spike_sum = spike_sum + spike.mean()
        membrane = torch.where(hard_spike.bool(), membrane.new_tensor(params.reset), membrane)

    spike_rate = spike_sum / currents.shape[0]
    return spike_rate_loss(spike_rate, target_rate)


def dense_triangular_surrogate_lif_spike_loss(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    params: LIFParams,
    *,
    width: float = 1.0,
    target_rate: float = 0.1,
) -> torch.Tensor:
    """Use a piecewise-linear triangular surrogate spike rate loss."""

    currents = torch.matmul(inputs, weight)
    membrane = torch.zeros(
        currents.shape[1:],
        dtype=currents.dtype,
        device=currents.device,
    )
    spike_sum = membrane.new_zeros(())

    for input_current in currents.unbind(dim=0):
        membrane = membrane * params.decay + input_current
        distance = (membrane - params.threshold).abs() / width
        spike = (1.0 - distance).clamp(min=0.0, max=1.0)
        spike_sum = spike_sum + spike.mean()
        membrane = membrane * (1.0 - spike)

    spike_rate = spike_sum / currents.shape[0]
    return spike_rate_loss(spike_rate, target_rate)


def looped_surrogate_lif_spike_loss(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    params: LIFParams,
    *,
    surrogate_slope: float = 10.0,
    target_rate: float = 0.1,
) -> torch.Tensor:
    """Avoid materializing ``[T, B, N]`` currents by matmuling per timestep."""

    membrane = torch.zeros(
        (inputs.shape[1], weight.shape[1]),
        dtype=inputs.dtype,
        device=inputs.device,
    )
    spike_sum = membrane.new_zeros(())

    for input_features in inputs.unbind(dim=0):
        input_current = torch.matmul(input_features, weight)
        membrane = membrane * params.decay + input_current
        spike = torch.sigmoid(surrogate_slope * (membrane - params.threshold))
        spike_sum = spike_sum + spike.mean()
        membrane = membrane * (1.0 - spike)

    spike_rate = spike_sum / inputs.shape[0]
    return spike_rate_loss(spike_rate, target_rate)


def looped_fast_surrogate_lif_spike_loss(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    params: LIFParams,
    *,
    surrogate_slope: float = 10.0,
    target_rate: float = 0.1,
) -> torch.Tensor:
    """Fast-sigmoid surrogate loss without materializing all timestep currents."""

    membrane = torch.zeros(
        (inputs.shape[1], weight.shape[1]),
        dtype=inputs.dtype,
        device=inputs.device,
    )
    spike_sum = membrane.new_zeros(())

    for input_features in inputs.unbind(dim=0):
        input_current = torch.matmul(input_features, weight)
        membrane = membrane * params.decay + input_current
        centered = surrogate_slope * (membrane - params.threshold)
        spike = 0.5 * (centered / (1.0 + centered.abs()) + 1.0)
        spike_sum = spike_sum + spike.mean()
        membrane = membrane * (1.0 - spike)

    spike_rate = spike_sum / inputs.shape[0]
    return spike_rate_loss(spike_rate, target_rate)


def looped_hard_fast_surrogate_lif_spike_loss(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    params: LIFParams,
    *,
    surrogate_slope: float = 5.0,
    target_rate: float = 0.1,
) -> torch.Tensor:
    """Hard-forward fast-surrogate loss without materialized currents."""

    membrane = torch.zeros(
        (inputs.shape[1], weight.shape[1]),
        dtype=inputs.dtype,
        device=inputs.device,
    )
    spike_sum = membrane.new_zeros(())

    for input_features in inputs.unbind(dim=0):
        input_current = torch.matmul(input_features, weight)
        membrane = membrane * params.decay + input_current
        centered = surrogate_slope * (membrane - params.threshold)
        soft_spike = 0.5 * (centered / (1.0 + centered.abs()) + 1.0)
        hard_spike = (membrane >= params.threshold).to(inputs.dtype)
        spike = hard_spike + soft_spike - soft_spike.detach()
        spike_sum = spike_sum + spike.mean()
        membrane = torch.where(hard_spike.bool(), membrane.new_tensor(params.reset), membrane)

    spike_rate = spike_sum / inputs.shape[0]
    return spike_rate_loss(spike_rate, target_rate)
