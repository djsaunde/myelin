"""Functional reference simulations."""

from __future__ import annotations

import torch

from myelin.neurons import (
    ALIFParams,
    ALIFState,
    IzhikevichParams,
    IzhikevichState,
    LIFParams,
    LIFState,
    _alif_step_unchecked,
    _izhikevich_step_unchecked,
    _lif_step_unchecked,
)
from myelin.surrogates import SurrogateFn, hard_surrogate_spike


def lif_unroll(
    inputs: torch.Tensor,
    initial_state: LIFState,
    params: LIFParams,
) -> tuple[LIFState, torch.Tensor]:
    """Run a time-major LIF simulation over inputs shaped ``[T, B, N]``."""

    if inputs.ndim != 3:
        raise ValueError(f"inputs must be shaped [T, B, N]; got {tuple(inputs.shape)}")

    expected_state_shape = inputs.shape[1:]
    if initial_state.membrane.shape != expected_state_shape:
        msg = (
            "initial_state.membrane must match inputs.shape[1:]; "
            f"got {tuple(initial_state.membrane.shape)} and {tuple(expected_state_shape)}"
        )
        raise ValueError(msg)

    spikes = []
    state = initial_state
    for input_current in inputs.unbind(dim=0):
        state, spike = _lif_step_unchecked(state, input_current, params)
        spikes.append(spike)

    return state, torch.stack(spikes, dim=0)


def alif_unroll(
    inputs: torch.Tensor,
    initial_state: ALIFState,
    params: ALIFParams,
) -> tuple[ALIFState, torch.Tensor]:
    """Run a time-major ALIF simulation over inputs shaped ``[T, B, N]``."""

    if inputs.ndim != 3:
        raise ValueError(f"inputs must be shaped [T, B, N]; got {tuple(inputs.shape)}")

    expected_state_shape = inputs.shape[1:]
    if initial_state.membrane.shape != expected_state_shape:
        msg = (
            "initial_state.membrane must match inputs.shape[1:]; "
            f"got {tuple(initial_state.membrane.shape)} and {tuple(expected_state_shape)}"
        )
        raise ValueError(msg)
    if initial_state.adaptation.shape != expected_state_shape:
        msg = (
            "initial_state.adaptation must match inputs.shape[1:]; "
            f"got {tuple(initial_state.adaptation.shape)} and {tuple(expected_state_shape)}"
        )
        raise ValueError(msg)

    spikes = []
    state = initial_state
    for input_current in inputs.unbind(dim=0):
        state, spike = _alif_step_unchecked(state, input_current, params)
        spikes.append(spike)

    return state, torch.stack(spikes, dim=0)


def izhikevich_unroll(
    inputs: torch.Tensor,
    initial_state: IzhikevichState,
    params: IzhikevichParams,
) -> tuple[IzhikevichState, torch.Tensor]:
    """Run a time-major Izhikevich simulation over inputs shaped ``[T, B, N]``."""

    params.validate()
    if inputs.ndim != 3:
        raise ValueError(f"inputs must be shaped [T, B, N]; got {tuple(inputs.shape)}")

    expected_state_shape = inputs.shape[1:]
    if initial_state.voltage.shape != expected_state_shape:
        msg = (
            "initial_state.voltage must match inputs.shape[1:]; "
            f"got {tuple(initial_state.voltage.shape)} and {tuple(expected_state_shape)}"
        )
        raise ValueError(msg)
    if initial_state.recovery.shape != expected_state_shape:
        msg = (
            "initial_state.recovery must match inputs.shape[1:]; "
            f"got {tuple(initial_state.recovery.shape)} and {tuple(expected_state_shape)}"
        )
        raise ValueError(msg)

    spikes = []
    state = initial_state
    for input_current in inputs.unbind(dim=0):
        state, spike = _izhikevich_step_unchecked(state, input_current, params)
        spikes.append(spike)

    return state, torch.stack(spikes, dim=0)


def surrogate_lif_unroll(
    inputs: torch.Tensor,
    initial_state: LIFState,
    params: LIFParams,
    *,
    surrogate: SurrogateFn,
    surrogate_slope: float = 10.0,
    hard_forward: bool = True,
) -> tuple[LIFState, torch.Tensor]:
    """Run a time-major surrogate-gradient LIF simulation."""

    if inputs.ndim != 3:
        raise ValueError(f"inputs must be shaped [T, B, N]; got {tuple(inputs.shape)}")

    expected_state_shape = inputs.shape[1:]
    if initial_state.membrane.shape != expected_state_shape:
        msg = (
            "initial_state.membrane must match inputs.shape[1:]; "
            f"got {tuple(initial_state.membrane.shape)} and {tuple(expected_state_shape)}"
        )
        raise ValueError(msg)

    spikes = []
    membrane = initial_state.membrane
    for input_current in inputs.unbind(dim=0):
        membrane = membrane * params.decay + input_current
        centered = surrogate_slope * (membrane - params.threshold)
        spike = hard_surrogate_spike(centered, surrogate) if hard_forward else surrogate(centered)
        membrane = membrane * (1.0 - spike) + params.reset * spike
        spikes.append(spike)

    return LIFState(membrane=membrane), torch.stack(spikes, dim=0)


def surrogate_alif_unroll(
    inputs: torch.Tensor,
    initial_state: ALIFState,
    params: ALIFParams,
    *,
    surrogate: SurrogateFn,
    surrogate_slope: float = 10.0,
    hard_forward: bool = True,
) -> tuple[ALIFState, torch.Tensor]:
    """Run a time-major surrogate-gradient ALIF simulation."""

    if inputs.ndim != 3:
        raise ValueError(f"inputs must be shaped [T, B, N]; got {tuple(inputs.shape)}")

    expected_state_shape = inputs.shape[1:]
    if initial_state.membrane.shape != expected_state_shape:
        msg = (
            "initial_state.membrane must match inputs.shape[1:]; "
            f"got {tuple(initial_state.membrane.shape)} and {tuple(expected_state_shape)}"
        )
        raise ValueError(msg)
    if initial_state.adaptation.shape != expected_state_shape:
        msg = (
            "initial_state.adaptation must match inputs.shape[1:]; "
            f"got {tuple(initial_state.adaptation.shape)} and {tuple(expected_state_shape)}"
        )
        raise ValueError(msg)

    spikes = []
    membrane = initial_state.membrane
    adaptation = initial_state.adaptation
    for input_current in inputs.unbind(dim=0):
        membrane = membrane * params.decay + input_current
        adaptive_threshold = params.threshold + params.beta * adaptation
        centered = surrogate_slope * (membrane - adaptive_threshold)
        spike = hard_surrogate_spike(centered, surrogate) if hard_forward else surrogate(centered)
        membrane = membrane * (1.0 - spike) + params.reset * spike
        adaptation = adaptation * params.adaptation_decay + spike
        spikes.append(spike)

    return ALIFState(membrane=membrane, adaptation=adaptation), torch.stack(spikes, dim=0)
