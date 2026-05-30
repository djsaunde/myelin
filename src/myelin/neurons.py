"""Reference neuron dynamics.

The functions in this module are intentionally ordinary PyTorch code. They are
the correctness oracle for future generated and Triton-backed kernels.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LIFParams:
    """Parameters for a leaky integrate-and-fire neuron population."""

    tau_mem: float = 20.0
    threshold: float = 1.0
    reset: float = 0.0

    @property
    def decay(self) -> float:
        """Euler-style membrane decay factor for one simulation step."""

        if self.tau_mem <= 0:
            raise ValueError("tau_mem must be positive")
        return 1.0 - (1.0 / self.tau_mem)


@dataclass(frozen=True)
class LIFState:
    """State carried between LIF simulation steps."""

    membrane: torch.Tensor


@dataclass(frozen=True)
class ALIFParams:
    """Parameters for an adaptive leaky integrate-and-fire neuron population."""

    tau_mem: float = 20.0
    tau_adaptation: float = 100.0
    threshold: float = 1.0
    reset: float = 0.0
    beta: float = 1.0

    @property
    def decay(self) -> float:
        """Euler-style membrane decay factor for one simulation step."""

        if self.tau_mem <= 0:
            raise ValueError("tau_mem must be positive")
        return 1.0 - (1.0 / self.tau_mem)

    @property
    def adaptation_decay(self) -> float:
        """Euler-style adaptation decay factor for one simulation step."""

        if self.tau_adaptation <= 0:
            raise ValueError("tau_adaptation must be positive")
        return 1.0 - (1.0 / self.tau_adaptation)


@dataclass(frozen=True)
class ALIFState:
    """State carried between ALIF simulation steps."""

    membrane: torch.Tensor
    adaptation: torch.Tensor


@dataclass(frozen=True)
class IzhikevichParams:
    """Parameters for an Izhikevich-style neuron population.

    The default values are the common regular-spiking configuration from the
    discrete Izhikevich model. ``dt`` scales both voltage and recovery updates.
    """

    recovery_decay: float = 0.02
    recovery_coupling: float = 0.2
    reset_voltage: float = -65.0
    recovery_jump: float = 8.0
    threshold: float = 30.0
    dt: float = 1.0
    voltage_square_coeff: float = 0.04
    voltage_coeff: float = 5.0
    voltage_bias: float = 140.0

    def validate(self) -> None:
        """Validate scalar parameters used by the reference implementation."""

        if self.dt <= 0:
            raise ValueError("dt must be positive")


@dataclass(frozen=True)
class IzhikevichState:
    """State carried between Izhikevich simulation steps."""

    voltage: torch.Tensor
    recovery: torch.Tensor


def _lif_step_unchecked(
    state: LIFState,
    input_current: torch.Tensor,
    params: LIFParams,
) -> tuple[LIFState, torch.Tensor]:
    membrane = state.membrane * params.decay + input_current
    spike = membrane >= params.threshold
    membrane = torch.where(spike, params.reset, membrane)
    return LIFState(membrane=membrane), spike.to(input_current.dtype)


def _alif_step_unchecked(
    state: ALIFState,
    input_current: torch.Tensor,
    params: ALIFParams,
) -> tuple[ALIFState, torch.Tensor]:
    membrane = state.membrane * params.decay + input_current
    adaptive_threshold = params.threshold + params.beta * state.adaptation
    spike = membrane >= adaptive_threshold
    spike_float = spike.to(input_current.dtype)
    next_membrane = torch.where(spike, params.reset, membrane)
    next_adaptation = state.adaptation * params.adaptation_decay + spike_float
    return ALIFState(membrane=next_membrane, adaptation=next_adaptation), spike_float


def _izhikevich_step_unchecked(
    state: IzhikevichState,
    input_current: torch.Tensor,
    params: IzhikevichParams,
) -> tuple[IzhikevichState, torch.Tensor]:
    voltage_delta = (
        params.voltage_square_coeff * state.voltage * state.voltage
        + params.voltage_coeff * state.voltage
        + params.voltage_bias
        - state.recovery
        + input_current
    )
    pre_reset_voltage = state.voltage + params.dt * voltage_delta
    recovery_delta = params.recovery_decay * (
        params.recovery_coupling * state.voltage - state.recovery
    )
    pre_spike_recovery = state.recovery + params.dt * recovery_delta
    spike = pre_reset_voltage >= params.threshold
    spike_float = spike.to(input_current.dtype)
    next_voltage = torch.where(spike, params.reset_voltage, pre_reset_voltage)
    next_recovery = torch.where(
        spike,
        pre_spike_recovery + params.recovery_jump,
        pre_spike_recovery,
    )
    return IzhikevichState(voltage=next_voltage, recovery=next_recovery), spike_float


def lif_step(
    state: LIFState,
    input_current: torch.Tensor,
    params: LIFParams,
) -> tuple[LIFState, torch.Tensor]:
    """Advance one LIF timestep.

    The reset is hard: neurons that spike are assigned ``params.reset`` after
    thresholding. This simple reference rule keeps M1 benchmarking honest before
    we add code-generated variants.
    """

    if state.membrane.shape != input_current.shape:
        msg = (
            "state.membrane and input_current must have the same shape; "
            f"got {tuple(state.membrane.shape)} and {tuple(input_current.shape)}"
        )
        raise ValueError(msg)

    return _lif_step_unchecked(state, input_current, params)


def alif_step(
    state: ALIFState,
    input_current: torch.Tensor,
    params: ALIFParams,
) -> tuple[ALIFState, torch.Tensor]:
    """Advance one adaptive LIF timestep."""

    if state.membrane.shape != input_current.shape:
        msg = (
            "state.membrane and input_current must have the same shape; "
            f"got {tuple(state.membrane.shape)} and {tuple(input_current.shape)}"
        )
        raise ValueError(msg)
    if state.adaptation.shape != input_current.shape:
        msg = (
            "state.adaptation and input_current must have the same shape; "
            f"got {tuple(state.adaptation.shape)} and {tuple(input_current.shape)}"
        )
        raise ValueError(msg)

    return _alif_step_unchecked(state, input_current, params)


def izhikevich_step(
    state: IzhikevichState,
    input_current: torch.Tensor,
    params: IzhikevichParams,
) -> tuple[IzhikevichState, torch.Tensor]:
    """Advance one Izhikevich-style timestep."""

    params.validate()
    if state.voltage.shape != input_current.shape:
        msg = (
            "state.voltage and input_current must have the same shape; "
            f"got {tuple(state.voltage.shape)} and {tuple(input_current.shape)}"
        )
        raise ValueError(msg)
    if state.recovery.shape != input_current.shape:
        msg = (
            "state.recovery and input_current must have the same shape; "
            f"got {tuple(state.recovery.shape)} and {tuple(input_current.shape)}"
        )
        raise ValueError(msg)

    return _izhikevich_step_unchecked(state, input_current, params)
