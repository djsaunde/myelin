"""Compile-friendly PyTorch modules for spiker."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

import torch
from torch import nn

from spiker._optional import has_triton
from spiker.checkpointing import CheckpointSize
from spiker.dsl import (
    NeuronBackwardPlan,
    NeuronIR,
    evaluate_neuron,
    evaluate_neuron_unroll,
    plan_generated_backward_ir,
    validate_generated_forward_ir,
)
from spiker.kernels import (
    Backend,
    alif_forward,
    izhikevich_forward,
    lif_forward,
    linear_surrogate_lif_forward,
    linear_surrogate_lif_packed_forward,
    linear_surrogate_lif_rate_forward,
    surrogate_alif_forward,
    surrogate_lif_forward,
)
from spiker.neurons import (
    ALIFParams,
    ALIFState,
    IzhikevichParams,
    IzhikevichState,
    LIFParams,
    LIFState,
)
from spiker.online import (
    OnlineALIFGrad,
    OnlineLIFGrad,
    OnlineSurrogate,
    linear_alif_online_eligibility_grad,
    linear_lif_online_eligibility_grad,
)
from spiker.packing import PackedSpikes
from spiker.surrogates import (
    SURROGATE_NAMES,
    SurrogateFn,
    atan_surrogate,
    fast_sigmoid_surrogate,
    hard_surrogate_spike,
    multi_gaussian_surrogate,
    sigmoid_surrogate,
    superspike_surrogate,
    surrogate_name,
    triangular_surrogate,
)

__all__ = [
    "ALIFCell",
    "CustomNeuronCell",
    "CustomSurrogateNeuronCell",
    "DenseLIF",
    "LIFCell",
    "LinearCustomSurrogateNeuron",
    "LinearCustomSurrogateNeuronRate",
    "LinearLIF",
    "LinearOnlineALIF",
    "LinearOnlineLIF",
    "LinearSurrogateLIF",
    "LinearSurrogateLIFPacked",
    "LinearSurrogateLIFRate",
    "LinearSynapse",
    "SurrogateDenseLIF",
    "SurrogateALIFCell",
    "SurrogateLIFCell",
    "SURROGATE_NAMES",
    "TimeUnroll",
    "atan_surrogate",
    "fast_sigmoid_surrogate",
    "hard_surrogate_spike",
    "multi_gaussian_surrogate",
    "sigmoid_surrogate",
    "superspike_surrogate",
    "triangular_surrogate",
]


class DenseLIF(nn.Module):
    """Deprecated convenience wrapper for ``LinearLIF``."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        params: LIFParams | None = None,
    ) -> None:
        super().__init__()
        self.layer = LinearLIF(in_features, out_features, params)

    @property
    def weight(self) -> torch.nn.Parameter:
        return self.layer.synapse.weight

    def forward(self, inputs: torch.Tensor) -> tuple[LIFState, torch.Tensor]:
        return self.layer(inputs)


class LinearSynapse(nn.Module):
    """Trainable dense projection over time-major inputs."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        params: LIFParams | None = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter((torch.rand((in_features, out_features)) - 0.5) * 0.02)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = torch.matmul(inputs, self.weight)
        if self.bias is not None:
            output = output + self.bias
        return output


class LIFCell(nn.Module):
    """Single-step LIF neuron population."""

    def __init__(self, params: LIFParams | None = None) -> None:
        super().__init__()
        self.params = params or LIFParams()

    def forward(
        self,
        state: LIFState,
        input_current: torch.Tensor,
    ) -> tuple[LIFState, torch.Tensor]:
        membrane = state.membrane * self.params.decay + input_current
        spike = membrane >= self.params.threshold
        membrane = torch.where(spike, self.params.reset, membrane)
        return LIFState(membrane=membrane), spike.to(input_current.dtype)


class ALIFCell(nn.Module):
    """Single-step adaptive LIF neuron population."""

    def __init__(self, params: ALIFParams | None = None) -> None:
        super().__init__()
        self.params = params or ALIFParams()

    def forward(
        self,
        state: ALIFState,
        input_current: torch.Tensor,
    ) -> tuple[ALIFState, torch.Tensor]:
        membrane = state.membrane * self.params.decay + input_current
        adaptive_threshold = self.params.threshold + self.params.beta * state.adaptation
        spike = membrane >= adaptive_threshold
        spike_float = spike.to(input_current.dtype)
        membrane = torch.where(spike, self.params.reset, membrane)
        adaptation = state.adaptation * self.params.adaptation_decay + spike_float
        return ALIFState(membrane=membrane, adaptation=adaptation), spike_float


class IzhikevichCell(nn.Module):
    """Single-step Izhikevich-style neuron population."""

    def __init__(self, params: IzhikevichParams | None = None) -> None:
        super().__init__()
        self.params = params or IzhikevichParams()

    def forward(
        self,
        state: IzhikevichState,
        input_current: torch.Tensor,
    ) -> tuple[IzhikevichState, torch.Tensor]:
        self.params.validate()
        voltage_delta = (
            self.params.voltage_square_coeff * state.voltage * state.voltage
            + self.params.voltage_coeff * state.voltage
            + self.params.voltage_bias
            - state.recovery
            + input_current
        )
        pre_reset_voltage = state.voltage + self.params.dt * voltage_delta
        recovery_delta = self.params.recovery_decay * (
            self.params.recovery_coupling * state.voltage - state.recovery
        )
        pre_spike_recovery = state.recovery + self.params.dt * recovery_delta
        spike = pre_reset_voltage >= self.params.threshold
        spike_float = spike.to(input_current.dtype)
        voltage = torch.where(spike, self.params.reset_voltage, pre_reset_voltage)
        recovery = torch.where(
            spike,
            pre_spike_recovery + self.params.recovery_jump,
            pre_spike_recovery,
        )
        return IzhikevichState(voltage=voltage, recovery=recovery), spike_float


class SurrogateLIFCell(nn.Module):
    """Single-step LIF population with hard-forward surrogate-gradient spikes."""

    def __init__(
        self,
        params: LIFParams | None = None,
        surrogate: SurrogateFn = atan_surrogate,
        surrogate_slope: float = 10.0,
        hard_forward: bool = True,
    ) -> None:
        super().__init__()
        self.params = params or LIFParams()
        self.surrogate = surrogate
        self.surrogate_slope = surrogate_slope
        self.hard_forward = hard_forward

    def forward(
        self,
        state: LIFState,
        input_current: torch.Tensor,
    ) -> tuple[LIFState, torch.Tensor]:
        membrane = state.membrane * self.params.decay + input_current
        centered = self.surrogate_slope * (membrane - self.params.threshold)
        if self.hard_forward:
            spike = hard_surrogate_spike(centered, self.surrogate)
        else:
            spike = self.surrogate(centered)
        membrane = membrane * (1.0 - spike) + self.params.reset * spike
        return LIFState(membrane=membrane), spike


class SurrogateALIFCell(nn.Module):
    """Single-step ALIF population with hard-forward surrogate-gradient spikes."""

    def __init__(
        self,
        params: ALIFParams | None = None,
        surrogate: SurrogateFn = atan_surrogate,
        surrogate_slope: float = 10.0,
        hard_forward: bool = True,
    ) -> None:
        super().__init__()
        self.params = params or ALIFParams()
        self.surrogate = surrogate
        self.surrogate_slope = surrogate_slope
        self.hard_forward = hard_forward

    def forward(
        self,
        state: ALIFState,
        input_current: torch.Tensor,
    ) -> tuple[ALIFState, torch.Tensor]:
        membrane = state.membrane * self.params.decay + input_current
        adaptive_threshold = self.params.threshold + self.params.beta * state.adaptation
        centered = self.surrogate_slope * (membrane - adaptive_threshold)
        if self.hard_forward:
            spike = hard_surrogate_spike(centered, self.surrogate)
        else:
            spike = self.surrogate(centered)
        membrane = membrane * (1.0 - spike) + self.params.reset * spike
        adaptation = state.adaptation * self.params.adaptation_decay + spike
        return ALIFState(membrane=membrane, adaptation=adaptation), spike


class CustomNeuronCell(nn.Module):
    """Single-step custom pointwise neuron backed by ``NeuronIR``."""

    def __init__(self, ir: NeuronIR, params: Mapping[str, float] | None = None) -> None:
        super().__init__()
        _validate_custom_neuron_cell_ir(ir)
        self.ir = ir
        self.params = dict(params or {})
        self._validate_params()

    def forward(
        self,
        state: Mapping[str, torch.Tensor],
        input_current: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        next_state, outputs = evaluate_neuron(
            self.ir,
            state_values=state,
            input_values={"input_current": input_current},
            param_values=self.params,
        )
        return next_state, outputs["spike"]

    def initial_state(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            state_name: torch.zeros(
                inputs.shape[1:],
                dtype=inputs.dtype,
                device=inputs.device,
            )
            for state_name in self.ir.state
        }

    def _validate_params(self) -> None:
        missing_params = set(self.ir.params) - set(self.params)
        if missing_params:
            raise ValueError(f"params is missing values: {sorted(missing_params)}")
        unexpected_params = set(self.params) - set(self.ir.params)
        if unexpected_params:
            raise ValueError(f"params contains undeclared values: {sorted(unexpected_params)}")

    def validate_initial_state(self, initial_state: Mapping[str, torch.Tensor]) -> None:
        missing_state = set(self.ir.state) - set(initial_state)
        if missing_state:
            raise ValueError(f"initial_state is missing state tensors: {sorted(missing_state)}")
        unexpected_state = set(initial_state) - set(self.ir.state)
        if unexpected_state:
            raise ValueError(
                f"initial_state contains undeclared state tensors: {sorted(unexpected_state)}"
            )


def _validate_custom_neuron_cell_ir(ir: NeuronIR) -> None:
    validate_generated_forward_ir(ir, context="CustomNeuronCell")


class CustomSurrogateNeuronCell(nn.Module):
    """Surrogate-gradient custom neuron for IRs matching the generated backward ABI."""

    def __init__(
        self,
        ir: NeuronIR,
        params: Mapping[str, float],
        surrogate: SurrogateFn = atan_surrogate,
        surrogate_slope: float = 10.0,
        hard_forward: bool = True,
    ) -> None:
        super().__init__()
        self.backward_plan = plan_generated_backward_ir(ir, context="CustomSurrogateNeuronCell")
        self.ir = ir
        self.params = dict(params)
        self.surrogate = surrogate
        self.surrogate_slope = surrogate_slope
        self.hard_forward = hard_forward
        self._validate_params()
        self.lif_params = _lif_params_from_custom_params(self.params, self.backward_plan)

    def forward(
        self,
        state: Mapping[str, torch.Tensor] | LIFState,
        input_current: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        lif_state = _custom_lif_state(state)
        next_state, spikes = SurrogateLIFCell(
            self.lif_params,
            self.surrogate,
            self.surrogate_slope,
            self.hard_forward,
        )(lif_state, input_current)
        return {"membrane": next_state.membrane}, spikes

    def initial_state(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "membrane": torch.zeros(
                inputs.shape[1:],
                dtype=inputs.dtype,
                device=inputs.device,
            )
        }

    def validate_initial_state(self, initial_state: Mapping[str, torch.Tensor]) -> None:
        missing_state = {"membrane"} - set(initial_state)
        if missing_state:
            raise ValueError(f"initial_state is missing state tensors: {sorted(missing_state)}")
        unexpected_state = set(initial_state) - {"membrane"}
        if unexpected_state:
            raise ValueError(
                f"initial_state contains undeclared state tensors: {sorted(unexpected_state)}"
            )

    def _validate_params(self) -> None:
        missing_params = set(self.ir.params) - set(self.params)
        if missing_params:
            raise ValueError(f"params is missing values: {sorted(missing_params)}")
        unexpected_params = set(self.params) - set(self.ir.params)
        if unexpected_params:
            raise ValueError(f"params contains undeclared values: {sorted(unexpected_params)}")


def _lif_params_from_custom_params(
    params: Mapping[str, float],
    plan: NeuronBackwardPlan,
) -> LIFParams:
    decay = params[plan.decay_param]
    if decay >= 1.0:
        raise ValueError("custom surrogate LIF decay must be less than 1.0")
    tau_mem = 1.0 / (1.0 - decay)
    return LIFParams(
        tau_mem=tau_mem,
        threshold=params[plan.threshold_param],
        reset=params[plan.reset_param],
    )


def _custom_lif_state(state: Mapping[str, torch.Tensor] | LIFState) -> LIFState:
    if isinstance(state, LIFState):
        return state
    return LIFState(membrane=state["membrane"])


class TimeUnroll(nn.Module):
    """Apply a cell across time-major inputs shaped ``[T, B, N]``."""

    def __init__(self, cell: nn.Module, *, backend: Backend = "torch") -> None:
        super().__init__()
        self.cell = cell
        self.backend: Backend = backend

    def forward(
        self,
        inputs: torch.Tensor,
        initial_state: (
            LIFState | ALIFState | IzhikevichState | Mapping[str, torch.Tensor] | None
        ) = None,
    ) -> tuple[
        LIFState | ALIFState | IzhikevichState | Mapping[str, torch.Tensor],
        torch.Tensor,
    ]:
        if initial_state is None:
            membrane = torch.zeros(
                inputs.shape[1:],
                dtype=inputs.dtype,
                device=inputs.device,
            )
            if isinstance(self.cell, CustomNeuronCell | CustomSurrogateNeuronCell):
                initial_state = self.cell.initial_state(inputs)
            elif isinstance(self.cell, ALIFCell | SurrogateALIFCell):
                initial_state = ALIFState(
                    membrane=membrane,
                    adaptation=torch.zeros(
                        inputs.shape[1:],
                        dtype=inputs.dtype,
                        device=inputs.device,
                    ),
                )
            elif isinstance(self.cell, IzhikevichCell):
                initial_state = IzhikevichState(
                    voltage=membrane,
                    recovery=torch.zeros(
                        inputs.shape[1:],
                        dtype=inputs.dtype,
                        device=inputs.device,
                    ),
                )
            else:
                initial_state = LIFState(
                    membrane=membrane,
                )

        if isinstance(self.cell, LIFCell):
            if not isinstance(initial_state, LIFState):
                raise TypeError("LIFCell requires LIFState")
            return lif_forward(
                inputs,
                initial_state,
                self.cell.params,
                backend=self.backend,
            )
        if isinstance(self.cell, ALIFCell):
            if not isinstance(initial_state, ALIFState):
                raise TypeError("ALIFCell requires ALIFState")
            return alif_forward(
                inputs,
                initial_state,
                self.cell.params,
                backend=self.backend,
            )
        if isinstance(self.cell, SurrogateALIFCell):
            if not isinstance(initial_state, ALIFState):
                raise TypeError("SurrogateALIFCell requires ALIFState")
            return surrogate_alif_forward(
                inputs,
                initial_state,
                self.cell.params,
                surrogate=surrogate_name(self.cell.surrogate),
                surrogate_slope=self.cell.surrogate_slope,
                hard_forward=self.cell.hard_forward,
                backend=self.backend,
            )
        if isinstance(self.cell, IzhikevichCell):
            if not isinstance(initial_state, IzhikevichState):
                raise TypeError("IzhikevichCell requires IzhikevichState")
            return izhikevich_forward(
                inputs,
                initial_state,
                self.cell.params,
                backend=self.backend,
            )
        if isinstance(self.cell, CustomSurrogateNeuronCell):
            if not isinstance(initial_state, Mapping):
                raise TypeError("CustomSurrogateNeuronCell requires a mapping initial_state")
            self.cell.validate_initial_state(initial_state)
            lif_initial_state = _custom_lif_state(initial_state)
            lif_state, spikes = surrogate_lif_forward(
                inputs,
                lif_initial_state,
                self.cell.lif_params,
                surrogate=surrogate_name(self.cell.surrogate),
                surrogate_slope=self.cell.surrogate_slope,
                hard_forward=self.cell.hard_forward,
                backend=self.backend,
            )
            return {"membrane": lif_state.membrane}, spikes
        if isinstance(self.cell, CustomNeuronCell):
            if not isinstance(initial_state, Mapping):
                raise TypeError("CustomNeuronCell requires a mapping initial_state")
            self.cell.validate_initial_state(initial_state)
            if self.backend == "triton":
                from spiker.triton import generated_neuron_forward

                return generated_neuron_forward(
                    self.cell.ir,
                    inputs,
                    initial_state,
                    self.cell.params,
                )
            if self.backend == "triton_generated":
                from spiker.triton import generated_neuron_forward

                return generated_neuron_forward(
                    self.cell.ir,
                    inputs,
                    initial_state,
                    self.cell.params,
                )
            if self.backend == "auto" and inputs.is_cuda and has_triton():
                from spiker.triton import generated_neuron_forward

                return generated_neuron_forward(
                    self.cell.ir,
                    inputs,
                    initial_state,
                    self.cell.params,
                )
            if self.backend not in {"auto", "torch"}:
                raise ValueError(f"unsupported backend: {self.backend}")
            return evaluate_neuron_unroll(
                self.cell.ir,
                inputs,
                initial_state,
                self.cell.params,
            )
        if (
            isinstance(self.cell, SurrogateLIFCell)
            and self.backend != "torch"
            and self.cell.hard_forward
        ):
            if not isinstance(initial_state, LIFState):
                raise TypeError("SurrogateLIFCell requires LIFState")
            return surrogate_lif_forward(
                inputs,
                initial_state,
                self.cell.params,
                surrogate=surrogate_name(self.cell.surrogate),
                surrogate_slope=self.cell.surrogate_slope,
                hard_forward=self.cell.hard_forward,
                backend=self.backend,
            )

        state = initial_state
        if state is None:
            raise TypeError("TimeUnroll could not infer an initial state")
        spikes = []
        for input_current in inputs.unbind(dim=0):
            state, spike = self.cell(state, input_current)
            spikes.append(spike)

        return state, torch.stack(spikes, dim=0)


class LinearLIF(nn.Module):
    """Convenience wrapper: ``LinearSynapse`` followed by ``TimeUnroll(LIFCell)``."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        params: LIFParams | None = None,
        bias: bool = True,
        backend: Backend = "torch",
    ) -> None:
        super().__init__()
        self.synapse = LinearSynapse(in_features, out_features, bias=bias)
        self.unroll = TimeUnroll(LIFCell(params), backend=backend)

    def forward(self, inputs: torch.Tensor) -> tuple[LIFState, torch.Tensor]:
        return self.unroll(self.synapse(inputs))


class LinearOnlineLIF(nn.Module):
    """Dense LIF layer exposing an online eligibility-trace update estimate."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        params: LIFParams | None = None,
        surrogate: SurrogateFn | NeuronIR = fast_sigmoid_surrogate,
        surrogate_slope: float = 5.0,
        surrogate_params: Mapping[str, float] | None = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.synapse = LinearSynapse(in_features, out_features, bias=bias)
        self.params = params or LIFParams()
        self.surrogate: OnlineSurrogate = (
            surrogate if isinstance(surrogate, NeuronIR) else surrogate_name(surrogate)
        )
        self.surrogate_slope = surrogate_slope
        self.surrogate_params = None if surrogate_params is None else dict(surrogate_params)

    def forward(self, inputs: torch.Tensor, learning_signal: torch.Tensor) -> OnlineLIFGrad:
        return linear_lif_online_eligibility_grad(
            inputs,
            self.synapse.weight,
            self.synapse.bias,
            learning_signal,
            self.params,
            surrogate=self.surrogate,
            surrogate_slope=self.surrogate_slope,
            surrogate_params=self.surrogate_params,
        )

    def step_online(
        self,
        inputs: torch.Tensor,
        learning_signal: torch.Tensor,
        *,
        lr: float,
    ) -> OnlineLIFGrad:
        if lr < 0:
            raise ValueError("lr must be non-negative")
        result = self(inputs, learning_signal)
        with torch.no_grad():
            self.synapse.weight -= lr * result.grad_weight
            if self.synapse.bias is not None and result.grad_bias is not None:
                self.synapse.bias -= lr * result.grad_bias
        return result


class LinearOnlineALIF(nn.Module):
    """Dense ALIF layer exposing an online eligibility-trace update estimate."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        params: ALIFParams | None = None,
        surrogate: SurrogateFn | NeuronIR = fast_sigmoid_surrogate,
        surrogate_slope: float = 5.0,
        surrogate_params: Mapping[str, float] | None = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.synapse = LinearSynapse(in_features, out_features, bias=bias)
        self.params = params or ALIFParams()
        self.surrogate: OnlineSurrogate = (
            surrogate if isinstance(surrogate, NeuronIR) else surrogate_name(surrogate)
        )
        self.surrogate_slope = surrogate_slope
        self.surrogate_params = None if surrogate_params is None else dict(surrogate_params)

    def forward(self, inputs: torch.Tensor, learning_signal: torch.Tensor) -> OnlineALIFGrad:
        return linear_alif_online_eligibility_grad(
            inputs,
            self.synapse.weight,
            self.synapse.bias,
            learning_signal,
            self.params,
            surrogate=self.surrogate,
            surrogate_slope=self.surrogate_slope,
            surrogate_params=self.surrogate_params,
        )

    def step_online(
        self,
        inputs: torch.Tensor,
        learning_signal: torch.Tensor,
        *,
        lr: float,
    ) -> OnlineALIFGrad:
        if lr < 0:
            raise ValueError("lr must be non-negative")
        result = self(inputs, learning_signal)
        with torch.no_grad():
            self.synapse.weight -= lr * result.grad_weight
            if self.synapse.bias is not None and result.grad_bias is not None:
                self.synapse.bias -= lr * result.grad_bias
        return result


class LinearSurrogateLIF(nn.Module):
    """Convenience wrapper for dense projection plus surrogate LIF unroll."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        params: LIFParams | None = None,
        surrogate: SurrogateFn = atan_surrogate,
        surrogate_slope: float = 10.0,
        hard_forward: bool = True,
        bias: bool = True,
        backend: Backend = "torch",
        stream_synapse: bool = False,
        checkpoint_size: CheckpointSize | None = None,
    ) -> None:
        super().__init__()
        self.synapse = LinearSynapse(in_features, out_features, bias=bias)
        self.stream_synapse = stream_synapse
        self.checkpoint_size: CheckpointSize | None = checkpoint_size
        self.unroll = TimeUnroll(
            SurrogateLIFCell(params, surrogate, surrogate_slope, hard_forward),
            backend=backend,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.stream_synapse:
            cell = self.unroll.cell
            if not isinstance(cell, SurrogateLIFCell):
                raise TypeError("stream_synapse requires SurrogateLIFCell")
            _state, spikes = linear_surrogate_lif_forward(
                inputs,
                self.synapse.weight,
                self.synapse.bias,
                cell.params,
                surrogate=surrogate_name(cell.surrogate),
                surrogate_slope=cell.surrogate_slope,
                hard_forward=cell.hard_forward,
                backend=self.unroll.backend,
                checkpoint_size=self.checkpoint_size,
            )
            return spikes

        _state, spikes = self.unroll(self.synapse(inputs))
        return spikes


class LinearCustomSurrogateNeuron(nn.Module):
    """Dense projection plus a surrogate custom neuron with generated-backward ABI."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        ir: NeuronIR,
        params: Mapping[str, float],
        surrogate: SurrogateFn = atan_surrogate,
        surrogate_slope: float = 10.0,
        hard_forward: bool = True,
        bias: bool = True,
        backend: Backend = "torch",
        stream_synapse: bool = False,
        checkpoint_size: CheckpointSize | None = None,
    ) -> None:
        super().__init__()
        self.synapse = LinearSynapse(in_features, out_features, bias=bias)
        self.stream_synapse = stream_synapse
        self.checkpoint_size: CheckpointSize | None = checkpoint_size
        self.cell = CustomSurrogateNeuronCell(
            ir,
            params,
            surrogate=surrogate,
            surrogate_slope=surrogate_slope,
            hard_forward=hard_forward,
        )
        self.unroll = TimeUnroll(self.cell, backend=backend)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.stream_synapse:
            _state, spikes = linear_surrogate_lif_forward(
                inputs,
                self.synapse.weight,
                self.synapse.bias,
                self.cell.lif_params,
                surrogate=surrogate_name(self.cell.surrogate),
                surrogate_slope=self.cell.surrogate_slope,
                hard_forward=self.cell.hard_forward,
                backend=self.unroll.backend,
                checkpoint_size=self.checkpoint_size,
            )
            return spikes

        currents = self.synapse(inputs)
        _state, spikes = self.unroll(currents)
        return spikes


class LinearSurrogateLIFRate(nn.Module):
    """Dense projection plus surrogate LIF returning spike rates instead of spikes.

    ``backend="auto"`` follows the stable training recommendation: CUDA tensors
    use the Triton rate path when Triton is installed, otherwise PyTorch is used.
    ``backend="triton_compile"`` is never selected automatically; it is an
    explicit experimental memory-oriented path for longer-T rate readouts.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        params: LIFParams | None = None,
        surrogate: SurrogateFn = atan_surrogate,
        surrogate_slope: float = 10.0,
        hard_forward: bool = True,
        bias: bool = True,
        backend: Backend = "torch",
        checkpoint_size: CheckpointSize = 25,
        reduction: Literal["mean", "none"] = "none",
    ) -> None:
        super().__init__()
        self.synapse = LinearSynapse(in_features, out_features, bias=bias)
        self.cell = SurrogateLIFCell(params, surrogate, surrogate_slope, hard_forward)
        self.backend: Backend = backend
        self.checkpoint_size: CheckpointSize = checkpoint_size
        self.reduction: Literal["mean", "none"] = reduction

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        _state, rates = linear_surrogate_lif_rate_forward(
            inputs,
            self.synapse.weight,
            self.synapse.bias,
            self.cell.params,
            surrogate=surrogate_name(self.cell.surrogate),
            surrogate_slope=self.cell.surrogate_slope,
            hard_forward=self.cell.hard_forward,
            backend=self.backend,
            checkpoint_size=self.checkpoint_size,
            reduction=self.reduction,
        )
        return rates


class LinearSurrogateLIFPacked(nn.Module):
    """Dense projection plus surrogate LIF returning bitpacked spike traces.

    The packed ``int32`` output is forward-only and not differentiable. Use
    ``LinearSurrogateLIF`` or ``LinearSurrogateLIFRate`` for training losses.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        params: LIFParams | None = None,
        surrogate: SurrogateFn = atan_surrogate,
        surrogate_slope: float = 10.0,
        hard_forward: bool = True,
        bias: bool = True,
        backend: Backend = "torch",
        checkpoint_size: CheckpointSize = 25,
    ) -> None:
        super().__init__()
        self.synapse = LinearSynapse(in_features, out_features, bias=bias)
        self.cell = SurrogateLIFCell(params, surrogate, surrogate_slope, hard_forward)
        self.backend: Backend = backend
        self.checkpoint_size: CheckpointSize = checkpoint_size

    def forward(self, inputs: torch.Tensor) -> PackedSpikes:
        _state, packed_spikes = linear_surrogate_lif_packed_forward(
            inputs,
            self.synapse.weight,
            self.synapse.bias,
            self.cell.params,
            surrogate=surrogate_name(self.cell.surrogate),
            surrogate_slope=self.cell.surrogate_slope,
            hard_forward=self.cell.hard_forward,
            backend=self.backend,
            checkpoint_size=self.checkpoint_size,
        )
        return packed_spikes


class LinearCustomSurrogateNeuronRate(nn.Module):
    """Dense projection plus custom surrogate neuron returning spike rates."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        ir: NeuronIR,
        params: Mapping[str, float],
        surrogate: SurrogateFn = atan_surrogate,
        surrogate_slope: float = 10.0,
        hard_forward: bool = True,
        bias: bool = True,
        backend: Backend = "torch",
        checkpoint_size: CheckpointSize = 25,
        reduction: Literal["mean", "none"] = "none",
    ) -> None:
        super().__init__()
        self.synapse = LinearSynapse(in_features, out_features, bias=bias)
        self.cell = CustomSurrogateNeuronCell(
            ir,
            params,
            surrogate=surrogate,
            surrogate_slope=surrogate_slope,
            hard_forward=hard_forward,
        )
        self.backend: Backend = backend
        self.checkpoint_size: CheckpointSize = checkpoint_size
        self.reduction: Literal["mean", "none"] = reduction

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        _state, rates = linear_surrogate_lif_rate_forward(
            inputs,
            self.synapse.weight,
            self.synapse.bias,
            self.cell.lif_params,
            surrogate=surrogate_name(self.cell.surrogate),
            surrogate_slope=self.cell.surrogate_slope,
            hard_forward=self.cell.hard_forward,
            backend=self.backend,
            checkpoint_size=self.checkpoint_size,
            reduction=self.reduction,
        )
        return rates


class SurrogateDenseLIF(nn.Module):
    """Deprecated convenience wrapper for ``LinearSurrogateLIF``."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        params: LIFParams | None = None,
        surrogate: SurrogateFn = atan_surrogate,
        surrogate_slope: float = 10.0,
        hard_forward: bool = True,
        bias: bool = True,
        backend: Backend = "torch",
        stream_synapse: bool = False,
        checkpoint_size: CheckpointSize | None = None,
    ) -> None:
        super().__init__()
        self.layer = LinearSurrogateLIF(
            in_features,
            out_features,
            params,
            surrogate,
            surrogate_slope,
            hard_forward,
            bias,
            backend,
            stream_synapse,
            checkpoint_size,
        )

    @property
    def weight(self) -> torch.nn.Parameter:
        return self.layer.synapse.weight

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layer(inputs)
