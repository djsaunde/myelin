"""Custom autograd boundaries for fused SNN kernels."""

from __future__ import annotations

from typing import Any, Literal

import torch

from myelin.neurons import LIFParams, LIFState
from myelin.surrogates import SurrogateName, surrogate_derivative, surrogate_from_name


class TritonLIFForwardFunction(torch.autograd.Function):
    """Autograd boundary for the Triton fused-time LIF forward kernel."""

    @staticmethod
    def forward(
        ctx: Any,
        inputs: torch.Tensor,
        initial_membrane: torch.Tensor,
        tau_mem: float,
        threshold: float,
        reset: float,
        block_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from myelin.triton import lif_forward

        ctx.set_materialize_grads(False)
        ctx.save_for_backward(inputs, initial_membrane)
        ctx.tau_mem = tau_mem
        ctx.threshold = threshold
        ctx.reset = reset

        params = LIFParams(tau_mem=tau_mem, threshold=threshold, reset=reset)
        state, spikes = lif_forward(
            inputs,
            LIFState(membrane=initial_membrane),
            params,
            block_size=block_size,
        )
        return state.membrane, spikes

    @staticmethod
    def backward(
        ctx: Any,
        *grad_outputs: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, None, None, None, None]:
        grad_final_membrane, grad_spikes = grad_outputs
        if grad_spikes is not None and bool(torch.any(grad_spikes != 0).item()):
            raise NotImplementedError(
                "Triton LIF spike-output backward is not implemented yet. Use "
                "backend='torch' for surrogate-gradient training."
            )

        if grad_final_membrane is None:
            return None, None, None, None, None, None

        inputs, initial_membrane = ctx.saved_tensors
        params = LIFParams(
            tau_mem=ctx.tau_mem,
            threshold=ctx.threshold,
            reset=ctx.reset,
        )

        needs_input_grad, needs_initial_grad = ctx.needs_input_grad[:2]
        replay_inputs = inputs.detach().requires_grad_(needs_input_grad)
        replay_initial = initial_membrane.detach().requires_grad_(needs_initial_grad)

        grad_targets = []
        if needs_input_grad:
            grad_targets.append(replay_inputs)
        if needs_initial_grad:
            grad_targets.append(replay_initial)
        if not grad_targets:
            return None, None, None, None, None, None

        from myelin.functional import lif_unroll

        with torch.enable_grad():
            final_state, _spikes = lif_unroll(
                replay_inputs,
                LIFState(membrane=replay_initial),
                params,
            )
            grads = torch.autograd.grad(
                final_state.membrane,
                grad_targets,
                grad_final_membrane,
                allow_unused=True,
            )

        grad_inputs = None
        grad_initial = None
        grad_index = 0
        if needs_input_grad:
            grad_inputs = grads[grad_index]
            grad_index += 1
        if needs_initial_grad:
            grad_initial = grads[grad_index]

        return grad_inputs, grad_initial, None, None, None, None


def triton_lif_forward_function(
    inputs: torch.Tensor,
    initial_state: LIFState,
    params: LIFParams,
    *,
    block_size: int = 256,
) -> tuple[LIFState, torch.Tensor]:
    """Run Triton LIF forward through an explicit autograd boundary."""

    final_membrane, spikes = TritonLIFForwardFunction.apply(
        inputs,
        initial_state.membrane,
        params.tau_mem,
        params.threshold,
        params.reset,
        block_size,
    )
    return LIFState(membrane=final_membrane), spikes


def fused_lif_function() -> type[TritonLIFForwardFunction]:
    """Return the current Triton LIF autograd Function class."""

    return TritonLIFForwardFunction


class TritonSurrogateLIFFunction(torch.autograd.Function):
    """Triton hard-forward LIF with explicit surrogate backward."""

    @staticmethod
    def forward(
        ctx: Any,
        inputs: torch.Tensor,
        initial_membrane: torch.Tensor,
        tau_mem: float,
        threshold: float,
        reset: float,
        surrogate: SurrogateName,
        surrogate_slope: float,
        block_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from myelin.triton import pack_spikes_triton, surrogate_lif_forward

        ctx.set_materialize_grads(False)
        ctx.tau_mem = tau_mem
        ctx.threshold = threshold
        ctx.reset = reset
        ctx.surrogate = surrogate
        ctx.surrogate_slope = surrogate_slope
        ctx.block_size = block_size

        params = LIFParams(tau_mem=tau_mem, threshold=threshold, reset=reset)
        state, spikes, pre_reset_membranes = surrogate_lif_forward(
            inputs,
            LIFState(membrane=initial_membrane),
            params,
            block_size=block_size,
        )
        packed_spikes = pack_spikes_triton(spikes)
        ctx.save_for_backward(pre_reset_membranes, packed_spikes.data)
        return state.membrane, spikes

    @staticmethod
    def backward(
        ctx: Any,
        *grad_outputs: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, None, None, None, None, None, None, None]:
        grad_final_membrane, grad_spikes = grad_outputs
        if grad_final_membrane is None and grad_spikes is None:
            return None, None, None, None, None, None, None, None, None

        needs_input_grad, needs_initial_grad = ctx.needs_input_grad[:2]
        if not needs_input_grad and not needs_initial_grad:
            return None, None, None, None, None, None, None, None, None

        pre_reset_membranes, packed_spikes = ctx.saved_tensors
        from myelin.triton import surrogate_lif_backward_packed_spikes

        params = LIFParams(
            tau_mem=ctx.tau_mem,
            threshold=ctx.threshold,
            reset=ctx.reset,
        )
        grad_inputs, grad_initial = surrogate_lif_backward_packed_spikes(
            pre_reset_membranes,
            packed_spikes,
            grad_final_membrane,
            grad_spikes,
            params,
            surrogate=ctx.surrogate,
            surrogate_slope=ctx.surrogate_slope,
            block_size=ctx.block_size,
        )
        if not needs_input_grad:
            grad_inputs = None
        if not needs_initial_grad:
            grad_initial = None
        return grad_inputs, grad_initial, None, None, None, None, None, None, None


def triton_surrogate_lif_function(
    inputs: torch.Tensor,
    initial_state: LIFState,
    params: LIFParams,
    *,
    surrogate: SurrogateName = "atan",
    surrogate_slope: float = 10.0,
    hard_forward: bool = True,
    block_size: int = 256,
) -> tuple[LIFState, torch.Tensor]:
    """Run hard-forward surrogate LIF through a Triton/autograd boundary."""

    if not hard_forward:
        raise ValueError("Triton surrogate LIF currently supports only hard_forward=True")

    final_membrane, spikes = TritonSurrogateLIFFunction.apply(
        inputs,
        initial_state.membrane,
        params.tau_mem,
        params.threshold,
        params.reset,
        surrogate,
        surrogate_slope,
        block_size,
    )
    return LIFState(membrane=final_membrane), spikes


class GeneratedTritonSurrogateLIFFunction(torch.autograd.Function):
    """Triton surrogate LIF using generated surrogate backward code."""

    @staticmethod
    def forward(
        ctx: Any,
        inputs: torch.Tensor,
        initial_membrane: torch.Tensor,
        tau_mem: float,
        threshold: float,
        reset: float,
        surrogate: SurrogateName,
        surrogate_slope: float,
        block_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from myelin.triton import pack_spikes_triton, surrogate_lif_forward

        ctx.set_materialize_grads(False)
        ctx.tau_mem = tau_mem
        ctx.threshold = threshold
        ctx.reset = reset
        ctx.surrogate = surrogate
        ctx.surrogate_slope = surrogate_slope
        ctx.block_size = block_size

        params = LIFParams(tau_mem=tau_mem, threshold=threshold, reset=reset)
        state, spikes, pre_reset_membranes = surrogate_lif_forward(
            inputs,
            LIFState(membrane=initial_membrane),
            params,
            block_size=block_size,
        )
        packed_spikes = pack_spikes_triton(spikes)
        ctx.save_for_backward(pre_reset_membranes, packed_spikes.data)
        return state.membrane, spikes

    @staticmethod
    def backward(
        ctx: Any,
        *grad_outputs: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, None, None, None, None, None, None, None]:
        grad_final_membrane, grad_spikes = grad_outputs
        if grad_final_membrane is None and grad_spikes is None:
            return None, None, None, None, None, None, None, None, None

        needs_input_grad, needs_initial_grad = ctx.needs_input_grad[:2]
        if not needs_input_grad and not needs_initial_grad:
            return None, None, None, None, None, None, None, None, None

        pre_reset_membranes, packed_spikes = ctx.saved_tensors
        from myelin.triton import generated_lif_surrogate_backward_packed_spikes

        params = LIFParams(
            tau_mem=ctx.tau_mem,
            threshold=ctx.threshold,
            reset=ctx.reset,
        )
        grad_inputs, grad_initial = generated_lif_surrogate_backward_packed_spikes(
            pre_reset_membranes,
            packed_spikes,
            grad_final_membrane,
            grad_spikes,
            params,
            surrogate=ctx.surrogate,
            surrogate_slope=ctx.surrogate_slope,
            block_size=ctx.block_size,
        )
        if not needs_input_grad:
            grad_inputs = None
        if not needs_initial_grad:
            grad_initial = None
        return grad_inputs, grad_initial, None, None, None, None, None, None, None


def generated_triton_surrogate_lif_function(
    inputs: torch.Tensor,
    initial_state: LIFState,
    params: LIFParams,
    *,
    surrogate: SurrogateName = "atan",
    surrogate_slope: float = 10.0,
    hard_forward: bool = True,
    block_size: int = 256,
) -> tuple[LIFState, torch.Tensor]:
    """Run hard-forward surrogate LIF with generated Triton backward code."""

    if not hard_forward:
        raise ValueError("generated Triton surrogate LIF currently supports only hard_forward=True")

    final_membrane, spikes = GeneratedTritonSurrogateLIFFunction.apply(
        inputs,
        initial_state.membrane,
        params.tau_mem,
        params.threshold,
        params.reset,
        surrogate,
        surrogate_slope,
        block_size,
    )
    return LIFState(membrane=final_membrane), spikes


class LinearSurrogateLIFStreamFunction(torch.autograd.Function):
    """Stream dense projection inside surrogate LIF and accumulate weight grads directly."""

    @staticmethod
    def forward(
        ctx: Any,
        inputs: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        tau_mem: float,
        threshold: float,
        reset: float,
        surrogate: SurrogateName,
        surrogate_slope: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ctx.set_materialize_grads(False)
        ctx.tau_mem = tau_mem
        ctx.threshold = threshold
        ctx.reset = reset
        ctx.surrogate = surrogate
        ctx.surrogate_slope = surrogate_slope
        ctx.has_bias = bias is not None

        params = LIFParams(tau_mem=tau_mem, threshold=threshold, reset=reset)
        membrane = torch.zeros(
            (inputs.shape[1], weight.shape[1]),
            dtype=inputs.dtype,
            device=inputs.device,
        )
        spikes = []
        pre_reset_membranes = []
        surrogate_fn = surrogate_from_name(surrogate)
        for input_features in inputs.unbind(dim=0):
            input_current = torch.matmul(input_features, weight)
            if bias is not None:
                input_current = input_current + bias
            membrane = membrane * params.decay + input_current
            centered = surrogate_slope * (membrane - threshold)
            smooth_spike = surrogate_fn(centered)
            spike = (membrane >= threshold).to(inputs.dtype) + smooth_spike - smooth_spike.detach()
            pre_reset_membranes.append(membrane)
            spikes.append(spike)
            membrane = membrane * (1.0 - spike) + reset * spike

        saved_bias = (
            torch.empty(0, dtype=inputs.dtype, device=inputs.device) if bias is None else bias
        )
        ctx.save_for_backward(
            inputs, weight, saved_bias, torch.stack(pre_reset_membranes), torch.stack(spikes)
        )
        return membrane, torch.stack(spikes)

    @staticmethod
    def backward(
        ctx: Any,
        *grad_outputs: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        None,
        None,
        None,
        None,
        None,
    ]:
        grad_final_membrane, grad_spikes = grad_outputs
        if grad_final_membrane is None and grad_spikes is None:
            return None, None, None, None, None, None, None, None

        inputs, weight, _bias, pre_reset_membranes, spikes = ctx.saved_tensors
        needs_input_grad, needs_weight_grad, needs_bias_grad = ctx.needs_input_grad[:3]
        decay = 1.0 - (1.0 / ctx.tau_mem)

        grad_membrane = (
            torch.zeros_like(pre_reset_membranes[0])
            if grad_final_membrane is None
            else grad_final_membrane
        )
        grad_inputs = torch.empty_like(inputs) if needs_input_grad else None
        grad_weight = torch.zeros_like(weight) if needs_weight_grad else None
        grad_bias = (
            torch.zeros(
                (weight.shape[1],),
                dtype=inputs.dtype,
                device=inputs.device,
            )
            if needs_bias_grad and ctx.has_bias
            else None
        )

        for t in range(inputs.shape[0] - 1, -1, -1):
            pre_reset = pre_reset_membranes[t]
            spike = spikes[t]
            centered = ctx.surrogate_slope * (pre_reset - ctx.threshold)
            d_spike_d_membrane = ctx.surrogate_slope * surrogate_derivative(
                centered,
                ctx.surrogate,
            )
            grad_pre_reset = grad_membrane * (
                (1.0 - spike) + (ctx.reset - pre_reset) * d_spike_d_membrane
            )
            if grad_spikes is not None:
                grad_pre_reset = grad_pre_reset + grad_spikes[t] * d_spike_d_membrane

            if grad_inputs is not None:
                grad_inputs[t] = torch.matmul(grad_pre_reset, weight.t())
            if grad_weight is not None:
                grad_weight = grad_weight + torch.matmul(inputs[t].t(), grad_pre_reset)
            if grad_bias is not None:
                grad_bias = grad_bias + grad_pre_reset.sum(dim=0)
            grad_membrane = grad_pre_reset * decay

        return grad_inputs, grad_weight, grad_bias, None, None, None, None, None


def linear_surrogate_lif_stream_function(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    params: LIFParams,
    *,
    surrogate: SurrogateName = "atan",
    surrogate_slope: float = 10.0,
    hard_forward: bool = True,
) -> tuple[LIFState, torch.Tensor]:
    """Run dense projection plus surrogate LIF without materializing currents."""

    if not hard_forward:
        raise ValueError("streamed LinearSurrogateLIF currently supports only hard_forward=True")
    final_membrane, spikes = LinearSurrogateLIFStreamFunction.apply(
        inputs,
        weight,
        bias,
        params.tau_mem,
        params.threshold,
        params.reset,
        surrogate,
        surrogate_slope,
    )
    return LIFState(membrane=final_membrane), spikes


class LinearSurrogateLIFCheckpointFunction(torch.autograd.Function):
    """Stream dense projection and recompute LIF traces chunk-by-chunk in backward."""

    @staticmethod
    def forward(
        ctx: Any,
        inputs: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        tau_mem: float,
        threshold: float,
        reset: float,
        surrogate: SurrogateName,
        surrogate_slope: float,
        checkpoint_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ctx.set_materialize_grads(False)
        ctx.tau_mem = tau_mem
        ctx.threshold = threshold
        ctx.reset = reset
        ctx.surrogate = surrogate
        ctx.surrogate_slope = surrogate_slope
        ctx.has_bias = bias is not None
        ctx.checkpoint_size = checkpoint_size

        params = LIFParams(tau_mem=tau_mem, threshold=threshold, reset=reset)
        membrane = torch.zeros(
            (inputs.shape[1], weight.shape[1]),
            dtype=inputs.dtype,
            device=inputs.device,
        )
        spikes = []
        chunk_start_membranes = []
        for t, input_features in enumerate(inputs.unbind(dim=0)):
            if t % checkpoint_size == 0:
                chunk_start_membranes.append(membrane)
            input_current = torch.matmul(input_features, weight)
            if bias is not None:
                input_current = input_current + bias
            membrane = membrane * params.decay + input_current
            spike = (membrane >= threshold).to(inputs.dtype)
            spikes.append(spike)
            membrane = membrane * (1.0 - spike) + reset * spike

        saved_bias = (
            torch.empty(0, dtype=inputs.dtype, device=inputs.device) if bias is None else bias
        )
        ctx.save_for_backward(inputs, weight, saved_bias, torch.stack(chunk_start_membranes))
        return membrane, torch.stack(spikes)

    @staticmethod
    def backward(
        ctx: Any,
        *grad_outputs: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        None,
        None,
        None,
        None,
        None,
        None,
    ]:
        grad_final_membrane, grad_spikes = grad_outputs
        if grad_final_membrane is None and grad_spikes is None:
            return None, None, None, None, None, None, None, None, None

        inputs, weight, _bias, chunk_start_membranes = ctx.saved_tensors
        needs_input_grad, needs_weight_grad, needs_bias_grad = ctx.needs_input_grad[:3]
        decay = 1.0 - (1.0 / ctx.tau_mem)

        grad_membrane = (
            torch.zeros(
                (inputs.shape[1], weight.shape[1]),
                dtype=inputs.dtype,
                device=inputs.device,
            )
            if grad_final_membrane is None
            else grad_final_membrane
        )
        grad_inputs = torch.empty_like(inputs) if needs_input_grad else None
        grad_weight = torch.zeros_like(weight) if needs_weight_grad else None
        grad_bias = (
            torch.zeros((weight.shape[1],), dtype=inputs.dtype, device=inputs.device)
            if needs_bias_grad and ctx.has_bias
            else None
        )

        for chunk_index in range(chunk_start_membranes.shape[0] - 1, -1, -1):
            chunk_start = chunk_index * ctx.checkpoint_size
            chunk_end = min(chunk_start + ctx.checkpoint_size, inputs.shape[0])
            membrane = chunk_start_membranes[chunk_index]
            pre_reset_membranes = []
            spikes = []
            for t in range(chunk_start, chunk_end):
                input_current = torch.matmul(inputs[t], weight)
                if ctx.has_bias:
                    input_current = input_current + _bias
                membrane = membrane * decay + input_current
                spike = (membrane >= ctx.threshold).to(inputs.dtype)
                pre_reset_membranes.append(membrane)
                spikes.append(spike)
                membrane = membrane * (1.0 - spike) + ctx.reset * spike

            for local_t in range(chunk_end - chunk_start - 1, -1, -1):
                t = chunk_start + local_t
                pre_reset = pre_reset_membranes[local_t]
                spike = spikes[local_t]
                centered = ctx.surrogate_slope * (pre_reset - ctx.threshold)
                d_spike_d_membrane = ctx.surrogate_slope * surrogate_derivative(
                    centered,
                    ctx.surrogate,
                )
                grad_pre_reset = grad_membrane * (
                    (1.0 - spike) + (ctx.reset - pre_reset) * d_spike_d_membrane
                )
                if grad_spikes is not None:
                    grad_pre_reset = grad_pre_reset + grad_spikes[t] * d_spike_d_membrane

                if grad_inputs is not None:
                    grad_inputs[t] = torch.matmul(grad_pre_reset, weight.t())
                if grad_weight is not None:
                    grad_weight = grad_weight + torch.matmul(inputs[t].t(), grad_pre_reset)
                if grad_bias is not None:
                    grad_bias = grad_bias + grad_pre_reset.sum(dim=0)
                grad_membrane = grad_pre_reset * decay

        return grad_inputs, grad_weight, grad_bias, None, None, None, None, None, None


def linear_surrogate_lif_checkpoint_function(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    params: LIFParams,
    *,
    surrogate: SurrogateName = "atan",
    surrogate_slope: float = 10.0,
    hard_forward: bool = True,
    checkpoint_size: int = 25,
) -> tuple[LIFState, torch.Tensor]:
    """Run streamed surrogate LIF while checkpointing recurrent traces across time."""

    if not hard_forward:
        raise ValueError(
            "checkpointed LinearSurrogateLIF currently supports only hard_forward=True"
        )
    if checkpoint_size <= 0:
        raise ValueError("checkpoint_size must be positive")
    final_membrane, spikes = LinearSurrogateLIFCheckpointFunction.apply(
        inputs,
        weight,
        bias,
        params.tau_mem,
        params.threshold,
        params.reset,
        surrogate,
        surrogate_slope,
        checkpoint_size,
    )
    return LIFState(membrane=final_membrane), spikes


class LinearSurrogateLIFCheckpointRateFunction(torch.autograd.Function):
    """Torch checkpointed dense projection returning spike rates directly."""

    @staticmethod
    def forward(
        ctx: Any,
        inputs: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        tau_mem: float,
        threshold: float,
        reset: float,
        surrogate: SurrogateName,
        surrogate_slope: float,
        checkpoint_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ctx.set_materialize_grads(False)
        ctx.tau_mem = tau_mem
        ctx.threshold = threshold
        ctx.reset = reset
        ctx.surrogate = surrogate
        ctx.surrogate_slope = surrogate_slope
        ctx.has_bias = bias is not None
        ctx.checkpoint_size = checkpoint_size

        params = LIFParams(tau_mem=tau_mem, threshold=threshold, reset=reset)
        membrane = torch.zeros(
            (inputs.shape[1], weight.shape[1]),
            dtype=inputs.dtype,
            device=inputs.device,
        )
        spike_counts = torch.zeros_like(membrane)
        chunk_start_membranes = []
        for t, input_features in enumerate(inputs.unbind(dim=0)):
            if t % checkpoint_size == 0:
                chunk_start_membranes.append(membrane)
            input_current = torch.matmul(input_features, weight)
            if bias is not None:
                input_current = input_current + bias
            membrane = membrane * params.decay + input_current
            spike = (membrane >= threshold).to(inputs.dtype)
            spike_counts = spike_counts + spike
            membrane = membrane * (1.0 - spike) + reset * spike

        saved_bias = (
            torch.empty(0, dtype=inputs.dtype, device=inputs.device) if bias is None else bias
        )
        ctx.save_for_backward(inputs, weight, saved_bias, torch.stack(chunk_start_membranes))
        return membrane, spike_counts / inputs.shape[0]

    @staticmethod
    def backward(
        ctx: Any,
        *grad_outputs: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        None,
        None,
        None,
        None,
        None,
        None,
    ]:
        grad_final_membrane, grad_spike_rates = grad_outputs
        if grad_final_membrane is None and grad_spike_rates is None:
            return None, None, None, None, None, None, None, None, None

        inputs, weight, _bias, chunk_start_membranes = ctx.saved_tensors
        needs_input_grad, needs_weight_grad, needs_bias_grad = ctx.needs_input_grad[:3]
        decay = 1.0 - (1.0 / ctx.tau_mem)

        grad_membrane = (
            torch.zeros(
                (inputs.shape[1], weight.shape[1]),
                dtype=inputs.dtype,
                device=inputs.device,
            )
            if grad_final_membrane is None
            else grad_final_membrane
        )
        grad_inputs = torch.empty_like(inputs) if needs_input_grad else None
        grad_weight = torch.zeros_like(weight) if needs_weight_grad else None
        grad_bias = (
            torch.zeros((weight.shape[1],), dtype=inputs.dtype, device=inputs.device)
            if needs_bias_grad and ctx.has_bias
            else None
        )
        grad_spike_per_timestep = (
            None if grad_spike_rates is None else grad_spike_rates / inputs.shape[0]
        )

        for chunk_index in range(chunk_start_membranes.shape[0] - 1, -1, -1):
            chunk_start = chunk_index * ctx.checkpoint_size
            chunk_end = min(chunk_start + ctx.checkpoint_size, inputs.shape[0])
            membrane = chunk_start_membranes[chunk_index]
            pre_reset_membranes = []
            spikes = []
            for t in range(chunk_start, chunk_end):
                input_current = torch.matmul(inputs[t], weight)
                if ctx.has_bias:
                    input_current = input_current + _bias
                membrane = membrane * decay + input_current
                spike = (membrane >= ctx.threshold).to(inputs.dtype)
                pre_reset_membranes.append(membrane)
                spikes.append(spike)
                membrane = membrane * (1.0 - spike) + ctx.reset * spike

            for local_t in range(chunk_end - chunk_start - 1, -1, -1):
                t = chunk_start + local_t
                pre_reset = pre_reset_membranes[local_t]
                spike = spikes[local_t]
                centered = ctx.surrogate_slope * (pre_reset - ctx.threshold)
                d_spike_d_membrane = ctx.surrogate_slope * surrogate_derivative(
                    centered,
                    ctx.surrogate,
                )
                grad_pre_reset = grad_membrane * (
                    (1.0 - spike) + (ctx.reset - pre_reset) * d_spike_d_membrane
                )
                if grad_spike_per_timestep is not None:
                    grad_pre_reset = grad_pre_reset + grad_spike_per_timestep * d_spike_d_membrane

                if grad_inputs is not None:
                    grad_inputs[t] = torch.matmul(grad_pre_reset, weight.t())
                if grad_weight is not None:
                    grad_weight = grad_weight + torch.matmul(inputs[t].t(), grad_pre_reset)
                if grad_bias is not None:
                    grad_bias = grad_bias + grad_pre_reset.sum(dim=0)
                grad_membrane = grad_pre_reset * decay

        return grad_inputs, grad_weight, grad_bias, None, None, None, None, None, None


def linear_surrogate_lif_checkpoint_rate_function(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    params: LIFParams,
    *,
    surrogate: SurrogateName = "atan",
    surrogate_slope: float = 10.0,
    hard_forward: bool = True,
    checkpoint_size: int = 25,
    reduction: Literal["mean", "none"] = "mean",
) -> tuple[LIFState, torch.Tensor]:
    """Run checkpointed Torch LIF and return final state plus spike rates."""

    if not hard_forward:
        raise ValueError(
            "checkpointed Torch spike-rate LIF currently supports only hard_forward=True"
        )
    if checkpoint_size <= 0:
        raise ValueError("checkpoint_size must be positive")
    if reduction not in {"mean", "none"}:
        raise ValueError("reduction must be 'mean' or 'none'")
    final_membrane, spike_rates = LinearSurrogateLIFCheckpointRateFunction.apply(
        inputs,
        weight,
        bias,
        params.tau_mem,
        params.threshold,
        params.reset,
        surrogate,
        surrogate_slope,
        checkpoint_size,
    )
    if reduction == "mean":
        return LIFState(membrane=final_membrane), spike_rates.mean()
    return LIFState(membrane=final_membrane), spike_rates


class TwoLayerSurrogateLIFRateRecomputeFunction(torch.autograd.Function):
    """Two-layer LIF rate readout that recomputes both layers during backward."""

    @staticmethod
    def forward(
        ctx: Any,
        inputs: torch.Tensor,
        hidden_weight: torch.Tensor,
        hidden_bias: torch.Tensor | None,
        output_weight: torch.Tensor,
        output_bias: torch.Tensor | None,
        tau_mem: float,
        threshold: float,
        reset: float,
        surrogate: SurrogateName,
        surrogate_slope: float,
        checkpoint_size: int,
    ) -> torch.Tensor:
        if checkpoint_size <= 0:
            raise ValueError("checkpoint_size must be positive")
        ctx.set_materialize_grads(False)
        ctx.has_hidden_bias = hidden_bias is not None
        ctx.has_output_bias = output_bias is not None
        ctx.tau_mem = tau_mem
        ctx.threshold = threshold
        ctx.reset = reset
        ctx.surrogate = surrogate
        ctx.surrogate_slope = surrogate_slope
        ctx.checkpoint_size = checkpoint_size
        saved_hidden_bias = (
            torch.empty(0, dtype=inputs.dtype, device=inputs.device)
            if hidden_bias is None
            else hidden_bias
        )
        saved_output_bias = (
            torch.empty(0, dtype=inputs.dtype, device=inputs.device)
            if output_bias is None
            else output_bias
        )
        ctx.save_for_backward(
            inputs,
            hidden_weight,
            saved_hidden_bias,
            output_weight,
            saved_output_bias,
        )

        params = LIFParams(tau_mem=tau_mem, threshold=threshold, reset=reset)
        _hidden_state, hidden_spikes = linear_surrogate_lif_checkpoint_function(
            inputs,
            hidden_weight,
            hidden_bias,
            params,
            surrogate=surrogate,
            surrogate_slope=surrogate_slope,
            hard_forward=True,
            checkpoint_size=checkpoint_size,
        )
        _output_state, rates = linear_surrogate_lif_checkpoint_rate_function(
            hidden_spikes,
            output_weight,
            output_bias,
            params,
            surrogate=surrogate,
            surrogate_slope=surrogate_slope,
            hard_forward=True,
            checkpoint_size=checkpoint_size,
            reduction="none",
        )
        return rates

    @staticmethod
    def backward(
        ctx: Any,
        *grad_outputs: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        None,
        None,
        None,
        None,
        None,
        None,
    ]:
        (grad_rates,) = grad_outputs
        if grad_rates is None:
            return None, None, None, None, None, None, None, None, None, None, None

        inputs, hidden_weight, saved_hidden_bias, output_weight, saved_output_bias = (
            ctx.saved_tensors
        )
        hidden_bias = saved_hidden_bias if ctx.has_hidden_bias else None
        output_bias = saved_output_bias if ctx.has_output_bias else None
        needs = ctx.needs_input_grad

        replay_inputs = inputs.detach().requires_grad_(needs[0])
        replay_hidden_weight = hidden_weight.detach().requires_grad_(needs[1])
        replay_hidden_bias = (
            None if hidden_bias is None else hidden_bias.detach().requires_grad_(needs[2])
        )
        replay_output_weight = output_weight.detach().requires_grad_(needs[3])
        replay_output_bias = (
            None if output_bias is None else output_bias.detach().requires_grad_(needs[4])
        )

        grad_targets: list[torch.Tensor] = []
        target_names: list[str] = []
        for name, tensor, needed in (
            ("inputs", replay_inputs, needs[0]),
            ("hidden_weight", replay_hidden_weight, needs[1]),
            ("hidden_bias", replay_hidden_bias, needs[2]),
            ("output_weight", replay_output_weight, needs[3]),
            ("output_bias", replay_output_bias, needs[4]),
        ):
            if needed and tensor is not None:
                grad_targets.append(tensor)
                target_names.append(name)

        if not grad_targets:
            return None, None, None, None, None, None, None, None, None, None, None

        params = LIFParams(tau_mem=ctx.tau_mem, threshold=ctx.threshold, reset=ctx.reset)
        with torch.enable_grad():
            _hidden_state, hidden_spikes = linear_surrogate_lif_checkpoint_function(
                replay_inputs,
                replay_hidden_weight,
                replay_hidden_bias,
                params,
                surrogate=ctx.surrogate,
                surrogate_slope=ctx.surrogate_slope,
                hard_forward=True,
                checkpoint_size=ctx.checkpoint_size,
            )
            _output_state, rates = linear_surrogate_lif_checkpoint_rate_function(
                hidden_spikes,
                replay_output_weight,
                replay_output_bias,
                params,
                surrogate=ctx.surrogate,
                surrogate_slope=ctx.surrogate_slope,
                hard_forward=True,
                checkpoint_size=ctx.checkpoint_size,
                reduction="none",
            )
            grads = torch.autograd.grad(
                rates,
                grad_targets,
                grad_rates,
                allow_unused=True,
            )

        grad_by_name = dict(zip(target_names, grads, strict=True))
        return (
            grad_by_name.get("inputs"),
            grad_by_name.get("hidden_weight"),
            grad_by_name.get("hidden_bias"),
            grad_by_name.get("output_weight"),
            grad_by_name.get("output_bias"),
            None,
            None,
            None,
            None,
            None,
            None,
        )


def two_layer_surrogate_lif_rate_recompute_function(
    inputs: torch.Tensor,
    hidden_weight: torch.Tensor,
    hidden_bias: torch.Tensor | None,
    output_weight: torch.Tensor,
    output_bias: torch.Tensor | None,
    params: LIFParams,
    *,
    surrogate: SurrogateName = "fast_sigmoid",
    surrogate_slope: float = 5.0,
    hard_forward: bool = True,
    checkpoint_size: int = 25,
) -> torch.Tensor:
    """Return class rates for a two-layer LIF network with whole-model recompute."""

    if not hard_forward:
        raise ValueError("two-layer recompute currently supports only hard_forward=True")
    return TwoLayerSurrogateLIFRateRecomputeFunction.apply(
        inputs,
        hidden_weight,
        hidden_bias,
        output_weight,
        output_bias,
        params.tau_mem,
        params.threshold,
        params.reset,
        surrogate,
        surrogate_slope,
        checkpoint_size,
    )


class TwoLayerSurrogateLIFRatePackedHiddenFunction(torch.autograd.Function):
    """Two-layer LIF rate model that stores the hidden spike boundary packed."""

    @staticmethod
    def forward(
        ctx: Any,
        inputs: torch.Tensor,
        hidden_weight: torch.Tensor,
        hidden_bias: torch.Tensor | None,
        output_weight: torch.Tensor,
        output_bias: torch.Tensor | None,
        tau_mem: float,
        threshold: float,
        reset: float,
        surrogate: SurrogateName,
        surrogate_slope: float,
        checkpoint_size: int,
        block_b: int,
        block_n: int,
        block_f: int,
    ) -> torch.Tensor:
        from myelin.triton import (
            linear_surrogate_lif_checkpoint_packed_forward,
            linear_surrogate_lif_checkpoint_rate_forward,
            unpack_spikes_triton,
        )

        if not inputs.is_cuda:
            raise ValueError("packed-hidden two-layer rate training requires CUDA tensors")

        ctx.set_materialize_grads(False)
        ctx.tau_mem = tau_mem
        ctx.threshold = threshold
        ctx.reset = reset
        ctx.surrogate = surrogate
        ctx.surrogate_slope = surrogate_slope
        ctx.checkpoint_size = checkpoint_size
        ctx.block_b = block_b
        ctx.block_n = block_n
        ctx.block_f = block_f
        ctx.has_hidden_bias = hidden_bias is not None
        ctx.has_output_bias = output_bias is not None
        ctx.hidden_shape = (int(inputs.shape[0]), int(inputs.shape[1]), int(hidden_weight.shape[1]))

        params = LIFParams(tau_mem=tau_mem, threshold=threshold, reset=reset)
        _hidden_state, packed_hidden, hidden_chunks = (
            linear_surrogate_lif_checkpoint_packed_forward(
                inputs,
                hidden_weight,
                hidden_bias,
                params,
                checkpoint_size=checkpoint_size,
                block_b=block_b,
                block_f=block_f,
            )
        )
        hidden_spikes = unpack_spikes_triton(packed_hidden, dtype=inputs.dtype)
        _output_state, rates, output_chunks = linear_surrogate_lif_checkpoint_rate_forward(
            hidden_spikes,
            output_weight,
            output_bias,
            params,
            checkpoint_size=checkpoint_size,
            block_b=block_b,
            block_n=block_n,
            block_f=block_f,
        )
        saved_hidden_bias = (
            torch.empty(0, dtype=inputs.dtype, device=inputs.device)
            if hidden_bias is None
            else hidden_bias
        )
        saved_output_bias = (
            torch.empty(0, dtype=inputs.dtype, device=inputs.device)
            if output_bias is None
            else output_bias
        )
        ctx.save_for_backward(
            inputs,
            hidden_weight,
            saved_hidden_bias,
            output_weight,
            saved_output_bias,
            hidden_chunks,
            output_chunks,
            packed_hidden.data,
        )
        return rates

    @staticmethod
    def backward(
        ctx: Any,
        *grad_outputs: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ]:
        (grad_rates,) = grad_outputs
        if grad_rates is None:
            return (
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )

        (
            inputs,
            hidden_weight,
            hidden_bias,
            output_weight,
            output_bias,
            hidden_chunks,
            output_chunks,
            packed_hidden_data,
        ) = ctx.saved_tensors
        needs_input_grad, needs_hidden_weight_grad, needs_hidden_bias_grad = ctx.needs_input_grad[
            :3
        ]
        needs_output_weight_grad, needs_output_bias_grad = ctx.needs_input_grad[3:5]

        from myelin.packing import PackedSpikes
        from myelin.triton import linear_surrogate_lif_checkpoint_backward, unpack_spikes_triton

        params = LIFParams(
            tau_mem=ctx.tau_mem,
            threshold=ctx.threshold,
            reset=ctx.reset,
        )
        packed_hidden = PackedSpikes(
            data=packed_hidden_data,
            original_shape=ctx.hidden_shape,
        )
        hidden_spikes = unpack_spikes_triton(packed_hidden, dtype=inputs.dtype)
        grad_hidden_spikes, grad_output_weight, grad_output_bias = (
            linear_surrogate_lif_checkpoint_backward(
                hidden_spikes,
                output_weight,
                None if not ctx.has_output_bias else output_bias,
                output_chunks,
                None,
                None,
                params,
                surrogate=ctx.surrogate,
                surrogate_slope=ctx.surrogate_slope,
                grad_spike_rates=grad_rates,
                needs_input_grad=(
                    needs_input_grad or needs_hidden_weight_grad or needs_hidden_bias_grad
                ),
                needs_weight_grad=needs_output_weight_grad,
                needs_bias_grad=needs_output_bias_grad and ctx.has_output_bias,
                checkpoint_size=ctx.checkpoint_size,
                block_b=ctx.block_b,
                block_n=ctx.block_n,
                block_f=ctx.block_f,
            )
        )

        grad_inputs = None
        grad_hidden_weight = None
        grad_hidden_bias = None
        if grad_hidden_spikes is not None:
            grad_inputs, grad_hidden_weight, grad_hidden_bias = (
                linear_surrogate_lif_checkpoint_backward(
                    inputs,
                    hidden_weight,
                    None if not ctx.has_hidden_bias else hidden_bias,
                    hidden_chunks,
                    None,
                    grad_hidden_spikes,
                    params,
                    surrogate=ctx.surrogate,
                    surrogate_slope=ctx.surrogate_slope,
                    needs_input_grad=needs_input_grad,
                    needs_weight_grad=needs_hidden_weight_grad,
                    needs_bias_grad=needs_hidden_bias_grad and ctx.has_hidden_bias,
                    checkpoint_size=ctx.checkpoint_size,
                    block_b=ctx.block_b,
                    block_n=ctx.block_n,
                    block_f=ctx.block_f,
                )
            )

        return (
            grad_inputs,
            grad_hidden_weight,
            grad_hidden_bias,
            grad_output_weight,
            grad_output_bias,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def two_layer_surrogate_lif_rate_packed_hidden_function(
    inputs: torch.Tensor,
    hidden_weight: torch.Tensor,
    hidden_bias: torch.Tensor | None,
    output_weight: torch.Tensor,
    output_bias: torch.Tensor | None,
    params: LIFParams,
    *,
    surrogate: SurrogateName = "fast_sigmoid",
    surrogate_slope: float = 5.0,
    hard_forward: bool = True,
    checkpoint_size: int = 25,
    block_b: int = 16,
    block_n: int = 32,
    block_f: int = 32,
) -> torch.Tensor:
    """Return rates for a two-layer LIF network with packed hidden spike storage."""

    if not hard_forward:
        raise ValueError("packed-hidden two-layer rate training supports only hard_forward=True")
    return TwoLayerSurrogateLIFRatePackedHiddenFunction.apply(
        inputs,
        hidden_weight,
        hidden_bias,
        output_weight,
        output_bias,
        params.tau_mem,
        params.threshold,
        params.reset,
        surrogate,
        surrogate_slope,
        checkpoint_size,
        block_b,
        block_n,
        block_f,
    )


class TritonLinearSurrogateLIFFunction(torch.autograd.Function):
    """Fused Triton dense projection + LIF forward with Triton backward kernels."""

    @staticmethod
    def forward(
        ctx: Any,
        inputs: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        tau_mem: float,
        threshold: float,
        reset: float,
        surrogate: SurrogateName,
        surrogate_slope: float,
        block_b: int,
        block_n: int,
        block_f: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from myelin.triton import linear_surrogate_lif_forward

        ctx.set_materialize_grads(False)
        ctx.tau_mem = tau_mem
        ctx.threshold = threshold
        ctx.reset = reset
        ctx.surrogate = surrogate
        ctx.surrogate_slope = surrogate_slope
        ctx.has_bias = bias is not None
        ctx.block_b = block_b
        ctx.block_n = block_n
        ctx.block_f = block_f

        params = LIFParams(tau_mem=tau_mem, threshold=threshold, reset=reset)
        state, spikes, pre_reset_membranes = linear_surrogate_lif_forward(
            inputs,
            weight,
            bias,
            params,
            block_b=block_b,
            block_n=block_n,
            block_f=block_f,
        )
        saved_bias = (
            torch.empty(0, dtype=inputs.dtype, device=inputs.device) if bias is None else bias
        )
        ctx.save_for_backward(inputs, weight, saved_bias, pre_reset_membranes, spikes)
        return state.membrane, spikes

    @staticmethod
    def backward(
        ctx: Any,
        *grad_outputs: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ]:
        grad_final_membrane, grad_spikes = grad_outputs
        if grad_final_membrane is None and grad_spikes is None:
            return None, None, None, None, None, None, None, None, None, None, None

        needs_input_grad, needs_weight_grad, needs_bias_grad = ctx.needs_input_grad[:3]
        if not needs_input_grad and not needs_weight_grad and not needs_bias_grad:
            return None, None, None, None, None, None, None, None, None, None, None

        inputs, weight, _bias, pre_reset_membranes, spikes = ctx.saved_tensors
        from myelin.triton import linear_surrogate_lif_backward

        params = LIFParams(
            tau_mem=ctx.tau_mem,
            threshold=ctx.threshold,
            reset=ctx.reset,
        )
        grad_inputs, grad_weight, grad_bias = linear_surrogate_lif_backward(
            inputs,
            weight,
            pre_reset_membranes,
            spikes,
            grad_final_membrane,
            grad_spikes,
            params,
            surrogate=ctx.surrogate,
            surrogate_slope=ctx.surrogate_slope,
            needs_input_grad=needs_input_grad,
            needs_weight_grad=needs_weight_grad,
            needs_bias_grad=needs_bias_grad and ctx.has_bias,
            block_b=ctx.block_b,
            block_n=ctx.block_n,
            block_f=ctx.block_f,
        )
        return grad_inputs, grad_weight, grad_bias, None, None, None, None, None, None, None, None


def triton_linear_surrogate_lif_function(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    params: LIFParams,
    *,
    surrogate: SurrogateName = "atan",
    surrogate_slope: float = 10.0,
    hard_forward: bool = True,
    block_b: int = 16,
    block_n: int = 32,
    block_f: int = 32,
) -> tuple[LIFState, torch.Tensor]:
    """Run fused dense projection plus surrogate LIF forward through Triton."""

    if not hard_forward:
        raise ValueError("Triton LinearSurrogateLIF currently supports only hard_forward=True")
    final_membrane, spikes = TritonLinearSurrogateLIFFunction.apply(
        inputs,
        weight,
        bias,
        params.tau_mem,
        params.threshold,
        params.reset,
        surrogate,
        surrogate_slope,
        block_b,
        block_n,
        block_f,
    )
    return LIFState(membrane=final_membrane), spikes


class GeneratedTritonLinearSurrogateLIFFunction(torch.autograd.Function):
    """Fused Triton dense projection + generated dweight/dbias backward."""

    @staticmethod
    def forward(
        ctx: Any,
        inputs: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        tau_mem: float,
        threshold: float,
        reset: float,
        surrogate: SurrogateName,
        surrogate_slope: float,
        block_b: int,
        block_n: int,
        block_f: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from myelin.triton import linear_surrogate_lif_forward, pack_spikes_triton

        ctx.set_materialize_grads(False)
        ctx.tau_mem = tau_mem
        ctx.threshold = threshold
        ctx.reset = reset
        ctx.surrogate = surrogate
        ctx.surrogate_slope = surrogate_slope
        ctx.has_bias = bias is not None
        ctx.block_b = block_b
        ctx.block_n = block_n
        ctx.block_f = block_f

        params = LIFParams(tau_mem=tau_mem, threshold=threshold, reset=reset)
        state, spikes, pre_reset_membranes = linear_surrogate_lif_forward(
            inputs,
            weight,
            bias,
            params,
            block_b=block_b,
            block_n=block_n,
            block_f=block_f,
        )
        saved_bias = (
            torch.empty(0, dtype=inputs.dtype, device=inputs.device) if bias is None else bias
        )
        packed_spikes = pack_spikes_triton(spikes)
        ctx.save_for_backward(inputs, weight, saved_bias, pre_reset_membranes, packed_spikes.data)
        return state.membrane, spikes

    @staticmethod
    def backward(
        ctx: Any,
        *grad_outputs: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ]:
        grad_final_membrane, grad_spikes = grad_outputs
        if grad_final_membrane is None and grad_spikes is None:
            return None, None, None, None, None, None, None, None, None, None, None

        needs_input_grad, needs_weight_grad, needs_bias_grad = ctx.needs_input_grad[:3]
        if not needs_input_grad and not needs_weight_grad and not needs_bias_grad:
            return None, None, None, None, None, None, None, None, None, None, None

        inputs, weight, _bias, pre_reset_membranes, packed_spikes = ctx.saved_tensors
        from myelin.triton import (
            generated_linear_lif_surrogate_backward_weight_bias_packed_spikes,
        )

        params = LIFParams(
            tau_mem=ctx.tau_mem,
            threshold=ctx.threshold,
            reset=ctx.reset,
        )
        grad_inputs, grad_weight, grad_bias = (
            generated_linear_lif_surrogate_backward_weight_bias_packed_spikes(
                inputs,
                weight,
                pre_reset_membranes,
                packed_spikes,
                grad_final_membrane,
                grad_spikes,
                params,
                surrogate=ctx.surrogate,
                surrogate_slope=ctx.surrogate_slope,
                needs_input_grad=needs_input_grad,
                needs_weight_grad=needs_weight_grad,
                needs_bias_grad=needs_bias_grad and ctx.has_bias,
                block_b=ctx.block_b,
                block_n=ctx.block_n,
                block_f=ctx.block_f,
            )
        )
        return grad_inputs, grad_weight, grad_bias, None, None, None, None, None, None, None, None


def generated_triton_linear_surrogate_lif_function(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    params: LIFParams,
    *,
    surrogate: SurrogateName = "atan",
    surrogate_slope: float = 10.0,
    hard_forward: bool = True,
    block_b: int = 16,
    block_n: int = 32,
    block_f: int = 32,
) -> tuple[LIFState, torch.Tensor]:
    """Run fused dense projection plus generated surrogate LIF backward through Triton."""

    if not hard_forward:
        raise ValueError(
            "generated Triton LinearSurrogateLIF currently supports only hard_forward=True"
        )
    final_membrane, spikes = GeneratedTritonLinearSurrogateLIFFunction.apply(
        inputs,
        weight,
        bias,
        params.tau_mem,
        params.threshold,
        params.reset,
        surrogate,
        surrogate_slope,
        block_b,
        block_n,
        block_f,
    )
    return LIFState(membrane=final_membrane), spikes


class TritonLinearSurrogateLIFCheckpointFunction(torch.autograd.Function):
    """Checkpointed fused Triton dense projection + LIF training."""

    @staticmethod
    def forward(
        ctx: Any,
        inputs: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        tau_mem: float,
        threshold: float,
        reset: float,
        surrogate: SurrogateName,
        surrogate_slope: float,
        checkpoint_size: int,
        block_b: int,
        block_n: int,
        block_f: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from myelin.triton import linear_surrogate_lif_checkpoint_forward

        ctx.set_materialize_grads(False)
        ctx.tau_mem = tau_mem
        ctx.threshold = threshold
        ctx.reset = reset
        ctx.surrogate = surrogate
        ctx.surrogate_slope = surrogate_slope
        ctx.has_bias = bias is not None
        ctx.checkpoint_size = checkpoint_size
        ctx.block_b = block_b
        ctx.block_n = block_n
        ctx.block_f = block_f

        params = LIFParams(tau_mem=tau_mem, threshold=threshold, reset=reset)
        state, spikes, chunk_start_membranes = linear_surrogate_lif_checkpoint_forward(
            inputs,
            weight,
            bias,
            params,
            checkpoint_size=checkpoint_size,
            block_b=block_b,
            block_n=block_n,
            block_f=block_f,
        )
        saved_bias = (
            torch.empty(0, dtype=inputs.dtype, device=inputs.device) if bias is None else bias
        )
        ctx.save_for_backward(inputs, weight, saved_bias, chunk_start_membranes)
        return state.membrane, spikes

    @staticmethod
    def backward(
        ctx: Any,
        *grad_outputs: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ]:
        grad_final_membrane, grad_spikes = grad_outputs
        if grad_final_membrane is None and grad_spikes is None:
            return None, None, None, None, None, None, None, None, None, None, None, None

        needs_input_grad, needs_weight_grad, needs_bias_grad = ctx.needs_input_grad[:3]
        if not needs_input_grad and not needs_weight_grad and not needs_bias_grad:
            return None, None, None, None, None, None, None, None, None, None, None, None

        inputs, weight, bias, chunk_start_membranes = ctx.saved_tensors
        from myelin.triton import linear_surrogate_lif_checkpoint_backward

        params = LIFParams(
            tau_mem=ctx.tau_mem,
            threshold=ctx.threshold,
            reset=ctx.reset,
        )
        grad_inputs, grad_weight, grad_bias = linear_surrogate_lif_checkpoint_backward(
            inputs,
            weight,
            None if not ctx.has_bias else bias,
            chunk_start_membranes,
            grad_final_membrane,
            grad_spikes,
            params,
            surrogate=ctx.surrogate,
            surrogate_slope=ctx.surrogate_slope,
            needs_input_grad=needs_input_grad,
            needs_weight_grad=needs_weight_grad,
            needs_bias_grad=needs_bias_grad and ctx.has_bias,
            checkpoint_size=ctx.checkpoint_size,
            block_b=ctx.block_b,
            block_n=ctx.block_n,
            block_f=ctx.block_f,
        )
        return (
            grad_inputs,
            grad_weight,
            grad_bias,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def triton_linear_surrogate_lif_checkpoint_function(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    params: LIFParams,
    *,
    surrogate: SurrogateName = "atan",
    surrogate_slope: float = 10.0,
    hard_forward: bool = True,
    checkpoint_size: int = 25,
    block_b: int = 16,
    block_n: int = 32,
    block_f: int = 32,
) -> tuple[LIFState, torch.Tensor]:
    """Run checkpointed fused dense projection plus surrogate LIF through Triton."""

    if not hard_forward:
        raise ValueError(
            "checkpointed Triton LinearSurrogateLIF currently supports only hard_forward=True"
        )
    if checkpoint_size <= 0:
        raise ValueError("checkpoint_size must be positive")
    final_membrane, spikes = TritonLinearSurrogateLIFCheckpointFunction.apply(
        inputs,
        weight,
        bias,
        params.tau_mem,
        params.threshold,
        params.reset,
        surrogate,
        surrogate_slope,
        checkpoint_size,
        block_b,
        block_n,
        block_f,
    )
    return LIFState(membrane=final_membrane), spikes


class TritonLinearSurrogateLIFCheckpointRateFunction(torch.autograd.Function):
    """Checkpointed fused Triton dense projection returning spike rate, not spikes."""

    @staticmethod
    def forward(
        ctx: Any,
        inputs: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        tau_mem: float,
        threshold: float,
        reset: float,
        surrogate: SurrogateName,
        surrogate_slope: float,
        checkpoint_size: int,
        block_b: int,
        block_n: int,
        block_f: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from myelin.triton import linear_surrogate_lif_checkpoint_rate_forward

        ctx.set_materialize_grads(False)
        ctx.tau_mem = tau_mem
        ctx.threshold = threshold
        ctx.reset = reset
        ctx.surrogate = surrogate
        ctx.surrogate_slope = surrogate_slope
        ctx.has_bias = bias is not None
        ctx.checkpoint_size = checkpoint_size
        ctx.block_b = block_b
        ctx.block_n = block_n
        ctx.block_f = block_f

        params = LIFParams(tau_mem=tau_mem, threshold=threshold, reset=reset)
        state, spike_rates, chunk_start_membranes = linear_surrogate_lif_checkpoint_rate_forward(
            inputs,
            weight,
            bias,
            params,
            checkpoint_size=checkpoint_size,
            block_b=block_b,
            block_n=block_n,
            block_f=block_f,
        )
        saved_bias = (
            torch.empty(0, dtype=inputs.dtype, device=inputs.device) if bias is None else bias
        )
        ctx.save_for_backward(inputs, weight, saved_bias, chunk_start_membranes)
        return state.membrane, spike_rates

    @staticmethod
    def backward(
        ctx: Any,
        *grad_outputs: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ]:
        grad_final_membrane, grad_spike_rates = grad_outputs
        if grad_final_membrane is None and grad_spike_rates is None:
            return None, None, None, None, None, None, None, None, None, None, None, None

        needs_input_grad, needs_weight_grad, needs_bias_grad = ctx.needs_input_grad[:3]
        if not needs_input_grad and not needs_weight_grad and not needs_bias_grad:
            return None, None, None, None, None, None, None, None, None, None, None, None

        inputs, weight, bias, chunk_start_membranes = ctx.saved_tensors
        from myelin.triton import linear_surrogate_lif_checkpoint_backward

        params = LIFParams(
            tau_mem=ctx.tau_mem,
            threshold=ctx.threshold,
            reset=ctx.reset,
        )
        grad_inputs, grad_weight, grad_bias = linear_surrogate_lif_checkpoint_backward(
            inputs,
            weight,
            None if not ctx.has_bias else bias,
            chunk_start_membranes,
            grad_final_membrane,
            None,
            params,
            surrogate=ctx.surrogate,
            surrogate_slope=ctx.surrogate_slope,
            grad_spike_rates=grad_spike_rates,
            needs_input_grad=needs_input_grad,
            needs_weight_grad=needs_weight_grad,
            needs_bias_grad=needs_bias_grad and ctx.has_bias,
            checkpoint_size=ctx.checkpoint_size,
            block_b=ctx.block_b,
            block_n=ctx.block_n,
            block_f=ctx.block_f,
        )
        return (
            grad_inputs,
            grad_weight,
            grad_bias,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def triton_linear_surrogate_lif_checkpoint_rate_function(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    params: LIFParams,
    *,
    surrogate: SurrogateName = "atan",
    surrogate_slope: float = 10.0,
    hard_forward: bool = True,
    checkpoint_size: int = 25,
    block_b: int = 16,
    block_n: int = 32,
    block_f: int = 32,
    reduction: Literal["mean", "none"] = "mean",
) -> tuple[LIFState, torch.Tensor]:
    """Run checkpointed Triton LIF and return final state plus spike rates."""

    if not hard_forward:
        raise ValueError(
            "checkpointed Triton spike-rate LIF currently supports only hard_forward=True"
        )
    if checkpoint_size <= 0:
        raise ValueError("checkpoint_size must be positive")
    if reduction not in {"mean", "none"}:
        raise ValueError("reduction must be 'mean' or 'none'")
    final_membrane, spike_rates = TritonLinearSurrogateLIFCheckpointRateFunction.apply(
        inputs,
        weight,
        bias,
        params.tau_mem,
        params.threshold,
        params.reset,
        surrogate,
        surrogate_slope,
        checkpoint_size,
        block_b,
        block_n,
        block_f,
    )
    if reduction == "mean":
        return LIFState(membrane=final_membrane), spike_rates.mean()
    return LIFState(membrane=final_membrane), spike_rates


class GeneratedTritonLinearSurrogateLIFCheckpointRateFunction(torch.autograd.Function):
    """Checkpointed fused Triton rate readout with generated backward chunk code."""

    @staticmethod
    def forward(
        ctx: Any,
        inputs: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        tau_mem: float,
        threshold: float,
        reset: float,
        surrogate: SurrogateName,
        surrogate_slope: float,
        checkpoint_size: int,
        block_b: int,
        block_n: int,
        block_f: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from myelin.triton import linear_surrogate_lif_checkpoint_rate_forward

        ctx.set_materialize_grads(False)
        ctx.tau_mem = tau_mem
        ctx.threshold = threshold
        ctx.reset = reset
        ctx.surrogate = surrogate
        ctx.surrogate_slope = surrogate_slope
        ctx.has_bias = bias is not None
        ctx.checkpoint_size = checkpoint_size
        ctx.block_b = block_b
        ctx.block_n = block_n
        ctx.block_f = block_f

        params = LIFParams(tau_mem=tau_mem, threshold=threshold, reset=reset)
        state, spike_rates, chunk_start_membranes = linear_surrogate_lif_checkpoint_rate_forward(
            inputs,
            weight,
            bias,
            params,
            checkpoint_size=checkpoint_size,
            block_b=block_b,
            block_n=block_n,
            block_f=block_f,
        )
        saved_bias = (
            torch.empty(0, dtype=inputs.dtype, device=inputs.device) if bias is None else bias
        )
        ctx.save_for_backward(inputs, weight, saved_bias, chunk_start_membranes)
        return state.membrane, spike_rates

    @staticmethod
    def backward(
        ctx: Any,
        *grad_outputs: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ]:
        grad_final_membrane, grad_spike_rates = grad_outputs
        if grad_final_membrane is None and grad_spike_rates is None:
            return None, None, None, None, None, None, None, None, None, None, None, None

        needs_input_grad, needs_weight_grad, needs_bias_grad = ctx.needs_input_grad[:3]
        if not needs_input_grad and not needs_weight_grad and not needs_bias_grad:
            return None, None, None, None, None, None, None, None, None, None, None, None

        inputs, weight, bias, chunk_start_membranes = ctx.saved_tensors
        from myelin.triton import generated_linear_lif_surrogate_checkpoint_backward

        params = LIFParams(
            tau_mem=ctx.tau_mem,
            threshold=ctx.threshold,
            reset=ctx.reset,
        )
        grad_inputs, grad_weight, grad_bias = generated_linear_lif_surrogate_checkpoint_backward(
            inputs,
            weight,
            None if not ctx.has_bias else bias,
            chunk_start_membranes,
            grad_final_membrane,
            None,
            params,
            surrogate=ctx.surrogate,
            surrogate_slope=ctx.surrogate_slope,
            grad_spike_rates=grad_spike_rates,
            needs_input_grad=needs_input_grad,
            needs_weight_grad=needs_weight_grad,
            needs_bias_grad=needs_bias_grad and ctx.has_bias,
            checkpoint_size=ctx.checkpoint_size,
            block_b=ctx.block_b,
            block_n=ctx.block_n,
            block_f=ctx.block_f,
        )
        return (
            grad_inputs,
            grad_weight,
            grad_bias,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def generated_triton_linear_surrogate_lif_checkpoint_rate_function(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    params: LIFParams,
    *,
    surrogate: SurrogateName = "atan",
    surrogate_slope: float = 10.0,
    hard_forward: bool = True,
    checkpoint_size: int = 25,
    block_b: int = 16,
    block_n: int = 32,
    block_f: int = 32,
    reduction: Literal["mean", "none"] = "mean",
) -> tuple[LIFState, torch.Tensor]:
    """Run generated checkpointed Triton LIF and return final state plus spike rates."""

    if not hard_forward:
        raise ValueError(
            "generated checkpointed Triton spike-rate LIF currently supports only hard_forward=True"
        )
    if checkpoint_size <= 0:
        raise ValueError("checkpoint_size must be positive")
    if reduction not in {"mean", "none"}:
        raise ValueError("reduction must be 'mean' or 'none'")
    final_membrane, spike_rates = GeneratedTritonLinearSurrogateLIFCheckpointRateFunction.apply(
        inputs,
        weight,
        bias,
        params.tau_mem,
        params.threshold,
        params.reset,
        surrogate,
        surrogate_slope,
        checkpoint_size,
        block_b,
        block_n,
        block_f,
    )
    if reduction == "mean":
        return LIFState(membrane=final_membrane), spike_rates.mean()
    return LIFState(membrane=final_membrane), spike_rates


class GeneratedTritonLinearSurrogateLIFCheckpointFunction(torch.autograd.Function):
    """Checkpointed fused Triton dense projection with generated backward chunk code."""

    @staticmethod
    def forward(
        ctx: Any,
        inputs: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        tau_mem: float,
        threshold: float,
        reset: float,
        surrogate: SurrogateName,
        surrogate_slope: float,
        checkpoint_size: int,
        block_b: int,
        block_n: int,
        block_f: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from myelin.triton import linear_surrogate_lif_checkpoint_forward

        ctx.set_materialize_grads(False)
        ctx.tau_mem = tau_mem
        ctx.threshold = threshold
        ctx.reset = reset
        ctx.surrogate = surrogate
        ctx.surrogate_slope = surrogate_slope
        ctx.has_bias = bias is not None
        ctx.checkpoint_size = checkpoint_size
        ctx.block_b = block_b
        ctx.block_n = block_n
        ctx.block_f = block_f

        params = LIFParams(tau_mem=tau_mem, threshold=threshold, reset=reset)
        state, spikes, chunk_start_membranes = linear_surrogate_lif_checkpoint_forward(
            inputs,
            weight,
            bias,
            params,
            checkpoint_size=checkpoint_size,
            block_b=block_b,
            block_n=block_n,
            block_f=block_f,
        )
        saved_bias = (
            torch.empty(0, dtype=inputs.dtype, device=inputs.device) if bias is None else bias
        )
        ctx.save_for_backward(inputs, weight, saved_bias, chunk_start_membranes)
        return state.membrane, spikes

    @staticmethod
    def backward(
        ctx: Any,
        *grad_outputs: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ]:
        grad_final_membrane, grad_spikes = grad_outputs
        if grad_final_membrane is None and grad_spikes is None:
            return None, None, None, None, None, None, None, None, None, None, None, None

        needs_input_grad, needs_weight_grad, needs_bias_grad = ctx.needs_input_grad[:3]
        if not needs_input_grad and not needs_weight_grad and not needs_bias_grad:
            return None, None, None, None, None, None, None, None, None, None, None, None

        inputs, weight, bias, chunk_start_membranes = ctx.saved_tensors
        from myelin.triton import generated_linear_lif_surrogate_checkpoint_backward

        params = LIFParams(
            tau_mem=ctx.tau_mem,
            threshold=ctx.threshold,
            reset=ctx.reset,
        )
        grad_inputs, grad_weight, grad_bias = generated_linear_lif_surrogate_checkpoint_backward(
            inputs,
            weight,
            None if not ctx.has_bias else bias,
            chunk_start_membranes,
            grad_final_membrane,
            grad_spikes,
            params,
            surrogate=ctx.surrogate,
            surrogate_slope=ctx.surrogate_slope,
            needs_input_grad=needs_input_grad,
            needs_weight_grad=needs_weight_grad,
            needs_bias_grad=needs_bias_grad and ctx.has_bias,
            checkpoint_size=ctx.checkpoint_size,
            block_b=ctx.block_b,
            block_n=ctx.block_n,
            block_f=ctx.block_f,
        )
        return (
            grad_inputs,
            grad_weight,
            grad_bias,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def generated_triton_linear_surrogate_lif_checkpoint_function(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    params: LIFParams,
    *,
    surrogate: SurrogateName = "atan",
    surrogate_slope: float = 10.0,
    hard_forward: bool = True,
    checkpoint_size: int = 25,
    block_b: int = 16,
    block_n: int = 32,
    block_f: int = 32,
) -> tuple[LIFState, torch.Tensor]:
    """Run generated checkpointed dense projection plus surrogate LIF through Triton."""

    if not hard_forward:
        raise ValueError(
            "generated checkpointed Triton LinearSurrogateLIF currently supports only "
            "hard_forward=True"
        )
    if checkpoint_size <= 0:
        raise ValueError("checkpoint_size must be positive")
    final_membrane, spikes = GeneratedTritonLinearSurrogateLIFCheckpointFunction.apply(
        inputs,
        weight,
        bias,
        params.tau_mem,
        params.threshold,
        params.reset,
        surrogate,
        surrogate_slope,
        checkpoint_size,
        block_b,
        block_n,
        block_f,
    )
    return LIFState(membrane=final_membrane), spikes


def _linear_surrogate_stream_backward(
    ctx: Any,
    *grad_outputs: torch.Tensor | None,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    grad_final_membrane, grad_spikes = grad_outputs
    if grad_final_membrane is None and grad_spikes is None:
        return None, None, None

    inputs, weight, _bias, pre_reset_membranes, spikes = ctx.saved_tensors
    needs_input_grad, needs_weight_grad, needs_bias_grad = ctx.needs_input_grad[:3]
    decay = 1.0 - (1.0 / ctx.tau_mem)

    grad_membrane = (
        torch.zeros_like(pre_reset_membranes[0])
        if grad_final_membrane is None
        else grad_final_membrane
    )
    grad_inputs = torch.empty_like(inputs) if needs_input_grad else None
    grad_weight = torch.zeros_like(weight) if needs_weight_grad else None
    grad_bias = (
        torch.zeros(
            (weight.shape[1],),
            dtype=inputs.dtype,
            device=inputs.device,
        )
        if needs_bias_grad and ctx.has_bias
        else None
    )

    for t in range(inputs.shape[0] - 1, -1, -1):
        pre_reset = pre_reset_membranes[t]
        spike = spikes[t]
        centered = ctx.surrogate_slope * (pre_reset - ctx.threshold)
        d_spike_d_membrane = ctx.surrogate_slope * surrogate_derivative(
            centered,
            ctx.surrogate,
        )
        grad_pre_reset = grad_membrane * (
            (1.0 - spike) + (ctx.reset - pre_reset) * d_spike_d_membrane
        )
        if grad_spikes is not None:
            grad_pre_reset = grad_pre_reset + grad_spikes[t] * d_spike_d_membrane

        if grad_inputs is not None:
            grad_inputs[t] = torch.matmul(grad_pre_reset, weight.t())
        if grad_weight is not None:
            grad_weight = grad_weight + torch.matmul(inputs[t].t(), grad_pre_reset)
        if grad_bias is not None:
            grad_bias = grad_bias + grad_pre_reset.sum(dim=0)
        grad_membrane = grad_pre_reset * decay

    return grad_inputs, grad_weight, grad_bias
