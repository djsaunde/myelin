"""Compile-visible Triton operators.

These wrappers expose selected Triton kernels through ``torch.library.triton_op``
so ``torch.compile`` can trace the allocation and launch boundary. They are
experimental and intentionally narrow; public dispatch only uses them for the
explicit ``backend="triton_compile"`` rate-training path.
"""

from __future__ import annotations

from typing import Any, cast

import torch
from torch.library import custom_op, triton_op, wrap_triton

from myelin._optional import require_triton
from myelin.neurons import LIFParams
from myelin.triton.lif import _linear_surrogate_lif_checkpoint_rate_forward_kernel

triton = require_triton()


@triton_op("myelin::linear_lif_checkpoint_rate_forward_no_bias", mutates_args={})
def linear_lif_checkpoint_rate_forward_no_bias_op(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    decay: float,
    threshold: float,
    reset: float,
    surrogate_slope: float,
    checkpoint_size: int,
    block_b: int,
    block_n: int,
    block_f: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the no-bias checkpointed rate forward as a compile-visible op.

    Args:
        inputs: Contiguous or non-contiguous CUDA tensor shaped ``[T, B, F]``.
        weight: Contiguous or non-contiguous CUDA tensor shaped ``[F, N]``.
        decay: LIF membrane decay.
        threshold: Spike threshold.
        reset: Reset membrane value after a spike.
        surrogate_slope: Surrogate derivative slope used by the registered backward.
        checkpoint_size: Timesteps per backward-recompute checkpoint.
        block_b: Batch tile size.
        block_n: Neuron tile size.
        block_f: Feature reduction tile size.

    Returns:
        ``(final_membrane, spike_rates, chunk_start_membranes)``.
    """

    del surrogate_slope
    contiguous_inputs = inputs.contiguous()
    contiguous_weight = weight.contiguous()
    timesteps = contiguous_inputs.shape[0]
    batch = contiguous_inputs.shape[1]
    features = contiguous_inputs.shape[2]
    neurons = contiguous_weight.shape[1]
    num_chunks = (timesteps + checkpoint_size - 1) // checkpoint_size
    final_membrane = torch.empty((batch, neurons), dtype=inputs.dtype, device=inputs.device)
    spike_rates = torch.empty((batch, neurons), dtype=inputs.dtype, device=inputs.device)
    chunk_start_membranes = torch.empty(
        (num_chunks, batch, neurons), dtype=inputs.dtype, device=inputs.device
    )

    def grid(meta: dict[str, int]):
        return (
            triton.cdiv(batch, meta["block_b"]),
            triton.cdiv(neurons, meta["block_n"]),
        )

    wrap_triton(_linear_surrogate_lif_checkpoint_rate_forward_kernel)[grid](
        contiguous_inputs,
        contiguous_weight,
        contiguous_weight,
        final_membrane,
        spike_rates,
        chunk_start_membranes,
        timesteps,
        batch,
        features,
        neurons,
        decay,
        threshold,
        reset,
        False,
        checkpoint_size,
        block_b,
        block_n,
        block_f,
    )
    return final_membrane, spike_rates, chunk_start_membranes


@triton_op("myelin::linear_lif_checkpoint_rate_forward_bias", mutates_args={})
def linear_lif_checkpoint_rate_forward_bias_op(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    decay: float,
    threshold: float,
    reset: float,
    surrogate_slope: float,
    checkpoint_size: int,
    block_b: int,
    block_n: int,
    block_f: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the bias checkpointed rate forward as a compile-visible op."""

    del surrogate_slope
    contiguous_inputs = inputs.contiguous()
    contiguous_weight = weight.contiguous()
    contiguous_bias = bias.contiguous()
    timesteps = contiguous_inputs.shape[0]
    batch = contiguous_inputs.shape[1]
    features = contiguous_inputs.shape[2]
    neurons = contiguous_weight.shape[1]
    num_chunks = (timesteps + checkpoint_size - 1) // checkpoint_size
    final_membrane = torch.empty((batch, neurons), dtype=inputs.dtype, device=inputs.device)
    spike_rates = torch.empty((batch, neurons), dtype=inputs.dtype, device=inputs.device)
    chunk_start_membranes = torch.empty(
        (num_chunks, batch, neurons), dtype=inputs.dtype, device=inputs.device
    )

    def grid(meta: dict[str, int]):
        return (
            triton.cdiv(batch, meta["block_b"]),
            triton.cdiv(neurons, meta["block_n"]),
        )

    wrap_triton(_linear_surrogate_lif_checkpoint_rate_forward_kernel)[grid](
        contiguous_inputs,
        contiguous_weight,
        contiguous_bias,
        final_membrane,
        spike_rates,
        chunk_start_membranes,
        timesteps,
        batch,
        features,
        neurons,
        decay,
        threshold,
        reset,
        True,
        checkpoint_size,
        block_b,
        block_n,
        block_f,
    )
    return final_membrane, spike_rates, chunk_start_membranes


@custom_op("myelin::linear_lif_checkpoint_rate_grads_no_bias", mutates_args={})
def linear_lif_checkpoint_rate_grads_no_bias_op(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    chunk_start_membranes: torch.Tensor,
    grad_final_membrane: torch.Tensor,
    grad_spike_rates: torch.Tensor,
    decay: float,
    threshold: float,
    reset: float,
    surrogate_slope: float,
    checkpoint_size: int,
    block_b: int,
    block_n: int,
    block_f: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the no-bias checkpointed rate backward and return ``(dinput, dweight)``."""

    from myelin.triton import linear_surrogate_lif_checkpoint_backward

    tau_mem = 1.0 / (1.0 - decay)
    params = LIFParams(tau_mem=tau_mem, threshold=threshold, reset=reset)
    grad_inputs, grad_weight, _grad_bias = linear_surrogate_lif_checkpoint_backward(
        inputs,
        weight,
        None,
        chunk_start_membranes,
        grad_final_membrane,
        None,
        params,
        surrogate="fast_sigmoid",
        surrogate_slope=surrogate_slope,
        grad_spike_rates=grad_spike_rates,
        needs_input_grad=True,
        needs_weight_grad=True,
        needs_bias_grad=False,
        checkpoint_size=checkpoint_size,
        block_b=block_b,
        block_n=block_n,
        block_f=block_f,
    )
    if grad_inputs is None or grad_weight is None:
        raise RuntimeError("expected input and weight gradients from Triton checkpoint backward")
    return grad_inputs, grad_weight


@custom_op("myelin::linear_lif_checkpoint_rate_grads_bias", mutates_args={})
def linear_lif_checkpoint_rate_grads_bias_op(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    chunk_start_membranes: torch.Tensor,
    grad_final_membrane: torch.Tensor,
    grad_spike_rates: torch.Tensor,
    decay: float,
    threshold: float,
    reset: float,
    surrogate_slope: float,
    checkpoint_size: int,
    block_b: int,
    block_n: int,
    block_f: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the bias checkpointed rate backward and return ``(dinput, dweight, dbias)``."""

    from myelin.triton import linear_surrogate_lif_checkpoint_backward

    tau_mem = 1.0 / (1.0 - decay)
    params = LIFParams(tau_mem=tau_mem, threshold=threshold, reset=reset)
    grad_inputs, grad_weight, grad_bias = linear_surrogate_lif_checkpoint_backward(
        inputs,
        weight,
        bias,
        chunk_start_membranes,
        grad_final_membrane,
        None,
        params,
        surrogate="fast_sigmoid",
        surrogate_slope=surrogate_slope,
        grad_spike_rates=grad_spike_rates,
        needs_input_grad=True,
        needs_weight_grad=True,
        needs_bias_grad=True,
        checkpoint_size=checkpoint_size,
        block_b=block_b,
        block_n=block_n,
        block_f=block_f,
    )
    if grad_inputs is None or grad_weight is None or grad_bias is None:
        raise RuntimeError(
            "expected input, weight, and bias gradients from Triton checkpoint backward"
        )
    return grad_inputs, grad_weight, grad_bias


@linear_lif_checkpoint_rate_grads_no_bias_op.register_fake
def _linear_lif_checkpoint_rate_grads_no_bias_fake(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    chunk_start_membranes: torch.Tensor,
    grad_final_membrane: torch.Tensor,
    grad_spike_rates: torch.Tensor,
    decay: float,
    threshold: float,
    reset: float,
    surrogate_slope: float,
    checkpoint_size: int,
    block_b: int,
    block_n: int,
    block_f: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    del (
        chunk_start_membranes,
        grad_final_membrane,
        grad_spike_rates,
        decay,
        threshold,
        reset,
        surrogate_slope,
        checkpoint_size,
        block_b,
        block_n,
        block_f,
    )
    return torch.empty_like(inputs), torch.empty_like(weight)


@linear_lif_checkpoint_rate_grads_bias_op.register_fake
def _linear_lif_checkpoint_rate_grads_bias_fake(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    chunk_start_membranes: torch.Tensor,
    grad_final_membrane: torch.Tensor,
    grad_spike_rates: torch.Tensor,
    decay: float,
    threshold: float,
    reset: float,
    surrogate_slope: float,
    checkpoint_size: int,
    block_b: int,
    block_n: int,
    block_f: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del (
        chunk_start_membranes,
        grad_final_membrane,
        grad_spike_rates,
        decay,
        threshold,
        reset,
        surrogate_slope,
        checkpoint_size,
        block_b,
        block_n,
        block_f,
    )
    return torch.empty_like(inputs), torch.empty_like(weight), torch.empty_like(bias)


@cast(Any, linear_lif_checkpoint_rate_forward_no_bias_op).register_fake
def _linear_lif_checkpoint_rate_forward_no_bias_fake(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    decay: float,
    threshold: float,
    reset: float,
    surrogate_slope: float,
    checkpoint_size: int,
    block_b: int,
    block_n: int,
    block_f: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del decay, threshold, reset, surrogate_slope, block_b, block_n, block_f
    timesteps = inputs.shape[0]
    batch = inputs.shape[1]
    neurons = weight.shape[1]
    num_chunks = (timesteps + checkpoint_size - 1) // checkpoint_size
    final_membrane = torch.empty((batch, neurons), dtype=inputs.dtype, device=inputs.device)
    spike_rates = torch.empty((batch, neurons), dtype=inputs.dtype, device=inputs.device)
    chunk_start_membranes = torch.empty(
        (num_chunks, batch, neurons), dtype=inputs.dtype, device=inputs.device
    )
    return final_membrane, spike_rates, chunk_start_membranes


@cast(Any, linear_lif_checkpoint_rate_forward_bias_op).register_fake
def _linear_lif_checkpoint_rate_forward_bias_fake(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    decay: float,
    threshold: float,
    reset: float,
    surrogate_slope: float,
    checkpoint_size: int,
    block_b: int,
    block_n: int,
    block_f: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del bias, decay, threshold, reset, surrogate_slope, block_b, block_n, block_f
    timesteps = inputs.shape[0]
    batch = inputs.shape[1]
    neurons = weight.shape[1]
    num_chunks = (timesteps + checkpoint_size - 1) // checkpoint_size
    final_membrane = torch.empty((batch, neurons), dtype=inputs.dtype, device=inputs.device)
    spike_rates = torch.empty((batch, neurons), dtype=inputs.dtype, device=inputs.device)
    chunk_start_membranes = torch.empty(
        (num_chunks, batch, neurons), dtype=inputs.dtype, device=inputs.device
    )
    return final_membrane, spike_rates, chunk_start_membranes


def _linear_lif_checkpoint_rate_forward_no_bias_setup_context(
    ctx: Any,
    inputs: tuple[
        torch.Tensor,
        torch.Tensor,
        float,
        float,
        float,
        float,
        int,
        int,
        int,
        int,
    ],
    output: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    (
        input_values,
        weight,
        decay,
        threshold,
        reset,
        surrogate_slope,
        checkpoint_size,
        block_b,
        block_n,
        block_f,
    ) = inputs
    _final_membrane, _spike_rates, chunk_start_membranes = output
    ctx.save_for_backward(input_values, weight, chunk_start_membranes)
    ctx.decay = decay
    ctx.threshold = threshold
    ctx.reset = reset
    ctx.surrogate_slope = surrogate_slope
    ctx.checkpoint_size = checkpoint_size
    ctx.block_b = block_b
    ctx.block_n = block_n
    ctx.block_f = block_f


def _linear_lif_checkpoint_rate_forward_no_bias_backward(
    ctx: Any,
    grad_final_membrane: torch.Tensor | None,
    grad_spike_rates: torch.Tensor | None,
    grad_chunk_start_membranes: torch.Tensor | None,
) -> tuple[None, torch.Tensor | None, None, None, None, None, None, None, None, None]:
    del grad_chunk_start_membranes
    if grad_final_membrane is None and grad_spike_rates is None:
        return None, None, None, None, None, None, None, None, None, None

    inputs, weight, chunk_start_membranes = ctx.saved_tensors
    grad_final = (
        torch.zeros((inputs.shape[1], weight.shape[1]), dtype=inputs.dtype, device=inputs.device)
        if grad_final_membrane is None
        else grad_final_membrane
    )
    grad_rates = (
        torch.zeros((inputs.shape[1], weight.shape[1]), dtype=inputs.dtype, device=inputs.device)
        if grad_spike_rates is None
        else grad_spike_rates
    )
    grad_inputs, grad_weight = linear_lif_checkpoint_rate_grads_no_bias_op(
        inputs,
        weight,
        chunk_start_membranes,
        grad_final,
        grad_rates,
        ctx.decay,
        ctx.threshold,
        ctx.reset,
        ctx.surrogate_slope,
        ctx.checkpoint_size,
        ctx.block_b,
        ctx.block_n,
        ctx.block_f,
    )
    if not ctx.needs_input_grad[0]:
        grad_inputs = None
    if not ctx.needs_input_grad[1]:
        grad_weight = None
    return grad_inputs, grad_weight, None, None, None, None, None, None, None, None


def _linear_lif_checkpoint_rate_forward_bias_setup_context(
    ctx: Any,
    inputs: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        float,
        float,
        float,
        float,
        int,
        int,
        int,
        int,
    ],
    output: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    (
        input_values,
        weight,
        bias,
        decay,
        threshold,
        reset,
        surrogate_slope,
        checkpoint_size,
        block_b,
        block_n,
        block_f,
    ) = inputs
    _final_membrane, _spike_rates, chunk_start_membranes = output
    ctx.save_for_backward(input_values, weight, bias, chunk_start_membranes)
    ctx.decay = decay
    ctx.threshold = threshold
    ctx.reset = reset
    ctx.surrogate_slope = surrogate_slope
    ctx.checkpoint_size = checkpoint_size
    ctx.block_b = block_b
    ctx.block_n = block_n
    ctx.block_f = block_f


def _linear_lif_checkpoint_rate_forward_bias_backward(
    ctx: Any,
    grad_final_membrane: torch.Tensor | None,
    grad_spike_rates: torch.Tensor | None,
    grad_chunk_start_membranes: torch.Tensor | None,
) -> tuple[
    None,
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
    del grad_chunk_start_membranes
    if grad_final_membrane is None and grad_spike_rates is None:
        return None, None, None, None, None, None, None, None, None, None, None

    inputs, weight, bias, chunk_start_membranes = ctx.saved_tensors
    grad_final = (
        torch.zeros((inputs.shape[1], weight.shape[1]), dtype=inputs.dtype, device=inputs.device)
        if grad_final_membrane is None
        else grad_final_membrane
    )
    grad_rates = (
        torch.zeros((inputs.shape[1], weight.shape[1]), dtype=inputs.dtype, device=inputs.device)
        if grad_spike_rates is None
        else grad_spike_rates
    )
    grad_inputs, grad_weight, grad_bias = linear_lif_checkpoint_rate_grads_bias_op(
        inputs,
        weight,
        bias,
        chunk_start_membranes,
        grad_final,
        grad_rates,
        ctx.decay,
        ctx.threshold,
        ctx.reset,
        ctx.surrogate_slope,
        ctx.checkpoint_size,
        ctx.block_b,
        ctx.block_n,
        ctx.block_f,
    )
    if not ctx.needs_input_grad[0]:
        grad_inputs = None
    if not ctx.needs_input_grad[1]:
        grad_weight = None
    if not ctx.needs_input_grad[2]:
        grad_bias = None
    return grad_inputs, grad_weight, grad_bias, None, None, None, None, None, None, None, None


cast(Any, linear_lif_checkpoint_rate_forward_no_bias_op).register_autograd(
    _linear_lif_checkpoint_rate_forward_no_bias_backward,
    setup_context=_linear_lif_checkpoint_rate_forward_no_bias_setup_context,
)

cast(Any, linear_lif_checkpoint_rate_forward_bias_op).register_autograd(
    _linear_lif_checkpoint_rate_forward_bias_backward,
    setup_context=_linear_lif_checkpoint_rate_forward_bias_setup_context,
)
