"""Triton forward kernels for dense LIF dynamics."""

from __future__ import annotations

import torch

from myelin._optional import require_triton
from myelin.neurons import LIFParams, LIFState
from myelin.packing import PackedSpikes, packed_last_dim_size
from myelin.surrogates import SurrogateName

triton = require_triton()
import triton.language as tl  # noqa: E402


@triton.jit
def _lif_forward_kernel(
    inputs_ptr,
    initial_ptr,
    final_ptr,
    spikes_ptr,
    total_elements: tl.constexpr,
    timesteps: tl.constexpr,
    decay: tl.constexpr,
    threshold: tl.constexpr,
    reset: tl.constexpr,
    block_size: tl.constexpr,
):
    program_id = tl.program_id(0)
    offsets = program_id * block_size + tl.arange(0, block_size)
    mask = offsets < total_elements
    membrane = tl.load(initial_ptr + offsets, mask=mask, other=0.0)

    for t in range(timesteps):
        input_current = tl.load(inputs_ptr + t * total_elements + offsets, mask=mask, other=0.0)
        membrane = membrane * decay + input_current
        spike = membrane >= threshold
        tl.store(spikes_ptr + t * total_elements + offsets, spike.to(tl.float32), mask=mask)
        membrane = tl.where(spike, reset, membrane)

    tl.store(final_ptr + offsets, membrane, mask=mask)


def lif_forward(
    inputs: torch.Tensor,
    initial_state: LIFState,
    params: LIFParams,
    *,
    block_size: int = 256,
) -> tuple[LIFState, torch.Tensor]:
    """Run dense LIF forward over time in one Triton launch.

    Args:
        inputs: Contiguous CUDA tensor shaped ``[T, B, N]``.
        initial_state: Initial membrane shaped ``[B, N]``.
        params: LIF dynamics parameters.
        block_size: Number of flattened neurons handled per Triton program.
    """

    if inputs.ndim != 3:
        raise ValueError(f"inputs must be shaped [T, B, N]; got {tuple(inputs.shape)}")
    if not inputs.is_cuda:
        raise ValueError("Triton LIF forward requires CUDA inputs")
    if initial_state.membrane.shape != inputs.shape[1:]:
        msg = (
            "initial_state.membrane must match inputs.shape[1:]; "
            f"got {tuple(initial_state.membrane.shape)} and {tuple(inputs.shape[1:])}"
        )
        raise ValueError(msg)
    if initial_state.membrane.device != inputs.device:
        raise ValueError("initial_state.membrane must be on the same device as inputs")
    if initial_state.membrane.dtype != inputs.dtype:
        raise ValueError("initial_state.membrane must have the same dtype as inputs")

    contiguous_inputs = inputs.contiguous()
    contiguous_initial = initial_state.membrane.contiguous()
    timesteps = contiguous_inputs.shape[0]
    total_elements = contiguous_initial.numel()
    final_membrane = torch.empty_like(contiguous_initial)
    spikes = torch.empty_like(contiguous_inputs)

    grid = (triton.cdiv(total_elements, block_size),)
    _lif_forward_kernel[grid](
        contiguous_inputs,
        contiguous_initial,
        final_membrane,
        spikes,
        total_elements,  # pyright: ignore[reportArgumentType]
        timesteps,  # pyright: ignore[reportArgumentType]
        params.decay,  # pyright: ignore[reportArgumentType]
        params.threshold,  # pyright: ignore[reportArgumentType]
        params.reset,  # pyright: ignore[reportArgumentType]
        block_size,  # pyright: ignore[reportArgumentType]
    )
    return LIFState(membrane=final_membrane), spikes


@triton.jit
def _lif_forward_packed_spikes_kernel(
    inputs_ptr,
    initial_ptr,
    final_ptr,
    packed_spikes_ptr,
    timesteps: tl.constexpr,
    batch: tl.constexpr,
    neurons: tl.constexpr,
    packed_neurons: tl.constexpr,
    decay: tl.constexpr,
    threshold: tl.constexpr,
    reset: tl.constexpr,
    block_b: tl.constexpr,
):
    program_b = tl.program_id(0) * block_b + tl.arange(0, block_b)
    program_word = tl.program_id(1)
    bit_offsets = tl.arange(0, 32)
    neuron_offsets = program_word * 32 + bit_offsets[None, :]
    mask = (program_b[:, None] < batch) & (neuron_offsets < neurons)
    state_offsets = program_b[:, None] * neurons + neuron_offsets
    membrane = tl.load(initial_ptr + state_offsets, mask=mask, other=0.0)
    weights = (1 << bit_offsets).to(tl.int64)

    for t in range(timesteps):
        input_current = tl.load(
            inputs_ptr + t * batch * neurons + state_offsets,
            mask=mask,
            other=0.0,
        )
        membrane = membrane * decay + input_current
        spike = membrane >= threshold
        packed = tl.sum(spike.to(tl.int64) * weights[None, :], axis=1).to(tl.int32)
        packed_offset = t * batch * packed_neurons + program_b * packed_neurons + program_word
        tl.store(packed_spikes_ptr + packed_offset, packed, mask=program_b < batch)
        membrane = tl.where(spike, reset, membrane)

    tl.store(final_ptr + state_offsets, membrane, mask=mask)


def lif_forward_packed_spikes(
    inputs: torch.Tensor,
    initial_state: LIFState,
    params: LIFParams,
    *,
    block_b: int = 8,
) -> tuple[LIFState, PackedSpikes]:
    """Run dense LIF forward and write bitpacked spikes directly."""

    if inputs.ndim != 3:
        raise ValueError(f"inputs must be shaped [T, B, N]; got {tuple(inputs.shape)}")
    if not inputs.is_cuda:
        raise ValueError("Triton packed LIF forward requires CUDA inputs")
    if initial_state.membrane.shape != inputs.shape[1:]:
        msg = (
            "initial_state.membrane must match inputs.shape[1:]; "
            f"got {tuple(initial_state.membrane.shape)} and {tuple(inputs.shape[1:])}"
        )
        raise ValueError(msg)
    if initial_state.membrane.device != inputs.device:
        raise ValueError("initial_state.membrane must be on the same device as inputs")
    if initial_state.membrane.dtype != inputs.dtype:
        raise ValueError("initial_state.membrane must have the same dtype as inputs")
    if block_b <= 0 or block_b & (block_b - 1):
        raise ValueError("block_b must be a positive power of two")

    contiguous_inputs = inputs.contiguous()
    contiguous_initial = initial_state.membrane.contiguous()
    timesteps, batch, neurons = contiguous_inputs.shape
    packed_neurons = packed_last_dim_size(neurons)
    final_membrane = torch.empty_like(contiguous_initial)
    packed_spikes = torch.empty(
        (timesteps, batch, packed_neurons),
        dtype=torch.int32,
        device=inputs.device,
    )
    grid = (triton.cdiv(batch, block_b), packed_neurons)
    _lif_forward_packed_spikes_kernel[grid](
        contiguous_inputs,
        contiguous_initial,
        final_membrane,
        packed_spikes,
        timesteps,  # pyright: ignore[reportArgumentType]
        batch,  # pyright: ignore[reportArgumentType]
        neurons,  # pyright: ignore[reportArgumentType]
        packed_neurons,  # pyright: ignore[reportArgumentType]
        params.decay,  # pyright: ignore[reportArgumentType]
        params.threshold,  # pyright: ignore[reportArgumentType]
        params.reset,  # pyright: ignore[reportArgumentType]
        block_b,  # pyright: ignore[reportArgumentType]
    )
    return LIFState(membrane=final_membrane), PackedSpikes(
        data=packed_spikes,
        original_shape=tuple(inputs.shape),
    )


@triton.jit
def _surrogate_lif_forward_kernel(
    inputs_ptr,
    initial_ptr,
    final_ptr,
    spikes_ptr,
    pre_reset_ptr,
    total_elements: tl.constexpr,
    timesteps: tl.constexpr,
    decay: tl.constexpr,
    threshold: tl.constexpr,
    reset: tl.constexpr,
    block_size: tl.constexpr,
):
    program_id = tl.program_id(0)
    offsets = program_id * block_size + tl.arange(0, block_size)
    mask = offsets < total_elements
    membrane = tl.load(initial_ptr + offsets, mask=mask, other=0.0)

    for t in range(timesteps):
        input_current = tl.load(inputs_ptr + t * total_elements + offsets, mask=mask, other=0.0)
        membrane = membrane * decay + input_current
        spike = membrane >= threshold
        tl.store(pre_reset_ptr + t * total_elements + offsets, membrane, mask=mask)
        tl.store(spikes_ptr + t * total_elements + offsets, spike.to(tl.float32), mask=mask)
        membrane = tl.where(spike, reset, membrane)

    tl.store(final_ptr + offsets, membrane, mask=mask)


@triton.jit
def _surrogate_derivative(
    centered,
    surrogate_id: tl.constexpr,
):
    if surrogate_id == 0:
        value = 1.0 / (1.0 + tl.exp(-centered))
        return value * (1.0 - value)
    if surrogate_id == 1:
        denom = 1.0 + tl.abs(centered)
        return 0.5 / (denom * denom)
    if surrogate_id == 2:
        scaled = 1.5707963267948966 * centered
        return 0.5 / (1.0 + scaled * scaled)
    if surrogate_id == 4:
        denom = 1.0 + tl.abs(centered)
        return 1.0 / (denom * denom)
    if surrogate_id == 5:
        center = 0.47873073648171924 * tl.exp(-0.5 * (centered / 0.5) * (centered / 0.5))
        right = 0.07978845608028655 * tl.exp(-0.5 * (centered - 1.0) * (centered - 1.0))
        left = 0.07978845608028655 * tl.exp(-0.5 * (centered + 1.0) * (centered + 1.0))
        return center + right + left

    sign = tl.where(centered > 0.0, 1.0, tl.where(centered < 0.0, -1.0, 0.0))
    return tl.where(tl.abs(centered) < 1.0, -sign, 0.0)


@triton.jit
def _surrogate_lif_backward_kernel(
    pre_reset_ptr,
    spikes_ptr,
    grad_final_ptr,
    grad_spikes_ptr,
    grad_inputs_ptr,
    grad_initial_ptr,
    total_elements: tl.constexpr,
    timesteps: tl.constexpr,
    decay: tl.constexpr,
    threshold: tl.constexpr,
    reset: tl.constexpr,
    surrogate_slope: tl.constexpr,
    surrogate_id: tl.constexpr,
    block_size: tl.constexpr,
):
    program_id = tl.program_id(0)
    offsets = program_id * block_size + tl.arange(0, block_size)
    mask = offsets < total_elements
    grad_membrane = tl.load(grad_final_ptr + offsets, mask=mask, other=0.0)

    for reverse_t in range(timesteps):
        t = timesteps - 1 - reverse_t
        time_offsets = t * total_elements + offsets
        pre_reset = tl.load(pre_reset_ptr + time_offsets, mask=mask, other=0.0)
        spike = tl.load(spikes_ptr + time_offsets, mask=mask, other=0.0)
        grad_spike = tl.load(grad_spikes_ptr + time_offsets, mask=mask, other=0.0)
        centered = surrogate_slope * (pre_reset - threshold)
        d_spike_d_membrane = surrogate_slope * _surrogate_derivative(centered, surrogate_id)

        grad_pre_reset = grad_membrane * ((1.0 - spike) + (reset - pre_reset) * d_spike_d_membrane)
        grad_pre_reset = grad_pre_reset + grad_spike * d_spike_d_membrane
        tl.store(grad_inputs_ptr + time_offsets, grad_pre_reset, mask=mask)
        grad_membrane = grad_pre_reset * decay

    tl.store(grad_initial_ptr + offsets, grad_membrane, mask=mask)


@triton.jit
def _surrogate_lif_backward_packed_spikes_kernel(
    pre_reset_ptr,
    packed_spikes_ptr,
    grad_final_ptr,
    grad_spikes_ptr,
    grad_inputs_ptr,
    grad_initial_ptr,
    total_elements: tl.constexpr,
    timesteps: tl.constexpr,
    batch: tl.constexpr,
    neurons: tl.constexpr,
    packed_neurons: tl.constexpr,
    decay: tl.constexpr,
    threshold: tl.constexpr,
    reset: tl.constexpr,
    surrogate_slope: tl.constexpr,
    surrogate_id: tl.constexpr,
    block_size: tl.constexpr,
):
    program_id = tl.program_id(0)
    offsets = program_id * block_size + tl.arange(0, block_size)
    mask = offsets < total_elements
    batch_offsets = offsets // neurons
    neuron_offsets = offsets - batch_offsets * neurons
    word_offsets = neuron_offsets // 32
    bit_offsets = neuron_offsets - word_offsets * 32
    grad_membrane = tl.load(grad_final_ptr + offsets, mask=mask, other=0.0)

    for reverse_t in range(timesteps):
        t = timesteps - 1 - reverse_t
        time_offsets = t * total_elements + offsets
        packed_offsets = t * batch * packed_neurons + batch_offsets * packed_neurons + word_offsets
        pre_reset = tl.load(pre_reset_ptr + time_offsets, mask=mask, other=0.0)
        packed_word = tl.load(packed_spikes_ptr + packed_offsets, mask=mask, other=0).to(tl.int64)
        spike = ((packed_word >> bit_offsets) & 1).to(tl.float32)
        grad_spike = tl.load(grad_spikes_ptr + time_offsets, mask=mask, other=0.0)
        centered = surrogate_slope * (pre_reset - threshold)
        d_spike_d_membrane = surrogate_slope * _surrogate_derivative(centered, surrogate_id)

        grad_pre_reset = grad_membrane * ((1.0 - spike) + (reset - pre_reset) * d_spike_d_membrane)
        grad_pre_reset = grad_pre_reset + grad_spike * d_spike_d_membrane
        tl.store(grad_inputs_ptr + time_offsets, grad_pre_reset, mask=mask)
        grad_membrane = grad_pre_reset * decay

    tl.store(grad_initial_ptr + offsets, grad_membrane, mask=mask)


def _validate_lif_tensors(inputs: torch.Tensor, initial_state: LIFState) -> None:
    if inputs.ndim != 3:
        raise ValueError(f"inputs must be shaped [T, B, N]; got {tuple(inputs.shape)}")
    if not inputs.is_cuda:
        raise ValueError("Triton LIF forward requires CUDA inputs")
    if initial_state.membrane.shape != inputs.shape[1:]:
        msg = (
            "initial_state.membrane must match inputs.shape[1:]; "
            f"got {tuple(initial_state.membrane.shape)} and {tuple(inputs.shape[1:])}"
        )
        raise ValueError(msg)
    if initial_state.membrane.device != inputs.device:
        raise ValueError("initial_state.membrane must be on the same device as inputs")
    if initial_state.membrane.dtype != inputs.dtype:
        raise ValueError("initial_state.membrane must have the same dtype as inputs")


def surrogate_id(name: SurrogateName) -> int:
    """Return the Triton constexpr id for a built-in surrogate."""

    if name == "sigmoid":
        return 0
    if name == "fast_sigmoid":
        return 1
    if name == "atan":
        return 2
    if name == "triangular":
        return 3
    if name == "superspike":
        return 4
    if name == "multi_gaussian":
        return 5


def surrogate_lif_forward(
    inputs: torch.Tensor,
    initial_state: LIFState,
    params: LIFParams,
    *,
    block_size: int = 256,
) -> tuple[LIFState, torch.Tensor, torch.Tensor]:
    """Run hard-forward LIF and save pre-reset membranes for surrogate backward."""

    _validate_lif_tensors(inputs, initial_state)
    contiguous_inputs = inputs.contiguous()
    contiguous_initial = initial_state.membrane.contiguous()
    timesteps = contiguous_inputs.shape[0]
    total_elements = contiguous_initial.numel()
    final_membrane = torch.empty_like(contiguous_initial)
    spikes = torch.empty_like(contiguous_inputs)
    pre_reset_membranes = torch.empty_like(contiguous_inputs)

    grid = (triton.cdiv(total_elements, block_size),)
    _surrogate_lif_forward_kernel[grid](
        contiguous_inputs,
        contiguous_initial,
        final_membrane,
        spikes,
        pre_reset_membranes,
        total_elements,  # pyright: ignore[reportArgumentType]
        timesteps,  # pyright: ignore[reportArgumentType]
        params.decay,  # pyright: ignore[reportArgumentType]
        params.threshold,  # pyright: ignore[reportArgumentType]
        params.reset,  # pyright: ignore[reportArgumentType]
        block_size,  # pyright: ignore[reportArgumentType]
    )
    return LIFState(membrane=final_membrane), spikes, pre_reset_membranes


def surrogate_lif_backward(
    pre_reset_membranes: torch.Tensor,
    spikes: torch.Tensor,
    grad_final_membrane: torch.Tensor | None,
    grad_spikes: torch.Tensor | None,
    params: LIFParams,
    *,
    surrogate: SurrogateName,
    surrogate_slope: float,
    block_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run fused reverse-time surrogate LIF backward in Triton."""

    if pre_reset_membranes.ndim != 3:
        raise ValueError(
            f"pre_reset_membranes must be shaped [T, B, N]; got {tuple(pre_reset_membranes.shape)}"
        )
    if not pre_reset_membranes.is_cuda:
        raise ValueError("Triton LIF backward requires CUDA tensors")
    if spikes.shape != pre_reset_membranes.shape:
        raise ValueError("spikes must have the same shape as pre_reset_membranes")
    if spikes.device != pre_reset_membranes.device or spikes.dtype != pre_reset_membranes.dtype:
        raise ValueError("spikes must have the same device and dtype as pre_reset_membranes")

    contiguous_pre_reset = pre_reset_membranes.contiguous()
    contiguous_spikes = spikes.contiguous()
    timesteps = contiguous_pre_reset.shape[0]
    state_shape = contiguous_pre_reset.shape[1:]
    total_elements = contiguous_pre_reset.shape[1] * contiguous_pre_reset.shape[2]

    if grad_final_membrane is None:
        contiguous_grad_final = torch.zeros(
            state_shape,
            dtype=contiguous_pre_reset.dtype,
            device=contiguous_pre_reset.device,
        )
    else:
        if grad_final_membrane.shape != state_shape:
            raise ValueError("grad_final_membrane must match pre_reset_membranes.shape[1:]")
        contiguous_grad_final = grad_final_membrane.contiguous()

    if grad_spikes is None:
        contiguous_grad_spikes = torch.zeros_like(contiguous_pre_reset)
    else:
        if grad_spikes.shape != contiguous_pre_reset.shape:
            raise ValueError("grad_spikes must match pre_reset_membranes")
        contiguous_grad_spikes = grad_spikes.contiguous()

    grad_inputs = torch.empty_like(contiguous_pre_reset)
    grad_initial = torch.empty_like(contiguous_grad_final)
    grid = (triton.cdiv(total_elements, block_size),)
    _surrogate_lif_backward_kernel[grid](
        contiguous_pre_reset,
        contiguous_spikes,
        contiguous_grad_final,
        contiguous_grad_spikes,
        grad_inputs,
        grad_initial,
        total_elements,  # pyright: ignore[reportArgumentType]
        timesteps,  # pyright: ignore[reportArgumentType]
        params.decay,  # pyright: ignore[reportArgumentType]
        params.threshold,  # pyright: ignore[reportArgumentType]
        params.reset,  # pyright: ignore[reportArgumentType]
        surrogate_slope,  # pyright: ignore[reportArgumentType]
        surrogate_id(surrogate),  # pyright: ignore[reportArgumentType]
        block_size,  # pyright: ignore[reportArgumentType]
    )
    return grad_inputs, grad_initial


def surrogate_lif_backward_packed_spikes(
    pre_reset_membranes: torch.Tensor,
    packed_spikes: torch.Tensor,
    grad_final_membrane: torch.Tensor | None,
    grad_spikes: torch.Tensor | None,
    params: LIFParams,
    *,
    surrogate: SurrogateName,
    surrogate_slope: float,
    block_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run surrogate LIF backward while reading saved spikes from packed words."""

    if pre_reset_membranes.ndim != 3:
        raise ValueError(
            f"pre_reset_membranes must be shaped [T, B, N]; got {tuple(pre_reset_membranes.shape)}"
        )
    if not pre_reset_membranes.is_cuda:
        raise ValueError("Triton packed-spike LIF backward requires CUDA tensors")
    if packed_spikes.dtype != torch.int32:
        raise ValueError("packed_spikes must have dtype torch.int32")
    if packed_spikes.device != pre_reset_membranes.device:
        raise ValueError("packed_spikes must be on the same device as pre_reset_membranes")

    contiguous_pre_reset = pre_reset_membranes.contiguous()
    timesteps, batch, neurons = contiguous_pre_reset.shape
    packed_neurons = packed_last_dim_size(neurons)
    if packed_spikes.shape != (timesteps, batch, packed_neurons):
        raise ValueError("packed_spikes must be shaped [T, B, ceil(N / 32)]")
    contiguous_packed_spikes = packed_spikes.contiguous()
    state_shape = contiguous_pre_reset.shape[1:]
    total_elements = batch * neurons

    if grad_final_membrane is None:
        contiguous_grad_final = torch.zeros(
            state_shape,
            dtype=contiguous_pre_reset.dtype,
            device=contiguous_pre_reset.device,
        )
    else:
        if grad_final_membrane.shape != state_shape:
            raise ValueError("grad_final_membrane must match pre_reset_membranes.shape[1:]")
        contiguous_grad_final = grad_final_membrane.contiguous()

    if grad_spikes is None:
        contiguous_grad_spikes = torch.zeros_like(contiguous_pre_reset)
    else:
        if grad_spikes.shape != contiguous_pre_reset.shape:
            raise ValueError("grad_spikes must match pre_reset_membranes")
        contiguous_grad_spikes = grad_spikes.contiguous()

    grad_inputs = torch.empty_like(contiguous_pre_reset)
    grad_initial = torch.empty_like(contiguous_grad_final)
    grid = (triton.cdiv(total_elements, block_size),)
    _surrogate_lif_backward_packed_spikes_kernel[grid](
        contiguous_pre_reset,
        contiguous_packed_spikes,
        contiguous_grad_final,
        contiguous_grad_spikes,
        grad_inputs,
        grad_initial,
        total_elements,  # pyright: ignore[reportArgumentType]
        timesteps,  # pyright: ignore[reportArgumentType]
        batch,  # pyright: ignore[reportArgumentType]
        neurons,  # pyright: ignore[reportArgumentType]
        packed_neurons,  # pyright: ignore[reportArgumentType]
        params.decay,  # pyright: ignore[reportArgumentType]
        params.threshold,  # pyright: ignore[reportArgumentType]
        params.reset,  # pyright: ignore[reportArgumentType]
        surrogate_slope,  # pyright: ignore[reportArgumentType]
        surrogate_id(surrogate),  # pyright: ignore[reportArgumentType]
        block_size,  # pyright: ignore[reportArgumentType]
    )
    return grad_inputs, grad_initial


@triton.jit
def _linear_surrogate_lif_forward_kernel(
    inputs_ptr,
    weight_ptr,
    bias_ptr,
    final_ptr,
    spikes_ptr,
    pre_reset_ptr,
    timesteps: tl.constexpr,
    batch: tl.constexpr,
    features: tl.constexpr,
    neurons: tl.constexpr,
    decay: tl.constexpr,
    threshold: tl.constexpr,
    reset: tl.constexpr,
    has_bias: tl.constexpr,
    block_b: tl.constexpr,
    block_n: tl.constexpr,
    block_f: tl.constexpr,
):
    program_b = tl.program_id(0)
    program_n = tl.program_id(1)
    offsets_b = program_b * block_b + tl.arange(0, block_b)
    offsets_n = program_n * block_n + tl.arange(0, block_n)
    mask_bn = (offsets_b[:, None] < batch) & (offsets_n[None, :] < neurons)
    membrane = tl.zeros((block_b, block_n), tl.float32)

    for t in range(timesteps):
        current = tl.zeros((block_b, block_n), tl.float32)
        for f_start in range(0, features, block_f):
            offsets_f = f_start + tl.arange(0, block_f)
            input_values = tl.load(
                inputs_ptr
                + t * batch * features
                + offsets_b[:, None] * features
                + offsets_f[None, :],
                mask=(offsets_b[:, None] < batch) & (offsets_f[None, :] < features),
                other=0.0,
            )
            weight_values = tl.load(
                weight_ptr + offsets_f[:, None] * neurons + offsets_n[None, :],
                mask=(offsets_f[:, None] < features) & (offsets_n[None, :] < neurons),
                other=0.0,
            )
            current += tl.dot(input_values, weight_values)

        if has_bias:
            bias = tl.load(bias_ptr + offsets_n, mask=offsets_n < neurons, other=0.0)
            current += bias[None, :]

        membrane = membrane * decay + current
        spike = membrane >= threshold
        output_offsets = t * batch * neurons + offsets_b[:, None] * neurons + offsets_n[None, :]
        tl.store(pre_reset_ptr + output_offsets, membrane, mask=mask_bn)
        tl.store(spikes_ptr + output_offsets, spike.to(tl.float32), mask=mask_bn)
        membrane = tl.where(spike, reset, membrane)

    final_offsets = offsets_b[:, None] * neurons + offsets_n[None, :]
    tl.store(final_ptr + final_offsets, membrane, mask=mask_bn)


def linear_surrogate_lif_forward(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    params: LIFParams,
    *,
    block_b: int = 16,
    block_n: int = 32,
    block_f: int = 32,
) -> tuple[LIFState, torch.Tensor, torch.Tensor]:
    """Run dense projection plus hard-forward LIF in one fused Triton kernel."""

    for name, value in [("block_b", block_b), ("block_n", block_n), ("block_f", block_f)]:
        if value <= 0 or value & (value - 1):
            raise ValueError(f"{name} must be a positive power of two")
    if block_f < 16:
        raise ValueError("block_f must be at least 16 for Triton dot")

    if inputs.ndim != 3:
        raise ValueError(f"inputs must be shaped [T, B, F]; got {tuple(inputs.shape)}")
    if weight.ndim != 2:
        raise ValueError(f"weight must be shaped [F, N]; got {tuple(weight.shape)}")
    if not inputs.is_cuda:
        raise ValueError("Triton linear surrogate LIF forward requires CUDA inputs")
    if weight.device != inputs.device or weight.dtype != inputs.dtype:
        raise ValueError("weight must have the same device and dtype as inputs")
    if inputs.shape[2] != weight.shape[0]:
        msg = (
            "inputs.shape[2] must match weight.shape[0]; "
            f"got {inputs.shape[2]} and {weight.shape[0]}"
        )
        raise ValueError(msg)
    if bias is not None:
        if bias.shape != (weight.shape[1],):
            raise ValueError("bias must be shaped [N]")
        if bias.device != inputs.device or bias.dtype != inputs.dtype:
            raise ValueError("bias must have the same device and dtype as inputs")

    contiguous_inputs = inputs.contiguous()
    contiguous_weight = weight.contiguous()
    contiguous_bias = None if bias is None else bias.contiguous()
    timesteps, batch, features = contiguous_inputs.shape
    neurons = contiguous_weight.shape[1]
    final_membrane = torch.empty((batch, neurons), dtype=inputs.dtype, device=inputs.device)
    spikes = torch.empty((timesteps, batch, neurons), dtype=inputs.dtype, device=inputs.device)
    pre_reset_membranes = torch.empty_like(spikes)
    bias_ptr = contiguous_weight if contiguous_bias is None else contiguous_bias

    grid = (triton.cdiv(batch, block_b), triton.cdiv(neurons, block_n))
    _linear_surrogate_lif_forward_kernel[grid](
        contiguous_inputs,
        contiguous_weight,
        bias_ptr,
        final_membrane,
        spikes,
        pre_reset_membranes,
        timesteps,  # pyright: ignore[reportArgumentType]
        batch,  # pyright: ignore[reportArgumentType]
        features,  # pyright: ignore[reportArgumentType]
        neurons,  # pyright: ignore[reportArgumentType]
        params.decay,  # pyright: ignore[reportArgumentType]
        params.threshold,  # pyright: ignore[reportArgumentType]
        params.reset,  # pyright: ignore[reportArgumentType]
        bias is not None,  # pyright: ignore[reportArgumentType]
        block_b,  # pyright: ignore[reportArgumentType]
        block_n,  # pyright: ignore[reportArgumentType]
        block_f,  # pyright: ignore[reportArgumentType]
    )
    return LIFState(membrane=final_membrane), spikes, pre_reset_membranes


@triton.jit
def _linear_surrogate_lif_checkpoint_forward_kernel(
    inputs_ptr,
    weight_ptr,
    bias_ptr,
    final_ptr,
    spikes_ptr,
    chunk_start_ptr,
    timesteps: tl.constexpr,
    batch: tl.constexpr,
    features: tl.constexpr,
    neurons: tl.constexpr,
    decay: tl.constexpr,
    threshold: tl.constexpr,
    reset: tl.constexpr,
    has_bias: tl.constexpr,
    checkpoint_size: tl.constexpr,
    block_b: tl.constexpr,
    block_n: tl.constexpr,
    block_f: tl.constexpr,
):
    program_b = tl.program_id(0)
    program_n = tl.program_id(1)
    offsets_b = program_b * block_b + tl.arange(0, block_b)
    offsets_n = program_n * block_n + tl.arange(0, block_n)
    mask_bn = (offsets_b[:, None] < batch) & (offsets_n[None, :] < neurons)
    membrane = tl.zeros((block_b, block_n), tl.float32)

    for t in range(timesteps):
        if t % checkpoint_size == 0:
            chunk_index = t // checkpoint_size
            chunk_offsets = (
                chunk_index * batch * neurons + offsets_b[:, None] * neurons + offsets_n[None, :]
            )
            tl.store(chunk_start_ptr + chunk_offsets, membrane, mask=mask_bn)

        current = tl.zeros((block_b, block_n), tl.float32)
        for f_start in range(0, features, block_f):
            offsets_f = f_start + tl.arange(0, block_f)
            input_values = tl.load(
                inputs_ptr
                + t * batch * features
                + offsets_b[:, None] * features
                + offsets_f[None, :],
                mask=(offsets_b[:, None] < batch) & (offsets_f[None, :] < features),
                other=0.0,
            )
            weight_values = tl.load(
                weight_ptr + offsets_f[:, None] * neurons + offsets_n[None, :],
                mask=(offsets_f[:, None] < features) & (offsets_n[None, :] < neurons),
                other=0.0,
            )
            current += tl.dot(input_values, weight_values)

        if has_bias:
            bias = tl.load(bias_ptr + offsets_n, mask=offsets_n < neurons, other=0.0)
            current += bias[None, :]

        membrane = membrane * decay + current
        spike = membrane >= threshold
        output_offsets = t * batch * neurons + offsets_b[:, None] * neurons + offsets_n[None, :]
        tl.store(spikes_ptr + output_offsets, spike.to(tl.float32), mask=mask_bn)
        membrane = tl.where(spike, reset, membrane)

    final_offsets = offsets_b[:, None] * neurons + offsets_n[None, :]
    tl.store(final_ptr + final_offsets, membrane, mask=mask_bn)


@triton.jit
def _linear_surrogate_lif_checkpoint_packed_forward_kernel(
    inputs_ptr,
    weight_ptr,
    bias_ptr,
    final_ptr,
    packed_spikes_ptr,
    chunk_start_ptr,
    timesteps: tl.constexpr,
    batch: tl.constexpr,
    features: tl.constexpr,
    neurons: tl.constexpr,
    packed_neurons: tl.constexpr,
    decay: tl.constexpr,
    threshold: tl.constexpr,
    reset: tl.constexpr,
    has_bias: tl.constexpr,
    checkpoint_size: tl.constexpr,
    block_b: tl.constexpr,
    block_f: tl.constexpr,
):
    program_b = tl.program_id(0)
    program_word = tl.program_id(1)
    offsets_b = program_b * block_b + tl.arange(0, block_b)
    bit_offsets = tl.arange(0, 32)
    offsets_n = program_word * 32 + bit_offsets
    mask_bn = (offsets_b[:, None] < batch) & (offsets_n[None, :] < neurons)
    state_offsets = offsets_b[:, None] * neurons + offsets_n[None, :]
    membrane = tl.zeros((block_b, 32), tl.float32)
    weights = (1 << bit_offsets).to(tl.int64)

    for t in range(timesteps):
        if t % checkpoint_size == 0:
            chunk_index = t // checkpoint_size
            chunk_offsets = chunk_index * batch * neurons + state_offsets
            tl.store(chunk_start_ptr + chunk_offsets, membrane, mask=mask_bn)

        current = tl.zeros((block_b, 32), tl.float32)
        for f_start in range(0, features, block_f):
            offsets_f = f_start + tl.arange(0, block_f)
            input_values = tl.load(
                inputs_ptr
                + t * batch * features
                + offsets_b[:, None] * features
                + offsets_f[None, :],
                mask=(offsets_b[:, None] < batch) & (offsets_f[None, :] < features),
                other=0.0,
            )
            weight_values = tl.load(
                weight_ptr + offsets_f[:, None] * neurons + offsets_n[None, :],
                mask=(offsets_f[:, None] < features) & (offsets_n[None, :] < neurons),
                other=0.0,
            )
            current += tl.dot(input_values, weight_values)

        if has_bias:
            bias = tl.load(bias_ptr + offsets_n, mask=offsets_n < neurons, other=0.0)
            current += bias[None, :]

        membrane = membrane * decay + current
        spike = membrane >= threshold
        packed = tl.sum(spike.to(tl.int64) * weights[None, :], axis=1).to(tl.int32)
        packed_offsets = t * batch * packed_neurons + offsets_b * packed_neurons + program_word
        tl.store(packed_spikes_ptr + packed_offsets, packed, mask=offsets_b < batch)
        membrane = tl.where(spike, reset, membrane)

    tl.store(final_ptr + state_offsets, membrane, mask=mask_bn)


@triton.jit
def _linear_surrogate_lif_checkpoint_rate_forward_kernel(
    inputs_ptr,
    weight_ptr,
    bias_ptr,
    final_ptr,
    rates_ptr,
    chunk_start_ptr,
    timesteps: tl.constexpr,
    batch: tl.constexpr,
    features: tl.constexpr,
    neurons: tl.constexpr,
    decay: tl.constexpr,
    threshold: tl.constexpr,
    reset: tl.constexpr,
    has_bias: tl.constexpr,
    checkpoint_size: tl.constexpr,
    block_b: tl.constexpr,
    block_n: tl.constexpr,
    block_f: tl.constexpr,
):
    program_b = tl.program_id(0)
    program_n = tl.program_id(1)
    offsets_b = program_b * block_b + tl.arange(0, block_b)
    offsets_n = program_n * block_n + tl.arange(0, block_n)
    mask_bn = (offsets_b[:, None] < batch) & (offsets_n[None, :] < neurons)
    membrane = tl.zeros((block_b, block_n), tl.float32)
    spike_counts = tl.zeros((block_b, block_n), tl.float32)

    for t in range(timesteps):
        if t % checkpoint_size == 0:
            chunk_index = t // checkpoint_size
            chunk_offsets = (
                chunk_index * batch * neurons + offsets_b[:, None] * neurons + offsets_n[None, :]
            )
            tl.store(chunk_start_ptr + chunk_offsets, membrane, mask=mask_bn)

        current = tl.zeros((block_b, block_n), tl.float32)
        for f_start in range(0, features, block_f):
            offsets_f = f_start + tl.arange(0, block_f)
            input_values = tl.load(
                inputs_ptr
                + t * batch * features
                + offsets_b[:, None] * features
                + offsets_f[None, :],
                mask=(offsets_b[:, None] < batch) & (offsets_f[None, :] < features),
                other=0.0,
            )
            weight_values = tl.load(
                weight_ptr + offsets_f[:, None] * neurons + offsets_n[None, :],
                mask=(offsets_f[:, None] < features) & (offsets_n[None, :] < neurons),
                other=0.0,
            )
            current += tl.dot(input_values, weight_values)

        if has_bias:
            bias = tl.load(bias_ptr + offsets_n, mask=offsets_n < neurons, other=0.0)
            current += bias[None, :]

        membrane = membrane * decay + current
        spike = membrane >= threshold
        spike_counts += spike.to(tl.float32)
        membrane = tl.where(spike, reset, membrane)

    final_offsets = offsets_b[:, None] * neurons + offsets_n[None, :]
    tl.store(final_ptr + final_offsets, membrane, mask=mask_bn)
    tl.store(rates_ptr + final_offsets, spike_counts / timesteps, mask=mask_bn)


def linear_surrogate_lif_checkpoint_forward(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    params: LIFParams,
    *,
    checkpoint_size: int,
    block_b: int = 16,
    block_n: int = 32,
    block_f: int = 32,
) -> tuple[LIFState, torch.Tensor, torch.Tensor]:
    """Run fused dense projection plus hard-forward LIF, saving chunk starts."""

    for name, value in [
        ("checkpoint_size", checkpoint_size),
        ("block_b", block_b),
        ("block_n", block_n),
        ("block_f", block_f),
    ]:
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    for name, value in [("block_b", block_b), ("block_n", block_n), ("block_f", block_f)]:
        if value & (value - 1):
            raise ValueError(f"{name} must be a power of two")
    if block_f < 16:
        raise ValueError("block_f must be at least 16 for Triton dot")
    if inputs.ndim != 3:
        raise ValueError(f"inputs must be shaped [T, B, F]; got {tuple(inputs.shape)}")
    if weight.ndim != 2:
        raise ValueError(f"weight must be shaped [F, N]; got {tuple(weight.shape)}")
    if not inputs.is_cuda:
        raise ValueError("Triton checkpoint LIF forward requires CUDA inputs")
    if weight.device != inputs.device or weight.dtype != inputs.dtype:
        raise ValueError("weight must have the same device and dtype as inputs")
    if inputs.shape[2] != weight.shape[0]:
        raise ValueError("inputs.shape[2] must match weight.shape[0]")
    if bias is not None:
        if bias.shape != (weight.shape[1],):
            raise ValueError("bias must be shaped [N]")
        if bias.device != inputs.device or bias.dtype != inputs.dtype:
            raise ValueError("bias must have the same device and dtype as inputs")

    contiguous_inputs = inputs.contiguous()
    contiguous_weight = weight.contiguous()
    contiguous_bias = None if bias is None else bias.contiguous()
    timesteps, batch, _features = contiguous_inputs.shape
    neurons = contiguous_weight.shape[1]
    num_chunks = (timesteps + checkpoint_size - 1) // checkpoint_size
    final_membrane = torch.empty((batch, neurons), dtype=inputs.dtype, device=inputs.device)
    spikes = torch.empty((timesteps, batch, neurons), dtype=inputs.dtype, device=inputs.device)
    chunk_start_membranes = torch.empty(
        (num_chunks, batch, neurons), dtype=inputs.dtype, device=inputs.device
    )
    bias_ptr = contiguous_weight if contiguous_bias is None else contiguous_bias

    grid = (triton.cdiv(batch, block_b), triton.cdiv(neurons, block_n))
    _linear_surrogate_lif_checkpoint_forward_kernel[grid](
        contiguous_inputs,
        contiguous_weight,
        bias_ptr,
        final_membrane,
        spikes,
        chunk_start_membranes,
        timesteps,  # pyright: ignore[reportArgumentType]
        batch,  # pyright: ignore[reportArgumentType]
        contiguous_inputs.shape[2],  # pyright: ignore[reportArgumentType]
        neurons,  # pyright: ignore[reportArgumentType]
        params.decay,  # pyright: ignore[reportArgumentType]
        params.threshold,  # pyright: ignore[reportArgumentType]
        params.reset,  # pyright: ignore[reportArgumentType]
        bias is not None,  # pyright: ignore[reportArgumentType]
        checkpoint_size,  # pyright: ignore[reportArgumentType]
        block_b,  # pyright: ignore[reportArgumentType]
        block_n,  # pyright: ignore[reportArgumentType]
        block_f,  # pyright: ignore[reportArgumentType]
    )
    return LIFState(membrane=final_membrane), spikes, chunk_start_membranes


def linear_surrogate_lif_checkpoint_packed_forward(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    params: LIFParams,
    *,
    checkpoint_size: int,
    block_b: int = 16,
    block_f: int = 32,
) -> tuple[LIFState, PackedSpikes, torch.Tensor]:
    """Run checkpointed linear LIF and write spike output as packed int32 words."""

    for name, value in [
        ("checkpoint_size", checkpoint_size),
        ("block_b", block_b),
        ("block_f", block_f),
    ]:
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    for name, value in [("block_b", block_b), ("block_f", block_f)]:
        if value & (value - 1):
            raise ValueError(f"{name} must be a power of two")
    if block_f < 16:
        raise ValueError("block_f must be at least 16 for Triton dot")
    if inputs.ndim != 3:
        raise ValueError(f"inputs must be shaped [T, B, F]; got {tuple(inputs.shape)}")
    if weight.ndim != 2:
        raise ValueError(f"weight must be shaped [F, N]; got {tuple(weight.shape)}")
    if not inputs.is_cuda:
        raise ValueError("Triton checkpoint packed LIF forward requires CUDA inputs")
    if weight.device != inputs.device or weight.dtype != inputs.dtype:
        raise ValueError("weight must have the same device and dtype as inputs")
    if inputs.shape[2] != weight.shape[0]:
        raise ValueError("inputs.shape[2] must match weight.shape[0]")
    if bias is not None:
        if bias.shape != (weight.shape[1],):
            raise ValueError("bias must be shaped [N]")
        if bias.device != inputs.device or bias.dtype != inputs.dtype:
            raise ValueError("bias must have the same device and dtype as inputs")

    contiguous_inputs = inputs.contiguous()
    contiguous_weight = weight.contiguous()
    contiguous_bias = None if bias is None else bias.contiguous()
    timesteps, batch, _features = contiguous_inputs.shape
    neurons = contiguous_weight.shape[1]
    packed_neurons = packed_last_dim_size(neurons)
    num_chunks = (timesteps + checkpoint_size - 1) // checkpoint_size
    final_membrane = torch.empty((batch, neurons), dtype=inputs.dtype, device=inputs.device)
    packed_spikes = torch.empty(
        (timesteps, batch, packed_neurons),
        dtype=torch.int32,
        device=inputs.device,
    )
    chunk_start_membranes = torch.empty(
        (num_chunks, batch, neurons), dtype=inputs.dtype, device=inputs.device
    )
    bias_ptr = contiguous_weight if contiguous_bias is None else contiguous_bias
    grid = (triton.cdiv(batch, block_b), packed_neurons)

    _linear_surrogate_lif_checkpoint_packed_forward_kernel[grid](
        contiguous_inputs,
        contiguous_weight,
        bias_ptr,
        final_membrane,
        packed_spikes,
        chunk_start_membranes,
        timesteps,  # pyright: ignore[reportArgumentType]
        batch,  # pyright: ignore[reportArgumentType]
        contiguous_inputs.shape[2],  # pyright: ignore[reportArgumentType]
        neurons,  # pyright: ignore[reportArgumentType]
        packed_neurons,  # pyright: ignore[reportArgumentType]
        params.decay,  # pyright: ignore[reportArgumentType]
        params.threshold,  # pyright: ignore[reportArgumentType]
        params.reset,  # pyright: ignore[reportArgumentType]
        bias is not None,  # pyright: ignore[reportArgumentType]
        checkpoint_size,  # pyright: ignore[reportArgumentType]
        block_b,  # pyright: ignore[reportArgumentType]
        block_f,  # pyright: ignore[reportArgumentType]
    )
    return (
        LIFState(membrane=final_membrane),
        PackedSpikes(data=packed_spikes, original_shape=(timesteps, batch, neurons)),
        chunk_start_membranes,
    )


def linear_surrogate_lif_checkpoint_rate_forward(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    params: LIFParams,
    *,
    checkpoint_size: int,
    block_b: int = 16,
    block_n: int = 32,
    block_f: int = 32,
) -> tuple[LIFState, torch.Tensor, torch.Tensor]:
    """Run checkpointed hard-forward LIF and return [B, N] spike rates."""

    for name, value in [
        ("checkpoint_size", checkpoint_size),
        ("block_b", block_b),
        ("block_n", block_n),
        ("block_f", block_f),
    ]:
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    for name, value in [("block_b", block_b), ("block_n", block_n), ("block_f", block_f)]:
        if value & (value - 1):
            raise ValueError(f"{name} must be a power of two")
    if block_f < 16:
        raise ValueError("block_f must be at least 16 for Triton dot")
    if inputs.ndim != 3:
        raise ValueError(f"inputs must be shaped [T, B, F]; got {tuple(inputs.shape)}")
    if weight.ndim != 2:
        raise ValueError(f"weight must be shaped [F, N]; got {tuple(weight.shape)}")
    if not inputs.is_cuda:
        raise ValueError("Triton checkpoint rate LIF forward requires CUDA inputs")
    if weight.device != inputs.device or weight.dtype != inputs.dtype:
        raise ValueError("weight must have the same device and dtype as inputs")
    if inputs.shape[2] != weight.shape[0]:
        raise ValueError("inputs.shape[2] must match weight.shape[0]")
    if bias is not None:
        if bias.shape != (weight.shape[1],):
            raise ValueError("bias must be shaped [N]")
        if bias.device != inputs.device or bias.dtype != inputs.dtype:
            raise ValueError("bias must have the same device and dtype as inputs")

    contiguous_inputs = inputs.contiguous()
    contiguous_weight = weight.contiguous()
    contiguous_bias = None if bias is None else bias.contiguous()
    timesteps, batch, _features = contiguous_inputs.shape
    neurons = contiguous_weight.shape[1]
    num_chunks = (timesteps + checkpoint_size - 1) // checkpoint_size
    final_membrane = torch.empty((batch, neurons), dtype=inputs.dtype, device=inputs.device)
    spike_rates = torch.empty((batch, neurons), dtype=inputs.dtype, device=inputs.device)
    chunk_start_membranes = torch.empty(
        (num_chunks, batch, neurons), dtype=inputs.dtype, device=inputs.device
    )
    bias_ptr = contiguous_weight if contiguous_bias is None else contiguous_bias
    grid = (triton.cdiv(batch, block_b), triton.cdiv(neurons, block_n))

    _linear_surrogate_lif_checkpoint_rate_forward_kernel[grid](
        contiguous_inputs,
        contiguous_weight,
        bias_ptr,
        final_membrane,
        spike_rates,
        chunk_start_membranes,
        timesteps,  # pyright: ignore[reportArgumentType]
        batch,  # pyright: ignore[reportArgumentType]
        contiguous_inputs.shape[2],  # pyright: ignore[reportArgumentType]
        neurons,  # pyright: ignore[reportArgumentType]
        params.decay,  # pyright: ignore[reportArgumentType]
        params.threshold,  # pyright: ignore[reportArgumentType]
        params.reset,  # pyright: ignore[reportArgumentType]
        bias is not None,  # pyright: ignore[reportArgumentType]
        checkpoint_size,  # pyright: ignore[reportArgumentType]
        block_b,  # pyright: ignore[reportArgumentType]
        block_n,  # pyright: ignore[reportArgumentType]
        block_f,  # pyright: ignore[reportArgumentType]
    )
    return LIFState(membrane=final_membrane), spike_rates, chunk_start_membranes


@triton.jit
def _linear_surrogate_lif_backward_inputs_kernel(
    grad_current_ptr,
    weight_ptr,
    grad_inputs_ptr,
    timesteps: tl.constexpr,
    batch: tl.constexpr,
    features: tl.constexpr,
    neurons: tl.constexpr,
    block_b: tl.constexpr,
    block_f: tl.constexpr,
    block_n: tl.constexpr,
):
    program_t = tl.program_id(0)
    program_b = tl.program_id(1)
    program_f = tl.program_id(2)
    offsets_b = program_b * block_b + tl.arange(0, block_b)
    offsets_f = program_f * block_f + tl.arange(0, block_f)
    acc = tl.zeros((block_b, block_f), tl.float32)

    for n_start in range(0, neurons, block_n):
        offsets_n = n_start + tl.arange(0, block_n)
        grad_values = tl.load(
            grad_current_ptr
            + program_t * batch * neurons
            + offsets_b[:, None] * neurons
            + offsets_n[None, :],
            mask=(offsets_b[:, None] < batch) & (offsets_n[None, :] < neurons),
            other=0.0,
        )
        weight_values = tl.load(
            weight_ptr + offsets_n[:, None] * features + offsets_f[None, :],
            mask=(offsets_n[:, None] < neurons) & (offsets_f[None, :] < features),
            other=0.0,
        )
        acc += tl.dot(grad_values, weight_values)

    output_offsets = (
        program_t * batch * features + offsets_b[:, None] * features + offsets_f[None, :]
    )
    tl.store(
        grad_inputs_ptr + output_offsets,
        acc,
        mask=(offsets_b[:, None] < batch) & (offsets_f[None, :] < features),
    )


@triton.jit
def _linear_surrogate_lif_backward_weight_kernel(
    inputs_ptr,
    grad_current_ptr,
    grad_weight_ptr,
    timesteps: tl.constexpr,
    batch: tl.constexpr,
    features: tl.constexpr,
    neurons: tl.constexpr,
    block_f: tl.constexpr,
    block_n: tl.constexpr,
    block_b: tl.constexpr,
):
    program_f = tl.program_id(0)
    program_n = tl.program_id(1)
    offsets_f = program_f * block_f + tl.arange(0, block_f)
    offsets_n = program_n * block_n + tl.arange(0, block_n)
    acc = tl.zeros((block_f, block_n), tl.float32)

    for t in range(timesteps):
        for b_start in range(0, batch, block_b):
            offsets_b = b_start + tl.arange(0, block_b)
            input_values = tl.load(
                inputs_ptr
                + t * batch * features
                + offsets_f[:, None]
                + offsets_b[None, :] * features,
                mask=(offsets_f[:, None] < features) & (offsets_b[None, :] < batch),
                other=0.0,
            )
            grad_values = tl.load(
                grad_current_ptr
                + t * batch * neurons
                + offsets_b[:, None] * neurons
                + offsets_n[None, :],
                mask=(offsets_b[:, None] < batch) & (offsets_n[None, :] < neurons),
                other=0.0,
            )
            acc += tl.dot(input_values, grad_values)

    output_offsets = offsets_f[:, None] * neurons + offsets_n[None, :]
    tl.store(
        grad_weight_ptr + output_offsets,
        acc,
        mask=(offsets_f[:, None] < features) & (offsets_n[None, :] < neurons),
    )


@triton.jit
def _linear_surrogate_lif_backward_bias_kernel(
    grad_current_ptr,
    grad_bias_ptr,
    timesteps: tl.constexpr,
    batch: tl.constexpr,
    neurons: tl.constexpr,
    block_n: tl.constexpr,
    block_b: tl.constexpr,
):
    program_n = tl.program_id(0)
    offsets_n = program_n * block_n + tl.arange(0, block_n)
    acc = tl.zeros((block_n,), tl.float32)

    for t in range(timesteps):
        for b_start in range(0, batch, block_b):
            offsets_b = b_start + tl.arange(0, block_b)
            values = tl.load(
                grad_current_ptr
                + t * batch * neurons
                + offsets_b[:, None] * neurons
                + offsets_n[None, :],
                mask=(offsets_b[:, None] < batch) & (offsets_n[None, :] < neurons),
                other=0.0,
            )
            acc += tl.sum(values, axis=0)

    tl.store(grad_bias_ptr + offsets_n, acc, mask=offsets_n < neurons)


@triton.jit
def _linear_surrogate_lif_backward_weight_bias_recurrent_kernel(
    inputs_ptr,
    pre_reset_ptr,
    spikes_ptr,
    grad_final_ptr,
    grad_spikes_ptr,
    grad_weight_ptr,
    grad_bias_ptr,
    timesteps: tl.constexpr,
    batch: tl.constexpr,
    features: tl.constexpr,
    neurons: tl.constexpr,
    decay: tl.constexpr,
    threshold: tl.constexpr,
    reset: tl.constexpr,
    surrogate_slope: tl.constexpr,
    surrogate_id: tl.constexpr,
    has_grad_final: tl.constexpr,
    has_grad_spikes: tl.constexpr,
    needs_weight_grad: tl.constexpr,
    needs_bias_grad: tl.constexpr,
    block_f: tl.constexpr,
    block_n: tl.constexpr,
    block_b: tl.constexpr,
):
    program_f = tl.program_id(0)
    program_n = tl.program_id(1)
    program_b = tl.program_id(2)
    offsets_f = program_f * block_f + tl.arange(0, block_f)
    offsets_n = program_n * block_n + tl.arange(0, block_n)
    offsets_b = program_b * block_b + tl.arange(0, block_b)
    mask_bn = (offsets_b[:, None] < batch) & (offsets_n[None, :] < neurons)

    if has_grad_final:
        grad_membrane = tl.load(
            grad_final_ptr + offsets_b[:, None] * neurons + offsets_n[None, :],
            mask=mask_bn,
            other=0.0,
        )
    else:
        grad_membrane = tl.zeros((block_b, block_n), tl.float32)

    weight_acc = tl.zeros((block_f, block_n), tl.float32)
    bias_acc = tl.zeros((block_n,), tl.float32)

    for reverse_t in range(timesteps):
        t = timesteps - 1 - reverse_t
        output_offsets = t * batch * neurons + offsets_b[:, None] * neurons + offsets_n[None, :]
        pre_reset = tl.load(pre_reset_ptr + output_offsets, mask=mask_bn, other=0.0)
        spike = tl.load(spikes_ptr + output_offsets, mask=mask_bn, other=0.0)
        centered = surrogate_slope * (pre_reset - threshold)
        d_spike_d_membrane = surrogate_slope * _surrogate_derivative(centered, surrogate_id)
        grad_pre_reset = grad_membrane * ((1.0 - spike) + (reset - pre_reset) * d_spike_d_membrane)
        if has_grad_spikes:
            grad_spike = tl.load(grad_spikes_ptr + output_offsets, mask=mask_bn, other=0.0)
            grad_pre_reset += grad_spike * d_spike_d_membrane

        if needs_weight_grad:
            input_values = tl.load(
                inputs_ptr
                + t * batch * features
                + offsets_f[:, None]
                + offsets_b[None, :] * features,
                mask=(offsets_f[:, None] < features) & (offsets_b[None, :] < batch),
                other=0.0,
            )
            weight_acc += tl.dot(input_values, grad_pre_reset)
        if needs_bias_grad:
            bias_acc += tl.sum(grad_pre_reset, axis=0)

        grad_membrane = grad_pre_reset * decay

    if needs_weight_grad:
        weight_offsets = offsets_f[:, None] * neurons + offsets_n[None, :]
        tl.atomic_add(
            grad_weight_ptr + weight_offsets,
            weight_acc,
            sem="relaxed",
            mask=(offsets_f[:, None] < features) & (offsets_n[None, :] < neurons),
        )
    if needs_bias_grad:
        tl.atomic_add(
            grad_bias_ptr + offsets_n,
            bias_acc,
            sem="relaxed",
            mask=(offsets_n < neurons) & (program_f == 0),
        )


def linear_surrogate_lif_backward(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    pre_reset_membranes: torch.Tensor,
    spikes: torch.Tensor,
    grad_final_membrane: torch.Tensor | None,
    grad_spikes: torch.Tensor | None,
    params: LIFParams,
    *,
    surrogate: SurrogateName,
    surrogate_slope: float,
    needs_input_grad: bool = True,
    needs_weight_grad: bool = True,
    needs_bias_grad: bool = True,
    block_b: int = 16,
    block_n: int = 32,
    block_f: int = 32,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Run Triton backward for fused dense projection plus surrogate LIF."""

    for name, value in [("block_b", block_b), ("block_n", block_n), ("block_f", block_f)]:
        if value <= 0 or value & (value - 1):
            raise ValueError(f"{name} must be a positive power of two")
    if min(block_b, block_n, block_f) < 16:
        raise ValueError("block_b, block_n, and block_f must be at least 16 for Triton dot")
    if inputs.ndim != 3:
        raise ValueError(f"inputs must be shaped [T, B, F]; got {tuple(inputs.shape)}")
    if weight.ndim != 2:
        raise ValueError(f"weight must be shaped [F, N]; got {tuple(weight.shape)}")
    if not inputs.is_cuda:
        raise ValueError("Triton linear surrogate LIF backward requires CUDA inputs")
    if weight.device != inputs.device or weight.dtype != inputs.dtype:
        raise ValueError("weight must have the same device and dtype as inputs")
    if inputs.shape[2] != weight.shape[0]:
        raise ValueError("inputs.shape[2] must match weight.shape[0]")
    if pre_reset_membranes.shape != (inputs.shape[0], inputs.shape[1], weight.shape[1]):
        raise ValueError("pre_reset_membranes must be shaped [T, B, N]")
    if spikes.shape != pre_reset_membranes.shape:
        raise ValueError("spikes must have the same shape as pre_reset_membranes")

    contiguous_inputs = inputs.contiguous()
    contiguous_weight = weight.contiguous()
    contiguous_pre_reset = pre_reset_membranes.contiguous()
    contiguous_spikes = spikes.contiguous()
    transposed_weight = contiguous_weight.t().contiguous()
    timesteps, batch, features = contiguous_inputs.shape
    neurons = contiguous_weight.shape[1]

    if not needs_input_grad and (needs_weight_grad or needs_bias_grad):
        grad_weight = torch.zeros_like(contiguous_weight) if needs_weight_grad else None
        grad_bias = (
            torch.zeros((neurons,), dtype=inputs.dtype, device=inputs.device)
            if needs_bias_grad
            else None
        )
        has_grad_final = grad_final_membrane is not None
        contiguous_grad_final = (
            contiguous_pre_reset
            if grad_final_membrane is None
            else grad_final_membrane.contiguous()
        )
        has_grad_spikes = grad_spikes is not None
        contiguous_grad_spikes = (
            contiguous_pre_reset if grad_spikes is None else grad_spikes.contiguous()
        )
        weight_ptr = contiguous_weight if grad_weight is None else grad_weight
        bias_ptr = contiguous_weight if grad_bias is None else grad_bias
        recurrent_grid = (
            triton.cdiv(features, block_f),
            triton.cdiv(neurons, block_n),
            triton.cdiv(batch, block_b),
        )
        _linear_surrogate_lif_backward_weight_bias_recurrent_kernel[recurrent_grid](
            contiguous_inputs,
            contiguous_pre_reset,
            contiguous_spikes,
            contiguous_grad_final,
            contiguous_grad_spikes,
            weight_ptr,
            bias_ptr,
            timesteps,  # pyright: ignore[reportArgumentType]
            batch,  # pyright: ignore[reportArgumentType]
            features,  # pyright: ignore[reportArgumentType]
            neurons,  # pyright: ignore[reportArgumentType]
            params.decay,  # pyright: ignore[reportArgumentType]
            params.threshold,  # pyright: ignore[reportArgumentType]
            params.reset,  # pyright: ignore[reportArgumentType]
            surrogate_slope,  # pyright: ignore[reportArgumentType]
            surrogate_id(surrogate),  # pyright: ignore[reportArgumentType]
            has_grad_final,  # pyright: ignore[reportArgumentType]
            has_grad_spikes,  # pyright: ignore[reportArgumentType]
            needs_weight_grad,  # pyright: ignore[reportArgumentType]
            needs_bias_grad,  # pyright: ignore[reportArgumentType]
            block_f,  # pyright: ignore[reportArgumentType]
            block_n,  # pyright: ignore[reportArgumentType]
            block_b,  # pyright: ignore[reportArgumentType]
        )
        return None, grad_weight, grad_bias

    grad_current, _grad_initial = surrogate_lif_backward(
        contiguous_pre_reset,
        contiguous_spikes,
        grad_final_membrane,
        grad_spikes,
        params,
        surrogate=surrogate,
        surrogate_slope=surrogate_slope,
        block_size=block_n,
    )

    grad_inputs = None
    if needs_input_grad:
        grad_inputs = torch.empty_like(contiguous_inputs)
        inputs_grid = (
            timesteps,
            triton.cdiv(batch, block_b),
            triton.cdiv(features, block_f),
        )
        _linear_surrogate_lif_backward_inputs_kernel[inputs_grid](
            grad_current,
            transposed_weight,
            grad_inputs,
            timesteps,  # pyright: ignore[reportArgumentType]
            batch,  # pyright: ignore[reportArgumentType]
            features,  # pyright: ignore[reportArgumentType]
            neurons,  # pyright: ignore[reportArgumentType]
            block_b,  # pyright: ignore[reportArgumentType]
            block_f,  # pyright: ignore[reportArgumentType]
            block_n,  # pyright: ignore[reportArgumentType]
        )

    grad_weight = None
    if needs_weight_grad:
        grad_weight = torch.empty_like(contiguous_weight)
        weight_grid = (triton.cdiv(features, block_f), triton.cdiv(neurons, block_n))
        _linear_surrogate_lif_backward_weight_kernel[weight_grid](
            contiguous_inputs,
            grad_current,
            grad_weight,
            timesteps,  # pyright: ignore[reportArgumentType]
            batch,  # pyright: ignore[reportArgumentType]
            features,  # pyright: ignore[reportArgumentType]
            neurons,  # pyright: ignore[reportArgumentType]
            block_f,  # pyright: ignore[reportArgumentType]
            block_n,  # pyright: ignore[reportArgumentType]
            block_b,  # pyright: ignore[reportArgumentType]
        )

    grad_bias = None
    if needs_bias_grad:
        grad_bias = torch.empty((neurons,), dtype=inputs.dtype, device=inputs.device)
        bias_grid = (triton.cdiv(neurons, block_n),)
        _linear_surrogate_lif_backward_bias_kernel[bias_grid](
            grad_current,
            grad_bias,
            timesteps,  # pyright: ignore[reportArgumentType]
            batch,  # pyright: ignore[reportArgumentType]
            neurons,  # pyright: ignore[reportArgumentType]
            block_n,  # pyright: ignore[reportArgumentType]
            block_b,  # pyright: ignore[reportArgumentType]
        )

    return grad_inputs, grad_weight, grad_bias


@triton.jit
def _linear_surrogate_lif_checkpoint_backward_weight_bias_kernel(
    inputs_ptr,
    weight_ptr,
    bias_ptr,
    chunk_start_ptr,
    grad_final_ptr,
    grad_spikes_ptr,
    grad_spike_rate_ptr,
    grad_weight_ptr,
    grad_bias_ptr,
    timesteps: tl.constexpr,
    batch: tl.constexpr,
    features: tl.constexpr,
    neurons: tl.constexpr,
    num_chunks: tl.constexpr,
    decay: tl.constexpr,
    threshold: tl.constexpr,
    reset: tl.constexpr,
    surrogate_slope: tl.constexpr,
    surrogate_id: tl.constexpr,
    has_bias: tl.constexpr,
    has_grad_final: tl.constexpr,
    has_grad_spikes: tl.constexpr,
    has_grad_spike_rate: tl.constexpr,
    spike_rate_scale: tl.constexpr,
    needs_weight_grad: tl.constexpr,
    needs_bias_grad: tl.constexpr,
    checkpoint_size: tl.constexpr,
    block_f: tl.constexpr,
    block_n: tl.constexpr,
    block_b: tl.constexpr,
):
    program_f = tl.program_id(0)
    program_n = tl.program_id(1)
    program_b = tl.program_id(2)
    offsets_f = program_f * block_f + tl.arange(0, block_f)
    offsets_n = program_n * block_n + tl.arange(0, block_n)
    offsets_b = program_b * block_b + tl.arange(0, block_b)
    mask_bn = (offsets_b[:, None] < batch) & (offsets_n[None, :] < neurons)

    if has_grad_final:
        grad_membrane = tl.load(
            grad_final_ptr + offsets_b[:, None] * neurons + offsets_n[None, :],
            mask=mask_bn,
            other=0.0,
        )
    else:
        grad_membrane = tl.zeros((block_b, block_n), tl.float32)

    weight_acc = tl.zeros((block_f, block_n), tl.float32)
    bias_acc = tl.zeros((block_n,), tl.float32)

    for reverse_chunk in range(num_chunks):
        chunk_index = num_chunks - 1 - reverse_chunk
        chunk_start = chunk_index * checkpoint_size
        chunk_stop = chunk_start + checkpoint_size
        if chunk_stop > timesteps:
            chunk_stop = timesteps

        for reverse_local in range(checkpoint_size):
            t = chunk_stop - 1 - reverse_local
            if t >= chunk_start:
                membrane = tl.load(
                    chunk_start_ptr
                    + chunk_index * batch * neurons
                    + offsets_b[:, None] * neurons
                    + offsets_n[None, :],
                    mask=mask_bn,
                    other=0.0,
                )
                pre_reset = membrane
                spike = membrane >= threshold

                for replay_local in range(checkpoint_size):
                    replay_t = chunk_start + replay_local
                    if replay_t <= t and replay_t < timesteps:
                        current = tl.zeros((block_b, block_n), tl.float32)
                        for f_start in range(0, features, block_f):
                            offsets_replay_f = f_start + tl.arange(0, block_f)
                            input_values = tl.load(
                                inputs_ptr
                                + replay_t * batch * features
                                + offsets_b[:, None] * features
                                + offsets_replay_f[None, :],
                                mask=(offsets_b[:, None] < batch)
                                & (offsets_replay_f[None, :] < features),
                                other=0.0,
                            )
                            weight_values = tl.load(
                                weight_ptr
                                + offsets_replay_f[:, None] * neurons
                                + offsets_n[None, :],
                                mask=(offsets_replay_f[:, None] < features)
                                & (offsets_n[None, :] < neurons),
                                other=0.0,
                            )
                            current += tl.dot(input_values, weight_values)
                        if has_bias:
                            bias = tl.load(
                                bias_ptr + offsets_n,
                                mask=offsets_n < neurons,
                                other=0.0,
                            )
                            current += bias[None, :]

                        pre_reset = membrane * decay + current
                        spike = pre_reset >= threshold
                        membrane = tl.where(spike, reset, pre_reset)

                centered = surrogate_slope * (pre_reset - threshold)
                d_spike_d_membrane = surrogate_slope * _surrogate_derivative(centered, surrogate_id)
                grad_pre_reset = grad_membrane * (
                    (1.0 - spike) + (reset - pre_reset) * d_spike_d_membrane
                )
                if has_grad_spikes:
                    grad_spike = tl.load(
                        grad_spikes_ptr
                        + t * batch * neurons
                        + offsets_b[:, None] * neurons
                        + offsets_n[None, :],
                        mask=mask_bn,
                        other=0.0,
                    )
                    grad_pre_reset += grad_spike * d_spike_d_membrane
                if has_grad_spike_rate:
                    grad_spike_rate = tl.load(grad_spike_rate_ptr)
                    grad_pre_reset += grad_spike_rate * spike_rate_scale * d_spike_d_membrane

                if needs_weight_grad:
                    input_values = tl.load(
                        inputs_ptr
                        + t * batch * features
                        + offsets_f[:, None]
                        + offsets_b[None, :] * features,
                        mask=(offsets_f[:, None] < features) & (offsets_b[None, :] < batch),
                        other=0.0,
                    )
                    weight_acc += tl.dot(input_values, grad_pre_reset)
                if needs_bias_grad:
                    bias_acc += tl.sum(grad_pre_reset, axis=0)

                grad_membrane = grad_pre_reset * decay

    if needs_weight_grad:
        weight_offsets = offsets_f[:, None] * neurons + offsets_n[None, :]
        tl.atomic_add(
            grad_weight_ptr + weight_offsets,
            weight_acc,
            sem="relaxed",
            mask=(offsets_f[:, None] < features) & (offsets_n[None, :] < neurons),
        )
    if needs_bias_grad:
        tl.atomic_add(
            grad_bias_ptr + offsets_n,
            bias_acc,
            sem="relaxed",
            mask=(offsets_n < neurons) & (program_f == 0),
        )


@triton.jit
def _linear_surrogate_lif_checkpoint_recompute_chunk_kernel(
    inputs_ptr,
    weight_ptr,
    bias_ptr,
    chunk_start_ptr,
    pre_reset_scratch_ptr,
    spikes_scratch_ptr,
    chunk_index: tl.constexpr,
    chunk_start: tl.constexpr,
    chunk_len: tl.constexpr,
    batch: tl.constexpr,
    features: tl.constexpr,
    neurons: tl.constexpr,
    decay: tl.constexpr,
    threshold: tl.constexpr,
    reset: tl.constexpr,
    has_bias: tl.constexpr,
    block_b: tl.constexpr,
    block_n: tl.constexpr,
    block_f: tl.constexpr,
):
    program_b = tl.program_id(0)
    program_n = tl.program_id(1)
    offsets_b = program_b * block_b + tl.arange(0, block_b)
    offsets_n = program_n * block_n + tl.arange(0, block_n)
    mask_bn = (offsets_b[:, None] < batch) & (offsets_n[None, :] < neurons)
    membrane = tl.load(
        chunk_start_ptr
        + chunk_index * batch * neurons
        + offsets_b[:, None] * neurons
        + offsets_n[None, :],
        mask=mask_bn,
        other=0.0,
    )

    for local_t in range(chunk_len):
        t = chunk_start + local_t
        current = tl.zeros((block_b, block_n), tl.float32)
        for f_start in range(0, features, block_f):
            offsets_f = f_start + tl.arange(0, block_f)
            input_values = tl.load(
                inputs_ptr
                + t * batch * features
                + offsets_b[:, None] * features
                + offsets_f[None, :],
                mask=(offsets_b[:, None] < batch) & (offsets_f[None, :] < features),
                other=0.0,
            )
            weight_values = tl.load(
                weight_ptr + offsets_f[:, None] * neurons + offsets_n[None, :],
                mask=(offsets_f[:, None] < features) & (offsets_n[None, :] < neurons),
                other=0.0,
            )
            current += tl.dot(input_values, weight_values)
        if has_bias:
            bias = tl.load(bias_ptr + offsets_n, mask=offsets_n < neurons, other=0.0)
            current += bias[None, :]

        pre_reset = membrane * decay + current
        spike = pre_reset >= threshold
        scratch_offsets = (
            local_t * batch * neurons + offsets_b[:, None] * neurons + offsets_n[None, :]
        )
        tl.store(pre_reset_scratch_ptr + scratch_offsets, pre_reset, mask=mask_bn)
        tl.store(spikes_scratch_ptr + scratch_offsets, spike.to(tl.float32), mask=mask_bn)
        membrane = tl.where(spike, reset, pre_reset)


@triton.jit
def _linear_surrogate_lif_checkpoint_recompute_chunk_packed_spikes_kernel(
    inputs_ptr,
    weight_ptr,
    bias_ptr,
    chunk_start_ptr,
    pre_reset_scratch_ptr,
    chunk_index: tl.constexpr,
    chunk_start: tl.constexpr,
    chunk_len: tl.constexpr,
    batch: tl.constexpr,
    features: tl.constexpr,
    neurons: tl.constexpr,
    decay: tl.constexpr,
    threshold: tl.constexpr,
    reset: tl.constexpr,
    has_bias: tl.constexpr,
    block_b: tl.constexpr,
    block_f: tl.constexpr,
):
    program_b = tl.program_id(0)
    program_word = tl.program_id(1)
    offsets_b = program_b * block_b + tl.arange(0, block_b)
    bit_offsets = tl.arange(0, 32)
    offsets_n = program_word * 32 + bit_offsets
    mask_bn = (offsets_b[:, None] < batch) & (offsets_n[None, :] < neurons)
    membrane = tl.load(
        chunk_start_ptr
        + chunk_index * batch * neurons
        + offsets_b[:, None] * neurons
        + offsets_n[None, :],
        mask=mask_bn,
        other=0.0,
    )

    for local_t in range(chunk_len):
        t = chunk_start + local_t
        current = tl.zeros((block_b, 32), tl.float32)
        for f_start in range(0, features, block_f):
            offsets_f = f_start + tl.arange(0, block_f)
            input_values = tl.load(
                inputs_ptr
                + t * batch * features
                + offsets_b[:, None] * features
                + offsets_f[None, :],
                mask=(offsets_b[:, None] < batch) & (offsets_f[None, :] < features),
                other=0.0,
            )
            weight_values = tl.load(
                weight_ptr + offsets_f[:, None] * neurons + offsets_n[None, :],
                mask=(offsets_f[:, None] < features) & (offsets_n[None, :] < neurons),
                other=0.0,
            )
            current += tl.dot(input_values, weight_values)
        if has_bias:
            bias = tl.load(bias_ptr + offsets_n, mask=offsets_n < neurons, other=0.0)
            current += bias[None, :]

        pre_reset = membrane * decay + current
        spike = pre_reset >= threshold
        scratch_offsets = (
            local_t * batch * neurons + offsets_b[:, None] * neurons + offsets_n[None, :]
        )
        tl.store(pre_reset_scratch_ptr + scratch_offsets, pre_reset, mask=mask_bn)
        membrane = tl.where(spike, reset, pre_reset)


@triton.jit
def _linear_surrogate_lif_checkpoint_backward_chunk_kernel(
    inputs_ptr,
    weight_ptr,
    pre_reset_scratch_ptr,
    spikes_scratch_ptr,
    grad_next_ptr,
    grad_spikes_ptr,
    grad_inputs_ptr,
    grad_prev_ptr,
    grad_weight_ptr,
    grad_bias_ptr,
    chunk_start: tl.constexpr,
    chunk_len: tl.constexpr,
    batch: tl.constexpr,
    features: tl.constexpr,
    neurons: tl.constexpr,
    decay: tl.constexpr,
    threshold: tl.constexpr,
    reset: tl.constexpr,
    surrogate_slope: tl.constexpr,
    surrogate_id: tl.constexpr,
    has_grad_spikes: tl.constexpr,
    needs_input_grad: tl.constexpr,
    needs_weight_grad: tl.constexpr,
    needs_bias_grad: tl.constexpr,
    block_f: tl.constexpr,
    block_n: tl.constexpr,
    block_b: tl.constexpr,
):
    program_f = tl.program_id(0)
    program_n = tl.program_id(1)
    program_b = tl.program_id(2)
    offsets_f = program_f * block_f + tl.arange(0, block_f)
    offsets_n = program_n * block_n + tl.arange(0, block_n)
    offsets_b = program_b * block_b + tl.arange(0, block_b)
    mask_bn = (offsets_b[:, None] < batch) & (offsets_n[None, :] < neurons)
    grad_membrane = tl.load(
        grad_next_ptr + offsets_b[:, None] * neurons + offsets_n[None, :],
        mask=mask_bn,
        other=0.0,
    )
    weight_acc = tl.zeros((block_f, block_n), tl.float32)
    bias_acc = tl.zeros((block_n,), tl.float32)

    for reverse_local in range(chunk_len):
        local_t = chunk_len - 1 - reverse_local
        t = chunk_start + local_t
        scratch_offsets = (
            local_t * batch * neurons + offsets_b[:, None] * neurons + offsets_n[None, :]
        )
        pre_reset = tl.load(pre_reset_scratch_ptr + scratch_offsets, mask=mask_bn, other=0.0)
        spike = tl.load(spikes_scratch_ptr + scratch_offsets, mask=mask_bn, other=0.0)
        centered = surrogate_slope * (pre_reset - threshold)
        d_spike_d_membrane = surrogate_slope * _surrogate_derivative(centered, surrogate_id)
        grad_pre_reset = grad_membrane * ((1.0 - spike) + (reset - pre_reset) * d_spike_d_membrane)
        if has_grad_spikes:
            grad_spike = tl.load(
                grad_spikes_ptr
                + t * batch * neurons
                + offsets_b[:, None] * neurons
                + offsets_n[None, :],
                mask=mask_bn,
                other=0.0,
            )
            grad_pre_reset += grad_spike * d_spike_d_membrane

        if needs_input_grad:
            weight_values = tl.load(
                weight_ptr + offsets_f[:, None] * neurons + offsets_n[None, :],
                mask=(offsets_f[:, None] < features) & (offsets_n[None, :] < neurons),
                other=0.0,
            )
            grad_input = tl.dot(grad_pre_reset, tl.trans(weight_values))
            input_offsets = (
                t * batch * features + offsets_b[:, None] * features + offsets_f[None, :]
            )
            tl.atomic_add(
                grad_inputs_ptr + input_offsets,
                grad_input,
                sem="relaxed",
                mask=(offsets_b[:, None] < batch) & (offsets_f[None, :] < features),
            )
        if needs_weight_grad:
            input_values = tl.load(
                inputs_ptr
                + t * batch * features
                + offsets_f[:, None]
                + offsets_b[None, :] * features,
                mask=(offsets_f[:, None] < features) & (offsets_b[None, :] < batch),
                other=0.0,
            )
            weight_acc += tl.dot(input_values, grad_pre_reset)
        if needs_bias_grad:
            bias_acc += tl.sum(grad_pre_reset, axis=0)

        grad_membrane = grad_pre_reset * decay

    if needs_weight_grad:
        weight_offsets = offsets_f[:, None] * neurons + offsets_n[None, :]
        tl.atomic_add(
            grad_weight_ptr + weight_offsets,
            weight_acc,
            sem="relaxed",
            mask=(offsets_f[:, None] < features) & (offsets_n[None, :] < neurons),
        )
    if needs_bias_grad:
        tl.atomic_add(
            grad_bias_ptr + offsets_n,
            bias_acc,
            sem="relaxed",
            mask=(offsets_n < neurons) & (program_f == 0),
        )
    if program_f == 0:
        tl.store(
            grad_prev_ptr + offsets_b[:, None] * neurons + offsets_n[None, :],
            grad_membrane,
            mask=mask_bn,
        )


@triton.jit
def _linear_surrogate_lif_checkpoint_backward_chunk_packed_spikes_kernel(
    inputs_ptr,
    weight_ptr,
    pre_reset_scratch_ptr,
    grad_next_ptr,
    grad_spikes_ptr,
    grad_spike_rate_ptr,
    grad_spike_rates_ptr,
    grad_inputs_ptr,
    grad_prev_ptr,
    grad_weight_ptr,
    grad_bias_ptr,
    chunk_start: tl.constexpr,
    chunk_len: tl.constexpr,
    batch: tl.constexpr,
    features: tl.constexpr,
    neurons: tl.constexpr,
    decay: tl.constexpr,
    threshold: tl.constexpr,
    reset: tl.constexpr,
    surrogate_slope: tl.constexpr,
    surrogate_id: tl.constexpr,
    has_grad_spikes: tl.constexpr,
    has_grad_spike_rate: tl.constexpr,
    has_grad_spike_rates: tl.constexpr,
    spike_rate_scale: tl.constexpr,
    spike_rates_scale: tl.constexpr,
    needs_input_grad: tl.constexpr,
    needs_weight_grad: tl.constexpr,
    needs_bias_grad: tl.constexpr,
    block_f: tl.constexpr,
    block_n: tl.constexpr,
    block_b: tl.constexpr,
):
    program_f = tl.program_id(0)
    program_n = tl.program_id(1)
    program_b = tl.program_id(2)
    offsets_f = program_f * block_f + tl.arange(0, block_f)
    offsets_n = program_n * block_n + tl.arange(0, block_n)
    offsets_b = program_b * block_b + tl.arange(0, block_b)
    mask_bn = (offsets_b[:, None] < batch) & (offsets_n[None, :] < neurons)
    grad_membrane = tl.load(
        grad_next_ptr + offsets_b[:, None] * neurons + offsets_n[None, :],
        mask=mask_bn,
        other=0.0,
    )
    weight_acc = tl.zeros((block_f, block_n), tl.float32)
    bias_acc = tl.zeros((block_n,), tl.float32)

    for reverse_local in range(chunk_len):
        local_t = chunk_len - 1 - reverse_local
        t = chunk_start + local_t
        scratch_offsets = (
            local_t * batch * neurons + offsets_b[:, None] * neurons + offsets_n[None, :]
        )
        pre_reset = tl.load(pre_reset_scratch_ptr + scratch_offsets, mask=mask_bn, other=0.0)
        spike = (pre_reset >= threshold).to(tl.float32)
        centered = surrogate_slope * (pre_reset - threshold)
        d_spike_d_membrane = surrogate_slope * _surrogate_derivative(centered, surrogate_id)
        grad_pre_reset = grad_membrane * ((1.0 - spike) + (reset - pre_reset) * d_spike_d_membrane)
        if has_grad_spikes:
            grad_spike = tl.load(
                grad_spikes_ptr
                + t * batch * neurons
                + offsets_b[:, None] * neurons
                + offsets_n[None, :],
                mask=mask_bn,
                other=0.0,
            )
            grad_pre_reset += grad_spike * d_spike_d_membrane
        if has_grad_spike_rate:
            grad_spike_rate = tl.load(grad_spike_rate_ptr)
            grad_pre_reset += grad_spike_rate * spike_rate_scale * d_spike_d_membrane
        if has_grad_spike_rates:
            grad_spike_rate_values = tl.load(
                grad_spike_rates_ptr + offsets_b[:, None] * neurons + offsets_n[None, :],
                mask=mask_bn,
                other=0.0,
            )
            grad_pre_reset += grad_spike_rate_values * spike_rates_scale * d_spike_d_membrane

        if needs_input_grad:
            weight_values = tl.load(
                weight_ptr + offsets_f[:, None] * neurons + offsets_n[None, :],
                mask=(offsets_f[:, None] < features) & (offsets_n[None, :] < neurons),
                other=0.0,
            )
            grad_input = tl.dot(grad_pre_reset, tl.trans(weight_values))
            input_offsets = (
                t * batch * features + offsets_b[:, None] * features + offsets_f[None, :]
            )
            tl.atomic_add(
                grad_inputs_ptr + input_offsets,
                grad_input,
                sem="relaxed",
                mask=(offsets_b[:, None] < batch) & (offsets_f[None, :] < features),
            )
        if needs_weight_grad:
            input_values = tl.load(
                inputs_ptr
                + t * batch * features
                + offsets_f[:, None]
                + offsets_b[None, :] * features,
                mask=(offsets_f[:, None] < features) & (offsets_b[None, :] < batch),
                other=0.0,
            )
            weight_acc += tl.dot(input_values, grad_pre_reset)
        if needs_bias_grad:
            bias_acc += tl.sum(grad_pre_reset, axis=0)

        grad_membrane = grad_pre_reset * decay

    if needs_weight_grad:
        weight_offsets = offsets_f[:, None] * neurons + offsets_n[None, :]
        tl.atomic_add(
            grad_weight_ptr + weight_offsets,
            weight_acc,
            sem="relaxed",
            mask=(offsets_f[:, None] < features) & (offsets_n[None, :] < neurons),
        )
    if needs_bias_grad:
        tl.atomic_add(
            grad_bias_ptr + offsets_n,
            bias_acc,
            sem="relaxed",
            mask=(offsets_n < neurons) & (program_f == 0),
        )
    if program_f == 0:
        tl.store(
            grad_prev_ptr + offsets_b[:, None] * neurons + offsets_n[None, :],
            grad_membrane,
            mask=mask_bn,
        )


def linear_surrogate_lif_checkpoint_backward(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    chunk_start_membranes: torch.Tensor,
    grad_final_membrane: torch.Tensor | None,
    grad_spikes: torch.Tensor | None,
    params: LIFParams,
    *,
    surrogate: SurrogateName,
    surrogate_slope: float,
    grad_spike_rate: torch.Tensor | None = None,
    grad_spike_rates: torch.Tensor | None = None,
    needs_input_grad: bool = False,
    needs_weight_grad: bool = True,
    needs_bias_grad: bool = True,
    checkpoint_size: int,
    block_b: int = 16,
    block_n: int = 32,
    block_f: int = 32,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Run checkpointed Triton backward without saving full recurrent traces."""
    for name, value in [
        ("checkpoint_size", checkpoint_size),
        ("block_b", block_b),
        ("block_n", block_n),
        ("block_f", block_f),
    ]:
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    for name, value in [("block_b", block_b), ("block_n", block_n), ("block_f", block_f)]:
        if value & (value - 1):
            raise ValueError(f"{name} must be a power of two")
    if min(block_b, block_n, block_f) < 16:
        raise ValueError("block_b, block_n, and block_f must be at least 16 for Triton dot")
    if inputs.ndim != 3:
        raise ValueError(f"inputs must be shaped [T, B, F]; got {tuple(inputs.shape)}")
    if weight.ndim != 2:
        raise ValueError(f"weight must be shaped [F, N]; got {tuple(weight.shape)}")
    if not inputs.is_cuda:
        raise ValueError("Triton checkpoint LIF backward requires CUDA inputs")
    if weight.device != inputs.device or weight.dtype != inputs.dtype:
        raise ValueError("weight must have the same device and dtype as inputs")
    if inputs.shape[2] != weight.shape[0]:
        raise ValueError("inputs.shape[2] must match weight.shape[0]")

    contiguous_inputs = inputs.contiguous()
    contiguous_weight = weight.contiguous()
    contiguous_bias = None if bias is None else bias.contiguous()
    contiguous_chunks = chunk_start_membranes.contiguous()
    timesteps, batch, features = contiguous_inputs.shape
    neurons = contiguous_weight.shape[1]
    neuron_words = packed_last_dim_size(neurons)
    num_chunks = (timesteps + checkpoint_size - 1) // checkpoint_size
    if contiguous_chunks.shape != (num_chunks, batch, neurons):
        raise ValueError("chunk_start_membranes must be shaped [ceil(T / chunk), B, N]")
    if bias is not None:
        if bias.shape != (neurons,):
            raise ValueError("bias must be shaped [N]")
        if bias.device != inputs.device or bias.dtype != inputs.dtype:
            raise ValueError("bias must have the same device and dtype as inputs")
    if grad_final_membrane is not None and grad_final_membrane.shape != (batch, neurons):
        raise ValueError("grad_final_membrane must be shaped [B, N]")
    if grad_spikes is not None and grad_spikes.shape != (timesteps, batch, neurons):
        raise ValueError("grad_spikes must be shaped [T, B, N]")
    if grad_spike_rate is not None:
        if grad_spike_rate.shape != ():
            raise ValueError("grad_spike_rate must be a scalar tensor")
        if grad_spike_rate.device != inputs.device or grad_spike_rate.dtype != inputs.dtype:
            raise ValueError("grad_spike_rate must have the same device and dtype as inputs")
    if grad_spike_rates is not None:
        if grad_spike_rates.shape != (batch, neurons):
            raise ValueError("grad_spike_rates must be shaped [B, N]")
        if grad_spike_rates.device != inputs.device or grad_spike_rates.dtype != inputs.dtype:
            raise ValueError("grad_spike_rates must have the same device and dtype as inputs")

    grad_inputs = torch.zeros_like(contiguous_inputs) if needs_input_grad else None
    grad_weight = torch.zeros_like(contiguous_weight) if needs_weight_grad else None
    grad_bias = (
        torch.zeros((neurons,), dtype=inputs.dtype, device=inputs.device)
        if needs_bias_grad and bias is not None
        else None
    )
    grad_next = (
        torch.zeros((batch, neurons), dtype=inputs.dtype, device=inputs.device)
        if grad_final_membrane is None
        else grad_final_membrane.contiguous()
    )
    has_grad_spikes = grad_spikes is not None
    contiguous_grad_spikes = contiguous_chunks if grad_spikes is None else grad_spikes.contiguous()
    has_grad_spike_rate = grad_spike_rate is not None
    contiguous_grad_spike_rate = (
        contiguous_inputs if grad_spike_rate is None else grad_spike_rate.contiguous()
    )
    has_grad_spike_rates = grad_spike_rates is not None
    contiguous_grad_spike_rates = (
        contiguous_chunks if grad_spike_rates is None else grad_spike_rates.contiguous()
    )
    bias_ptr = contiguous_weight if contiguous_bias is None else contiguous_bias
    grad_inputs_ptr = contiguous_inputs if grad_inputs is None else grad_inputs
    grad_weight_ptr = contiguous_weight if grad_weight is None else grad_weight
    grad_bias_ptr = contiguous_weight if grad_bias is None else grad_bias
    pre_reset_scratch = torch.empty(
        (checkpoint_size, batch, neurons), dtype=inputs.dtype, device=inputs.device
    )
    recompute_grid = (triton.cdiv(batch, block_b), neuron_words)
    backward_grid = (
        triton.cdiv(features, block_f),
        triton.cdiv(neurons, block_n),
        triton.cdiv(batch, block_b),
    )
    if grad_final_membrane is None:
        grad_prev_buffers = (grad_next, torch.empty_like(grad_next))
        grad_prev_buffer_index = 1
    else:
        grad_prev_buffers = (torch.empty_like(grad_next), torch.empty_like(grad_next))
        grad_prev_buffer_index = 0

    for chunk_index in range(num_chunks - 1, -1, -1):
        chunk_start = chunk_index * checkpoint_size
        chunk_len = min(checkpoint_size, timesteps - chunk_start)
        _linear_surrogate_lif_checkpoint_recompute_chunk_packed_spikes_kernel[recompute_grid](
            contiguous_inputs,
            contiguous_weight,
            bias_ptr,
            contiguous_chunks,
            pre_reset_scratch,
            chunk_index,  # pyright: ignore[reportArgumentType]
            chunk_start,  # pyright: ignore[reportArgumentType]
            chunk_len,  # pyright: ignore[reportArgumentType]
            batch,  # pyright: ignore[reportArgumentType]
            features,  # pyright: ignore[reportArgumentType]
            neurons,  # pyright: ignore[reportArgumentType]
            params.decay,  # pyright: ignore[reportArgumentType]
            params.threshold,  # pyright: ignore[reportArgumentType]
            params.reset,  # pyright: ignore[reportArgumentType]
            bias is not None,  # pyright: ignore[reportArgumentType]
            block_b,  # pyright: ignore[reportArgumentType]
            block_f,  # pyright: ignore[reportArgumentType]
        )
        grad_prev = grad_prev_buffers[grad_prev_buffer_index]
        _linear_surrogate_lif_checkpoint_backward_chunk_packed_spikes_kernel[backward_grid](
            contiguous_inputs,
            contiguous_weight,
            pre_reset_scratch,
            grad_next,
            contiguous_grad_spikes,
            contiguous_grad_spike_rate,
            contiguous_grad_spike_rates,
            grad_inputs_ptr,
            grad_prev,
            grad_weight_ptr,
            grad_bias_ptr,
            chunk_start,  # pyright: ignore[reportArgumentType]
            chunk_len,  # pyright: ignore[reportArgumentType]
            batch,  # pyright: ignore[reportArgumentType]
            features,  # pyright: ignore[reportArgumentType]
            neurons,  # pyright: ignore[reportArgumentType]
            params.decay,  # pyright: ignore[reportArgumentType]
            params.threshold,  # pyright: ignore[reportArgumentType]
            params.reset,  # pyright: ignore[reportArgumentType]
            surrogate_slope,  # pyright: ignore[reportArgumentType]
            surrogate_id(surrogate),  # pyright: ignore[reportArgumentType]
            has_grad_spikes,  # pyright: ignore[reportArgumentType]
            has_grad_spike_rate,  # pyright: ignore[reportArgumentType]
            has_grad_spike_rates,  # pyright: ignore[reportArgumentType]
            1.0 / float(timesteps * batch * neurons),  # pyright: ignore[reportArgumentType]
            1.0 / float(timesteps),  # pyright: ignore[reportArgumentType]
            needs_input_grad,  # pyright: ignore[reportArgumentType]
            needs_weight_grad,  # pyright: ignore[reportArgumentType]
            grad_bias is not None,  # pyright: ignore[reportArgumentType]
            block_f,  # pyright: ignore[reportArgumentType]
            block_n,  # pyright: ignore[reportArgumentType]
            block_b,  # pyright: ignore[reportArgumentType]
        )
        grad_next = grad_prev
        grad_prev_buffer_index = 1 - grad_prev_buffer_index
    return grad_inputs, grad_weight, grad_bias


def linear_surrogate_lif_checkpoint_backward_replay_weight_bias(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    chunk_start_membranes: torch.Tensor,
    grad_final_membrane: torch.Tensor | None,
    grad_spikes: torch.Tensor | None,
    params: LIFParams,
    *,
    surrogate: SurrogateName,
    surrogate_slope: float,
    grad_spike_rate: torch.Tensor | None = None,
    needs_weight_grad: bool = True,
    needs_bias_grad: bool = True,
    checkpoint_size: int,
    block_b: int = 16,
    block_n: int = 32,
    block_f: int = 32,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Run memory-minimal checkpoint backward by replaying chunks in-kernel.

    This path avoids the chunk-sized pre-reset scratch used by
    ``linear_surrogate_lif_checkpoint_backward``. It is intended for benchmark
    and memory-floor experiments in the common no-``dinputs`` case; the replay
    work is O(checkpoint_size^2), so the scratch path remains the default.
    """

    for name, value in [
        ("checkpoint_size", checkpoint_size),
        ("block_b", block_b),
        ("block_n", block_n),
        ("block_f", block_f),
    ]:
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    for name, value in [("block_b", block_b), ("block_n", block_n), ("block_f", block_f)]:
        if value & (value - 1):
            raise ValueError(f"{name} must be a power of two")
    if min(block_b, block_n, block_f) < 16:
        raise ValueError("block_b, block_n, and block_f must be at least 16 for Triton dot")
    if inputs.ndim != 3:
        raise ValueError(f"inputs must be shaped [T, B, F]; got {tuple(inputs.shape)}")
    if weight.ndim != 2:
        raise ValueError(f"weight must be shaped [F, N]; got {tuple(weight.shape)}")
    if not inputs.is_cuda:
        raise ValueError("Triton replay checkpoint LIF backward requires CUDA inputs")
    if weight.device != inputs.device or weight.dtype != inputs.dtype:
        raise ValueError("weight must have the same device and dtype as inputs")
    if inputs.shape[2] != weight.shape[0]:
        raise ValueError("inputs.shape[2] must match weight.shape[0]")

    contiguous_inputs = inputs.contiguous()
    contiguous_weight = weight.contiguous()
    contiguous_bias = None if bias is None else bias.contiguous()
    contiguous_chunks = chunk_start_membranes.contiguous()
    timesteps, batch, features = contiguous_inputs.shape
    neurons = contiguous_weight.shape[1]
    num_chunks = (timesteps + checkpoint_size - 1) // checkpoint_size
    if contiguous_chunks.shape != (num_chunks, batch, neurons):
        raise ValueError("chunk_start_membranes must be shaped [ceil(T / chunk), B, N]")
    if bias is not None:
        if bias.shape != (neurons,):
            raise ValueError("bias must be shaped [N]")
        if bias.device != inputs.device or bias.dtype != inputs.dtype:
            raise ValueError("bias must have the same device and dtype as inputs")
    if grad_final_membrane is not None and grad_final_membrane.shape != (batch, neurons):
        raise ValueError("grad_final_membrane must be shaped [B, N]")
    if grad_spikes is not None and grad_spikes.shape != (timesteps, batch, neurons):
        raise ValueError("grad_spikes must be shaped [T, B, N]")
    if grad_spike_rate is not None:
        if grad_spike_rate.shape != ():
            raise ValueError("grad_spike_rate must be a scalar tensor")
        if grad_spike_rate.device != inputs.device or grad_spike_rate.dtype != inputs.dtype:
            raise ValueError("grad_spike_rate must have the same device and dtype as inputs")

    grad_weight = torch.zeros_like(contiguous_weight) if needs_weight_grad else None
    grad_bias = (
        torch.zeros((neurons,), dtype=inputs.dtype, device=inputs.device)
        if needs_bias_grad and bias is not None
        else None
    )
    if grad_weight is None and grad_bias is None:
        return None, None

    bias_ptr = contiguous_weight if contiguous_bias is None else contiguous_bias
    grad_final_ptr = contiguous_chunks if grad_final_membrane is None else grad_final_membrane
    grad_spikes_ptr = contiguous_chunks if grad_spikes is None else grad_spikes.contiguous()
    grad_spike_rate_ptr = contiguous_inputs if grad_spike_rate is None else grad_spike_rate
    grad_weight_ptr = contiguous_weight if grad_weight is None else grad_weight
    grad_bias_ptr = contiguous_weight if grad_bias is None else grad_bias
    grid = (
        triton.cdiv(features, block_f),
        triton.cdiv(neurons, block_n),
        triton.cdiv(batch, block_b),
    )
    _linear_surrogate_lif_checkpoint_backward_weight_bias_kernel[grid](
        contiguous_inputs,
        contiguous_weight,
        bias_ptr,
        contiguous_chunks,
        grad_final_ptr,
        grad_spikes_ptr,
        grad_spike_rate_ptr,
        grad_weight_ptr,
        grad_bias_ptr,
        timesteps,  # pyright: ignore[reportArgumentType]
        batch,  # pyright: ignore[reportArgumentType]
        features,  # pyright: ignore[reportArgumentType]
        neurons,  # pyright: ignore[reportArgumentType]
        num_chunks,  # pyright: ignore[reportArgumentType]
        params.decay,  # pyright: ignore[reportArgumentType]
        params.threshold,  # pyright: ignore[reportArgumentType]
        params.reset,  # pyright: ignore[reportArgumentType]
        surrogate_slope,  # pyright: ignore[reportArgumentType]
        surrogate_id(surrogate),  # pyright: ignore[reportArgumentType]
        bias is not None,  # pyright: ignore[reportArgumentType]
        grad_final_membrane is not None,  # pyright: ignore[reportArgumentType]
        grad_spikes is not None,  # pyright: ignore[reportArgumentType]
        grad_spike_rate is not None,  # pyright: ignore[reportArgumentType]
        1.0 / float(timesteps * batch * neurons),  # pyright: ignore[reportArgumentType]
        grad_weight is not None,  # pyright: ignore[reportArgumentType]
        grad_bias is not None,  # pyright: ignore[reportArgumentType]
        checkpoint_size,  # pyright: ignore[reportArgumentType]
        block_f,  # pyright: ignore[reportArgumentType]
        block_n,  # pyright: ignore[reportArgumentType]
        block_b,  # pyright: ignore[reportArgumentType]
    )
    return grad_weight, grad_bias
