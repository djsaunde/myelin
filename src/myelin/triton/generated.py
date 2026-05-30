"""Generated Triton kernels from neuron IR."""

from __future__ import annotations

import linecache
from collections.abc import Callable, Mapping
from types import ModuleType
from typing import Any

import torch

from myelin._optional import require_triton
from myelin.codegen import (
    lower_neuron_to_ssa,
    render_lif_surrogate_backward_step_body,
    render_triton_step_body,
)
from myelin.dsl import (
    Expr,
    NeuronIR,
    alif_ir,
    izhikevich_ir,
    lif_ir,
    validate_generated_forward_ir,
)
from myelin.neurons import (
    ALIFParams,
    ALIFState,
    IzhikevichParams,
    IzhikevichState,
    LIFParams,
    LIFState,
)
from myelin.packing import packed_last_dim_size
from myelin.surrogates import SurrogateName

triton = require_triton()
import triton.language as tl  # noqa: E402

_GeneratedNeuronForwardKey = tuple[object, ...]
_GENERATED_NEURON_FORWARD_CACHE: dict[_GeneratedNeuronForwardKey, Any] = {}
_GeneratedKernelKey = tuple[str, str, str, str]
_GENERATED_KERNEL_CACHE: dict[_GeneratedKernelKey, Any] = {}
_TORCH_VERSION_KEY = str(torch.__version__)


def render_forward_kernel_source(
    ir: NeuronIR,
    *,
    function_name: str,
) -> str:
    """Render a full Triton forward kernel source from neuron IR."""

    _validate_forward_kernel_ir(ir)

    initial_args = "\n".join(f"    initial_{state_name}_ptr," for state_name in ir.state)
    final_args = "\n".join(f"    final_{state_name}_ptr," for state_name in ir.state)
    param_args = "\n".join(f"    {param_name}: tl.constexpr," for param_name in ir.params)
    state_loads = "\n".join(
        (f"    {state_name} = tl.load(initial_{state_name}_ptr + offsets, mask=mask, other=0.0)")
        for state_name in ir.state
    )
    final_stores = "\n".join(
        (f"    tl.store(final_{state_name}_ptr + offsets, {state_name}, mask=mask)")
        for state_name in ir.state
    )

    step_body = render_triton_step_body(
        lower_neuron_to_ssa(ir, dialect="triton"),
        variable_map={
            **{state_name: state_name for state_name in ir.state},
            "input_current": "input_current",
            **{param_name: param_name for param_name in ir.params},
        },
        output_map={"spike": "spike"},
        indent="        ",
    )
    return f"""@triton.jit
def {function_name}(
    inputs_ptr,
{initial_args}
{final_args}
    spikes_ptr,
    total_elements: tl.constexpr,
    timesteps: tl.constexpr,
{param_args}
    block_size: tl.constexpr,
):
    program_id = tl.program_id(0)
    offsets = program_id * block_size + tl.arange(0, block_size)
    mask = offsets < total_elements
{state_loads}

    for t in range(timesteps):
        input_current = tl.load(inputs_ptr + t * total_elements + offsets, mask=mask, other=0.0)
{step_body}
        tl.store(spikes_ptr + t * total_elements + offsets, spike.to(tl.float32), mask=mask)

{final_stores}
"""


def render_lif_forward_kernel_source(
    *, function_name: str = "_generated_lif_forward_kernel"
) -> str:
    """Render a full Triton LIF forward kernel source from ``lif_ir()``."""

    return render_forward_kernel_source(lif_ir(), function_name=function_name)


def render_alif_forward_kernel_source(
    *, function_name: str = "_generated_alif_forward_kernel"
) -> str:
    """Render a full Triton ALIF forward kernel source from ``alif_ir()``."""

    return render_forward_kernel_source(alif_ir(), function_name=function_name)


def render_izhikevich_forward_kernel_source(
    *, function_name: str = "_generated_izhikevich_forward_kernel"
) -> str:
    """Render a full Triton Izhikevich forward kernel from ``izhikevich_ir()``."""

    return render_forward_kernel_source(izhikevich_ir(), function_name=function_name)


def render_lif_surrogate_backward_kernel_source(
    surrogate: str,
    *,
    function_name: str = "_generated_lif_surrogate_backward_kernel",
) -> str:
    """Render a generated Triton surrogate LIF backward kernel source."""

    step_body = render_lif_surrogate_backward_step_body(
        surrogate,
        dialect="triton",
        pre_reset_name="pre_reset",
        spike_name="spike",
        grad_membrane_name="grad_membrane",
        grad_spike_name="grad_spike",
        threshold_name="threshold",
        reset_name="reset",
        decay_name="decay",
        surrogate_slope_name="surrogate_slope",
        grad_pre_reset_name="grad_pre_reset",
        indent="        ",
    )
    return f"""@triton.jit
def {function_name}(
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
{step_body}
        tl.store(grad_inputs_ptr + time_offsets, grad_pre_reset, mask=mask)

    tl.store(grad_initial_ptr + offsets, grad_membrane, mask=mask)
"""


def render_lif_surrogate_backward_packed_spikes_kernel_source(
    surrogate: str,
    *,
    function_name: str = "_generated_lif_surrogate_backward_packed_spikes_kernel",
) -> str:
    """Render a generated Triton surrogate LIF backward kernel using packed spikes."""

    step_body = render_lif_surrogate_backward_step_body(
        surrogate,
        dialect="triton",
        pre_reset_name="pre_reset",
        spike_name="spike",
        grad_membrane_name="grad_membrane",
        grad_spike_name="grad_spike",
        threshold_name="threshold",
        reset_name="reset",
        decay_name="decay",
        surrogate_slope_name="surrogate_slope",
        grad_pre_reset_name="grad_pre_reset",
        indent="        ",
    )
    return f"""@triton.jit
def {function_name}(
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
{step_body}
        tl.store(grad_inputs_ptr + time_offsets, grad_pre_reset, mask=mask)

    tl.store(grad_initial_ptr + offsets, grad_membrane, mask=mask)
"""


def render_linear_lif_surrogate_backward_weight_bias_kernel_source(
    surrogate: str,
    *,
    function_name: str = "_generated_linear_lif_surrogate_backward_weight_bias_kernel",
) -> str:
    """Render generated fused dense-synapse surrogate LIF backward kernel source."""

    step_body = render_lif_surrogate_backward_step_body(
        surrogate,
        dialect="triton",
        pre_reset_name="pre_reset",
        spike_name="spike",
        grad_membrane_name="grad_membrane",
        grad_spike_name="grad_spike",
        threshold_name="threshold",
        reset_name="reset",
        decay_name="decay",
        surrogate_slope_name="surrogate_slope",
        grad_pre_reset_name="grad_pre_reset",
        indent="        ",
    )
    return f"""@triton.jit
def {function_name}(
    inputs_ptr,
    weight_ptr,
    pre_reset_ptr,
    spikes_ptr,
    grad_final_ptr,
    grad_spikes_ptr,
    grad_inputs_ptr,
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
    has_grad_final: tl.constexpr,
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
        if has_grad_spikes:
            grad_spike = tl.load(grad_spikes_ptr + output_offsets, mask=mask_bn, other=0.0)
        else:
            grad_spike = tl.zeros((block_b, block_n), tl.float32)
{step_body}

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
"""


def render_linear_lif_surrogate_backward_weight_bias_packed_spikes_kernel_source(
    surrogate: str,
    *,
    function_name: str = "_generated_linear_lif_surrogate_backward_packed_spikes_kernel",
) -> str:
    """Render generated fused dense-synapse backward using packed saved spikes."""

    step_body = render_lif_surrogate_backward_step_body(
        surrogate,
        dialect="triton",
        pre_reset_name="pre_reset",
        spike_name="spike",
        grad_membrane_name="grad_membrane",
        grad_spike_name="grad_spike",
        threshold_name="threshold",
        reset_name="reset",
        decay_name="decay",
        surrogate_slope_name="surrogate_slope",
        grad_pre_reset_name="grad_pre_reset",
        indent="        ",
    )
    return f"""@triton.jit
def {function_name}(
    inputs_ptr,
    weight_ptr,
    pre_reset_ptr,
    packed_spikes_ptr,
    grad_final_ptr,
    grad_spikes_ptr,
    grad_inputs_ptr,
    grad_weight_ptr,
    grad_bias_ptr,
    timesteps: tl.constexpr,
    batch: tl.constexpr,
    features: tl.constexpr,
    neurons: tl.constexpr,
    packed_neurons: tl.constexpr,
    decay: tl.constexpr,
    threshold: tl.constexpr,
    reset: tl.constexpr,
    surrogate_slope: tl.constexpr,
    has_grad_final: tl.constexpr,
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
    word_offsets = offsets_n // 32
    bit_offsets = offsets_n - word_offsets * 32

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
        packed_offsets = (
            t * batch * packed_neurons
            + offsets_b[:, None] * packed_neurons
            + word_offsets[None, :]
        )
        pre_reset = tl.load(pre_reset_ptr + output_offsets, mask=mask_bn, other=0.0)
        packed_word = tl.load(
            packed_spikes_ptr + packed_offsets,
            mask=mask_bn,
            other=0,
        ).to(tl.int64)
        spike = ((packed_word >> bit_offsets[None, :]) & 1).to(tl.float32)
        if has_grad_spikes:
            grad_spike = tl.load(grad_spikes_ptr + output_offsets, mask=mask_bn, other=0.0)
        else:
            grad_spike = tl.zeros((block_b, block_n), tl.float32)
{step_body}

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
"""


def render_linear_lif_surrogate_checkpoint_backward_chunk_kernel_source(
    surrogate: str,
    *,
    function_name: str = "_generated_linear_lif_surrogate_checkpoint_backward_chunk_kernel",
) -> str:
    """Render generated checkpointed fused dense-synapse chunk backward kernel."""

    step_body = render_lif_surrogate_backward_step_body(
        surrogate,
        dialect="triton",
        pre_reset_name="pre_reset",
        spike_name="spike",
        grad_membrane_name="grad_membrane",
        grad_spike_name="grad_spike",
        threshold_name="threshold",
        reset_name="reset",
        decay_name="decay",
        surrogate_slope_name="surrogate_slope",
        grad_pre_reset_name="grad_pre_reset",
        indent="        ",
    )
    return f"""@triton.jit
def {function_name}(
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
        if has_grad_spikes:
            grad_spike = tl.load(
                grad_spikes_ptr
                + t * batch * neurons
                + offsets_b[:, None] * neurons
                + offsets_n[None, :],
                mask=mask_bn,
                other=0.0,
            )
        else:
            grad_spike = tl.zeros((block_b, block_n), tl.float32)
{step_body}

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
"""


def render_linear_lif_surrogate_checkpoint_backward_chunk_packed_spikes_kernel_source(
    surrogate: str,
    *,
    function_name: str = "_generated_linear_lif_surrogate_checkpoint_backward_packed_spikes_kernel",
) -> str:
    """Render generated checkpointed chunk backward deriving spikes from pre-reset state."""

    step_body = render_lif_surrogate_backward_step_body(
        surrogate,
        dialect="triton",
        pre_reset_name="pre_reset",
        spike_name="spike",
        grad_membrane_name="grad_membrane",
        grad_spike_name="grad_spike",
        threshold_name="threshold",
        reset_name="reset",
        decay_name="decay",
        surrogate_slope_name="surrogate_slope",
        grad_pre_reset_name="grad_pre_reset",
        indent="        ",
    )
    return f"""@triton.jit
def {function_name}(
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
        if has_grad_spikes:
            grad_spike = tl.load(
                grad_spikes_ptr
                + t * batch * neurons
                + offsets_b[:, None] * neurons
                + offsets_n[None, :],
                mask=mask_bn,
                other=0.0,
            )
        else:
            grad_spike = tl.zeros((block_b, block_n), tl.float32)
        if has_grad_spike_rate:
            grad_spike_rate = tl.load(grad_spike_rate_ptr)
            grad_spike += grad_spike_rate * spike_rate_scale
        if has_grad_spike_rates:
            grad_spike_rate_values = tl.load(
                grad_spike_rates_ptr + offsets_b[:, None] * neurons + offsets_n[None, :],
                mask=mask_bn,
                other=0.0,
            )
            grad_spike += grad_spike_rate_values * spike_rates_scale
{step_body}

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
"""


def load_generated_forward_kernel(
    source: str,
    *,
    function_name: str,
    module_name: str,
) -> Any:
    """Compile generated Triton kernel source into a Python function."""

    filename = f"<myelin_generated_{function_name}>"
    linecache.cache[filename] = (
        len(source),
        None,
        [f"{line}\n" for line in source.splitlines()],
        filename,
    )
    module = ModuleType(module_name)
    module.__file__ = filename
    module.__dict__.update({"triton": triton, "tl": tl})
    exec(compile(source, filename, "exec"), module.__dict__)  # noqa: S102 - local IR.
    return module.__dict__[function_name]


def _load_generated_kernel_cached(
    *,
    kind: str,
    function_name: str,
    module_name: str,
    source_factory: Callable[[], str],
) -> Any:
    cache_key = (kind, function_name, module_name, _TORCH_VERSION_KEY)
    cached_kernel = _GENERATED_KERNEL_CACHE.get(cache_key)
    if cached_kernel is not None:
        return cached_kernel

    kernel = load_generated_forward_kernel(
        source_factory(),
        function_name=function_name,
        module_name=module_name,
    )
    _GENERATED_KERNEL_CACHE[cache_key] = kernel
    return kernel


def load_generated_neuron_forward_kernel(
    ir: NeuronIR,
    *,
    function_name: str | None = None,
    module_name: str | None = None,
) -> Any:
    """Compile a generated Triton forward kernel for a custom neuron IR."""

    _validate_forward_kernel_ir(ir)
    resolved_function_name = (
        f"_generated_{ir.name}_forward_kernel" if function_name is None else function_name
    )
    resolved_module_name = (
        f"myelin_generated_{ir.name}_forward" if module_name is None else module_name
    )
    cache_key = (_neuron_ir_cache_key(ir), resolved_function_name, resolved_module_name)
    cached_kernel = _GENERATED_NEURON_FORWARD_CACHE.get(cache_key)
    if cached_kernel is not None:
        return cached_kernel

    kernel = load_generated_forward_kernel(
        render_forward_kernel_source(ir, function_name=resolved_function_name),
        function_name=resolved_function_name,
        module_name=resolved_module_name,
    )
    _GENERATED_NEURON_FORWARD_CACHE[cache_key] = kernel
    return kernel


def load_generated_lif_forward_kernel(function_name: str = "_generated_lif_forward_kernel") -> Any:
    """Compile the generated LIF forward kernel source into a Python function."""

    return _load_generated_kernel_cached(
        kind="lif_forward",
        function_name=function_name,
        module_name="myelin_generated_lif",
        source_factory=lambda: render_lif_forward_kernel_source(function_name=function_name),
    )


def load_generated_alif_forward_kernel(
    function_name: str = "_generated_alif_forward_kernel",
) -> Any:
    """Compile the generated ALIF forward kernel source into a Python function."""

    return _load_generated_kernel_cached(
        kind="alif_forward",
        function_name=function_name,
        module_name="myelin_generated_alif",
        source_factory=lambda: render_alif_forward_kernel_source(function_name=function_name),
    )


def load_generated_izhikevich_forward_kernel(
    function_name: str = "_generated_izhikevich_forward_kernel",
) -> Any:
    """Compile the generated Izhikevich forward kernel source into a Python function."""

    return _load_generated_kernel_cached(
        kind="izhikevich_forward",
        function_name=function_name,
        module_name="myelin_generated_izhikevich",
        source_factory=lambda: render_izhikevich_forward_kernel_source(function_name=function_name),
    )


def load_generated_lif_surrogate_backward_kernel(
    surrogate: str,
    function_name: str = "_generated_lif_surrogate_backward_kernel",
) -> Any:
    """Compile a generated surrogate LIF backward kernel into a Python function."""

    return _load_generated_kernel_cached(
        kind=f"lif_surrogate_backward:{surrogate}",
        function_name=function_name,
        module_name=f"myelin_generated_lif_{surrogate}_backward",
        source_factory=lambda: render_lif_surrogate_backward_kernel_source(
            surrogate,
            function_name=function_name,
        ),
    )


def load_generated_lif_surrogate_backward_packed_spikes_kernel(
    surrogate: str,
    function_name: str = "_generated_lif_surrogate_backward_packed_spikes_kernel",
) -> Any:
    """Compile a generated surrogate LIF packed-spike backward kernel."""

    return _load_generated_kernel_cached(
        kind=f"lif_surrogate_backward_packed_spikes:{surrogate}",
        function_name=function_name,
        module_name=f"myelin_generated_lif_{surrogate}_backward_packed_spikes",
        source_factory=lambda: render_lif_surrogate_backward_packed_spikes_kernel_source(
            surrogate,
            function_name=function_name,
        ),
    )


def load_generated_linear_lif_surrogate_backward_weight_bias_kernel(
    surrogate: str,
    function_name: str = "_generated_linear_lif_surrogate_backward_weight_bias_kernel",
) -> Any:
    """Compile generated fused dense-synapse dweight/dbias kernel source."""

    return _load_generated_kernel_cached(
        kind=f"linear_lif_surrogate_backward_weight_bias:{surrogate}",
        function_name=function_name,
        module_name=f"myelin_generated_linear_lif_{surrogate}_backward_weight_bias",
        source_factory=lambda: render_linear_lif_surrogate_backward_weight_bias_kernel_source(
            surrogate,
            function_name=function_name,
        ),
    )


def load_generated_linear_lif_surrogate_backward_weight_bias_packed_spikes_kernel(
    surrogate: str,
    function_name: str = "_generated_linear_lif_surrogate_backward_packed_spikes_kernel",
) -> Any:
    """Compile generated fused dense-synapse backward with packed saved spikes."""

    return _load_generated_kernel_cached(
        kind=f"linear_lif_surrogate_backward_weight_bias_packed_spikes:{surrogate}",
        function_name=function_name,
        module_name=(f"myelin_generated_linear_lif_{surrogate}_backward_weight_bias_packed_spikes"),
        source_factory=lambda: (
            render_linear_lif_surrogate_backward_weight_bias_packed_spikes_kernel_source(
                surrogate,
                function_name=function_name,
            )
        ),
    )


def load_generated_linear_lif_surrogate_checkpoint_backward_chunk_kernel(
    surrogate: str,
    function_name: str = "_generated_linear_lif_surrogate_checkpoint_backward_chunk_kernel",
) -> Any:
    """Compile generated checkpointed dense-synapse chunk backward kernel."""

    return _load_generated_kernel_cached(
        kind=f"linear_lif_surrogate_checkpoint_backward_chunk:{surrogate}",
        function_name=function_name,
        module_name=f"myelin_generated_linear_lif_{surrogate}_checkpoint_backward_chunk",
        source_factory=lambda: render_linear_lif_surrogate_checkpoint_backward_chunk_kernel_source(
            surrogate,
            function_name=function_name,
        ),
    )


def load_generated_linear_lif_surrogate_checkpoint_backward_chunk_packed_spikes_kernel(
    surrogate: str,
    function_name: str = "_generated_linear_lif_surrogate_checkpoint_backward_packed_spikes_kernel",
) -> Any:
    """Compile generated checkpointed backward deriving hard spikes from pre-reset state."""

    return _load_generated_kernel_cached(
        kind=f"linear_lif_surrogate_checkpoint_backward_packed_spikes:{surrogate}",
        function_name=function_name,
        module_name=f"myelin_generated_linear_lif_{surrogate}_checkpoint_backward_packed_spikes",
        source_factory=lambda: (
            render_linear_lif_surrogate_checkpoint_backward_chunk_packed_spikes_kernel_source(
                surrogate,
                function_name=function_name,
            )
        ),
    )


def generated_forward(
    ir: NeuronIR,
    inputs: torch.Tensor,
    initial_state: Mapping[str, torch.Tensor],
    params: Mapping[str, float],
    *,
    kernel: Any,
    block_size: int = 256,
    label: str | None = None,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Run a generated Triton forward kernel with an IR-defined launch ABI."""

    _validate_forward_kernel_ir(ir)
    kernel_label = ir.name if label is None else label
    if inputs.ndim != 3:
        raise ValueError(f"inputs must be shaped [T, B, N]; got {tuple(inputs.shape)}")
    if not inputs.is_cuda:
        raise ValueError(f"generated Triton {kernel_label} forward requires CUDA inputs")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive; got {block_size}")

    expected_state_shape = inputs.shape[1:]
    missing_state = set(ir.state) - set(initial_state)
    if missing_state:
        raise ValueError(f"initial_state is missing state tensors: {sorted(missing_state)}")
    unexpected_state = set(initial_state) - set(ir.state)
    if unexpected_state:
        raise ValueError(
            f"initial_state contains undeclared state tensors: {sorted(unexpected_state)}"
        )
    missing_params = set(ir.params) - set(params)
    if missing_params:
        raise ValueError(f"params is missing values: {sorted(missing_params)}")
    unexpected_params = set(params) - set(ir.params)
    if unexpected_params:
        raise ValueError(f"params contains undeclared values: {sorted(unexpected_params)}")

    contiguous_inputs = inputs.contiguous()
    contiguous_initial: dict[str, torch.Tensor] = {}
    for state_name in ir.state:
        state_tensor = initial_state[state_name]
        if state_tensor.shape != expected_state_shape:
            raise ValueError(f"initial_state.{state_name} must match inputs.shape[1:]")
        if state_tensor.device != inputs.device:
            raise ValueError(f"initial_state.{state_name} must be on the same device as inputs")
        if state_tensor.dtype != inputs.dtype:
            raise ValueError(f"initial_state.{state_name} must have the same dtype as inputs")
        contiguous_initial[state_name] = state_tensor.contiguous()

    first_state = contiguous_initial[ir.state[0]]
    timesteps = contiguous_inputs.shape[0]
    total_elements = first_state.numel()
    final_state = {
        state_name: torch.empty_like(contiguous_initial[state_name]) for state_name in ir.state
    }
    spikes = torch.empty_like(contiguous_inputs)
    grid = (triton.cdiv(total_elements, block_size),)
    kernel_args: list[object] = [
        contiguous_inputs,
        *(contiguous_initial[state_name] for state_name in ir.state),
        *(final_state[state_name] for state_name in ir.state),
        spikes,
        total_elements,
        timesteps,
        *(params[param_name] for param_name in ir.params),
        block_size,
    ]
    kernel[grid](*kernel_args)
    return final_state, spikes


def _validate_forward_kernel_ir(ir: NeuronIR) -> None:
    validate_generated_forward_ir(ir)


def _neuron_ir_cache_key(ir: NeuronIR) -> _GeneratedNeuronForwardKey:
    return (
        ir.name,
        ir.state,
        ir.params,
        ir.inputs,
        tuple((name, _expr_cache_key(expr)) for name, expr in ir.next_state.items()),
        tuple((name, _expr_cache_key(expr)) for name, expr in ir.outputs.items()),
    )


def _expr_cache_key(expr: Expr) -> tuple[object, ...]:
    return (
        expr.kind,
        expr.name,
        expr.value,
        tuple(_expr_cache_key(arg) for arg in expr.args),
    )


def generated_neuron_forward(
    ir: NeuronIR,
    inputs: torch.Tensor,
    initial_state: Mapping[str, torch.Tensor],
    params: Mapping[str, float],
    *,
    block_size: int = 256,
    label: str | None = None,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Compile and run a generated Triton forward kernel for a custom neuron IR."""

    return generated_forward(
        ir,
        inputs,
        initial_state,
        params,
        kernel=load_generated_neuron_forward_kernel(ir),
        block_size=block_size,
        label=label,
    )


def generated_lif_forward(
    inputs: torch.Tensor,
    initial_state: LIFState,
    params: LIFParams,
    *,
    block_size: int = 256,
) -> tuple[LIFState, torch.Tensor]:
    """Run a LIF forward kernel generated from the neuron IR."""

    final_state, spikes = generated_forward(
        lif_ir(),
        inputs,
        {"membrane": initial_state.membrane},
        {"decay": params.decay, "threshold": params.threshold, "reset": params.reset},
        kernel=load_generated_lif_forward_kernel(),
        block_size=block_size,
        label="LIF",
    )
    return LIFState(membrane=final_state["membrane"]), spikes


def generated_alif_forward(
    inputs: torch.Tensor,
    initial_state: ALIFState,
    params: ALIFParams,
    *,
    block_size: int = 256,
) -> tuple[ALIFState, torch.Tensor]:
    """Run an ALIF forward kernel generated from the neuron IR."""

    final_state, spikes = generated_forward(
        alif_ir(),
        inputs,
        {
            "membrane": initial_state.membrane,
            "adaptation": initial_state.adaptation,
        },
        {
            "decay": params.decay,
            "adaptation_decay": params.adaptation_decay,
            "threshold": params.threshold,
            "reset": params.reset,
            "beta": params.beta,
        },
        kernel=load_generated_alif_forward_kernel(),
        block_size=block_size,
        label="ALIF",
    )
    return ALIFState(
        membrane=final_state["membrane"],
        adaptation=final_state["adaptation"],
    ), spikes


def generated_izhikevich_forward(
    inputs: torch.Tensor,
    initial_state: IzhikevichState,
    params: IzhikevichParams,
    *,
    block_size: int = 256,
) -> tuple[IzhikevichState, torch.Tensor]:
    """Run an Izhikevich forward kernel generated from the neuron IR."""

    params.validate()
    final_state, spikes = generated_forward(
        izhikevich_ir(),
        inputs,
        {
            "voltage": initial_state.voltage,
            "recovery": initial_state.recovery,
        },
        {
            "recovery_decay": params.recovery_decay,
            "recovery_coupling": params.recovery_coupling,
            "reset_voltage": params.reset_voltage,
            "recovery_jump": params.recovery_jump,
            "threshold": params.threshold,
            "dt": params.dt,
            "voltage_square_coeff": params.voltage_square_coeff,
            "voltage_coeff": params.voltage_coeff,
            "voltage_bias": params.voltage_bias,
        },
        kernel=load_generated_izhikevich_forward_kernel(),
        block_size=block_size,
        label="Izhikevich",
    )
    return IzhikevichState(
        voltage=final_state["voltage"],
        recovery=final_state["recovery"],
    ), spikes


def generated_lif_surrogate_backward(
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
    """Run a generated Triton surrogate LIF backward kernel."""

    if pre_reset_membranes.ndim != 3:
        raise ValueError(
            f"pre_reset_membranes must be shaped [T, B, N]; got {tuple(pre_reset_membranes.shape)}"
        )
    if not pre_reset_membranes.is_cuda:
        raise ValueError("generated Triton LIF backward requires CUDA tensors")
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
    kernel = load_generated_lif_surrogate_backward_kernel(surrogate)
    grid = (triton.cdiv(total_elements, block_size),)
    kernel[grid](
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
        block_size,  # pyright: ignore[reportArgumentType]
    )
    return grad_inputs, grad_initial


def generated_lif_surrogate_backward_packed_spikes(
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
    """Run a generated Triton surrogate LIF backward kernel with packed saved spikes."""

    if pre_reset_membranes.ndim != 3:
        raise ValueError(
            f"pre_reset_membranes must be shaped [T, B, N]; got {tuple(pre_reset_membranes.shape)}"
        )
    if not pre_reset_membranes.is_cuda:
        raise ValueError("generated Triton LIF packed backward requires CUDA tensors")
    if packed_spikes.device != pre_reset_membranes.device:
        raise ValueError("packed_spikes must be on the same device as pre_reset_membranes")
    if packed_spikes.dtype != torch.int32:
        raise ValueError("packed_spikes must have dtype torch.int32")

    contiguous_pre_reset = pre_reset_membranes.contiguous()
    timesteps, batch, neurons = contiguous_pre_reset.shape
    packed_neurons = packed_last_dim_size(neurons)
    expected_packed_shape = (timesteps, batch, packed_neurons)
    if packed_spikes.shape != expected_packed_shape:
        raise ValueError(
            "packed_spikes must be shaped "
            f"{expected_packed_shape}; got {tuple(packed_spikes.shape)}"
        )

    contiguous_packed_spikes = packed_spikes.contiguous()
    state_shape = (batch, neurons)
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
    kernel = load_generated_lif_surrogate_backward_packed_spikes_kernel(surrogate)
    grid = (triton.cdiv(total_elements, block_size),)
    kernel[grid](
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
        block_size,  # pyright: ignore[reportArgumentType]
    )
    return grad_inputs, grad_initial


def generated_linear_lif_surrogate_backward_weight_bias(
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
    """Run generated fused dense-synapse surrogate LIF backward."""

    for name, value in [("block_b", block_b), ("block_n", block_n), ("block_f", block_f)]:
        if value <= 0 or value & (value - 1):
            raise ValueError(f"{name} must be a positive power of two")
    if min(block_b, block_n, block_f) < 16:
        raise ValueError("block_b, block_n, and block_f must be at least 16 for Triton dot")
    if inputs.ndim != 3:
        raise ValueError(f"inputs must be shaped [T, B, F]; got {tuple(inputs.shape)}")
    if weight.ndim != 2:
        raise ValueError(f"weight must be shaped [F, N]; got {tuple(weight.shape)}")
    if pre_reset_membranes.ndim != 3:
        raise ValueError(
            f"pre_reset_membranes must be shaped [T, B, N]; got {tuple(pre_reset_membranes.shape)}"
        )
    if not inputs.is_cuda:
        raise ValueError("generated Triton linear surrogate LIF backward requires CUDA inputs")
    if pre_reset_membranes.device != inputs.device or pre_reset_membranes.dtype != inputs.dtype:
        raise ValueError("pre_reset_membranes must have the same device and dtype as inputs")
    if weight.device != inputs.device or weight.dtype != inputs.dtype:
        raise ValueError("weight must have the same device and dtype as inputs")
    if spikes.shape != pre_reset_membranes.shape:
        raise ValueError("spikes must have the same shape as pre_reset_membranes")
    if spikes.device != inputs.device or spikes.dtype != inputs.dtype:
        raise ValueError("spikes must have the same device and dtype as inputs")
    if pre_reset_membranes.shape[:2] != inputs.shape[:2]:
        raise ValueError("inputs and pre_reset_membranes must agree on T and B")
    if inputs.shape[2] != weight.shape[0] or pre_reset_membranes.shape[2] != weight.shape[1]:
        raise ValueError("weight must connect inputs features to pre_reset neurons")

    contiguous_inputs = inputs.contiguous()
    contiguous_weight = weight.contiguous()
    contiguous_pre_reset = pre_reset_membranes.contiguous()
    contiguous_spikes = spikes.contiguous()
    timesteps, batch, features = contiguous_inputs.shape
    neurons = contiguous_pre_reset.shape[2]

    if grad_final_membrane is None:
        contiguous_grad_final = contiguous_pre_reset
        has_grad_final = False
    else:
        if grad_final_membrane.shape != (batch, neurons):
            raise ValueError("grad_final_membrane must match pre_reset_membranes.shape[1:]")
        if grad_final_membrane.device != inputs.device or grad_final_membrane.dtype != inputs.dtype:
            raise ValueError("grad_final_membrane must have the same device and dtype as inputs")
        contiguous_grad_final = grad_final_membrane.contiguous()
        has_grad_final = True

    if grad_spikes is None:
        contiguous_grad_spikes = contiguous_pre_reset
        has_grad_spikes = False
    else:
        if grad_spikes.shape != contiguous_pre_reset.shape:
            raise ValueError("grad_spikes must match pre_reset_membranes")
        if grad_spikes.device != inputs.device or grad_spikes.dtype != inputs.dtype:
            raise ValueError("grad_spikes must have the same device and dtype as inputs")
        contiguous_grad_spikes = grad_spikes.contiguous()
        has_grad_spikes = True

    grad_inputs = torch.zeros_like(contiguous_inputs) if needs_input_grad else None
    grad_weight = (
        torch.zeros((features, neurons), dtype=inputs.dtype, device=inputs.device)
        if needs_weight_grad
        else None
    )
    grad_bias = (
        torch.zeros((neurons,), dtype=inputs.dtype, device=inputs.device)
        if needs_bias_grad
        else None
    )
    grad_inputs_ptr = contiguous_inputs if grad_inputs is None else grad_inputs
    grad_weight_ptr = contiguous_inputs if grad_weight is None else grad_weight
    bias_ptr = contiguous_inputs if grad_bias is None else grad_bias
    kernel = load_generated_linear_lif_surrogate_backward_weight_bias_kernel(surrogate)
    grid = (
        triton.cdiv(features, block_f),
        triton.cdiv(neurons, block_n),
        triton.cdiv(batch, block_b),
    )
    kernel[grid](
        contiguous_inputs,
        contiguous_weight,
        contiguous_pre_reset,
        contiguous_spikes,
        contiguous_grad_final,
        contiguous_grad_spikes,
        grad_inputs_ptr,
        grad_weight_ptr,
        bias_ptr,
        timesteps,  # pyright: ignore[reportArgumentType]
        batch,  # pyright: ignore[reportArgumentType]
        features,  # pyright: ignore[reportArgumentType]
        neurons,  # pyright: ignore[reportArgumentType]
        params.decay,  # pyright: ignore[reportArgumentType]
        params.threshold,  # pyright: ignore[reportArgumentType]
        params.reset,  # pyright: ignore[reportArgumentType]
        surrogate_slope,  # pyright: ignore[reportArgumentType]
        has_grad_final,  # pyright: ignore[reportArgumentType]
        has_grad_spikes,  # pyright: ignore[reportArgumentType]
        needs_input_grad,  # pyright: ignore[reportArgumentType]
        needs_weight_grad,  # pyright: ignore[reportArgumentType]
        needs_bias_grad,  # pyright: ignore[reportArgumentType]
        block_f,  # pyright: ignore[reportArgumentType]
        block_n,  # pyright: ignore[reportArgumentType]
        block_b,  # pyright: ignore[reportArgumentType]
    )
    return grad_inputs, grad_weight, grad_bias


def generated_linear_lif_surrogate_backward_weight_bias_packed_spikes(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    pre_reset_membranes: torch.Tensor,
    packed_spikes: torch.Tensor,
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
    """Run generated fused dense-synapse backward with packed saved spikes."""

    for name, value in [("block_b", block_b), ("block_n", block_n), ("block_f", block_f)]:
        if value <= 0 or value & (value - 1):
            raise ValueError(f"{name} must be a positive power of two")
    if min(block_b, block_n, block_f) < 16:
        raise ValueError("block_b, block_n, and block_f must be at least 16 for Triton dot")
    if inputs.ndim != 3:
        raise ValueError(f"inputs must be shaped [T, B, F]; got {tuple(inputs.shape)}")
    if weight.ndim != 2:
        raise ValueError(f"weight must be shaped [F, N]; got {tuple(weight.shape)}")
    if pre_reset_membranes.ndim != 3:
        raise ValueError(
            f"pre_reset_membranes must be shaped [T, B, N]; got {tuple(pre_reset_membranes.shape)}"
        )
    if not inputs.is_cuda:
        raise ValueError("generated Triton linear surrogate LIF backward requires CUDA inputs")
    if pre_reset_membranes.device != inputs.device or pre_reset_membranes.dtype != inputs.dtype:
        raise ValueError("pre_reset_membranes must have the same device and dtype as inputs")
    if weight.device != inputs.device or weight.dtype != inputs.dtype:
        raise ValueError("weight must have the same device and dtype as inputs")
    if packed_spikes.device != inputs.device:
        raise ValueError("packed_spikes must have the same device as inputs")
    if packed_spikes.dtype != torch.int32:
        raise ValueError("packed_spikes must have dtype torch.int32")
    if pre_reset_membranes.shape[:2] != inputs.shape[:2]:
        raise ValueError("inputs and pre_reset_membranes must agree on T and B")
    if inputs.shape[2] != weight.shape[0] or pre_reset_membranes.shape[2] != weight.shape[1]:
        raise ValueError("weight must connect inputs features to pre_reset neurons")

    contiguous_inputs = inputs.contiguous()
    contiguous_weight = weight.contiguous()
    contiguous_pre_reset = pre_reset_membranes.contiguous()
    timesteps, batch, features = contiguous_inputs.shape
    neurons = contiguous_pre_reset.shape[2]
    packed_neurons = packed_last_dim_size(neurons)
    expected_packed_shape = (timesteps, batch, packed_neurons)
    if packed_spikes.shape != expected_packed_shape:
        raise ValueError(
            "packed_spikes must be shaped "
            f"{expected_packed_shape}; got {tuple(packed_spikes.shape)}"
        )
    contiguous_packed_spikes = packed_spikes.contiguous()

    if grad_final_membrane is None:
        contiguous_grad_final = contiguous_pre_reset
        has_grad_final = False
    else:
        if grad_final_membrane.shape != (batch, neurons):
            raise ValueError("grad_final_membrane must match pre_reset_membranes.shape[1:]")
        if grad_final_membrane.device != inputs.device or grad_final_membrane.dtype != inputs.dtype:
            raise ValueError("grad_final_membrane must have the same device and dtype as inputs")
        contiguous_grad_final = grad_final_membrane.contiguous()
        has_grad_final = True

    if grad_spikes is None:
        contiguous_grad_spikes = contiguous_pre_reset
        has_grad_spikes = False
    else:
        if grad_spikes.shape != contiguous_pre_reset.shape:
            raise ValueError("grad_spikes must match pre_reset_membranes")
        if grad_spikes.device != inputs.device or grad_spikes.dtype != inputs.dtype:
            raise ValueError("grad_spikes must have the same device and dtype as inputs")
        contiguous_grad_spikes = grad_spikes.contiguous()
        has_grad_spikes = True

    grad_inputs = torch.zeros_like(contiguous_inputs) if needs_input_grad else None
    grad_weight = (
        torch.zeros((features, neurons), dtype=inputs.dtype, device=inputs.device)
        if needs_weight_grad
        else None
    )
    grad_bias = (
        torch.zeros((neurons,), dtype=inputs.dtype, device=inputs.device)
        if needs_bias_grad
        else None
    )
    grad_inputs_ptr = contiguous_inputs if grad_inputs is None else grad_inputs
    grad_weight_ptr = contiguous_inputs if grad_weight is None else grad_weight
    bias_ptr = contiguous_inputs if grad_bias is None else grad_bias
    kernel = load_generated_linear_lif_surrogate_backward_weight_bias_packed_spikes_kernel(
        surrogate
    )
    grid = (
        triton.cdiv(features, block_f),
        triton.cdiv(neurons, block_n),
        triton.cdiv(batch, block_b),
    )
    kernel[grid](
        contiguous_inputs,
        contiguous_weight,
        contiguous_pre_reset,
        contiguous_packed_spikes,
        contiguous_grad_final,
        contiguous_grad_spikes,
        grad_inputs_ptr,
        grad_weight_ptr,
        bias_ptr,
        timesteps,  # pyright: ignore[reportArgumentType]
        batch,  # pyright: ignore[reportArgumentType]
        features,  # pyright: ignore[reportArgumentType]
        neurons,  # pyright: ignore[reportArgumentType]
        packed_neurons,  # pyright: ignore[reportArgumentType]
        params.decay,  # pyright: ignore[reportArgumentType]
        params.threshold,  # pyright: ignore[reportArgumentType]
        params.reset,  # pyright: ignore[reportArgumentType]
        surrogate_slope,  # pyright: ignore[reportArgumentType]
        has_grad_final,  # pyright: ignore[reportArgumentType]
        has_grad_spikes,  # pyright: ignore[reportArgumentType]
        needs_input_grad,  # pyright: ignore[reportArgumentType]
        needs_weight_grad,  # pyright: ignore[reportArgumentType]
        needs_bias_grad,  # pyright: ignore[reportArgumentType]
        block_f,  # pyright: ignore[reportArgumentType]
        block_n,  # pyright: ignore[reportArgumentType]
        block_b,  # pyright: ignore[reportArgumentType]
    )
    return grad_inputs, grad_weight, grad_bias


def generated_linear_lif_surrogate_checkpoint_backward(
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
    """Run generated checkpointed fused dense-synapse backward."""
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
        raise ValueError("generated Triton checkpoint LIF backward requires CUDA inputs")
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
    from myelin.triton.lif import (
        _linear_surrogate_lif_checkpoint_recompute_chunk_packed_spikes_kernel,
    )

    backward_kernel = (
        load_generated_linear_lif_surrogate_checkpoint_backward_chunk_packed_spikes_kernel(
            surrogate
        )
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
        backward_kernel[backward_grid](
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
