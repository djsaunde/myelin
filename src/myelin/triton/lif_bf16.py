"""Prototype: bf16-I/O surrogate-LIF kernel with an fp32 membrane carry.

The production kernel (``lif.py``) computes the recurrence in the I/O dtype, so
``SpikingSequenceLIF`` upcasts bf16 inputs to fp32 before calling it (to keep the
loop-carried membrane in fp32). That doubles the LIF's memory traffic under bf16
autocast. Here the loads are cast to fp32 in-register, so the membrane carry is
fp32 while inputs / spikes / pre-reset stay bf16 — halving the bandwidth of the
~23%-of-step LIF. ``pre_reset`` is stored bf16 (used by the surrogate backward),
which trades a little gradient precision for bandwidth; parity is validated
against the eager loop oracle on Modal before this is promoted into ``lif.py``.
"""

from __future__ import annotations

import torch

from myelin._optional import has_triton
from myelin.neurons import LIFParams, LIFState

if has_triton():
    import triton
    import triton.language as tl

    from myelin.triton.lif import _surrogate_derivative

    @triton.jit
    def _fwd_kernel(
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
        pid = tl.program_id(0)
        offs = pid * block_size + tl.arange(0, block_size)
        mask = offs < total_elements
        membrane = tl.load(initial_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        for t in range(timesteps):
            toff = t * total_elements + offs
            cur = tl.load(inputs_ptr + toff, mask=mask, other=0.0).to(tl.float32)
            membrane = membrane * decay + cur
            spike = membrane >= threshold
            tl.store(pre_reset_ptr + t * total_elements + offs, membrane, mask=mask)
            tl.store(spikes_ptr + t * total_elements + offs, spike.to(tl.float32), mask=mask)
            membrane = tl.where(spike, reset, membrane)
        tl.store(final_ptr + offs, membrane, mask=mask)

    @triton.jit
    def _bwd_kernel(
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
        pid = tl.program_id(0)
        offs = pid * block_size + tl.arange(0, block_size)
        mask = offs < total_elements
        grad_membrane = tl.load(grad_final_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        for reverse_t in range(timesteps):
            t = timesteps - 1 - reverse_t
            toff = t * total_elements + offs
            pre_reset = tl.load(pre_reset_ptr + toff, mask=mask, other=0.0).to(tl.float32)
            spike = tl.load(spikes_ptr + toff, mask=mask, other=0.0).to(tl.float32)
            grad_spike = tl.load(grad_spikes_ptr + toff, mask=mask, other=0.0).to(tl.float32)
            centered = surrogate_slope * (pre_reset - threshold)
            d = surrogate_slope * _surrogate_derivative(centered, surrogate_id)
            grad_pre = grad_membrane * ((1.0 - spike) + (reset - pre_reset) * d) + grad_spike * d
            tl.store(grad_inputs_ptr + toff, grad_pre, mask=mask)
            grad_membrane = grad_pre * decay
        tl.store(grad_initial_ptr + offs, grad_membrane, mask=mask)

    class _BF16LIF(torch.autograd.Function):
        @staticmethod
        def forward(ctx, inputs, initial, decay, threshold, reset, slope, surrogate_id, block):
            inp = inputs.contiguous()
            init = initial.contiguous()
            total = init.numel()
            final = torch.empty_like(init)
            spikes = torch.empty_like(inp)
            pre = torch.empty_like(inp)
            grid = (triton.cdiv(total, block),)
            _fwd_kernel[grid](
                inp, init, final, spikes, pre, total, inp.shape[0], decay, threshold, reset, block
            )
            ctx.save_for_backward(pre, spikes)
            ctx.meta = (decay, threshold, reset, slope, surrogate_id, block, total)
            return final, spikes

        @staticmethod
        def backward(ctx, *grad_outputs):
            grad_final, grad_spikes = grad_outputs
            pre, spikes = ctx.saved_tensors
            decay, threshold, reset, slope, sid, block, total = ctx.meta
            gi = torch.empty_like(spikes)
            g0 = torch.empty_like(grad_final.contiguous())
            grid = (triton.cdiv(total, block),)
            _bwd_kernel[grid](
                pre,
                spikes,
                grad_final.contiguous(),
                grad_spikes.contiguous(),
                gi,
                g0,
                total,
                spikes.shape[0],
                decay,
                threshold,
                reset,
                slope,
                sid,
                block,
            )
            return gi, g0, None, None, None, None, None, None


def surrogate_lif_bf16io(
    inputs: torch.Tensor,
    initial: LIFState,
    params: LIFParams,
    *,
    surrogate_slope: float = 2.0,
    surrogate_id: int = 2,
    block_size: int = 128,  # ~2.5% faster than 256 here (occupancy; Modal sm_120 A/B)
) -> torch.Tensor:
    """Spikes from the bf16-I/O / fp32-carry surrogate LIF (CUDA only)."""
    _, spikes = _BF16LIF.apply(
        inputs,
        initial.membrane,
        params.decay,
        params.threshold,
        params.reset,
        surrogate_slope,
        surrogate_id,
        block_size,
    )
    return spikes
