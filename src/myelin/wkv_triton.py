"""Fused Triton WKV kernel (RWKV-v4 recurrence) — forward + backward.

A hand-written sequential-over-time kernel that is far faster than the generic
``associative_scan`` path (~8x on the forward at ctx 1024) and needs no nvcc —
Triton compiles through the bundled ptxas. The math is the standard RWKV-v4 WKV;
the backward is ported verbatim from the reference SpikeGPT/RWKV CUDA kernel
(``cuda/wkv_cuda.cu``): a forward-replay pass that accumulates the time_decay /
time_first gradients, then a reverse pass for the key/value gradients. The
``y``/``z``/``zexp`` replay buffers the CUDA version keeps in per-thread register
arrays are stored in global scratch here (one program per (batch, channel-block)
owns its rows, so there are no cross-program races).

The recurrence runs in float32 regardless of the model's autocast dtype (matching
the reference kernel), with results cast back to the caller's dtype.
"""

from __future__ import annotations

import torch

from myelin._optional import has_triton

if has_triton():
    import triton
    import triton.language as tl

    @triton.jit
    def _wkv_forward_kernel(k_ptr, v_ptr, w_ptr, u_ptr, y_ptr, T, C, BLOCK: tl.constexpr):
        pid_b = tl.program_id(0)
        cs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = cs < C
        w = tl.load(w_ptr + cs, mask=mask, other=0.0)
        u = tl.load(u_ptr + cs, mask=mask, other=0.0)
        p = tl.zeros([BLOCK], tl.float32)
        q = tl.zeros([BLOCK], tl.float32)
        o = tl.full([BLOCK], -1e38, tl.float32)
        base = pid_b * T * C
        for t in range(T):
            off = base + t * C + cs
            k = tl.load(k_ptr + off, mask=mask, other=0.0)
            v = tl.load(v_ptr + off, mask=mask, other=0.0)
            no = tl.maximum(o, u + k)
            a = tl.exp(o - no)
            b = tl.exp(u + k - no)
            tl.store(y_ptr + off, (a * p + b * v) / (a * q + b), mask=mask)
            no = tl.maximum(w + o, k)
            a = tl.exp(w + o - no)
            b = tl.exp(k - no)
            p = a * p + b * v
            q = a * q + b
            o = no

    @triton.jit
    def _wkv_backward_kernel(
        w_ptr,
        u_ptr,
        k_ptr,
        v_ptr,
        gy_ptr,
        y_ptr,
        z_ptr,
        zexp_ptr,
        gw_ptr,
        gu_ptr,
        gk_ptr,
        gv_ptr,
        T,
        C,
        BLOCK: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        cs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = cs < C
        w = tl.load(w_ptr + cs, mask=mask, other=0.0)
        u = tl.load(u_ptr + cs, mask=mask, other=0.0)
        base = pid_b * T * C

        # Pass 1: forward replay; cache y/z/zexp and accumulate gw, gu.
        p = tl.zeros([BLOCK], tl.float32)
        q = tl.zeros([BLOCK], tl.float32)
        dpdw = tl.zeros([BLOCK], tl.float32)
        dqdw = tl.zeros([BLOCK], tl.float32)
        o = tl.full([BLOCK], -1e38, tl.float32)
        gw = tl.zeros([BLOCK], tl.float32)
        gu = tl.zeros([BLOCK], tl.float32)
        for t in range(T):
            off = base + t * C + cs
            k = tl.load(k_ptr + off, mask=mask, other=0.0)
            v = tl.load(v_ptr + off, mask=mask, other=0.0)
            gy = tl.load(gy_ptr + off, mask=mask, other=0.0)
            no = tl.maximum(o, k + u)
            a = tl.exp(o - no)
            b = tl.exp(k + u - no)
            iden = 1.0 / (a * q + b)
            y = (a * p + b * v) * iden
            tl.store(y_ptr + off, y, mask=mask)
            tl.store(z_ptr + off, iden, mask=mask)
            tl.store(zexp_ptr + off, k + u - no, mask=mask)
            gw += gy * (dpdw - dqdw * y) * iden * a
            gu += gy * (v - y) * b * iden
            no = tl.maximum(w + o, k)
            a = tl.exp(w + o - no)
            b = tl.exp(k - no)
            dpdw = a * (p + dpdw)
            dqdw = a * (q + dqdw)
            p = a * p + b * v
            q = a * q + b
            o = no
        bc = pid_b * C + cs
        tl.store(gw_ptr + bc, gw * w, mask=mask)  # chain rule: w = -exp(time_decay)
        tl.store(gu_ptr + bc, gu, mask=mask)

        # Pass 2: reverse accumulation for gk, gv.
        gp = tl.zeros([BLOCK], tl.float32)
        gq = tl.zeros([BLOCK], tl.float32)
        o = tl.full([BLOCK], -1e38, tl.float32)
        for ti in range(T):
            t = T - 1 - ti
            off = base + t * C + cs
            k = tl.load(k_ptr + off, mask=mask, other=0.0)
            v = tl.load(v_ptr + off, mask=mask, other=0.0)
            gy = tl.load(gy_ptr + off, mask=mask, other=0.0)
            y = tl.load(y_ptr + off, mask=mask, other=0.0)
            z = tl.load(z_ptr + off, mask=mask, other=0.0)
            zx = tl.load(zexp_ptr + off, mask=mask, other=0.0)
            a = gy * z * tl.exp(zx)
            b = tl.exp(k + o)
            tl.store(gk_ptr + off, a * (v - y) + b * (gp * v + gq), mask=mask)
            tl.store(gv_ptr + off, a + b * gp, mask=mask)
            no = tl.maximum(w + o, zx - k - u)
            a = tl.exp(w + o - no)
            b = gy * z * tl.exp(zx - k - u - no)
            gp = a * gp + b
            gq = a * gq - b * y
            o = no


def _grid(b: int, c: int, block: int) -> tuple[int, int]:
    return (b, triton.cdiv(c, block))


class _WKVTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, key, value, time_decay, time_first):
        block = 64
        b, t, c = key.shape
        w = (-torch.exp(time_decay.float())).contiguous()
        u = time_first.float().contiguous()
        k = key.float().contiguous()
        v = value.float().contiguous()
        y = torch.empty_like(k)
        _wkv_forward_kernel[_grid(b, c, block)](k, v, w, u, y, t, c, BLOCK=block)
        ctx.save_for_backward(w, u, k, v)
        ctx.block = block
        ctx.orig_dtype = key.dtype
        ctx.td_dtype = time_decay.dtype
        ctx.tf_dtype = time_first.dtype
        return y.to(key.dtype)

    @staticmethod
    def backward(ctx, grad_y):
        w, u, k, v = ctx.saved_tensors
        b, t, c = k.shape
        block = ctx.block
        gy = grad_y.float().contiguous()
        ys = torch.empty_like(k)
        zs = torch.empty_like(k)
        zexps = torch.empty_like(k)
        gw = torch.empty((b, c), dtype=torch.float32, device=k.device)
        gu = torch.empty((b, c), dtype=torch.float32, device=k.device)
        gk = torch.empty_like(k)
        gv = torch.empty_like(k)
        _wkv_backward_kernel[_grid(b, c, block)](
            w, u, k, v, gy, ys, zs, zexps, gw, gu, gk, gv, t, c, BLOCK=block
        )
        return (
            gk.to(ctx.orig_dtype),
            gv.to(ctx.orig_dtype),
            gw.sum(0).to(ctx.td_dtype),  # grad time_decay (kernel already applied *w)
            gu.sum(0).to(ctx.tf_dtype),  # grad time_first
        )


def weighted_key_value_triton(
    key: torch.Tensor,
    value: torch.Tensor,
    time_decay: torch.Tensor,
    time_first: torch.Tensor,
) -> torch.Tensor:
    """RWKV WKV recurrence via the fused Triton kernel (CUDA only)."""
    return _WKVTriton.apply(key, value, time_decay, time_first)
