"""Backend-selecting fused kernel entrypoints."""

from __future__ import annotations

import warnings
from typing import Literal

import torch

from myelin._optional import has_triton, require_triton
from myelin.checkpointing import CheckpointSize, resolve_checkpoint_size
from myelin.functional import (
    alif_unroll,
    izhikevich_unroll,
    lif_unroll,
    surrogate_alif_unroll,
    surrogate_lif_unroll,
)
from myelin.neurons import (
    ALIFParams,
    ALIFState,
    IzhikevichParams,
    IzhikevichState,
    LIFParams,
    LIFState,
)
from myelin.packing import PackedSpikes, pack_spikes
from myelin.surrogates import SurrogateName, surrogate_from_name

Backend = Literal["auto", "torch", "triton", "triton_generated", "triton_compile"]


def _warn_if_cuda_torch_backend(
    inputs: torch.Tensor,
    *,
    operation: str,
    faster_backend: str = "triton",
) -> None:
    if torch.compiler.is_compiling():
        return
    if inputs.is_cuda and has_triton():
        warnings.warn(
            "CUDA inputs detected and Triton is available, but "
            f"{operation} is running with backend='torch'. Use "
            f"backend='{faster_backend}' or backend='auto' to enable the fused "
            "Triton path.",
            RuntimeWarning,
            stacklevel=3,
        )


def lif_forward(
    inputs: torch.Tensor,
    initial_state: LIFState,
    params: LIFParams,
    *,
    backend: Backend = "auto",
    block_size: int = 256,
) -> tuple[LIFState, torch.Tensor]:
    """Run LIF forward with an explicit backend choice.

    ``backend="auto"`` uses the Triton fused-time kernel only when inputs are
    CUDA tensors and Triton is importable; otherwise it falls back to the
    reference PyTorch implementation.
    """

    if backend == "torch":
        _warn_if_cuda_torch_backend(inputs, operation="lif_forward")
        return lif_unroll(inputs, initial_state, params)
    if backend == "triton":
        require_triton()
        from myelin.autograd import triton_lif_forward_function

        return triton_lif_forward_function(inputs, initial_state, params, block_size=block_size)
    if backend == "triton_generated":
        raise ValueError("backend='triton_generated' is only supported for surrogate LIF")
    if backend == "auto":
        if inputs.is_cuda and has_triton():
            from myelin.autograd import triton_lif_forward_function

            return triton_lif_forward_function(inputs, initial_state, params, block_size=block_size)
        if inputs.is_cuda:
            warnings.warn(
                "CUDA inputs detected but Triton is unavailable; falling back to the "
                "PyTorch LIF reference path. Install with `uv sync --extra cuda` "
                "to enable the fused-time Triton kernel.",
                RuntimeWarning,
                stacklevel=2,
            )
        return lif_unroll(inputs, initial_state, params)

    msg = f"unsupported backend: {backend}"
    raise ValueError(msg)


def lif_forward_packed_spikes(
    inputs: torch.Tensor,
    initial_state: LIFState,
    params: LIFParams,
    *,
    backend: Backend = "auto",
    block_b: int = 8,
) -> tuple[LIFState, PackedSpikes]:
    """Run LIF forward and return spikes packed along the neuron dimension.

    ``backend="triton"`` writes packed int32 words directly and avoids
    materializing the dense ``[T, B, N]`` spike output. ``backend="torch"`` is
    the correctness fallback: it runs the reference dense forward and packs the
    resulting spikes.
    """

    if backend == "torch":
        _warn_if_cuda_torch_backend(
            inputs,
            operation="lif_forward_packed_spikes",
        )
        state, spikes = lif_unroll(inputs, initial_state, params)
        return state, pack_spikes(spikes)
    if backend == "triton":
        require_triton()
        from myelin.triton import lif_forward_packed_spikes as triton_lif_forward_packed_spikes

        return triton_lif_forward_packed_spikes(
            inputs,
            initial_state,
            params,
            block_b=block_b,
        )
    if backend == "triton_generated":
        raise ValueError(
            "backend='triton_generated' is not implemented for packed LIF forward; "
            "use 'triton' or 'auto'"
        )
    if backend == "auto":
        if inputs.is_cuda and has_triton():
            from myelin.triton import lif_forward_packed_spikes as triton_lif_forward_packed_spikes

            return triton_lif_forward_packed_spikes(
                inputs,
                initial_state,
                params,
                block_b=block_b,
            )
        if inputs.is_cuda:
            warnings.warn(
                "CUDA inputs detected but Triton is unavailable; falling back to dense "
                "PyTorch LIF forward followed by packing. Install with "
                "`uv sync --extra cuda` to enable direct packed spike writes.",
                RuntimeWarning,
                stacklevel=2,
            )
        state, spikes = lif_unroll(inputs, initial_state, params)
        return state, pack_spikes(spikes)

    msg = f"unsupported backend: {backend}"
    raise ValueError(msg)


def alif_forward(
    inputs: torch.Tensor,
    initial_state: ALIFState,
    params: ALIFParams,
    *,
    backend: Backend = "auto",
    block_size: int = 256,
) -> tuple[ALIFState, torch.Tensor]:
    """Run ALIF forward with an explicit backend choice."""

    if backend == "torch":
        _warn_if_cuda_torch_backend(
            inputs,
            operation="alif_forward",
            faster_backend="triton_generated",
        )
        return alif_unroll(inputs, initial_state, params)
    if backend == "triton":
        raise ValueError("backend='triton' is not implemented for ALIF; use 'triton_generated'")
    if backend == "triton_generated":
        require_triton()
        from myelin.triton import generated_alif_forward

        return generated_alif_forward(inputs, initial_state, params, block_size=block_size)
    if backend == "auto":
        if inputs.is_cuda and has_triton():
            from myelin.triton import generated_alif_forward

            return generated_alif_forward(inputs, initial_state, params, block_size=block_size)
        if inputs.is_cuda:
            warnings.warn(
                "CUDA inputs detected but Triton is unavailable; falling back to the "
                "PyTorch ALIF reference path. Install with `uv sync --extra cuda` "
                "to enable the generated fused-time Triton kernel.",
                RuntimeWarning,
                stacklevel=2,
            )
        return alif_unroll(inputs, initial_state, params)

    msg = f"unsupported backend: {backend}"
    raise ValueError(msg)


def izhikevich_forward(
    inputs: torch.Tensor,
    initial_state: IzhikevichState,
    params: IzhikevichParams,
    *,
    backend: Backend = "auto",
    block_size: int = 256,
) -> tuple[IzhikevichState, torch.Tensor]:
    """Run Izhikevich forward with an explicit backend choice."""

    if backend == "torch":
        _warn_if_cuda_torch_backend(
            inputs,
            operation="izhikevich_forward",
            faster_backend="triton_generated",
        )
        return izhikevich_unroll(inputs, initial_state, params)
    if backend == "triton":
        raise ValueError(
            "backend='triton' is not implemented for Izhikevich; use 'triton_generated'"
        )
    if backend == "triton_generated":
        require_triton()
        from myelin.triton import generated_izhikevich_forward

        return generated_izhikevich_forward(inputs, initial_state, params, block_size=block_size)
    if backend == "auto":
        if inputs.is_cuda and has_triton():
            from myelin.triton import generated_izhikevich_forward

            return generated_izhikevich_forward(
                inputs, initial_state, params, block_size=block_size
            )
        if inputs.is_cuda:
            warnings.warn(
                "CUDA inputs detected but Triton is unavailable; falling back to the "
                "PyTorch Izhikevich reference path. Install with `uv sync --extra cuda` "
                "to enable the generated fused-time Triton kernel.",
                RuntimeWarning,
                stacklevel=2,
            )
        return izhikevich_unroll(inputs, initial_state, params)

    msg = f"unsupported backend: {backend}"
    raise ValueError(msg)


def surrogate_lif_forward(
    inputs: torch.Tensor,
    initial_state: LIFState,
    params: LIFParams,
    *,
    surrogate: SurrogateName = "atan",
    surrogate_slope: float = 10.0,
    hard_forward: bool = True,
    backend: Backend = "auto",
    block_size: int = 256,
) -> tuple[LIFState, torch.Tensor]:
    """Run surrogate LIF forward with an explicit backend choice."""

    surrogate_fn = surrogate_from_name(surrogate)
    if backend == "torch":
        if hard_forward:
            _warn_if_cuda_torch_backend(inputs, operation="surrogate_lif_forward")
        return surrogate_lif_unroll(
            inputs,
            initial_state,
            params,
            surrogate=surrogate_fn,
            surrogate_slope=surrogate_slope,
            hard_forward=hard_forward,
        )
    if backend == "triton":
        if not hard_forward:
            raise ValueError("Triton surrogate LIF currently supports only hard_forward=True")
        require_triton()
        from myelin.autograd import triton_surrogate_lif_function

        return triton_surrogate_lif_function(
            inputs,
            initial_state,
            params,
            surrogate=surrogate,
            surrogate_slope=surrogate_slope,
            hard_forward=hard_forward,
            block_size=block_size,
        )
    if backend == "triton_generated":
        if not hard_forward:
            raise ValueError(
                "generated Triton surrogate LIF currently supports only hard_forward=True"
            )
        require_triton()
        from myelin.autograd import generated_triton_surrogate_lif_function

        return generated_triton_surrogate_lif_function(
            inputs,
            initial_state,
            params,
            surrogate=surrogate,
            surrogate_slope=surrogate_slope,
            hard_forward=hard_forward,
            block_size=block_size,
        )
    if backend == "auto":
        if inputs.is_cuda and has_triton() and hard_forward:
            from myelin.autograd import triton_surrogate_lif_function

            return triton_surrogate_lif_function(
                inputs,
                initial_state,
                params,
                surrogate=surrogate,
                surrogate_slope=surrogate_slope,
                hard_forward=hard_forward,
                block_size=block_size,
            )
        if inputs.is_cuda:
            warnings.warn(
                "CUDA inputs detected but Triton is unavailable; falling back to the "
                "PyTorch surrogate LIF reference path. Install with "
                "`uv sync --extra cuda` to enable the fused-time Triton kernel.",
                RuntimeWarning,
                stacklevel=2,
            )
        return surrogate_lif_unroll(
            inputs,
            initial_state,
            params,
            surrogate=surrogate_fn,
            surrogate_slope=surrogate_slope,
            hard_forward=hard_forward,
        )

    msg = f"unsupported backend: {backend}"
    raise ValueError(msg)


def surrogate_alif_forward(
    inputs: torch.Tensor,
    initial_state: ALIFState,
    params: ALIFParams,
    *,
    surrogate: SurrogateName = "atan",
    surrogate_slope: float = 10.0,
    hard_forward: bool = True,
    backend: Backend = "auto",
) -> tuple[ALIFState, torch.Tensor]:
    """Run surrogate ALIF forward with an explicit backend choice."""

    surrogate_fn = surrogate_from_name(surrogate)
    if backend == "torch":
        if hard_forward:
            _warn_if_cuda_torch_backend(
                inputs,
                operation="surrogate_alif_forward",
                faster_backend="triton_generated",
            )
        return surrogate_alif_unroll(
            inputs,
            initial_state,
            params,
            surrogate=surrogate_fn,
            surrogate_slope=surrogate_slope,
            hard_forward=hard_forward,
        )
    if backend in {"triton", "triton_generated"}:
        raise ValueError(
            "surrogate ALIF generated backward is planned but not implemented yet; "
            "use backend='torch'"
        )
    if backend == "auto":
        if inputs.is_cuda and has_triton() and hard_forward:
            warnings.warn(
                "CUDA inputs detected and Triton is available, but surrogate ALIF "
                "generated backward is not implemented yet; falling back to the "
                "PyTorch surrogate ALIF reference path.",
                RuntimeWarning,
                stacklevel=2,
            )
        return surrogate_alif_unroll(
            inputs,
            initial_state,
            params,
            surrogate=surrogate_fn,
            surrogate_slope=surrogate_slope,
            hard_forward=hard_forward,
        )

    msg = f"unsupported backend: {backend}"
    raise ValueError(msg)


def linear_surrogate_lif_forward(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    params: LIFParams,
    *,
    surrogate: SurrogateName = "atan",
    surrogate_slope: float = 10.0,
    hard_forward: bool = True,
    backend: Backend = "auto",
    checkpoint_size: CheckpointSize | None = None,
    block_b: int = 16,
    block_n: int = 32,
    block_f: int = 32,
) -> tuple[LIFState, torch.Tensor]:
    """Run dense projection plus surrogate LIF through a selected backend.

    The Torch backend streams the projection through a custom autograd boundary
    instead of requiring users to materialize currents themselves. Triton
    backends use fused dense-synapse kernels when CUDA and Triton are available.
    The Triton checkpoint path recomputes recurrent traces chunk-by-chunk during
    backward, including optional input, weight, and bias gradients.
    """

    resolved_checkpoint_size = (
        None
        if checkpoint_size is None
        else resolve_checkpoint_size(int(inputs.shape[0]), checkpoint_size)
    )

    if backend == "torch":
        if hard_forward:
            _warn_if_cuda_torch_backend(
                inputs,
                operation="linear_surrogate_lif_forward",
            )
        if resolved_checkpoint_size is not None:
            from myelin.autograd import linear_surrogate_lif_checkpoint_function

            return linear_surrogate_lif_checkpoint_function(
                inputs,
                weight,
                bias,
                params,
                surrogate=surrogate,
                surrogate_slope=surrogate_slope,
                hard_forward=hard_forward,
                checkpoint_size=resolved_checkpoint_size,
            )
        from myelin.autograd import linear_surrogate_lif_stream_function

        return linear_surrogate_lif_stream_function(
            inputs,
            weight,
            bias,
            params,
            surrogate=surrogate,
            surrogate_slope=surrogate_slope,
            hard_forward=hard_forward,
        )

    if backend == "triton":
        require_triton()
        if resolved_checkpoint_size is not None:
            from myelin.autograd import triton_linear_surrogate_lif_checkpoint_function

            return triton_linear_surrogate_lif_checkpoint_function(
                inputs,
                weight,
                bias,
                params,
                surrogate=surrogate,
                surrogate_slope=surrogate_slope,
                hard_forward=hard_forward,
                checkpoint_size=resolved_checkpoint_size,
                block_b=block_b,
                block_n=block_n,
                block_f=block_f,
            )
        from myelin.autograd import triton_linear_surrogate_lif_function

        return triton_linear_surrogate_lif_function(
            inputs,
            weight,
            bias,
            params,
            surrogate=surrogate,
            surrogate_slope=surrogate_slope,
            hard_forward=hard_forward,
            block_b=block_b,
            block_n=block_n,
            block_f=block_f,
        )

    if backend == "triton_generated":
        require_triton()
        if resolved_checkpoint_size is not None:
            from myelin.autograd import generated_triton_linear_surrogate_lif_checkpoint_function

            return generated_triton_linear_surrogate_lif_checkpoint_function(
                inputs,
                weight,
                bias,
                params,
                surrogate=surrogate,
                surrogate_slope=surrogate_slope,
                hard_forward=hard_forward,
                checkpoint_size=resolved_checkpoint_size,
                block_b=block_b,
                block_n=block_n,
                block_f=block_f,
            )
        from myelin.autograd import generated_triton_linear_surrogate_lif_function

        return generated_triton_linear_surrogate_lif_function(
            inputs,
            weight,
            bias,
            params,
            surrogate=surrogate,
            surrogate_slope=surrogate_slope,
            hard_forward=hard_forward,
            block_b=block_b,
            block_n=block_n,
            block_f=block_f,
        )

    if backend == "auto":
        if inputs.is_cuda and has_triton():
            return linear_surrogate_lif_forward(
                inputs,
                weight,
                bias,
                params,
                surrogate=surrogate,
                surrogate_slope=surrogate_slope,
                hard_forward=hard_forward,
                backend="triton",
                checkpoint_size=resolved_checkpoint_size,
                block_b=block_b,
                block_n=block_n,
                block_f=block_f,
            )
        if inputs.is_cuda:
            warnings.warn(
                "CUDA inputs detected but Triton is unavailable; falling back to the "
                "PyTorch streamed LinearSurrogateLIF path. Install with "
                "`uv sync --extra cuda` to enable fused dense-synapse Triton kernels.",
                RuntimeWarning,
                stacklevel=2,
            )
        return linear_surrogate_lif_forward(
            inputs,
            weight,
            bias,
            params,
            surrogate=surrogate,
            surrogate_slope=surrogate_slope,
            hard_forward=hard_forward,
            backend="torch",
            checkpoint_size=resolved_checkpoint_size,
            block_b=block_b,
            block_n=block_n,
            block_f=block_f,
        )

    msg = f"unsupported backend: {backend}"
    raise ValueError(msg)


def linear_surrogate_lif_rate_forward(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    params: LIFParams,
    *,
    surrogate: SurrogateName = "atan",
    surrogate_slope: float = 10.0,
    hard_forward: bool = True,
    backend: Backend = "auto",
    checkpoint_size: CheckpointSize = 25,
    block_b: int = 16,
    block_n: int = 32,
    block_f: int = 32,
    reduction: Literal["mean", "none"] = "mean",
) -> tuple[LIFState, torch.Tensor]:
    """Run dense projection plus surrogate LIF and return spike rates.

    Triton uses a checkpointed path that avoids materializing dense spike
    outputs. ``backend="triton_generated"`` uses the same low-memory forward
    with generated checkpoint backward code. Torch computes rates from the dense
    reference output. ``backend="triton_compile"`` is an explicit experimental
    path for compile-visible fast-sigmoid hard-forward rate training. It is
    intended for longer-T memory pressure and is not selected by
    ``backend="auto"``.
    """

    resolved_checkpoint_size = resolve_checkpoint_size(int(inputs.shape[0]), checkpoint_size)

    if backend == "auto":
        backend = "triton" if inputs.is_cuda and has_triton() else "torch"

    if backend == "triton":
        require_triton()
        from myelin.autograd import triton_linear_surrogate_lif_checkpoint_rate_function

        return triton_linear_surrogate_lif_checkpoint_rate_function(
            inputs,
            weight,
            bias,
            params,
            surrogate=surrogate,
            surrogate_slope=surrogate_slope,
            hard_forward=hard_forward,
            checkpoint_size=resolved_checkpoint_size,
            block_b=block_b,
            block_n=block_n,
            block_f=block_f,
            reduction=reduction,
        )

    if backend == "triton_generated":
        require_triton()
        from myelin.autograd import (
            generated_triton_linear_surrogate_lif_checkpoint_rate_function,
        )

        return generated_triton_linear_surrogate_lif_checkpoint_rate_function(
            inputs,
            weight,
            bias,
            params,
            surrogate=surrogate,
            surrogate_slope=surrogate_slope,
            hard_forward=hard_forward,
            checkpoint_size=resolved_checkpoint_size,
            block_b=block_b,
            block_n=block_n,
            block_f=block_f,
            reduction=reduction,
        )

    if backend == "triton_compile":
        require_triton()
        if not hard_forward:
            raise ValueError(
                "backend='triton_compile' is experimental and currently supports only "
                "hard_forward=True"
            )
        if surrogate != "fast_sigmoid":
            raise ValueError(
                "backend='triton_compile' is experimental and currently supports only "
                "surrogate='fast_sigmoid'"
            )
        if bias is None:
            from myelin.triton import linear_lif_checkpoint_rate_forward_no_bias_op

            final_membrane, spike_rates, _chunk_starts = (
                linear_lif_checkpoint_rate_forward_no_bias_op(
                    inputs,
                    weight,
                    params.decay,
                    params.threshold,
                    params.reset,
                    surrogate_slope,
                    resolved_checkpoint_size,
                    block_b,
                    block_n,
                    block_f,
                )
            )
        else:
            from myelin.triton import linear_lif_checkpoint_rate_forward_bias_op

            final_membrane, spike_rates, _chunk_starts = linear_lif_checkpoint_rate_forward_bias_op(
                inputs,
                weight,
                bias,
                params.decay,
                params.threshold,
                params.reset,
                surrogate_slope,
                resolved_checkpoint_size,
                block_b,
                block_n,
                block_f,
            )
        if reduction == "mean":
            return LIFState(membrane=final_membrane), spike_rates.mean()
        return LIFState(membrane=final_membrane), spike_rates

    if backend == "torch":
        from myelin.autograd import linear_surrogate_lif_checkpoint_rate_function

        return linear_surrogate_lif_checkpoint_rate_function(
            inputs,
            weight,
            bias,
            params,
            surrogate=surrogate,
            surrogate_slope=surrogate_slope,
            hard_forward=hard_forward,
            checkpoint_size=resolved_checkpoint_size,
            reduction=reduction,
        )

    msg = f"unsupported backend: {backend}"
    raise ValueError(msg)


def linear_surrogate_lif_packed_forward(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    params: LIFParams,
    *,
    surrogate: SurrogateName = "atan",
    surrogate_slope: float = 10.0,
    hard_forward: bool = True,
    backend: Backend = "auto",
    checkpoint_size: CheckpointSize = 25,
    block_b: int = 16,
    block_f: int = 32,
) -> tuple[LIFState, PackedSpikes]:
    """Run dense projection plus surrogate LIF and return packed spike bits.

    This is a forward-only memory/communication surface. The packed ``int32``
    output is intentionally not differentiable; use ``linear_surrogate_lif_forward``
    or ``linear_surrogate_lif_rate_forward`` for training losses.
    """

    resolved_checkpoint_size = resolve_checkpoint_size(int(inputs.shape[0]), checkpoint_size)

    if backend == "auto":
        backend = "triton" if inputs.is_cuda and has_triton() else "torch"

    if backend == "triton":
        require_triton()
        if not hard_forward:
            raise ValueError(
                "Triton packed surrogate LIF currently supports only hard_forward=True"
            )
        from myelin.triton import linear_surrogate_lif_checkpoint_packed_forward

        state, packed_spikes, _chunk_starts = linear_surrogate_lif_checkpoint_packed_forward(
            inputs,
            weight,
            bias,
            params,
            checkpoint_size=resolved_checkpoint_size,
            block_b=block_b,
            block_f=block_f,
        )
        return state, packed_spikes

    if backend == "triton_generated":
        raise ValueError(
            "backend='triton_generated' is not implemented for packed linear surrogate LIF; "
            "use 'triton', 'torch', or 'auto'"
        )

    if backend == "torch":
        if hard_forward:
            _warn_if_cuda_torch_backend(
                inputs,
                operation="linear_surrogate_lif_packed_forward",
            )
        state, spikes = linear_surrogate_lif_forward(
            inputs,
            weight,
            bias,
            params,
            surrogate=surrogate,
            surrogate_slope=surrogate_slope,
            hard_forward=hard_forward,
            backend="torch",
            checkpoint_size=resolved_checkpoint_size,
        )
        return state, pack_spikes(spikes)

    msg = f"unsupported backend: {backend}"
    raise ValueError(msg)


def two_layer_surrogate_lif_rate_recompute_forward(
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
    checkpoint_size: CheckpointSize = 25,
) -> torch.Tensor:
    """Run a narrow two-layer LIF rate model with whole-model backward recompute."""

    resolved_checkpoint_size = resolve_checkpoint_size(int(inputs.shape[0]), checkpoint_size)
    from myelin.autograd import two_layer_surrogate_lif_rate_recompute_function

    return two_layer_surrogate_lif_rate_recompute_function(
        inputs,
        hidden_weight,
        hidden_bias,
        output_weight,
        output_bias,
        params,
        surrogate=surrogate,
        surrogate_slope=surrogate_slope,
        hard_forward=hard_forward,
        checkpoint_size=resolved_checkpoint_size,
    )


def two_layer_surrogate_lif_rate_packed_hidden_forward(
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
    checkpoint_size: CheckpointSize = 25,
    block_b: int = 16,
    block_n: int = 32,
    block_f: int = 32,
) -> torch.Tensor:
    """Run a two-layer LIF rate model while saving hidden spikes bitpacked."""

    resolved_checkpoint_size = resolve_checkpoint_size(int(inputs.shape[0]), checkpoint_size)
    from myelin.autograd import two_layer_surrogate_lif_rate_packed_hidden_function

    return two_layer_surrogate_lif_rate_packed_hidden_function(
        inputs,
        hidden_weight,
        hidden_bias,
        output_weight,
        output_bias,
        params,
        surrogate=surrogate,
        surrogate_slope=surrogate_slope,
        hard_forward=hard_forward,
        checkpoint_size=resolved_checkpoint_size,
        block_b=block_b,
        block_n=block_n,
        block_f=block_f,
    )


def fused_lif_unroll(
    inputs: torch.Tensor,
    initial_state: LIFState,
    params: LIFParams,
    *,
    backend: Backend = "auto",
    block_size: int = 256,
) -> tuple[LIFState, torch.Tensor]:
    """Compatibility alias for ``lif_forward``."""

    return lif_forward(
        inputs,
        initial_state,
        params,
        backend=backend,
        block_size=block_size,
    )
