"""Benchmark custom surrogate-neuron training against the built-in LIF path."""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

import torch

from spiker import (
    CustomSurrogateNeuronCell,
    LIFParams,
    LIFState,
    LinearCustomSurrogateNeuron,
    LinearCustomSurrogateNeuronRate,
    LinearSurrogateLIF,
    LinearSurrogateLIFRate,
    SurrogateLIFCell,
    TimeUnroll,
)
from spiker._optional import has_triton
from spiker.baselines import synchronize_if_needed
from spiker.benchmarks.custom_neuron_module import build_custom_lif_ir
from spiker.benchmarks.lif import format_memory, format_ms, format_speedup, gpu_name
from spiker.modules import fast_sigmoid_surrogate

StepOutput = tuple[
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
]


@dataclass(frozen=True)
class TrainingResult:
    """One forward+backward benchmark row."""

    variant: str
    backend: str
    seconds: float | None
    speedup_vs_builtin_torch: float | None
    peak_bytes: int | None
    loss: float | None
    final_membrane_max_error: float | None
    loss_max_error: float | None
    input_grad_max_error: float | None
    weight_grad_max_error: float | None
    bias_grad_max_error: float | None
    error: str | None = None


@dataclass(frozen=True)
class PairwiseResult:
    """Custom-vs-built-in correctness comparison for one backend surface."""

    surface: str
    backend: str
    loss_max_error: float | None
    final_membrane_max_error: float | None
    input_grad_max_error: float | None
    weight_grad_max_error: float | None
    bias_grad_max_error: float | None
    error: str | None = None


def make_inputs(
    *,
    timesteps: int,
    batch: int,
    neurons: int,
    device: str,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return torch.rand((timesteps, batch, neurons), device=device, generator=generator)


def make_feature_inputs(
    *,
    timesteps: int,
    batch: int,
    features: int,
    device: str,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + 1)
    return torch.rand((timesteps, batch, features), device=device, generator=generator)


def params_from_decay(
    *,
    decay: float,
    threshold: float,
    reset: float,
) -> LIFParams:
    if decay >= 1.0:
        raise ValueError("decay must be less than 1.0")
    return LIFParams(tau_mem=1.0 / (1.0 - decay), threshold=threshold, reset=reset)


def initial_membrane(inputs: torch.Tensor) -> torch.Tensor:
    return torch.zeros(inputs.shape[1:], dtype=inputs.dtype, device=inputs.device)


def make_builtin_step(
    inputs: torch.Tensor,
    params: LIFParams,
    *,
    backend: str,
) -> Callable[[], StepOutput]:
    unroll = TimeUnroll(
        SurrogateLIFCell(
            params,
            surrogate=fast_sigmoid_surrogate,
            surrogate_slope=5.0,
        ),
        backend=backend,  # pyright: ignore[reportArgumentType]
    )

    def step() -> StepOutput:
        step_inputs = inputs.detach().clone().requires_grad_(True)
        initial = LIFState(membrane=initial_membrane(step_inputs))
        state, spikes = unroll(step_inputs, initial)
        if not isinstance(state, LIFState):
            raise TypeError("built-in surrogate LIF returned non-LIF state")
        loss = spikes.mean()
        loss.backward()
        if step_inputs.grad is None:
            raise RuntimeError("built-in surrogate LIF did not produce input gradients")
        return loss.detach(), state.membrane.detach(), step_inputs.grad.detach(), None, None

    return step


def make_custom_step(
    inputs: torch.Tensor,
    custom_params: dict[str, float],
    *,
    backend: str,
) -> Callable[[], StepOutput]:
    cell = CustomSurrogateNeuronCell(
        build_custom_lif_ir(),
        custom_params,
        surrogate=fast_sigmoid_surrogate,
        surrogate_slope=5.0,
    )
    unroll = TimeUnroll(cell, backend=backend)  # pyright: ignore[reportArgumentType]

    def step() -> StepOutput:
        step_inputs = inputs.detach().clone().requires_grad_(True)
        state, spikes = unroll(step_inputs, {"membrane": initial_membrane(step_inputs)})
        if not isinstance(state, dict):
            raise TypeError("custom surrogate neuron returned non-dict state")
        loss = spikes.mean()
        loss.backward()
        if step_inputs.grad is None:
            raise RuntimeError("custom surrogate neuron did not produce input gradients")
        return loss.detach(), state["membrane"].detach(), step_inputs.grad.detach(), None, None

    return step


def make_linear_step(
    inputs: torch.Tensor,
    layer: (
        LinearSurrogateLIF
        | LinearCustomSurrogateNeuron
        | LinearSurrogateLIFRate
        | LinearCustomSurrogateNeuronRate
    ),
) -> Callable[[], StepOutput]:
    def step() -> StepOutput:
        layer.zero_grad(set_to_none=True)
        step_inputs = inputs.detach().clone().requires_grad_(True)
        spikes = layer(step_inputs)
        loss = spikes.mean()
        loss.backward()
        if step_inputs.grad is None:
            raise RuntimeError("linear surrogate layer did not produce input gradients")
        weight_grad = layer.synapse.weight.grad
        if weight_grad is None:
            raise RuntimeError("linear surrogate layer did not produce weight gradients")
        bias_grad = None if layer.synapse.bias is None else layer.synapse.bias.grad
        if layer.synapse.bias is not None and bias_grad is None:
            raise RuntimeError("linear surrogate layer did not produce bias gradients")
        return (
            loss.detach(),
            None,
            step_inputs.grad.detach(),
            weight_grad.detach().clone(),
            None if bias_grad is None else bias_grad.detach().clone(),
        )

    return step


def time_step(
    step: Callable[[], object],
    *,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> tuple[float, int | None]:
    for _ in range(warmup):
        step()
    synchronize_if_needed(device)

    peak = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    for _ in range(repeats):
        step()
    synchronize_if_needed(device)
    seconds = (time.perf_counter() - start) / repeats
    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated(device)
    return seconds, peak


def run_training_variant(
    *,
    variant: str,
    backend: str,
    step: Callable[[], StepOutput],
    expected: StepOutput | None,
    builtin_torch_seconds: float | None,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> TrainingResult:
    try:
        actual = step()
        seconds, peak = time_step(step, device=device, warmup=warmup, repeats=repeats)
        speedup = None if builtin_torch_seconds is None else builtin_torch_seconds / seconds
        if expected is None:
            final_membrane_error = 0.0
            loss_error = 0.0
            input_grad_error = 0.0
            weight_grad_error = None
            bias_grad_error = None
        else:
            (
                expected_loss,
                expected_final,
                expected_input_grad,
                expected_weight_grad,
                expected_bias_grad,
            ) = expected
            if actual[1] is None or expected_final is None:
                final_membrane_error = None
            else:
                final_membrane_error = float((actual[1] - expected_final).abs().max().item())
            loss_error = float((actual[0] - expected_loss).abs().max().item())
            input_grad_error = float((actual[2] - expected_input_grad).abs().max().item())
            if actual[3] is None or expected_weight_grad is None:
                weight_grad_error = None
            else:
                weight_grad_error = float((actual[3] - expected_weight_grad).abs().max().item())
            if actual[4] is None or expected_bias_grad is None:
                bias_grad_error = None
            else:
                bias_grad_error = float((actual[4] - expected_bias_grad).abs().max().item())
        return TrainingResult(
            variant=variant,
            backend=backend,
            seconds=seconds,
            speedup_vs_builtin_torch=speedup,
            peak_bytes=peak,
            loss=float(actual[0].item()),
            final_membrane_max_error=final_membrane_error,
            loss_max_error=loss_error,
            input_grad_max_error=input_grad_error,
            weight_grad_max_error=weight_grad_error,
            bias_grad_max_error=bias_grad_error,
        )
    except Exception as exc:  # noqa: BLE001 - benchmark should report backend failures.
        return TrainingResult(
            variant=variant,
            backend=backend,
            seconds=None,
            speedup_vs_builtin_torch=None,
            peak_bytes=None,
            loss=None,
            final_membrane_max_error=None,
            loss_max_error=None,
            input_grad_max_error=None,
            weight_grad_max_error=None,
            bias_grad_max_error=None,
            error=f"{type(exc).__name__}: {exc}",
        )


def compare_outputs(
    *,
    surface: str,
    backend: str,
    builtin: StepOutput,
    custom: StepOutput,
) -> PairwiseResult:
    """Compare custom wrapper outputs against the built-in wrapper for one backend."""

    if builtin[1] is None or custom[1] is None:
        final_membrane_error = None
    else:
        final_membrane_error = float((custom[1] - builtin[1]).abs().max().item())
    if builtin[3] is None or custom[3] is None:
        weight_grad_error = None
    else:
        weight_grad_error = float((custom[3] - builtin[3]).abs().max().item())
    if builtin[4] is None or custom[4] is None:
        bias_grad_error = None
    else:
        bias_grad_error = float((custom[4] - builtin[4]).abs().max().item())
    return PairwiseResult(
        surface=surface,
        backend=backend,
        loss_max_error=float((custom[0] - builtin[0]).abs().max().item()),
        final_membrane_max_error=final_membrane_error,
        input_grad_max_error=float((custom[2] - builtin[2]).abs().max().item()),
        weight_grad_max_error=weight_grad_error,
        bias_grad_max_error=bias_grad_error,
    )


def compare_steps(
    *,
    surface: str,
    backend: str,
    builtin_step: Callable[[], StepOutput],
    custom_step: Callable[[], StepOutput],
) -> PairwiseResult:
    try:
        return compare_outputs(
            surface=surface,
            backend=backend,
            builtin=builtin_step(),
            custom=custom_step(),
        )
    except Exception as exc:  # noqa: BLE001 - benchmark should report backend failures.
        return PairwiseResult(
            surface=surface,
            backend=backend,
            loss_max_error=None,
            final_membrane_max_error=None,
            input_grad_max_error=None,
            weight_grad_max_error=None,
            bias_grad_max_error=None,
            error=f"{type(exc).__name__}: {exc}",
        )


def make_linear_layers(
    *,
    features: int,
    neurons: int,
    params: LIFParams,
    custom_params: dict[str, float],
    backend: str,
    stream_synapse: bool,
    checkpoint_size: int | None,
    linear_bias: float,
    device: torch.device,
) -> tuple[LinearSurrogateLIF, LinearCustomSurrogateNeuron]:
    builtin = LinearSurrogateLIF(
        features,
        neurons,
        params,
        surrogate=fast_sigmoid_surrogate,
        surrogate_slope=5.0,
        backend=backend,  # pyright: ignore[reportArgumentType]
        stream_synapse=stream_synapse,
        checkpoint_size=checkpoint_size,
    ).to(device)
    custom = LinearCustomSurrogateNeuron(
        features,
        neurons,
        build_custom_lif_ir(),
        custom_params,
        surrogate=fast_sigmoid_surrogate,
        surrogate_slope=5.0,
        backend=backend,  # pyright: ignore[reportArgumentType]
        stream_synapse=stream_synapse,
        checkpoint_size=checkpoint_size,
    ).to(device)
    custom.synapse.weight.data.copy_(builtin.synapse.weight)
    if builtin.synapse.bias is not None and custom.synapse.bias is not None:
        builtin.synapse.bias.data.fill_(linear_bias)
        custom.synapse.bias.data.copy_(builtin.synapse.bias)
    return builtin, custom


def make_rate_layers(
    *,
    features: int,
    neurons: int,
    params: LIFParams,
    custom_params: dict[str, float],
    backend: str,
    checkpoint_size: int,
    linear_bias: float,
    device: torch.device,
) -> tuple[LinearSurrogateLIFRate, LinearCustomSurrogateNeuronRate]:
    builtin = LinearSurrogateLIFRate(
        features,
        neurons,
        params,
        surrogate=fast_sigmoid_surrogate,
        surrogate_slope=5.0,
        backend=backend,  # pyright: ignore[reportArgumentType]
        checkpoint_size=checkpoint_size,
        reduction="none",
    ).to(device)
    custom = LinearCustomSurrogateNeuronRate(
        features,
        neurons,
        build_custom_lif_ir(),
        custom_params,
        surrogate=fast_sigmoid_surrogate,
        surrogate_slope=5.0,
        backend=backend,  # pyright: ignore[reportArgumentType]
        checkpoint_size=checkpoint_size,
        reduction="none",
    ).to(device)
    custom.synapse.weight.data.copy_(builtin.synapse.weight)
    if builtin.synapse.bias is not None and custom.synapse.bias is not None:
        builtin.synapse.bias.data.fill_(linear_bias)
        custom.synapse.bias.data.copy_(builtin.synapse.bias)
    return builtin, custom


def run_pairwise_comparisons(args: argparse.Namespace) -> list[PairwiseResult]:
    """Compare custom wrappers directly against their built-in counterparts."""

    device = torch.device(args.device)
    inputs = make_inputs(
        timesteps=args.timesteps,
        batch=args.batch,
        neurons=args.neurons,
        device=args.device,
        seed=args.seed,
    )
    feature_inputs = make_feature_inputs(
        timesteps=args.timesteps,
        batch=args.batch,
        features=args.features,
        device=args.device,
        seed=args.seed,
    )
    custom_params = {"decay": args.decay, "threshold": args.threshold, "reset": args.reset}
    params = params_from_decay(
        decay=args.decay,
        threshold=args.threshold,
        reset=args.reset,
    )

    pairwise = [
        compare_steps(
            surface="cell",
            backend="torch",
            builtin_step=make_builtin_step(inputs, params, backend="torch"),
            custom_step=make_custom_step(inputs, custom_params, backend="torch"),
        )
    ]

    linear_builtin, linear_custom = make_linear_layers(
        features=args.features,
        neurons=args.neurons,
        params=params,
        custom_params=custom_params,
        backend="torch",
        stream_synapse=False,
        checkpoint_size=None,
        linear_bias=args.linear_bias,
        device=device,
    )
    pairwise.append(
        compare_steps(
            surface="linear",
            backend="torch",
            builtin_step=make_linear_step(feature_inputs, linear_builtin),
            custom_step=make_linear_step(feature_inputs, linear_custom),
        )
    )

    rate_builtin, rate_custom = make_rate_layers(
        features=args.features,
        neurons=args.neurons,
        params=params,
        custom_params=custom_params,
        backend="torch",
        checkpoint_size=args.checkpoint_size,
        linear_bias=args.linear_bias,
        device=device,
    )
    pairwise.append(
        compare_steps(
            surface="rate",
            backend="torch",
            builtin_step=make_linear_step(feature_inputs, rate_builtin),
            custom_step=make_linear_step(feature_inputs, rate_custom),
        )
    )

    if device.type == "cuda" and has_triton():
        pairwise.append(
            compare_steps(
                surface="cell",
                backend="triton_generated",
                builtin_step=make_builtin_step(inputs, params, backend="triton_generated"),
                custom_step=make_custom_step(inputs, custom_params, backend="triton_generated"),
            )
        )
        linear_generated_builtin, linear_generated_custom = make_linear_layers(
            features=args.features,
            neurons=args.neurons,
            params=params,
            custom_params=custom_params,
            backend="triton_generated",
            stream_synapse=True,
            checkpoint_size=args.checkpoint_size,
            linear_bias=args.linear_bias,
            device=device,
        )
        pairwise.append(
            compare_steps(
                surface="linear",
                backend="triton_generated_stream",
                builtin_step=make_linear_step(feature_inputs, linear_generated_builtin),
                custom_step=make_linear_step(feature_inputs, linear_generated_custom),
            )
        )
        rate_generated_builtin, rate_generated_custom = make_rate_layers(
            features=args.features,
            neurons=args.neurons,
            params=params,
            custom_params=custom_params,
            backend="triton_generated",
            checkpoint_size=args.checkpoint_size,
            linear_bias=args.linear_bias,
            device=device,
        )
        pairwise.append(
            compare_steps(
                surface="rate",
                backend="triton_generated_rate",
                builtin_step=make_linear_step(feature_inputs, rate_generated_builtin),
                custom_step=make_linear_step(feature_inputs, rate_generated_custom),
            )
        )

    return pairwise


def run_benchmark(args: argparse.Namespace) -> list[TrainingResult]:
    device = torch.device(args.device)
    inputs = make_inputs(
        timesteps=args.timesteps,
        batch=args.batch,
        neurons=args.neurons,
        device=args.device,
        seed=args.seed,
    )
    feature_inputs = make_feature_inputs(
        timesteps=args.timesteps,
        batch=args.batch,
        features=args.features,
        device=args.device,
        seed=args.seed,
    )
    custom_params = {"decay": args.decay, "threshold": args.threshold, "reset": args.reset}
    params = params_from_decay(
        decay=args.decay,
        threshold=args.threshold,
        reset=args.reset,
    )

    builtin_torch_step = make_builtin_step(inputs, params, backend="torch")
    expected = builtin_torch_step()
    builtin_torch_seconds, builtin_torch_peak = time_step(
        builtin_torch_step,
        device=device,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    results = [
        TrainingResult(
            variant="builtin",
            backend="torch",
            seconds=builtin_torch_seconds,
            speedup_vs_builtin_torch=1.0,
            peak_bytes=builtin_torch_peak,
            loss=float(expected[0].item()),
            final_membrane_max_error=0.0,
            loss_max_error=0.0,
            input_grad_max_error=0.0,
            weight_grad_max_error=None,
            bias_grad_max_error=None,
        )
    ]

    custom_torch_step = make_custom_step(inputs, custom_params, backend="torch")
    results.append(
        run_training_variant(
            variant="custom_ir",
            backend="torch",
            step=custom_torch_step,
            expected=expected,
            builtin_torch_seconds=builtin_torch_seconds,
            device=device,
            warmup=args.warmup,
            repeats=args.repeats,
        )
    )
    if device.type == "cuda" and has_triton():
        for variant, step in (
            ("builtin", make_builtin_step(inputs, params, backend="triton_generated")),
            ("custom_ir", make_custom_step(inputs, custom_params, backend="triton_generated")),
        ):
            results.append(
                run_training_variant(
                    variant=variant,
                    backend="triton_generated",
                    step=step,
                    expected=expected,
                    builtin_torch_seconds=builtin_torch_seconds,
                    device=device,
                    warmup=args.warmup,
                    repeats=args.repeats,
                )
            )

    linear_builtin, linear_custom = make_linear_layers(
        features=args.features,
        neurons=args.neurons,
        params=params,
        custom_params=custom_params,
        backend="torch",
        stream_synapse=False,
        checkpoint_size=None,
        linear_bias=args.linear_bias,
        device=device,
    )
    linear_builtin_step = make_linear_step(feature_inputs, linear_builtin)
    linear_expected = linear_builtin_step()
    linear_builtin_seconds, linear_builtin_peak = time_step(
        linear_builtin_step,
        device=device,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    results.append(
        TrainingResult(
            variant="linear_builtin",
            backend="torch",
            seconds=linear_builtin_seconds,
            speedup_vs_builtin_torch=linear_builtin_seconds / linear_builtin_seconds,
            peak_bytes=linear_builtin_peak,
            loss=float(linear_expected[0].item()),
            final_membrane_max_error=None,
            loss_max_error=0.0,
            input_grad_max_error=0.0,
            weight_grad_max_error=0.0,
            bias_grad_max_error=0.0,
        )
    )
    results.append(
        run_training_variant(
            variant="linear_custom_ir",
            backend="torch",
            step=make_linear_step(feature_inputs, linear_custom),
            expected=linear_expected,
            builtin_torch_seconds=linear_builtin_seconds,
            device=device,
            warmup=args.warmup,
            repeats=args.repeats,
        )
    )
    if device.type == "cuda" and has_triton():
        linear_generated_builtin, linear_generated_custom = make_linear_layers(
            features=args.features,
            neurons=args.neurons,
            params=params,
            custom_params=custom_params,
            backend="triton_generated",
            stream_synapse=True,
            checkpoint_size=args.checkpoint_size,
            linear_bias=args.linear_bias,
            device=device,
        )
        for variant, step in (
            ("linear_builtin", make_linear_step(feature_inputs, linear_generated_builtin)),
            ("linear_custom_ir", make_linear_step(feature_inputs, linear_generated_custom)),
        ):
            results.append(
                run_training_variant(
                    variant=variant,
                    backend="triton_generated_stream",
                    step=step,
                    expected=linear_expected,
                    builtin_torch_seconds=linear_builtin_seconds,
                    device=device,
                    warmup=args.warmup,
                    repeats=args.repeats,
                )
            )

    rate_builtin, rate_custom = make_rate_layers(
        features=args.features,
        neurons=args.neurons,
        params=params,
        custom_params=custom_params,
        backend="torch",
        checkpoint_size=args.checkpoint_size,
        linear_bias=args.linear_bias,
        device=device,
    )
    rate_builtin_step = make_linear_step(feature_inputs, rate_builtin)
    rate_expected = rate_builtin_step()
    rate_builtin_seconds, rate_builtin_peak = time_step(
        rate_builtin_step,
        device=device,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    results.append(
        TrainingResult(
            variant="rate_builtin",
            backend="torch",
            seconds=rate_builtin_seconds,
            speedup_vs_builtin_torch=rate_builtin_seconds / rate_builtin_seconds,
            peak_bytes=rate_builtin_peak,
            loss=float(rate_expected[0].item()),
            final_membrane_max_error=None,
            loss_max_error=0.0,
            input_grad_max_error=0.0,
            weight_grad_max_error=0.0,
            bias_grad_max_error=0.0,
        )
    )
    results.append(
        run_training_variant(
            variant="rate_custom_ir",
            backend="torch",
            step=make_linear_step(feature_inputs, rate_custom),
            expected=rate_expected,
            builtin_torch_seconds=rate_builtin_seconds,
            device=device,
            warmup=args.warmup,
            repeats=args.repeats,
        )
    )
    if device.type == "cuda" and has_triton():
        rate_generated_builtin, rate_generated_custom = make_rate_layers(
            features=args.features,
            neurons=args.neurons,
            params=params,
            custom_params=custom_params,
            backend="triton_generated",
            checkpoint_size=args.checkpoint_size,
            linear_bias=args.linear_bias,
            device=device,
        )
        for variant, step in (
            ("rate_builtin", make_linear_step(feature_inputs, rate_generated_builtin)),
            ("rate_custom_ir", make_linear_step(feature_inputs, rate_generated_custom)),
        ):
            results.append(
                run_training_variant(
                    variant=variant,
                    backend="triton_generated_rate",
                    step=step,
                    expected=rate_expected,
                    builtin_torch_seconds=rate_builtin_seconds,
                    device=device,
                    warmup=args.warmup,
                    repeats=args.repeats,
                )
            )
    return results


def print_markdown(
    args: argparse.Namespace,
    results: list[TrainingResult],
    pairwise: list[PairwiseResult] | None = None,
) -> None:
    print("# Custom Surrogate Training Benchmark")
    print()
    print("Compares custom LIF-shaped surrogate wrappers against the built-in surrogate LIF paths.")
    print()
    print("## Environment")
    print()
    print(f"- `generated_utc`: `{datetime.now(UTC).isoformat(timespec='seconds')}`")
    print(f"- `device`: `{args.device}`")
    print(f"- `gpu`: `{gpu_name(args.device)}`")
    print(f"- `torch`: `{torch.__version__}`")
    print(f"- `cuda_available`: `{torch.cuda.is_available()}`")
    print(f"- `cuda_version`: `{torch.version.cuda}`")
    print(f"- `shape`: `T={args.timesteps}, B={args.batch}, N={args.neurons}`")
    print(f"- `features`: `{args.features}`")
    print(f"- `checkpoint_size`: `{args.checkpoint_size}`")
    print(f"- `linear_bias`: `{args.linear_bias}`")
    print(f"- `decay`: `{args.decay}`")
    print(f"- `threshold`: `{args.threshold}`")
    print(f"- `reset`: `{args.reset}`")
    print(f"- `warmup`: `{args.warmup}`")
    print(f"- `repeats`: `{args.repeats}`")
    print()
    print("## Results")
    print()
    print(
        "| Variant | Backend | Fwd+Bwd ms | Speedup vs Builtin Torch | Peak MB | "
        "Loss | Final Membrane Max Error | Loss Max Error | Input Grad Max Error | "
        "Weight Grad Max Error | Bias Grad Max Error | Error |"
    )
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for result in results:
        loss = "" if result.loss is None else f"{result.loss:.8f}"
        final_error = (
            ""
            if result.final_membrane_max_error is None
            else f"{result.final_membrane_max_error:.3e}"
        )
        loss_error = "" if result.loss_max_error is None else f"{result.loss_max_error:.3e}"
        grad_error = (
            "" if result.input_grad_max_error is None else f"{result.input_grad_max_error:.3e}"
        )
        weight_grad_error = (
            "" if result.weight_grad_max_error is None else f"{result.weight_grad_max_error:.3e}"
        )
        bias_grad_error = (
            "" if result.bias_grad_max_error is None else f"{result.bias_grad_max_error:.3e}"
        )
        print(
            f"| {result.variant} | "
            f"{result.backend} | "
            f"{format_ms(result.seconds)} | "
            f"{format_speedup(result.speedup_vs_builtin_torch)} | "
            f"{format_memory(result.peak_bytes)} | "
            f"{loss} | "
            f"{final_error} | "
            f"{loss_error} | "
            f"{grad_error} | "
            f"{weight_grad_error} | "
            f"{bias_grad_error} | "
            f"{result.error or ''} |"
        )
    if pairwise is not None:
        print()
        print("## Built-In vs Custom Pairwise")
        print()
        print(
            "| Surface | Backend | Loss Max Error | Final Membrane Max Error | "
            "Input Grad Max Error | Weight Grad Max Error | Bias Grad Max Error | Error |"
        )
        print("|---|---|---:|---:|---:|---:|---:|---|")
        for result in pairwise:
            loss_error = "" if result.loss_max_error is None else f"{result.loss_max_error:.3e}"
            final_error = (
                ""
                if result.final_membrane_max_error is None
                else f"{result.final_membrane_max_error:.3e}"
            )
            input_grad_error = (
                "" if result.input_grad_max_error is None else f"{result.input_grad_max_error:.3e}"
            )
            weight_grad_error = (
                ""
                if result.weight_grad_max_error is None
                else f"{result.weight_grad_max_error:.3e}"
            )
            bias_grad_error = (
                "" if result.bias_grad_max_error is None else f"{result.bias_grad_max_error:.3e}"
            )
            print(
                f"| {result.surface} | "
                f"{result.backend} | "
                f"{loss_error} | "
                f"{final_error} | "
                f"{input_grad_error} | "
                f"{weight_grad_error} | "
                f"{bias_grad_error} | "
                f"{result.error or ''} |"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--neurons", type=int, default=2048)
    parser.add_argument("--features", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint-size", type=int, default=25)
    parser.add_argument("--linear-bias", type=float, default=0.25)
    parser.add_argument("--decay", type=float, default=0.85)
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--reset", type=float, default=0.0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print_markdown(args, run_benchmark(args), run_pairwise_comparisons(args))


if __name__ == "__main__":
    main()
