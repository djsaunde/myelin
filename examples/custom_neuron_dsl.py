"""Author and run custom neuron variants through the DSL."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from typing import Literal

import torch

from myelin import (
    CustomNeuronCell,
    CustomSurrogateNeuronCell,
    NeuronBuilder,
    TimeUnroll,
    analyze_neuron_ir,
    evaluate_neuron_unroll,
    fast_sigmoid_surrogate,
)
from myelin.dsl import where

Variant = Literal["lif", "alif", "refractory_lif"]


def build_custom_lif():
    """Build a hard-reset LIF neuron through the public DSL builder."""

    builder = NeuronBuilder("custom_lif")
    membrane = builder.state("membrane")
    current = builder.input("input_current")
    decay = builder.param("decay")
    threshold = builder.param("threshold")
    reset = builder.param("reset")

    pre_reset = membrane * decay + current
    did_spike = pre_reset.ge(threshold)
    spike = where(did_spike, 1.0, 0.0)
    return builder.build(
        next_state={"membrane": where(did_spike, reset, pre_reset)},
        outputs={"spike": spike},
    )


def build_custom_alif():
    """Build an ALIF-style neuron using only public DSL builder methods."""

    builder = NeuronBuilder("custom_alif")
    membrane = builder.state("membrane")
    adaptation = builder.state("adaptation")
    current = builder.input("input_current")
    decay = builder.param("decay")
    adaptation_decay = builder.param("adaptation_decay")
    threshold = builder.param("threshold")
    reset = builder.param("reset")
    beta = builder.param("beta")

    pre_reset = membrane * decay + current
    adaptive_threshold = threshold + beta * adaptation
    did_spike = pre_reset.ge(adaptive_threshold)
    spike = where(did_spike, 1.0, 0.0)
    return builder.build(
        next_state={
            "membrane": where(did_spike, reset, pre_reset),
            "adaptation": adaptation * adaptation_decay + spike,
        },
        outputs={"spike": spike},
    )


def build_custom_refractory_lif():
    """Build a LIF neuron with an integer-like refractory counter state."""

    builder = NeuronBuilder("custom_refractory_lif")
    membrane = builder.state("membrane")
    refractory = builder.state("refractory")
    current = builder.input("input_current")
    decay = builder.param("decay")
    threshold = builder.param("threshold")
    reset = builder.param("reset")
    refractory_steps = builder.param("refractory_steps")

    active_refractory = refractory.ge(0.5)
    pre_reset = membrane * decay + current
    candidate_spike = pre_reset.ge(threshold)
    spike = where(active_refractory, 0.0, where(candidate_spike, 1.0, 0.0))
    did_spike = spike.ge(0.5)
    next_refractory = where(
        did_spike,
        refractory_steps,
        where(active_refractory, refractory - 1.0, 0.0),
    )
    return builder.build(
        next_state={
            "membrane": where(active_refractory, reset, where(did_spike, reset, pre_reset)),
            "refractory": next_refractory,
        },
        outputs={"spike": spike},
    )


def build_variant(variant: Variant):
    if variant == "lif":
        return build_custom_lif()
    if variant == "alif":
        return build_custom_alif()
    if variant == "refractory_lif":
        return build_custom_refractory_lif()
    raise ValueError(f"unsupported variant: {variant}")


def params_for_variant(variant: Variant) -> dict[str, float]:
    if variant == "lif":
        return {
            "decay": 0.85,
            "threshold": 1.0,
            "reset": 0.0,
        }
    if variant == "alif":
        return {
            "decay": 0.85,
            "adaptation_decay": 0.9,
            "threshold": 1.0,
            "reset": 0.0,
            "beta": 0.5,
        }
    if variant == "refractory_lif":
        return {
            "decay": 0.85,
            "threshold": 1.0,
            "reset": 0.0,
            "refractory_steps": 2.0,
        }
    raise ValueError(f"unsupported variant: {variant}")


def initial_state_for_variant(
    variant: Variant,
    inputs: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if variant == "lif":
        return {
            "membrane": torch.zeros(inputs.shape[1:], device=inputs.device),
        }
    if variant == "alif":
        return {
            "membrane": torch.zeros(inputs.shape[1:], device=inputs.device),
            "adaptation": torch.zeros(inputs.shape[1:], device=inputs.device),
        }
    if variant == "refractory_lif":
        return {
            "membrane": torch.zeros(inputs.shape[1:], device=inputs.device),
            "refractory": torch.zeros(inputs.shape[1:], device=inputs.device),
        }
    raise ValueError(f"unsupported variant: {variant}")


def run_reference(
    inputs: torch.Tensor,
    *,
    variant: Variant,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    ir = build_variant(variant)
    params = params_for_variant(variant)
    state = initial_state_for_variant(variant, inputs)
    return evaluate_neuron_unroll(ir, inputs, state, params)


def run_module(
    inputs: torch.Tensor,
    *,
    variant: Variant,
    backend: str = "auto",
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    ir = build_variant(variant)
    params = params_for_variant(variant)
    state = initial_state_for_variant(variant, inputs)
    cell = CustomNeuronCell(ir, params)
    unroll = TimeUnroll(cell, backend=backend)  # type: ignore[arg-type]
    return unroll(inputs, state)


def print_summary_header(state_names: tuple[str, ...]) -> None:
    columns = " | ".join(f"Final {name.title()} Mean" for name in state_names)
    print(f"| Backend | Spike Rate | {columns} |", flush=True)
    print(f"|---|---:|{''.join('---:|' for _ in state_names)}", flush=True)


def print_validation_report(variant: Variant) -> None:
    report = analyze_neuron_ir(build_variant(variant))
    print("| Check | Value |", flush=True)
    print("|---|---:|", flush=True)
    print(f"| variant | {variant} |", flush=True)
    print(f"| valid | {report.is_valid} |", flush=True)
    print(f"| supports_unroll_api | {report.supports_unroll_api} |", flush=True)
    print(f"| supports_generated_forward | {report.supports_generated_forward} |", flush=True)
    print(
        f"| generated_forward_errors | {', '.join(report.generated_forward_errors) or 'none'} |",
        flush=True,
    )
    print(f"| supports_generated_backward | {report.supports_generated_backward} |", flush=True)
    print(
        f"| generated_backward_errors | {', '.join(report.generated_backward_errors) or 'none'} |",
        flush=True,
    )
    print()


def print_summary_row(
    *,
    backend: str,
    final_state: Mapping[str, torch.Tensor],
    spikes: torch.Tensor,
) -> None:
    state_means = " | ".join(f"{float(value.mean()):.6f}" for value in final_state.values())
    print(
        f"| {backend} | {float(spikes.mean()):.6f} | {state_means} |",
        flush=True,
    )


def assert_states_close(
    actual: Mapping[str, torch.Tensor],
    expected: Mapping[str, torch.Tensor],
) -> None:
    assert tuple(actual) == tuple(expected)
    for name, expected_value in expected.items():
        assert torch.allclose(actual[name], expected_value)


def print_training_check(
    inputs: torch.Tensor,
    *,
    variant: Variant,
    backend: str = "auto",
) -> None:
    report = analyze_neuron_ir(build_variant(variant))
    if not report.supports_generated_backward:
        return

    training_inputs = inputs.detach().clone().requires_grad_(True)
    ir = build_variant(variant)
    cell = CustomSurrogateNeuronCell(
        ir,
        params_for_variant(variant),
        surrogate=fast_sigmoid_surrogate,
        surrogate_slope=5.0,
    )
    unroll = TimeUnroll(cell, backend=backend)  # type: ignore[arg-type]
    _state, spikes = unroll(training_inputs, initial_state_for_variant(variant, training_inputs))
    loss = spikes.mean()
    loss.backward()
    grad_norm = 0.0 if training_inputs.grad is None else float(training_inputs.grad.norm().item())

    print()
    print("## Surrogate Training Check", flush=True)
    print("| Backend | Loss | Input Grad Norm |", flush=True)
    print("|---|---:|---:|", flush=True)
    print(f"| {backend} | {float(loss.item()):.8f} | {grad_norm:.8f} |", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--variant", choices=("lif", "alif", "refractory_lif"), default="alif")
    parser.add_argument("--timesteps", type=int, default=8)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--neurons", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    inputs = torch.rand((args.timesteps, args.batch, args.neurons), device=args.device)
    variant: Variant = args.variant
    final_state, spikes = run_reference(inputs, variant=variant)
    module_state, module_spikes = run_module(inputs, variant=variant, backend="auto")

    print_validation_report(variant)
    print_summary_header(tuple(final_state))
    print_summary_row(backend="python_evaluator", final_state=final_state, spikes=spikes)
    print_summary_row(backend="module_auto", final_state=module_state, spikes=module_spikes)
    assert_states_close(module_state, final_state)
    assert torch.equal(module_spikes, spikes)
    print_training_check(inputs, variant=variant, backend="auto")

    if inputs.is_cuda:
        generated_state, generated_spikes = run_module(
            inputs,
            variant=variant,
            backend="triton_generated",
        )
        print_summary_row(
            backend="module_triton_generated",
            final_state=generated_state,
            spikes=generated_spikes,
        )
        assert_states_close(generated_state, final_state)
        assert torch.equal(generated_spikes, spikes)


if __name__ == "__main__":
    main()
