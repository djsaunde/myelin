"""Use a custom surrogate derivative IR in an online LIF layer."""

from __future__ import annotations

import argparse

import torch

from myelin import (
    LIFParams,
    LinearOnlineLIF,
    SurrogateBuilder,
    linear_lif_online_eligibility_grad,
)


def build_wide_fast_surrogate():
    """Build a parameterized fast-sigmoid-style derivative IR."""

    builder = SurrogateBuilder("wide_fast")
    centered = builder.centered()
    width = builder.param("width")
    return builder.build(0.5 / (1.0 + (centered / width).abs()).square())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--timesteps", type=int, default=6)
    parser.add_argument("--batch", type=int, default=3)
    parser.add_argument("--features", type=int, default=4)
    parser.add_argument("--neurons", type=int, default=5)
    parser.add_argument("--width", type=float, default=1.25)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    inputs = torch.rand(
        (args.timesteps, args.batch, args.features),
        device=args.device,
    )
    learning_signal = (
        torch.rand(
            (args.timesteps, args.batch, args.neurons),
            device=args.device,
        )
        - 0.5
    )
    params = LIFParams(tau_mem=8.0, threshold=0.7, reset=-0.25)
    surrogate_ir = build_wide_fast_surrogate()
    surrogate_params = {"width": args.width}
    layer = LinearOnlineLIF(
        args.features,
        args.neurons,
        params,
        surrogate=surrogate_ir,
        surrogate_slope=5.0,
        surrogate_params=surrogate_params,
    ).to(device=args.device)

    module_result = layer(inputs, learning_signal)
    functional_result = linear_lif_online_eligibility_grad(
        inputs,
        layer.synapse.weight,
        layer.synapse.bias,
        learning_signal,
        params,
        surrogate=surrogate_ir,
        surrogate_slope=5.0,
        surrogate_params=surrogate_params,
    )
    grad_weight_error = float(
        (module_result.grad_weight - functional_result.grad_weight).abs().max().item()
    )
    grad_bias_error = (
        0.0
        if module_result.grad_bias is None or functional_result.grad_bias is None
        else float((module_result.grad_bias - functional_result.grad_bias).abs().max().item())
    )

    print("| Metric | Value |", flush=True)
    print("|---|---:|", flush=True)
    print(f"| surrogate_params.width | {args.width:.6f} |", flush=True)
    print(f"| spike_rate | {float(module_result.spikes.mean()):.6f} |", flush=True)
    print(
        f"| grad_weight_norm | {float(module_result.grad_weight.detach().norm()):.6f} |",
        flush=True,
    )
    print(f"| grad_weight_max_error | {grad_weight_error:.6e} |", flush=True)
    print(f"| grad_bias_max_error | {grad_bias_error:.6e} |", flush=True)


if __name__ == "__main__":
    main()
