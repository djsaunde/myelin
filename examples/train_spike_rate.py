"""Train a single surrogate LIF layer toward a target spike rate."""

from __future__ import annotations

import argparse
import time

import torch
from example_utils import (
    add_compile_policy_arg,
    add_grad_clip_arg,
    add_matmul_precision_arg,
    add_surrogate_args,
    add_wandb_args,
    clip_gradients,
    compile_training_model,
    configure_matmul_precision,
    finish_wandb,
    init_wandb,
    log_wandb,
    print_model_summary,
    print_step_time_summary,
    resolve_compile_policy,
)

from spiker import LinearSurrogateLIF, SpikeRateLoss, fast_sigmoid_surrogate


def synchronize_if_needed(device: str) -> None:
    torch_device = torch.device(device)
    if torch_device.type == "cuda":
        torch.cuda.synchronize(torch_device)


def evaluate(
    model: torch.nn.Module,
    loss_fn: SpikeRateLoss,
    inputs: torch.Tensor,
) -> tuple[float, float]:
    with torch.no_grad():
        spikes = model(inputs)
        loss = loss_fn(spikes)
        spike_rate = spikes.mean()
    return float(loss), float(spike_rate)


def train(
    *,
    timesteps: int,
    batch: int,
    features: int,
    neurons: int,
    target_rate: float,
    steps: int,
    lr: float,
    grad_clip: float,
    surrogate_slope: float,
    hard_forward: bool,
    device: str,
    compile_model: bool,
    seed: int,
    log_every: int | None = None,
    wandb_run=None,
) -> tuple[list[tuple[int, float, float, float | None]], float]:
    torch.manual_seed(seed)
    inputs = torch.rand((timesteps, batch, features), device=device)
    model = LinearSurrogateLIF(
        features,
        neurons,
        surrogate=fast_sigmoid_surrogate,
        surrogate_slope=surrogate_slope,
        hard_forward=hard_forward,
    ).to(device=device)
    loss_fn = SpikeRateLoss(target_rate=target_rate)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    print_model_summary(model)
    print()

    model = compile_training_model(model, compile_model)

    history: list[tuple[int, float, float, float | None]] = [
        (0, *evaluate(model, loss_fn, inputs), None)
    ]
    log_wandb(
        wandb_run,
        {"train/loss": history[0][1], "train/spike_rate": history[0][2]},
        step=0,
    )
    if log_every is not None:
        print("| Step | Loss | Spike Rate | Step ms |", flush=True)
        print("|---:|---:|---:|---:|", flush=True)
        print_history_row(history[0])

    synchronize_if_needed(device)
    training_start = time.perf_counter()
    for step in range(1, steps + 1):
        synchronize_if_needed(device)
        start = time.perf_counter()
        optimizer.zero_grad()
        loss = loss_fn(model(inputs))
        loss.backward()
        grad_norm = clip_gradients(model, grad_clip)
        optimizer.step()
        synchronize_if_needed(device)
        step_seconds = time.perf_counter() - start
        history.append((step, *evaluate(model, loss_fn, inputs), step_seconds))
        log_wandb(
            wandb_run,
            {
                "train/loss": history[-1][1],
                "train/spike_rate": history[-1][2],
                "train/step_ms": step_seconds * 1000,
                **({} if grad_norm is None else {"train/grad_norm": grad_norm}),
            },
            step=step,
        )
        if log_every is not None and (step == steps or step % log_every == 0):
            print_history_row(history[-1])
    synchronize_if_needed(device)
    total_seconds = time.perf_counter() - training_start

    return history, total_seconds


def print_history_row(row: tuple[int, float, float, float | None]) -> None:
    step, loss, spike_rate, step_seconds = row
    step_ms = "" if step_seconds is None else f"{step_seconds * 1000:.3f}"
    print(f"| {step} | {loss:.8f} | {spike_rate:.8f} | {step_ms} |", flush=True)


def print_history(
    history: list[tuple[int, float, float, float | None]],
    *,
    log_every: int,
    target_rate: float,
    total_seconds: float,
) -> None:
    print("| Step | Loss | Spike Rate | Step ms |")
    print("|---:|---:|---:|---:|")
    final_step = history[-1][0]
    for row in history:
        step = row[0]
        if step == 0 or step == final_step or step % log_every == 0:
            print_history_row(row)

    print_summary(history, target_rate=target_rate, total_seconds=total_seconds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=16)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--features", type=int, default=16)
    parser.add_argument("--neurons", type=int, default=32)
    parser.add_argument("--target-rate", type=float, default=0.2)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1.0)
    add_grad_clip_arg(parser)
    add_surrogate_args(parser)
    parser.add_argument("--device", default="cpu")
    add_compile_policy_arg(parser)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--no-stream-log", action="store_true")
    add_matmul_precision_arg(parser)
    add_wandb_args(parser)
    args = parser.parse_args()
    configure_matmul_precision(args.matmul_precision)
    compile_model = resolve_compile_policy(args.compile, args.device)

    config = {
        "device": args.device,
        "compile": compile_model,
        "compile_policy": args.compile,
        "timesteps": args.timesteps,
        "batch": args.batch,
        "features": args.features,
        "neurons": args.neurons,
        "target_rate": args.target_rate,
        "steps": args.steps,
        "lr": args.lr,
        "grad_clip": args.grad_clip,
        "matmul_precision": args.matmul_precision,
        "surrogate_slope": args.surrogate_slope,
        "hard_forward": not args.smooth_forward,
        "seed": args.seed,
    }
    wandb_run = init_wandb(
        enabled=args.wandb,
        project=args.wandb_project,
        run_name=args.wandb_run_name,
        config=config,
    )

    print(
        "config="
        f"device:{args.device},compile:{compile_model},compile_policy:{args.compile},"
        f"T:{args.timesteps},B:{args.batch},F:{args.features},N:{args.neurons},"
        f"target_rate:{args.target_rate},steps:{args.steps},lr:{args.lr},"
        f"grad_clip:{args.grad_clip},matmul_precision:{args.matmul_precision},"
        f"surrogate_slope:{args.surrogate_slope},hard_forward:{not args.smooth_forward}"
    )
    print()

    history, total_seconds = train(
        timesteps=args.timesteps,
        batch=args.batch,
        features=args.features,
        neurons=args.neurons,
        target_rate=args.target_rate,
        steps=args.steps,
        lr=args.lr,
        grad_clip=args.grad_clip,
        surrogate_slope=args.surrogate_slope,
        hard_forward=not args.smooth_forward,
        device=args.device,
        compile_model=compile_model,
        seed=args.seed,
        log_every=None if args.no_stream_log else args.log_every,
        wandb_run=wandb_run,
    )
    if args.no_stream_log:
        print_history(
            history,
            log_every=args.log_every,
            target_rate=args.target_rate,
            total_seconds=total_seconds,
        )
    else:
        print_summary(history, target_rate=args.target_rate, total_seconds=total_seconds)
    finish_wandb(wandb_run)


def print_summary(
    history: list[tuple[int, float, float, float | None]],
    *,
    target_rate: float,
    total_seconds: float,
) -> None:
    step_times = [step_seconds for _, _, _, step_seconds in history if step_seconds is not None]
    final_spike_rate = history[-1][2]
    print(flush=True)
    print(f"target_rate={target_rate:.8f}", flush=True)
    print(f"final_target_error={abs(final_spike_rate - target_rate):.8f}", flush=True)
    print(f"total_training_seconds={total_seconds:.3f}", flush=True)
    print_step_time_summary(step_times)


if __name__ == "__main__":
    main()
