"""Train a convolutional surrogate SNN on MNIST."""

from __future__ import annotations

import argparse
import math
import time
from collections.abc import Sized
from pathlib import Path
from typing import cast

import torch
from example_utils import (
    add_compile_policy_arg,
    add_encoding_arg,
    add_grad_clip_arg,
    add_matmul_precision_arg,
    add_surrogate_args,
    add_wandb_args,
    clip_gradients,
    compile_training_model,
    configure_matmul_precision,
    encode_time_series,
    finish_wandb,
    init_wandb,
    log_wandb,
    print_cuda_peak_memory_summary,
    print_model_summary,
    print_step_time_summary,
    reset_cuda_peak_memory,
    resolve_compile_policy,
)
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from train_mnist import accuracy, limited_dataset, synchronize_if_needed

from myelin import LinearSurrogateLIF, fast_sigmoid_surrogate


class ConvMNISTSNN(nn.Module):
    def __init__(
        self,
        hidden: int,
        classes: int,
        *,
        surrogate_slope: float,
        hard_forward: bool,
        dropout: float,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.feature_dropout = nn.Dropout(dropout)
        self.hidden = LinearSurrogateLIF(
            32 * 7 * 7,
            hidden,
            surrogate=fast_sigmoid_surrogate,
            surrogate_slope=surrogate_slope,
            hard_forward=hard_forward,
        )
        self.hidden_dropout = nn.Dropout(dropout)
        self.output = LinearSurrogateLIF(
            hidden,
            classes,
            surrogate=fast_sigmoid_surrogate,
            surrogate_slope=surrogate_slope,
            hard_forward=hard_forward,
        )

    def encode(self, image_series: torch.Tensor) -> torch.Tensor:
        spike_features = self.features(image_series.flatten(0, 1)).flatten(start_dim=1)
        spike_features = self.feature_dropout(spike_features)
        return spike_features.view(image_series.shape[0], image_series.shape[1], -1).contiguous()

    def forward(self, image_series: torch.Tensor) -> torch.Tensor:
        inputs = self.encode(image_series)
        hidden_spikes = self.hidden(inputs)
        hidden_spikes = self.hidden_dropout(hidden_spikes)
        output_spikes = self.output(hidden_spikes)
        return output_spikes.mean(dim=0)


def reset_synapse_fan_in(layer: LinearSurrogateLIF) -> None:
    fan_in = layer.synapse.weight.shape[0]
    bound = 1.0 / math.sqrt(fan_in)
    nn.init.uniform_(layer.synapse.weight, -bound, bound)
    if layer.synapse.bias is not None:
        nn.init.uniform_(layer.synapse.bias, -bound, bound)


def apply_synapse_init(model: ConvMNISTSNN, init: str) -> None:
    if init == "myelin":
        return
    if init == "fan_in":
        reset_synapse_fan_in(model.hidden)
        reset_synapse_fan_in(model.output)
        return
    msg = f"unsupported synapse init: {init}"
    raise ValueError(msg)


def make_image_series(
    images: torch.Tensor,
    timesteps: int,
    device: str,
    *,
    encoding: str,
) -> torch.Tensor:
    return encode_time_series(images.to(device=device), timesteps, encoding)


@torch.no_grad()
def compute_dynamics(
    model: ConvMNISTSNN,
    images: torch.Tensor,
    targets: torch.Tensor,
    *,
    timesteps: int,
    encoding: str,
) -> dict[str, float]:
    was_training = model.training
    model.eval()

    image_series = make_image_series(images, timesteps, str(images.device), encoding=encoding)
    inputs = model.encode(image_series)
    hidden_current = model.hidden.synapse(inputs)
    _hidden_state, hidden_spikes = model.hidden.unroll(hidden_current)
    output_current = model.output.synapse(hidden_spikes)
    _output_state, output_spikes = model.output.unroll(output_current)
    logits = output_spikes.mean(dim=0)
    probabilities = logits.softmax(dim=1)
    top2 = logits.topk(2, dim=1).values
    predictions = logits.argmax(dim=1)

    if was_training:
        model.train()

    input_delta_std = float((inputs[1:] - inputs[:-1]).std()) if timesteps > 1 else 0.0

    return {
        "dynamics/input_mean": float(inputs.mean()),
        "dynamics/input_std": float(inputs.std()),
        "dynamics/input_delta_std": input_delta_std,
        "dynamics/hidden_current_mean": float(hidden_current.mean()),
        "dynamics/hidden_current_std": float(hidden_current.std()),
        "dynamics/hidden_spike_rate": float(hidden_spikes.mean()),
        "dynamics/output_current_mean": float(output_current.mean()),
        "dynamics/output_current_std": float(output_current.std()),
        "dynamics/output_spike_rate": float(output_spikes.mean()),
        "dynamics/logit_std": float(logits.std()),
        "dynamics/logit_entropy": float(
            (-(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=1)).mean()
        ),
        "dynamics/logit_margin": float((top2[:, 0] - top2[:, 1]).mean()),
        "dynamics/batch_accuracy": float((predictions == targets).float().mean()),
    }


def print_dynamics(metrics: dict[str, float]) -> None:
    print(
        "| Dynamics | Input Std | Input Delta Std | Hidden Current Std | "
        "Hidden Spike Rate | Output Current Std | Output Spike Rate | "
        "Logit Std | Entropy | Margin | Batch Acc |",
        flush=True,
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|", flush=True)
    print(
        "| value | "
        f"{metrics['dynamics/input_std']:.6f} | "
        f"{metrics['dynamics/input_delta_std']:.6f} | "
        f"{metrics['dynamics/hidden_current_std']:.6f} | "
        f"{metrics['dynamics/hidden_spike_rate']:.6f} | "
        f"{metrics['dynamics/output_current_std']:.6f} | "
        f"{metrics['dynamics/output_spike_rate']:.6f} | "
        f"{metrics['dynamics/logit_std']:.6f} | "
        f"{metrics['dynamics/logit_entropy']:.6f} | "
        f"{metrics['dynamics/logit_margin']:.6f} | "
        f"{metrics['dynamics/batch_accuracy']:.6f} |",
        flush=True,
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    *,
    timesteps: int,
    device: str,
    encoding: str,
    max_batches: int | None,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for batch_index, (images, targets) in enumerate(loader, start=1):
        image_series = make_image_series(images, timesteps, device, encoding=encoding)
        targets = targets.to(device=device)
        logits = model(image_series)
        loss = loss_fn(logits, targets)
        total_loss += float(loss) * targets.numel()
        total_correct += int((logits.argmax(dim=1) == targets).sum())
        total_examples += targets.numel()
        if max_batches is not None and batch_index >= max_batches:
            break

    model.train()
    return total_loss / total_examples, total_correct / total_examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--timesteps", type=int, default=25)
    add_encoding_arg(parser)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    add_grad_clip_arg(parser, default=1.0)
    add_surrogate_args(parser)
    parser.add_argument("--seed", type=int, default=0)
    add_compile_policy_arg(parser)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--test-limit", type=int)
    parser.add_argument("--log-dynamics", action="store_true")
    parser.add_argument("--synapse-init", choices=("myelin", "fan_in"), default="fan_in")
    add_matmul_precision_arg(parser)
    add_wandb_args(parser)
    args = parser.parse_args()
    configure_matmul_precision(args.matmul_precision)
    compile_model = resolve_compile_policy(args.compile, args.device)

    torch.manual_seed(args.seed)
    transform = transforms.ToTensor()
    train_data = limited_dataset(
        datasets.MNIST(Path(args.data_dir), train=True, download=True, transform=transform),
        args.train_limit,
    )
    test_data = limited_dataset(
        datasets.MNIST(Path(args.data_dir), train=False, download=True, transform=transform),
        args.test_limit,
    )
    train_loader = DataLoader(train_data, batch_size=args.batch, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=args.batch, shuffle=False, num_workers=2)
    train_examples = len(cast(Sized, train_data))
    test_examples = len(cast(Sized, test_data))
    config = {
        "device": args.device,
        "compile": compile_model,
        "compile_policy": args.compile,
        "timesteps": args.timesteps,
        "encoding": args.encoding,
        "batch": args.batch,
        "hidden": args.hidden,
        "epochs": args.epochs,
        "lr": args.lr,
        "dropout": args.dropout,
        "label_smoothing": args.label_smoothing,
        "grad_clip": args.grad_clip,
        "matmul_precision": args.matmul_precision,
        "surrogate_slope": args.surrogate_slope,
        "hard_forward": not args.smooth_forward,
        "seed": args.seed,
        "train_examples": train_examples,
        "test_examples": test_examples,
        "model": "conv_mnist_snn",
        "log_dynamics": args.log_dynamics,
        "synapse_init": args.synapse_init,
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
        f"T:{args.timesteps},"
        f"encoding:{args.encoding},"
        f"batch:{args.batch},hidden:{args.hidden},epochs:{args.epochs},lr:{args.lr},"
        f"dropout:{args.dropout},label_smoothing:{args.label_smoothing},"
        f"grad_clip:{args.grad_clip},matmul_precision:{args.matmul_precision},"
        f"surrogate_slope:{args.surrogate_slope},hard_forward:{not args.smooth_forward},"
        f"synapse_init:{args.synapse_init},"
        f"train_examples:{train_examples},test_examples:{test_examples}",
        flush=True,
    )
    print()

    base_model = ConvMNISTSNN(
        hidden=args.hidden,
        classes=10,
        surrogate_slope=args.surrogate_slope,
        hard_forward=not args.smooth_forward,
        dropout=args.dropout,
    ).to(device=args.device)
    apply_synapse_init(base_model, args.synapse_init)
    print_model_summary(base_model)
    print()
    model: nn.Module = base_model
    model = compile_training_model(base_model, compile_model)

    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print("| Step | Epoch | Loss | Train Acc | Val Loss | Val Acc | Step ms |", flush=True)
    print("|---:|---:|---:|---:|---:|---:|---:|", flush=True)

    global_step = 0
    step_times = []
    reset_cuda_peak_memory(args.device)
    training_start = time.perf_counter()
    model.train()

    for epoch in range(1, args.epochs + 1):
        for images, targets in train_loader:
            global_step += 1
            image_series = make_image_series(
                images,
                args.timesteps,
                args.device,
                encoding=args.encoding,
            )
            targets = targets.to(device=args.device)

            synchronize_if_needed(args.device)
            step_start = time.perf_counter()
            optimizer.zero_grad()
            logits = model(image_series)
            loss = loss_fn(logits, targets)
            loss.backward()
            grad_norm = clip_gradients(model, args.grad_clip)
            optimizer.step()
            synchronize_if_needed(args.device)
            step_seconds = time.perf_counter() - step_start
            step_times.append(step_seconds)

            should_eval = global_step == 1 or global_step % args.eval_every == 0
            should_log = global_step == 1 or global_step % args.log_every == 0
            is_last_batch = epoch == args.epochs and global_step == len(train_loader) * args.epochs
            if should_log or should_eval or is_last_batch:
                train_acc = accuracy(logits.detach(), targets)
                wandb_metrics = {
                    "train/loss": float(loss.detach()),
                    "train/accuracy": train_acc,
                    "train/step_ms": step_seconds * 1000,
                }
                if grad_norm is not None:
                    wandb_metrics["train/grad_norm"] = grad_norm
                dynamics_metrics = None
                if args.log_dynamics:
                    dynamics_metrics = compute_dynamics(
                        base_model,
                        images.to(device=args.device),
                        targets,
                        timesteps=args.timesteps,
                        encoding=args.encoding,
                    )
                    wandb_metrics.update(dynamics_metrics)
                val_loss = ""
                val_acc = ""
                if should_eval or is_last_batch:
                    evaluated_loss, evaluated_acc = evaluate(
                        model,
                        test_loader,
                        loss_fn,
                        timesteps=args.timesteps,
                        device=args.device,
                        encoding=args.encoding,
                        max_batches=args.eval_batches,
                    )
                    val_loss = f"{evaluated_loss:.6f}"
                    val_acc = f"{evaluated_acc:.4f}"
                    wandb_metrics.update(
                        {"val/loss": evaluated_loss, "val/accuracy": evaluated_acc}
                    )
                print(
                    f"| {global_step} | {epoch} | {float(loss.detach()):.6f} | "
                    f"{train_acc:.4f} | {val_loss} | {val_acc} | "
                    f"{step_seconds * 1000:.3f} |",
                    flush=True,
                )
                if dynamics_metrics is not None:
                    print_dynamics(dynamics_metrics)
                log_wandb(wandb_run, wandb_metrics, step=global_step)

    synchronize_if_needed(args.device)
    total_seconds = time.perf_counter() - training_start
    final_loss, final_acc = evaluate(
        model,
        test_loader,
        loss_fn,
        timesteps=args.timesteps,
        device=args.device,
        encoding=args.encoding,
        max_batches=None,
    )

    print()
    print(f"final_test_loss={final_loss:.6f}", flush=True)
    print(f"final_test_accuracy={final_acc:.4f}", flush=True)
    print(f"total_training_seconds={total_seconds:.3f}", flush=True)
    print_cuda_peak_memory_summary(args.device)
    print_step_time_summary(step_times)
    finish_wandb(wandb_run)


if __name__ == "__main__":
    main()
