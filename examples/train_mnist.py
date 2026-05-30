"""Train a small surrogate SNN on MNIST."""

from __future__ import annotations

import argparse
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
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from myelin import LinearSurrogateLIF, fast_sigmoid_surrogate


class MNISTSNN(nn.Module):
    def __init__(
        self,
        features: int,
        hidden: int,
        classes: int,
        *,
        surrogate_slope: float,
        hard_forward: bool,
    ) -> None:
        super().__init__()
        self.hidden = LinearSurrogateLIF(
            features,
            hidden,
            surrogate=fast_sigmoid_surrogate,
            surrogate_slope=surrogate_slope,
            hard_forward=hard_forward,
        )
        self.output = LinearSurrogateLIF(
            hidden,
            classes,
            surrogate=fast_sigmoid_surrogate,
            surrogate_slope=surrogate_slope,
            hard_forward=hard_forward,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden_spikes = self.hidden(inputs)
        output_spikes = self.output(hidden_spikes)
        return output_spikes.mean(dim=0)


def synchronize_if_needed(device: str) -> None:
    torch_device = torch.device(device)
    if torch_device.type == "cuda":
        torch.cuda.synchronize(torch_device)


def make_time_inputs(
    images: torch.Tensor,
    timesteps: int,
    device: str,
    *,
    encoding: str,
) -> torch.Tensor:
    flat = images.flatten(start_dim=1).to(device=device)
    return encode_time_series(flat, timesteps, encoding)


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return float((logits.argmax(dim=1) == targets).float().mean())


def limited_dataset(
    dataset: torch.utils.data.Dataset, limit: int | None
) -> torch.utils.data.Dataset:
    if limit is None:
        return dataset
    return Subset(dataset, range(min(limit, len(cast(Sized, dataset)))))


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
        inputs = make_time_inputs(images, timesteps, device, encoding=encoding)
        targets = targets.to(device=device)
        logits = model(inputs)
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
    parser.add_argument("--timesteps", type=int, default=10)
    add_encoding_arg(parser)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-3)
    add_grad_clip_arg(parser)
    add_surrogate_args(parser)
    parser.add_argument("--seed", type=int, default=0)
    add_compile_policy_arg(parser)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--test-limit", type=int)
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
        "grad_clip": args.grad_clip,
        "matmul_precision": args.matmul_precision,
        "surrogate_slope": args.surrogate_slope,
        "hard_forward": not args.smooth_forward,
        "seed": args.seed,
        "train_examples": train_examples,
        "test_examples": test_examples,
        "model": "dense_mnist_snn",
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
        f"grad_clip:{args.grad_clip},matmul_precision:{args.matmul_precision},"
        f"surrogate_slope:{args.surrogate_slope},hard_forward:{not args.smooth_forward},"
        f"train_examples:{train_examples},test_examples:{test_examples}",
        flush=True,
    )
    print()

    model = MNISTSNN(
        features=28 * 28,
        hidden=args.hidden,
        classes=10,
        surrogate_slope=args.surrogate_slope,
        hard_forward=not args.smooth_forward,
    ).to(device=args.device)
    print_model_summary(model)
    print()
    model = compile_training_model(model, compile_model)

    loss_fn = nn.CrossEntropyLoss()
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
            inputs = make_time_inputs(
                images,
                args.timesteps,
                args.device,
                encoding=args.encoding,
            )
            targets = targets.to(device=args.device)

            synchronize_if_needed(args.device)
            step_start = time.perf_counter()
            optimizer.zero_grad()
            logits = model(inputs)
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
                    f"{train_acc:.4f} | "
                    f"{val_loss} | {val_acc} | {step_seconds * 1000:.3f} |",
                    flush=True,
                )
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
