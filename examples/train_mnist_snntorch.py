"""Train comparable snnTorch SNNs on MNIST."""

from __future__ import annotations

import argparse
import time
from collections.abc import Sized
from pathlib import Path
from typing import cast

import snntorch as snn
import torch
from example_utils import (
    add_encoding_arg,
    add_grad_clip_arg,
    add_matmul_precision_arg,
    add_surrogate_args,
    add_wandb_args,
    clip_gradients,
    configure_matmul_precision,
    encode_time_series,
    finish_wandb,
    init_wandb,
    log_wandb,
    print_cuda_peak_memory_summary,
    print_model_summary,
    print_step_time_summary,
    reset_cuda_peak_memory,
)
from snntorch import surrogate
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def synchronize_if_needed(device: str) -> None:
    torch_device = torch.device(device)
    if torch_device.type == "cuda":
        torch.cuda.synchronize(torch_device)


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return float((logits.argmax(dim=1) == targets).float().mean())


def limited_dataset(
    dataset: torch.utils.data.Dataset, limit: int | None
) -> torch.utils.data.Dataset:
    if limit is None:
        return dataset
    return Subset(dataset, range(min(limit, len(cast(Sized, dataset)))))


def make_time_inputs(
    images: torch.Tensor,
    timesteps: int,
    device: str,
    *,
    encoding: str,
) -> torch.Tensor:
    flat = images.flatten(start_dim=1).to(device=device)
    return encode_time_series(flat, timesteps, encoding)


class SnnTorchDenseMNIST(nn.Module):
    def __init__(
        self,
        features: int,
        hidden: int,
        classes: int,
        *,
        beta: float,
        surrogate_slope: float,
    ) -> None:
        super().__init__()
        spike_grad = surrogate.fast_sigmoid(slope=int(surrogate_slope))
        self.hidden_synapse = nn.Linear(features, hidden)
        self.hidden_lif = snn.Leaky(
            beta=beta,
            spike_grad=spike_grad,
            reset_mechanism="zero",
            reset_delay=False,
        )
        self.output_synapse = nn.Linear(hidden, classes)
        self.output_lif = snn.Leaky(
            beta=beta,
            spike_grad=spike_grad,
            reset_mechanism="zero",
            reset_delay=False,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        mem_hidden = torch.zeros(
            inputs.shape[1],
            self.hidden_synapse.out_features,
            dtype=inputs.dtype,
            device=inputs.device,
        )
        mem_output = torch.zeros(
            inputs.shape[1],
            self.output_synapse.out_features,
            dtype=inputs.dtype,
            device=inputs.device,
        )

        output_spikes = []
        for input_current in inputs.unbind(dim=0):
            hidden_current = self.hidden_synapse(input_current)
            hidden_spike, mem_hidden = self.hidden_lif(hidden_current, mem_hidden)
            output_current = self.output_synapse(hidden_spike)
            output_spike, mem_output = self.output_lif(output_current, mem_output)
            output_spikes.append(output_spike)
        return torch.stack(output_spikes, dim=0).mean(dim=0)


class SnnTorchConvMNIST(nn.Module):
    def __init__(
        self,
        hidden: int,
        classes: int,
        *,
        beta: float,
        surrogate_slope: float,
    ) -> None:
        super().__init__()
        spike_grad = surrogate.fast_sigmoid(slope=int(surrogate_slope))
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.hidden_synapse = nn.Linear(32 * 7 * 7, hidden)
        self.hidden_lif = snn.Leaky(
            beta=beta,
            spike_grad=spike_grad,
            reset_mechanism="zero",
            reset_delay=False,
        )
        self.output_synapse = nn.Linear(hidden, classes)
        self.output_lif = snn.Leaky(
            beta=beta,
            spike_grad=spike_grad,
            reset_mechanism="zero",
            reset_delay=False,
        )

    def encode(self, images: torch.Tensor, timesteps: int, encoding: str) -> torch.Tensor:
        if encoding == "repeat":
            conv_features = self.features(images).flatten(start_dim=1)
            return encode_time_series(conv_features, timesteps, encoding)
        image_series = encode_time_series(images, timesteps, encoding)
        spike_features = self.features(image_series.flatten(0, 1)).flatten(start_dim=1)
        return spike_features.view(timesteps, images.shape[0], -1).contiguous()

    def forward(self, images: torch.Tensor, timesteps: int, encoding: str) -> torch.Tensor:
        inputs = self.encode(images, timesteps, encoding)
        mem_hidden = torch.zeros(
            inputs.shape[1],
            self.hidden_synapse.out_features,
            dtype=inputs.dtype,
            device=inputs.device,
        )
        mem_output = torch.zeros(
            inputs.shape[1],
            self.output_synapse.out_features,
            dtype=inputs.dtype,
            device=inputs.device,
        )

        output_spikes = []
        for input_current in inputs.unbind(dim=0):
            hidden_current = self.hidden_synapse(input_current)
            hidden_spike, mem_hidden = self.hidden_lif(hidden_current, mem_hidden)
            output_current = self.output_synapse(hidden_spike)
            output_spike, mem_output = self.output_lif(output_current, mem_output)
            output_spikes.append(output_spike)
        return torch.stack(output_spikes, dim=0).mean(dim=0)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    *,
    model_kind: str,
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
        targets = targets.to(device=device)
        if model_kind == "dense":
            inputs = make_time_inputs(images, timesteps, device, encoding=encoding)
            logits = model(inputs)
        else:
            logits = model(images.to(device=device), timesteps, encoding)
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
    parser.add_argument("--model", choices=("dense", "conv"), default="conv")
    parser.add_argument("--timesteps", type=int, default=10)
    add_encoding_arg(parser)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-3)
    add_grad_clip_arg(parser)
    add_surrogate_args(parser)
    parser.add_argument("--beta", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--test-limit", type=int)
    add_matmul_precision_arg(parser)
    add_wandb_args(parser)
    args = parser.parse_args()
    configure_matmul_precision(args.matmul_precision)

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
        "library": "snntorch",
        "model": args.model,
        "device": args.device,
        "compile": args.compile,
        "timesteps": args.timesteps,
        "encoding": args.encoding,
        "batch": args.batch,
        "hidden": args.hidden,
        "epochs": args.epochs,
        "lr": args.lr,
        "grad_clip": args.grad_clip,
        "matmul_precision": args.matmul_precision,
        "surrogate_slope": args.surrogate_slope,
        "beta": args.beta,
        "seed": args.seed,
        "train_examples": train_examples,
        "test_examples": test_examples,
    }
    wandb_run = init_wandb(
        enabled=args.wandb,
        project=args.wandb_project,
        run_name=args.wandb_run_name,
        config=config,
    )

    print(
        "config="
        f"library:snntorch,model:{args.model},device:{args.device},"
        f"compile:{args.compile},T:{args.timesteps},encoding:{args.encoding},"
        f"batch:{args.batch},hidden:{args.hidden},epochs:{args.epochs},lr:{args.lr},"
        f"grad_clip:{args.grad_clip},matmul_precision:{args.matmul_precision},"
        f"surrogate_slope:{args.surrogate_slope},"
        f"beta:{args.beta},train_examples:{train_examples},test_examples:{test_examples}",
        flush=True,
    )
    print()

    if args.model == "dense":
        model = SnnTorchDenseMNIST(
            features=28 * 28,
            hidden=args.hidden,
            classes=10,
            beta=args.beta,
            surrogate_slope=args.surrogate_slope,
        ).to(device=args.device)
    else:
        model = SnnTorchConvMNIST(
            hidden=args.hidden,
            classes=10,
            beta=args.beta,
            surrogate_slope=args.surrogate_slope,
        ).to(device=args.device)
    print_model_summary(model)
    print()
    if args.compile:
        model = cast(nn.Module, torch.compile(model, mode="reduce-overhead", fullgraph=True))

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
            targets = targets.to(device=args.device)

            if args.model == "dense":
                model_inputs = make_time_inputs(
                    images,
                    args.timesteps,
                    args.device,
                    encoding=args.encoding,
                )
                forward_args = (model_inputs,)
            else:
                forward_args = (images.to(device=args.device), args.timesteps, args.encoding)

            synchronize_if_needed(args.device)
            step_start = time.perf_counter()
            optimizer.zero_grad()
            try:
                logits = model(*forward_args)
            except Exception as exc:
                if args.compile:
                    print(
                        f"compile_failed={type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    finish_wandb(wandb_run)
                    return
                raise
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
                        model_kind=args.model,
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
                log_wandb(wandb_run, wandb_metrics, step=global_step)

    synchronize_if_needed(args.device)
    total_seconds = time.perf_counter() - training_start
    final_loss, final_acc = evaluate(
        model,
        test_loader,
        loss_fn,
        model_kind=args.model,
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
