"""Train the rate-readout MNIST SNN with ordinary PyTorch DDP."""

from __future__ import annotations

import argparse
import os
import time
from collections.abc import Sized
from pathlib import Path
from typing import Any, Literal, cast

import torch
import torch.distributed as dist
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
    finish_wandb,
    init_wandb,
    log_wandb,
    print_cuda_peak_memory_summary,
    print_model_summary,
    print_resident_memory_summary,
    print_step_time_summary,
    reset_cuda_peak_memory,
    resolve_compile_policy,
)
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
from train_mnist import accuracy, limited_dataset, make_time_inputs, synchronize_if_needed
from train_mnist_rate import resolve_backend

from spiker import RateReadoutClassifier, parse_checkpoint_size, resolve_checkpoint_size

RateBackend = Literal["auto", "torch", "triton", "triton_generated", "triton_compile"]


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def setup_distributed(requested_device: str) -> tuple[bool, int, int, int, str]:
    world_size = env_int("WORLD_SIZE", 1)
    rank = env_int("RANK", 0)
    local_rank = env_int("LOCAL_RANK", 0)
    torch_device = torch.device(requested_device)
    if torch_device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but CUDA is unavailable")
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
        backend = "nccl"
    else:
        device = requested_device
        backend = "gloo"

    distributed = world_size > 1
    if distributed:
        if torch.device(device).type == "cuda":
            dist.init_process_group(
                backend=backend,
                rank=rank,
                world_size=world_size,
                device_id=torch.device(device),
            )
        else:
            dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    return distributed, rank, local_rank, world_size, device


def cleanup_distributed(distributed: bool) -> None:
    if distributed:
        dist.destroy_process_group()


def is_rank0(rank: int) -> bool:
    return rank == 0


def reduced_sums(values: torch.Tensor, distributed: bool) -> torch.Tensor:
    if distributed:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return values


def distributed_barrier(distributed: bool, device: str) -> None:
    if not distributed:
        return
    torch_device = torch.device(device)
    if torch_device.type == "cuda":
        dist.barrier(device_ids=[torch_device.index or 0])
    else:
        dist.barrier()


def prepare_mnist_datasets(
    data_dir: str,
    *,
    transform: Any,
    train_limit: int | None,
    test_limit: int | None,
    distributed: bool,
    rank: int,
    device: str,
) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    path = Path(data_dir)
    if distributed and rank != 0:
        distributed_barrier(distributed, device)
    download = not distributed or rank == 0
    train_data = limited_dataset(
        datasets.MNIST(path, train=True, download=download, transform=transform),
        train_limit,
    )
    test_data = limited_dataset(
        datasets.MNIST(path, train=False, download=download, transform=transform),
        test_limit,
    )
    if distributed and rank == 0:
        distributed_barrier(distributed, device)
    return train_data, test_data


@torch.no_grad()
def evaluate_distributed(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    *,
    timesteps: int,
    device: str,
    encoding: str,
    max_batches: int | None,
    distributed: bool,
) -> tuple[float, float]:
    model.eval()
    totals = torch.zeros(3, dtype=torch.float64, device=device)

    for batch_index, (images, targets) in enumerate(loader, start=1):
        inputs = make_time_inputs(images, timesteps, device, encoding=encoding)
        targets = targets.to(device=device)
        logits = model(inputs)
        loss = loss_fn(logits, targets)
        totals[0] += loss.detach().to(torch.float64) * targets.numel()
        totals[1] += (logits.argmax(dim=1) == targets).sum().to(torch.float64)
        totals[2] += targets.numel()
        if max_batches is not None and batch_index >= max_batches:
            break

    reduced_sums(totals, distributed)
    model.train()
    return float(totals[0] / totals[2]), float(totals[1] / totals[2])


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
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    add_grad_clip_arg(parser)
    add_surrogate_args(parser)
    parser.add_argument(
        "--backend",
        choices=("auto", "torch", "triton", "triton_generated", "triton_compile"),
        default="auto",
    )
    parser.add_argument(
        "--checkpoint-size",
        type=parse_checkpoint_size,
        default="balanced",
    )
    parser.add_argument("--seed", type=int, default=0)
    add_compile_policy_arg(parser)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--test-limit", type=int)
    parser.add_argument("--num-workers", type=int, default=2)
    add_matmul_precision_arg(parser)
    add_wandb_args(parser)
    args = parser.parse_args()

    configure_matmul_precision(args.matmul_precision)
    distributed, rank, local_rank, world_size, device = setup_distributed(args.device)
    try:
        resolved_backend = resolve_backend(cast(RateBackend, args.backend), device)
        compile_model = resolve_compile_policy(args.compile, device)
        resolved_checkpoint_size = resolve_checkpoint_size(args.timesteps, args.checkpoint_size)
        torch.manual_seed(args.seed + rank)

        transform = transforms.ToTensor()
        train_data, test_data = prepare_mnist_datasets(
            args.data_dir,
            transform=transform,
            train_limit=args.train_limit,
            test_limit=args.test_limit,
            distributed=distributed,
            rank=rank,
            device=device,
        )
        train_sampler = (
            DistributedSampler(
                train_data,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                seed=args.seed,
            )
            if distributed
            else None
        )
        test_sampler = (
            DistributedSampler(
                test_data,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
                seed=args.seed,
            )
            if distributed
            else None
        )
        train_loader = DataLoader(
            train_data,
            batch_size=args.batch,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            num_workers=args.num_workers,
        )
        test_loader = DataLoader(
            test_data,
            batch_size=args.batch,
            shuffle=False,
            sampler=test_sampler,
            num_workers=args.num_workers,
        )
        train_examples = len(cast(Sized, train_data))
        test_examples = len(cast(Sized, test_data))
        config = {
            "distributed": "ddp",
            "rank": rank,
            "local_rank": local_rank,
            "world_size": world_size,
            "device": device,
            "compile": compile_model,
            "compile_policy": args.compile,
            "timesteps": args.timesteps,
            "encoding": args.encoding,
            "batch_per_rank": args.batch,
            "hidden": args.hidden,
            "epochs": args.epochs,
            "lr": args.lr,
            "dropout": args.dropout,
            "label_smoothing": args.label_smoothing,
            "grad_clip": args.grad_clip,
            "matmul_precision": args.matmul_precision,
            "surrogate_slope": args.surrogate_slope,
            "hard_forward": not args.smooth_forward,
            "backend": args.backend,
            "resolved_backend": resolved_backend,
            "checkpoint_size": args.checkpoint_size,
            "resolved_checkpoint_size": resolved_checkpoint_size,
            "seed": args.seed,
            "train_examples": train_examples,
            "test_examples": test_examples,
            "model": "rate_readout_mnist_snn",
        }
        wandb_run = init_wandb(
            enabled=args.wandb and is_rank0(rank),
            project=args.wandb_project,
            run_name=args.wandb_run_name,
            config=config,
        )

        if is_rank0(rank):
            print(
                "config="
                f"rank:{rank},local_rank:{local_rank},world_size:{world_size},"
                f"device:{device},compile:{compile_model},compile_policy:{args.compile},"
                f"T:{args.timesteps},encoding:{args.encoding},"
                f"batch_per_rank:{args.batch},hidden:{args.hidden},epochs:{args.epochs},"
                f"lr:{args.lr},dropout:{args.dropout},label_smoothing:{args.label_smoothing},"
                f"grad_clip:{args.grad_clip},matmul_precision:{args.matmul_precision},"
                f"surrogate_slope:{args.surrogate_slope},hard_forward:{not args.smooth_forward},"
                f"backend:{args.backend},resolved_backend:{resolved_backend},"
                f"checkpoint_size:{args.checkpoint_size},"
                f"resolved_checkpoint_size:{resolved_checkpoint_size},"
                f"train_examples:{train_examples},test_examples:{test_examples}",
                flush=True,
            )
            print()

        model: nn.Module = RateReadoutClassifier(
            in_features=28 * 28,
            hidden_features=args.hidden,
            out_features=10,
            surrogate_slope=args.surrogate_slope,
            hard_forward=not args.smooth_forward,
            backend=resolved_backend,
            checkpoint_size=resolved_checkpoint_size,
            dropout=args.dropout,
        ).to(device=device)
        if is_rank0(rank):
            print_model_summary(model)
            print()
            print_resident_memory_summary(model)
            print()

        model = compile_training_model(model, compile_model)
        if distributed:
            if torch.device(device).type == "cuda":
                model = DistributedDataParallel(model, device_ids=[local_rank])
            else:
                model = DistributedDataParallel(model)

        loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        if is_rank0(rank):
            print("| Step | Epoch | Loss | Train Acc | Val Loss | Val Acc | Step ms |", flush=True)
            print("|---:|---:|---:|---:|---:|---:|---:|", flush=True)

        global_step = 0
        step_times: list[float] = []
        reset_cuda_peak_memory(device)
        training_start = time.perf_counter()
        model.train()

        for epoch in range(1, args.epochs + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            for images, targets in train_loader:
                global_step += 1
                inputs = make_time_inputs(images, args.timesteps, device, encoding=args.encoding)
                targets = targets.to(device=device)

                synchronize_if_needed(device)
                step_start = time.perf_counter()
                optimizer.zero_grad()
                logits = model(inputs)
                loss = loss_fn(logits, targets)
                loss.backward()
                clip_gradients(model, args.grad_clip)
                optimizer.step()
                synchronize_if_needed(device)
                step_seconds = time.perf_counter() - step_start
                step_times.append(step_seconds)

                should_eval = global_step == 1 or global_step % args.eval_every == 0
                should_log = global_step == 1 or global_step % args.log_every == 0
                is_last_batch = (
                    epoch == args.epochs and global_step == len(train_loader) * args.epochs
                )
                evaluated_loss = None
                evaluated_acc = None
                if should_eval or is_last_batch:
                    evaluated_loss, evaluated_acc = evaluate_distributed(
                        model,
                        test_loader,
                        loss_fn,
                        timesteps=args.timesteps,
                        device=device,
                        encoding=args.encoding,
                        max_batches=args.eval_batches,
                        distributed=distributed,
                    )
                if is_rank0(rank) and (should_log or should_eval or is_last_batch):
                    train_acc = accuracy(logits.detach(), targets)
                    val_loss = ""
                    val_acc = ""
                    if evaluated_loss is not None and evaluated_acc is not None:
                        val_loss = f"{evaluated_loss:.6f}"
                        val_acc = f"{evaluated_acc:.4f}"
                    wandb_metrics = {
                        "train/loss": float(loss.detach()),
                        "train/accuracy": train_acc,
                        "train/step_ms": step_seconds * 1000,
                    }
                    if evaluated_loss is not None and evaluated_acc is not None:
                        wandb_metrics.update(
                            {"val/loss": evaluated_loss, "val/accuracy": evaluated_acc}
                        )
                    log_wandb(wandb_run, wandb_metrics, step=global_step)
                    print(
                        f"| {global_step} | {epoch} | {float(loss.detach()):.6f} | "
                        f"{train_acc:.4f} | {val_loss} | {val_acc} | "
                        f"{step_seconds * 1000:.3f} |",
                        flush=True,
                    )

        synchronize_if_needed(device)
        total_seconds = time.perf_counter() - training_start
        final_loss, final_acc = evaluate_distributed(
            model,
            test_loader,
            loss_fn,
            timesteps=args.timesteps,
            device=device,
            encoding=args.encoding,
            max_batches=None,
            distributed=distributed,
        )

        if is_rank0(rank):
            print()
            print(f"final_test_loss={final_loss:.6f}", flush=True)
            print(f"final_test_accuracy={final_acc:.4f}", flush=True)
            print(f"total_training_seconds={total_seconds:.3f}", flush=True)
            print_cuda_peak_memory_summary(device)
            print_resident_memory_summary(model, optimizer)
            print_step_time_summary(step_times)
            finish_wandb(wandb_run)
    finally:
        cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
