"""Modal launchers for distributed myelin training examples."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Literal

import modal

APP_NAME = "myelin-distributed"
REMOTE_ROOT = "/root/myelin"
DEFAULT_GPU = "L4:2"


@dataclass(frozen=True)
class TrainConfig:
    script: str
    nproc_per_node: int = 2
    timesteps: int = 25
    batch: int = 128
    hidden: int = 128
    epochs: int = 1
    lr: float = 3e-3
    train_limit: int = 4096
    test_limit: int = 1024
    eval_batches: int = 4
    log_every: int = 100
    eval_every: int = 250
    backend: str | None = "auto"
    compile_policy: str = "off"
    matmul_precision: str = "high"
    wandb: bool = False
    wandb_project: str = "myelin"
    wandb_run_name: str | None = None


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy>=1.24.4",
        "torch",
        "torchvision>=0.27.0",
        "triton",
        "wandb",
    )
    .add_local_dir(
        ".",
        remote_path=REMOTE_ROOT,
        ignore=[
            ".git",
            ".venv",
            ".pytest_cache",
            ".ruff_cache",
            "data",
            "dist",
            "wandb",
        ],
    )
)

app = modal.App(APP_NAME, image=image)


def _base_command(config: TrainConfig) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node",
        str(config.nproc_per_node),
        f"{REMOTE_ROOT}/{config.script}",
        "--device",
        "cuda",
        "--timesteps",
        str(config.timesteps),
        "--batch",
        str(config.batch),
        "--hidden",
        str(config.hidden),
        "--epochs",
        str(config.epochs),
        "--lr",
        str(config.lr),
        "--train-limit",
        str(config.train_limit),
        "--test-limit",
        str(config.test_limit),
        "--eval-batches",
        str(config.eval_batches),
        "--log-every",
        str(config.log_every),
        "--eval-every",
        str(config.eval_every),
        "--compile",
        config.compile_policy,
        "--matmul-precision",
        config.matmul_precision,
    ]
    if config.backend is not None:
        command.extend(["--backend", config.backend])
    if config.wandb:
        command.extend(["--wandb", "--wandb-project", config.wandb_project])
        if config.wandb_run_name is not None:
            command.extend(["--wandb-run-name", config.wandb_run_name])
    return command


def _run_training(config: TrainConfig) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{REMOTE_ROOT}/src:{REMOTE_ROOT}/examples"
    subprocess.run(
        _base_command(config),
        cwd=REMOTE_ROOT,
        env=env,
        check=True,
    )


@app.function(gpu=DEFAULT_GPU, timeout=60 * 60)
def run_ddp_smoke(
    *,
    timesteps: int = 25,
    batch: int = 128,
    hidden: int = 128,
    epochs: int = 1,
    train_limit: int = 4096,
    test_limit: int = 1024,
    wandb: bool = False,
    wandb_project: str = "myelin",
) -> None:
    """Run the DDP rate-readout example on a cheap single-node multi-GPU Modal box."""

    _run_training(
        TrainConfig(
            script="examples/train_mnist_rate_ddp.py",
            timesteps=timesteps,
            batch=batch,
            hidden=hidden,
            epochs=epochs,
            train_limit=train_limit,
            test_limit=test_limit,
            wandb=wandb,
            wandb_project=wandb_project,
            wandb_run_name="modal-ddp-smoke" if wandb else None,
        )
    )


@app.function(gpu=DEFAULT_GPU, timeout=60 * 60)
def run_ddp_conv(
    *,
    timesteps: int = 25,
    batch: int = 128,
    hidden: int = 256,
    epochs: int = 4,
    train_limit: int = 60_000,
    test_limit: int = 10_000,
    wandb: bool = False,
    wandb_project: str = "myelin",
) -> None:
    """Run the DDP convolutional MNIST example on a cheap single-node multi-GPU box."""

    _run_training(
        TrainConfig(
            script="examples/train_mnist_conv_ddp.py",
            timesteps=timesteps,
            batch=batch,
            hidden=hidden,
            epochs=epochs,
            train_limit=train_limit,
            test_limit=test_limit,
            eval_batches=20,
            log_every=100,
            eval_every=250,
            backend=None,
            wandb=wandb,
            wandb_project=wandb_project,
            wandb_run_name="modal-ddp-conv" if wandb else None,
        )
    )


@app.function(gpu=DEFAULT_GPU, timeout=60 * 60)
def run_fsdp2_smoke(
    *,
    timesteps: int = 25,
    batch: int = 128,
    hidden: int = 128,
    epochs: int = 1,
    train_limit: int = 4096,
    test_limit: int = 1024,
    wandb: bool = False,
    wandb_project: str = "myelin",
) -> None:
    """Run the FSDP2 rate-readout example on a cheap single-node multi-GPU Modal box."""

    _run_training(
        TrainConfig(
            script="examples/train_mnist_rate_fsdp2.py",
            timesteps=timesteps,
            batch=batch,
            hidden=hidden,
            epochs=epochs,
            train_limit=train_limit,
            test_limit=test_limit,
            wandb=wandb,
            wandb_project=wandb_project,
            wandb_run_name="modal-fsdp2-smoke" if wandb else None,
        )
    )


@app.local_entrypoint()
def main(
    target: Literal["ddp", "ddp-conv", "fsdp2"] = "ddp",
    timesteps: int = 25,
    batch: int = 128,
    hidden: int = 128,
    epochs: int = 1,
    train_limit: int = 4096,
    test_limit: int = 1024,
    wandb: bool = False,
    wandb_project: str = "myelin",
) -> None:
    """Launch a cheap Modal smoke run.

    Use ``target=ddp`` for rate-readout DDP, ``target=ddp-conv`` for
    convolutional DDP, and ``target=fsdp2`` for FSDP2.
    """

    kwargs = {
        "timesteps": timesteps,
        "batch": batch,
        "hidden": hidden,
        "epochs": epochs,
        "train_limit": train_limit,
        "test_limit": test_limit,
        "wandb": wandb,
        "wandb_project": wandb_project,
    }
    if target == "ddp":
        run_ddp_smoke.remote(**kwargs)
        return
    if target == "ddp-conv":
        run_ddp_conv.remote(**kwargs)
        return
    if target == "fsdp2":
        run_fsdp2_smoke.remote(**kwargs)
        return
    raise ValueError("target must be 'ddp', 'ddp-conv', or 'fsdp2'")
