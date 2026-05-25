# Modal Distributed Launchers

These launchers run the existing distributed examples on Modal without making
Modal a runtime dependency of `spiker`.

The default GPU target is `L4:2`, which is intentionally cheaper than H100/B200
and enough for smoke testing our DDP/FSDP2 wiring. Increase the GPU class later
only when a benchmark needs it.

The Modal smoke launcher defaults to `--compile off`. That keeps cloud smoke
runs focused on distributed correctness and avoids spending most of the run on
first-step compilation. Use the local examples directly when benchmarking the
compiled CUDA path.

## Setup

Install the optional Modal dependency and authenticate:

```bash
uv sync --extra cloud
uv run modal setup
```

For W&B logging, make `WANDB_API_KEY` available in the Modal container before
running with `--wandb`. The simplest version is to add a Modal Secret to the
launcher function decorator for your workspace. Run without `--wandb` for smoke
tests that do not need remote logging.

## DDP Smoke

```bash
uv run modal run modal/train_distributed.py --target ddp
```

## DDP Conv MNIST

```bash
uv run modal run modal/train_distributed.py \
  --target ddp-conv \
  --timesteps 25 \
  --hidden 256 \
  --epochs 4 \
  --train-limit 60000 \
  --test-limit 10000
```

## FSDP2 Smoke

```bash
uv run modal run modal/train_distributed.py --target fsdp2
```

## Larger Cheap Run

```bash
uv run modal run modal/train_distributed.py \
  --target ddp \
  --timesteps 50 \
  --hidden 256 \
  --epochs 2 \
  --train-limit 8192 \
  --test-limit 2048
```

The launcher shells out to `torch.distributed.run` inside a single Modal
container with two GPUs. Multi-node Modal clusters are a separate step because
the clustered API is currently beta and has stricter GPU-count constraints.
