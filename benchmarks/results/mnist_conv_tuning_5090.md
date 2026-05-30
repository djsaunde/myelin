# MNIST Conv Tuning Notes

Short directional runs on RTX 5090. These use only 1024 training examples for
one epoch, so they are tuning signals rather than final accuracy numbers.

## Baseline Conv Dynamics

```bash
uv run python examples/train_mnist_conv.py \
  --device cuda \
  --timesteps 10 \
  --encoding poisson \
  --batch 128 \
  --hidden 128 \
  --epochs 1 \
  --train-limit 1024 \
  --test-limit 1024 \
  --eval-batches 4 \
  --log-every 1000 \
  --eval-every 1000 \
  --log-dynamics \
  --synapse-init myelin
```

| Setting | Final Test Loss | Final Test Acc | Hidden Spike Rate | Output Spike Rate | Logit Std |
|---|---:|---:|---:|---:|---:|
| `synapse_init=myelin`, `grad_clip=0.1` | 2.211106 | 0.3711 | 0.310461 | 0.051875 | 0.087693 |

The initial output layer was silent at step 1, and the final output spike rate
was still low. This pointed to readout/current scaling rather than the
convolutional feature extractor itself.

## Init And Clip Sweep

| Setting | Final Test Loss | Final Test Acc | Notes |
|---|---:|---:|---|
| `synapse_init=fan_in`, `grad_clip=0.1`, `lr=0.003` | 2.039673 | 0.5039 | Output layer spikes from step 1 |
| `synapse_init=fan_in`, `grad_clip=1.0`, `lr=0.003` | 2.044660 | 0.5869 | Best direct single-script run |
| `synapse_init=fan_in`, `grad_clip=0.1`, `lr=0.01` | 1.946561 | 0.5059 | Lower loss, no accuracy win |
| `synapse_init=fan_in`, `grad_clip=1.0`, `lr=0.01` | 1.899308 | 0.5762 | Did not beat lower LR |
| `synapse_init=fan_in`, `grad_clip=0.1`, `slope=10` | 2.222672 | 0.2686 | Higher slope hurt here |

## Tuned Conv vs snnTorch Conv

```bash
uv run python -m myelin.benchmarks.mnist_compare \
  --device cuda \
  --variant conv \
  --variant snntorch_conv \
  --timesteps 10 \
  --encoding poisson \
  --batch 128 \
  --hidden 128 \
  --epochs 1 \
  --train-limit 1024 \
  --test-limit 1024 \
  --eval-batches 4 \
  --log-every 1000 \
  --eval-every 1000 \
  --grad-clip 1.0 \
  --conv-synapse-init fan_in
```

| Variant | Final Test Loss | Final Test Acc | Total s | Avg Step ms | Steady Step ms | Status |
|---|---:|---:|---:|---:|---:|---:|
| conv | 2.044464 | 0.5703 | 1.067 | 102.775 | 11.045 | ok |
| snntorch_conv | 2.141364 | 0.5566 | 1.032 | 96.486 | 17.152 | ok |

## Compiled Tuned Conv vs snnTorch Conv

After moving Poisson image-series encoding outside `ConvMNISTSNN.forward`, the
compiled conv module no longer traces random encoding work or an encoding
string through the model forward.

```bash
uv run python -m myelin.benchmarks.mnist_compare \
  --device cuda \
  --compile-myelin-only \
  --variant conv \
  --variant snntorch_conv \
  --timesteps 10 \
  --encoding poisson \
  --batch 128 \
  --hidden 128 \
  --epochs 1 \
  --train-limit 1024 \
  --test-limit 1024 \
  --eval-batches 4 \
  --log-every 1000 \
  --eval-every 1000 \
  --grad-clip 1.0 \
  --conv-synapse-init fan_in
```

| Variant | Final Test Loss | Final Test Acc | Total s | Avg Step ms | Steady Step ms | Status |
|---|---:|---:|---:|---:|---:|---:|
| conv | 2.041711 | 0.5908 | 2.728 | 245.530 | 1.804 | ok |
| snntorch_conv | 2.141364 | 0.5566 | 1.124 | 106.259 | 18.807 | ok |

## Matmul Precision Check

SNN training can be sensitive to small matmul differences because threshold
crossings turn those differences into spike/no-spike decisions. On the same
short compiled conv setup, `--matmul-precision high` removed PyTorch's TF32
warning and reduced steady-state step time, but it also hurt this one-epoch
accuracy check.

| Matmul Precision | Final Test Loss | Final Test Acc | Steady Step ms | Notes |
|---|---:|---:|---:|---|
| `highest` | 2.044058 | 0.5791 | 22.374 | Stricter PyTorch default; emits TF32 warning |
| `high` | 2.015691 | 0.5264 | 1.879 | Faster; lower short-run accuracy here |

## Takeaway

The conv example was brittle primarily because the SNN synapse layers used the
small default myelin initialization. Fan-in initialization gives the readout
enough current to spike early. Less aggressive clipping (`1.0`) helps this
conv recipe, while higher surrogate slope hurts. The compiled conv path now
matches the tuned eager accuracy after moving stochastic Poisson encoding
outside the compiled model forward. Training examples default to
`--matmul-precision highest`; use `high` explicitly for speed-focused runs.

## Larger Tuned Conv

```bash
uv run python examples/train_mnist_conv.py \
  --device cuda \
  --compile \
  --timesteps 25 \
  --hidden 256 \
  --epochs 4 \
  --train-limit 10000 \
  --test-limit 4096 \
  --eval-batches 8 \
  --log-every 200 \
  --eval-every 1000 \
  --grad-clip 1.0 \
  --synapse-init fan_in
```

| Final Test Loss | Final Test Acc | Total s | Peak CUDA MB | Avg Step ms | Steady Step ms |
|---:|---:|---:|---:|---:|---:|
| 1.497434 | 0.9578 | 8.090 | 900.3 | 18.860 | 10.108 |

This is now the example's default quality recipe: `T=25`, `hidden=256`,
`epochs=4`, `grad_clip=1.0`, fan-in synapse initialization, and strict
`matmul_precision=highest`. Use `--train-limit`/`--test-limit` for bounded
local checks, or omit them for the full MNIST split.

## Dropout And Label Smoothing

```bash
uv run python examples/train_mnist_conv.py \
  --device cuda \
  --compile \
  --timesteps 25 \
  --hidden 256 \
  --epochs 4 \
  --train-limit 10000 \
  --test-limit 4096 \
  --eval-batches 8 \
  --log-every 200 \
  --eval-every 1000 \
  --grad-clip 1.0 \
  --synapse-init fan_in \
  --dropout 0.1 \
  --label-smoothing 0.05
```

| Final Test Loss | Final Test Acc | Total s | Peak CUDA MB | Avg Step ms | Steady Step ms |
|---:|---:|---:|---:|---:|---:|
| 1.534343 | 0.9646 | 51.206 | 899.4 | 150.972 | 86.496 |

Dropout plus label smoothing improved bounded test accuracy from 95.78% to
96.46%, but the single run had higher wall time because compile/runtime caching
noise dominated. Treat this as a quality recipe, not a speed recipe.
