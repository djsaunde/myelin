# MNIST spiker vs snnTorch Comparison

Runs the repo MNIST examples and comparable snnTorch examples with matched
training knobs.

These are short directional runs, not final accuracy benchmarks.

## Eager spiker vs Eager snnTorch

```bash
uv run python -m spiker.benchmarks.mnist_compare \
  --device cuda \
  --variant dense \
  --variant rate \
  --variant conv \
  --variant snntorch_dense \
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
  --eval-every 1000
```

- `generated_utc`: `2026-05-24T09:20:48+00:00`
- `device`: `cuda`
- `gpu`: `NVIDIA GeForce RTX 5090`
- `timesteps`: `10`
- `encoding`: `poisson`
- `batch`: `128`
- `hidden`: `128`
- `epochs`: `1`
- `train_limit`: `1024`
- `test_limit`: `1024`
- `snntorch_beta`: `0.95`

| Variant | Final Test Loss | Final Test Acc | Total s | Avg Step ms | Steady Step ms | Status |
|---|---:|---:|---:|---:|---:|---:|
| dense | 2.136469 | 0.6006 | 0.893 | 76.325 | 8.391 | ok |
| rate | 2.136011 | 0.6055 | 0.799 | 65.154 | 1.225 | ok |
| conv | 2.208766 | 0.3594 | 0.978 | 91.578 | 9.824 | ok |
| snntorch_dense | 1.966508 | 0.6240 | 0.951 | 82.073 | 15.711 | ok |
| snntorch_conv | 2.158583 | 0.5312 | 1.115 | 106.587 | 17.310 | ok |

## Compiled spiker vs Eager snnTorch

```bash
uv run python -m spiker.benchmarks.mnist_compare \
  --device cuda \
  --compile-spiker-only \
  --variant dense \
  --variant rate \
  --variant conv \
  --variant snntorch_dense \
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
  --eval-every 1000
```

- `generated_utc`: `2026-05-24T09:21:49+00:00`
- `device`: `cuda`
- `gpu`: `NVIDIA GeForce RTX 5090`
- `timesteps`: `10`
- `encoding`: `poisson`
- `batch`: `128`
- `hidden`: `128`
- `epochs`: `1`
- `train_limit`: `1024`
- `test_limit`: `1024`
- `snntorch_beta`: `0.95`

| Variant | Final Test Loss | Final Test Acc | Total s | Avg Step ms | Steady Step ms | Status |
|---|---:|---:|---:|---:|---:|---:|
| dense | 2.136557 | 0.5967 | 2.913 | 226.310 | 1.212 | ok |
| rate | 2.136011 | 0.6055 | 2.392 | 238.244 | 1.207 | ok |
| conv | 2.198888 | 0.3721 | 3.143 | 231.662 | 1.958 | ok |
| snntorch_dense | 1.966508 | 0.6240 | 0.969 | 83.381 | 15.692 | ok |
| snntorch_conv | 2.158583 | 0.5312 | 1.029 | 95.671 | 17.151 | ok |

## Takeaway

For this short subset, snnTorch has better early accuracy than the current
spiker convolutional example, so our conv recipe still needs tuning. The
steady-state timing story is clearer: compiled spiker dense/rate steps are
about 13x faster than snnTorch dense eager, and compiled spiker conv is about
9x faster than snnTorch conv eager. First-step compile overhead dominates total
seconds in the tiny compiled run.
