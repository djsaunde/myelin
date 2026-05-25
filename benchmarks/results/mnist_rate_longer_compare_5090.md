# MNIST Rate Longer Comparison

Runs the dense MNIST example and low-memory rate-readout MNIST example with
matched training knobs beyond the smallest smoke size.

## Command

```bash
uv run python -m spiker.benchmarks.mnist_compare \
  --variant dense --variant rate \
  --device cuda --timesteps 10 --batch 128 --hidden 128 \
  --epochs 2 --train-limit 4096 --test-limit 2048 \
  --eval-batches 4 --log-every 200 --eval-every 200 \
  --matmul-precision highest --grad-clip 0.1 --compile-spiker-only
```

## Environment

- `generated_utc`: `2026-05-24T12:03:25+00:00`
- `device`: `cuda`
- `gpu`: `NVIDIA GeForce RTX 5090`
- `timesteps`: `10`
- `encoding`: `poisson`
- `batch`: `128`
- `hidden`: `128`
- `epochs`: `2`
- `train_limit`: `4096`
- `test_limit`: `2048`
- `compile_spiker_only`: `True`
- `matmul_precision`: `highest`
- `grad_clip`: `0.1`

## Results

| Variant | Final Test Loss | Final Test Acc | Total s | Peak CUDA MB | Avg Step ms | Steady Step ms | Status |
|---|---:|---:|---:|---:|---:|---:|---:|
| dense | 1.606427 | 0.8286 | 2.917 | 45.3 | 28.257 | 1.252 | ok |
| rate | 1.608823 | 0.8330 | 4.613 | 18.7 | 58.211 | 1.106 | ok |

## Takeaway

At this longer smoke scale, the rate-readout path preserved training quality and
slightly exceeded dense accuracy while using 58.7% less peak CUDA memory. The
steady-state step time was also slightly faster after compile/warmup noise.
