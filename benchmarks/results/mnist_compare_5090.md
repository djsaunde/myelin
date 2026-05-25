# MNIST Example Comparison

Runs the repo MNIST examples with matched training knobs.

This is a short directional run, not a final accuracy benchmark:

```bash
uv run python -m spiker.benchmarks.mnist_compare \
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
  --eval-every 1000
```

## Environment

- `generated_utc`: `2026-05-24T09:13:04+00:00`
- `device`: `cuda`
- `gpu`: `NVIDIA GeForce RTX 5090`
- `timesteps`: `10`
- `encoding`: `poisson`
- `batch`: `128`
- `hidden`: `128`
- `epochs`: `1`
- `train_limit`: `1024`
- `test_limit`: `1024`
- `compile`: `False`

## Results

| Variant | Final Test Loss | Final Test Acc | Total s | Avg Step ms | Steady Step ms | Status |
|---|---:|---:|---:|---:|---:|---:|
| dense | 2.136469 | 0.6006 | 0.892 | 75.870 | 8.483 | ok |
| rate | 2.136011 | 0.6055 | 1.678 | 169.008 | 1.198 | ok |
| conv | 2.208766 | 0.3594 | 1.030 | 97.997 | 10.220 | ok |

## Takeaway

The rate-readout example has high average step time because its first step pays
Triton compilation cost, but its post-warmup steady-state step time is much
lower than the dense eager example in this short run. Accuracy is essentially
tied with dense on this small subset. The convolutional example is not tuned
yet and underperforms here despite adding useful image features.
