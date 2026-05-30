# Compiled MNIST Example Comparison

Runs the repo MNIST examples with matched training knobs and `--compile`.

This is a short directional run, not a final accuracy benchmark:

```bash
uv run python -m myelin.benchmarks.mnist_compare \
  --device cuda \
  --compile \
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

- `generated_utc`: `2026-05-24T09:15:29+00:00` and `2026-05-24T09:15:49+00:00`
- `device`: `cuda`
- `gpu`: `NVIDIA GeForce RTX 5090`
- `timesteps`: `10`
- `encoding`: `poisson`
- `batch`: `128`
- `hidden`: `128`
- `epochs`: `1`
- `train_limit`: `1024`
- `test_limit`: `1024`
- `compile`: `True`

## Results

| Variant | Final Test Loss | Final Test Acc | Total s | Avg Step ms | Steady Step ms | Status |
|---|---:|---:|---:|---:|---:|---:|
| dense | 2.136557 | 0.5967 | 2.853 | 218.455 | 1.263 | ok |
| rate | 2.136011 | 0.6055 | 2.441 | 243.995 | 1.138 | ok |
| conv | 2.198888 | 0.3721 | 11.074 | 881.208 | 1.785 | ok |

## Takeaway

`torch.compile` is compatible with all three example models after resolving
`backend="auto"` outside the rate-readout model forward. Compile overhead
dominates this tiny run, but steady-state step time is close between dense and
rate, with rate slightly faster here. Conv still underperforms on accuracy in
this short, untuned setup.
