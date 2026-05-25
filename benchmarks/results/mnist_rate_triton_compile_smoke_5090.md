# MNIST Example Comparison

Runs the repo MNIST examples with matched training knobs.

## Environment

- `generated_utc`: `2026-05-24T17:07:34+00:00`
- `device`: `cuda`
- `timesteps`: `10`
- `encoding`: `poisson`
- `batch`: `128`
- `hidden`: `128`
- `epochs`: `1`
- `grad_clip`: `None`
- `rate_checkpoint_size`: `balanced`
- `train_limit`: `1024`
- `test_limit`: `1024`
- `compile`: `False`
- `compile_spiker_only`: `True`
- `conv_synapse_init`: `None`
- `matmul_precision`: `high`
- `snntorch_beta`: `0.95`

## Results

| Variant | Final Test Loss | Final Test Acc | Total s | Peak CUDA MB | Avg Step ms | Steady Step ms | Compiled | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| rate | 2.136011 | 0.6055 | 3.604 | 18.7 | 374.912 | 2.331 | True | ok |
| rate_triton_compile | 2.136011 | 0.6055 | 2.990 | 18.7 | 276.441 | 1.566 | True | ok |
