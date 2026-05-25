# MNIST Example Comparison

Runs the repo MNIST examples with matched training knobs.

## Environment

- `generated_utc`: `2026-05-24T17:16:58+00:00`
- `device`: `cuda`
- `timesteps`: `10`
- `encoding`: `poisson`
- `batch`: `128`
- `hidden`: `128`
- `epochs`: `2`
- `grad_clip`: `0.1`
- `rate_checkpoint_size`: `balanced`
- `train_limit`: `4096`
- `test_limit`: `2048`
- `compile`: `False`
- `compile_spiker_only`: `True`
- `conv_synapse_init`: `None`
- `matmul_precision`: `highest`
- `snntorch_beta`: `0.95`

## Results

| Variant | Final Test Loss | Final Test Acc | Total s | Peak CUDA MB | Avg Step ms | Steady Step ms | Compiled | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| rate | 1.607841 | 0.8242 | 5.054 | 18.7 | 62.752 | 1.607 | True | ok |
| rate_triton_compile | 1.609889 | 0.8267 | 4.368 | 18.7 | 51.912 | 1.925 | True | ok |
