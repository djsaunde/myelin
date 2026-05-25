# MNIST Example Comparison

Runs the repo MNIST examples with matched training knobs.

## Environment

- `generated_utc`: `2026-05-24T13:48:01+00:00`
- `device`: `cuda`
- `timesteps`: `10`
- `encoding`: `poisson`
- `batch`: `128`
- `hidden`: `128`
- `epochs`: `2`
- `grad_clip`: `0.1`
- `rate_checkpoint_size`: `memory`
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
| dense | 1.606427 | 0.8286 | 2.840 | 45.3 | 27.345 | 1.256 | True | ok |
| rate | 1.606508 | 0.8320 | 3.319 | 18.7 | 34.929 | 1.193 | True | ok |
| rate_generated | 1.611092 | 0.8247 | 2.560 | 18.7 | 30.196 | 1.269 | True | ok |
