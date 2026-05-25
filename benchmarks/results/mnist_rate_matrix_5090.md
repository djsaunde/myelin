# MNIST Rate-Readout Matrix

Compares rate-readout variants across matched MNIST training settings.

## Environment

- `generated_utc`: `2026-05-24T17:23:20+00:00`
- `device`: `cuda`
- `encoding`: `poisson`
- `batch`: `128`
- `epochs`: `2`
- `grad_clip`: `0.1`
- `rate_checkpoint_size`: `balanced`
- `train_limit`: `4096`
- `test_limit`: `2048`
- `compile_spiker_only`: `True`
- `matmul_precision`: `highest`

## Results

| Setting | T | Hidden | Variant | Final Test Loss | Final Test Acc | Total s | Peak CUDA MB | Avg Step ms | Steady Step ms | Compiled | Status |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| t10_h128 | 10 | 128 | rate | 1.606884 | 0.8286 | 3.665 | 18.7 | 45.851 | 1.764 | True | ok |
| t10_h128 | 10 | 128 | rate_triton_compile | 1.606329 | 0.8364 | 3.457 | 18.7 | 38.819 | 1.900 | True | ok |
| t25_h128 | 25 | 128 | rate | 1.608791 | 0.8408 | 5.485 | 110.3 | 69.671 | 1.555 | True | ok |
| t25_h128 | 25 | 128 | rate_triton_compile | 1.606392 | 0.8452 | 5.796 | 44.0 | 73.868 | 1.732 | True | ok |
| t10_h256 | 10 | 256 | rate | 1.595077 | 0.8535 | 5.728 | 103.9 | 73.137 | 1.475 | True | ok |
| t10_h256 | 10 | 256 | rate_triton_compile | 1.596204 | 0.8403 | 6.038 | 103.9 | 78.220 | 1.839 | True | ok |
