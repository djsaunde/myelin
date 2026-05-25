# LIF Compile Baseline

Compile time is excluded from latency measurements.

## Environment

- `generated_utc`: `2026-05-24T04:38:46+00:00`
- `device`: `cuda`
- `gpu`: `NVIDIA GeForce RTX 5090`
- `torch`: `2.12.0+cu130`
- `cuda_available`: `True`
- `cuda_version`: `13.0`
- `warmup`: `5`
- `repeats`: `20`
- `compile_enabled`: `True`
- `compile_mode`: `reduce-overhead`
- `compile_fullgraph`: `True`
- `compile_time_included`: `False`
- `dtype`: `torch.float32`

## Results

| T | Batch | N | Eager ms | Compiled ms | Triton ms | Compile Warmup ms | Compiled Speedup | Triton Speedup | Peak CUDA MB | Compile Error | Triton Error |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 25 | 16 | 512 | 1.300 | 0.071 | 0.018 | 1423.614 | 18.33x | 70.54x | 2.4 |  |  |
| 25 | 64 | 2048 | 1.287 | 0.073 | 0.017 | 396.330 | 17.73x | 73.75x | 39.0 |  |  |
| 100 | 64 | 2048 | 5.124 | 0.133 | 0.055 | 1187.888 | 38.46x | 93.97x | 163.5 |  |  |
| 200 | 64 | 2048 | 10.504 | 0.354 | 0.279 | 2139.040 | 29.64x | 37.64x | 326.0 |  |  |
