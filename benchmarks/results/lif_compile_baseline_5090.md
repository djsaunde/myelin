# LIF Compile Baseline

Compile time is excluded from latency measurements.

## Environment

- `generated_utc`: `2026-05-23T03:30:09+00:00`
- `device`: `cuda`
- `gpu`: `NVIDIA GeForce RTX 5090`
- `torch`: `2.12.0+cu130`
- `cuda_available`: `True`
- `cuda_version`: `13.0`
- `warmup`: `3`
- `repeats`: `10`
- `compile_enabled`: `True`
- `compile_mode`: `reduce-overhead`
- `compile_fullgraph`: `True`
- `compile_time_included`: `False`
- `dtype`: `torch.float32`

## Results

| T | Batch | N | Eager ms | Compiled ms | Speedup | Peak CUDA MB | Compile Error |
|---:|---:|---:|---:|---:|---:|---:|---|
| 25 | 16 | 512 | 1.459 | 0.093 | 15.66x | 97.6 |  |
| 25 | 64 | 2048 | 1.441 | 0.083 | 17.39x | 122.0 |  |
| 25 | 256 | 8192 | 1.464 | 0.596 | 2.46x | 624.0 |  |
| 100 | 16 | 512 | 5.780 | 0.105 | 55.20x | 411.2 |  |
| 100 | 64 | 2048 | 5.855 | 0.143 | 41.00x | 253.7 |  |
| 100 | 256 | 8192 | 6.047 | 2.713 | 2.23x | 2616.0 |  |
| 200 | 16 | 512 | 11.756 | 0.126 | 93.45x | 1614.3 |  |
| 200 | 64 | 2048 | 12.682 | 0.358 | 35.45x | 906.8 |  |
| 200 | 256 | 8192 | 17.500 | 5.767 | 3.03x | 5216.0 |  |
