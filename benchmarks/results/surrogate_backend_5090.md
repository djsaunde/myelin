# Surrogate LIF Backend Benchmark

Compile time is excluded from latency measurements.

## Workload

```text
[T, B, F] inputs
      |
      v
[F, N] trainable weight
      |
      v
[T, B, N] currents
      |
      v
surrogate_lif_forward(..., backend=?)
      |
      v
spike-rate style scalar loss
      |
      v
loss.backward() -> d(weight)
```

## Environment

- `generated_utc`: `2026-05-24T05:12:59+00:00`
- `device`: `cuda`
- `gpu`: `NVIDIA GeForce RTX 5090`
- `torch`: `2.12.0+cu130`
- `cuda_available`: `True`
- `cuda_version`: `13.0`
- `features`: `128`
- `surrogate`: `fast_sigmoid`
- `surrogate_slope`: `5.0`
- `warmup`: `3`
- `repeats`: `10`
- `compile_enabled`: `True`
- `compile_mode`: `reduce-overhead`
- `compile_fullgraph`: `True`
- `compile_time_included`: `False`
- `matmul_precision`: `high`
- `dtype`: `torch.float32`

## Results

### Latency

| T | Batch | Features | N | Eager Fwd-only ms | Compiled Fwd-only ms | Triton Fwd-only ms | Eager Split Fwd ms | Compiled Split Fwd ms | Triton Split Fwd ms | Eager Bwd ms | Compiled Bwd ms | Triton Bwd ms | Eager Fwd+Bwd ms | Compiled Fwd+Bwd ms | Triton Fwd+Bwd ms | Compiled Fwd+Bwd Speedup | Triton Fwd+Bwd Speedup | Compile Warmup ms | Compile Error | Triton Error |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 25 | 64 | 128 | 2048 | 4.241 | 1.020 | 0.215 | 4.873 | 0.167 | 0.247 | 4.433 | 0.247 | 8.713 | 9.306 | 0.414 | 8.960 | 22.45x | 1.04x | 10150.331 |  |  |
| 100 | 64 | 128 | 2048 | 16.866 | 2.919 | 0.211 | 16.901 | 0.337 | 0.286 | 20.187 | 0.415 | 43.228 | 37.088 | 0.752 | 43.513 | 49.31x | 0.85x | 37396.704 |  |  |
| 200 | 64 | 128 | 2048 | 30.378 | 7.608 | 0.296 | 33.648 | 0.730 | 0.459 | 43.868 | 0.685 | 66.731 | 77.516 | 1.414 | 67.190 | 54.80x | 1.15x | 61314.050 |  |  |

### CUDA Memory

| T | Batch | Features | N | Eager Fwd Peak MB | Compiled Fwd Peak MB | Triton Fwd Peak MB | Eager Bwd Peak MB | Compiled Bwd Peak MB | Triton Bwd Peak MB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 25 | 64 | 128 | 2048 | 156.3 | 4.8 | 95.3 | 156.3 | 4.8 | 170.3 |
| 100 | 64 | 128 | 2048 | 422.1 | 7.1 | 172.6 | 422.1 | 7.1 | 472.6 |
| 200 | 64 | 128 | 2048 | 775.3 | 10.3 | 275.8 | 775.3 | 10.3 | 875.8 |
