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

- `generated_utc`: `2026-05-24T05:21:45+00:00`
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
| 25 | 64 | 128 | 2048 | 5.404 | 0.986 | 0.195 | 4.845 | 0.161 | 0.230 | 4.051 | 0.232 | 0.325 | 8.897 | 0.393 | 0.555 | 22.62x | 16.02x | 1110.836 |  |  |
| 100 | 64 | 128 | 2048 | 15.613 | 2.747 | 0.214 | 16.953 | 0.344 | 0.265 | 17.136 | 0.397 | 0.408 | 34.089 | 0.741 | 0.673 | 45.99x | 50.63x | 1104.509 |  |  |
| 200 | 64 | 128 | 2048 | 33.021 | 7.495 | 0.716 | 36.398 | 1.058 | 0.709 | 36.927 | 0.796 | 0.949 | 73.325 | 1.854 | 1.658 | 39.55x | 44.22x | 1904.738 |  |  |

### CUDA Memory

| T | Batch | Features | N | Eager Fwd Peak MB | Compiled Fwd Peak MB | Triton Fwd Peak MB | Eager Bwd Peak MB | Compiled Bwd Peak MB | Triton Bwd Peak MB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 25 | 64 | 128 | 2048 | 156.3 | 4.8 | 107.8 | 156.3 | 4.8 | 119.8 |
| 100 | 64 | 128 | 2048 | 422.1 | 7.1 | 222.6 | 422.1 | 7.1 | 272.1 |
| 200 | 64 | 128 | 2048 | 775.3 | 10.3 | 375.8 | 775.3 | 10.3 | 475.3 |
