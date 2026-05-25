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

- `generated_utc`: `2026-05-24T05:16:44+00:00`
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
| 25 | 64 | 128 | 2048 | 4.186 | 0.984 | 0.211 | 5.022 | 0.160 | 0.242 | 4.233 | 0.254 | 5.371 | 9.255 | 0.414 | 5.612 | 22.37x | 1.65x | 1115.409 |  |  |
| 100 | 64 | 128 | 2048 | 21.172 | 2.833 | 0.194 | 16.268 | 0.341 | 0.262 | 19.280 | 0.401 | 19.700 | 35.548 | 0.743 | 19.962 | 47.87x | 1.78x | 1086.556 |  |  |
| 200 | 64 | 128 | 2048 | 32.926 | 23.298 | 0.497 | 32.164 | 1.073 | 0.451 | 47.871 | 0.798 | 39.279 | 80.035 | 1.871 | 39.730 | 42.78x | 2.01x | 1711.858 |  |  |

### CUDA Memory

| T | Batch | Features | N | Eager Fwd Peak MB | Compiled Fwd Peak MB | Triton Fwd Peak MB | Eager Bwd Peak MB | Compiled Bwd Peak MB | Triton Bwd Peak MB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 25 | 64 | 128 | 2048 | 156.3 | 4.8 | 95.3 | 156.3 | 4.8 | 146.8 |
| 100 | 64 | 128 | 2048 | 422.1 | 7.1 | 172.6 | 422.1 | 7.1 | 374.1 |
| 200 | 64 | 128 | 2048 | 775.3 | 10.3 | 275.8 | 775.3 | 10.3 | 677.3 |
