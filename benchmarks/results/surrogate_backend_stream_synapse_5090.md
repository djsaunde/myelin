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

- `generated_utc`: `2026-05-24T05:36:16+00:00`
- `device`: `cuda`
- `gpu`: `NVIDIA GeForce RTX 5090`
- `torch`: `2.12.0+cu130`
- `cuda_available`: `True`
- `cuda_version`: `13.0`
- `features`: `128`
- `surrogate`: `fast_sigmoid`
- `surrogate_slope`: `5.0`
- `warmup`: `2`
- `repeats`: `5`
- `compile_enabled`: `True`
- `compile_mode`: `reduce-overhead`
- `compile_fullgraph`: `True`
- `compile_time_included`: `False`
- `matmul_precision`: `high`
- `dtype`: `torch.float32`

## Results

### Latency

| T | Batch | Features | N | Eager Fwd-only ms | Compiled Fwd-only ms | Stream Fwd-only ms | Triton Fwd-only ms | Eager Split Fwd ms | Compiled Split Fwd ms | Stream Split Fwd ms | Triton Split Fwd ms | Eager Bwd ms | Compiled Bwd ms | Stream Bwd ms | Triton Bwd ms | Eager Fwd+Bwd ms | Compiled Fwd+Bwd ms | Stream Fwd+Bwd ms | Triton Fwd+Bwd ms | Compiled Fwd+Bwd Speedup | Stream Fwd+Bwd Speedup | Triton Fwd+Bwd Speedup | Compile Warmup ms | Compile Error | Triton Error |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 25 | 64 | 128 | 2048 | 5.537 | 1.251 | 4.126 | 0.296 | 4.060 | 0.176 | 4.271 | 0.366 | 5.132 | 0.249 | 5.601 | 0.449 | 9.192 | 0.426 | 9.872 | 0.815 | 21.59x | 0.93x | 11.28x | 1124.458 |  |  |

### CUDA Memory

| T | Batch | Features | N | Eager Fwd Peak MB | Compiled Fwd Peak MB | Stream Fwd Peak MB | Triton Fwd Peak MB | Eager Bwd Peak MB | Compiled Bwd Peak MB | Stream Bwd Peak MB | Triton Bwd Peak MB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 25 | 64 | 128 | 2048 | 157.3 | 6.8 | 134.3 | 109.8 | 157.3 | 6.8 | 134.3 | 121.8 |
