# Dense LIF BPTT Baseline

Compile time is excluded from latency measurements.

## Workload

```text
[T, B, F] inputs
      |
      v
[F, N] trainable weight
      |
      v
[T, B, N] input currents
      |
      v
differentiable surrogate LIF unroll across time
      |
      v
spike-rate loss over all timesteps
      |
      v
loss.backward() -> d(weight)
```

## Environment

- `generated_utc`: `2026-05-24T05:31:21+00:00`
- `device`: `cuda`
- `gpu`: `NVIDIA GeForce RTX 5090`
- `torch`: `2.12.0+cu130`
- `cuda_available`: `True`
- `cuda_version`: `13.0`
- `features`: `128`
- `workload`: `surrogate-fast`
- `warmup`: `1`
- `repeats`: `3`
- `compile_enabled`: `True`
- `compile_mode`: `reduce-overhead`
- `compile_fullgraph`: `True`
- `compile_time_included`: `False`
- `matmul_precision`: `high`
- `dtype`: `torch.float32`

## Results

### Latency

| T | Batch | Features | N | Eager Fwd-only ms | Compiled Fwd-only ms | Fwd-only Speedup | Eager Split Fwd ms | Compiled Split Fwd ms | Split Fwd Speedup | Eager Bwd ms | Compiled Bwd ms | Bwd Speedup | Eager Fwd+Bwd ms | Compiled Fwd+Bwd ms | Fwd+Bwd Speedup | Compile Warmup ms | Compile Error |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 25 | 64 | 128 | 2048 | 4.474 | 1.077 | 4.16x | 3.617 | 0.267 | 13.53x | 4.344 | 0.828 | 5.25x | 7.961 | 1.095 | 7.27x | 8421.395 |  |

### CUDA Memory

| T | Batch | Features | N | Eager Alloc MB | Eager Fwd Peak MB | Eager Bwd Peak MB | Compiled Alloc MB | Compiled Fwd Peak MB | Compiled Bwd Peak MB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 25 | 64 | 128 | 2048 | 67.8 | 130.3 | 130.3 | 3.8 | 3.8 | 3.8 |
