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

- `generated_utc`: `2026-05-23T03:53:46+00:00`
- `device`: `cuda`
- `gpu`: `NVIDIA GeForce RTX 5090`
- `torch`: `2.12.0+cu130`
- `cuda_available`: `True`
- `cuda_version`: `13.0`
- `features`: `128`
- `workload`: `surrogate-spike`
- `warmup`: `2`
- `repeats`: `5`
- `compile_enabled`: `True`
- `compile_mode`: `reduce-overhead`
- `compile_fullgraph`: `True`
- `compile_time_included`: `False`
- `dtype`: `torch.float32`

## Results

### Latency

| T | Batch | Features | N | Eager Fwd-only ms | Compiled Fwd-only ms | Fwd-only Speedup | Eager Split Fwd ms | Compiled Split Fwd ms | Split Fwd Speedup | Eager Bwd ms | Compiled Bwd ms | Bwd Speedup | Eager Fwd+Bwd ms | Compiled Fwd+Bwd ms | Fwd+Bwd Speedup | Compile Error |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 25 | 16 | 128 | 512 | 2.514 | 1.364 | 1.84x | 2.798 | 0.317 | 8.82x | 3.279 | 0.319 | 10.28x | 6.077 | 0.636 | 9.56x |  |
| 25 | 64 | 128 | 2048 | 2.761 | 1.666 | 1.66x | 3.192 | 0.370 | 8.62x | 10.628 | 0.337 | 31.52x | 13.820 | 0.707 | 19.53x |  |
| 25 | 256 | 128 | 8192 | 10.515 | 4.071 | 2.58x | 8.898 | 2.852 | 3.12x | 3.650 | 0.767 | 4.76x | 12.548 | 3.619 | 3.47x |  |
| 100 | 16 | 128 | 512 | 9.288 | 3.861 | 2.41x | 10.103 | 0.594 | 17.01x | 12.351 | 0.457 | 27.05x | 22.454 | 1.051 | 21.37x |  |
| 100 | 64 | 128 | 2048 | 14.520 | 5.334 | 2.72x | 11.911 | 2.673 | 4.46x | 13.617 | 0.803 | 16.97x | 25.528 | 3.476 | 7.34x |  |
| 100 | 256 | 128 | 8192 | 11.059 | 48.114 | 0.23x | 11.438 | 44.997 | 0.25x | 14.620 | 3.207 | 4.56x | 26.058 | 48.204 | 0.54x |  |
| 200 | 16 | 128 | 512 | 19.880 | 8.994 | 2.21x | 20.171 | 1.012 | 19.94x | 25.018 | 0.716 | 34.95x | 45.189 | 1.728 | 26.16x |  |
| 200 | 64 | 128 | 2048 | 23.482 | 12.857 | 1.83x | 22.709 | 6.297 | 3.61x | 27.233 | 1.580 | 17.23x | 49.942 | 7.877 | 6.34x |  |
| 200 | 256 | 128 | 8192 | 23.126 | 101.382 | 0.23x | 23.548 | 94.881 | 0.25x | 28.872 | 5.734 | 5.04x | 52.420 | 100.616 | 0.52x |  |

### CUDA Memory

| T | Batch | Features | N | Eager Alloc MB | Eager Fwd Peak MB | Eager Bwd Peak MB | Compiled Alloc MB | Compiled Fwd Peak MB | Compiled Bwd Peak MB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 25 | 16 | 128 | 512 | 64.9 | 67.9 | 67.9 | 1.0 | 1.0 | 1.0 |
| 25 | 64 | 128 | 2048 | 68.0 | 117.5 | 117.5 | 3.8 | 3.8 | 3.8 |
| 25 | 256 | 128 | 8192 | 80.1 | 885.0 | 885.0 | 15.1 | 15.1 | 15.1 |
| 100 | 16 | 128 | 512 | 69.5 | 81.8 | 81.8 | 1.6 | 1.6 | 1.6 |
| 100 | 64 | 128 | 2048 | 70.4 | 269.9 | 269.9 | 6.2 | 6.2 | 6.2 |
| 100 | 256 | 128 | 8192 | 89.5 | 3293.5 | 3293.5 | 24.5 | 24.5 | 24.5 |
| 200 | 16 | 128 | 512 | 70.3 | 95.1 | 95.1 | 2.3 | 2.3 | 2.3 |
| 200 | 64 | 128 | 2048 | 73.5 | 473.0 | 473.0 | 9.3 | 9.3 | 9.3 |
| 200 | 256 | 128 | 8192 | 103.0 | 6507.0 | 6507.0 | 38.0 | 38.0 | 38.0 |
