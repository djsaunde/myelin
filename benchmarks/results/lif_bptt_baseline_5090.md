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
LIF unroll across time
      |
      v
final membrane energy loss
      |
      v
loss.backward() -> d(weight)
```

## Environment

- `generated_utc`: `2026-05-23T03:45:41+00:00`
- `device`: `cuda`
- `gpu`: `NVIDIA GeForce RTX 5090`
- `torch`: `2.12.0+cu130`
- `cuda_available`: `True`
- `cuda_version`: `13.0`
- `features`: `128`
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
| 25 | 16 | 128 | 512 | 1.643 | 1.279 | 1.28x | 1.808 | 0.359 | 5.03x | 1.223 | 0.425 | 2.87x | 3.031 | 0.785 | 3.86x |  |
| 25 | 64 | 128 | 2048 | 1.708 | 1.171 | 1.46x | 1.888 | 0.231 | 8.18x | 1.318 | 0.273 | 4.82x | 3.206 | 0.504 | 6.36x |  |
| 25 | 256 | 128 | 8192 | 5.088 | 1.769 | 2.88x | 2.752 | 0.708 | 3.89x | 2.154 | 0.770 | 2.80x | 4.907 | 1.479 | 3.32x |  |
| 100 | 16 | 128 | 512 | 9.234 | 2.809 | 3.29x | 8.683 | 0.603 | 14.39x | 5.249 | 0.482 | 10.90x | 13.932 | 1.085 | 12.84x |  |
| 100 | 64 | 128 | 2048 | 6.704 | 3.643 | 1.84x | 7.150 | 0.636 | 11.25x | 4.693 | 0.490 | 9.59x | 11.843 | 1.125 | 10.53x |  |
| 100 | 256 | 128 | 8192 | 8.646 | 7.188 | 1.20x | 9.103 | 4.931 | 1.85x | 7.395 | 2.441 | 3.03x | 16.498 | 7.372 | 2.24x |  |
| 200 | 16 | 128 | 512 | 13.202 | 7.541 | 1.75x | 13.831 | 0.818 | 16.92x | 8.833 | 0.697 | 12.68x | 22.664 | 1.515 | 14.96x |  |
| 200 | 64 | 128 | 2048 | 14.218 | 7.930 | 1.79x | 16.757 | 1.032 | 16.24x | 13.321 | 0.616 | 21.64x | 30.078 | 1.648 | 18.26x |  |
| 200 | 256 | 128 | 8192 | 34.530 | 14.642 | 2.36x | 17.481 | 8.546 | 2.05x | 11.874 | 4.371 | 2.72x | 29.354 | 12.917 | 2.27x |  |

### CUDA Memory

| T | Batch | Features | N | Eager Alloc MB | Eager Fwd Peak MB | Eager Bwd Peak MB | Compiled Alloc MB | Compiled Fwd Peak MB | Compiled Bwd Peak MB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 25 | 16 | 128 | 512 | 64.9 | 67.3 | 67.3 | 1.0 | 1.0 | 1.0 |
| 25 | 64 | 128 | 2048 | 68.0 | 108.7 | 108.7 | 3.8 | 3.8 | 3.8 |
| 25 | 256 | 128 | 8192 | 80.1 | 743.0 | 743.0 | 15.1 | 15.1 | 15.1 |
| 100 | 16 | 128 | 512 | 69.5 | 80.4 | 80.4 | 1.6 | 1.6 | 1.6 |
| 100 | 64 | 128 | 2048 | 70.4 | 232.9 | 232.9 | 6.2 | 6.2 | 6.2 |
| 100 | 256 | 128 | 8192 | 89.5 | 2701.5 | 2701.5 | 24.5 | 24.5 | 24.5 |
| 200 | 16 | 128 | 512 | 70.3 | 90.5 | 90.5 | 2.3 | 2.3 | 2.3 |
| 200 | 64 | 128 | 2048 | 73.5 | 398.5 | 398.5 | 9.3 | 9.3 | 9.3 |
| 200 | 256 | 128 | 8192 | 103.0 | 5315.0 | 5315.0 | 38.0 | 38.0 | 38.0 |
