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
backend boundary:
    eager/compiled/Triton LIF: materialized [T, B, N] currents
    stream/Triton synapse: per-timestep or fused current computation
      |
      v
surrogate LIF recurrence
      |
      v
spike-rate style scalar loss
      |
      v
loss.backward() -> d(weight), optional d(inputs)
```

## Environment

- `generated_utc`: `2026-05-24T08:52:53+00:00`
- `device`: `cuda`
- `gpu`: `NVIDIA GeForce RTX 5090`
- `torch`: `2.12.0+cu130`
- `cuda_available`: `True`
- `cuda_version`: `13.0`
- `features`: `128`
- `surrogate`: `fast_sigmoid`
- `surrogate_slope`: `5.0`
- `checkpoint_size`: `25`
- `warmup`: `2`
- `repeats`: `5`
- `compile_enabled`: `True`
- `compile_mode`: `reduce-overhead`
- `compile_fullgraph`: `True`
- `compile_time_included`: `False`
- `matmul_precision`: `high`
- `dtype`: `torch.float32`
- `input_grad`: `False`

## Results

### Latency

| T | Batch | Features | N | Eager Fwd-only ms | Compiled Fwd-only ms | Stream Fwd-only ms | Checkpoint Fwd-only ms | Triton LIF Fwd-only ms | Triton Synapse Fwd-only ms | Triton Checkpoint Fwd-only ms | Eager Split Fwd ms | Compiled Split Fwd ms | Stream Split Fwd ms | Checkpoint Split Fwd ms | Triton Split Fwd ms | Triton Synapse Split Fwd ms | Triton Checkpoint Split Fwd ms | Eager Bwd ms | Compiled Bwd ms | Stream Bwd ms | Checkpoint Bwd ms | Triton LIF Bwd ms | Triton Synapse Bwd ms | Triton Checkpoint Bwd ms | Eager Fwd+Bwd ms | Compiled Fwd+Bwd ms | Stream Fwd+Bwd ms | Checkpoint Fwd+Bwd ms | Triton LIF Fwd+Bwd ms | Triton Synapse Fwd+Bwd ms | Triton Checkpoint Fwd+Bwd ms | Compiled Fwd+Bwd Speedup | Stream Fwd+Bwd Speedup | Checkpoint Fwd+Bwd Speedup | Triton LIF Fwd+Bwd Speedup | Triton Synapse Fwd+Bwd Speedup | Triton Checkpoint Fwd+Bwd Speedup | Compile Warmup ms | Compile Error | Triton Error | Triton Synapse Error | Triton Checkpoint Error |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| 100 | 64 | 128 | 2048 | 19.743 | 3.829 | 17.910 | 9.257 | 0.264 | 0.200 | 0.161 | 19.844 | 0.346 | 17.722 | 10.061 | 0.285 | 0.251 | 0.242 | 15.926 | 0.439 | 16.936 | 25.366 | 0.425 | 0.489 | 0.648 | 35.770 | 0.785 | 34.658 | 35.427 | 0.710 | 0.740 | 0.890 | 45.58x | 1.03x | 1.01x | 50.36x | 48.36x | 40.20x | 1779.922 |  |  |  |  |
| 200 | 64 | 128 | 2048 | 42.530 | 9.148 | 34.957 | 18.424 | 0.786 | 0.594 | 0.478 | 37.748 | 1.128 | 39.645 | 17.673 | 0.837 | 0.641 | 0.446 | 37.692 | 0.790 | 41.327 | 54.632 | 0.832 | 0.831 | 1.109 | 75.441 | 1.918 | 80.972 | 72.305 | 1.669 | 1.472 | 1.556 | 39.32x | 0.93x | 1.04x | 45.20x | 51.25x | 48.50x | 1910.102 |  |  |  |  |
| 500 | 64 | 128 | 2048 | 86.968 | 49.850 | 105.036 | 51.453 | 1.133 | 0.933 | 0.833 | 101.626 | 1.657 | 105.935 | 48.532 | 1.192 | 1.024 | 1.189 | 93.939 | 1.541 | 87.705 | 148.566 | 1.066 | 1.234 | 2.532 | 195.565 | 3.198 | 193.640 | 197.098 | 2.257 | 2.259 | 3.720 | 61.16x | 1.01x | 0.99x | 86.64x | 86.58x | 52.56x | 124376.955 |  |  |  |  |

### Generated Triton Comparison

| T | Batch | Features | N | Handwritten LIF Fwd+Bwd ms | Generated LIF Fwd+Bwd ms | Generated LIF vs Handwritten | Handwritten Synapse Fwd+Bwd ms | Generated Synapse Fwd+Bwd ms | Generated Synapse vs Handwritten | Handwritten Checkpoint Fwd+Bwd ms | Generated Checkpoint Fwd+Bwd ms | Generated Checkpoint vs Handwritten | Generated LIF Error | Generated Synapse Error | Generated Checkpoint Error |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| 100 | 64 | 128 | 2048 | 0.710 | 0.754 | 0.94x | 0.740 | 0.795 | 0.93x | 0.890 | 0.950 | 0.94x |  |  |  |
| 200 | 64 | 128 | 2048 | 1.669 | 1.729 | 0.97x | 1.472 | 1.564 | 0.94x | 1.556 | 1.646 | 0.94x |  |  |  |
| 500 | 64 | 128 | 2048 | 2.257 | 2.244 | 1.01x | 2.259 | 2.575 | 0.88x | 3.720 | 3.640 | 1.02x |  |  |  |

### CUDA Memory

| T | Batch | Features | N | Eager Fwd Peak MB | Compiled Fwd Peak MB | Stream Fwd Peak MB | Checkpoint Fwd Peak MB | Triton LIF Fwd Peak MB | Triton Synapse Fwd Peak MB | Triton Checkpoint Fwd Peak MB | Eager Bwd Peak MB | Compiled Bwd Peak MB | Stream Bwd Peak MB | Checkpoint Bwd Peak MB | Triton LIF Bwd Peak MB | Triton Synapse Bwd Peak MB | Triton Checkpoint Bwd Peak MB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 64 | 128 | 2048 | 428.1 | 16.1 | 330.1 | 184.1 | 233.2 | 183.1 | 137.1 | 428.1 | 16.1 | 330.1 | 184.1 | 233.2 | 234.6 | 151.5 |
| 200 | 64 | 128 | 2048 | 782.3 | 19.3 | 584.3 | 292.3 | 387.9 | 286.3 | 192.3 | 782.3 | 19.3 | 584.3 | 292.3 | 387.9 | 387.8 | 206.7 |
| 500 | 64 | 128 | 2048 | 1841.6 | 28.6 | 1343.6 | 613.6 | 852.0 | 595.6 | 357.6 | 1841.6 | 28.6 | 1343.6 | 613.6 | 852.0 | 847.1 | 372.0 |

### Generated Triton CUDA Memory

| T | Batch | Features | N | Handwritten LIF Bwd Peak MB | Generated LIF Bwd Peak MB | Handwritten Synapse Bwd Peak MB | Generated Synapse Bwd Peak MB | Handwritten Checkpoint Bwd Peak MB | Generated Checkpoint Bwd Peak MB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 64 | 128 | 2048 | 233.2 | 234.2 | 234.6 | 186.2 | 151.5 | 152.5 |
| 200 | 64 | 128 | 2048 | 387.9 | 388.9 | 387.8 | 290.9 | 206.7 | 207.7 |
| 500 | 64 | 128 | 2048 | 852.0 | 853.0 | 847.1 | 605.0 | 372.0 | 373.0 |
