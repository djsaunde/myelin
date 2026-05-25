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

- `generated_utc`: `2026-05-24T11:49:38+00:00`
- `device`: `cuda`
- `gpu`: `NVIDIA GeForce RTX 5090`
- `torch`: `2.12.0+cu130`
- `cuda_available`: `True`
- `cuda_version`: `13.0`
- `features`: `128`
- `surrogate`: `fast_sigmoid`
- `surrogate_slope`: `5.0`
- `checkpoint_size`: `25`
- `warmup`: `3`
- `repeats`: `10`
- `compile_enabled`: `True`
- `compile_mode`: `reduce-overhead`
- `compile_fullgraph`: `True`
- `compile_time_included`: `False`
- `matmul_precision`: `high`
- `dtype`: `torch.float32`
- `input_grad`: `True`

## Results

### Latency

| T | Batch | Features | N | Eager Fwd-only ms | Compiled Fwd-only ms | Stream Fwd-only ms | Checkpoint Fwd-only ms | Triton LIF Fwd-only ms | Triton Synapse Fwd-only ms | Triton Checkpoint Fwd-only ms | Eager Split Fwd ms | Compiled Split Fwd ms | Stream Split Fwd ms | Checkpoint Split Fwd ms | Triton Split Fwd ms | Triton Synapse Split Fwd ms | Triton Checkpoint Split Fwd ms | Eager Bwd ms | Compiled Bwd ms | Stream Bwd ms | Checkpoint Bwd ms | Triton LIF Bwd ms | Triton Synapse Bwd ms | Triton Checkpoint Bwd ms | Eager Fwd+Bwd ms | Compiled Fwd+Bwd ms | Stream Fwd+Bwd ms | Checkpoint Fwd+Bwd ms | Triton LIF Fwd+Bwd ms | Triton Synapse Fwd+Bwd ms | Triton Checkpoint Fwd+Bwd ms | Compiled Fwd+Bwd Speedup | Stream Fwd+Bwd Speedup | Checkpoint Fwd+Bwd Speedup | Triton LIF Fwd+Bwd Speedup | Triton Synapse Fwd+Bwd Speedup | Triton Checkpoint Fwd+Bwd Speedup | Compile Warmup ms | Compile Error | Triton Error | Triton Synapse Error | Triton Checkpoint Error |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| 100 | 64 | 128 | 2048 | 17.309 | 3.101 | 15.951 | 8.445 | 0.213 | 0.190 | 0.153 | 15.831 | 0.349 | 18.720 | 9.290 | 0.312 | 0.246 | 0.232 | 21.791 | 0.458 | 21.886 | 33.230 | 0.447 | 0.668 | 0.642 | 37.622 | 0.807 | 40.606 | 42.519 | 0.759 | 0.914 | 0.874 | 46.63x | 0.93x | 0.88x | 49.55x | 41.18x | 43.05x | 1781.380 |  |  |  |  |

### Generated Triton Comparison

| T | Batch | Features | N | Handwritten LIF Fwd+Bwd ms | Generated LIF Fwd+Bwd ms | Generated LIF vs Handwritten | Handwritten Synapse Fwd+Bwd ms | Generated Synapse Fwd+Bwd ms | Generated Synapse vs Handwritten | Handwritten Checkpoint Fwd+Bwd ms | Generated Checkpoint Fwd+Bwd ms | Generated Checkpoint vs Handwritten | Generated LIF Error | Generated Synapse Error | Generated Checkpoint Error |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| 100 | 64 | 128 | 2048 | 0.759 | 0.756 | 1.00x | 0.914 | 0.938 | 0.97x | 0.874 | 0.947 | 0.92x |  |  |  |

### CUDA Memory

| T | Batch | Features | N | Eager Fwd Peak MB | Compiled Fwd Peak MB | Stream Fwd Peak MB | Checkpoint Fwd Peak MB | Triton LIF Fwd Peak MB | Triton Synapse Fwd Peak MB | Triton Checkpoint Fwd Peak MB | Eager Bwd Peak MB | Compiled Bwd Peak MB | Stream Bwd Peak MB | Checkpoint Bwd Peak MB | Triton LIF Bwd Peak MB | Triton Synapse Bwd Peak MB | Triton Checkpoint Bwd Peak MB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 64 | 128 | 2048 | 428.1 | 16.1 | 330.1 | 184.1 | 233.2 | 183.1 | 137.1 | 428.1 | 16.1 | 330.1 | 184.1 | 233.2 | 288.3 | 154.3 |

### Generated Triton CUDA Memory

| T | Batch | Features | N | Handwritten LIF Bwd Peak MB | Generated LIF Bwd Peak MB | Handwritten Synapse Bwd Peak MB | Generated Synapse Bwd Peak MB | Handwritten Checkpoint Bwd Peak MB | Generated Checkpoint Bwd Peak MB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 64 | 128 | 2048 | 233.2 | 234.2 | 288.3 | 189.3 | 154.3 | 155.3 |
