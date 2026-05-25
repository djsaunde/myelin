# Scalar Loss Boundary Benchmark

Compares scalar-loss PyTorch graphs captured by `torch.compile` against
Triton checkpoint paths that keep spike-rate objectives out of the public
dense `[T, B, N]` spike-output contract.

Rows marked hard-forward use thresholded spikes in forward with
fast-sigmoid straight-through gradients, matching the Triton surrogate
contract more closely than the soft-forward rows.

## Environment

- `generated_utc`: `2026-05-25T16:10:07+00:00`
- `device`: `cuda`
- `gpu`: `NVIDIA GeForce RTX 5090`
- `torch`: `2.12.0+cu130`
- `cuda_available`: `True`
- `cuda_version`: `13.0`
- `features`: `128`
- `checkpoint_size`: `25`
- `resolved_checkpoint_sizes`: `{100: 25}`
- `surrogate`: `fast_sigmoid`
- `surrogate_slope`: `5.0`
- `compile_mode`: `reduce-overhead`
- `matmul_precision`: `high`
- `warmup`: `3`
- `repeats`: `10`

## Results

| T | Batch | Features | N | Variant | Fwd+Bwd ms | Baseline Alloc MB | Peak MB | Increment MB | Compile Warmup ms | Error |
|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---|
| 100 | 64 | 128 | 2048 | torch.compile materialized soft-surrogate scalar loss | 0.710 | 5.1 | 5.1 | 0.0 | 76966.388 |  |
| 100 | 64 | 128 | 2048 | torch.compile looped soft-surrogate scalar loss | 1.272 | 6.1 | 6.1 | 0.0 | 24029.262 |  |
| 100 | 64 | 128 | 2048 | torch.compile materialized hard-forward scalar loss | 0.733 | 7.2 | 7.2 | 0.0 | 32609.426 |  |
| 100 | 64 | 128 | 2048 | torch.compile looped hard-forward scalar loss | 1.250 | 8.2 | 8.2 | 0.0 | 41370.064 |  |
| 100 | 64 | 128 | 2048 | Triton checkpoint dense-spike scalar loss | 0.912 | 10.2 | 76.2 | 66.0 |  |  |
| 100 | 64 | 128 | 2048 | Triton checkpoint scalar-rate loss | 0.950 | 12.2 | 28.7 | 16.5 |  |  |
| 100 | 64 | 128 | 2048 | Generated Triton checkpoint scalar-rate loss | 0.802 | 14.2 | 30.7 | 16.5 |  |  |
| 100 | 64 | 128 | 2048 | Triton checkpoint scalar-rate replay-no-scratch loss | 4.973 | 16.2 | 19.7 | 3.5 |  |  |
