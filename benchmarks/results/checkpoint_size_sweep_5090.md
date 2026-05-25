# Checkpoint Size Sweep

Sweeps Triton checkpoint chunk size for dense and rate-output LIF training.

## Environment

- `generated_utc`: `2026-05-24T13:38:17+00:00`
- `device`: `cuda`
- `gpu`: `NVIDIA GeForce RTX 5090`
- `torch`: `2.12.0+cu130`
- `cuda_available`: `True`
- `cuda_version`: `13.0`
- `shape`: `T=100, B=64, F=128, N=2048`
- `checkpoint_sizes`: `(5, 10, 25, 50, 100)`
- `expected_currents_mb`: `50.0`
- `warmup`: `3`
- `repeats`: `10`

## Results

| Checkpoint | Chunks | Variant | Fwd ms | Bwd ms | Fwd+Bwd ms | Expected Scratch MB | Fwd Increment MB | Bwd Increment MB | Error |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| 5 | 20 | dense | 0.236 | 0.809 | 1.044 | 2.5 | 60.0 | 64.0 |  |
| 5 | 20 | rate | 0.220 | 0.817 | 1.036 | 2.5 | 10.0 | 14.5 |  |
| 5 | 20 | generated_rate | 0.221 | 0.895 | 1.116 | 2.5 | 10.0 | 14.5 |  |
| 5 | 20 | replay_rate |  |  | 1.339 | 2.5 | 11.5 | 11.5 |  |
| 10 | 10 | dense | 0.236 | 0.583 | 0.820 | 5.0 | 55.0 | 61.5 |  |
| 10 | 10 | rate | 0.219 | 0.585 | 0.804 | 5.0 | 5.0 | 12.0 |  |
| 10 | 10 | generated_rate | 0.220 | 0.617 | 0.838 | 5.0 | 5.0 | 12.0 |  |
| 10 | 10 | replay_rate |  |  | 2.139 | 5.0 | 6.5 | 6.5 |  |
| 25 | 4 | dense | 0.234 | 0.498 | 0.732 | 12.5 | 52.0 | 66.0 |  |
| 25 | 4 | rate | 0.220 | 0.469 | 0.689 | 12.5 | 2.0 | 16.5 |  |
| 25 | 4 | generated_rate | 0.225 | 0.548 | 0.772 | 12.5 | 2.0 | 16.5 |  |
| 25 | 4 | replay_rate |  |  | 5.007 | 12.5 | 3.5 | 3.5 |  |
| 50 | 2 | dense | 0.229 | 0.486 | 0.716 | 25.0 | 51.0 | 78.5 |  |
| 50 | 2 | rate | 0.218 | 0.455 | 0.673 | 25.0 | 1.0 | 29.0 |  |
| 50 | 2 | generated_rate | 0.219 | 0.500 | 0.719 | 25.0 | 1.0 | 29.0 |  |
| 50 | 2 | replay_rate |  |  | 8.924 | 25.0 | 2.5 | 2.5 |  |
| 100 | 1 | dense | 0.237 | 0.494 | 0.731 | 50.0 | 50.5 | 102.0 |  |
| 100 | 1 | rate | 0.228 | 0.449 | 0.677 | 50.0 | 0.5 | 52.5 |  |
| 100 | 1 | generated_rate | 0.220 | 0.499 | 0.719 | 50.0 | 0.5 | 52.5 |  |
| 100 | 1 | replay_rate |  |  | 17.569 | 50.0 | 2.0 | 2.0 |  |

## Rate Pareto Frontier

Non-dominated rate-output choices when minimizing fwd+bwd latency and backward peak-memory increment.

| Checkpoint | Chunks | Variant | Fwd+Bwd ms | Bwd Increment MB |
|---:|---:|---|---:|---:|
| 100 | 1 | replay_rate | 17.569 | 2.0 |
| 50 | 2 | replay_rate | 8.924 | 2.5 |
| 25 | 4 | replay_rate | 5.007 | 3.5 |
| 10 | 10 | replay_rate | 2.139 | 6.5 |
| 5 | 20 | replay_rate | 1.339 | 11.5 |
| 10 | 10 | rate | 0.804 | 12.0 |
| 25 | 4 | rate | 0.689 | 16.5 |
| 50 | 2 | rate | 0.673 | 29.0 |

## Takeaway

The replay-rate variant avoids the chunk-sized pre-reset scratch by replaying chunk prefixes inside the backward kernel. It gives a lower memory floor, but its latency grows quickly with checkpoint size.

Use the regular `rate` rows for the default speed/memory tradeoff and `replay_rate` as a memory-floor experiment.
