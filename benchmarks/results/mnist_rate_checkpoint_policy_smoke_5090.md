# MNIST Rate Checkpoint Policy Smoke

Compares the adaptive `checkpoint_size="balanced"` default against an explicit
`checkpoint_size=10` on the low-memory MNIST rate-readout example.

## Environment

- `device`: `cuda`
- `gpu`: `NVIDIA GeForce RTX 5090`
- `torch`: `2.12.0+cu130`
- `shape`: `T=10, B=128, hidden=128`
- `train_examples`: `512`
- `test_examples`: `512`
- `epochs`: `1`
- `backend`: `triton`
- `matmul_precision`: `highest`
- `grad_clip`: `0.1`

## Results

| Requested Checkpoint | Resolved Checkpoint | Final Test Loss | Final Test Accuracy | Peak CUDA MB | Steady Step ms |
|---|---:|---:|---:|---:|---:|
| balanced | 3 | 2.265670 | 0.3730 | 19.078 | 1.473 |
| 10 | 10 | 2.265670 | 0.3730 | 19.078 | 1.436 |

## Takeaway

At `T=10`, the adaptive `balanced` policy resolves to a smaller chunk than the
old explicit default path, but the smoke behavior is unchanged for loss,
accuracy, and peak memory. Steady-state step time is within run noise.
