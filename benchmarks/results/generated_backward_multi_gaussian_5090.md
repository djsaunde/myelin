# Generated Surrogate Backward Benchmark

## Environment

- `generated_utc`: `2026-05-24T10:15:05+00:00`
- `device`: `cuda`
- `gpu`: `NVIDIA GeForce RTX 5090`
- `torch`: `2.12.0+cu130`
- `cuda_available`: `True`
- `cuda_version`: `13.0`
- `shape`: `T=100, B=64, N=2048`
- `surrogate`: `multi_gaussian`
- `surrogate_slope`: `5.0`
- `block_size`: `256`
- `warmup`: `5`
- `repeats`: `20`

## Results

| Variant | Backward ms | Peak MB | Speedup vs handwritten | Error |
|---|---:|---:|---:|---|
| handwritten Triton backward | 0.128 | 201.0 |  |  |
| generated Triton backward | 0.137 | 201.0 | 0.94x |  |
