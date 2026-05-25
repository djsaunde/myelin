# Generated Fused Synapse Backward Benchmark

## Environment

- `generated_utc`: `2026-05-24T06:54:49+00:00`
- `device`: `cuda`
- `gpu`: `NVIDIA GeForce RTX 5090`
- `torch`: `2.12.0+cu130`
- `cuda_available`: `True`
- `cuda_version`: `13.0`
- `shape`: `T=100, B=64, F=128, N=2048`
- `surrogate`: `fast_sigmoid`
- `surrogate_slope`: `5.0`
- `warmup`: `5`
- `repeats`: `20`

## Results

| Variant | Fwd+Bwd ms | Peak MB | Speedup vs handwritten | Error |
|---|---:|---:|---:|---|
| handwritten Triton fused synapse | 0.496 | 158.1 |  |  |
| generated Triton fused synapse | 0.516 | 158.2 | 0.96x |  |
