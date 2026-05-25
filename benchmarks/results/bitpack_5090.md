# Bitpacked Spike Storage Benchmark

## Environment

- `generated_utc`: `2026-05-24T16:11:33+00:00`
- `device`: `cuda`
- `gpu`: `NVIDIA GeForce RTX 5090`
- `torch`: `2.12.0+cu130`
- `cuda_available`: `True`
- `cuda_version`: `13.0`
- `rate`: `0.05`
- `warmup`: `5`
- `repeats`: `20`

## Results

| T | Batch | N | Dense MB | Packed MB | Compression | Packed Shape | Torch Pack ms | Triton Pack ms | Count Backend | Packed Count ms | Unpack+Sum ms | Per-Neuron Count ms | Unpack+Per-Neuron Sum ms | Count OK | Per-Neuron OK | Round Trip | Triton Round Trip |
|---:|---:|---:|---:|---:|---:|---|---:|---:|---|---:|---:|---:|---:|---|---|---|---|
| 25 | 64 | 2048 | 12.5 | 0.4 | 32.00x | `(25, 64, 64)` | 0.079 | 0.016 | Triton auto | 0.034 | 0.089 | 0.093 | 0.085 | True | True | True | True |
| 100 | 64 | 2048 | 50.0 | 1.6 | 32.00x | `(100, 64, 64)` | 0.312 | 0.016 | Triton auto | 0.028 | 0.295 | 0.258 | 0.301 | True | True | True | True |
| 200 | 64 | 2048 | 100.0 | 3.1 | 32.00x | `(200, 64, 64)` | 0.664 | 0.060 | Triton auto | 0.028 | 0.674 | 0.552 | 0.682 | True | True | True | True |
