# Packed LIF Forward Benchmark

## Environment

- `generated_utc`: `2026-05-24T07:27:32+00:00`
- `device`: `cuda`
- `gpu`: `NVIDIA GeForce RTX 5090`
- `torch`: `2.12.0+cu130`
- `cuda_available`: `True`
- `cuda_version`: `13.0`
- `warmup`: `5`
- `repeats`: `20`

## Results

| T | Batch | N | Dense ms | Dense+Pack ms | Direct Packed ms | Packed vs Dense | Packed vs Dense+Pack | Dense Spikes MB | Packed Spikes MB | Round Trip | Error |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 25 | 64 | 2048 | 0.023 | 0.036 | 0.024 | 0.99x | 1.54x | 12.5 | 0.4 | True |  |
| 100 | 64 | 2048 | 0.055 | 0.066 | 0.028 | 1.94x | 2.34x | 50.0 | 1.6 | True |  |
| 200 | 64 | 2048 | 0.149 | 0.190 | 0.091 | 1.63x | 2.08x | 100.0 | 3.1 | True |  |
