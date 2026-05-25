# Rate Readout Benchmark

Compares a dense `[T, B, classes]` spike readout against direct `[B, classes]` spike rates for a classifier loss.

## Environment

- `generated_utc`: `2026-05-24T10:33:23+00:00`
- `device`: `cuda`
- `gpu`: `NVIDIA GeForce RTX 5090`
- `torch`: `2.12.0+cu130`
- `features`: `128`
- `checkpoint_size`: `25`
- `warmup`: `3`
- `repeats`: `10`

## Results

| T | Batch | Features | Classes | Dense Fwd+Bwd ms | Rate Fwd+Bwd ms | Module Rate Fwd+Bwd ms | Generated Rate Fwd+Bwd ms | Generated Module Rate Fwd+Bwd ms | Dense Bwd Peak MB | Rate Bwd Peak MB | Module Rate Bwd Peak MB | Generated Rate Bwd Peak MB | Generated Module Rate Bwd Peak MB | Generated Errors |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 100 | 128 | 128 | 10 | 0.512 | 0.466 | 0.504 | 0.532 | 0.543 | 6.9 | 6.5 | 6.5 | 6.5 | 6.5 |  |
| 100 | 128 | 128 | 1000 | 0.751 | 0.686 | 0.688 | 0.751 | 0.743 | 74.1 | 26.2 | 26.7 | 27.2 | 27.7 |  |
| 100 | 128 | 128 | 2048 | 1.116 | 1.010 | 1.043 | 1.054 | 1.061 | 145.1 | 47.1 | 48.1 | 49.1 | 50.1 |  |
