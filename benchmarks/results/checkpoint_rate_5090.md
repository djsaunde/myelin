# Checkpoint Spike-Rate Benchmark

Compares dense checkpointed spike output against a scalar spike-rate output.

## Environment

- `generated_utc`: `2026-05-24T09:03:08+00:00`
- `device`: `cuda`
- `gpu`: `NVIDIA GeForce RTX 5090`
- `torch`: `2.12.0+cu130`
- `features`: `128`
- `checkpoint_size`: `25`
- `warmup`: `2`
- `repeats`: `5`

## Results

| T | Batch | Features | N | Dense Fwd+Bwd ms | Rate Fwd+Bwd ms | Dense Bwd Peak MB | Rate Bwd Peak MB |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 64 | 128 | 2048 | 0.858 | 0.847 | 73.5 | 25.0 |
| 200 | 64 | 128 | 2048 | 1.535 | 1.292 | 128.6 | 30.1 |
| 500 | 64 | 128 | 2048 | 35.944 | 2.612 | 294.0 | 45.5 |
