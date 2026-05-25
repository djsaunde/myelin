# Distributed Packed Collectives Benchmark

Local multi-process benchmark using `torch.distributed` and packed spike helpers.

## Environment

- `generated_utc`: `2026-05-24T16:08:51+00:00`
- `backend`: `gloo`
- `device`: `cpu`
- `world_size`: `2`
- `shape`: `T=100, B=64, N=2048`
- `spike_probability`: `0.05`
- `seed`: `0`
- `warmup`: `3`
- `repeats`: `10`

## Results

| Rank | Dense AllGather ms | Packed AllGather ms | Dense Count AllReduce ms | Packed Count AllReduce ms | Packed Rate AllReduce ms | Dense Payload MB | Packed Payload MB | Compression | Device | Count Shape | Rate Shape | Packed Gather OK | Count Max Error | Rate Max Error |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---:|---:|---:|
| 0 | 102.063 | 1.804 | 2.519 | 3.436 | 4.016 | 50.0 | 1.6 | 32.00x | `cpu` | `(100, 64)` | `(100, 64)` | True | 0 | 0.000e+00 |
| 1 | 102.064 | 1.804 | 2.523 | 3.436 | 4.015 | 50.0 | 1.6 | 32.00x | `cpu` | `(100, 64)` | `(100, 64)` | True | 0 | 0.000e+00 |
