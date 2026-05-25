# Distributed Packed Collectives Benchmark

Local multi-process benchmark using `torch.distributed` and packed spike helpers.

## Environment

- `generated_utc`: `2026-05-24T16:07:57+00:00`
- `backend`: `nccl`
- `device`: `cuda`
- `world_size`: `1`
- `shape`: `T=100, B=64, N=2048`
- `spike_probability`: `0.05`
- `seed`: `0`
- `warmup`: `3`
- `repeats`: `10`

## Results

| Rank | Dense AllGather ms | Packed AllGather ms | Dense Count AllReduce ms | Packed Count AllReduce ms | Packed Rate AllReduce ms | Dense Payload MB | Packed Payload MB | Compression | Device | Count Shape | Rate Shape | Packed Gather OK | Count Max Error | Rate Max Error |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---:|---:|---:|
| 0 | 0.123 | 0.064 | 0.059 | 0.085 | 0.101 | 50.0 | 1.6 | 32.00x | `cuda:0` | `(100, 64)` | `(100, 64)` | True | 0 | 0.000e+00 |
