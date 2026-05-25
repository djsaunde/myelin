# Generated Forward Benchmark

## Environment

- `generated_utc`: `2026-05-24T08:29:12+00:00`
- `device`: `cuda`
- `gpu`: `NVIDIA GeForce RTX 5090`
- `torch`: `2.12.0+cu130`
- `cuda_available`: `True`
- `cuda_version`: `13.0`
- `shape`: `T=100, B=64, N=2048`
- `block_size`: `256`
- `warmup`: `5`
- `repeats`: `20`

## Results

| Neuron | Torch ms | Generated ms | Speedup | Peak MB | State Max Error | Spike Mismatch Rate | Error |
|---|---:|---:|---:|---:|---:|---:|---|
| LIF | 6.824 | 0.097 | 70.66x | 202.0 | 1.788e-07 | 0.000e+00 |  |
| ALIF | 9.594 | 0.159 | 60.47x | 204.0 | 2.384e-06 | 0.000e+00 |  |
| Izhikevich | 18.009 | 0.354 | 50.93x | 204.0 | 7.629e-06 | 0.000e+00 |  |
