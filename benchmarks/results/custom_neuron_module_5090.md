# Custom Neuron Module Benchmark

## Environment

- `generated_utc`: `2026-05-24T13:50:45+00:00`
- `device`: `cuda`
- `gpu`: `NVIDIA GeForce RTX 5090`
- `torch`: `2.12.0+cu130`
- `cuda_available`: `True`
- `cuda_version`: `13.0`
- `variant`: `all`
- `shape`: `T=100, B=64, N=2048`
- `warmup`: `5`
- `repeats`: `20`

## Workload

- `alif`: Custom two-state ALIF-style `NeuronIR`
- `refractory_lif`: Custom refractory-LIF `NeuronIR` with counter-like state

Each variant is wrapped in `CustomNeuronCell` and `TimeUnroll`.

## Results

| Variant | Backend | Forward ms | Speedup vs torch | Peak MB | State Max Error | Spike Mismatch Rate | Error |
|---|---|---:|---:|---:|---:|---:|---|
| alif | torch | 30.245 | 1.00x | 254.0 | 0.000e+00 | 0.000e+00 |  |
| alif | auto | 0.077 | 393.62x | 204.0 | 7.153e-07 | 0.000e+00 |  |
| alif | triton_generated | 0.076 | 396.54x | 204.0 | 7.153e-07 | 0.000e+00 |  |
| refractory_lif | torch | 50.495 | 1.00x | 254.0 | 0.000e+00 | 0.000e+00 |  |
| refractory_lif | auto | 0.094 | 535.01x | 204.0 | 1.788e-07 | 0.000e+00 |  |
| refractory_lif | triton_generated | 0.084 | 600.54x | 204.0 | 1.788e-07 | 0.000e+00 |  |
