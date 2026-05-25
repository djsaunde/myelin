# Custom Neuron Module Variant Benchmark

- `generated_utc`: `2026-05-24T12:11:06+00:00`
- `device`: `cuda`
- `gpu`: `NVIDIA GeForce RTX 5090`
- `torch`: `2.12.0+cu130`
- `cuda_version`: `13.0`
- `shape`: `T=100, B=64, N=2048`
- `warmup`: `5`
- `repeats`: `20`

## Results

| Variant | Backend | Forward ms | Speedup vs torch | Peak MB | State Max Error | Spike Mismatch Rate |
|---|---|---:|---:|---:|---:|---:|
| alif | torch | 38.939 | 1.00x | 254.0 | 0.000e+00 | 0.000e+00 |
| alif | auto | 0.091 | 428.18x | 204.0 | 7.892e-04 | 3.052e-07 |
| alif | triton_generated | 0.086 | 454.28x | 204.0 | 7.892e-04 | 3.052e-07 |
| refractory_lif | torch | 55.036 | 1.00x | 254.0 | 0.000e+00 | 0.000e+00 |
| refractory_lif | auto | 0.094 | 583.91x | 204.0 | 1.192e-07 | 0.000e+00 |
| refractory_lif | triton_generated | 0.093 | 592.81x | 204.0 | 1.192e-07 | 0.000e+00 |

## Notes

- Both variants use public `NeuronBuilder` IR wrapped in `CustomNeuronCell` and `TimeUnroll`.
- `auto` dispatches to the generic generated Triton forward path on CUDA when Triton is installed.
- The refractory-LIF variant encodes counter-like state with tensor predicates and `where`, avoiding Python data-dependent control flow.
