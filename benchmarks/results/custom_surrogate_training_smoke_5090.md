<!-- benchmark_runner_name: custom_surrogate_training -->
<!-- benchmark_runner_started: 2026-05-24T14:38:23+00:00 -->
<!-- benchmark_runner_finished: 2026-05-24T14:38:26+00:00 -->
<!-- benchmark_runner_command: /home/danjs/code/spiker/.venv/bin/python3 -m spiker.benchmarks.custom_surrogate_training --device cuda --timesteps 16 --batch 8 --neurons 64 --warmup 1 --repeats 1 -->
# Custom Surrogate Training Benchmark

Compares custom LIF-shaped surrogate wrappers against the built-in surrogate LIF paths.

## Environment

- `generated_utc`: `2026-05-24T14:38:25+00:00`
- `device`: `cuda`
- `gpu`: `NVIDIA GeForce RTX 5090`
- `torch`: `2.12.0+cu130`
- `cuda_available`: `True`
- `cuda_version`: `13.0`
- `shape`: `T=16, B=8, N=64`
- `features`: `128`
- `checkpoint_size`: `25`
- `linear_bias`: `0.25`
- `decay`: `0.85`
- `threshold`: `1.0`
- `reset`: `0.0`
- `warmup`: `1`
- `repeats`: `1`

## Results

| Variant | Backend | Fwd+Bwd ms | Speedup vs Builtin Torch | Peak MB | Loss | Final Membrane Max Error | Loss Max Error | Input Grad Max Error | Weight Grad Max Error | Bias Grad Max Error | Error |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| builtin | torch | 6.200 | 1.00x | 0.4 | 0.31860352 | 0.000e+00 | 0.000e+00 | 0.000e+00 |  |  |  |
| custom_ir | torch | 5.990 | 1.04x | 0.4 | 0.31860352 | 0.000e+00 | 0.000e+00 | 0.000e+00 |  |  |  |
| builtin | triton_generated | 0.437 | 14.17x | 0.3 | 0.31860352 | 1.192e-07 | 0.000e+00 | 3.929e-10 |  |  |  |
| custom_ir | triton_generated | 0.404 | 15.36x | 0.3 | 0.31860352 | 1.192e-07 | 0.000e+00 | 3.929e-10 |  |  |  |
| linear_builtin | torch | 6.152 | 1.00x | 64.6 | 0.13574219 |  | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 |  |
| linear_custom_ir | torch | 6.041 | 1.02x | 64.7 | 0.13574219 |  | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 |  |
| linear_builtin | triton_generated_stream | 0.451 | 13.65x | 64.8 | 0.13806152 |  | 2.319e-03 | 3.011e-05 | 1.979e-03 | 2.527e-03 |  |
| linear_custom_ir | triton_generated_stream | 0.448 | 13.74x | 64.8 | 0.13806152 |  | 2.319e-03 | 3.011e-05 | 1.979e-03 | 2.527e-03 |  |
| rate_builtin | torch | 8.490 | 1.00x | 64.9 | 0.13684082 |  | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 |  |
| rate_custom_ir | torch | 8.338 | 1.02x | 65.1 | 0.13684082 |  | 0.000e+00 | 0.000e+00 | 0.000e+00 | 0.000e+00 |  |
| rate_builtin | triton_generated_rate | 0.510 | 16.66x | 65.1 | 0.13464355 |  | 2.197e-03 | 3.058e-05 | 2.046e-03 | 2.695e-03 |  |
| rate_custom_ir | triton_generated_rate | 0.461 | 18.40x | 65.1 | 0.13464355 |  | 2.197e-03 | 3.058e-05 | 2.046e-03 | 2.695e-03 |  |

## Built-In vs Custom Pairwise

| Surface | Backend | Loss Max Error | Final Membrane Max Error | Input Grad Max Error | Weight Grad Max Error | Bias Grad Max Error | Error |
|---|---|---:|---:|---:|---:|---:|---|
| cell | torch | 0.000e+00 | 0.000e+00 | 0.000e+00 |  |  |  |
| linear | torch | 0.000e+00 |  | 0.000e+00 | 0.000e+00 | 0.000e+00 |  |
| rate | torch | 0.000e+00 |  | 0.000e+00 | 0.000e+00 | 0.000e+00 |  |
| cell | triton_generated | 0.000e+00 | 0.000e+00 | 0.000e+00 |  |  |  |
| linear | triton_generated_stream | 0.000e+00 |  | 0.000e+00 | 0.000e+00 | 0.000e+00 |  |
| rate | triton_generated_rate | 0.000e+00 |  | 0.000e+00 | 0.000e+00 | 0.000e+00 |  |
