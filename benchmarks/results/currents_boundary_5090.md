# Currents Boundary Check

Compile time is excluded from latency measurements. This compares whether
the PyTorch workload explicitly materializes `[T, B, N]` currents before
the recurrent loop or computes `[B, N]` currents inside the timestep loop.

## Environment

- `generated_utc`: `2026-05-24T11:13:55+00:00`
- `device`: `cuda`
- `gpu`: `NVIDIA GeForce RTX 5090`
- `torch`: `2.12.0+cu130`
- `cuda_available`: `True`
- `cuda_version`: `13.0`
- `features`: `128`
- `warmup`: `1`
- `repeats`: `3`
- `compile_mode`: `reduce-overhead`
- `compile_fullgraph`: `True`
- `matmul_precision`: `high`

## Results

| T | Batch | F | N | Workload | Expected Currents MB | Eager Fwd+Bwd ms | Compiled Fwd+Bwd ms | Speedup | Eager Fwd Peak MB | Compiled Fwd Peak MB | Eager Fwd Increment MB | Compiled Fwd Increment MB | Eager Increment / Currents | Compiled Increment / Currents | Eager Bwd Peak MB | Compiled Bwd Peak MB | Compile Warmup ms | Error |
|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 100 | 64 | 128 | 2048 | Materialized currents | 50.0 | 32.463 | 3.025 | 10.73x | 320.1 | 6.1 | 250.0 | 0.0 | 5.00x | 0.00x | 320.1 | 6.1 | 1324.301 |  |
| 100 | 64 | 128 | 2048 | Per-timestep matmul | 50.0 | 42.207 | 6.309 | 6.69x | 271.6 | 6.1 | 200.5 | 0.0 | 4.01x | 0.00x | 271.6 | 6.1 | 645.449 |  |

## Takeaway

`torch.compile` can reduce the materialized-looking PyTorch graph to a small peak allocation, so source-level `currents = matmul(...)` is not proof that `[T, B, N]` survives as a distinct runtime allocation.
The per-timestep graph remains useful as a semantic boundary check: it forces current production inside the recurrence, while the materialized graph lets Inductor decide whether to eliminate or rematerialize the temporary.
