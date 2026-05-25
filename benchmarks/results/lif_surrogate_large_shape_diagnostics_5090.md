# Large-Shape Surrogate Diagnostics

Compile time is excluded from latency measurements. Rows are flushed as they complete.

## Environment

- `generated_utc`: `2026-05-23T04:08:52+00:00`
- `device`: `cuda`
- `gpu`: `NVIDIA GeForce RTX 5090`
- `torch`: `2.12.0+cu130`
- `cuda_version`: `13.0`
- `features`: `128`
- `warmup`: `1`
- `repeats`: `3`
- `compile_fullgraph`: `True`
- `compile_time_included`: `False`

## Compile-Friendliness Audit

- Workloads use fixed-shape tensor inputs and static Python loops over `inputs.unbind(dim=0)`.
- There is no data-dependent Python control flow in the timestep loop.
- Public shape validation is outside the hot `lif_unroll` loop.
- Scalar neuron parameters are Python floats, so compiled graphs specialize/guard on them.
- Dense surrogate variants materialize `[T, B, N]` currents before the recurrent loop.
- `fullgraph=True` is used so graph breaks fail instead of silently benchmarking a partial compile.

## Results

| Workload | Compile Mode | Matmul Precision | T | B | F | N | Compiled Fwd+Bwd ms | Compiled Fwd-only ms | Compiled Bwd ms | Compiled Bwd Peak MB | Eager Fwd+Bwd ms | Eager Bwd Peak MB | Fwd+Bwd Speedup | Compile Error |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| surrogate-spike | reduce-overhead | highest | 100 | 256 | 128 | 8192 | 8.541 | 22.047 | 5.266 | 24.5 | 25.639 | 3292.5 | 3.00x |  |
