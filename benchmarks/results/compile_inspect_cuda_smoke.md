# Torch Compile Inspection Smoke

This inspection dumps `torch.compile`/Inductor artifacts for the pure PyTorch
surrogate-rate workloads. It uses a small CUDA shape so the generated
`output_code.py` files are easy to inspect.

## Environment

- `generated_utc`: `2026-05-24T13:10:54+00:00`
- `device`: `cuda`
- `gpu`: `NVIDIA GeForce RTX 5090`
- `torch`: `2.12.0+cu130`
- `cuda_version`: `13.0`
- `shape`: `T=16, B=8, F=16, N=32`
- `compile_mode`: `reduce-overhead`
- `force_disable_caches`: `True`

## Results

| Workload | Graph Count | Graph Breaks | Op Count | Compile First Run ms | Output Code Files | Triton JIT Kernels | `extern_kernels.mm` | CUDA Alloc Sites | `reinterpret_tensor` Calls | Explicit `del` Calls | Reuse Assignments |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Materialized currents | 1 | 0 | 197 | 4971.488 | 2 | 2 | 2 | 24 | 19 | 17 | 0 |
| Per-timestep matmul | 1 | 0 | 212 | 5719.663 | 2 | 3 | 16 | 47 | 32 | 56 | 27 |

## Inspection Notes

Materialized-source forward code performs one flattened matmul:

```python
buf0 = empty_strided_cuda((128, 32), (32, 1), torch.float32)
extern_kernels.mm(reinterpret_tensor(primals_1, (128, 16), (16, 1), 0), primals_2, out=buf0)
```

It returns that full currents buffer, plus per-timestep recurrent state buffers,
to the compiled backward:

```python
return (buf33, reinterpret_tensor(buf0, (8, 32), (32, 1), 0), buf2, ..., buf30, ...)
```

Looped-source forward code emits one matmul per timestep and returns those
per-timestep current/state buffers. It also shows explicit reuse:

```python
buf21 = buf20; del buf20  # reuse
buf24 = buf21; del buf21  # reuse
```

Backward code for the materialized workload allocates one contiguous gradient
scratch and aliases each timestep view into it before the final weight-gradient
matmul:

```python
buf46 = empty_strided_cuda((128, 32), (32, 1), torch.float32)
buf44 = reinterpret_tensor(buf46, (8, 32), (32, 1), 3584)  # alias
...
extern_kernels.mm(permute, buf46, out=buf47)
```

## Takeaway

This smoke inspection does **not** show Inductor avoiding all dense
time-by-neuron storage for dense-output BPTT. It materializes or returns
currents/state tensors needed by backward, but it gets strong memory behavior
from whole-graph capture, fusion, deletion, and buffer reuse/aliasing. The most
direct Triton lesson is to improve scratch planning and aliasing for dense
outputs, while still preferring rate/packed output contracts when users do not
need dense `[T, B, N]` spikes.

The inspection tool now disables compiler caches by default so fresh
`output_code.py` files are generated during inspection runs. Use
`--allow-compile-cache` when the debug files are already present and compile
latency matters more than fresh artifacts.
