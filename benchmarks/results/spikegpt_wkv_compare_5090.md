# SpikeGPT WKV Compile/Correctness Comparison

Generated: 2026-05-30T16:36:12+00:00
Device: cuda (NVIDIA GeForce RTX 5090)
torch: 2.12.0+cu130
batch=8, channels=128, chunk_size=32, matmul_precision=highest, repeats=10, compile_mode=default, fullgraph=True

Timings include forward+backward. `compile_first_ms` includes compilation.

| Variant | T | Fwd err | Bwd err | Eager ms | Compile+1st ms | Steady ms | Error |
|---|---:|---:|---:|---:|---:|---:|---|
| reference_loop | 16 | 0.00e+00 | 0.00e+00 | 16.272 | 18751.575 | 0.756 |  |
| scan | 16 | 0.00e+00 | 3.82e+01 | 133.240 |  |  | compile BackendCompilerFailed: backend='inductor' raised: |
| parallel | 16 | 7.15e-07 | 7.63e-06 | 1.830 | 3309.183 | 0.873 |  |
| chunked | 16 | 7.15e-07 | 7.63e-06 | 1.934 | 3006.001 | 0.884 |  |
| reference_loop | 32 | 0.00e+00 | 0.00e+00 | 28.577 | 44530.324 | 1.037 |  |
| scan | 32 | 0.00e+00 | 6.48e+01 | 200.733 |  |  | compile BackendCompilerFailed: backend='inductor' raised: |
| parallel | 32 | 7.15e-07 | 1.53e-05 | 1.800 | 3078.871 | 0.905 |  |
| chunked | 32 | 7.15e-07 | 1.53e-05 | 1.936 | 2848.095 | 0.841 |  |
| reference_loop | 64 | 0.00e+00 | 0.00e+00 | 60.443 | 103355.121 | 1.576 |  |
| scan | 64 | 0.00e+00 | 1.39e+02 | 268.931 |  |  | compile BackendCompilerFailed: backend='inductor' raised: |
| parallel | 64 | 7.15e-07 | 4.58e-05 | 1.711 | 2976.643 | 0.897 |  |
| chunked | 64 | 5.96e-07 | 4.58e-05 | 4.225 | 5029.490 | 1.468 |  |
| reference_loop | 128 | 0.00e+00 | 0.00e+00 | 129.970 |  |  | compile skipped (slow) |
| scan | 128 | 0.00e+00 | 2.92e+02 | 270.129 |  |  | compile BackendCompilerFailed: backend='inductor' raised: |
| parallel | 128 | 8.34e-07 | 7.63e-05 | 1.947 | 4447.452 | 1.074 |  |
| chunked | 128 | 9.54e-07 | 7.63e-05 | 8.608 | 11953.165 | 2.060 |  |
| reference_loop | 256 | 0.00e+00 | 0.00e+00 | 261.176 |  |  | compile skipped (slow) |
| scan | 256 | 0.00e+00 | 5.35e+02 | 457.814 |  |  | compile BackendCompilerFailed: backend='inductor' raised: |
| parallel | 256 | 7.15e-07 | 2.44e-04 | 1.754 | 3693.086 | 0.963 |  |
| chunked | 256 | 7.15e-07 | 2.75e-04 | 17.774 | 21113.264 | 4.042 |  |
| reference_loop | 512 | 0.00e+00 | 0.00e+00 | 491.839 |  |  | compile skipped (slow) |
| scan | 512 | 0.00e+00 | 1.24e+03 | 786.484 |  |  | compile BackendCompilerFailed: backend='inductor' raised: |
| parallel | 512 | 9.54e-07 | 1.10e-03 | 4.400 | 3544.034 | 1.425 |  |
| chunked | 512 | 7.15e-07 | 1.10e-03 | 35.930 | 45915.417 | 7.173 |  |

## Notes

- `compile_first_ms` is a cold compile (Inductor caches disabled via `--cold-compile`);
  `torch._dynamo.reset()` runs before each row so compiles are independent.
- Timings include forward+backward. `matmul_precision=highest` (no TF32); under `high`/TF32
  the matrix-form backward error rises to ~1e-2 because the intra-span contraction is a matmul.

## Verdict

- **reference_loop**: correct, but cold compile is catastrophic and super-linear in `T`
  (18.8s -> 44.5s -> 103s at T=16/32/64) because Dynamo unrolls the per-step recurrence.
  Impractical to compile beyond short contexts.
- **scan** (`torch._higher_order_ops.scan`): exact forward, but **wrong gradients eagerly**
  (bwd err 38-1240) **and fails Inductor compilation** (`scan might be aliasing`) in torch
  2.12. Not viable here.
- **parallel** (O(T^2) decay-matrix, loop-free): correct (fwd ~7e-7, bwd <=1.1e-3 fp32) and
  **compile time is ~flat at ~3s for every T** -- the single biggest compile win. Cost is
  O(T^2 * C) intra-span memory/matmul, so it is memory-bound at long `T`.
- **chunked** (matrix per chunk, chunk=32): correct, O(T * chunk) memory, but compile time
  **grows with `T / chunk_size`** (3s -> 46s at T=512) since the chunk loop still unrolls under
  `fullgraph`. Eager runtime is also higher (many small kernels).

**Recommendation**: replace the per-step WKV loop with the loop-free `parallel` matrix form for
flat ~3s compiles at the repo's context lengths. If memory at long `T` becomes the constraint,
chunk it with a *large* chunk size (few chunks) to keep the unroll small, or compile per chunk
regionally so the chunk graph compiles once and is reused. `scan` is not usable in torch 2.12.
