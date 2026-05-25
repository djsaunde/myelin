# Torch Rate Backend Direct-Rate Benchmark

- `generated_utc`: `2026-05-24T12:14:29+00:00`
- `device`: `cuda`
- `gpu`: `NVIDIA GeForce RTX 5090`
- `torch`: `2.12.0+cu130`
- `shape`: `T=100, B=64, F=128, N=2048`
- `checkpoint_size`: `25`
- `surrogate`: `fast_sigmoid`
- `surrogate_slope`: `5.0`
- `repeats`: `5`

## Results

| Torch Backend Pattern | Fwd+Bwd ms | Peak MB | Delta vs Dense |
|---|---:|---:|---:|
| Dense spikes then `spikes.mean(dim=0)` | 46.289 | 175.1 | baseline |
| Direct checkpoint rate output | 42.638 | 105.2 | -70.0 MB |

## Notes

- Both rows force `backend="torch"` on CUDA to isolate the Torch backend behavior.
- The direct-rate path returns `[B, N]` rates and recomputes checkpoint chunks in backward, so it no longer materializes the public dense `[T, B, N]` spike tensor for rate-readout training.
- The Triton and generated Triton rate paths already used direct checkpoint rate outputs; this change makes the Torch fallback/baseline follow the same output contract.
