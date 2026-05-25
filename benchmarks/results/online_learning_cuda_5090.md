# Online Learning CUDA Benchmark

Timing excludes warmup. Online variants compute local eligibility gradients;
BPTT variants build a PyTorch graph through the full surrogate recurrence.

## Environment

- `generated_utc`: `2026-05-24T13:07:26+00:00`
- `device`: `cuda`
- `gpu`: `NVIDIA GeForce RTX 5090`
- `torch`: `2.12.0+cu130`
- `shape`: `T=100, B=64, F=128, N=512`
- `bias`: `True`
- `seed`: `0`
- `warmup`: `5`
- `repeats`: `30`

## Results

| Variant | Update ms | Peak MB | Grad Weight Norm | Error |
|---|---:|---:|---:|---|
| Online LIF eligibility | 37.556 | 74.2 | 2399.719238 |  |
| Online ALIF eligibility | 53.922 | 111.2 | 1134.823364 |  |
| Online LIF custom surrogate IR | 38.097 | 74.2 | 2399.719238 |  |
| Online ALIF custom surrogate IR | 51.137 | 111.2 | 1134.823364 |  |
| BPTT LIF surrogate | 39.365 | 131.4 | 1085.648560 |  |
| BPTT ALIF surrogate | 49.219 | 131.6 | 462.622681 |  |

## Notes

- This benchmark is a functional comparison of the current dense PyTorch online
  eligibility oracle against surrogate BPTT references.
- Online LIF stores its eligibility trace as `[B, F]` because the documented
  local rule ignores reset/future-state gradients; this is faster and
  lower-memory than the LIF BPTT reference at this shape.
- Online ALIF now stores the neuron-indexed centered eligibility trace directly
  instead of keeping a separate adaptation eligibility trace plus centered
  temporaries. Its peak memory dropped from the previous 143.3 MB readout to
  111.2 MB on this shape, below the ALIF BPTT reference.
- The custom surrogate IR variants use a `SurrogateBuilder` derivative that is
  mathematically equivalent to the built-in fast-sigmoid derivative. They match
  gradient norms and memory. The generated Python derivative callable is cached
  by surrogate IR structure, so repeated online helper calls do not re-render or
  recompile the derivative. On this run the LIF custom path is within run noise
  of the built-in path and the ALIF custom path is slightly faster.
