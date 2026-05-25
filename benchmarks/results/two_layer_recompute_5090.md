# Two-Layer Rate Recompute Benchmark

Compares the composed two-layer rate model with packed-hidden boundary storage and a whole-model recompute path.

## Environment

- `generated_utc`: `2026-05-24T18:49:04+00:00`
- `device`: `cuda`
- `checkpoint_size`: `balanced`
- `warmup`: `2`
- `repeats`: `5`

## Results

| T | B | F | H | C | Variant | ms | Peak MB | Error |
|---:|---:|---:|---:|---:|---|---:|---:|---|
| 10 | 128 | 784 | 128 | 10 | composed | 1.297 | 6.6 |  |
| 10 | 128 | 784 | 128 | 10 | packed_hidden | 1.365 | 7.6 |  |
| 10 | 128 | 784 | 128 | 10 | recompute | 19.149 | 73.3 |  |
| 25 | 128 | 784 | 128 | 10 | composed | 0.843 | 78.9 |  |
| 25 | 128 | 784 | 128 | 10 | packed_hidden | 0.814 | 80.2 |  |
| 25 | 128 | 784 | 128 | 10 | recompute | 29.870 | 82.1 |  |
| 50 | 128 | 784 | 128 | 10 | composed | 0.933 | 91.8 |  |
| 50 | 128 | 784 | 128 | 10 | packed_hidden | 0.901 | 93.5 |  |
| 50 | 128 | 784 | 128 | 10 | recompute | 61.186 | 95.7 |  |

## Interpretation

Whole-model PyTorch recompute was slower at every tested shape and did not reduce peak memory. Packed-hidden boundary storage was faster than the composed path on 2/3 shapes, but lower-memory on 0/3 shapes. This says packed storage alone is not enough at this MNIST-hidden-size shape; the remaining lever is avoiding dense unpack/backward scratch or using a larger hidden/readout boundary.
