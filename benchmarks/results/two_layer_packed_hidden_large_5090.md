# Two-Layer Rate Recompute Benchmark

Compares the composed two-layer rate model with packed-hidden boundary storage and a whole-model recompute path.

## Environment

- `generated_utc`: `2026-05-24T19:01:10+00:00`
- `device`: `cuda`
- `checkpoint_size`: `balanced`
- `warmup`: `2`
- `repeats`: `5`

## Results

| T | B | F | H | C | Variant | ms | Peak MB | Error |
|---:|---:|---:|---:|---:|---|---:|---:|---|
| 25 | 128 | 784 | 512 | 10 | composed | 0.995 | 27.8 |  |
| 25 | 128 | 784 | 512 | 10 | packed_hidden | 0.923 | 33.3 |  |
| 25 | 128 | 784 | 512 | 10 | recompute | 34.926 | 104.7 |  |
| 25 | 128 | 784 | 1024 | 10 | composed | 1.037 | 110.9 |  |
| 25 | 128 | 784 | 1024 | 10 | packed_hidden | 1.105 | 121.9 |  |
| 25 | 128 | 784 | 1024 | 10 | recompute | 34.875 | 136.8 |  |
| 25 | 128 | 784 | 2048 | 10 | composed | 1.592 | 148.4 |  |
| 25 | 128 | 784 | 2048 | 10 | packed_hidden | 1.623 | 171.1 |  |
| 25 | 128 | 784 | 2048 | 10 | recompute | 35.767 | 201.2 |  |
| 50 | 128 | 784 | 512 | 10 | composed | 1.039 | 114.1 |  |
| 50 | 128 | 784 | 512 | 10 | packed_hidden | 1.026 | 121.8 |  |
| 50 | 128 | 784 | 512 | 10 | recompute | 85.599 | 130.7 |  |
| 50 | 128 | 784 | 1024 | 10 | composed | 1.580 | 147.5 |  |
| 50 | 128 | 784 | 1024 | 10 | packed_hidden | 1.616 | 161.8 |  |
| 50 | 128 | 784 | 1024 | 10 | recompute | 77.520 | 179.3 |  |
| 50 | 128 | 784 | 2048 | 10 | composed | 3.026 | 207.5 |  |
| 50 | 128 | 784 | 2048 | 10 | packed_hidden | 3.092 | 237.4 |  |
| 50 | 128 | 784 | 2048 | 10 | recompute | 69.681 | 271.2 |  |

## Interpretation

Whole-model PyTorch recompute was slower at every tested shape and did not reduce peak memory. Packed-hidden boundary storage was faster than the composed path on 2/6 shapes, but lower-memory on 0/6 shapes. This says packed storage alone is not enough for these shapes; the remaining lever is avoiding dense unpack/backward scratch.
