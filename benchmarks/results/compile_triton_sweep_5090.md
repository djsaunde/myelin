# Compile/Triton Rate Training Sweep

Compares regular compiled PyTorch scalar-loss training with the existing
Triton rate path and the public compile-visible `triton_compile` backend.

## Environment

- `generated_utc`: `2026-05-24T16:54:04+00:00`
- `device`: `cuda`
- `gpu`: `NVIDIA GeForce RTX 5090`
- `torch`: `2.12.0+cu130`
- `cuda_available`: `True`
- `cuda_version`: `13.0`
- `checkpoint_size`: `25`
- `compile_mode`: `reduce-overhead`
- `warmup`: `3`
- `repeats`: `10`

## Results

| T | B | F | N | Path | ms | Peak MB | Compile Warmup ms | Error |
|---:|---:|---:|---:|---|---:|---:|---:|---|
| 50 | 64 | 128 | 1024 | regular torch.compile scalar-loss training | 0.478 | 2.6 | 1634.7 |  |
| 50 | 64 | 128 | 1024 | existing Triton rate training | 0.587 | 11.3 |  |  |
| 50 | 64 | 128 | 1024 | torch.compile public backend="triton_compile" | 0.417 | 4.1 | 82.6 |  |
| 100 | 64 | 128 | 2048 | regular torch.compile scalar-loss training | 0.852 | 5.1 | 2623.7 |  |
| 100 | 64 | 128 | 2048 | existing Triton rate training | 1.048 | 23.6 |  |  |
| 100 | 64 | 128 | 2048 | torch.compile public backend="triton_compile" | 0.522 | 8.1 | 136.8 |  |
| 200 | 64 | 128 | 2048 | regular torch.compile scalar-loss training | 16.677 | 8.3 | 4694.8 |  |
| 200 | 64 | 128 | 2048 | existing Triton rate training | 0.882 | 28.8 |  |  |
| 200 | 64 | 128 | 2048 | torch.compile public backend="triton_compile" | 0.730 | 11.3 | 131.5 |  |
