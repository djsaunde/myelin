<!-- benchmark_runner_name: compile_triton_sweep -->
<!-- benchmark_runner_started: 2026-05-24T16:47:45+00:00 -->
<!-- benchmark_runner_finished: 2026-05-24T16:47:56+00:00 -->
<!-- benchmark_runner_command: /home/danjs/code/myelin/.venv/bin/python3 -m myelin.benchmarks.compile_triton_sweep --device cuda --shape 16,8,16,64 --checkpoint-size 4 --warmup 1 --repeats 1 -->
# Compile/Triton Rate Training Sweep

Compares regular compiled PyTorch scalar-loss training with the existing
Triton rate path and the public compile-visible `triton_compile` backend.

## Environment

- `generated_utc`: `2026-05-24T16:47:54+00:00`
- `device`: `cuda`
- `gpu`: `NVIDIA GeForce RTX 5090`
- `torch`: `2.12.0+cu130`
- `cuda_available`: `True`
- `cuda_version`: `13.0`
- `checkpoint_size`: `4`
- `compile_mode`: `reduce-overhead`
- `warmup`: `1`
- `repeats`: `1`

## Results

| T | B | F | N | Path | ms | Peak MB | Compile Warmup ms | Error |
|---:|---:|---:|---:|---|---:|---:|---:|---|
| 16 | 8 | 16 | 64 | regular torch.compile scalar-loss training | 0.759 | 0.0 | 4935.3 |  |
| 16 | 8 | 16 | 64 | existing Triton rate training | 1.020 | 0.1 |  |  |
| 16 | 8 | 16 | 64 | torch.compile public backend="triton_compile" | 0.714 | 0.0 | 866.7 |  |
