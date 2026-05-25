<!-- benchmark_runner_name: compile_triton_boundary -->
<!-- benchmark_runner_started: 2026-05-24T17:10:29+00:00 -->
<!-- benchmark_runner_finished: 2026-05-24T17:10:39+00:00 -->
<!-- benchmark_runner_command: /home/danjs/code/spiker/.venv/bin/python3 -m spiker.benchmarks.compile_triton_boundary --device cuda --timesteps 16 --batch 8 --features 16 --neurons 64 --checkpoint-size 4 --warmup 1 --repeats 1 -->
# Compile-Visible Triton Boundary

Measures an experimental `torch.library.triton_op` wrapper around the
checkpointed linear LIF rate-forward kernel.

## Environment

- `generated_utc`: `2026-05-24T17:10:37+00:00`
- `device`: `cuda`
- `gpu`: `NVIDIA GeForce RTX 5090`
- `torch`: `2.12.0+cu130`
- `cuda_available`: `True`
- `cuda_version`: `13.0`
- `shape`: `T=16, B=8, F=16, N=64`
- `checkpoint_size`: `4`
- `resolved_checkpoint_size`: `4`
- `compile_mode`: `reduce-overhead`
- `warmup`: `1`
- `repeats`: `1`

## Results

| Path | ms | Peak MB | Graph Count | Graph Breaks | Note | Error |
|---|---:|---:|---:|---:|---|---|
| raw Python Triton wrapper + eager loss | 0.391 | 0.0 |  |  | existing forward wrapper; downstream loss is outside torch.compile |  |
| triton_op wrapper + eager loss | 0.241 | 0.0 | 1 | 0 | same kernel exposed through torch.library.triton_op |  |
| torch.compile(triton_op wrapper + loss) | 40.235 | 0.1 |  |  | tests whether Inductor can capture Triton launch plus downstream loss |  |
| existing Triton custom autograd rate training | 1.143 | 0.1 |  |  | current public rate-training path, including loss.backward() |  |
| triton_op registered-autograd rate training | 1.406 | 0.1 |  |  | compile-visible forward with registered custom-op backward |  |
| torch.compile(triton_op registered-autograd rate training) | 3.396 | 0.1 |  |  | compiled forward/loss with registered custom-op backward |  |
| torch.compile(public triton_compile rate training) | 2.937 | 0.1 |  |  | same compile-visible path through public backend='triton_compile' |  |
| torch.compile(public triton_compile rate training + bias) | 3.567 | 0.1 |  |  | same public backend with bias gradients enabled |  |
