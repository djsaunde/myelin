# Compile-Visible Triton Boundary

Measures an experimental `torch.library.triton_op` wrapper around the
checkpointed linear LIF rate-forward kernel.

## Environment

- `generated_utc`: `2026-05-24T17:07:28+00:00`
- `device`: `cuda`
- `gpu`: `NVIDIA GeForce RTX 5090`
- `torch`: `2.12.0+cu130`
- `cuda_available`: `True`
- `cuda_version`: `13.0`
- `shape`: `T=100, B=64, F=128, N=2048`
- `checkpoint_size`: `25`
- `resolved_checkpoint_size`: `25`
- `compile_mode`: `reduce-overhead`
- `warmup`: `3`
- `repeats`: `10`

## Results

| Path | ms | Peak MB | Graph Count | Graph Breaks | Note | Error |
|---|---:|---:|---:|---:|---|---|
| raw Python Triton wrapper + eager loss | 0.144 | 13.6 |  |  | existing forward wrapper; downstream loss is outside torch.compile |  |
| triton_op wrapper + eager loss | 0.167 | 13.6 | 1 | 0 | same kernel exposed through torch.library.triton_op |  |
| torch.compile(triton_op wrapper + loss) | 0.144 | 9.6 |  |  | tests whether Inductor can capture Triton launch plus downstream loss |  |
| existing Triton custom autograd rate training | 0.775 | 28.1 |  |  | current public rate-training path, including loss.backward() |  |
| triton_op registered-autograd rate training | 0.880 | 33.3 |  |  | compile-visible forward with registered custom-op backward |  |
| torch.compile(triton_op registered-autograd rate training) | 0.525 | 11.6 |  |  | compiled forward/loss with registered custom-op backward |  |
| torch.compile(public triton_compile rate training) | 0.511 | 11.6 |  |  | same compile-visible path through public backend='triton_compile' |  |
| torch.compile(public triton_compile rate training + bias) | 0.549 | 11.6 |  |  | same public backend with bias gradients enabled |  |
