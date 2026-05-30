<!-- benchmark_runner_name: performance_frontier -->
<!-- benchmark_runner_started: 2026-05-24T15:20:51+00:00 -->
<!-- benchmark_runner_finished: 2026-05-24T15:20:57+00:00 -->
<!-- benchmark_runner_command: /home/danjs/code/myelin/.venv/bin/python3 -m myelin.benchmarks.performance_frontier --device cuda --timesteps 100 --batch 64 --features 128 --neurons 2048 --checkpoint-size 25 --matmul-precision high --warmup 3 --repeats 10 -->
# Performance Frontier

Canonical compiled-vs-Triton comparison for equal or explicit output contracts.

## Environment

- `generated_utc`: `2026-05-24T15:20:56+00:00`
- `device`: `cuda`
- `gpu`: `NVIDIA GeForce RTX 5090`
- `torch`: `2.12.0+cu130`
- `cuda_available`: `True`
- `cuda_version`: `13.0`
- `shape`: `T=100, B=64, F=128, N=2048`
- `dense_spike_or_current_mb`: `50.0`
- `checkpoint_size`: `25`
- `resolved_checkpoint_size`: `25`
- `compile_mode`: `reduce-overhead`
- `warmup`: `3`
- `repeats`: `10`

## Frontier

| Contract | Variant | Dense Spike Output | Fwd ms | Bwd ms | Fwd+Bwd ms | Bwd Increment MB | Speedup vs Compile | Compile Warmup ms | Note | Error |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
| hard LIF forward | torch.compile | yes | 0.140 |  |  |  | 1.00x | 1172.697 | forward-only baseline; compile time excluded |  |
| hard LIF forward | Triton fused-time | yes | 0.057 |  |  |  | 2.46x |  | single fused-time forward kernel |  |
| dense-output training | torch.compile materialized graph | no | 0.348 | 0.399 | 0.747 | 0.0 | 1.00x |  | whole scalar loss captured by Inductor |  |
| dense-output training | Triton checkpoint recompute | yes | 0.231 | 0.507 | 0.739 | 66.4 | 1.01x |  | returns dense spikes; checkpointed backward |  |
| rate/scalar training | Triton checkpoint rate output | no | 0.220 | 0.478 | 0.697 | 16.9 | 1.07x |  | avoids dense spike output; handwritten backward |  |
| rate/scalar training | Generated Triton checkpoint rate output | no | 0.219 | 0.499 | 0.718 | 16.9 | 1.04x |  | avoids dense spike output; generated backward chunk |  |

## Interpretation

- Forward-only fused-time Triton is the clean launch-overhead win.
- Dense-output training is a close fight because `torch.compile` sees the whole loss.
- Triton should win by changing the contract: rate/scalar outputs, packing, and sparse comms.
