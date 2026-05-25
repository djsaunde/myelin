<!-- benchmark_runner_name: performance_frontier -->
<!-- benchmark_runner_started: 2026-05-24T15:19:36+00:00 -->
<!-- benchmark_runner_finished: 2026-05-24T15:19:51+00:00 -->
<!-- benchmark_runner_command: /home/danjs/code/spiker/.venv/bin/python3 -m spiker.benchmarks.performance_frontier --device cuda --timesteps 16 --batch 8 --features 16 --neurons 64 --checkpoint-size 4 --warmup 1 --repeats 1 -->
# Performance Frontier

Canonical compiled-vs-Triton comparison for equal or explicit output contracts.

## Environment

- `generated_utc`: `2026-05-24T15:19:50+00:00`
- `device`: `cuda`
- `gpu`: `NVIDIA GeForce RTX 5090`
- `torch`: `2.12.0+cu130`
- `cuda_available`: `True`
- `cuda_version`: `13.0`
- `shape`: `T=16, B=8, F=16, N=64`
- `dense_spike_or_current_mb`: `0.0`
- `checkpoint_size`: `4`
- `resolved_checkpoint_size`: `4`
- `compile_mode`: `reduce-overhead`
- `warmup`: `1`
- `repeats`: `1`

## Frontier

| Contract | Variant | Dense Spike Output | Fwd ms | Bwd ms | Fwd+Bwd ms | Bwd Increment MB | Speedup vs Compile | Compile Warmup ms | Note | Error |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
| hard LIF forward | torch.compile | yes | 0.278 |  |  |  | 1.00x | 3572.949 | forward-only baseline; compile time excluded |  |
| hard LIF forward | Triton fused-time | yes | 0.113 |  |  |  | 2.45x |  | single fused-time forward kernel |  |
| dense-output training | torch.compile materialized graph | no | 3.061 | 1.460 | 4.521 | 0.0 | 1.00x |  | whole scalar loss captured by Inductor |  |
| dense-output training | Triton checkpoint recompute | yes | 0.367 | 0.867 | 1.234 | 0.1 | 3.66x |  | returns dense spikes; checkpointed backward |  |
| rate/scalar training | Triton checkpoint rate output | no | 0.360 | 0.852 | 1.212 | 0.0 | 3.73x |  | avoids dense spike output; handwritten backward |  |
| rate/scalar training | Generated Triton checkpoint rate output | no | 0.363 | 0.619 | 0.982 | 0.0 | 4.60x |  | avoids dense spike output; generated backward chunk |  |

## Interpretation

- Forward-only fused-time Triton is the clean launch-overhead win.
- Dense-output training is a close fight because `torch.compile` sees the whole loss.
- Triton should win by changing the contract: rate/scalar outputs, packing, and sparse comms.
