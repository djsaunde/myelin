# Triton Checkpoint/Recompute Slice

RTX 5090, CUDA, `T=100`, `B=64`, `F=128`, `N=2048`, fast-sigmoid surrogate,
hard-forward LIF, `checkpoint_size=25`.

Benchmark command:

```bash
uv run python -m spiker.benchmarks.surrogate_backend \
  --device cuda --timesteps 100 --batch 64 --features 128 --neurons 2048 \
  --warmup 2 --repeats 5 --matmul-precision high --checkpoint-size 25
```

| Path | Fwd+Bwd ms | Forward Peak MB | Backward Peak MB |
|---|---:|---:|---:|
| Eager materialized currents | 40.351 | 425.1 | 425.1 |
| PyTorch streamed custom autograd | 37.946 | 327.1 | 327.1 |
| PyTorch checkpoint/recompute oracle | 37.080 | 181.1 | 181.1 |
| Triton LIF over materialized currents | 0.677 | 228.6 | 278.1 |
| Triton fused synapse full trace | 0.717 | 179.1 | 230.6 |
| Triton checkpoint recompute | 0.817 | 132.1 | 158.6 |
| `torch.compile` materialized graph | 0.755 | 13.1 | 13.1 |

Notes:

- This is the first Triton checkpoint/recompute correctness slice for the
  common case where `dinputs` are not needed.
- It stores chunk-start membranes instead of full pre-reset traces.
- The checkpoint backward now recomputes each chunk into a bounded scratch
  buffer and then runs a reverse chunk kernel, avoiding the original local
  `O(checkpoint_size^2)` replay. This keeps the lower memory footprint while
  making latency competitive with the full-trace Triton path.
- The memory numbers above are from the shared benchmark harness, so they
  include the same allocator baseline as the other surrogate backend results.
