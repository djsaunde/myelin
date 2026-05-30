# Surrogate Checkpoint/Recompute Oracle

RTX 5090, CUDA, `T=100`, `B=64`, `F=128`, `N=2048`, fast-sigmoid surrogate,
hard-forward LIF, `checkpoint_size=25`. Timing excludes first-use compile/warmup.
Benchmark command:

```bash
uv run python -m myelin.benchmarks.surrogate_backend \
  --device cuda --timesteps 100 --batch 64 --features 128 --neurons 2048 \
  --warmup 2 --repeats 5 --matmul-precision high --checkpoint-size 25
```

| Path | Fwd+Bwd ms | Forward Peak MB | Backward Peak MB |
|---|---:|---:|---:|
| Eager materialized currents | 34.437 | 424.1 | 424.1 |
| PyTorch streamed custom autograd | 35.891 | 326.1 | 326.1 |
| PyTorch checkpoint/recompute oracle | 43.243 | 180.1 | 180.1 |
| Triton LIF over materialized currents | 0.745 | 227.6 | 277.1 |
| Triton fused synapse + LIF | 0.718 | 178.1 | 229.6 |
| `torch.compile` materialized graph | 0.747 | 12.1 | 12.1 |

Notes:

- The checkpoint oracle stores chunk-start membrane states instead of the full
  pre-reset trace, then recomputes chunk traces during backward.
- It proves the memory direction for M3: saved recurrent traces can be reduced
  substantially before moving the mechanism into Triton.
- The Python oracle is slower than full-trace streaming because recompute runs
  in Python loops. The intended fast path is a Triton chunked backward.
