# Fused Synapse + LIF Triton Backward

RTX 5090, CUDA, `T=25`, `B=64`, `F=128`, `N=2048`, fast-sigmoid surrogate,
hard-forward LIF, loss = `spikes.mean() + 0.1 * final_membrane.square().mean()`.
Timing excludes first-use compile/warmup. Benchmark command:

```bash
uv run python -m spiker.benchmarks.surrogate_backend \
  --device cuda --timesteps 25 --batch 64 --features 128 --neurons 2048 \
  --warmup 3 --repeats 10 --matmul-precision high
```

| Path | Fwd+Bwd ms |
|---|---:|
| Eager materialized currents | 9.903 |
| PyTorch streamed custom autograd | 10.020 |
| Triton LIF over materialized currents | 0.556 |
| `torch.compile` materialized graph | 0.453 |
| Triton fused synapse + LIF | 0.496 |

| Path | Forward Peak MB | Backward Peak MB |
|---|---:|---:|
| Eager materialized currents | 158.3 | 158.3 |
| PyTorch streamed custom autograd | 135.3 | 135.3 |
| Triton LIF over materialized currents | 110.8 | 122.8 |
| `torch.compile` materialized graph | 7.8 | 7.8 |
| Triton fused synapse + LIF | 98.8 | 112.8 |

Notes:

- The Triton path computes dense currents inside the forward kernel instead of
  materializing `[T, B, N]` currents before LIF.
- The backward path is now all Triton kernels. When `dinputs` are not needed,
  it accumulates `dweight`/`dbias` during the reverse recurrence instead of
  materializing a `[T, B, N]` current-gradient scratch. When `dinputs` are
  needed, the fallback still materializes that scratch.
- Backward peak is still far from the `torch.compile` result because the
  current Triton training path saves the full pre-reset and spike traces for
  BPTT.
- The next optimization target is eliminating or tiling that scratch while
  accumulating `dweight`.
