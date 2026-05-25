# Tiny SpikeGPT CPU Smoke

Date: 2026-05-25

Command:

```bash
uv run python examples/train_tiny_spikegpt.py \
  --device cpu \
  --compile off \
  --context-length 32 \
  --layers 2 \
  --embedding 64 \
  --batch 16 \
  --steps 80 \
  --log-every 20 \
  --sample-prompt spik \
  --sample-tokens 48
```

Setup:

| Setting | Value |
|---|---:|
| Device | CPU |
| Compile | off |
| Context length | 32 |
| Layers | 2 |
| Embedding | 64 |
| Batch | 16 |
| Steps | 80 |
| Learning rate | 0.003 |
| LIF threshold | 0.0 |
| Spike embedding | true |
| Vocab size | 23 |
| Parameters | 111,104 |

Results:

| Step | Loss | Emb Spike Rate | Mean Block Spike Rate | Step ms |
|---:|---:|---:|---:|---:|
| 1 | 3.138231 | 0.4889 | 0.4307 | 76.404 |
| 20 | 1.634518 | 0.5026 | 0.4571 | 44.842 |
| 40 | 0.573685 | 0.4835 | 0.4638 | 35.697 |
| 60 | 0.221309 | 0.4951 | 0.4722 | 42.801 |
| 80 | 0.142532 | 0.4944 | 0.4761 | 45.813 |

Summary:

| Metric | Value |
|---|---:|
| Initial loss | 3.138231 |
| Final loss | 0.142532 |
| Average step time | 48.653 ms |
| Post-warmup average step time | 48.302 ms |
| Steady-state average step time | 48.288 ms |

Generated sample:

```text
spiker explores fast training paths for those event 
```

Takeaways:

- The torch-native SpikeGPT-style model can overfit a tiny character language-model workload.
- Spike-rate diagnostics are nonzero and stable in this smoke, which confirms that the residual LIF activations are active rather than dead.
- This is a correctness and API smoke only. It is not a meaningful language modeling benchmark.
