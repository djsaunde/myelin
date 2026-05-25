# Modal DDP vs FSDP2 Resident Memory

Date: 2026-05-25

Commands:

```bash
uv run modal run modal/train_distributed.py \
  --target ddp \
  --timesteps 10 \
  --hidden 4096 \
  --epochs 1 \
  --train-limit 512 \
  --test-limit 256 \
  --batch 64

uv run modal run modal/train_distributed.py \
  --target fsdp2 \
  --timesteps 10 \
  --hidden 4096 \
  --epochs 1 \
  --train-limit 512 \
  --test-limit 256 \
  --batch 64
```

Runs:

| Path | Modal run |
|---|---|
| DDP | https://modal.com/apps/djsaunde/main/ap-Rsfqsezwxit8l8qSEobc8c |
| FSDP2 | https://modal.com/apps/djsaunde/main/ap-Dx5rOa7nfOLseZDNzJZaFj |

Setup:

| Setting | Value |
|---|---:|
| GPUs | 2 x NVIDIA L4 |
| World size | 2 |
| Model | RateReadoutClassifier |
| Hidden size | 4096 |
| Parameters | 3,256,330 |
| Timesteps | 10 |
| Batch per rank | 64 |
| Train examples | 512 |
| Test examples | 256 |
| Compile policy | off |

Results:

| Path | Local params | Local optimizer state | Peak CUDA | Steady-state step |
|---|---:|---:|---:|---:|
| DDP | 12.422 MB | 24.844 MB | 83.952 MB | 12.047 ms |
| FSDP2 | 6.211 MB | 12.422 MB | 68.620 MB | 12.999 ms |

Ratios:

| Quantity | FSDP2 / DDP |
|---|---:|
| Local params | 0.500 |
| Local optimizer state | 0.500 |
| Peak CUDA | 0.817 |
| Steady-state step | 1.079 |

Takeaways:

- The expected `1 / world_size` local residency factor is visible for sharded parameters and Adam optimizer state.
- Peak CUDA memory does not fall by 2x because it also includes activations, temporary buffers, communication buffers, and runtime overhead that are not model-state shards.
- This is a small-model sanity check. FSDP2 should become more compelling as model-state memory dominates more of the peak.
