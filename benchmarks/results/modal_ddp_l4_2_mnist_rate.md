# Modal DDP MNIST Rate Smoke

Date: 2026-05-25

Modal run: https://modal.com/apps/djsaunde/main/ap-PG6q450NuLKD96vcr4GLhC

Command:

```bash
uv run modal run modal/train_distributed.py \
  --target ddp \
  --timesteps 25 \
  --hidden 128 \
  --epochs 3 \
  --train-limit 8192 \
  --test-limit 2048 \
  --batch 128
```

Setup:

| Setting | Value |
|---|---:|
| GPUs | 2 x NVIDIA L4 |
| World size | 2 |
| Model | RateSNNClassifier MLP |
| Encoding | poisson |
| Timesteps | 25 |
| Hidden size | 128 |
| Train examples | 8192 |
| Test examples | 2048 |
| Batch per rank | 128 |
| Epochs | 3 |
| Learning rate | 0.003 |
| Grad clip | 0.1 |
| Compile policy | off |
| Backend | auto -> triton |
| Parameters | 101,770 |

Results:

| Metric | Value |
|---|---:|
| Initial train loss | 2.302585 |
| Initial train accuracy | 0.1172 |
| Initial validation loss | 2.302585 |
| Initial validation accuracy | 0.0850 |
| Final logged train loss | 1.549112 |
| Final logged train accuracy | 0.9219 |
| Final logged validation loss | 1.560909 |
| Final logged validation accuracy | 0.8936 |
| Final test loss | 1.570481 |
| Final test accuracy | 0.8809 |
| Total training seconds | 9.395 |
| First logged step time | 6228.160 ms |
| Average step time | 72.843 ms |
| Post-warmup average step time | 8.050 ms |
| Steady-state average step time | 7.647 ms |
| Peak CUDA memory | 44.731 MB |

Takeaways:

- The distributed DDP path learns on a cheap 2 x L4 Modal run: validation accuracy moved from 8.50% to 89.36%, with final test accuracy at 88.09%.
- The first logged step is dominated by startup, dataloader, NCCL, and backend initialization overhead even with `--compile off`; steady-state timing is the useful throughput number here.
- Accuracy is below the tuned local single-GPU examples because this is a small rate-MLP smoke run over 8192 train examples for 3 epochs, not a tuned full-data training job.
- Peak memory is modest at 44.731 MB, so this path is suitable for cheap distributed CI or remote smoke testing.
