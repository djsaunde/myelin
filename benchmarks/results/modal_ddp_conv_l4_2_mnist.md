# Modal DDP Conv MNIST

Date: 2026-05-25

Modal run: https://modal.com/apps/djsaunde/main/ap-zZfR41zUdVNwLHhidhG50A

Command:

```bash
uv run modal run modal/train_distributed.py \
  --target ddp-conv \
  --timesteps 25 \
  --hidden 256 \
  --epochs 4 \
  --train-limit 60000 \
  --test-limit 10000 \
  --batch 128
```

Setup:

| Setting | Value |
|---|---:|
| GPUs | 2 x NVIDIA L4 |
| World size | 2 |
| Model | ConvMNISTSNN |
| Encoding | poisson |
| Timesteps | 25 |
| Hidden size | 256 |
| Train examples | 60000 |
| Test examples | 10000 |
| Batch per rank | 128 |
| Epochs | 4 |
| Learning rate | 0.003 |
| Grad clip | 1.0 |
| Compile policy | off |
| Synapse init | fan_in |
| Parameters | 409,034 |

Results:

| Step | Epoch | Loss | Train Acc | Val Loss | Val Acc | Step ms |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 2.303768 | 0.0625 | 2.274770 | 0.2836 | 1192.128 |
| 250 | 2 | 1.487345 | 0.9766 | 1.501023 | 0.9561 | 48.661 |
| 500 | 3 | 1.472981 | 0.9922 | 1.489190 | 0.9662 | 45.078 |
| 750 | 4 | 1.486067 | 0.9531 | 1.482629 | 0.9732 | 45.770 |
| 940 | 4 | 1.466147 | 0.9792 | 1.481148 | 0.9748 | 42.454 |

Summary:

| Metric | Value |
|---|---:|
| Final test loss | 1.475713 |
| Final test accuracy | 0.9829 |
| Total training seconds | 55.097 |
| Peak CUDA memory | 534.936 MB |
| Average step time | 47.941 ms |
| Post-warmup average step time | 46.722 ms |
| Steady-state average step time | 46.677 ms |

Takeaways:

- The stronger DDP conv path reaches 98.29% test accuracy on full MNIST in a cheap 2 x L4 Modal run.
- The run output did not show the previous NCCL barrier device warning after switching process-group init and barriers to explicit CUDA devices.
- This is now a better distributed-training artifact than the earlier rate-MLP smoke: the smoke proved plumbing, while this proves the path can train a useful model.
