# SpikeGPT Training Benchmark

Generated: 2026-05-25T18:26:33.180853+00:00
Device: cuda (NVIDIA GeForce RTX 5090)
Shape: batch=4, context_length=64, layers=2, embedding=64, vocab_size=512
Warmup: 2; repeats: 5; seed: 0; compile=False; matmul_precision=high

| Path | Step time | Tokens/s | Peak memory | Loss | Error |
|---|---:|---:|---:|---:|---|
| eager | 227.220 | 1126.7 | 74.0 | 5.886495 |  |
