# Training Breakdown

Breaks surrogate LIF training into projection, forward, and backward components.

## Environment

- `generated_utc`: `2026-05-25T16:06:54+00:00`
- `device`: `cuda`
- `gpu`: `NVIDIA GeForce RTX 5090`
- `torch`: `2.12.0+cu130`
- `cuda_available`: `True`
- `cuda_version`: `13.0`
- `shape`: `T=100, B=64, F=128, N=2048`
- `checkpoint_size`: `25`
- `resolved_checkpoint_size`: `25`
- `compile_mode`: `reduce-overhead`
- `no_compile`: `False`
- `warmup`: `3`
- `repeats`: `10`

## Results

| Component | Path | ms | Peak MB | Note | Error |
|---|---|---:|---:|---|---|
| dense projection | torch matmul | 0.067 | 86.1 | dense [T,B,F] x [F,N] projection only |  |
| full training | torch.compile materialized scalar loss | 0.575 | 5.1 | whole scalar loss inside compiled graph |  |
| checkpoint forward | Triton dense spikes | 0.127 | 160.6 | fused projection + LIF forward returning [T,B,N] spikes |  |
| checkpoint forward | Triton packed spikes | 0.118 | 112.2 | fused projection + LIF forward returning packed [T,B,ceil(N/32)] spikes |  |
| checkpoint forward | Triton rate output | 0.112 | 111.1 | fused projection + LIF forward returning [B,N] rates |  |
| checkpoint backward | Triton recurrent only | 0.195 | 121.6 | proxy for reverse recurrence; kernel grid still follows training layout |  |
| checkpoint backward | Triton recurrent + dweight | 0.237 | 122.6 | default training target when input gradients are not needed |  |
| checkpoint backward | Triton recurrent + dweight + dinput | 0.383 | 125.8 | training target when gradients through inputs are requested |  |
| checkpoint backward | Triton rate recurrent + dweight | 0.217 | 126.5 | rate-output backward avoids dense grad_spikes |  |
