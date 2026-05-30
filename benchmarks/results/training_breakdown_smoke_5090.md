<!-- benchmark_runner_name: training_breakdown -->
<!-- benchmark_runner_started: 2026-05-24T15:55:08+00:00 -->
<!-- benchmark_runner_finished: 2026-05-24T15:55:12+00:00 -->
<!-- benchmark_runner_command: /home/danjs/code/myelin/.venv/bin/python3 -m myelin.benchmarks.training_breakdown --device cuda --timesteps 16 --batch 8 --features 16 --neurons 64 --checkpoint-size 4 --warmup 1 --repeats 1 --no-compile -->
# Training Breakdown

Breaks surrogate LIF training into projection, forward, and backward components.

## Environment

- `generated_utc`: `2026-05-24T15:55:12+00:00`
- `device`: `cuda`
- `gpu`: `NVIDIA GeForce RTX 5090`
- `torch`: `2.12.0+cu130`
- `cuda_available`: `True`
- `cuda_version`: `13.0`
- `shape`: `T=16, B=8, F=16, N=64`
- `checkpoint_size`: `4`
- `resolved_checkpoint_size`: `4`
- `compile_mode`: `reduce-overhead`
- `no_compile`: `True`
- `warmup`: `1`
- `repeats`: `1`

## Results

| Component | Path | ms | Peak MB | Note | Error |
|---|---|---:|---:|---|---|
| dense projection | torch matmul | 0.111 | 32.0 | dense [T,B,F] x [F,N] projection only |  |
| full training | torch.compile materialized scalar loss |  |  | whole scalar loss inside compiled graph | compile disabled |
| checkpoint forward | Triton dense spikes | 0.062 | 32.1 | fused projection + LIF forward returning [T,B,N] spikes |  |
| checkpoint forward | Triton packed spikes | 0.141 | 32.1 | fused projection + LIF forward returning packed [T,B,ceil(N/32)] spikes |  |
| checkpoint forward | Triton rate output | 0.118 | 32.1 | fused projection + LIF forward returning [B,N] rates |  |
| checkpoint backward | Triton recurrent only | 0.276 | 32.1 | proxy for reverse recurrence; kernel grid still follows training layout |  |
| checkpoint backward | Triton recurrent + dweight | 0.347 | 32.1 | default training target when input gradients are not needed |  |
| checkpoint backward | Triton recurrent + dweight + dinput | 0.310 | 32.1 | training target when gradients through inputs are requested |  |
| checkpoint backward | Triton rate recurrent + dweight | 0.314 | 32.1 | rate-output backward avoids dense grad_spikes |  |
