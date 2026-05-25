# MNIST Rate Readout Memory Comparison

Runs the dense and low-memory rate-readout MNIST examples with matched knobs.
This is a short smoke comparison intended to verify the example-level memory
metric and the practical effect of avoiding dense output spikes.

## Environment

- `generated_utc`: `2026-05-24T11:21:01+00:00`
- `device`: `cuda`
- `gpu`: `NVIDIA GeForce RTX 5090`
- `timesteps`: `10`
- `encoding`: `poisson`
- `batch`: `128`
- `hidden`: `128`
- `epochs`: `1`
- `grad_clip`: `0.1`
- `train_limit`: `512`
- `test_limit`: `512`
- `compile_spiker_only`: `True`
- `matmul_precision`: `highest`

## Results

| Variant | Final Test Loss | Final Test Acc | Total s | Peak CUDA MB | Avg Step ms | Steady Step ms | Status |
|---|---:|---:|---:|---:|---:|---:|---:|
| dense | 2.265670 | 0.3672 | 8.672 | 45.3 | 1448.881 | 1.425 | ok |
| rate | 2.265670 | 0.3730 | 3.653 | 18.7 | 721.254 | 1.168 | ok |

## Takeaway

At this smoke scale, the rate-readout example keeps accuracy essentially
matched while reducing peak CUDA memory by 26.6 MB, a 58.7% reduction relative
to the dense output-spike example.
