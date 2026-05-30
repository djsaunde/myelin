# myelin vs snnTorch Matrix

Compares compiled myelin examples against eager snnTorch examples across timesteps.

## Environment

- `generated_utc`: `2026-05-25T16:01:18+00:00`
- `device`: `cuda`
- `encoding`: `poisson`
- `batch`: `128`
- `hidden`: `128`
- `epochs`: `1`
- `train_limit`: `1024`
- `test_limit`: `1024`
- `compile_myelin_only`: `True`
- `matmul_precision`: `highest`

## Results

| T | Variant | Final Test Loss | Final Test Acc | Total s | Peak CUDA MB | Avg Step ms | Steady Step ms | Compiled | Status |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 10 | rate | 2.136011 | 0.6055 | 4.562 | 18.7 | 492.807 | 1.174 | True | ok |
| 10 | conv | 2.039137 | 0.4727 | 3.605 | 382.8 | 330.265 | 3.064 | True | ok |
| 10 | snntorch_dense | 1.966508 | 0.6240 | 1.063 | 83.2 | 85.446 | 18.910 | False | ok |
| 10 | snntorch_conv | 2.158583 | 0.5312 | 1.230 | 483.6 | 109.899 | 20.146 | False | ok |
| 25 | rate | 2.137167 | 0.6182 | 3.524 | 44.0 | 362.443 | 2.422 | True | ok |
| 25 | conv | 2.231659 | 0.2383 | 4.120 | 897.0 | 340.103 | 5.034 | True | ok |
| 25 | snntorch_dense | 1.976218 | 0.6465 | 1.370 | 108.5 | 106.900 | 43.277 | False | ok |
| 25 | snntorch_conv | 2.171570 | 0.2734 | 1.628 | 1102.8 | 139.067 | 51.411 | False | ok |
| 50 | rate | 2.178982 | 0.6289 | 3.574 | 86.3 | 368.122 | 2.438 | True | ok |
| 50 | conv | 2.258209 | 0.2412 | 5.543 | 1755.5 | 426.170 | 8.568 | True | ok |
| 50 | snntorch_dense | 1.966261 | 0.6641 | 1.942 | 150.9 | 154.518 | 90.328 | False | ok |
| 50 | snntorch_conv | 2.158333 | 0.5527 | 2.263 | 2133.2 | 192.611 | 98.715 | False | ok |

## Derived Speedups

| T | Comparison | Steady Step Speedup | Peak Memory Ratio |
|---:|---|---:|---:|
| 10 | rate vs snntorch_dense | 16.11x | 4.45x |
| 10 | conv vs snntorch_conv | 6.58x | 1.26x |
| 25 | rate vs snntorch_dense | 17.87x | 2.47x |
| 25 | conv vs snntorch_conv | 10.21x | 1.23x |
| 50 | rate vs snntorch_dense | 37.05x | 1.75x |
| 50 | conv vs snntorch_conv | 11.52x | 1.22x |
