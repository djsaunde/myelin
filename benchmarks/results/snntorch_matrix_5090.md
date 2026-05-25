# spiker vs snnTorch Matrix

Compares compiled spiker examples against eager snnTorch examples across timesteps.

## Environment

- `generated_utc`: `2026-05-24T18:17:47+00:00`
- `device`: `cuda`
- `encoding`: `poisson`
- `batch`: `128`
- `hidden`: `128`
- `epochs`: `1`
- `train_limit`: `1024`
- `test_limit`: `1024`
- `compile_spiker_only`: `True`
- `matmul_precision`: `highest`

## Results

| T | Variant | Final Test Loss | Final Test Acc | Total s | Peak CUDA MB | Avg Step ms | Steady Step ms | Compiled | Status |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 10 | rate | 2.136011 | 0.6055 | 4.673 | 18.7 | 484.956 | 1.936 | True | ok |
| 10 | conv | 2.040249 | 0.4668 | 13.942 | 382.0 | 1168.907 | 2.151 | True | ok |
| 10 | snntorch_dense | 1.966508 | 0.6240 | 1.042 | 83.2 | 84.240 | 19.214 | False | ok |
| 10 | snntorch_conv | 2.158583 | 0.5312 | 1.218 | 483.6 | 111.696 | 19.606 | False | ok |
| 25 | rate | 2.137167 | 0.6182 | 3.511 | 44.0 | 348.250 | 1.760 | True | ok |
| 25 | conv | 2.231659 | 0.2383 | 28.913 | 897.0 | 2407.815 | 4.830 | True | ok |
| 25 | snntorch_dense | 1.976218 | 0.6465 | 1.379 | 108.5 | 109.915 | 47.369 | False | ok |
| 25 | snntorch_conv | 2.171570 | 0.2734 | 1.490 | 1102.8 | 130.483 | 46.790 | False | ok |
| 50 | rate | 2.178982 | 0.6289 | 5.706 | 123.0 | 594.521 | 1.756 | True | ok |
| 50 | conv | 2.254294 | 0.2197 | 62.754 | 1754.6 | 5372.341 | 8.064 | True | ok |
| 50 | snntorch_dense | 1.966261 | 0.6641 | 1.927 | 150.9 | 152.076 | 88.542 | False | ok |
| 50 | snntorch_conv | 2.157638 | 0.5596 | 2.113 | 2133.2 | 178.890 | 95.224 | False | ok |

## Derived Speedups

| T | Comparison | Steady Step Speedup | Peak Memory Ratio |
|---:|---|---:|---:|
| 10 | rate vs snntorch_dense | 9.92x | 4.45x |
| 10 | conv vs snntorch_conv | 9.11x | 1.27x |
| 25 | rate vs snntorch_dense | 26.91x | 2.47x |
| 25 | conv vs snntorch_conv | 9.69x | 1.23x |
| 50 | rate vs snntorch_dense | 50.42x | 1.23x |
| 50 | conv vs snntorch_conv | 11.81x | 1.22x |
