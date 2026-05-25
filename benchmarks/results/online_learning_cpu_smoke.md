# Online Learning CPU Smoke

Small correctness/perf smoke for the online-learning benchmark. This is not a
GPU performance result.

```bash
uv run python -m spiker.benchmarks.online_learning \
  --device cpu \
  --timesteps 4 \
  --batch 2 \
  --features 3 \
  --neurons 5 \
  --warmup 1 \
  --repeats 2
```

| Variant | Update ms | Peak MB | Grad Weight Norm | Error |
|---|---:|---:|---:|---|
| Online LIF eligibility | 0.303 |  | 0.064320 |  |
| Online ALIF eligibility | 0.437 |  | 0.063876 |  |
| BPTT LIF surrogate | 0.554 |  | 0.063825 |  |
| BPTT ALIF surrogate | 0.520 |  | 0.061770 |  |

