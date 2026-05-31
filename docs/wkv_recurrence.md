# WKV recurrence: why `associative_scan`

The SpikeGPT time-mixing block uses the RWKV "weighted key-value" (WKV)
recurrence (`myelin.language.weighted_key_value`). It is a first-order linear
recurrence over the token/time axis: a decay-weighted running average of past
values. This note records why the default implementation is an
`associative_scan` higher-order op rather than a Python time loop.

## The problem

The obvious implementation is a `for t in range(T)` loop
(`weighted_key_value_loop`, kept as the correctness oracle). It is correct and
memory-light, but `torch.compile` **unrolls** the Python loop, so the graph
grows with the context length `T` and Inductor compile time explodes
super-linearly. On an RTX 5090 (cold compile, fwd+bwd): ~107 s at `T=64`,
~294 s at `T=128`, and impractical beyond. Compiling the model — which is where
`torch.compile` pays off for SNNs — becomes the bottleneck.

## Variants considered

All are exact in the forward; the differences are compile time, peak memory, and
whether backward (training) is correct. RTX 5090, batch 8, channels 128, fp32,
cold compile, fwd+bwd (see `benchmarks/results/spikegpt_wkv_compare_5090.md` and
`spikegpt_wkv_compare.py`):

| Variant | Compile @ T=1024 | Steady runtime @ T=1024 | Peak mem @ T=1024 | Training backward |
|---|--:|--:|--:|:--:|
| loop (`weighted_key_value_loop`) | uncompilable (>>100 s) | — | low | correct |
| parallel (O(T²) decay matrix) | ~3.9 s | 4.7 ms | 2299 MB | correct |
| chunked (decay matrix per chunk) | ~49 s | 6.6 ms | 232 MB | correct |
| `scan` HOP | fails / wrong | — | low | **WRONG** |
| **`associative_scan` (generic)** | ~51 s | **1.8 ms** | **231 MB** | **correct** |

## Why `associative_scan` (generic mode)

WKV is a linear recurrence, so it fits an associative scan: each token is a
log-space monoid element `(acc_decay, log_scale, num, den)` with true
accumulator `exp(log_scale) * num`, and the associative `combine` decays the
earlier segment by the later segment's total decay before merging (with a shared
max-exponent for stability). This gives, at long context:

- **Low, linear memory** — 231 MB at `T=1024`, vs the parallel form's O(T²)
  2.3 GB.
- **Best compiled runtime** — flat ~1.8 ms, beating both parallel and chunked.
- **Correct training** — forward exact, backward correct *including the
  `time_decay`/carry gradient*, numerically stable to `T=1024`.

The one wart is compile time: it grows with `T` (the prototype generic-mode
autograd materializes a joint backward graph), ~51 s cold at `T=1024`. Unlike
the loop it always completes, and it is a one-time, cacheable cost.

### Rejected alternatives

- **`scan` HOP**: forward is exact and compiles (with `.clone()` on the
  `combine_fn` outputs to satisfy the no-aliasing rule), but its backward
  **drops the carry-recurrence gradient** — a minimal `s_t = 0.9·s_{t-1} + x_t`
  test returns gradient `[1,1,…]` instead of the correct `[4.69, 4.10, …]`. This
  holds eager and compiled, in torch 2.12 and 2.13.dev. Unusable for training.
- **`associative_scan` pointwise mode** (CUDA-only): faster/flatter compile, but
  produces NaN/exploding gradients and parallelizes the whole scan tree into
  ~34 GB at `T=1024`. Unusable.

## End-to-end: the next bottleneck is `SpikingSequenceLIF`

Fixing WKV does **not** by itself make the whole SpikeGPT model compile fast.
`SpikeGPTBlock.forward` also applies two `SpikingSequenceLIF` activations, each of
which is its own `for step in range(T)` loop that `torch.compile` unrolls. An
end-to-end `spikegpt_compile_probe` (context 32, 2 layers, 128 embedding, RTX
5090, `fullgraph=True`) is correct (compiled loss matches eager) and fast once
warm (compiled steady 4.6 ms), but the cold compile still takes ~57 s at `T=32`
— now dominated by the four unrolled LIF loops, not WKV.

So the WKV is no longer the compile bottleneck, but the LIF activations are. The
same treatment applies: `SpikingSequenceLIF` is a linear recurrence with a hard
reset (a per-step multiplier in `{decay, 0}`), so it can be made loop-free via
`associative_scan` or routed through `myelin`'s existing fused-time LIF kernels.
That is the next step toward a flat whole-model compile.

## torch 2.13 requirement

`associative_scan` autograd is only correct in torch 2.13+, which at time of
writing is a **nightly** build (`pyproject.toml` pulls torch/torchvision/triton
from the PyTorch nightly CUDA index). Caveat: a pinned nightly dev build is
garbage-collected from the index after a few weeks — re-run `uv lock` if a sync
fails. Switch the index back to PyPI and pin `torch>=2.13` once 2.13 ships
stable.
