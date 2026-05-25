# Training Recommendation

This is the current v0 recommendation for training paths. It reflects the
benchmark evidence in `benchmarks/results/` and should be updated when the
backend tradeoffs change.

## Default Position

Use PyTorch plus `torch.compile` as the default training baseline. It is not a
toy fallback: on dense scalar-loss workloads it is often the fastest and lowest
memory implementation because Inductor can fuse loops, shorten buffer lifetimes,
and avoid materializing intermediates.

Use Triton when the library changes the SNN contract in a way PyTorch cannot
infer from ordinary dense tensors:

- rate-readout outputs instead of full `[T, B, N]` spike traces
- checkpointed recurrence across time
- packed spike storage and packed spike reductions
- generated kernels from the restricted neuron DSL
- future sparse/packed distributed spike communication

## Recommended Paths

| Use case | Path | Why |
|---|---|---|
| Correctness and research baselines | PyTorch reference functions | Simple, debuggable, and stable |
| General dense spike training | `LinearSurrogateLIF(..., stream_synapse=True)` with `torch.compile` | Keeps the ordinary dense spike contract and benefits from Inductor |
| Classifier/readout objectives | `LinearSurrogateLIFRate(..., backend="triton", checkpoint_size="balanced")` with `torch.compile` | Avoids output spike traces while preserving the current stable training recipe |
| Longer-T memory pressure | Try explicit `backend="triton_compile"` on `LinearSurrogateLIFRate` | Can reduce memory when the readout recurrence dominates, but it is experimental |
| Forward-only or communication paths | packed spike APIs | Avoids dense binary spike tensors |

## Current `triton_compile` Position

`backend="triton_compile"` is compile-visible and supports input, weight, and
bias gradients for the fast-sigmoid hard-forward rate path. It should remain
explicit rather than automatic.

The MNIST rate matrix on RTX 5090 shows why:

| Setting | Regular rate | `rate_triton_compile` | Takeaway |
|---|---:|---:|---|
| `T=10, hidden=128` | 82.86%, 18.7 MB, 1.764 ms | 83.64%, 18.7 MB, 1.900 ms | Quality holds; no memory win; slower steady step |
| `T=25, hidden=128` | 84.08%, 110.3 MB, 1.555 ms | 84.52%, 44.0 MB, 1.732 ms | Quality holds; large memory win; slower steady step |
| `T=10, hidden=256` | 85.35%, 103.9 MB, 1.475 ms | 84.03%, 103.9 MB, 1.839 ms | Quality roughly holds; no memory win; slower steady step |

So the right interpretation is not "Triton is the faster default." The current
evidence says `triton_compile` is a memory-oriented experimental option for
rate-readout training, especially when `T` is high enough that the readout
state/output contract matters.

## Benchmark Reporting Rules

When comparing training paths, always report:

- final test loss
- final test accuracy
- total wall time
- peak CUDA memory
- average step time
- steady-state step time
- whether `torch.compile` was enabled

Do not make a performance claim from only total wall time or only steady-state
step time. Compile warmup, optimizer state, and evaluation cadence can move
those numbers independently.
