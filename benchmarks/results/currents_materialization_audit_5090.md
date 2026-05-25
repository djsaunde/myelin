# Currents Materialization Audit

The compiled PyTorch workload returns a scalar loss from inside the compiled
graph. Dense-output Triton variants return `[T, B, N]` spikes to Python
before the scalar loss consumes them, so they have a dense-output allocation
lower bound that the compiled scalar-loss graph does not expose.

## Environment

- `generated_utc`: `2026-05-24T13:16:43+00:00`
- `device`: `cuda`
- `gpu`: `NVIDIA GeForce RTX 5090`
- `torch`: `2.12.0+cu130`
- `cuda_available`: `True`
- `cuda_version`: `13.0`
- `shape`: `T=100, B=64, F=128, N=2048`
- `expected_currents_bytes`: `52428800`
- `expected_currents_mb`: `50.0`
- `expected_dense_spike_output_bytes`: `52428800`
- `expected_dense_spike_output_mb`: `50.0`
- `expected_chunk_start_bytes`: `2097152`
- `expected_chunk_start_mb`: `2.0`
- `expected_checkpoint_scratch_bytes`: `13107200`
- `expected_checkpoint_scratch_mb`: `12.5`
- `checkpoint_size`: `25`
- `resolved_checkpoint_size`: `25`
- `compile_mode`: `reduce-overhead`

## Results

| Variant | Dense Spike Output | Fwd ms | Bwd ms | Baseline Alloc MB | Fwd Peak MB | Fwd Increment MB | Fwd Increment / Currents | Bwd Peak MB | Bwd Increment MB | Fwd Extra Over Dense Output MB | Bwd Extra Over Dense Output MB | Error |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Eager materialized currents | yes | 16.593 | 17.997 | 75.1 | 425.1 | 350.0 | 7.00x | 425.1 | 350.0 | 300.0 | 300.0 |  |
| torch.compile materialized graph | no | 0.404 | 0.511 | 11.1 | 11.1 | 0.0 | 0.00x | 11.1 | 0.0 |  |  |  |
| PyTorch streamed custom autograd | yes | 21.009 | 21.535 | 76.1 | 327.1 | 251.0 | 5.02x | 327.1 | 251.0 | 201.0 | 201.0 |  |
| Triton fused synapse full trace | yes | 0.258 | 0.468 | 77.1 | 177.1 | 100.0 | 2.00x | 228.6 | 151.5 | 50.0 | 101.5 |  |
| Triton checkpoint recompute | yes | 0.235 | 0.518 | 78.1 | 130.1 | 52.0 | 1.04x | 144.1 | 66.0 | 2.0 | 16.0 |  |
| Triton checkpoint rate output | no | 0.220 | 0.478 | 79.1 | 81.1 | 2.0 | 0.04x | 95.6 | 16.5 |  |  |  |
| Generated Triton checkpoint rate output | no | 0.223 | 0.526 | 80.1 | 82.1 | 2.0 | 0.04x | 96.6 | 16.5 |  |  |  |

## Scratch Reuse Comparison

This run includes two-buffer ping-pong reuse for the checkpoint-backward
`grad_prev` scratch, replacing a per-chunk `torch.empty_like` allocation. Peak
memory is unchanged because the same live scratch is still required, but
backward allocator churn drops.

| Variant | Before Bwd ms | After Bwd ms | Change |
|---|---:|---:|---:|
| Triton checkpoint recompute | 0.643 | 0.523 | 18.7% faster |
| Triton checkpoint rate output | 0.611 | 0.478 | 21.8% faster |
| Generated Triton checkpoint rate output | 0.750 | 0.525 | 30.0% faster |

## Spike Scratch Elision

Checkpoint backward now derives the hard spike from `pre_reset >= threshold`
instead of storing and reloading a packed spike scratch. This is valid for the
current hard-forward LIF contract and removes one chunk-local scratch tensor
plus the associated Triton memory traffic.

## Takeaway

The compiled materialized PyTorch graph did not show a currents-sized peak-memory
increment in this run, but it also returns a scalar loss from inside the compiled
graph instead of exposing dense spikes to Python. That suggests Inductor
eliminated, rematerialized, or reused storage for the nominal `[T, B, N]`
currents and spike-like tensors well enough that they do not appear as separate
peak allocations at the Python boundary.

Triton fused synapse avoids the explicit currents tensor too, but the full-trace
path still stores recurrent BPTT traces. Checkpoint/recompute cuts that cost
substantially. For dense spike-output training, the checkpoint path's forward
increment is now almost exactly the dense output lower bound: 50.0 MB of spikes
plus 2.0 MB of chunk-start membrane state. Its backward increment is 16.0 MB
above the dense output lower bound, roughly the expected chunk-start state plus
one 12.5 MB checkpoint scratch buffer.

For scalar spike-rate objectives, the Triton checkpoint rate-output path avoids
both dense currents and dense spike outputs. In this run both the handwritten
and generated rate-output paths had a forward increment of only 0.04x the
expected currents size, and a backward increment of 16.9 MB.
