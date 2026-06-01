# myelin

Fast, time-fused spiking neural network simulation for PyTorch.

`myelin` is a research-grade SNN library targeting the bottleneck that dominates
many training workloads: Python-side time loops and per-timestep GPU kernel
launches. PyTorch is a required dependency and the correctness oracle; Triton is
the optional CUDA backend for fused-time kernels.

## Main takeaway so far

**`torch.compile` is the baseline to beat — and so far it wins.** On dense
training workloads, Inductor fuses the time loop, shortens buffer lifetimes, and
captures the whole scalar loss, so compiled PyTorch is often the fastest *and*
lowest-memory path. On the RTX 5090 performance frontier
(`benchmarks/results/performance_frontier_5090.md`), our handwritten and
generated Triton training kernels only **tie** compiled PyTorch on dense-output
training (1.01x) and barely edge it on rate output (1.07x).

Where Triton actually wins is narrower than we expected, and only when it
changes a contract PyTorch cannot infer from ordinary dense tensors:

- **Forward-only fused-time** launch-overhead win (2.46x over compiled hard-LIF
  forward).
- **Memory**, via checkpointed recompute and rate-only readouts that never
  materialize `[T, B, N]` spikes.
- **Packed spikes** and packed/distributed spike reductions.

So treat `torch.compile` as the default and reach for Triton when you need one
of those contract changes — not as a faster drop-in. Full rationale:
[docs/training_recommendation.md](docs/training_recommendation.md); milestone
status: [docs/roadmap.md](docs/roadmap.md).

## Reproduction: SpikeGPT on enwik8

We reproduce the SpikeGPT paper's enwik8 result. At the paper's **ctx-1024**
setting our 41M model (12 layers / 512 embd) reaches **test BPC 1.281**
(full-context strided eval on the held-out last 5M bytes) versus the paper's
**1.283** — a match.

Honest caveats: we use a **stabilized recipe** (cosine LR `4e-4 → 1e-5`, weight
decay `0.1`, bf16) rather than the paper's literal `6e-4` / `wd 0`, because the
literal recipe **diverges** in our setup (the LR is too high to converge and the
iterate diffuses out of the minimum); training uses the first 95M bytes (vs the
standard 90M), with the last 5M held out for test either way; and bf16 (with the
WKV recurrence and LIF membrane kept in fp32) was validated to track fp32. bf16
gives ~1.6x step-time under `torch.compile`, which makes the full ~10B-token
budget tractable (~15h on one RTX 5090).

```bash
# train (full budget); checkpoints the best-val model
uv run --extra tracking python examples/train_tiny_spikegpt.py \
  --text-file data/enwik8 --vocab byte --context-length 1024 --layers 12 --embedding 512 \
  --batch 12 --steps 833000 --lr 4e-4 --lr-final 1e-5 --warmup-steps 2000 \
  --weight-decay 0.1 --dropout 0.03 --amp bf16 --compile regional \
  --best-checkpoint-out runs/enwik8_repro.best.pt
# evaluate (full-context BPC by default)
uv run python examples/evaluate_spikegpt_checkpoint.py runs/enwik8_repro.best.pt \
  --text-file data/enwik8_test --no-sample
```

## Quickstart

```bash
uv sync --extra dev
uv run pytest
```

`myelin` requires **torch 2.13** (currently a nightly): the SpikeGPT WKV
recurrence is a single `associative_scan` higher-order op, which only has
correct autograd in 2.13+. `pyproject.toml` pulls torch/torchvision/triton from
the PyTorch nightly CUDA index automatically, so `uv sync` just works; see
`docs/wkv_recurrence.md` for the rationale and benchmarks.

Core deps are `torch`, `torchvision`, and `numpy`. CUDA/Triton, benchmark
comparison, and W&B tracking are optional extras:

```bash
uv sync --extra dev --extra cuda --extra bench --extra compare --extra tracking
```

The CPU CI gate runs locally as:

```bash
uv run ruff format --check . && uv run ruff check . && uv run pyright \
  && uv run pytest -q && uv build
```

## Package map

- `myelin.neurons` — correctness-first neuron dynamics (the oracle).
- `myelin.functional` — unfused PyTorch reference simulations.
- `myelin.kernels` — stable backend-dispatched forward/training entry points.
- `myelin.triton` — raw Triton kernels and packing helpers.
- `myelin.autograd` — custom autograd boundaries used by the dispatchers.
- `myelin.packing` — bitpacked spikes (32 spikes per signed int32 word).
- `myelin.modules` — small PyTorch training modules over the kernel paths.
- `myelin.online` — e-prop/OSTL-style online eligibility-trace rules.
- `myelin.hardware` — JSON hardware-bridge export (dense LIF, quantized,
  placement, SpiNNaker 2 manifest).

The package ships `py.typed`. Inputs are time-major: `[T, B, N]`.

## Backend selection

`lif_forward`, `alif_forward`, the surrogate forward APIs, `LinearLIF`, and
`LinearSurrogateLIF` accept an explicit `backend`:

- `torch` — PyTorch reference; always available, supports autograd.
- `triton` — requires CUDA tensors and the optional `triton` dependency; fused
  reverse-time surrogate backward.
- `triton_generated` — same path with DSL/codegen-generated kernels.
- `auto` — Triton when CUDA + Triton are available, else PyTorch (warns on
  CUDA-without-Triton, since it may leave a large speedup unused).

```python
from myelin import LIFParams, LIFState, lif_forward

params = LIFParams(tau_mem=20.0, threshold=1.0, reset=0.0)
initial = LIFState(membrane=current.new_zeros(current.shape[1:]))
final_state, spikes = lif_forward(current, initial, params, backend="auto")
```

`LinearSurrogateLIF(stream_synapse=True, backend="triton")` computes currents
inside the recurrent kernel instead of materializing `[T, B, N]` currents, with
Triton backward for the reverse-time recurrence plus `dinput`/`dweight`/`dbias`.
`checkpoint_size=` (an int, or the policy strings `"memory"`/`"balanced"`/
`"speed"`) enables chunked recompute across time. For classifier objectives,
`LinearSurrogateLIFRate` returns spike rates directly and never stores dense
spikes.

Built-in surrogates: `sigmoid`, `fast_sigmoid`, `atan`, `triangular`,
`superspike`, `multi_gaussian`. `SurrogateBuilder` authors custom pointwise
derivative IRs (evaluated in PyTorch, rendered to a Triton/Python body); backend
dispatch still accepts only the built-in names.

## Custom neurons (DSL)

`NeuronBuilder`/`SurrogateBuilder` author restricted neuron and surrogate IRs.
`analyze_neuron_ir(ir)` reports whether an IR fits the unroll/generated-forward
boundary and whether it supports the current generated-backward ABI (hard-reset
LIF-shaped only; ALIF has a recognized-but-unimplemented plan). The
`CustomSurrogateNeuronCell` / `LinearCustomSurrogateNeuron` /
`LinearCustomSurrogateNeuronRate` wrappers train those IRs through the surrogate
LIF backend.

```bash
uv run python examples/custom_neuron_dsl.py --device cpu
```

## Online learning

`linear_lif_online_eligibility_grad` (and the ALIF variant) compute dense
eligibility-trace updates from `[T, B, F]` inputs, `[F, N]` weights, and a
`[T, B, N]` learning signal — an e-prop/OSTL-style local oracle, not full BPTT
(it ignores gradients through reset and future recurrent state).
`LinearOnlineLIF`/`LinearOnlineALIF` wrap these as `nn.Module`s, where
`step_online(..., lr=...)` applies the update in place. Custom surrogate
derivative IRs plug in via `surrogate=`.

## Hardware export

`myelin.hardware` writes a generic JSON bridge for dense LIF layers: float
export → signed-symmetric quantized export → core placement plan → bundle
manifest, plus a SpiNNaker 2 adapter manifest. These are validated handoff
artifacts, not SDK programs.

```bash
uv run python examples/export_hardware_bundle.py --output-dir hardware_exports
```

## Bitpacked spikes

`pack_spikes`/`unpack_spikes` pack along the last dimension: `[T, B, N]` →
`[T, B, ceil(N / 32)]` int32 (a 32x reduction when `N % 32 == 0`).
`packed_spike_count`/`packed_spike_rate` reduce packed words directly (Triton
row-count fast path on CUDA). `lif_forward_packed_spikes(..., backend="auto")`
writes packed output directly on CUDA + Triton. `myelin.distributed` adds packed
all-gather and count/rate all-reduce helpers over `PackedSpikes`.

## Examples and benchmarks

Runnable training examples live in `examples/` (e.g. `train_mnist_rate.py`,
`train_mnist_conv.py`, `train_custom_lif_rate.py`). Benchmark results live in
`benchmarks/results/`. Regenerate the canonical CUDA set with:

```bash
uv run python -m myelin.benchmarks.performance_frontier --device cuda  # compiled-vs-Triton frontier
uv run python -m myelin.benchmarks.training_breakdown --device cuda    # projection/forward/backward split
uv run python -m myelin.benchmarks.runner --preset core --require-cuda --suffix 5090
```

`myelin.benchmarks.headline` summarizes saved artifacts without retraining, and
the runner is wired into the manual `GPU Benchmarks` GitHub Actions workflow
(`--dry-run` prints the command/artifact table). As one external reference
point, on `benchmarks/results/snntorch_matrix_5090.md` compiled myelin
rate-readout steady-step speedup over eager snnTorch grew from 9.92x at `T=10`
to 50.42x at `T=50` — another facet of the same compile-first takeaway.
