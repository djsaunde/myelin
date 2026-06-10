# myelin

Fast, time-fused spiking neural network simulation for PyTorch.

`myelin` targets the bottleneck that dominates many SNN training workloads:
Python-side time loops and per-timestep GPU kernel launches. PyTorch is a
required dependency and the correctness oracle; Triton is the optional CUDA
backend for fused-time kernels. The package ships `py.typed`; inputs are
time-major `[T, B, N]`.

> The SpikeGPT language-model reproduction that previously lived here now lives
> in its own repository,
> [spikegpt-myelin](https://github.com/djsaunde/spikegpt-myelin), which depends
> on this package.

## Main takeaway

**`torch.compile` is the baseline to beat, and so far it wins.** Inductor fuses
the time loop, shortens buffer lifetimes, and captures the whole scalar loss, so
compiled PyTorch is often the fastest *and* lowest-memory path. On the RTX 5090
frontier (`benchmarks/results/performance_frontier_5090.md`), our Triton training
kernels only **tie** compiled PyTorch on dense-output training (1.01x) and barely
edge it on rate output (1.07x). Triton wins only where it changes a contract
PyTorch cannot infer from dense tensors: forward-only fused-time (2.46x over
compiled hard-LIF), memory (checkpointed recompute, rate-only readouts that never
materialize `[T, B, N]` spikes), and packed/distributed spike reductions.

So treat `torch.compile` as the default and reach for Triton for one of those
contract changes, not as a faster drop-in. Full rationale:
[docs/training_recommendation.md](docs/training_recommendation.md); milestones:
[docs/roadmap.md](docs/roadmap.md).

## Quickstart

```bash
uv sync --extra dev
uv run pytest
```

`myelin` requires **torch 2.13** (currently a nightly); `pyproject.toml` pulls
torch/torchvision/triton from the PyTorch nightly CUDA index automatically. Core
deps are `torch`, `torchvision`, `numpy`; CUDA/Triton, benchmark comparison, and
W&B tracking are optional extras (`--extra cuda --extra compare --extra
tracking`). The CPU CI gate runs locally as:

```bash
uv run ruff format --check . && uv run ruff check . && uv run pyright \
  && uv run pytest -q && uv build
```

## Package map

- `myelin.neurons` / `myelin.functional`: correctness-first neuron dynamics (the
  oracle) and unfused PyTorch reference simulations.
- `myelin.kernels`: stable backend-dispatched forward/training entry points.
- `myelin.triton` / `myelin.autograd`: raw Triton kernels and the custom
  autograd boundaries the dispatchers use.
- `myelin.modules`: small PyTorch training modules over the kernel paths.
- `myelin.packing` / `myelin.distributed`: bitpacked spikes (32 per int32 word)
  and packed all-gather / count-rate all-reduce helpers.
- `myelin.online`: e-prop/OSTL-style online eligibility-trace rules.
- `myelin.hardware`: JSON hardware-bridge export (dense LIF → quantized →
  placement → bundle manifest, plus a SpiNNaker 2 adapter).

## Backends

`lif_forward`, `alif_forward`, the surrogate forward APIs, and the `Linear*`
modules accept an explicit `backend`: `torch` (reference, always available,
autograd), `triton` (CUDA + optional `triton` dep; fused reverse-time surrogate
backward), `triton_generated` (DSL/codegen kernels), or `auto`.

`LinearSurrogateLIF(stream_synapse=True, backend="triton")` computes currents
inside the recurrent kernel instead of materializing `[T, B, N]` currents;
`checkpoint_size=` (`int` or `"memory"`/`"balanced"`/`"speed"`) enables chunked
recompute across time; `LinearSurrogateLIFRate` returns spike rates directly and
never stores dense spikes. Built-in surrogates: `sigmoid`, `fast_sigmoid`,
`atan`, `triangular`, `superspike`, `multi_gaussian`.

## Custom neurons (DSL)

`NeuronBuilder`/`SurrogateBuilder` author restricted neuron and surrogate IRs;
`analyze_neuron_ir(ir)` reports whether an IR fits the unroll/generated-forward
boundary and the generated-backward ABI (hard-reset LIF-shaped only). The
`CustomSurrogateNeuronCell` / `LinearCustomSurrogateNeuron` wrappers train them
through the surrogate LIF backend (`examples/custom_neuron_dsl.py`).

## Examples and benchmarks

Runnable training examples live in `examples/` (`train_mnist_rate.py`,
`train_mnist_conv.py`, `train_custom_lif_rate.py`, `export_hardware_bundle.py`);
benchmark results live in `benchmarks/results/`. Regenerate the canonical CUDA
set with `python -m myelin.benchmarks.performance_frontier --device cuda` and
`python -m myelin.benchmarks.runner --preset core --require-cuda`. As one
external reference point (`benchmarks/results/snntorch_matrix_5090.md`), compiled
myelin rate-readout steady-step speedup over eager snnTorch grew from 9.92x at
`T=10` to 50.42x at `T=50`.
