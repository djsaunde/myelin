# myelin

Fast spiking neural network simulation with PyTorch.

`myelin` targets the bottleneck that dominates many SNN training workloads:
Python-side time loops and per-timestep GPU kernel launches. PyTorch is a
required dependency and the correctness oracle. In practice `torch.compile` is
the strong default: Inductor fuses the time loop, shortens buffer lifetimes,
and captures the scalar loss, so compiled PyTorch is often the fastest *and*
lowest-memory path. The optional Triton backend earns its place for the cases
compile cannot infer from ordinary dense tensors: memory (checkpointed recompute,
rate-only readouts that never materialize `[T, B, N]` spikes), packed/distributed
spike reductions, and forward-only fused-time. Full rationale:
[docs/training_recommendation.md](docs/training_recommendation.md).

The package ships `py.typed`; inputs are time-major `[T, B, N]`.

## Quickstart

```bash
uv sync --extra dev
uv run pytest
```

`myelin` requires torch 2.13 (currently a nightly); `pyproject.toml` pulls
torch/torchvision/triton from the PyTorch nightly CUDA index automatically. Core
deps are `torch`, `torchvision`, `numpy`; CUDA/Triton and W&B tracking are
optional extras (`--extra cuda --extra tracking`). The CPU CI gate runs locally
as:

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
backward), `triton_generated` (DSL/codegen kernels; see below), or `auto`.

`LinearSurrogateLIF(stream_synapse=True, backend="triton")` computes currents
inside the recurrent kernel instead of materializing `[T, B, N]` currents;
`checkpoint_size=` (`int` or `"memory"`/`"balanced"`/`"speed"`) enables chunked
recompute across time; `LinearSurrogateLIFRate` returns spike rates directly and
never stores dense spikes. Built-in surrogates: `sigmoid`, `fast_sigmoid`,
`atan`, `triangular`, `superspike`, `multi_gaussian`.

## Custom neurons (DSL, experimental)

> Partial. Generated Triton kernels currently cover pointwise fused-time
> forward only, and generated backward is limited to hard-reset LIF-shaped
> IRs. Anything outside that boundary falls back to the PyTorch reference path,
> which handles arbitrary IRs. `analyze_neuron_ir(ir)` reports exactly which path
> a given IR qualifies for.

`NeuronBuilder`/`SurrogateBuilder` author restricted neuron and surrogate IRs.
The `CustomSurrogateNeuronCell` / `LinearCustomSurrogateNeuron` wrappers train
them through the surrogate LIF backend (`examples/custom_neuron_dsl.py`).

## Examples and benchmarks

Runnable training examples live in `examples/` (`train_mnist_rate.py`,
`train_mnist_conv.py`, `train_custom_lif_rate.py`, `export_hardware_bundle.py`).
The benchmark suite lives in `myelin.benchmarks`; generate the canonical CUDA set
locally with `python -m myelin.benchmarks.performance_frontier --device cuda` and
`python -m myelin.benchmarks.runner --preset core --require-cuda`, which write
results into `benchmarks/results/`.
