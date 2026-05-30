# spiker

Fast, time-fused spiking neural network simulation for PyTorch.

`spiker` is starting as a research-grade SNN library focused on the bottleneck
that dominates many training workloads: Python-side loops and per-timestep GPU
kernel launches. PyTorch is a required dependency and the correctness oracle;
Triton is the optional CUDA backend for fused-time kernels.

The current path has moved past a single LIF proof of concept. Triton-backed
surrogate training has fused forward and reverse-time backward kernels for dense
hard-forward LIF currents, fused dense-synapse + LIF training for
`LinearSurrogateLIF`, checkpointed recompute across time, direct spike-rate
readouts for low-memory classifier objectives, and generated fused-time forward
kernels for builder-authored custom neurons. Generic generated backward for
custom neuron IRs is still future work, but the DSL now recognizes the first
narrow generated-backward contract: builder-authored hard-reset LIF IRs that
match the existing surrogate LIF recurrence. ALIF also has an explicit
unimplemented backward plan, which is the next kernel target.

## Quickstart

```bash
uv sync --extra dev
uv run pytest
```

Core install dependencies include `torch`, `torchvision`, and `numpy`.
CUDA/Triton, benchmark comparison, and W&B tracking dependencies are optional
extras:

```bash
uv sync --extra dev --extra cuda --extra bench --extra compare --extra tracking
```

The CPU CI gate runs the same basic checks locally:

```bash
uv run ruff format --check .
uv run ruff check .
uv run pyright
uv run pytest -q
uv build
```

The smaller CUDA development install is:

```bash
uv sync --extra dev --extra cuda --extra bench
uv run python -m spiker.benchmarks.lif --device cuda
```

To regenerate a consistent CUDA benchmark subset and write markdown artifacts:

```bash
uv run python -m spiker.benchmarks.performance_frontier --device cuda
uv run python -m spiker.benchmarks.training_breakdown --device cuda
uv run python -m spiker.benchmarks.runner --preset smoke --require-cuda --suffix 5090
uv run python -m spiker.benchmarks.runner --preset core --require-cuda --suffix 5090
```

`performance_frontier` is the canonical compiled-vs-Triton comparison;
`training_breakdown` isolates projection, forward, and backward costs, and
`compile_triton_boundary` checks whether selected Triton kernels remain visible
inside `torch.compile`. Use `--dry-run` on the runner to print the
command/artifact table without running the suite.
The same runner is wired into the manual `GPU Benchmarks` GitHub Actions
workflow for self-hosted Linux runners labeled with CUDA support. The presets
also include distributed packed-collective smoke/core artifacts: CPU/Gloo for
portable two-rank local payload checks, and CUDA/NCCL for the packed count/rate
fast path.

## Current Shape

- `spiker.neurons` contains the correctness-first neuron dynamics.
- `spiker.functional` contains unfused PyTorch reference simulations.
- `spiker.kernels` exposes the stable backend-dispatched forward/training
  entry points, including fused dense-synapse surrogate LIF and direct
  spike-rate and packed-spike readouts.
- `spiker.triton` contains raw Triton kernels and packing helpers.
- `spiker.autograd` contains internal custom autograd boundaries used by the
  dispatchers; hard LIF final-membrane gradients still replay the PyTorch
  reference, while surrogate spike-output gradients use Triton backward
  kernels.
- `spiker.packing` contains the first bitpacked spike representation:
  32 spikes per signed int32 word along the neuron dimension.
- `spiker.modules` contains small PyTorch modules for training examples and
  delegates optimized paths to `spiker.kernels`.
- `spiker.hardware` contains the first JSON hardware-bridge export schemas for
  dense LIF layers, quantized artifacts, placement plans, and target handoff
  manifests.
- The package ships a `py.typed` marker so downstream type checkers can consume
  the inline type hints.

Current milestone status is tracked in [docs/roadmap.md](docs/roadmap.md). The
current v0 training-path recommendation is in
[docs/training_recommendation.md](docs/training_recommendation.md).

The practical training path today is:

1. Use PyTorch references as correctness baselines.
2. Use `torch.compile` as the default fair-performance baseline for ordinary
   dense training.
3. Use `LinearSurrogateLIF(..., stream_synapse=True, backend="triton")` for
   dense-output training when full spike traces are needed.
4. Use `LinearSurrogateLIFRate(..., backend="triton", checkpoint_size=...)` for
   classifier/readout objectives that only need spike rates.
   `backend="triton_compile"` is an explicit experimental memory-oriented
   variant for compile-visible fast-sigmoid rate training when the surrounding
   loss is wrapped in `torch.compile`; it supports input, weight, and bias
   gradients, but it is not the default speed path.
5. Use `LinearSurrogateLIFPacked(..., backend="auto")` or
   `spiker.linear_surrogate_lif_packed_forward(..., backend="auto")` for
   forward-only simulation or communication paths that should return packed
   spike bits instead of dense `[T,B,N]` float spikes.
6. Use `CustomNeuronCell` + `TimeUnroll(..., backend="auto")` for generated
   fused-time custom-neuron forward on CUDA, with PyTorch evaluator fallback on
   CPU.
7. Use `LinearCustomSurrogateNeuron(..., stream_synapse=True)` for the current
   custom LIF-shaped surrogate training path.
8. Use `LinearCustomSurrogateNeuronRate(...)` for custom LIF-shaped rate
   readouts that should avoid dense output spike traces.

For a minimal user-authored neuron example, run:

```bash
uv run python examples/custom_neuron_dsl.py --device cpu
```

On CUDA with Triton installed, the same example also checks the generic
generated forward launcher against the PyTorch DSL evaluator.

To time the public module path for a custom two-state ALIF-style neuron:

```bash
uv run python -m spiker.benchmarks.custom_neuron_module
```

To compare the custom LIF-shaped surrogate training wrapper against the
built-in surrogate LIF path:

```bash
uv run python -m spiker.benchmarks.custom_surrogate_training --device cuda
```

To train a builder-authored LIF-shaped IR through the low-memory rate readout
module:

```bash
uv run python examples/train_custom_lif_rate.py --device cuda --backend triton_generated
```

To write a generic train-then-export hardware bundle for a small `LinearLIF`
layer:

```bash
uv run python examples/export_hardware_bundle.py --output-dir hardware_exports
```

To compare online eligibility updates against surrogate BPTT on a small dense
workload:

```bash
uv run python -m spiker.benchmarks.online_learning --device cpu
```

For the larger CUDA online-learning readout used by the benchmark runner:

```bash
uv run python -m spiker.benchmarks.online_learning \
  --device cuda --timesteps 100 --batch 64 --features 128 --neurons 512 --seed 0
```

To inspect what `torch.compile`/Inductor emits for the pure PyTorch current
materialization baselines:

```bash
uv run python -m spiker.benchmarks.compile_inspect --workload materialized --device cuda
```

To sweep the checkpoint chunk-size memory/latency tradeoff for Triton
checkpointed training:

```bash
uv run python -m spiker.benchmarks.checkpoint_size_sweep --device cuda
```

To measure the experimental `torch.library.triton_op` boundary for combining
custom Triton kernels with `torch.compile`:

```bash
uv run python -m spiker.benchmarks.compile_triton_boundary --device cuda
```

To sweep regular compiled PyTorch against the existing Triton rate path and the
compile-visible `triton_compile` backend:

```bash
uv run python -m spiker.benchmarks.compile_triton_sweep --device cuda
```

To run the current MNIST rate-readout quality/performance matrix:

```bash
uv run python -m spiker.benchmarks.mnist_rate_matrix --device cuda
```

To summarize the current saved headline benchmark artifacts without rerunning
training:

```bash
uv run python -m spiker.benchmarks.headline
```

To compare compiled spiker examples against eager snnTorch across timestep
counts:

```bash
uv run python -m spiker.benchmarks.snntorch_matrix --device cuda
```

The default matrix runs `T=10/25/50` with a small matched MNIST budget. On the
RTX 5090 artifact in `benchmarks/results/snntorch_matrix_5090.md`, compiled
spiker rate-readout steady-step speedup over eager snnTorch dense grew from
9.92x at `T=10` to 50.42x at `T=50`.

Other checkpointed benchmark CLIs accept the same `--checkpoint-size` values as
the public API, for example `--checkpoint-size memory` or
`--checkpoint-size speed`.

To train the current low-memory MNIST rate-readout example:

```bash
uv run python examples/train_mnist_rate.py --device cuda --compile --backend triton
```

Omitting `--backend` is equivalent on a CUDA machine with Triton installed:
`backend="auto"` resolves to the recommended `triton` rate path, not to the
experimental `triton_compile` path.

Add `--wandb` after running `uv run wandb login` to send serial metric tables to
Weights & Biases.

To train the tuned convolutional MNIST example:

```bash
uv run python examples/train_mnist_conv.py --device cuda --compile
```

The current conv defaults use `T=25`, `hidden=256`, `epochs=4`,
`grad_clip=1.0`, and fan-in SNN synapse initialization. On the RTX 5090 tuned
subset run in `benchmarks/results/mnist_conv_tuned_5090.md`, that recipe reached
95.78% test accuracy.

For the stronger bounded quality recipe, add dropout and label smoothing:

```bash
uv run python examples/train_mnist_conv.py --device cuda --compile \
  --train-limit 10000 --test-limit 4096 \
  --dropout 0.1 --label-smoothing 0.05
```

That run reached 96.46% in
`benchmarks/results/mnist_conv_dropout_label_smoothing_5090.md`.

The current public v0 API is intentionally small:

```python
from spiker import (
    ALIFParams,
    ALIFState,
    IzhikevichParams,
    IzhikevichState,
    CustomNeuronCell,
    CustomSurrogateNeuronCell,
    LIFParams,
    LIFState,
    LinearCustomSurrogateNeuron,
    LinearCustomSurrogateNeuronRate,
    LinearLIF,
    LinearSurrogateLIF,
    LinearSurrogateLIFRate,
    NeuronBuilder,
    SurrogateBuilder,
    TimeUnroll,
    alif_forward,
    analyze_neuron_ir,
    evaluate_neuron_unroll,
    izhikevich_forward,
    lif_forward,
    linear_surrogate_lif_rate_forward,
    linear_surrogate_lif_forward,
    lif_step,
    lif_unroll,
    pack_spikes,
    packed_spike_rate,
    surrogate_lif_forward,
    unpack_spikes,
)
```

Inputs use time-major layout: `[T, B, N]`.

For custom neurons, `analyze_neuron_ir(ir)` returns structured validation
diagnostics and reports whether the IR fits the default public unroll/generated
forward boundary before codegen is attempted. It also reports
`supports_generated_backward=True` only for the current hard-reset LIF-shaped
custom-neuron backward ABI and exposes the recognized `NeuronBackwardPlan` via
`plan_generated_backward_ir(ir)`. ALIF now has a recognized but unimplemented
`alif_adaptive_threshold` backward plan, exposed with
`plan_generated_backward_ir(ir, allow_unimplemented=True)`. Refractory LIF and
richer custom dynamics remain forward-codegen only for now.

`CustomSurrogateNeuronCell` is the first public training wrapper for that narrow
backward ABI. It accepts a builder-authored LIF-shaped `NeuronIR`, resolves the
generated-backward plan, and then reuses the existing surrogate LIF backend
path, including `backend="triton_generated"` on CUDA.
`LinearCustomSurrogateNeuron` adds the matching dense trainable layer wrapper
and can stream the dense synapse through the same fused surrogate LIF backend.
`LinearCustomSurrogateNeuronRate` exposes the corresponding rate-output wrapper
for classifier/readout objectives.

## Backend Selection

`lif_forward`, `alif_forward`, `surrogate_lif_forward`, `surrogate_alif_forward`,
`LinearLIF`, and `LinearSurrogateLIF` accept explicit backend choices.
`backend="torch"` and `backend="auto"` are available across the public forward
APIs. `lif_forward` and surrogate LIF support `backend="triton"`;
`alif_forward`, `izhikevich_forward`, `surrogate_lif_forward`, and
`LinearSurrogateLIF` also accept `backend="triton_generated"` for opt-in
generated kernels. `surrogate_alif_forward` is currently a PyTorch oracle;
Triton/generated backends are reserved for the planned ALIF backward kernel.

```python
from spiker import LIFParams, LIFState, lif_forward

params = LIFParams(tau_mem=20.0, threshold=1.0, reset=0.0)
initial = LIFState(membrane=input_current.new_zeros(input_current.shape[1:]))
final_state, spikes = lif_forward(input_current, initial, params, backend="auto")
```

- `backend="torch"` always uses the PyTorch reference implementation and
  supports autograd.
- `backend="triton"` requires CUDA tensors and the optional `triton`
  dependency. Surrogate spike-output gradients are supported by a fused
  reverse-time Triton backward kernel.
- `backend="triton_generated"` uses the same Triton surrogate forward path but
  swaps in DSL/codegen-generated kernels. For ALIF and Izhikevich this is
  generated fused-time forward. For surrogate LIF this is generated
  reverse-time backward. For `stream_synapse=True`, the generated path supports
  full-trace, checkpointed training, and checkpointed rate readout, including
  optional input, weight, and bias gradients.
- `backend="auto"` uses Triton when CUDA tensors and Triton are available,
  otherwise it falls back to PyTorch. CUDA fallback without Triton emits a
  warning because it may leave a large speedup unused.

The current Triton surrogate path stores pre-reset membranes for backward. That
uses more memory than the forward-only path but lets backward run as one fused
reverse-time Triton kernel.

`LinearSurrogateLIF(stream_synapse=True, backend="triton")` extends the Triton
boundary across the dense projection. It computes currents inside the forward
kernel instead of materializing `[T, B, N]` currents first, and its backward uses
Triton kernels for the reverse-time recurrence plus `dinput`, `dweight`, and
`dbias`.

`torch.compile` can sometimes erase the cost of a source-level
`currents = inputs @ weight` temporary, so compiled PyTorch remains the fair
baseline. The streamed synapse path makes the stronger backend contract
explicit: current production is inside the recurrent boundary, one timestep at
a time or inside the fused Triton kernel.

Training examples expose `--matmul-precision`, which calls
`torch.set_float32_matmul_precision(...)`. They default to `highest` because
SNN thresholds can make training sensitive to small matmul differences. Use
`--matmul-precision high` to enable TF32 matmul paths on CUDA hardware that
supports them when speed is more important than bit-level agreement with the
default PyTorch behavior.

The same backend-selected path is available as a function:

```python
from spiker import linear_surrogate_lif_forward

final_state, spikes = linear_surrogate_lif_forward(
    inputs,
    weight,
    bias,
    params,
    surrogate="fast_sigmoid",
    backend="auto",
)
```

Built-in hard-forward surrogate gradients currently include `sigmoid`,
`fast_sigmoid`, `atan`, `triangular`, `superspike`, and `multi_gaussian`.
For research surrogates that are not built in, `SurrogateBuilder` authors a
restricted pointwise derivative IR over `centered` and scalar params. That IR
can be evaluated with PyTorch and rendered into a Triton/Python derivative body;
full backend dispatch still accepts only the built-in names.

`LinearSurrogateLIF(..., stream_synapse=True, checkpoint_size=...)` enables
chunked recompute across time. On the Triton backend this lower-memory path is
used for optional input, weight, and bias gradients. `backend="triton_generated"`
uses generated reverse-time chunk backward code for the same checkpoint
contract.

`checkpoint_size` can be a positive integer or a policy string:
`"memory"`, `"balanced"`, or `"speed"`. `recommended_checkpoint_size(T)`
returns the concrete chunk size for a policy; at `T=100`, those policies map to
`10`, `25`, and `50` respectively. The low-memory MNIST rate example defaults
to `"balanced"`.

For spike-rate objectives that do not need the full `[T, B, N]` spike tensor,
`linear_surrogate_lif_rate_forward` returns a scalar spike rate by default, or
`[B, N]` rates with `reduction="none"`. `LinearSurrogateLIFRate` wraps the same
path as a module and defaults to `reduction="none"` for classifier-style
readouts. On the Triton backend it uses checkpointed recompute without
materializing dense spike outputs or dense spike-output gradients. The
`triton_generated` backend uses the same low-memory rate forward with generated
checkpoint backward code.

`examples/train_mnist_rate.py` uses this path for an MNIST classifier readout:
the hidden layer still emits a dense spike trace, but the output layer returns
class spike rates directly instead of storing `[T, B, classes]` spikes.

## Online Learning

`linear_lif_online_eligibility_grad` is the first online-learning reference
primitive. It computes a dense LIF eligibility-trace update from
`[T, B, F]` inputs, `[F, N]` weights, and a per-timestep learning signal
`[T, B, N]`. This is an e-prop/OSTL-style local update oracle, not full BPTT:
it intentionally ignores gradients through reset and future recurrent state.
`LinearOnlineLIF` and `LinearOnlineALIF` wrap these rules as `nn.Module`s;
`forward` returns the local gradient estimate, and `step_online(..., lr=...)`
applies it directly to the layer parameters. The functional helpers and module
wrappers accept a `SurrogateBuilder` derivative IR via `surrogate=...` plus
`surrogate_params`, which is the first path where custom surrogate derivatives
participate in a learning rule.

## Hardware Export

`export_dense_lif_layer` writes the first hardware-bridge artifact: a
JSON-serializable dense LIF layer with weights, optional bias, LIF parameters,
timestep metadata, and string metadata. This is a generic compatibility schema;
target SDK integration remains future work.

```python
from spiker import LIFParams, export_dense_lif_layer

export = export_dense_lif_layer(weight, bias, LIFParams(), dt=0.001)
json_payload = export.to_json()
```

`export_linear_lif_module` exports `LinearLIF`-style modules that expose a
`synapse` and LIF `TimeUnroll` cell, and `read_hardware_export` validates a
saved JSON artifact before returning the typed export object.

`quantize_dense_lif_export` converts the float artifact to a signed symmetric
fixed-point artifact (`spiker.dense_lif_quantized.v0`) with explicit bit width,
integer range, and per-tensor scales for weights and bias. This is target-prep,
not final deployment: real hardware adapters still need to decide acceptable
bit widths, quantization calibration, core placement, and runtime format.

`plan_dense_lif_placement` tiles a quantized dense layer into generic target
core ranges with explicit input/output spans and synapse counts. When a layer's
input dimension is split across cores for the same output range, the plan marks
those cores as requiring accumulation, which is the next routing constraint a
real adapter must lower into target-specific communication.

`export_linear_lif_hardware_bundle` runs the whole generic export pipeline for a
`LinearLIF`-style module and writes a manifest plus the float, quantized, and
placement JSON artifacts into one directory.

`export_spinnaker2_dense_lif_manifest` builds a target-specific adapter
artifact. It references the quantized dense export and placement plan, records
per-core neuron and incoming-synapse limits, maps the generic placement tiles
into target handoff metadata, and emits a manifest for a later SDK lowering
pass. It does not generate SpiNNaker SDK objects yet.

```bash
uv run python examples/export_hardware_bundle.py --adapter spinnaker2
```

## Bitpacked Spikes

`pack_spikes` and `unpack_spikes` define the v0 packed spike format. Packing is
along the last dimension, so `[T, B, N]` float or bool spikes become
`[T, B, ceil(N / 32)]` int32 words. This is a fixed 32x storage reduction
versus float32 dense spikes when `N` is divisible by 32.

```python
from spiker import pack_spikes, packed_spike_counts, packed_spike_rate, unpack_spikes

packed = pack_spikes(spikes)
round_trip = unpack_spikes(packed, dtype=spikes.dtype)
row_counts = packed_spike_counts(packed)
per_timestep_counts = packed_spike_counts(packed, dim=(1, -1))
per_neuron_counts = packed_spike_count(packed, dim=(0, 1))
rate = packed_spike_rate(packed)
row_rates = packed_spike_rate(packed, dim=-1)
per_neuron_rates = packed_spike_rate(packed, dim=(0, 1))
```

CUDA tensors automatically use the Triton row-count kernel for packed
count/rate reductions when Triton is available. CUDA tensors can also use
`spiker.triton.pack_spikes_triton` and `spiker.triton.unpack_spikes_triton` for
the same representation.
`spiker.lif_forward_packed_spikes(..., backend="auto")` is the public forward
entry point for direct packed output. On CUDA with Triton installed it writes
packed spike words directly and avoids dense spike output allocation; otherwise
it runs the PyTorch reference forward and packs the dense result. That path is
forward-only for now; training kernels still consume dense spike traces.

`spiker.distributed` contains the first packed collective helpers:
`packed_spike_all_gather`, `packed_spike_count_all_reduce`, and
`packed_spike_rate_all_reduce`. These operate on `PackedSpikes` so early
distributed experiments can move or aggregate bitpacked spike payloads instead
of dense float activations. They require an initialized `torch.distributed`
process group. CUDA tensors use a Triton packed row-count fast path for
count/rate reductions when Triton is installed; FSDP and custom sparse
collectives are still future work.

```bash
uv run python -m spiker.benchmarks.distributed_collectives
uv run python -m spiker.benchmarks.distributed_collectives --backend nccl --device cuda --world-size 2
```

## M1 Goal

Implement and benchmark fused-time LIF kernels. The forward kernel has landed,
and the first autograd slices support final-membrane and surrogate spike-output
gradients. Dense-synapse fusion and checkpoint/recompute across time have also
landed for the first training path. The DSL/codegen path now reaches generated
ALIF forward plus generated backward for materialized LIF, fused dense synapse,
and checkpointed dense synapse training. The reference implementation remains
the correctness oracle for optimized kernels.
