# Architecture Notes

## Direction

`myelin` is designed around fused simulation across time. The intended path is:

1. Define neuron dynamics with a restricted Python DSL.
2. Lower the DSL to a backend-agnostic IR.
3. Emit fused-time kernels, starting with Triton.
4. Wrap kernels in PyTorch modules and custom autograd Functions.

The scaffold starts with ordinary PyTorch reference functions. They are the
correctness oracle for future generated kernels and should stay simple.

## Backend Boundary

The current backend boundary is deliberately narrow:

- `myelin.functional.lif_unroll`, `surrogate_lif_unroll`, and
  `surrogate_alif_unroll` are the PyTorch reference implementations.
- `myelin.triton.lif_forward` is the raw fused-time CUDA/Triton forward kernel.
- `myelin.autograd.triton_lif_forward_function` wraps the Triton forward kernel
  in a custom `torch.autograd.Function`. Its backward currently replays the
  PyTorch reference for exact final-membrane gradients.
- `myelin.autograd.triton_surrogate_lif_function` uses the same Triton forward
  family with saved pre-reset membranes and a fused reverse-time Triton
  backward kernel for spike-output gradients.
- `myelin.autograd.generated_triton_surrogate_lif_function` keeps the same
  forward boundary but replaces the handwritten surrogate backward with a
  DSL/codegen-generated reverse-time Triton kernel.
- `myelin.autograd.generated_triton_linear_surrogate_lif_function` extends that
  generated backward foothold to fused dense synapse training for the
  `dinputs`/`dweight`/`dbias` path.
- `myelin.autograd.generated_triton_linear_surrogate_lif_checkpoint_function`
  extends the generated backward path to checkpointed fused dense-synapse
  training by generating the reverse-time chunk kernel.
- `myelin.kernels.lif_forward`, `alif_forward`, `surrogate_lif_forward`,
  `surrogate_alif_forward`, and `linear_surrogate_lif_forward` are the public
  dispatchers for `backend="torch"` and `backend="auto"`. `lif_forward`,
  surrogate LIF, and linear surrogate LIF support `backend="triton"`;
  `alif_forward`, surrogate LIF, and linear surrogate LIF also accept
  `backend="triton_generated"` as explicit opt-in generated paths.
  `surrogate_alif_forward` is currently torch-only and raises for Triton
  backends until the planned ALIF generated backward kernel exists.
- `myelin.modules.TimeUnroll` uses the dispatchers for hard LIF, ALIF,
  hard-forward surrogate LIF, and hard-forward surrogate ALIF cells.

This keeps correctness, raw kernels, autograd integration, and module-level API
separate. It also leaves room for a future DSL/codegen path to target the same
backend interface instead of coupling generated code directly to modules.

`backend="auto"` means:

- CUDA tensor plus Triton installed: use the Triton forward kernel.
- CUDA tensor without Triton: warn and fall back to PyTorch.
- CPU tensor: use PyTorch without warning.

Optional dependency availability is intentionally cached at module import time
in `myelin._optional`. This keeps `backend="auto"` from calling
`importlib.find_spec` inside model forwards, which is important for
`torch.compile(fullgraph=True)` compatibility.

Example training scripts expose float32 matmul precision as an explicit CLI
setting and default to `highest`. TF32-friendly `high` precision is useful for
speed on supported NVIDIA GPUs, but SNN threshold crossings can make training
sensitive to small matmul differences, so examples keep the stricter PyTorch
behavior unless the user opts into `--matmul-precision high`.

The surrogate Triton path stores pre-reset membranes during forward and consumes
them in one reverse-time backward kernel. This is the first kernel-level BPTT
slice. It is fast, but it still has the classic BPTT memory issue because the
full `[T, B, N]` pre-reset trace is saved. Built-in hard-forward surrogate
gradients are `sigmoid`, `fast_sigmoid`, `atan`, `triangular`, and
`superspike`, and `multi_gaussian`. `SurrogateBuilder` provides the first
custom-surrogate authoring boundary for pointwise derivative IRs: users define a
derivative as a restricted expression over `centered` and scalar params, then
evaluate it with PyTorch or render a Python/Triton derivative body. The current
high-level training backend dispatch still accepts built-in surrogate names.

## M1 Boundary

M1 does not need the full DSL. It needs:

- a correct LIF reference implementation,
- a fused Triton LIF forward path, completed,
- a custom final-membrane backward path, completed via reference replay,
- a surrogate spike-output backward path, completed as a fused Triton kernel,
- checkpointing across time for backward memory, completed for the PyTorch
  streamed oracle and for the common Triton no-`dinputs` dense-synapse path,
- benchmark scripts that can compare launch-heavy reference code against the
  fused kernel.

The initial tensor convention is `[T, B, N]` for time, batch, and neurons.

## Neuron DSL v0

`myelin.dsl` is the first narrow DSL/IR foothold. It represents neuron dynamics
as pointwise expression graphs over named state, input, and parameter values.
The current public builder is `lif_ir()`, which lowers hard-reset LIF into:

```text
pre_reset = membrane * decay + input_current
spike = pre_reset >= threshold
next_membrane = where(spike, reset, pre_reset)
output_spike = where(spike, 1.0, 0.0)
```

The evaluator is intentionally PyTorch-based and correctness-first. Its job is
to prove the IR contract and provide a backend-independent oracle. `NeuronIR`
can be constructed directly, but `NeuronBuilder` is the preferred v0 authoring
helper for custom pointwise neurons because it records state/input/parameter
declarations while expressions are created and validates that final expressions
only reference declared names. Names must be unique across state, input, and
parameter declarations because generated kernels lower all symbols into one
kernel-local namespace. Neuron, state, input, parameter, and output names must
also be valid Python-style identifiers and not keywords, so invalid custom IR
fails before the generated Python/Triton source is rendered. Triton emission
should consume this IR later; optimized hand-written kernels remain the current
fast path while the codegen path is built up.
Direct `NeuronIR(...)` construction is still supported for tests and advanced
use, but callers should run `validate_neuron_ir(ir)` before passing custom IR to
evaluators or codegen. The evaluator and lowering path also run this validation
themselves so invalid IR fails before rendering ambiguous code. `validate_expr`
is also public for checking individual expression nodes when building custom
tools on top of the DSL.

```python
from myelin.dsl import NeuronBuilder, where

builder = NeuronBuilder("custom_lif")
membrane = builder.state("membrane")
current = builder.input("input_current")
decay = builder.param("decay")
threshold = builder.param("threshold")
reset = builder.param("reset")

pre_reset = membrane * decay + current
spike = pre_reset.ge(threshold)
ir = builder.build(
    next_state={"membrane": where(spike, reset, pre_reset)},
    outputs={"spike": where(spike, 1.0, 0.0)},
)
```

The CPU oracle for time-major inputs is `evaluate_neuron_unroll`:

```python
from myelin import evaluate_neuron_unroll

final_state, spikes = evaluate_neuron_unroll(
    ir,
    inputs,
    {"membrane": initial_membrane},
    {"decay": 0.9, "threshold": 1.0, "reset": 0.0},
)
```

On CUDA, the same custom IR can be passed to the generated forward helper:

```python
from myelin.triton import generated_neuron_forward

final_state, spikes = generated_neuron_forward(
    ir,
    inputs,
    {"membrane": initial_membrane},
    {"decay": 0.9, "threshold": 1.0, "reset": 0.0},
)
```

For module code, wrap the same IR in `CustomNeuronCell` and use `TimeUnroll`:

```python
from myelin import CustomNeuronCell, TimeUnroll

cell = CustomNeuronCell(ir, {"decay": 0.9, "threshold": 1.0, "reset": 0.0})
unroll = TimeUnroll(cell, backend="auto")
final_state, spikes = unroll(inputs, {"membrane": initial_membrane})
```

On CPU this uses the PyTorch IR evaluator. On CUDA with Triton installed,
`backend="auto"` dispatches to the generated fused-time Triton forward helper.
If no initial state is passed, custom cells initialize every declared state to
zeros with shape `inputs.shape[1:]`. `CustomNeuronCell` enforces the same ABI as
the generated forward helper at construction time, so CPU and CUDA accept the
same module-compatible IR subset.

The current generic generated-forward ABI is deliberately narrow: one time-major
input named `input_current`, at least one state tensor, and exactly one output
named `spike`. That maps directly to the launcher signature
`inputs[T, B, N] -> final_state dict + spikes[T, B, N]`; unsupported shapes fail
validation before any Python/Triton source is rendered.

`analyze_neuron_ir(ir)` provides a non-throwing diagnostic layer for this
boundary. It returns structural errors, warnings about the default
unroll/generated-forward ABI, and booleans for `supports_unroll_api` and
`supports_generated_forward`. It also returns `generated_forward_errors`, which
names the exact generated-backend ABI requirements the IR misses, such as
stateless neurons, non-`input_current` inputs, or extra outputs. The same report
now includes `supports_generated_backward` and `generated_backward_errors`. The
first generated-backward ABI is deliberately narrow: it recognizes a
single-state hard-reset LIF recurrence with `membrane`, `input_current`,
`decay`, `threshold`, `reset`, and one `spike` output. For recognized IRs,
`generated_backward_plan` and `plan_generated_backward_ir(ir)` expose a
`NeuronBackwardPlan` naming the recurrence kind, state/input/output names,
parameter bindings, and saved values required by the backward kernel. Other
custom neuron IRs still report actionable generated-backward errors. ALIF is
now structurally recognized as `alif_adaptive_threshold`, including adaptation
state, adaptation decay, beta, adaptive-threshold scratch, and spike scratch in
the plan. That plan is marked `is_implemented=False`, so
`supports_generated_backward` remains false until a generated reverse-time ALIF
kernel consumes the contract.
`validate_neuron_ir(ir)` remains the strict structural gate;
`validate_generated_forward_ir(ir)` is the stricter forward-backend gate used by
`CustomNeuronCell` and generated Triton rendering, while
`validate_generated_backward_ir(ir)` checks the narrow generated-backward ABI.
`CustomSurrogateNeuronCell` is the first public wrapper that consumes that
backward ABI: it resolves the plan for a builder-authored LIF-shaped IR,
converts the declared `decay`/`threshold`/`reset` params into the existing
`LIFParams` surface, and dispatches through the same surrogate LIF backend path
used by `SurrogateLIFCell`. `LinearCustomSurrogateNeuron` extends that wrapper across a
trainable dense projection; with `stream_synapse=True`, it calls the same fused
dense-synapse surrogate LIF backend used by `LinearSurrogateLIF`.
`LinearCustomSurrogateNeuronRate` exposes the corresponding direct spike-rate
readout path, matching `LinearSurrogateLIFRate`.
`myelin.benchmarks.custom_surrogate_training` compares both the cell wrapper
and the dense linear/rate wrappers against the built-in surrogate LIF paths and
reports loss, final-state, input-gradient, weight-gradient, and bias-gradient
agreement where those quantities apply. It also prints a direct pairwise
built-in-vs-custom table for each backend surface, so generated-backend parity
does not have to be inferred through the torch reference rows.

`examples/custom_neuron_dsl.py` shows custom LIF, ALIF, and refractory-LIF
variants. LIF proves the first generated-backward recognition contract for a
builder-authored hard-reset neuron. ALIF proves multiple state variables, while
refractory LIF proves counter-like state updates encoded with pointwise `where`
expressions instead of Python control flow. Each variant runs first through the
PyTorch evaluator and then through `CustomNeuronCell`/`TimeUnroll`; on CUDA with
Triton installed, `backend="auto"` uses the generic generated forward launcher.
The `myelin.benchmarks.custom_neuron_module` benchmark times this public module
path and records correctness against the evaluator oracle. Its combined
`--variant all` mode keeps the LIF, ALIF, and refractory-LIF composability proof
in one artifact, including per-variant speedups, state error, and spike mismatch
rate. This is still mostly a forward-codegen proof: the current implemented
generated backward ABI only supports hard-reset LIF-shaped custom IRs. ALIF now
has an explicit unimplemented backward plan, while refractory LIF and richer
custom dynamics remain future work for generic generated backward.

`myelin.codegen` lowers this IR to a deterministic SSA-like fragment. For LIF,
the current lowering produces temporaries for membrane decay, pre-reset
membrane, threshold comparison, reset selection, and spike output. It can render
Python-style fragments (`torch.where`) and Triton-style fragments (`tl.where`),
and it can render a Triton step-body fragment with kernel-local variable names.
`myelin.triton.generated` now splices lowered step bodies into full fused-time
Triton forward kernel sources, loads them through `triton.jit`, and verifies
them against reference unrolls. The renderer is shared across neuron IRs: state
names become initial/final state pointers, parameter names become constexpr
kernel arguments, and the IR's `spike` output is written as the time-major
output trace. The generated kernels are not the default fast path yet; the
hand-written kernels remain the optimized implementation while the generated
path grows to support more neuron definitions.

ALIF is the first composability proof beyond LIF. It adds a second state
variable, `adaptation`, plus an adaptive threshold:

```text
adaptive_threshold = threshold + beta * adaptation
spike = pre_reset >= adaptive_threshold
next_adaptation = adaptation * adaptation_decay + spike
```

The existing IR/evaluator/codegen path handles this without new operators, and
ALIF now emits and runs through the same generated fused-time Triton forward
template as LIF. It is also exposed through `myelin.kernels.alif_forward` and
`TimeUnroll(ALIFCell(...), backend="triton_generated")`, so the composability
proof reaches the same public dispatch surface instead of stopping at raw
kernel helpers. That is the intended M2 direction: new pointwise neuron dynamics
should add IR definitions and state/parameter bindings before they require new
kernel templates.

The Izhikevich-style reference neuron is the next proof point. It uses two
states, `voltage` and `recovery`, plus a quadratic voltage update:

```text
voltage_delta = 0.04 * voltage^2 + 5 * voltage + 140 - recovery + input
pre_reset_voltage = voltage + dt * voltage_delta
recovery = recovery + dt * a * (b * voltage - recovery)
spike = pre_reset_voltage >= threshold
```

This goes through the same evaluator and generated fused-time Triton forward
path via `izhikevich_ir()` and `myelin.kernels.izhikevich_forward`. It is still
inside the v1 DSL boundary because the update is pointwise and keeps all state
in fixed-shape tensors.

## Dense Synapse Boundary

The current Triton surrogate kernel starts after dense synaptic projection:

```text
[T, B, F] inputs x [F, N] weight -> [T, B, N] currents -> Triton LIF
```

That is a useful kernel boundary for proving fused LIF dynamics, but it is not
the boundary that solves training memory. At this boundary, autograd still needs
the input features and current gradients to form `dweight`, and the forward pass
has already produced a dense `[T, B, N]` current tensor.

`torch.compile` has a structural advantage in the PyTorch baseline because it
sees the whole workload from input features through matmul, recurrence, loss,
and backward. On a small RTX 5090 check (`T=25, B=64, F=128, N=2048`), both the
materialized-currents and per-timestep-matmul PyTorch workloads compiled down to
the same 3.8 MB CUDA peak, even though eager materialized currents peaked at
130.3 MB and eager per-timestep matmul peaked at 105.3 MB.

A focused allocation audit at the larger checkpoint benchmark shape
(`T=100, B=64, F=128, N=2048`) makes this more explicit. The dense currents
tensor would be 50.0 MB on its own, but the compiled materialized graph showed
no forward peak increment above its baseline allocation in that run. That is
evidence that Inductor can eliminate, rematerialize, or reuse storage for the
nominal `[T, B, N]` currents tensor. We should therefore treat compiled PyTorch
as the serious memory baseline, not merely eager PyTorch.
The comparison has one important boundary caveat: that compiled workload returns
a scalar loss from inside the compiled graph. Public dense-output Triton paths
return `[T, B, N]` spikes to Python before the loss consumes them, so they have a
dense-output allocation lower bound that the scalar compiled graph can avoid.

`myelin.benchmarks.compile_inspect` is the current way to inspect that behavior
instead of guessing. On the CUDA smoke shape (`T=16, B=8, F=16, N=32`), the
compiled materialized workload had one Dynamo graph, no graph breaks, two
Inductor Triton kernels, two `extern_kernels.mm` calls, 24 CUDA allocation
sites, 19 `reinterpret_tensor` calls, and 17 explicit buffer `del` calls in the
generated code. The looped source had more matmul calls and allocation sites,
but also 27 direct buffer-reuse assignments. The practical lesson is that the
compiled baseline is not just fusing math; it is also aggressively planning
scratch lifetimes and view aliasing across forward and backward.

Implication: the next Triton training boundary should include the dense synapse
for `LinearSurrogateLIF`, not only the LIF recurrence:

```text
[T, B, F] inputs, [F, N] weight -> fused/streamed synapse + LIF -> loss grads
```

That larger boundary lets us stream or recompute currents during backward and
accumulate `dweight` directly, which is the path toward matching
`torch.compile`'s memory behavior while keeping the fused-time SNN control.

`linear_surrogate_lif_forward(...)` is the public backend-selected function for
that boundary. `LinearSurrogateLIF(stream_synapse=True)` delegates to the same
path. With `backend="torch"`, it is the PyTorch custom-autograd oracle: it
computes `[B, N]` currents one timestep at a time, stores the recurrent
spike/pre-reset trace, and accumulates `dweight`, `dbias`, and optional
`dinputs` directly in its backward pass. It is not the final fast backend; it
defines the contract the Triton synapse+LIF kernels must match. On the same
small RTX 5090 check, this streamed Python autograd path reduced eager peak from
157.3 MB to 134.3 MB, but remained much slower than the current Triton LIF-only
backend and far above `torch.compile`'s 6.8 MB peak. That confirms the boundary
is right, but the implementation needs to be a fused Triton kernel rather than a
Python custom-autograd loop.

The first fused synapse+LIF Triton slice is now forward plus an explicit
backward:

```text
[T, B, F] inputs, [F, N] weight, optional [N] bias
        -> fused dense current + hard LIF forward
        -> spikes, final membrane, pre-reset trace
        -> reverse-time LIF gradient + Triton dinput/dweight/dbias kernels
```

It proves the larger kernel boundary and avoids materializing forward currents.
When `dinputs` are not needed, the current backward accumulates `dweight` and
`dbias` during the reverse recurrence instead of materializing a `[T, B, N]`
current-gradient scratch. That matches the common supervised-training case
where dataset inputs do not require gradients. If `dinputs` are needed, the
fallback still materializes the scratch so it can multiply by `weight.T`.

This is a good correctness and performance foothold, but not the final memory
story. The remaining large BPTT cost is the saved full `[T, B, N]` pre-reset and
spike trace. The next optimization is checkpointing/recompute across time so
the trace does not scale linearly with long sequences.

`LinearSurrogateLIF(stream_synapse=True, checkpoint_size=...)` is the first
PyTorch oracle for that checkpoint contract. Forward stores only chunk-start
membrane states plus the returned spikes, and backward recomputes each chunk's
pre-reset/spike trace before applying the reverse recurrence. On an RTX 5090
shape (`T=100, B=64, F=128, N=2048`), this reduced the streamed PyTorch peak
from 326.1 MB to 180.1 MB with `checkpoint_size=25`, at the expected cost of a
slower Python backward. The next Triton step is to move this chunked recompute
pattern into the fused synapse+LIF backward.

The first Triton checkpoint/recompute slice now exists for the common
training case and now also supports optional `dinputs`. It stores chunk-start
membranes instead of full pre-reset traces and accumulates `dinputs`, `dweight`,
and `dbias` during backward. The backward uses a chunk-scratch strategy:
recompute one chunk into a bounded `[chunk, B, N]` scratch buffer, then run the
reverse chunk recurrence and pass the chunk-start gradient to the previous
chunk. In the shared benchmark harness (`T=100, B=64, F=128, N=2048`), the
no-`dinputs` path ran in 0.756 ms with a 144.1 MB backward peak, compared with
0.723 ms and 228.6 MB for the full-trace fused Triton path. Measured as an
increment above baseline allocation, the checkpoint dense-output path needs
52.0 MB in forward: 50.0 MB for the returned dense spike tensor plus 2.0 MB for
chunk-start membrane state. Its backward increment is 66.0 MB, or 16.0 MB above
the dense output lower bound, roughly matching chunk-start state plus one
12.5 MB checkpoint scratch buffer. The compiled materialized graph still reports
no currents-sized increment because the scalar loss stays inside the compiled
graph. The `dinputs` path
does extra weight-transpose work during each chunk backward, so it should be
benchmarked separately for input-optimization workloads.

`LinearSurrogateLIF` now uses the lower-memory Triton checkpoint path whenever
`checkpoint_size` is provided for `backend="triton"` or
`backend="triton_generated"`, including when `inputs.requires_grad` is true.
Checkpoint size can be a positive integer or one of the policy strings
`"memory"`, `"balanced"`, or `"speed"`. The policies resolve from the input
time dimension through `recommended_checkpoint_size`; for `T=100`, they map to
`10`, `25`, and `50`.
`LinearSurrogateLIFRate` exposes the spike-rate path as a module; it shares the
same dense projection and surrogate LIF configuration but returns scalar or
`[B, N]` rates instead of dense `[T, B, N]` spikes. Torch, handwritten Triton,
and generated Triton rate backends all use direct checkpoint rate outputs;
backward recomputes chunks and distributes rate gradients across time. The
generated path swaps in generated reverse-time chunk code for the backward pass.
On the same audit shape, the handwritten and generated checkpoint rate-output
paths had a forward increment of only 2.0 MB and a backward increment of
16.5 MB. This is the right path for rate objectives, but it is an explicit
specialized output contract rather than a replacement for dense spike-output
BPTT.

`myelin.benchmarks.scalar_loss_boundary` is the current like-for-like-ish
latency probe against `torch.compile`'s scalar-loss graph. It keeps the
Triton rate objective inside the checkpointed backend boundary instead of
returning dense spikes. On the shared RTX 5090 shape with
`matmul_precision=high`, the handwritten Triton scalar-rate checkpoint path and
the compiled materialized PyTorch hard-forward straight-through graph are within
run noise: 0.671 ms versus 0.658 ms fwd+bwd on the latest run. The compiled
graph still reports no incremental CUDA allocation, while the default Triton
scalar-rate path reports the expected 16.5 MB checkpoint scratch increment. A
memory-minimal replay variant avoids the chunk pre-reset scratch and drops the
increment to 3.5 MB, but latency rises to 4.726 ms because it replays chunk
prefixes inside the backward kernel. The hard-forward PyTorch row uses
thresholded spikes in forward and a fast-sigmoid straight-through gradient,
making it a closer semantic comparison to the Triton hard-forward surrogate path
than the older soft-forward rows.

`myelin.benchmarks.checkpoint_size_sweep` now includes that replay-rate path
across chunk sizes and prints the rate-output Pareto frontier. The result is
useful but not a new default: checkpoint `5` replay-rate took 1.339 ms with an
11.5 MB increment, while the scratch-backed rate path at checkpoint `10` took
0.804 ms with a 12.0 MB increment on the latest RTX 5090 run. Replay is
therefore best treated as a memory-floor diagnostic unless a future kernel can
avoid the full O(checkpoint_size^2) prefix replay.

The generated Triton checkpoint path keeps the same recompute kernel and swaps
in generated reverse-time chunk code. It supports the same input, weight, and
bias gradient contract as the handwritten chunk backward kernel, which makes the
checkpointed memory path part of the DSL/codegen proof instead of being only
handwritten Triton. Generated training paths are now compile-compatible when
their kernels are materialized before Dynamo traces the model. The generated
loader cache is keyed before source rendering, and `train_mnist_rate.py`
prewarms generated kernels before `torch.compile`, avoiding Python source
rendering, `exec`, and `linecache` mutation inside the compiled autograd path.

## Bitpacked Spike Format

`myelin.packing` defines the first M4 spike storage contract. Spikes are packed
along the last dimension into signed int32 words:

```text
[T, B, N] -> [T, B, ceil(N / 32)]
```

Bit `i` inside a word represents neuron offset `i` within that group of 32.
The implementation uses signed int32 storage because that maps directly to a
portable PyTorch dtype; the high bit is represented as a negative int32 value
and unpacks correctly through bitwise masking.

This format gives a fixed 32x storage reduction versus float32 dense spikes
when `N` is divisible by 32. It is deliberately dense-bitpacked rather than
index-sparse: the next Triton kernels can compute word offsets directly without
per-row sparse metadata, and sparse-aware collectives can still build on the
same representation later.

The format now has both PyTorch and Triton pack/unpack implementations. On the
RTX 5090 benchmark shape `T=100, B=64, N=2048`, Triton packing produced the
same round-trip result and reduced pack time from 0.317 ms to 0.015 ms.
Basic metrics can be computed from the packed representation with
`packed_spike_counts`, `packed_spike_count`, and `packed_spike_rate`; these
helpers validate the packed shape and ignore padding bits in the last word.
`packed_spike_counts` reduces over the packed neuron dimension and returns one
count per original row, for example `[T, B]` from `[T, B, N]` spikes. It also
supports packed-safe reductions such as `dim=(0, -1)` for per-batch counts or
`dim=(1, -1)` for per-timestep counts. Reductions that keep the original
neuron dimension are available through `packed_spike_count`, which unpacks as a
convenience path.

`myelin.lif_forward_packed_spikes` is the public backend-selecting entry point
for packed LIF forward. With CUDA tensors and Triton installed, it calls the
direct Triton writer instead of materializing dense spikes. It stores 50.0 MB of
dense spikes as 1.6 MB of packed words at `T=100, B=64, N=2048`. After batching
multiple rows per Triton program, direct packed forward runs in 0.028 ms versus
0.056 ms for dense forward on that shape. The Torch fallback still computes the
dense reference result and then packs it. Training kernels still use dense spike
outputs.

The first training-facing packed trace slice is in the handwritten Triton
surrogate LIF autograd path. The forward still returns dense spikes to preserve
the public differentiable API, but the custom autograd context saves the spike
trace as packed int32 words and the backward kernel reads those bits directly.
On the shared `T=100, B=64, N=2048` materialized-current benchmark, this reduced
the Triton LIF backward peak from about 281 MB to 233 MB. This does not yet make
the whole training interface packed, but it proves packed traces can participate
in BPTT kernels without unpacking to a dense saved tensor.

## Distributed Foothold

`myelin.distributed` is the first M5 surface. It intentionally starts with the
packed spike format rather than a full training strategy:

- `packed_spike_all_gather` all-gathers packed int32 spike words and preserves
  the `PackedSpikes.original_shape` metadata for each rank.
- `packed_spike_count_all_reduce` computes packed-safe spike counts locally and
  sums those counts across ranks.
- `packed_spike_rate_all_reduce` turns those summed counts into global rates,
  assuming each rank contributes the same local shape.
- `wrap_fsdp_if_initialized` wraps a module in PyTorch FSDP when a distributed
  process group is initialized, and otherwise returns the original module unless
  `require_initialized=True` is passed.

These helpers require an initialized `torch.distributed` process group. They do
not replace tensor parallelism or custom sparse collectives yet; they make the
packed activation representation and a conservative FSDP wrapping boundary
usable from tested public primitives. `myelin.benchmarks.distributed_collectives`
is the first benchmark harness for this surface and now reports correctness
columns next to timing: packed all-gather must unpack to the dense all-gather
payload, and count/rate all-reduce must match dense count/rate reductions with
zero max error. On a local two-rank Gloo run at `T=100, B=64, N=2048`, dense
float all-gather moves a 50.0 MB spike payload per rank and took about
103.1 ms, while packed all-gather moves 1.6 MB per rank and took about
1.6 ms. The CUDA path uses `packed_spike_counts_triton` for row counts over the
packed neuron dimension before the distributed all-reduce. On the same shape
with single-GPU NCCL (`world_size=1`), that path runs packed count/rate in about
0.078/0.104 ms. That validates the CUDA primitive, but a real multi-GPU NCCL
benchmark is still required before claiming distributed training speedups.

## Online Learning Foothold

`myelin.online.linear_lif_online_eligibility_grad` is the first M6 reference
primitive. It accepts time-major dense inputs, dense output weights, optional
bias, and a per-timestep learning signal interpreted as `dL/d spike`. It
updates an eligibility trace online:

```text
eligibility[t] = decay * eligibility[t - 1] + input[t]
grad_weight += learning_signal[t] * surrogate_derivative[t] * eligibility[t]
```

The helper returns hard spikes, final membrane state, and local weight/bias
gradients. It is deliberately labeled as an online estimate rather than BPTT:
it ignores reset-gradient and future-state terms. Tests compare it against
autograd in a no-reset regime where the eligibility recurrence is exact, which
makes it a useful correctness oracle before adding ALIF/e-prop variants or a
larger online-training stack. Because this LIF rule ignores reset/future-state
feedback, its membrane eligibility is shared across output neurons and is stored
as `[B, F]`; the weight gradient is formed with an outer product against the
neuron-local learning factor. The functional online helpers accept either a
built-in surrogate name or a custom `SurrogateBuilder` derivative IR with
explicit `surrogate_params`, so custom pointwise derivatives can participate in
this learning rule before they are threaded through generated Triton backward
kernels. `examples/custom_surrogate_online.py` is the smallest public workflow:
it builds a parameterized derivative IR, passes it to `LinearOnlineLIF`, and
checks the module wrapper against the functional online oracle.

`linear_alif_online_eligibility_grad` extends the same reference path to ALIF
with adaptation-aware eligibility. Its local gradient uses the effective
centered trace `d(pre_reset - beta * adaptation) / d weight` and keeps that
centered trace directly as `[B, F, N]`, avoiding a separate adaptation
eligibility tensor. The same e-prop/OSTL caveat applies: reset-gradient and
full future-state BPTT are not modeled exactly.

`LinearOnlineLIF` and `LinearOnlineALIF` are the module wrappers for these
rules. Their `forward` methods return local gradient estimates for
caller-controlled learning signals, and `step_online(..., lr=...)` applies the
local update directly to the owned dense synapse parameters. They accept the
same custom `SurrogateBuilder` derivative IRs as the functional helpers. This
keeps online learning explicit and separate from autograd-based BPTT modules.

`myelin.benchmarks.online_learning` compares the LIF/ALIF online update cost
against surrogate BPTT baselines on the same dense workload. It is meant to
track the online-learning path independently from fused BPTT kernels. The
benchmark also includes custom surrogate IR variants that alias the built-in
fast-sigmoid derivative, so the current generic IR-evaluator overhead is visible
beside the built-in path. Those custom derivative IRs are resolved to reusable
Python callables before the online time loop, keeping the LIF custom path within
benchmark noise of the built-in derivative path at the current CUDA shape.

## Hardware Bridge Foothold

`myelin.hardware` defines the first M7 export boundary:

```text
dense weight [F, N]
optional bias [N]
LIF scalars: tau_mem, decay, threshold, reset
timestep metadata: dt
string metadata
        |
        v
myelin.dense_lif.v0 JSON artifact
        |
        v
signed symmetric fixed-point preparation
        |
        v
myelin.dense_lif_quantized.v0 JSON artifact
        |
        v
generic core tiling / accumulator marking
        |
        v
myelin.dense_lif_placement.v0 JSON artifact
        |
        v
artifact manifest
        |
        v
myelin.hardware_bundle.v0 JSON artifact
        |
        v
target adapter manifest
        |
        v
myelin.spinnaker2_dense_lif_manifest.v0 JSON artifact
```

`export_dense_lif_layer` accepts raw tensors plus `LIFParams` and returns a
`DenseLIFHardwareExport` dataclass that can be serialized with `to_json()`.
`export_linear_lif_module` handles `LinearLIF`-style modules by reading their
owned `LinearSynapse` weights and LIF cell parameters. `read_hardware_export`
and `dense_lif_hardware_export_from_dict` parse and validate artifacts on the
way back in.

`quantize_dense_lif_export` derives a fixed-point artifact from the float
schema. It uses signed symmetric per-tensor quantization with explicit
`num_bits`, `qmin`, `qmax`, `weight_scale`, and `bias_scale` metadata, and
`dequantize_dense_lif_export` provides a simple round-trip check. The float
artifact remains the canonical compatibility boundary; quantized artifacts are
target-prep artifacts that hardware adapters can further constrain or reject.

`plan_dense_lif_placement` adds the first routing-oriented artifact. It tiles
the quantized dense layer into core-local input and output ranges, records each
tile's synapse count, and marks tiles that require cross-core accumulation
because the same output range receives multiple input shards. This is still
generic: it does not choose SpiNNaker vertices, packet formats, or host/runtime
APIs. It gives those future adapters a concrete plan to accept, refine, or
reject.

`export_linear_lif_hardware_bundle` runs this full generic pipeline for a
`LinearLIF`-style module and writes the float export, quantized export,
placement plan, and `myelin.hardware_bundle.v0` manifest into a single
directory. The manifest keeps artifact filenames, format versions, target name,
and summary counts together so a future SpiNNaker adapter has one stable entry
point.

`examples/export_hardware_bundle.py` is the runnable smoke path for this bridge.
It creates a deterministic `LinearLIF`, prints a model parameter summary, writes
the bundle directory, and reports artifact paths in Markdown tables. Its
`--adapter spinnaker2` mode keeps the primary bundle generic while also writing
the SpiNNaker2 placement/manifest, which makes the current M7 handoff surface
reproducible from one command.

`export_spinnaker2_dense_lif_manifest` is the target-specific adapter artifact.
It accepts a quantized dense LIF export plus a placement plan whose target is
`spinnaker2`, validates per-core neuron and incoming-synapse limits, records
timestep in milliseconds, and emits `myelin.spinnaker2_dense_lif_manifest.v0`.
The manifest references the generic quantized and placement artifacts instead of
duplicating weights. Dense input shard accumulation is preserved in each mapping
entry through `requires_accumulator` so a later SpiNNaker lowerer can decide how
to route partial sums. It is not a SpiNNaker SDK program yet; it is a validated
handoff object for a future SDK lowering pass.

The bridge still needs real target SDK adapters and checks that hardware
timestep/reset semantics match the training-time model.

## TimeUnroll

`TimeUnroll` is the small module-level adapter that runs one recurrent cell over
a time-major sequence. Conceptually:

```text
input_current[T, B, N]
        |
        v
for t in 0..T-1:
    state, spike[t] = cell(input_current[t], state)
        |
        v
spikes[T, B, N], final_state[B, N]
```

For hard `LIFCell`, `TimeUnroll` can dispatch to `myelin.kernels.lif_forward`,
which lets `backend="auto"` choose the fused Triton forward kernel on CUDA. For
hard-forward `SurrogateLIFCell`, it can dispatch to
`myelin.kernels.surrogate_lif_forward`; Triton handles the forward pass and the
custom autograd boundary dispatches to the fused Triton backward kernel.
