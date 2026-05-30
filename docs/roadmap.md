# Roadmap Status

This file tracks the original milestone plan against the current repository
state. It is intentionally evidence-based: each status points to implementation
or benchmark artifacts that can be re-run or inspected.

The current v0 training-path recommendation is tracked separately in
`docs/training_recommendation.md`.

## Summary

| Milestone | Status | Current Evidence | Main Remaining Gap |
|---|---|---|---|
| M1 fused-time LIF proof of concept | Mostly complete | `spiker.kernels.lif_forward`, Triton forward/backward surrogate paths, `benchmarks/results/lif_triton_forward_5090.md`, `benchmarks/results/surrogate_backend_generated_triton_5090.md` | Broader third-party benchmark comparison and more tuned kernels |
| M2 neuron DSL v0 + composability | Partial but real | `NeuronBuilder`, custom ALIF/refractory-LIF examples, generic `generated_forward`, generated LIF/ALIF/Izhikevich forward, `benchmarks/results/generated_forward_5090.md` | Generated kernels beyond pointwise forward |
| M3 checkpoint across time | Partial | PyTorch oracle plus handwritten/generated Triton checkpoint paths with optional input gradients, `benchmarks/results/surrogate_backend_time_scaling_5090.md`, `benchmarks/results/checkpoint_size_sweep_5090.md` | Tuning against compiled PyTorch memory behavior at long `T` |
| M4 bitpacked spikes | Partial | `PackedSpikes`, PyTorch/Triton pack/unpack/count/rate, public `lif_forward_packed_spikes`, `linear_surrogate_lif_packed_forward`, and `LinearSurrogateLIFPacked`, packed saved spike traces, `benchmarks/results/packed_forward_5090.md`, `benchmarks/results/bitpack_5090.md` | Public training outputs and some internal spike tensors still use dense storage |
| M5 distributed | Started | Packed spike all-gather/count/rate all-reduce helpers, CUDA packed row-count kernel, `wrap_fsdp_if_initialized`, two-rank Gloo smoke, and local NCCL smoke artifacts | Custom sparse/packed collectives and real multi-GPU benchmarks |
| M6 online learning rules | Started | LIF and ALIF dense eligibility-trace oracles, `LinearOnlineLIF`/`LinearOnlineALIF` wrappers, and CPU/CUDA `online_learning` benchmarks | Richer e-prop variants and lower-memory online traces |
| M7 hardware bridge | Started | `spiker.hardware` dense LIF JSON export schema, signed symmetric quantized export schema, generic placement plan, bundle manifest, SpiNNaker 2 adapter manifest, and round-trip tests | Target SDK object generation |

## What Is Working

- CPU-first reference implementations exist for LIF, ALIF, surrogate LIF,
  surrogate ALIF, and dense-synapse surrogate training.
- Built-in surrogate-gradient functions include sigmoid, fast sigmoid, ATan,
  triangular, SuperSpike, and multi-Gaussian derivatives across the Python,
  DSL/codegen, and Triton backward paths. `SurrogateBuilder` adds the first
  public custom surrogate derivative authoring path for pointwise derivatives,
  with PyTorch evaluation and derivative-body rendering.
- Public backend dispatchers exist for `lif_forward`, `alif_forward`,
  `surrogate_lif_forward`, `surrogate_alif_forward`,
  `linear_surrogate_lif_forward`, `linear_surrogate_lif_rate_forward`,
  `lif_forward_packed_spikes`, and `linear_surrogate_lif_packed_forward`.
- `backend="auto"` uses Triton on CUDA when available and falls back to PyTorch
  otherwise.
- `NeuronBuilder` provides the first public authoring helper for custom
  pointwise neuron IRs, and custom builder-authored IRs can run through the
  generic generated Triton forward launcher. `CustomNeuronCell` wraps this IR in
  the module API, so `TimeUnroll(..., backend="auto")` uses the PyTorch evaluator
  on CPU and the generated fused-time Triton path on CUDA. The
  `custom_neuron_module` benchmark shows the public module path matching the
  evaluator with zero spike mismatches and a large CUDA speedup.
- The custom DSL example includes ALIF and refractory-LIF variants. Refractory
  LIF is the current recipe for avoiding Python-side data-dependent control
  flow: encode the refractory branch as tensor predicates and `where`
  expressions so the same IR can lower into Triton.
- `analyze_neuron_ir` provides non-throwing DSL diagnostics and reports whether
  a custom IR fits the current public unroll/generated-forward ABI, including
  exact generated-forward ABI misses through `generated_forward_errors`.
  It now recognizes the first generated-backward ABI for builder-authored
  hard-reset LIF IRs and returns `supports_generated_backward=True` only for
  that shape. Recognized IRs now also expose a `NeuronBackwardPlan`, which is
  the first explicit contract between DSL analysis and generated backward
  kernels. ALIF is now recognized as an unimplemented `alif_adaptive_threshold`
  backward plan, separating structural planning from backend availability.
  `CustomSurrogateNeuronCell` wires the implemented LIF shape into the public
  module API by dispatching through the existing surrogate LIF backend path, and
  `LinearCustomSurrogateNeuron` extends it across a trainable dense projection
  with the same streamed/checkpointed backend options.
  `LinearCustomSurrogateNeuronRate` adds the matching low-memory rate-readout
  wrapper. Refractory LIF and richer custom dynamics still return actionable
  `generated_backward_errors` without a plan.
- Generated Triton forward works for LIF, ALIF, and Izhikevich-style neurons,
  proving multi-state and quadratic-update paths through the DSL/IR/codegen
  stack.
- Generated Triton backward paths exist for materialized surrogate LIF, fused
  dense-synapse surrogate LIF, and checkpointed dense-synapse surrogate LIF.
- Fused dense-synapse Triton training avoids materializing forward currents.
- Checkpoint/recompute across time exists for dense-synapse Triton training,
  including optional input gradients.
- Bitpacked spike storage is specified and implemented in both PyTorch and
  Triton, with direct packed LIF forward output on CUDA and public packed
  count/rate helpers that auto-dispatch to Triton row counts on CUDA.
- The handwritten and generated materialized Triton surrogate LIF autograd
  paths now save their spike traces in packed form and read packed bits directly
  during backward.
- Generated fused dense-synapse Triton autograd also saves its spike trace in
  packed form while keeping dense spike outputs for user-facing losses.
- Handwritten and generated checkpointed dense-synapse Triton backward derive
  hard spikes from per-chunk pre-reset membrane scratch, avoiding separate
  checkpoint spike scratch.
- `spiker.distributed` provides the first packed spike collective helpers:
  packed all-gather and packed count/rate all-reduce over initialized
  `torch.distributed` process groups. `wrap_fsdp_if_initialized` provides a
  conservative FSDP integration boundary for distributed launches. The local
  two-rank Gloo benchmark shows the packed payload path clearly, while also
  showing that CPU/Gloo count/rate reductions need a faster CUDA-aware
  implementation. `spiker.triton.packed_spike_counts_triton` is the first CUDA
  row-count fast path for those reductions.
- `linear_surrogate_lif_rate_forward` provides specialized checkpointed
  spike-rate objectives that return either a scalar rate or `[B, N]` rates
  instead of a dense `[T, B, N]` spike tensor. Torch, Triton, and generated
  Triton backends now all use direct rate outputs with chunk recompute backward.
  `LinearSurrogateLIFRate` exposes that path through the module API.
- `linear_lif_online_eligibility_grad` and
  `linear_alif_online_eligibility_grad` provide the first online-learning
  oracles: dense eligibility-trace gradient estimates from local learning
  signals. `LinearOnlineLIF` and `LinearOnlineALIF` wrap the same rules in
  modules with explicit `step_online` parameter updates. The ALIF oracle stores
  the centered eligibility trace directly, reducing CUDA peak memory versus the
  earlier adaptation-trace implementation. The functional online helpers and
  module wrappers now accept custom `SurrogateBuilder` derivative IRs, giving
  custom surrogates a real learning-rule path before full generated-backward
  integration. `examples/custom_surrogate_online.py` shows that workflow through
  `LinearOnlineLIF` and validates it against the functional oracle.
  `spiker.benchmarks.online_learning` compares these online updates against
  surrogate BPTT baselines on CPU and CUDA.
- `spiker.hardware` provides the first hardware-bridge compatibility artifact:
  a validated `spiker.dense_lif.v0` JSON schema for dense LIF weights, optional
  bias, neuron parameters, timestep metadata, and string metadata.
  `quantize_dense_lif_export` derives a `spiker.dense_lif_quantized.v0`
  fixed-point artifact with explicit bit width, integer range, and scales.
  `plan_dense_lif_placement` derives a `spiker.dense_lif_placement.v0`
  core-tiling artifact with accumulator requirements for split input shards.
  `export_linear_lif_hardware_bundle` writes the float, quantized, placement,
  and manifest artifacts for a `LinearLIF`-style module into one directory.
  `export_spinnaker2_dense_lif_manifest` derives a
  `spiker.spinnaker2_dense_lif_manifest.v0` handoff artifact with per-core
  neuron and incoming-synapse constraints and mapping ranges for a later
  SpiNNaker 2 SDK lowering pass.
  `examples/export_hardware_bundle.py --adapter spinnaker2` is the runnable
  smoke path and writes the target handoff manifest from one deterministic
  layer.
- MNIST, convolutional MNIST, low-memory MNIST rate-readout, spike-rate, and
  custom LIF rate-readout, and snnTorch comparison examples are present, with
  model summaries, step-time logging, grad clipping, and optional W&B logging
  where the example supports tracking.
- CPU CI now covers formatting, linting, type checking, examples, and tests.
- `spiker.benchmarks.runner` provides smoke/core CUDA benchmark presets with
  dry-run command tables and markdown artifact output, so the key 5090 readouts
  can be regenerated from one entry point. The runner now includes the hardware
  export smoke, compile-visible Triton boundary/sweep, and distributed
  packed-collective Gloo and NCCL specs; the hardware export spec does not
  receive a device argument, Gloo overrides the benchmark device to CPU, and
  NCCL follows the CUDA runner device.
- `.github/workflows/gpu-benchmarks.yml` wires that runner into a manual
  self-hosted CUDA workflow and uploads the generated markdown artifacts.

## Performance Frontier

`spiker.benchmarks.performance_frontier` is the canonical compiled-vs-Triton
comparison. It separates equal-contract rows from explicit SNN-specific output
contracts so we do not overread one-off benchmark artifacts.

| Contract | Variant | Shape | Current Readout |
|---|---|---|---|
| Hard LIF forward | `torch.compile` vs Triton fused-time | `T=100, B=64, N=2048` | 0.140 ms compiled vs 0.057 ms Triton; Triton 2.46x faster, compile warmup 1172.7 ms |
| Dense-output training | `torch.compile` materialized graph vs Triton checkpoint recompute | `T=100, B=64, F=128, N=2048` | 0.747 ms compiled vs 0.739 ms Triton; effectively tied, Triton backward increment 66.4 MB |
| Rate/scalar training | `torch.compile` materialized graph vs Triton checkpoint rate output | `T=100, B=64, F=128, N=2048` | 0.747 ms compiled baseline vs 0.697 ms Triton rate and 0.718 ms generated Triton rate; Triton 1.04-1.07x faster, 16.9 MB backward increment |

`spiker.benchmarks.training_breakdown` isolates the training pieces behind that
tie on the same `T=100, B=64, F=128, N=2048` workload:

| Component | Path | Current Readout |
|---|---|---|
| Dense projection | `torch.matmul` | 0.067 ms |
| Full training | `torch.compile` materialized scalar loss | 0.483 ms |
| Checkpoint forward | Triton dense spikes | 0.116 ms |
| Checkpoint forward | Triton packed spikes | 0.117 ms |
| Checkpoint forward | Triton rate output | 0.112 ms |
| Checkpoint backward | Triton recurrent only | 0.195 ms |
| Checkpoint backward | Triton recurrent + dweight | 0.274 ms |
| Checkpoint backward | Triton recurrent + dweight + dinput | 0.382 ms |
| Checkpoint backward | Triton rate recurrent + dweight | 0.210 ms |

Interpretation: Triton clearly wins the fused-time forward kernel, but plain LIF
training is now a close fight with `torch.compile`. The credible path to a large
win is not rewriting the same dense graph in Triton; it is changing the SNN
contract with direct rate/scalar outputs, packed traces, sparse communication,
generated custom-neuron kernels, and compile-visible Triton operators that let
Inductor optimize the launch boundary and downstream tensor work.

## Historical Supporting Readouts

The rows below are retained as supporting evidence. Use the performance frontier
above for strategy-level compiled-vs-Triton conclusions.

| Result | Shape | Current Readout |
|---|---|---|
| Generated LIF forward | `T=100, B=64, N=2048` | 0.097 ms generated vs 6.824 ms Torch, 70.66x |
| Generated ALIF forward | `T=100, B=64, N=2048` | 0.159 ms generated vs 9.594 ms Torch, 60.47x |
| Generated Izhikevich forward | `T=100, B=64, N=2048` | 0.354 ms generated vs 18.009 ms Torch, 50.93x |
| Custom ALIF module | `T=100, B=64, N=2048` | 0.076 ms generated vs 30.245 ms Torch, 396.54x, zero spike mismatch |
| Custom refractory-LIF module | `T=100, B=64, N=2048` | 0.084 ms generated vs 50.495 ms Torch, 600.54x, zero spike mismatch |
| Custom surrogate LIF training smoke | cell `T=16, B=8, N=64`; linear/rate `F=128, N=64` | custom IR cell generated Triton 0.404 ms vs built-in Torch 6.200 ms, 15.36x; linear custom generated stream 0.448 ms vs linear built-in Torch 6.152 ms, 13.74x; rate custom generated 0.461 ms vs rate built-in Torch 8.490 ms, 18.40x; built-in-vs-custom pairwise loss/grad errors are zero for torch and generated backends |
| Generated multi-Gaussian surrogate backward | `T=100, B=64, N=2048` | 0.137 ms generated vs 0.128 ms handwritten Triton, same 201.0 MB peak |
| Triton LIF with packed saved spikes | `T=100, B=64, F=128, N=2048` | 0.697 ms fwd+bwd, 233.2 MB backward peak |
| Generated Triton LIF with packed saved spikes | `T=100, B=64, F=128, N=2048` | 0.711 ms fwd+bwd, 234.2 MB backward peak |
| Triton synapse training | `T=100, B=64, F=128, N=2048` | 0.696 ms fwd+bwd, 234.6 MB backward peak |
| Generated Triton synapse with packed saved spikes | `T=100, B=64, F=128, N=2048` | 0.799 ms fwd+bwd, 186.2 MB backward peak |
| Triton checkpoint recompute | `T=100, B=64, F=128, N=2048` | 0.757 ms fwd+bwd, 144.1 MB backward peak |
| Triton checkpoint rate output | `T=100, B=64, F=128, N=2048` | 0.700 ms fwd+bwd, 95.6 MB backward peak |
| Generated Triton checkpoint rate output | `T=100, B=64, F=128, N=2048` | 0.748 ms fwd+bwd, 96.6 MB backward peak |
| Torch direct rate output | `T=100, B=64, F=128, N=2048` | 42.638 ms fwd+bwd vs 46.289 ms dense-rate pattern, 105.2 MB vs 175.1 MB |
| Triton checkpoint with `dinputs` | `T=100, B=64, F=128, N=2048` | 0.874 ms fwd+bwd, 154.3 MB backward peak |
| Generated checkpoint with `dinputs` | `T=100, B=64, F=128, N=2048` | 0.947 ms fwd+bwd, 155.3 MB backward peak |
| `torch.compile` materialized baseline | `T=100, B=64, F=128, N=2048` | 0.742 ms fwd+bwd, 16.1 MB backward peak |
| Triton checkpoint time scaling | `T=100/200/500, B=64, F=128, N=2048` | 0.890/1.556/3.720 ms fwd+bwd, 151.5/206.7/372.0 MB |
| Generated checkpoint time scaling | `T=100/200/500, B=64, F=128, N=2048` | 0.950/1.646/3.640 ms fwd+bwd, 152.5/207.7/373.0 MB |
| `torch.compile` time scaling | `T=100/200/500, B=64, F=128, N=2048` | 0.785/1.918/3.198 ms fwd+bwd, 16.1/19.3/28.6 MB; T=500 compile warmup was 124.4 s |
| Triton checkpoint spike-rate output | `T=100/200/500, B=64, F=128, N=2048` | 0.847/1.292/2.612 ms fwd+bwd, 25.0/30.1/45.5 MB |
| Materialization audit | `T=100, B=64, F=128, N=2048` | `torch.compile` scalar-loss graph: 0.404/0.511 ms fwd/bwd and 0.0 MB increment; Triton checkpoint dense output: 0.235/0.518 ms, 52.0 MB fwd increment, only 2.0 MB over dense-output lower bound |
| Materialization audit with rate output | `T=100, B=64, F=128, N=2048` | Handwritten/generated Triton checkpoint rate output: 0.04x currents forward increment, 16.5 MB backward increment |
| Checkpoint-size sweep | `T=100, B=64, F=128, N=2048` | Pareto frontier now printed. Regular rate checkpoint `10`: 0.804 ms / 12.0 MB; checkpoint `25`: 0.689 ms / 16.5 MB; checkpoint `50`: 0.673 ms / 29.0 MB; replay-rate memory floor checkpoint `100`: 17.569 ms / 2.0 MB |
| Scalar-loss boundary | `T=100, B=64, F=128, N=2048` | compiled materialized hard-forward straight-through 0.658 ms / 0.0 MB increment; Triton scalar-rate checkpoint 0.671 ms / 16.5 MB increment; replay-no-scratch 4.726 ms / 3.5 MB increment |
| `torch.compile` code inspection | `T=16, B=8, F=16, N=32` | One captured graph, no graph breaks; materialized source emitted 2 Inductor Triton kernels, 2 `mm` calls, 24 alloc sites, 19 alias calls, and 17 explicit buffer deletions |
| Compile-visible Triton boundary | `T=100, B=64, F=128, N=2048` | forward/loss: raw Triton wrapper 0.144 ms / 13.6 MB, eager `triton_op` 0.167 ms / 13.6 MB with 1 graph and 0 graph breaks, compiled `triton_op` 0.144 ms / 9.6 MB; full rate training with input and weight gradients: existing custom autograd 0.775 ms / 28.1 MB, compiled registered-autograd `triton_op` 0.525 ms / 11.6 MB, compiled public `backend="triton_compile"` 0.511 ms / 11.6 MB, compiled public `backend="triton_compile"` with bias gradients 0.549 ms / 11.6 MB |
| Compile/Triton rate sweep | `T=50/100/200, B=64, F=128, N=1024/2048` | compiled public `backend="triton_compile"` beat existing Triton rate training on all three tested shapes in the targeted sweep: 0.417/0.522/0.730 ms vs 0.587/1.048/0.882 ms, with much lower peak memory. Regular `torch.compile` was 0.478/0.852/16.677 ms in the same sweep; isolated `T=200` and frontier reruns put regular compile closer to 2.1-3.2 ms, so the long-`T` regular-compile row remains noisy but still slower than `triton_compile` in these reruns. |
| Triton classifier rate readout | `T=100, B=128, F=128, classes=10/1000/2048` | handwritten functional 0.466/0.686/1.010 ms, generated functional 0.532/0.751/1.054 ms fwd+bwd; generated module peak 6.5/27.7/50.1 MB |
| MNIST example smoke comparison | `T=10, B=128, hidden=128, 1024 train examples` | dense 60.1%, rate 60.5%, conv 35.9%; steady step 8.483/1.198/10.220 ms |
| Compiled MNIST example smoke comparison | `T=10, B=128, hidden=128, 1024 train examples` | dense 59.7%, rate 60.5%, conv 37.2%; steady step 1.263/1.138/1.785 ms |
| MNIST rate `triton_compile` smoke comparison | `T=10, B=128, hidden=128, 1024 train examples` | compiled rate and rate-triton-compile both reached 60.55% test accuracy and 2.136011 test loss; steady step improved from 2.331 ms to 1.566 ms |
| MNIST rate `triton_compile` longer comparison | `T=10, B=128, hidden=128, 4096 train examples, 2 epochs` | regular rate and rate-triton-compile reached 82.42%/82.67% accuracy with similar loss; total wall time improved from 5.054 s to 4.368 s, but steady step was slower at 1.925 ms vs 1.607 ms |
| MNIST rate matrix | `T=10/25, hidden=128/256, 4096 train examples, 2 epochs` | rate-triton-compile preserved quality across all three tested settings. It reduced memory at `T=25, hidden=128` from 110.3 MB to 44.0 MB, but steady step was slower in all settings: 1.900/1.732/1.839 ms vs regular rate 1.764/1.555/1.475 ms |
| spiker vs snnTorch timestep matrix | `T=10/25/50, B=128, hidden=128, 1024 train examples, 1 epoch` | compiled spiker rate steady-step speedup over eager snnTorch dense grew from 9.92x to 26.91x to 50.42x as `T` increased; compiled spiker conv was 9.11x/9.69x/11.81x faster than eager snnTorch conv. Rate-readout memory was lower than snnTorch dense at every tested `T` |
| Two-layer packed-hidden/recompute probe | `T=10/25/50, B=128, F=784, H=128, C=10` | packed-hidden boundary storage was faster than composed on 2/3 shapes but lower-memory on 0/3: composed 1.297/0.843/0.933 ms and 6.6/78.9/91.8 MB vs packed 1.365/0.814/0.901 ms and 7.6/80.2/93.5 MB. Whole-model recompute remained much slower at 19.149/29.870/61.186 ms |
| Large packed-hidden sweep | `T=25/50, B=128, F=784, H=512/1024/2048, C=10` | packed-hidden storage was still lower-memory on 0/6 shapes. It was roughly time-neutral, but peak memory was consistently higher: composed 27.8-207.5 MB vs packed 33.3-237.4 MB |
| MNIST rate memory smoke comparison | `T=10, B=128, hidden=128, 512 train examples` | dense/rate 36.7%/37.3%; peak CUDA 45.3/18.7 MB |
| MNIST rate checkpoint policy smoke | `T=10, B=128, hidden=128, 512 train examples` | `balanced` resolves to checkpoint 3; same 37.3% accuracy and 19.1 MB peak as explicit checkpoint 10 |
| MNIST rate longer comparison | `T=10, B=128, hidden=128, 4096 train examples, 2 epochs` | dense/rate 82.86%/83.30%; peak CUDA 45.3/18.7 MB; steady step 1.252/1.106 ms |
| MNIST rate backend/policy comparison | `T=10, B=128, hidden=128, 4096 train examples, 2 epochs, rate checkpoint=memory` | compiled dense/rate/generated-rate 82.86%/83.20%/82.47%; peak CUDA 45.3/18.7/18.7 MB; steady step 1.256/1.193/1.269 ms |
| Tuned MNIST rate MLP | `T=25, B=128, hidden=256, 10000 train examples, 3 epochs` | recommended compiled `backend=auto -> triton` rate path reached 90.82% accuracy and 1.541005 final test loss; peak CUDA 114.0 MB |
| Larger regularized MNIST rate MLP | `T=25, B=128, hidden=512, 20000 train examples, 4 epochs, dropout=0.1, label smoothing=0.05` | recommended compiled `backend=auto -> triton` rate path reached 95.51% accuracy and 1.544614 final test loss; peak CUDA 145.0 MB |
| MNIST vs snnTorch smoke comparison | `T=10, B=128, hidden=128, 1024 train examples` | compiled spiker steady step dense/rate/conv 1.212/1.207/1.958 ms vs snnTorch eager dense/conv 15.692/17.151 ms |
| Tuned MNIST conv smoke comparison | `T=10, B=128, hidden=128, 1024 train examples` | compiled spiker conv 59.1% vs snnTorch conv 55.7%; steady step 1.804 vs 18.807 ms |
| Tuned MNIST conv example | `T=25, B=128, hidden=256, 10000 train examples, 4 epochs` | compiled conv example reached 95.78% accuracy and 1.497434 final test loss; peak CUDA 900.3 MB |
| Regularized MNIST conv example | `T=25, B=128, hidden=256, 10000 train examples, 4 epochs, dropout=0.1, label smoothing=0.05` | compiled conv example reached 96.46% accuracy and 1.534343 final test loss; peak CUDA 899.4 MB |
| Direct packed LIF forward | `T=100, B=64, N=2048` | 0.028 ms direct packed vs 0.055 ms dense |
| Packed spike memory/count | `T=100, B=64, N=2048` | 50.0 MB dense spikes to 1.6 MB packed words; public packed count 0.028 ms vs unpack+sum 0.295 ms; per-neuron packed count 0.258 ms vs unpack+sum 0.301 ms |
| Local Gloo packed all-gather | `T=100, B=64, N=2048, world=2` | 103.1 ms dense all-gather vs 1.6 ms packed all-gather, 32.0x payload compression, packed gather correctness true |
| Local Gloo packed count/rate all-reduce | `T=100, B=64, N=2048, world=2` | dense count 2.52 ms vs packed count/rate 3.44/4.02 ms, zero count/rate max error; useful API proof, not yet the target optimized collective |
| Single-GPU NCCL packed count/rate smoke | `T=100, B=64, N=2048, world=1` | dense count 0.059 ms vs packed count/rate 0.085/0.101 ms, zero count/rate max error; validates CUDA path, not multi-GPU communication |
| Online learning CPU smoke | `T=4, B=2, F=3, N=5` | LIF online 0.303 ms, ALIF online 0.437 ms, LIF BPTT 0.554 ms, ALIF BPTT 0.520 ms |
| Online learning CUDA | `T=100, B=64, F=128, N=512` | LIF online 37.556 ms / 74.2 MB, ALIF online 53.922 ms / 111.2 MB, custom surrogate LIF/ALIF 38.097/51.137 ms at same memory |
| Hardware bridge bundle | `LinearLIF F=3, N=5` | Runner-generated smoke writes generic float/quantized/placement bundle plus SpiNNaker2 adapter manifest; 4 generic placement cores, 15 synapses |

The most important lesson so far is that `torch.compile` is a serious baseline,
but the benchmark boundary matters. It can eliminate or hide nominal
`[T, B, N]` current and spike-like materialization when the scalar loss stays
inside the compiled graph. Public dense-output Triton paths intentionally expose
`[T, B, N]` spikes to Python, so they have a dense-output lower bound that the
scalar compiled graph avoids. On the current audit shape, Triton checkpoint
dense output is close to that lower bound in forward: 52.0 MB total increment,
with only 2.0 MB above the returned spike tensor. Triton must therefore win on
explicit SNN-specific behavior: fused-time control, generated custom neuron
kernels, bitpacked spikes, predictable memory contracts, and future
sparse/packed communication. The specialized spike-rate path closes much of the
memory gap for rate objectives by avoiding dense spike outputs entirely. The
classifier readout benchmark also shows the important limit: direct rates are
not a major memory lever for MNIST-sized 10-class outputs, but they matter once
the readout dimension is large. For training quality, the conv MNIST example is
sensitive to synapse initialization and clipping; fan-in synapse init plus less
aggressive clipping closed the short-run accuracy gap to snnTorch conv. The
conv example now keeps stochastic image encoding outside the compiled model
forward, which fixed the compiled conv accuracy drop seen in the earlier smoke
comparison.

`spiker.benchmarks.currents_boundary` now reports forward peak increments above
baseline allocation and normalizes those increments by expected `[T, B, N]`
current bytes. Use that table when auditing whether a compiled graph appears to
keep, eliminate, rematerialize, or reuse storage for the nominal currents
tensor.
`spiker.benchmarks.currents_audit` includes handwritten and generated Triton
checkpoint rate-output variants so the same materialization table also exposes
when dense spike outputs, not dense currents, are the dominant remaining
allocation.
`spiker.benchmarks.compile_inspect` dumps Dynamo/Inductor artifacts for the pure
PyTorch workloads. The smoke inspection suggests the compiled baseline's memory
behavior comes from whole-graph fusion plus scratch lifetime planning, buffer
deletion, reuse, and view aliasing, not from avoiding all dense current/state
storage in dense-output BPTT.
`spiker.benchmarks.checkpoint_size_sweep` sweeps chunk size and shows the
checkpoint memory tradeoff directly: chunk-start states shrink as chunk size
grows, while per-chunk recompute scratch grows with chunk size.
`spiker.benchmarks.two_layer_recompute` probes whether packed hidden-boundary
storage or a whole-model PyTorch recompute wrapper can remove the hidden spike
trace in a two-layer rate model. The current result is mixed/negative:
packed-hidden storage preserves gradients and can slightly improve time, but it
does not reduce peak memory at the MNIST-sized hidden boundary because dense
unpack/backward scratch still dominates. The larger `H=512/1024/2048` sweep had
the same memory result, so packed-hidden training should be deprioritized unless
we avoid dense unpack/backward scratch. Replay through PyTorch is much slower.

## Next Best Work

1. Use `spiker.benchmarks.performance_frontier` as the default performance
   report and keep the scratch-backed checkpoint rate path as the practical
   Triton training default. The latest frontier shows dense-output training is
   effectively tied with `torch.compile`, while rate/scalar outputs are the
   credible near-term Triton edge.
2. Deprioritize packed-hidden training until there is a plan to avoid dense
   unpack/backward scratch. The large-boundary sweep did not show a memory win.
3. Implement the generated ALIF backward kernel from the new
   `alif_adaptive_threshold` plan. `CustomSurrogateNeuronCell` now exercises the
   narrow supported LIF shape, ALIF has a structural plan, and refractory-LIF
   still needs a real generic backward contract.
4. Run the distributed NCCL benchmark on a multi-GPU host and add those numbers
   beside the current single-GPU NCCL smoke artifact.
5. Add a real SDK lowerer for the current SpiNNaker 2 adapter manifest.
