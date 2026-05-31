from __future__ import annotations

import pytest
import torch

from myelin._optional import has_triton
from myelin.functional import alif_unroll, izhikevich_unroll, lif_unroll, surrogate_lif_unroll
from myelin.neurons import (
    ALIFParams,
    ALIFState,
    IzhikevichParams,
    IzhikevichState,
    LIFParams,
    LIFState,
)
from myelin.surrogates import (
    atan_surrogate,
    fast_sigmoid_surrogate,
    multi_gaussian_surrogate,
    sigmoid_surrogate,
    superspike_surrogate,
    triangular_surrogate,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not has_triton(),
    reason="Triton LIF tests require CUDA and Triton",
)

SLOW_TORCH_BACKEND_WARNING = "CUDA inputs detected and Triton is available"
REPRESENTATIVE_SURROGATE_NAMES = ("fast_sigmoid", "multi_gaussian")


def test_generated_kernel_cache_skips_source_factory_on_hit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from myelin.triton import generated as generated_module

    compiled_kernel = object()
    sources: list[str] = []

    def fake_load_generated_forward_kernel(
        source: str,
        *,
        function_name: str,
        module_name: str,
    ) -> object:
        assert function_name == "_unit_kernel"
        assert module_name == "unit_module"
        sources.append(source)
        return compiled_kernel

    generated_module._GENERATED_KERNEL_CACHE.clear()
    monkeypatch.setattr(
        generated_module,
        "load_generated_forward_kernel",
        fake_load_generated_forward_kernel,
    )

    first = generated_module._load_generated_kernel_cached(
        kind="unit",
        function_name="_unit_kernel",
        module_name="unit_module",
        source_factory=lambda: "source",
    )
    second = generated_module._load_generated_kernel_cached(
        kind="unit",
        function_name="_unit_kernel",
        module_name="unit_module",
        source_factory=lambda: pytest.fail("source factory should not run on cache hit"),
    )

    assert first is compiled_kernel
    assert second is compiled_kernel
    assert sources == ["source"]


def test_triton_lif_forward_matches_reference() -> None:
    from myelin.triton import lif_forward

    torch.manual_seed(0)
    inputs = torch.rand((7, 3, 19), device="cuda")
    initial = LIFState(membrane=torch.rand((3, 19), device="cuda") * 0.2)
    params = LIFParams(tau_mem=10.0, threshold=1.0, reset=-0.25)

    expected_state, expected_spikes = lif_unroll(inputs, initial, params)
    actual_state, actual_spikes = lif_forward(inputs, initial, params, block_size=32)

    assert torch.allclose(actual_state.membrane, expected_state.membrane, atol=1e-4, rtol=1e-3)
    assert torch.equal(actual_spikes, expected_spikes)


@pytest.mark.parametrize("neurons", [1, 31, 32, 33, 65])
def test_triton_lif_forward_packed_spikes_matches_dense_forward(neurons: int) -> None:
    from myelin.triton import lif_forward, lif_forward_packed_spikes, unpack_spikes_triton

    torch.manual_seed(10 + neurons)
    inputs = torch.rand((7, 3, neurons), device="cuda")
    initial = LIFState(membrane=torch.rand((3, neurons), device="cuda") * 0.2)
    params = LIFParams(tau_mem=10.0, threshold=1.0, reset=-0.25)

    expected_state, expected_spikes = lif_forward(inputs, initial, params, block_size=32)
    actual_state, packed_spikes = lif_forward_packed_spikes(inputs, initial, params)
    actual_spikes = unpack_spikes_triton(packed_spikes, dtype=inputs.dtype)

    assert torch.allclose(actual_state.membrane, expected_state.membrane, atol=1e-4, rtol=1e-3)
    assert torch.equal(actual_spikes, expected_spikes)


def test_public_lif_forward_packed_spikes_auto_backend_matches_dense_forward() -> None:
    from myelin.kernels import lif_forward_packed_spikes
    from myelin.triton import lif_forward, unpack_spikes_triton

    torch.manual_seed(12)
    inputs = torch.rand((7, 3, 65), device="cuda")
    initial = LIFState(membrane=torch.rand((3, 65), device="cuda") * 0.2)
    params = LIFParams(tau_mem=10.0, threshold=1.0, reset=-0.25)

    expected_state, expected_spikes = lif_forward(inputs, initial, params, block_size=32)
    actual_state, packed_spikes = lif_forward_packed_spikes(inputs, initial, params)
    actual_spikes = unpack_spikes_triton(packed_spikes, dtype=inputs.dtype)

    assert torch.allclose(actual_state.membrane, expected_state.membrane, atol=1e-4, rtol=1e-3)
    assert torch.equal(actual_spikes, expected_spikes)


@pytest.mark.parametrize("neurons", [31, 32, 33])
def test_linear_checkpoint_packed_forward_matches_dense_checkpoint_forward(neurons: int) -> None:
    from myelin.triton import (
        linear_surrogate_lif_checkpoint_forward,
        linear_surrogate_lif_checkpoint_packed_forward,
        unpack_spikes_triton,
    )

    torch.manual_seed(20 + neurons)
    inputs = torch.rand((7, 3, 11), device="cuda")
    weight = (torch.rand((11, neurons), device="cuda") - 0.5) * 0.05
    bias = (torch.rand((neurons,), device="cuda") - 0.5) * 0.01
    params = LIFParams(tau_mem=10.0, threshold=0.1, reset=-0.25)

    expected_state, expected_spikes, expected_chunks = linear_surrogate_lif_checkpoint_forward(
        inputs,
        weight,
        bias,
        params,
        checkpoint_size=3,
        block_b=16,
        block_n=32,
        block_f=16,
    )
    actual_state, packed_spikes, actual_chunks = linear_surrogate_lif_checkpoint_packed_forward(
        inputs,
        weight,
        bias,
        params,
        checkpoint_size=3,
        block_b=16,
        block_f=16,
    )
    actual_spikes = unpack_spikes_triton(packed_spikes, dtype=inputs.dtype)

    assert packed_spikes.original_shape == tuple(expected_spikes.shape)
    assert torch.allclose(actual_state.membrane, expected_state.membrane, atol=1e-4, rtol=1e-3)
    assert torch.allclose(actual_chunks, expected_chunks, atol=1e-4, rtol=1e-3)
    assert torch.equal(actual_spikes, expected_spikes)


def test_generated_lif_forward_source_contains_dsl_step_body() -> None:
    from myelin.triton import render_lif_forward_kernel_source

    source = render_lif_forward_kernel_source()

    assert "tmp0 = (membrane * decay)" in source
    assert "tmp1 = (tmp0 + input_current)" in source
    assert "tmp2 = (tmp1 >= threshold)" in source
    assert "membrane = tmp3" in source
    assert "spike = tmp4" in source


def test_generated_lif_surrogate_backward_source_contains_codegen_step() -> None:
    from myelin.triton import render_lif_surrogate_backward_kernel_source

    source = render_lif_surrogate_backward_kernel_source("fast_sigmoid")

    assert "for reverse_t in range(timesteps):" in source
    assert "pre_reset = tl.load" in source
    assert "grad_spike = tl.load" in source
    assert "tl.abs(centered)" in source
    assert "grad_pre_reset = grad_membrane *" in source
    assert "tl.store(grad_inputs_ptr" in source
    assert "tl.store(grad_initial_ptr" in source


def test_generated_lif_surrogate_backward_packed_spikes_source_contains_codegen_step() -> None:
    from myelin.triton import render_lif_surrogate_backward_packed_spikes_kernel_source

    source = render_lif_surrogate_backward_packed_spikes_kernel_source("fast_sigmoid")

    assert "packed_spikes_ptr" in source
    assert "packed_offsets = t * batch * packed_neurons" in source
    assert "packed_word = tl.load" in source
    assert "spike = ((packed_word >> bit_offsets) & 1).to(tl.float32)" in source
    assert "tl.abs(centered)" in source
    assert "grad_pre_reset = grad_membrane *" in source
    assert "tl.store(grad_inputs_ptr" in source
    assert "tl.store(grad_initial_ptr" in source


def test_generated_linear_lif_surrogate_backward_source_contains_codegen_step() -> None:
    from myelin.triton import render_linear_lif_surrogate_backward_weight_bias_kernel_source

    source = render_linear_lif_surrogate_backward_weight_bias_kernel_source("fast_sigmoid")

    assert "def _generated_linear_lif_surrogate_backward_weight_bias_kernel(" in source
    assert "for reverse_t in range(timesteps):" in source
    assert "pre_reset = tl.load" in source
    assert "grad_spike = tl.load" in source
    assert "tl.abs(centered)" in source
    assert "grad_input = tl.dot(grad_pre_reset, tl.trans(weight_values))" in source
    assert "weight_acc += tl.dot(input_values, grad_pre_reset)" in source
    assert "tl.atomic_add(" in source


def test_generated_linear_lif_surrogate_backward_packed_spikes_source_contains_codegen_step() -> (
    None
):
    from myelin.triton import (
        render_linear_lif_surrogate_backward_weight_bias_packed_spikes_kernel_source,
    )

    source = render_linear_lif_surrogate_backward_weight_bias_packed_spikes_kernel_source(
        "fast_sigmoid"
    )

    assert "packed_spikes_ptr" in source
    assert "packed_offsets = (" in source
    assert "packed_word = tl.load" in source
    assert "spike = ((packed_word >> bit_offsets[None, :]) & 1).to(tl.float32)" in source
    assert "tl.abs(centered)" in source
    assert "grad_input = tl.dot(grad_pre_reset, tl.trans(weight_values))" in source
    assert "weight_acc += tl.dot(input_values, grad_pre_reset)" in source


def test_generated_linear_lif_surrogate_checkpoint_backward_source_contains_codegen_step() -> None:
    from myelin.triton import render_linear_lif_surrogate_checkpoint_backward_chunk_kernel_source

    source = render_linear_lif_surrogate_checkpoint_backward_chunk_kernel_source("fast_sigmoid")

    assert "def _generated_linear_lif_surrogate_checkpoint_backward_chunk_kernel(" in source
    assert "for reverse_local in range(chunk_len):" in source
    assert "pre_reset = tl.load" in source
    assert "grad_spike = tl.load" in source
    assert "tl.abs(centered)" in source
    assert "grad_input = tl.dot(grad_pre_reset, tl.trans(weight_values))" in source
    assert "weight_acc += tl.dot(input_values, grad_pre_reset)" in source
    assert "tl.store(" in source


def test_generated_checkpoint_backward_packed_spikes_source_contains_codegen_step() -> None:
    from myelin.triton import (
        render_linear_lif_surrogate_checkpoint_backward_chunk_packed_spikes_kernel_source,
    )

    source = render_linear_lif_surrogate_checkpoint_backward_chunk_packed_spikes_kernel_source(
        "fast_sigmoid"
    )

    assert "packed_spikes_scratch_ptr" not in source
    assert "packed_word = tl.load" not in source
    assert "spike = (pre_reset >= threshold).to(tl.float32)" in source
    assert "grad_spike_rate_ptr" in source
    assert "grad_spike_rates_ptr" in source
    assert "tl.abs(centered)" in source
    assert "grad_input = tl.dot(grad_pre_reset, tl.trans(weight_values))" in source
    assert "tl.store(" in source


def test_generic_generated_forward_renderer_uses_ir_boundary() -> None:
    from myelin.dsl import lif_ir
    from myelin.triton import render_forward_kernel_source

    source = render_forward_kernel_source(
        lif_ir(),
        function_name="_test_generated_forward_kernel",
    )

    assert "def _test_generated_forward_kernel(" in source
    assert "initial_membrane_ptr" in source
    assert "final_membrane_ptr" in source
    assert "decay: tl.constexpr" in source
    assert "threshold: tl.constexpr" in source
    assert "reset: tl.constexpr" in source


def test_generated_izhikevich_forward_matches_reference() -> None:
    from myelin.triton import generated_izhikevich_forward

    torch.manual_seed(13)
    inputs = torch.rand((7, 3, 19), device="cuda") * 20.0
    initial = IzhikevichState(
        voltage=torch.full((3, 19), -65.0, device="cuda"),
        recovery=torch.full((3, 19), -13.0, device="cuda"),
    )
    params = IzhikevichParams()

    expected_state, expected_spikes = izhikevich_unroll(inputs, initial, params)
    actual_state, actual_spikes = generated_izhikevich_forward(
        inputs,
        initial,
        params,
        block_size=32,
    )

    assert torch.allclose(actual_state.voltage, expected_state.voltage, atol=1e-4, rtol=1e-3)
    assert torch.allclose(actual_state.recovery, expected_state.recovery, atol=1e-4, rtol=1e-3)
    assert torch.equal(actual_spikes, expected_spikes)


def test_generated_kernel_loader_reuses_cached_function() -> None:
    from myelin.triton import load_generated_alif_forward_kernel, load_generated_lif_forward_kernel

    assert load_generated_lif_forward_kernel() is load_generated_lif_forward_kernel()
    assert load_generated_alif_forward_kernel() is load_generated_alif_forward_kernel()


def test_generated_lif_forward_matches_reference() -> None:
    from myelin.triton import generated_lif_forward

    torch.manual_seed(1)
    inputs = torch.rand((7, 3, 19), device="cuda")
    initial = LIFState(membrane=torch.rand((3, 19), device="cuda") * 0.2)
    params = LIFParams(tau_mem=10.0, threshold=1.0, reset=-0.25)

    expected_state, expected_spikes = lif_unroll(inputs, initial, params)
    actual_state, actual_spikes = generated_lif_forward(inputs, initial, params, block_size=32)

    assert torch.allclose(actual_state.membrane, expected_state.membrane, atol=1e-4, rtol=1e-3)
    assert torch.equal(actual_spikes, expected_spikes)


def test_generic_generated_forward_launcher_matches_lif_reference() -> None:
    from myelin.dsl import lif_ir
    from myelin.triton import generated_forward, load_generated_lif_forward_kernel

    torch.manual_seed(11)
    inputs = torch.rand((5, 2, 13), device="cuda")
    initial = LIFState(membrane=torch.rand((2, 13), device="cuda") * 0.2)
    params = LIFParams(tau_mem=8.0, threshold=0.7, reset=-0.1)

    expected_state, expected_spikes = lif_unroll(inputs, initial, params)
    actual_state, actual_spikes = generated_forward(
        lif_ir(),
        inputs,
        {"membrane": initial.membrane},
        {"decay": params.decay, "threshold": params.threshold, "reset": params.reset},
        kernel=load_generated_lif_forward_kernel(),
        block_size=16,
    )

    assert torch.allclose(
        actual_state["membrane"],
        expected_state.membrane,
        atol=1e-4,
        rtol=1e-3,
    )
    assert torch.equal(actual_spikes, expected_spikes)


def test_neuron_builder_ir_runs_through_generated_forward_launcher() -> None:
    from myelin.dsl import NeuronBuilder, evaluate_neuron_unroll, where
    from myelin.triton import (
        generated_forward,
        load_generated_neuron_forward_kernel,
    )

    builder = NeuronBuilder("custom_lif_codegen")
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

    torch.manual_seed(13)
    inputs = torch.rand((5, 2, 13), device="cuda")
    initial_membrane = torch.rand((2, 13), device="cuda") * 0.2
    params = {"decay": 0.9, "threshold": 0.7, "reset": -0.1}
    expected_state, expected_spikes = evaluate_neuron_unroll(
        ir,
        inputs,
        {"membrane": initial_membrane},
        params,
    )

    kernel = load_generated_neuron_forward_kernel(ir)
    actual_state, actual_spikes = generated_forward(
        ir,
        inputs,
        {"membrane": initial_membrane},
        params,
        kernel=kernel,
        block_size=16,
    )

    assert torch.allclose(
        actual_state["membrane"], expected_state["membrane"], atol=1e-4, rtol=1e-3
    )
    assert torch.equal(actual_spikes, expected_spikes)


def test_generated_custom_forward_kernel_cache_skips_re_render(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import myelin.triton.generated as generated
    from myelin.dsl import NeuronBuilder, where

    builder = NeuronBuilder("custom_lif_cache_probe")
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

    calls = 0
    original_render = generated.render_forward_kernel_source

    def counting_render(*args: object, **kwargs: object) -> str:
        nonlocal calls
        calls += 1
        return original_render(*args, **kwargs)  # pyright: ignore[reportArgumentType]

    monkeypatch.setattr(generated, "render_forward_kernel_source", counting_render)

    first_kernel = generated.load_generated_neuron_forward_kernel(ir)
    second_kernel = generated.load_generated_neuron_forward_kernel(ir)

    assert first_kernel is second_kernel
    assert calls == 1


def test_neuron_builder_ir_runs_through_generated_neuron_forward_wrapper() -> None:
    from myelin.dsl import NeuronBuilder, evaluate_neuron_unroll, where
    from myelin.triton import generated_neuron_forward

    builder = NeuronBuilder("custom_lif_wrapper")
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

    torch.manual_seed(17)
    inputs = torch.rand((5, 2, 13), device="cuda")
    initial_membrane = torch.rand((2, 13), device="cuda") * 0.2
    params = {"decay": 0.9, "threshold": 0.7, "reset": -0.1}
    expected_state, expected_spikes = evaluate_neuron_unroll(
        ir,
        inputs,
        {"membrane": initial_membrane},
        params,
    )
    actual_state, actual_spikes = generated_neuron_forward(
        ir,
        inputs,
        {"membrane": initial_membrane},
        params,
        block_size=16,
    )

    assert torch.allclose(
        actual_state["membrane"], expected_state["membrane"], atol=1e-4, rtol=1e-3
    )
    assert torch.equal(actual_spikes, expected_spikes)


def test_generated_custom_forward_rejects_ir_shapes_outside_current_abi() -> None:
    from myelin.dsl import NeuronIR, const, input_, state
    from myelin.triton.generated import render_forward_kernel_source

    stateless = NeuronIR(
        name="stateless",
        state=(),
        inputs=("input_current",),
        params=(),
        next_state={},
        outputs={"spike": const(0.0)},
    )
    with pytest.raises(ValueError, match="at least one state"):
        render_forward_kernel_source(stateless, function_name="_stateless")

    wrong_input = NeuronIR(
        name="wrong_input",
        state=("membrane",),
        inputs=("current",),
        params=(),
        next_state={"membrane": input_("current")},
        outputs={"spike": state("membrane")},
    )
    with pytest.raises(ValueError, match="one input named input_current"):
        render_forward_kernel_source(wrong_input, function_name="_wrong_input")

    extra_output = NeuronIR(
        name="extra_output",
        state=("membrane",),
        inputs=("input_current",),
        params=(),
        next_state={"membrane": input_("input_current")},
        outputs={"spike": state("membrane"), "rate": state("membrane")},
    )
    with pytest.raises(ValueError, match="exactly one output named spike"):
        render_forward_kernel_source(extra_output, function_name="_extra_output")


def test_generated_custom_forward_rejects_unexpected_launch_inputs() -> None:
    from myelin.dsl import NeuronBuilder, where
    from myelin.triton import generated_neuron_forward

    builder = NeuronBuilder("custom_lif_strict_launch")
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

    inputs = torch.rand((3, 2, 5), device="cuda")
    initial_membrane = torch.zeros((2, 5), device="cuda")
    params = {"decay": 0.9, "threshold": 1.0, "reset": 0.0}

    with pytest.raises(ValueError, match="undeclared state tensors"):
        generated_neuron_forward(
            ir,
            inputs,
            {"membrane": initial_membrane, "extra": initial_membrane},
            params,
            block_size=16,
        )

    with pytest.raises(ValueError, match="undeclared values"):
        generated_neuron_forward(
            ir,
            inputs,
            {"membrane": initial_membrane},
            {**params, "extra": 1.0},
            block_size=16,
        )

    with pytest.raises(ValueError, match="block_size must be positive"):
        generated_neuron_forward(
            ir,
            inputs,
            {"membrane": initial_membrane},
            params,
            block_size=0,
        )


def test_custom_neuron_cell_auto_backend_uses_generated_triton_path_on_cuda() -> None:
    from myelin.dsl import NeuronBuilder, evaluate_neuron_unroll, where
    from myelin.modules import CustomNeuronCell, TimeUnroll

    builder = NeuronBuilder("module_custom_lif_cuda")
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

    torch.manual_seed(19)
    inputs = torch.rand((5, 2, 13), device="cuda")
    initial_state = {"membrane": torch.rand((2, 13), device="cuda") * 0.2}
    params = {"decay": 0.9, "threshold": 0.7, "reset": -0.1}
    expected_state, expected_spikes = evaluate_neuron_unroll(ir, inputs, initial_state, params)

    actual_state, actual_spikes = TimeUnroll(CustomNeuronCell(ir, params), backend="auto")(
        inputs,
        initial_state,
    )

    assert torch.allclose(
        actual_state["membrane"], expected_state["membrane"], atol=1e-4, rtol=1e-3
    )
    assert torch.equal(actual_spikes, expected_spikes)


def test_generated_alif_forward_source_contains_dsl_step_body() -> None:
    from myelin.triton import render_alif_forward_kernel_source

    source = render_alif_forward_kernel_source()

    assert "tmp0 = (membrane * decay)" in source
    assert "tmp3 = (threshold + tmp2)" in source
    assert "tmp4 = (tmp1 >= tmp3)" in source
    assert "tmp5 = tl.where(tmp4, reset, tmp1)" in source
    assert "adaptation = tmp8" in source
    assert "spike = tmp7" in source


def test_generated_alif_forward_matches_reference() -> None:
    from myelin.triton import generated_alif_forward

    torch.manual_seed(2)
    inputs = torch.rand((9, 4, 23), device="cuda")
    initial = ALIFState(
        membrane=torch.rand((4, 23), device="cuda") * 0.2,
        adaptation=torch.rand((4, 23), device="cuda") * 0.1,
    )
    params = ALIFParams(
        tau_mem=10.0,
        tau_adaptation=25.0,
        threshold=1.0,
        reset=-0.25,
        beta=0.5,
    )

    expected_state, expected_spikes = alif_unroll(inputs, initial, params)
    actual_state, actual_spikes = generated_alif_forward(inputs, initial, params, block_size=32)

    assert torch.allclose(actual_state.membrane, expected_state.membrane, atol=1e-4, rtol=1e-3)
    assert torch.allclose(
        actual_state.adaptation,
        expected_state.adaptation,
        atol=1e-4,
        rtol=1e-3,
    )
    assert torch.equal(actual_spikes, expected_spikes)


def test_generated_alif_backend_forward_matches_reference() -> None:
    from myelin.kernels import alif_forward

    torch.manual_seed(3)
    inputs = torch.rand((9, 4, 23), device="cuda")
    initial = ALIFState(
        membrane=torch.rand((4, 23), device="cuda") * 0.2,
        adaptation=torch.rand((4, 23), device="cuda") * 0.1,
    )
    params = ALIFParams(
        tau_mem=10.0,
        tau_adaptation=25.0,
        threshold=1.0,
        reset=-0.25,
        beta=0.5,
    )

    expected_state, expected_spikes = alif_unroll(inputs, initial, params)
    actual_state, actual_spikes = alif_forward(
        inputs,
        initial,
        params,
        backend="triton_generated",
        block_size=32,
    )

    assert torch.allclose(actual_state.membrane, expected_state.membrane, atol=1e-4, rtol=1e-3)
    assert torch.allclose(
        actual_state.adaptation,
        expected_state.adaptation,
        atol=1e-4,
        rtol=1e-3,
    )
    assert torch.equal(actual_spikes, expected_spikes)


def test_time_unroll_alif_generated_backend_matches_reference() -> None:
    from myelin.modules import ALIFCell, TimeUnroll

    torch.manual_seed(4)
    inputs = torch.rand((9, 4, 23), device="cuda")
    initial = ALIFState(
        membrane=torch.rand((4, 23), device="cuda") * 0.2,
        adaptation=torch.rand((4, 23), device="cuda") * 0.1,
    )
    params = ALIFParams(
        tau_mem=10.0,
        tau_adaptation=25.0,
        threshold=1.0,
        reset=-0.25,
        beta=0.5,
    )
    unroll = TimeUnroll(ALIFCell(params), backend="triton_generated")

    expected_state, expected_spikes = alif_unroll(inputs, initial, params)
    actual_state, actual_spikes = unroll(inputs, initial)

    assert isinstance(actual_state, ALIFState)
    assert torch.allclose(actual_state.membrane, expected_state.membrane, atol=1e-4, rtol=1e-3)
    assert torch.allclose(
        actual_state.adaptation,
        expected_state.adaptation,
        atol=1e-4,
        rtol=1e-3,
    )
    assert torch.equal(actual_spikes, expected_spikes)


@pytest.mark.parametrize(
    ("shape", "params", "block_size"),
    [
        ((1, 1, 1), LIFParams(tau_mem=2.0, threshold=0.2, reset=0.0), 16),
        ((3, 2, 7), LIFParams(tau_mem=5.0, threshold=0.6, reset=-0.5), 8),
        ((11, 4, 33), LIFParams(tau_mem=20.0, threshold=1.0, reset=0.0), 32),
        ((17, 3, 65), LIFParams(tau_mem=50.0, threshold=1.5, reset=0.25), 64),
    ],
)
def test_triton_lif_forward_matches_reference_across_shapes(
    shape: tuple[int, int, int],
    params: LIFParams,
    block_size: int,
) -> None:
    from myelin.triton import lif_forward

    torch.manual_seed(sum(shape) + block_size)
    timesteps, batch, neurons = shape
    inputs = torch.rand(shape, device="cuda") * 1.5
    initial = LIFState(membrane=torch.rand((batch, neurons), device="cuda") * 0.5)

    expected_state, expected_spikes = lif_unroll(inputs, initial, params)
    actual_state, actual_spikes = lif_forward(inputs, initial, params, block_size=block_size)

    assert torch.allclose(actual_state.membrane, expected_state.membrane, atol=1e-4, rtol=1e-3)
    assert torch.equal(actual_spikes, expected_spikes)


def test_triton_lif_forward_handles_partial_blocks() -> None:
    from myelin.triton import lif_forward

    inputs = torch.full((4, 2, 13), 0.6, device="cuda")
    initial = LIFState(membrane=torch.zeros((2, 13), device="cuda"))
    params = LIFParams(tau_mem=20.0, threshold=1.0, reset=0.0)

    expected_state, expected_spikes = lif_unroll(inputs, initial, params)
    actual_state, actual_spikes = lif_forward(inputs, initial, params, block_size=16)

    assert torch.allclose(actual_state.membrane, expected_state.membrane, atol=1e-4, rtol=1e-3)
    assert torch.equal(actual_spikes, expected_spikes)


def test_backend_lif_forward_auto_matches_reference() -> None:
    from myelin.kernels import lif_forward

    torch.manual_seed(12)
    inputs = torch.rand((5, 2, 17), device="cuda")
    initial = LIFState(membrane=torch.zeros((2, 17), device="cuda"))
    params = LIFParams(tau_mem=7.0, threshold=0.9, reset=-0.1)

    expected_state, expected_spikes = lif_unroll(inputs, initial, params)
    actual_state, actual_spikes = lif_forward(inputs, initial, params, backend="auto")

    assert torch.allclose(actual_state.membrane, expected_state.membrane, atol=1e-4, rtol=1e-3)
    assert torch.equal(actual_spikes, expected_spikes)


def test_autograd_boundary_forward_matches_reference() -> None:
    from myelin.autograd import triton_lif_forward_function

    torch.manual_seed(21)
    inputs = torch.rand((6, 2, 9), device="cuda")
    initial = LIFState(membrane=torch.rand((2, 9), device="cuda") * 0.1)
    params = LIFParams(tau_mem=11.0, threshold=0.8, reset=-0.2)

    expected_state, expected_spikes = lif_unroll(inputs, initial, params)
    actual_state, actual_spikes = triton_lif_forward_function(
        inputs,
        initial,
        params,
        block_size=16,
    )

    assert torch.allclose(actual_state.membrane, expected_state.membrane)
    assert torch.equal(actual_spikes, expected_spikes)


def test_autograd_boundary_backward_matches_reference_for_final_membrane() -> None:
    from myelin.kernels import lif_forward

    torch.manual_seed(31)
    inputs = torch.rand((5, 2, 7), device="cuda", requires_grad=True)
    initial_membrane = (torch.rand((2, 7), device="cuda") * 0.2).requires_grad_()
    params = LIFParams(tau_mem=10.0, threshold=0.8, reset=-0.1)

    expected_state, _expected_spikes = lif_unroll(
        inputs,
        LIFState(membrane=initial_membrane),
        params,
    )
    expected_loss = expected_state.membrane.square().mean()
    expected_loss.backward()
    expected_input_raw_grad = inputs.grad
    assert expected_input_raw_grad is not None
    expected_input_grad = expected_input_raw_grad.detach().clone()
    expected_initial_raw_grad = initial_membrane.grad
    assert expected_initial_raw_grad is not None
    expected_initial_grad = expected_initial_raw_grad.detach().clone()

    inputs.grad = None
    initial_membrane.grad = None
    actual_state, _actual_spikes = lif_forward(
        inputs,
        LIFState(membrane=initial_membrane),
        params,
        backend="triton",
        block_size=16,
    )
    actual_loss = actual_state.membrane.square().mean()
    actual_loss.backward()

    actual_input_grad = inputs.grad
    assert actual_input_grad is not None
    actual_initial_grad = initial_membrane.grad
    assert actual_initial_grad is not None
    assert torch.allclose(actual_input_grad, expected_input_grad)
    assert torch.allclose(actual_initial_grad, expected_initial_grad)


def test_autograd_boundary_spike_backward_raises_clear_error() -> None:
    from myelin.kernels import lif_forward

    inputs = torch.rand((4, 2, 7), device="cuda", requires_grad=True)
    initial = LIFState(membrane=torch.zeros((2, 7), device="cuda"))
    params = LIFParams(tau_mem=10.0, threshold=0.5, reset=0.0)

    _state, spikes = lif_forward(inputs, initial, params, backend="triton", block_size=16)
    loss = spikes.sum()

    with pytest.raises(NotImplementedError, match="spike-output backward"):
        loss.backward()


def test_triton_surrogate_lif_spike_backward_matches_reference() -> None:
    from myelin.kernels import surrogate_lif_forward

    torch.manual_seed(41)
    inputs = torch.rand((5, 2, 7), device="cuda", requires_grad=True)
    initial_membrane = (torch.rand((2, 7), device="cuda") * 0.2).requires_grad_()
    params = LIFParams(tau_mem=10.0, threshold=0.8, reset=-0.1)

    expected_state, expected_spikes = surrogate_lif_unroll(
        inputs,
        LIFState(membrane=initial_membrane),
        params,
        surrogate=fast_sigmoid_surrogate,
        surrogate_slope=5.0,
        hard_forward=True,
    )
    expected_loss = expected_spikes.square().mean() + expected_state.membrane.square().mean()
    expected_loss.backward()
    expected_input_raw_grad = inputs.grad
    expected_initial_raw_grad = initial_membrane.grad
    assert expected_input_raw_grad is not None
    assert expected_initial_raw_grad is not None
    expected_input_grad = expected_input_raw_grad.detach().clone()
    expected_initial_grad = expected_initial_raw_grad.detach().clone()

    inputs.grad = None
    initial_membrane.grad = None
    actual_state, actual_spikes = surrogate_lif_forward(
        inputs,
        LIFState(membrane=initial_membrane),
        params,
        surrogate="fast_sigmoid",
        surrogate_slope=5.0,
        backend="triton",
        block_size=16,
    )
    actual_loss = actual_spikes.square().mean() + actual_state.membrane.square().mean()
    actual_loss.backward()

    actual_input_grad = inputs.grad
    actual_initial_grad = initial_membrane.grad
    assert actual_input_grad is not None
    assert actual_initial_grad is not None
    assert torch.allclose(actual_spikes, expected_spikes.detach())
    assert torch.allclose(
        actual_state.membrane,
        expected_state.membrane.detach(),
        atol=1e-4,
        rtol=1e-3,
    )
    assert torch.allclose(actual_input_grad, expected_input_grad)
    assert torch.allclose(actual_initial_grad, expected_initial_grad)


@pytest.mark.parametrize(
    ("surrogate_name", "surrogate_fn"),
    [
        ("sigmoid", sigmoid_surrogate),
        ("fast_sigmoid", fast_sigmoid_surrogate),
        ("atan", atan_surrogate),
        ("triangular", triangular_surrogate),
        ("superspike", superspike_surrogate),
        ("multi_gaussian", multi_gaussian_surrogate),
    ],
)
def test_triton_surrogate_lif_backward_matches_reference_for_builtin_surrogates(
    surrogate_name: str,
    surrogate_fn,
) -> None:
    from myelin.kernels import surrogate_lif_forward

    torch.manual_seed(42)
    inputs = torch.rand((4, 2, 5), device="cuda", requires_grad=True)
    initial_membrane = (torch.rand((2, 5), device="cuda") * 0.2).requires_grad_()
    params = LIFParams(tau_mem=8.0, threshold=0.7, reset=-0.2)

    expected_state, expected_spikes = surrogate_lif_unroll(
        inputs,
        LIFState(membrane=initial_membrane),
        params,
        surrogate=surrogate_fn,
        surrogate_slope=4.0,
        hard_forward=True,
    )
    expected_loss = expected_spikes.mean() + 0.1 * expected_state.membrane.square().mean()
    expected_loss.backward()
    expected_input_raw_grad = inputs.grad
    expected_initial_raw_grad = initial_membrane.grad
    assert expected_input_raw_grad is not None
    assert expected_initial_raw_grad is not None
    expected_input_grad = expected_input_raw_grad.detach().clone()
    expected_initial_grad = expected_initial_raw_grad.detach().clone()

    inputs.grad = None
    initial_membrane.grad = None
    actual_state, actual_spikes = surrogate_lif_forward(
        inputs,
        LIFState(membrane=initial_membrane),
        params,
        surrogate=surrogate_name,  # pyright: ignore[reportArgumentType]
        surrogate_slope=4.0,
        backend="triton",
        block_size=16,
    )
    actual_loss = actual_spikes.mean() + 0.1 * actual_state.membrane.square().mean()
    actual_loss.backward()

    actual_input_grad = inputs.grad
    actual_initial_grad = initial_membrane.grad
    assert actual_input_grad is not None
    assert actual_initial_grad is not None
    assert torch.allclose(actual_spikes, expected_spikes.detach())
    assert torch.allclose(actual_input_grad, expected_input_grad, atol=1e-6, rtol=1e-5)
    assert torch.allclose(actual_initial_grad, expected_initial_grad, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("surrogate_name", REPRESENTATIVE_SURROGATE_NAMES)
def test_generated_lif_surrogate_backward_matches_handwritten_triton(
    surrogate_name: str,
) -> None:
    from myelin.triton import generated_lif_surrogate_backward, surrogate_lif_backward

    torch.manual_seed(43)
    pre_reset = torch.rand((5, 3, 13), device="cuda") * 1.4 - 0.2
    spikes = (pre_reset >= 0.7).to(pre_reset.dtype)
    grad_final = torch.rand((3, 13), device="cuda") - 0.5
    grad_spikes = torch.rand_like(pre_reset) - 0.5
    params = LIFParams(tau_mem=8.0, threshold=0.7, reset=-0.2)

    expected_inputs, expected_initial = surrogate_lif_backward(
        pre_reset,
        spikes,
        grad_final,
        grad_spikes,
        params,
        surrogate=surrogate_name,  # pyright: ignore[reportArgumentType]
        surrogate_slope=4.0,
        block_size=16,
    )
    actual_inputs, actual_initial = generated_lif_surrogate_backward(
        pre_reset,
        spikes,
        grad_final,
        grad_spikes,
        params,
        surrogate=surrogate_name,  # pyright: ignore[reportArgumentType]
        surrogate_slope=4.0,
        block_size=16,
    )

    assert torch.allclose(actual_inputs, expected_inputs, atol=1e-6, rtol=1e-5)
    assert torch.allclose(actual_initial, expected_initial, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("neurons", [13, 32])
@pytest.mark.parametrize("surrogate_name", ["fast_sigmoid"])
def test_triton_surrogate_lif_backward_packed_spikes_matches_dense_backward(
    neurons: int,
    surrogate_name: str,
) -> None:
    from myelin.triton import (
        pack_spikes_triton,
        surrogate_lif_backward,
        surrogate_lif_backward_packed_spikes,
    )

    torch.manual_seed(45 + neurons)
    pre_reset = torch.rand((5, 3, neurons), device="cuda") * 1.4 - 0.2
    spikes = (pre_reset >= 0.7).to(pre_reset.dtype)
    packed_spikes = pack_spikes_triton(spikes)
    grad_final = torch.rand((3, neurons), device="cuda") - 0.5
    grad_spikes = torch.rand_like(pre_reset) - 0.5
    params = LIFParams(tau_mem=8.0, threshold=0.7, reset=-0.2)

    expected_inputs, expected_initial = surrogate_lif_backward(
        pre_reset,
        spikes,
        grad_final,
        grad_spikes,
        params,
        surrogate=surrogate_name,  # pyright: ignore[reportArgumentType]
        surrogate_slope=4.0,
        block_size=16,
    )
    actual_inputs, actual_initial = surrogate_lif_backward_packed_spikes(
        pre_reset,
        packed_spikes.data,
        grad_final,
        grad_spikes,
        params,
        surrogate=surrogate_name,  # pyright: ignore[reportArgumentType]
        surrogate_slope=4.0,
        block_size=16,
    )

    assert torch.allclose(actual_inputs, expected_inputs, atol=1e-6, rtol=1e-5)
    assert torch.allclose(actual_initial, expected_initial, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("neurons", [13, 32])
@pytest.mark.parametrize("surrogate_name", REPRESENTATIVE_SURROGATE_NAMES)
def test_generated_lif_surrogate_backward_packed_spikes_matches_dense_generated(
    neurons: int,
    surrogate_name: str,
) -> None:
    from myelin.triton import (
        generated_lif_surrogate_backward,
        generated_lif_surrogate_backward_packed_spikes,
        pack_spikes_triton,
    )

    torch.manual_seed(46 + neurons)
    pre_reset = torch.rand((5, 3, neurons), device="cuda") * 1.4 - 0.2
    spikes = (pre_reset >= 0.7).to(pre_reset.dtype)
    packed_spikes = pack_spikes_triton(spikes)
    grad_final = torch.rand((3, neurons), device="cuda") - 0.5
    grad_spikes = torch.rand_like(pre_reset) - 0.5
    params = LIFParams(tau_mem=8.0, threshold=0.7, reset=-0.2)

    expected_inputs, expected_initial = generated_lif_surrogate_backward(
        pre_reset,
        spikes,
        grad_final,
        grad_spikes,
        params,
        surrogate=surrogate_name,  # pyright: ignore[reportArgumentType]
        surrogate_slope=4.0,
        block_size=16,
    )
    actual_inputs, actual_initial = generated_lif_surrogate_backward_packed_spikes(
        pre_reset,
        packed_spikes.data,
        grad_final,
        grad_spikes,
        params,
        surrogate=surrogate_name,  # pyright: ignore[reportArgumentType]
        surrogate_slope=4.0,
        block_size=16,
    )

    assert torch.allclose(actual_inputs, expected_inputs, atol=1e-6, rtol=1e-5)
    assert torch.allclose(actual_initial, expected_initial, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("surrogate_name", REPRESENTATIVE_SURROGATE_NAMES)
def test_generated_triton_surrogate_lif_autograd_matches_handwritten_triton(
    surrogate_name: str,
) -> None:
    from myelin.autograd import (
        generated_triton_surrogate_lif_function,
        triton_surrogate_lif_function,
    )

    torch.manual_seed(44)
    inputs = torch.rand((4, 2, 7), device="cuda", requires_grad=True)
    initial_membrane = (torch.rand((2, 7), device="cuda") * 0.2).requires_grad_()
    generated_inputs = inputs.detach().clone().requires_grad_()
    generated_initial = initial_membrane.detach().clone().requires_grad_()
    params = LIFParams(tau_mem=9.0, threshold=0.75, reset=-0.15)

    expected_state, expected_spikes = triton_surrogate_lif_function(
        inputs,
        LIFState(membrane=initial_membrane),
        params,
        surrogate=surrogate_name,  # pyright: ignore[reportArgumentType]
        surrogate_slope=4.0,
        block_size=16,
    )
    expected_loss = expected_spikes.mean() + 0.1 * expected_state.membrane.square().mean()
    expected_loss.backward()

    actual_state, actual_spikes = generated_triton_surrogate_lif_function(
        generated_inputs,
        LIFState(membrane=generated_initial),
        params,
        surrogate=surrogate_name,  # pyright: ignore[reportArgumentType]
        surrogate_slope=4.0,
        block_size=16,
    )
    actual_loss = actual_spikes.mean() + 0.1 * actual_state.membrane.square().mean()
    actual_loss.backward()

    assert torch.equal(actual_spikes, expected_spikes.detach())
    assert torch.allclose(
        actual_state.membrane,
        expected_state.membrane.detach(),
        atol=1e-4,
        rtol=1e-3,
    )
    assert inputs.grad is not None
    assert initial_membrane.grad is not None
    assert generated_inputs.grad is not None
    assert generated_initial.grad is not None
    assert torch.allclose(generated_inputs.grad, inputs.grad, atol=1e-6, rtol=1e-5)
    assert torch.allclose(generated_initial.grad, initial_membrane.grad, atol=1e-6, rtol=1e-5)


def test_generated_triton_surrogate_backend_matches_handwritten_triton_backend() -> None:
    from myelin.kernels import surrogate_lif_forward

    torch.manual_seed(45)
    inputs = torch.rand((5, 2, 7), device="cuda", requires_grad=True)
    initial_membrane = (torch.rand((2, 7), device="cuda") * 0.2).requires_grad_()
    generated_inputs = inputs.detach().clone().requires_grad_()
    generated_initial = initial_membrane.detach().clone().requires_grad_()
    params = LIFParams(tau_mem=9.0, threshold=0.75, reset=-0.15)

    expected_state, expected_spikes = surrogate_lif_forward(
        inputs,
        LIFState(membrane=initial_membrane),
        params,
        surrogate="fast_sigmoid",
        surrogate_slope=4.0,
        backend="triton",
        block_size=16,
    )
    expected_loss = expected_spikes.mean() + 0.1 * expected_state.membrane.square().mean()
    expected_loss.backward()

    actual_state, actual_spikes = surrogate_lif_forward(
        generated_inputs,
        LIFState(membrane=generated_initial),
        params,
        surrogate="fast_sigmoid",
        surrogate_slope=4.0,
        backend="triton_generated",
        block_size=16,
    )
    actual_loss = actual_spikes.mean() + 0.1 * actual_state.membrane.square().mean()
    actual_loss.backward()

    assert torch.equal(actual_spikes, expected_spikes.detach())
    assert torch.allclose(
        actual_state.membrane,
        expected_state.membrane.detach(),
        atol=1e-4,
        rtol=1e-3,
    )
    assert inputs.grad is not None
    assert initial_membrane.grad is not None
    assert generated_inputs.grad is not None
    assert generated_initial.grad is not None
    assert torch.allclose(generated_inputs.grad, inputs.grad, atol=1e-6, rtol=1e-5)
    assert torch.allclose(generated_initial.grad, initial_membrane.grad, atol=1e-6, rtol=1e-5)


def test_linear_lif_triton_backend_matches_torch_backend_on_cuda() -> None:
    from myelin.modules import LinearLIF

    torch.manual_seed(123)
    inputs = torch.rand((6, 3, 11), device="cuda")
    params = LIFParams(tau_mem=9.0, threshold=0.7, reset=-0.2)
    torch_layer = LinearLIF(11, 17, params, backend="torch").to(device="cuda")
    triton_layer = LinearLIF(11, 17, params, backend="triton").to(device="cuda")
    triton_layer.synapse.weight.data.copy_(torch_layer.synapse.weight)
    assert torch_layer.synapse.bias is not None
    assert triton_layer.synapse.bias is not None
    triton_layer.synapse.bias.data.copy_(torch_layer.synapse.bias)

    with pytest.warns(RuntimeWarning, match=SLOW_TORCH_BACKEND_WARNING):
        torch_state, torch_spikes = torch_layer(inputs)
    triton_state, triton_spikes = triton_layer(inputs)

    assert torch.allclose(triton_state.membrane, torch_state.membrane)
    assert torch.equal(triton_spikes, torch_spikes)


def test_linear_surrogate_lif_triton_backend_matches_torch_backend_on_cuda() -> None:
    from myelin.modules import LinearSurrogateLIF

    torch.manual_seed(124)
    inputs = torch.rand((6, 3, 11), device="cuda", requires_grad=True)
    params = LIFParams(tau_mem=9.0, threshold=0.7, reset=-0.2)
    torch_layer = LinearSurrogateLIF(
        11,
        17,
        params,
        surrogate=fast_sigmoid_surrogate,
        surrogate_slope=5.0,
        backend="torch",
    ).to(device="cuda")
    triton_layer = LinearSurrogateLIF(
        11,
        17,
        params,
        surrogate=fast_sigmoid_surrogate,
        surrogate_slope=5.0,
        backend="triton",
    ).to(device="cuda")
    triton_layer.synapse.weight.data.copy_(torch_layer.synapse.weight)
    assert torch_layer.synapse.bias is not None
    assert triton_layer.synapse.bias is not None
    triton_layer.synapse.bias.data.copy_(torch_layer.synapse.bias)

    expected_spikes = torch_layer(inputs)
    expected_loss = expected_spikes.mean()
    expected_loss.backward()
    expected_weight_grad = torch_layer.synapse.weight.grad
    assert expected_weight_grad is not None

    actual_spikes = triton_layer(inputs)
    actual_loss = actual_spikes.mean()
    actual_loss.backward()
    actual_weight_grad = triton_layer.synapse.weight.grad
    assert actual_weight_grad is not None

    assert torch.allclose(actual_spikes, expected_spikes.detach())
    assert torch.allclose(actual_weight_grad, expected_weight_grad)


def test_linear_surrogate_lif_generated_triton_backend_matches_triton_backend_on_cuda() -> None:
    from myelin.modules import LinearSurrogateLIF

    torch.manual_seed(126)
    inputs = torch.rand((6, 3, 11), device="cuda", requires_grad=True)
    generated_inputs = inputs.detach().clone().requires_grad_()
    params = LIFParams(tau_mem=9.0, threshold=0.7, reset=-0.2)
    triton_layer = LinearSurrogateLIF(
        11,
        17,
        params,
        surrogate=fast_sigmoid_surrogate,
        surrogate_slope=5.0,
        backend="triton",
    ).to(device="cuda")
    generated_layer = LinearSurrogateLIF(
        11,
        17,
        params,
        surrogate=fast_sigmoid_surrogate,
        surrogate_slope=5.0,
        backend="triton_generated",
    ).to(device="cuda")
    generated_layer.synapse.weight.data.copy_(triton_layer.synapse.weight)
    assert triton_layer.synapse.bias is not None
    assert generated_layer.synapse.bias is not None
    generated_layer.synapse.bias.data.copy_(triton_layer.synapse.bias)

    expected_spikes = triton_layer(inputs)
    expected_loss = expected_spikes.mean()
    expected_loss.backward()
    expected_weight_grad = triton_layer.synapse.weight.grad
    expected_input_grad = inputs.grad
    assert expected_weight_grad is not None
    assert expected_input_grad is not None

    actual_spikes = generated_layer(generated_inputs)
    actual_loss = actual_spikes.mean()
    actual_loss.backward()
    actual_weight_grad = generated_layer.synapse.weight.grad
    actual_input_grad = generated_inputs.grad
    assert actual_weight_grad is not None
    assert actual_input_grad is not None

    assert torch.allclose(actual_spikes, expected_spikes.detach())
    assert torch.allclose(actual_input_grad, expected_input_grad)
    assert torch.allclose(actual_weight_grad, expected_weight_grad)


def test_generated_triton_linear_surrogate_lif_stream_backend_matches_triton_backend_on_cuda() -> (
    None
):
    from myelin.modules import LinearSurrogateLIF

    torch.manual_seed(127)
    inputs = torch.rand((6, 3, 11), device="cuda", requires_grad=True)
    generated_inputs = inputs.detach().clone().requires_grad_()
    params = LIFParams(tau_mem=9.0, threshold=0.7, reset=-0.2)
    triton_layer = LinearSurrogateLIF(
        11,
        17,
        params,
        surrogate=fast_sigmoid_surrogate,
        surrogate_slope=5.0,
        backend="triton",
        stream_synapse=True,
    ).to(device="cuda")
    generated_layer = LinearSurrogateLIF(
        11,
        17,
        params,
        surrogate=fast_sigmoid_surrogate,
        surrogate_slope=5.0,
        backend="triton_generated",
        stream_synapse=True,
    ).to(device="cuda")
    generated_layer.synapse.weight.data.copy_(triton_layer.synapse.weight)
    assert triton_layer.synapse.bias is not None
    assert generated_layer.synapse.bias is not None
    generated_layer.synapse.bias.data.copy_(triton_layer.synapse.bias)

    expected_spikes = triton_layer(inputs)
    expected_loss = expected_spikes.mean()
    expected_loss.backward()
    expected_input_grad = inputs.grad
    expected_weight_grad = triton_layer.synapse.weight.grad
    expected_bias_grad = triton_layer.synapse.bias.grad
    assert expected_input_grad is not None
    assert expected_weight_grad is not None
    assert expected_bias_grad is not None

    actual_spikes = generated_layer(generated_inputs)
    actual_loss = actual_spikes.mean()
    actual_loss.backward()
    actual_input_grad = generated_inputs.grad
    actual_weight_grad = generated_layer.synapse.weight.grad
    actual_bias_grad = generated_layer.synapse.bias.grad
    assert actual_input_grad is not None
    assert actual_weight_grad is not None
    assert actual_bias_grad is not None

    assert torch.allclose(actual_spikes, expected_spikes.detach())
    assert torch.allclose(actual_input_grad, expected_input_grad, atol=1e-6, rtol=1e-5)
    assert torch.allclose(actual_weight_grad, expected_weight_grad, atol=1e-6, rtol=1e-5)
    assert torch.allclose(actual_bias_grad, expected_bias_grad, atol=1e-6, rtol=1e-5)


def test_generated_triton_linear_surrogate_lif_stream_supports_input_gradients() -> None:
    from myelin.modules import LinearSurrogateLIF

    torch.manual_seed(128)
    inputs = torch.rand((6, 3, 11), device="cuda", requires_grad=True)
    layer = LinearSurrogateLIF(
        11,
        17,
        surrogate=fast_sigmoid_surrogate,
        surrogate_slope=5.0,
        backend="triton_generated",
        stream_synapse=True,
    ).to(device="cuda")

    layer(inputs).mean().backward()

    assert inputs.grad is not None
    assert layer.synapse.weight.grad is not None


def test_triton_linear_surrogate_lif_forward_matches_streamed_reference() -> None:
    from myelin.autograd import linear_surrogate_lif_stream_function
    from myelin.triton import linear_surrogate_lif_forward

    torch.manual_seed(125)
    inputs = torch.rand((7, 3, 11), device="cuda")
    weight = (torch.rand((11, 17), device="cuda") - 0.5) * 0.02
    bias = torch.rand((17,), device="cuda") * 0.01
    params = LIFParams(tau_mem=9.0, threshold=0.7, reset=-0.2)

    expected_state, expected_spikes = linear_surrogate_lif_stream_function(
        inputs,
        weight,
        bias,
        params,
        surrogate="fast_sigmoid",
        surrogate_slope=5.0,
    )
    actual_state, actual_spikes, actual_pre_reset = linear_surrogate_lif_forward(
        inputs,
        weight,
        bias,
        params,
        block_b=16,
        block_n=16,
        block_f=16,
    )

    assert torch.allclose(actual_state.membrane, expected_state.membrane, atol=1e-4, rtol=1e-3)
    assert torch.equal(actual_spikes, expected_spikes.detach())
    assert actual_pre_reset.shape == actual_spikes.shape


def test_public_linear_surrogate_lif_forward_triton_matches_torch_dispatcher() -> None:
    from myelin.kernels import linear_surrogate_lif_forward

    torch.manual_seed(135)
    inputs = torch.rand((6, 3, 11), device="cuda", requires_grad=True)
    triton_inputs = inputs.detach().clone().requires_grad_()
    weight = ((torch.rand((11, 17), device="cuda") - 0.5) * 0.02).requires_grad_(True)
    triton_weight = weight.detach().clone().requires_grad_()
    bias = torch.zeros((17,), device="cuda", requires_grad=True)
    triton_bias = bias.detach().clone().requires_grad_()
    params = LIFParams(tau_mem=9.0, threshold=0.7, reset=-0.2)

    with pytest.warns(RuntimeWarning, match=SLOW_TORCH_BACKEND_WARNING):
        expected_state, expected_spikes = linear_surrogate_lif_forward(
            inputs,
            weight,
            bias,
            params,
            surrogate="fast_sigmoid",
            surrogate_slope=5.0,
            backend="torch",
        )
    expected_loss = expected_spikes.mean() + 0.01 * expected_state.membrane.square().mean()
    expected_loss.backward()

    actual_state, actual_spikes = linear_surrogate_lif_forward(
        triton_inputs,
        triton_weight,
        triton_bias,
        params,
        surrogate="fast_sigmoid",
        surrogate_slope=5.0,
        backend="triton",
    )
    actual_loss = actual_spikes.mean() + 0.01 * actual_state.membrane.square().mean()
    actual_loss.backward()

    assert torch.equal(actual_spikes, expected_spikes.detach())
    assert torch.allclose(
        actual_state.membrane,
        expected_state.membrane.detach(),
        atol=1e-4,
        rtol=1e-3,
    )
    assert weight.grad is not None
    assert triton_weight.grad is not None
    assert bias.grad is not None
    assert triton_bias.grad is not None
    assert torch.allclose(triton_weight.grad, weight.grad, atol=1e-4, rtol=1e-3)
    assert torch.allclose(triton_bias.grad, bias.grad, atol=1e-4, rtol=1e-3)


def test_triton_linear_surrogate_lif_forward_matches_materialized_reference_without_bias() -> None:
    from myelin.triton import linear_surrogate_lif_forward

    torch.manual_seed(126)
    inputs = torch.rand((5, 5, 13), device="cuda")
    weight = (torch.rand((13, 19), device="cuda") - 0.5) * 0.02
    params = LIFParams(tau_mem=7.0, threshold=0.8, reset=0.1)
    currents = torch.matmul(inputs, weight)
    expected_state, expected_spikes = surrogate_lif_unroll(
        currents,
        LIFState(membrane=torch.zeros((5, 19), device="cuda")),
        params,
        surrogate=fast_sigmoid_surrogate,
        surrogate_slope=5.0,
        hard_forward=True,
    )

    actual_state, actual_spikes, _pre_reset = linear_surrogate_lif_forward(
        inputs,
        weight,
        None,
        params,
        block_b=4,
        block_n=16,
        block_f=16,
    )

    assert torch.allclose(actual_state.membrane, expected_state.membrane, atol=1e-4, rtol=1e-3)
    assert torch.equal(actual_spikes, expected_spikes.detach())


def test_triton_linear_surrogate_lif_backward_matches_streamed_reference_without_bias() -> None:
    from myelin.autograd import (
        linear_surrogate_lif_stream_function,
        triton_linear_surrogate_lif_function,
    )

    torch.manual_seed(128)
    inputs = torch.rand((6, 5, 19), device="cuda", requires_grad=True)
    weight = ((torch.rand((19, 23), device="cuda") - 0.5) * 0.02).requires_grad_()
    ref_inputs = inputs.detach().clone().requires_grad_()
    ref_weight = weight.detach().clone().requires_grad_()
    params = LIFParams(tau_mem=7.0, threshold=0.8, reset=0.1)

    expected_state, expected_spikes = linear_surrogate_lif_stream_function(
        ref_inputs,
        ref_weight,
        None,
        params,
        surrogate="fast_sigmoid",
        surrogate_slope=5.0,
    )
    expected_loss = expected_spikes.mean() + 0.1 * expected_state.membrane.square().mean()
    expected_loss.backward()

    actual_state, actual_spikes = triton_linear_surrogate_lif_function(
        inputs,
        weight,
        None,
        params,
        surrogate="fast_sigmoid",
        surrogate_slope=5.0,
    )
    actual_loss = actual_spikes.mean() + 0.1 * actual_state.membrane.square().mean()
    actual_loss.backward()

    assert torch.equal(actual_spikes, expected_spikes.detach())
    assert torch.allclose(
        actual_state.membrane, expected_state.membrane.detach(), atol=1e-4, rtol=1e-3
    )
    assert inputs.grad is not None
    assert ref_inputs.grad is not None
    assert weight.grad is not None
    assert ref_weight.grad is not None
    assert torch.allclose(inputs.grad, ref_inputs.grad, atol=1e-4, rtol=1e-3)
    assert torch.allclose(weight.grad, ref_weight.grad, atol=1e-4, rtol=1e-3)


def test_triton_linear_surrogate_lif_backward_matches_reference_without_input_grad() -> None:
    from myelin.autograd import (
        linear_surrogate_lif_stream_function,
        triton_linear_surrogate_lif_function,
    )

    torch.manual_seed(129)
    inputs = torch.rand((6, 5, 19), device="cuda")
    weight = ((torch.rand((19, 23), device="cuda") - 0.5) * 0.02).requires_grad_()
    bias = (torch.rand((23,), device="cuda") * 0.01).requires_grad_()
    ref_weight = weight.detach().clone().requires_grad_()
    ref_bias = bias.detach().clone().requires_grad_()
    params = LIFParams(tau_mem=7.0, threshold=0.8, reset=0.1)

    expected_state, expected_spikes = linear_surrogate_lif_stream_function(
        inputs,
        ref_weight,
        ref_bias,
        params,
        surrogate="fast_sigmoid",
        surrogate_slope=5.0,
    )
    expected_loss = expected_spikes.mean() + 0.1 * expected_state.membrane.square().mean()
    expected_loss.backward()

    actual_state, actual_spikes = triton_linear_surrogate_lif_function(
        inputs,
        weight,
        bias,
        params,
        surrogate="fast_sigmoid",
        surrogate_slope=5.0,
    )
    actual_loss = actual_spikes.mean() + 0.1 * actual_state.membrane.square().mean()
    actual_loss.backward()

    assert torch.equal(actual_spikes, expected_spikes.detach())
    assert torch.allclose(
        actual_state.membrane, expected_state.membrane.detach(), atol=1e-4, rtol=1e-3
    )
    assert weight.grad is not None
    assert ref_weight.grad is not None
    assert bias.grad is not None
    assert ref_bias.grad is not None
    assert torch.allclose(weight.grad, ref_weight.grad, atol=1e-4, rtol=1e-3)
    assert torch.allclose(bias.grad, ref_bias.grad, atol=1e-4, rtol=1e-3)


@pytest.mark.parametrize("neurons", [17, 32])
def test_generated_linear_lif_surrogate_backward_packed_spikes_matches_dense_generated(
    neurons: int,
) -> None:
    from myelin.triton import (
        generated_linear_lif_surrogate_backward_weight_bias,
        generated_linear_lif_surrogate_backward_weight_bias_packed_spikes,
        linear_surrogate_lif_forward,
        pack_spikes_triton,
    )

    torch.manual_seed(130 + neurons)
    inputs = torch.rand((6, 5, 19), device="cuda")
    weight = (torch.rand((19, neurons), device="cuda") - 0.5) * 0.02
    bias = torch.rand((neurons,), device="cuda") * 0.01
    params = LIFParams(tau_mem=7.0, threshold=0.8, reset=0.1)
    _state, spikes, pre_reset = linear_surrogate_lif_forward(
        inputs,
        weight,
        bias,
        params,
        block_b=4,
        block_n=16,
        block_f=16,
    )
    packed_spikes = pack_spikes_triton(spikes)
    grad_final = torch.rand((5, neurons), device="cuda") - 0.5
    grad_spikes = torch.rand_like(spikes) - 0.5

    expected_inputs, expected_weight, expected_bias = (
        generated_linear_lif_surrogate_backward_weight_bias(
            inputs,
            weight,
            pre_reset,
            spikes,
            grad_final,
            grad_spikes,
            params,
            surrogate="fast_sigmoid",
            surrogate_slope=5.0,
            block_b=16,
            block_n=16,
            block_f=16,
        )
    )
    actual_inputs, actual_weight, actual_bias = (
        generated_linear_lif_surrogate_backward_weight_bias_packed_spikes(
            inputs,
            weight,
            pre_reset,
            packed_spikes.data,
            grad_final,
            grad_spikes,
            params,
            surrogate="fast_sigmoid",
            surrogate_slope=5.0,
            block_b=16,
            block_n=16,
            block_f=16,
        )
    )

    assert expected_inputs is not None
    assert expected_weight is not None
    assert expected_bias is not None
    assert actual_inputs is not None
    assert actual_weight is not None
    assert actual_bias is not None
    assert torch.allclose(actual_inputs, expected_inputs, atol=1e-4, rtol=1e-3)
    assert torch.allclose(actual_weight, expected_weight, atol=1e-4, rtol=1e-3)
    assert torch.allclose(actual_bias, expected_bias, atol=1e-4, rtol=1e-3)


def test_triton_linear_surrogate_lif_checkpoint_matches_reference_without_input_grad() -> None:
    from myelin.autograd import (
        linear_surrogate_lif_checkpoint_function,
        triton_linear_surrogate_lif_checkpoint_function,
    )

    torch.manual_seed(130)
    inputs = torch.rand((7, 5, 19), device="cuda")
    weight = ((torch.rand((19, 23), device="cuda") - 0.5) * 0.02).requires_grad_()
    bias = (torch.rand((23,), device="cuda") * 0.01).requires_grad_()
    ref_weight = weight.detach().clone().requires_grad_()
    ref_bias = bias.detach().clone().requires_grad_()
    params = LIFParams(tau_mem=7.0, threshold=0.8, reset=0.1)

    expected_state, expected_spikes = linear_surrogate_lif_checkpoint_function(
        inputs,
        ref_weight,
        ref_bias,
        params,
        surrogate="fast_sigmoid",
        surrogate_slope=5.0,
        checkpoint_size=3,
    )
    expected_loss = expected_spikes.mean() + 0.1 * expected_state.membrane.square().mean()
    expected_loss.backward()

    actual_state, actual_spikes = triton_linear_surrogate_lif_checkpoint_function(
        inputs,
        weight,
        bias,
        params,
        surrogate="fast_sigmoid",
        surrogate_slope=5.0,
        checkpoint_size=3,
        block_b=16,
        block_n=16,
        block_f=16,
    )
    actual_loss = actual_spikes.mean() + 0.1 * actual_state.membrane.square().mean()
    actual_loss.backward()

    assert torch.equal(actual_spikes, expected_spikes.detach())
    assert torch.allclose(
        actual_state.membrane, expected_state.membrane.detach(), atol=1e-4, rtol=1e-3
    )
    assert weight.grad is not None
    assert ref_weight.grad is not None
    assert bias.grad is not None
    assert ref_bias.grad is not None
    assert torch.allclose(weight.grad, ref_weight.grad, atol=1e-4, rtol=1e-3)
    assert torch.allclose(bias.grad, ref_bias.grad, atol=1e-4, rtol=1e-3)


@pytest.mark.parametrize("checkpoint_size", [1, 4, 9])
@pytest.mark.parametrize("has_bias", [False, True])
def test_triton_linear_surrogate_lif_checkpoint_matches_reference_across_chunk_edges(
    checkpoint_size: int,
    has_bias: bool,
) -> None:
    from myelin.autograd import (
        linear_surrogate_lif_checkpoint_function,
        triton_linear_surrogate_lif_checkpoint_function,
    )

    torch.manual_seed(132 + checkpoint_size + int(has_bias))
    inputs = torch.rand((9, 3, 7), device="cuda", requires_grad=True)
    weight = ((torch.rand((7, 11), device="cuda") - 0.5) * 0.03).requires_grad_()
    ref_weight = weight.detach().clone().requires_grad_()
    bias = (torch.rand((11,), device="cuda") * 0.01).requires_grad_() if has_bias else None
    ref_bias = None if bias is None else bias.detach().clone().requires_grad_()
    params = LIFParams(tau_mem=6.0, threshold=0.65, reset=-0.15)
    spike_grad_scale = torch.linspace(0.1, 1.0, steps=9, device="cuda").reshape(9, 1, 1)

    expected_state, expected_spikes = linear_surrogate_lif_checkpoint_function(
        inputs,
        ref_weight,
        ref_bias,
        params,
        surrogate="fast_sigmoid",
        surrogate_slope=4.0,
        checkpoint_size=checkpoint_size,
    )
    expected_loss = (
        (expected_spikes * spike_grad_scale).mean()
        + 0.2 * expected_spikes.square().mean()
        + 0.05 * expected_state.membrane.square().mean()
    )
    expected_loss.backward()

    actual_state, actual_spikes = triton_linear_surrogate_lif_checkpoint_function(
        inputs,
        weight,
        bias,
        params,
        surrogate="fast_sigmoid",
        surrogate_slope=4.0,
        checkpoint_size=checkpoint_size,
        block_b=16,
        block_n=16,
        block_f=16,
    )
    actual_loss = (
        (actual_spikes * spike_grad_scale).mean()
        + 0.2 * actual_spikes.square().mean()
        + 0.05 * actual_state.membrane.square().mean()
    )
    actual_loss.backward()

    assert torch.equal(actual_spikes, expected_spikes.detach())
    assert torch.allclose(
        actual_state.membrane,
        expected_state.membrane.detach(),
        atol=1e-4,
        rtol=1e-3,
    )
    assert weight.grad is not None
    assert ref_weight.grad is not None
    assert torch.allclose(weight.grad, ref_weight.grad, atol=1e-4, rtol=1e-3)
    if bias is not None and ref_bias is not None:
        assert bias.grad is not None
        assert ref_bias.grad is not None
        assert torch.allclose(bias.grad, ref_bias.grad, atol=1e-4, rtol=1e-3)


@pytest.mark.parametrize("checkpoint_size", [1, 9])
@pytest.mark.parametrize("has_bias", [False, True])
def test_generated_triton_linear_surrogate_lif_checkpoint_matches_triton_backend(
    checkpoint_size: int,
    has_bias: bool,
) -> None:
    from myelin.autograd import (
        generated_triton_linear_surrogate_lif_checkpoint_function,
        triton_linear_surrogate_lif_checkpoint_function,
    )

    torch.manual_seed(140 + checkpoint_size + int(has_bias))
    inputs = torch.rand((9, 3, 7), device="cuda", requires_grad=True)
    generated_inputs = inputs.detach().clone().requires_grad_()
    weight = ((torch.rand((7, 11), device="cuda") - 0.5) * 0.03).requires_grad_()
    generated_weight = weight.detach().clone().requires_grad_()
    bias = (torch.rand((11,), device="cuda") * 0.01).requires_grad_() if has_bias else None
    generated_bias = None if bias is None else bias.detach().clone().requires_grad_()
    params = LIFParams(tau_mem=6.0, threshold=0.65, reset=-0.15)
    spike_grad_scale = torch.linspace(0.1, 1.0, steps=9, device="cuda").reshape(9, 1, 1)

    expected_state, expected_spikes = triton_linear_surrogate_lif_checkpoint_function(
        inputs,
        weight,
        bias,
        params,
        surrogate="fast_sigmoid",
        surrogate_slope=4.0,
        checkpoint_size=checkpoint_size,
        block_b=16,
        block_n=16,
        block_f=16,
    )
    expected_loss = (
        (expected_spikes * spike_grad_scale).mean()
        + 0.2 * expected_spikes.square().mean()
        + 0.05 * expected_state.membrane.square().mean()
    )
    expected_loss.backward()

    actual_state, actual_spikes = generated_triton_linear_surrogate_lif_checkpoint_function(
        generated_inputs,
        generated_weight,
        generated_bias,
        params,
        surrogate="fast_sigmoid",
        surrogate_slope=4.0,
        checkpoint_size=checkpoint_size,
        block_b=16,
        block_n=16,
        block_f=16,
    )
    actual_loss = (
        (actual_spikes * spike_grad_scale).mean()
        + 0.2 * actual_spikes.square().mean()
        + 0.05 * actual_state.membrane.square().mean()
    )
    actual_loss.backward()

    assert torch.equal(actual_spikes, expected_spikes.detach())
    assert torch.allclose(
        actual_state.membrane,
        expected_state.membrane.detach(),
        atol=1e-4,
        rtol=1e-3,
    )
    assert weight.grad is not None
    assert generated_weight.grad is not None
    assert inputs.grad is not None
    assert generated_inputs.grad is not None
    assert torch.allclose(generated_inputs.grad, inputs.grad, atol=1e-4, rtol=1e-3)
    assert torch.allclose(generated_weight.grad, weight.grad, atol=1e-4, rtol=1e-3)
    if bias is not None and generated_bias is not None:
        assert bias.grad is not None
        assert generated_bias.grad is not None
        assert torch.allclose(generated_bias.grad, bias.grad, atol=1e-4, rtol=1e-3)


def test_linear_surrogate_lif_stream_synapse_triton_backend_matches_torch_backend() -> None:
    from myelin.modules import LinearSurrogateLIF

    torch.manual_seed(127)
    inputs = torch.rand((6, 3, 11), device="cuda", requires_grad=True)
    torch_inputs = inputs.detach().clone().requires_grad_()
    params = LIFParams(tau_mem=9.0, threshold=0.7, reset=-0.2)
    torch_layer = LinearSurrogateLIF(
        11,
        17,
        params,
        surrogate=fast_sigmoid_surrogate,
        surrogate_slope=5.0,
        backend="torch",
        stream_synapse=True,
    ).to(device="cuda")
    triton_layer = LinearSurrogateLIF(
        11,
        17,
        params,
        surrogate=fast_sigmoid_surrogate,
        surrogate_slope=5.0,
        backend="triton",
        stream_synapse=True,
    ).to(device="cuda")
    triton_layer.synapse.weight.data.copy_(torch_layer.synapse.weight)
    assert torch_layer.synapse.bias is not None
    assert triton_layer.synapse.bias is not None
    triton_layer.synapse.bias.data.copy_(torch_layer.synapse.bias)

    with pytest.warns(RuntimeWarning, match=SLOW_TORCH_BACKEND_WARNING):
        expected_spikes = torch_layer(torch_inputs)
    expected_loss = expected_spikes.mean()
    expected_loss.backward()
    actual_spikes = triton_layer(inputs)
    actual_loss = actual_spikes.mean()
    actual_loss.backward()

    assert torch.allclose(actual_spikes, expected_spikes.detach())
    assert inputs.grad is not None
    assert torch_inputs.grad is not None
    assert triton_layer.synapse.weight.grad is not None
    assert torch_layer.synapse.weight.grad is not None
    assert triton_layer.synapse.bias.grad is not None
    assert torch_layer.synapse.bias.grad is not None
    assert torch.allclose(inputs.grad, torch_inputs.grad, atol=1e-4, rtol=1e-3)
    assert torch.allclose(
        triton_layer.synapse.weight.grad,
        torch_layer.synapse.weight.grad,
        atol=1e-4,
        rtol=1e-3,
    )
    assert torch.allclose(
        triton_layer.synapse.bias.grad,
        torch_layer.synapse.bias.grad,
        atol=1e-4,
        rtol=1e-3,
    )


def test_linear_surrogate_lif_checkpoint_triton_backend_matches_torch_backend() -> None:
    from myelin.modules import LinearSurrogateLIF

    torch.manual_seed(131)
    inputs = torch.rand((7, 3, 11), device="cuda")
    params = LIFParams(tau_mem=9.0, threshold=0.7, reset=-0.2)
    torch_layer = LinearSurrogateLIF(
        11,
        17,
        params,
        surrogate=fast_sigmoid_surrogate,
        surrogate_slope=5.0,
        backend="torch",
        stream_synapse=True,
        checkpoint_size=3,
    ).to(device="cuda")
    triton_layer = LinearSurrogateLIF(
        11,
        17,
        params,
        surrogate=fast_sigmoid_surrogate,
        surrogate_slope=5.0,
        backend="triton",
        stream_synapse=True,
        checkpoint_size=3,
    ).to(device="cuda")
    triton_layer.synapse.weight.data.copy_(torch_layer.synapse.weight)
    assert torch_layer.synapse.bias is not None
    assert triton_layer.synapse.bias is not None
    triton_layer.synapse.bias.data.copy_(torch_layer.synapse.bias)

    with pytest.warns(RuntimeWarning, match=SLOW_TORCH_BACKEND_WARNING):
        expected_spikes = torch_layer(inputs)
    expected_loss = expected_spikes.mean()
    expected_loss.backward()
    actual_spikes = triton_layer(inputs)
    actual_loss = actual_spikes.mean()
    actual_loss.backward()

    assert torch.allclose(actual_spikes, expected_spikes.detach())
    assert triton_layer.synapse.weight.grad is not None
    assert torch_layer.synapse.weight.grad is not None
    assert triton_layer.synapse.bias.grad is not None
    assert torch_layer.synapse.bias.grad is not None
    assert torch.allclose(
        triton_layer.synapse.weight.grad,
        torch_layer.synapse.weight.grad,
        atol=1e-4,
        rtol=1e-3,
    )
    assert torch.allclose(
        triton_layer.synapse.bias.grad,
        torch_layer.synapse.bias.grad,
        atol=1e-4,
        rtol=1e-3,
    )


def test_linear_surrogate_lif_checkpoint_generated_triton_backend_matches_triton_backend() -> None:
    from myelin.modules import LinearSurrogateLIF

    torch.manual_seed(141)
    inputs = torch.rand((7, 3, 11), device="cuda")
    params = LIFParams(tau_mem=9.0, threshold=0.7, reset=-0.2)
    triton_layer = LinearSurrogateLIF(
        11,
        17,
        params,
        surrogate=fast_sigmoid_surrogate,
        surrogate_slope=5.0,
        backend="triton",
        stream_synapse=True,
        checkpoint_size=3,
    ).to(device="cuda")
    generated_layer = LinearSurrogateLIF(
        11,
        17,
        params,
        surrogate=fast_sigmoid_surrogate,
        surrogate_slope=5.0,
        backend="triton_generated",
        stream_synapse=True,
        checkpoint_size=3,
    ).to(device="cuda")
    generated_layer.synapse.weight.data.copy_(triton_layer.synapse.weight)
    assert triton_layer.synapse.bias is not None
    assert generated_layer.synapse.bias is not None
    generated_layer.synapse.bias.data.copy_(triton_layer.synapse.bias)

    expected_spikes = triton_layer(inputs)
    expected_loss = expected_spikes.mean()
    expected_loss.backward()
    actual_spikes = generated_layer(inputs)
    actual_loss = actual_spikes.mean()
    actual_loss.backward()

    assert torch.allclose(actual_spikes, expected_spikes.detach())
    assert triton_layer.synapse.weight.grad is not None
    assert generated_layer.synapse.weight.grad is not None
    assert triton_layer.synapse.bias.grad is not None
    assert generated_layer.synapse.bias.grad is not None
    assert torch.allclose(
        generated_layer.synapse.weight.grad,
        triton_layer.synapse.weight.grad,
        atol=1e-4,
        rtol=1e-3,
    )
    assert torch.allclose(
        generated_layer.synapse.bias.grad,
        triton_layer.synapse.bias.grad,
        atol=1e-4,
        rtol=1e-3,
    )


def test_linear_surrogate_lif_checkpoint_triton_backend_preserves_input_gradients() -> None:
    from myelin.modules import LinearSurrogateLIF

    torch.manual_seed(133)
    inputs = torch.rand((6, 3, 11), device="cuda", requires_grad=True)
    torch_inputs = inputs.detach().clone().requires_grad_()
    params = LIFParams(tau_mem=9.0, threshold=0.7, reset=-0.2)
    torch_layer = LinearSurrogateLIF(
        11,
        17,
        params,
        surrogate=fast_sigmoid_surrogate,
        surrogate_slope=5.0,
        backend="torch",
        stream_synapse=True,
        checkpoint_size=3,
    ).to(device="cuda")
    triton_layer = LinearSurrogateLIF(
        11,
        17,
        params,
        surrogate=fast_sigmoid_surrogate,
        surrogate_slope=5.0,
        backend="triton",
        stream_synapse=True,
        checkpoint_size=3,
    ).to(device="cuda")
    triton_layer.synapse.weight.data.copy_(torch_layer.synapse.weight)
    assert torch_layer.synapse.bias is not None
    assert triton_layer.synapse.bias is not None
    triton_layer.synapse.bias.data.copy_(torch_layer.synapse.bias)

    with pytest.warns(RuntimeWarning, match=SLOW_TORCH_BACKEND_WARNING):
        expected_spikes = torch_layer(torch_inputs)
    expected_loss = expected_spikes.mean() + 0.1 * expected_spikes.square().mean()
    expected_loss.backward()
    actual_spikes = triton_layer(inputs)
    actual_loss = actual_spikes.mean() + 0.1 * actual_spikes.square().mean()
    actual_loss.backward()

    assert torch.allclose(actual_spikes, expected_spikes.detach())
    assert inputs.grad is not None
    assert torch_inputs.grad is not None
    assert triton_layer.synapse.weight.grad is not None
    assert torch_layer.synapse.weight.grad is not None
    assert triton_layer.synapse.bias.grad is not None
    assert torch_layer.synapse.bias.grad is not None
    assert torch.allclose(inputs.grad, torch_inputs.grad, atol=1e-4, rtol=1e-3)
    assert torch.allclose(
        triton_layer.synapse.weight.grad,
        torch_layer.synapse.weight.grad,
        atol=1e-4,
        rtol=1e-3,
    )
    assert torch.allclose(
        triton_layer.synapse.bias.grad,
        torch_layer.synapse.bias.grad,
        atol=1e-4,
        rtol=1e-3,
    )


def test_direct_triton_checkpoint_function_matches_reference_with_input_gradients() -> None:
    from myelin.autograd import (
        linear_surrogate_lif_checkpoint_function,
        triton_linear_surrogate_lif_checkpoint_function,
    )

    torch.manual_seed(134)
    inputs = torch.rand((5, 2, 7), device="cuda", requires_grad=True)
    ref_inputs = inputs.detach().clone().requires_grad_()
    weight = ((torch.rand((7, 11), device="cuda") - 0.5) * 0.02).requires_grad_()
    ref_weight = weight.detach().clone().requires_grad_()
    params = LIFParams(tau_mem=8.0, threshold=0.7, reset=-0.2)

    expected_state, expected_spikes = linear_surrogate_lif_checkpoint_function(
        ref_inputs,
        ref_weight,
        None,
        params,
        surrogate="fast_sigmoid",
        surrogate_slope=5.0,
        checkpoint_size=2,
    )
    expected_loss = expected_spikes.mean() + 0.1 * expected_state.membrane.square().mean()
    expected_loss.backward()

    actual_state, actual_spikes = triton_linear_surrogate_lif_checkpoint_function(
        inputs,
        weight,
        None,
        params,
        surrogate="fast_sigmoid",
        surrogate_slope=5.0,
        checkpoint_size=2,
        block_b=16,
        block_n=16,
        block_f=16,
    )
    actual_loss = actual_spikes.mean() + 0.1 * actual_state.membrane.square().mean()
    actual_loss.backward()

    assert torch.equal(actual_spikes, expected_spikes.detach())
    assert torch.allclose(
        actual_state.membrane,
        expected_state.membrane.detach(),
        atol=1e-4,
        rtol=1e-3,
    )
    assert inputs.grad is not None
    assert ref_inputs.grad is not None
    assert weight.grad is not None
    assert ref_weight.grad is not None
    assert torch.allclose(inputs.grad, ref_inputs.grad, atol=1e-4, rtol=1e-3)
    assert torch.allclose(weight.grad, ref_weight.grad, atol=1e-4, rtol=1e-3)


def test_direct_triton_checkpoint_function_matches_reference_with_unaligned_neurons() -> None:
    from myelin.autograd import (
        linear_surrogate_lif_checkpoint_function,
        triton_linear_surrogate_lif_checkpoint_function,
    )

    torch.manual_seed(136)
    inputs = torch.rand((7, 3, 13), device="cuda", requires_grad=True)
    ref_inputs = inputs.detach().clone().requires_grad_()
    weight = ((torch.rand((13, 35), device="cuda") - 0.5) * 0.02).requires_grad_()
    ref_weight = weight.detach().clone().requires_grad_()
    bias = (torch.rand((35,), device="cuda") * 0.01).requires_grad_()
    ref_bias = bias.detach().clone().requires_grad_()
    params = LIFParams(tau_mem=8.0, threshold=0.7, reset=-0.2)

    expected_state, expected_spikes = linear_surrogate_lif_checkpoint_function(
        ref_inputs,
        ref_weight,
        ref_bias,
        params,
        surrogate="fast_sigmoid",
        surrogate_slope=5.0,
        checkpoint_size=3,
    )
    expected_loss = expected_spikes.mean() + 0.1 * expected_state.membrane.square().mean()
    expected_loss.backward()

    actual_state, actual_spikes = triton_linear_surrogate_lif_checkpoint_function(
        inputs,
        weight,
        bias,
        params,
        surrogate="fast_sigmoid",
        surrogate_slope=5.0,
        checkpoint_size=3,
        block_b=16,
        block_n=16,
        block_f=16,
    )
    actual_loss = actual_spikes.mean() + 0.1 * actual_state.membrane.square().mean()
    actual_loss.backward()

    assert torch.equal(actual_spikes, expected_spikes.detach())
    assert torch.allclose(
        actual_state.membrane,
        expected_state.membrane.detach(),
        atol=1e-4,
        rtol=1e-3,
    )
    assert inputs.grad is not None
    assert ref_inputs.grad is not None
    assert weight.grad is not None
    assert ref_weight.grad is not None
    assert bias.grad is not None
    assert ref_bias.grad is not None
    assert torch.allclose(inputs.grad, ref_inputs.grad, atol=1e-4, rtol=1e-3)
    assert torch.allclose(weight.grad, ref_weight.grad, atol=1e-4, rtol=1e-3)
    assert torch.allclose(bias.grad, ref_bias.grad, atol=1e-4, rtol=1e-3)


def test_triton_checkpoint_rate_function_matches_dense_checkpoint_reference() -> None:
    from myelin.autograd import (
        linear_surrogate_lif_checkpoint_function,
        triton_linear_surrogate_lif_checkpoint_rate_function,
    )

    torch.manual_seed(137)
    inputs = torch.rand((7, 3, 13), device="cuda", requires_grad=True)
    ref_inputs = inputs.detach().clone().requires_grad_()
    weight = ((torch.rand((13, 35), device="cuda") - 0.5) * 0.02).requires_grad_()
    ref_weight = weight.detach().clone().requires_grad_()
    bias = (torch.rand((35,), device="cuda") * 0.01).requires_grad_()
    ref_bias = bias.detach().clone().requires_grad_()
    params = LIFParams(tau_mem=8.0, threshold=0.7, reset=-0.2)

    expected_state, expected_spikes = linear_surrogate_lif_checkpoint_function(
        ref_inputs,
        ref_weight,
        ref_bias,
        params,
        surrogate="fast_sigmoid",
        surrogate_slope=5.0,
        checkpoint_size=3,
    )
    expected_rate = expected_spikes.mean()
    expected_loss = expected_rate.square() + 0.1 * expected_state.membrane.square().mean()
    expected_loss.backward()

    actual_state, actual_rate = triton_linear_surrogate_lif_checkpoint_rate_function(
        inputs,
        weight,
        bias,
        params,
        surrogate="fast_sigmoid",
        surrogate_slope=5.0,
        checkpoint_size=3,
        block_b=16,
        block_n=16,
        block_f=16,
    )
    actual_loss = actual_rate.square() + 0.1 * actual_state.membrane.square().mean()
    actual_loss.backward()

    assert torch.allclose(actual_rate, expected_rate.detach())
    assert torch.allclose(
        actual_state.membrane,
        expected_state.membrane.detach(),
        atol=1e-4,
        rtol=1e-3,
    )
    assert inputs.grad is not None
    assert ref_inputs.grad is not None
    assert weight.grad is not None
    assert ref_weight.grad is not None
    assert bias.grad is not None
    assert ref_bias.grad is not None
    assert torch.allclose(inputs.grad, ref_inputs.grad, atol=1e-4, rtol=1e-3)
    assert torch.allclose(weight.grad, ref_weight.grad, atol=1e-4, rtol=1e-3)
    assert torch.allclose(bias.grad, ref_bias.grad, atol=1e-4, rtol=1e-3)


def test_generated_triton_checkpoint_rate_function_matches_triton_rate_function() -> None:
    from myelin.autograd import (
        generated_triton_linear_surrogate_lif_checkpoint_rate_function,
        triton_linear_surrogate_lif_checkpoint_rate_function,
    )

    torch.manual_seed(143)
    inputs = torch.rand((7, 3, 13), device="cuda", requires_grad=True)
    generated_inputs = inputs.detach().clone().requires_grad_()
    weight = ((torch.rand((13, 35), device="cuda") - 0.5) * 0.02).requires_grad_()
    generated_weight = weight.detach().clone().requires_grad_()
    bias = (torch.rand((35,), device="cuda") * 0.01).requires_grad_()
    generated_bias = bias.detach().clone().requires_grad_()
    params = LIFParams(tau_mem=8.0, threshold=0.7, reset=-0.2)

    expected_state, expected_rate = triton_linear_surrogate_lif_checkpoint_rate_function(
        inputs,
        weight,
        bias,
        params,
        surrogate="fast_sigmoid",
        surrogate_slope=5.0,
        checkpoint_size=3,
        block_b=16,
        block_n=16,
        block_f=16,
    )
    expected_loss = expected_rate.square() + 0.1 * expected_state.membrane.square().mean()
    expected_loss.backward()

    actual_state, actual_rate = generated_triton_linear_surrogate_lif_checkpoint_rate_function(
        generated_inputs,
        generated_weight,
        generated_bias,
        params,
        surrogate="fast_sigmoid",
        surrogate_slope=5.0,
        checkpoint_size=3,
        block_b=16,
        block_n=16,
        block_f=16,
    )
    actual_loss = actual_rate.square() + 0.1 * actual_state.membrane.square().mean()
    actual_loss.backward()

    assert torch.allclose(actual_rate, expected_rate.detach())
    assert torch.allclose(
        actual_state.membrane,
        expected_state.membrane.detach(),
        atol=1e-4,
        rtol=1e-3,
    )
    assert inputs.grad is not None
    assert generated_inputs.grad is not None
    assert weight.grad is not None
    assert generated_weight.grad is not None
    assert bias.grad is not None
    assert generated_bias.grad is not None
    assert torch.allclose(generated_inputs.grad, inputs.grad, atol=1e-4, rtol=1e-3)
    assert torch.allclose(generated_weight.grad, weight.grad, atol=1e-4, rtol=1e-3)
    assert torch.allclose(generated_bias.grad, bias.grad, atol=1e-4, rtol=1e-3)


def test_triton_checkpoint_replay_weight_bias_matches_rate_backward() -> None:
    from myelin.autograd import triton_linear_surrogate_lif_checkpoint_rate_function
    from myelin.triton import (
        linear_surrogate_lif_checkpoint_backward_replay_weight_bias,
        linear_surrogate_lif_checkpoint_rate_forward,
    )

    torch.manual_seed(144)
    inputs = torch.rand((7, 3, 13), device="cuda")
    weight = ((torch.rand((13, 35), device="cuda") - 0.5) * 0.02).requires_grad_()
    replay_weight = weight.detach().clone()
    bias = (torch.rand((35,), device="cuda") * 0.01).requires_grad_()
    replay_bias = bias.detach().clone()
    params = LIFParams(tau_mem=8.0, threshold=0.7, reset=-0.2)

    expected_state, expected_rate = triton_linear_surrogate_lif_checkpoint_rate_function(
        inputs,
        weight,
        bias,
        params,
        surrogate="fast_sigmoid",
        surrogate_slope=5.0,
        checkpoint_size=3,
        block_b=16,
        block_n=16,
        block_f=16,
    )
    expected_loss = expected_rate.square() + 0.1 * expected_state.membrane.square().mean()
    expected_loss.backward()

    actual_state, actual_rates, chunk_start_membranes = (
        linear_surrogate_lif_checkpoint_rate_forward(
            inputs,
            replay_weight,
            replay_bias,
            params,
            checkpoint_size=3,
            block_b=16,
            block_n=16,
            block_f=16,
        )
    )
    actual_rate = actual_rates.mean()
    grad_final = 0.2 * actual_state.membrane / actual_state.membrane.numel()
    grad_rate = 2.0 * actual_rate
    actual_grad_weight, actual_grad_bias = (
        linear_surrogate_lif_checkpoint_backward_replay_weight_bias(
            inputs,
            replay_weight,
            replay_bias,
            chunk_start_membranes,
            grad_final,
            None,
            params,
            surrogate="fast_sigmoid",
            surrogate_slope=5.0,
            grad_spike_rate=grad_rate,
            checkpoint_size=3,
            block_b=16,
            block_n=16,
            block_f=16,
        )
    )

    assert torch.allclose(actual_rate, expected_rate.detach())
    assert weight.grad is not None
    assert bias.grad is not None
    assert actual_grad_weight is not None
    assert actual_grad_bias is not None
    assert torch.allclose(actual_grad_weight, weight.grad, atol=1e-4, rtol=1e-3)
    assert torch.allclose(actual_grad_bias, bias.grad, atol=1e-4, rtol=1e-3)


def test_public_linear_surrogate_lif_rate_forward_matches_dense_checkpoint_reference() -> None:
    from myelin.kernels import linear_surrogate_lif_forward, linear_surrogate_lif_rate_forward

    torch.manual_seed(138)
    inputs = torch.rand((6, 3, 13), device="cuda", requires_grad=True)
    ref_inputs = inputs.detach().clone().requires_grad_()
    weight = ((torch.rand((13, 35), device="cuda") - 0.5) * 0.02).requires_grad_()
    ref_weight = weight.detach().clone().requires_grad_()
    params = LIFParams(tau_mem=8.0, threshold=0.7, reset=-0.2)

    expected_state, expected_spikes = linear_surrogate_lif_forward(
        ref_inputs,
        ref_weight,
        None,
        params,
        surrogate="fast_sigmoid",
        surrogate_slope=5.0,
        backend="triton",
        checkpoint_size=3,
        block_b=16,
        block_n=16,
        block_f=16,
    )
    expected_rate = expected_spikes.mean()
    expected_loss = expected_rate.square() + 0.1 * expected_state.membrane.square().mean()
    expected_loss.backward()

    actual_state, actual_rate = linear_surrogate_lif_rate_forward(
        inputs,
        weight,
        None,
        params,
        surrogate="fast_sigmoid",
        surrogate_slope=5.0,
        backend="triton",
        checkpoint_size=3,
        block_b=16,
        block_n=16,
        block_f=16,
    )
    actual_loss = actual_rate.square() + 0.1 * actual_state.membrane.square().mean()
    actual_loss.backward()

    assert torch.allclose(actual_rate, expected_rate.detach())
    assert torch.allclose(actual_state.membrane, expected_state.membrane.detach())
    assert inputs.grad is not None
    assert ref_inputs.grad is not None
    assert weight.grad is not None
    assert ref_weight.grad is not None
    assert torch.allclose(inputs.grad, ref_inputs.grad, atol=1e-4, rtol=1e-3)
    assert torch.allclose(weight.grad, ref_weight.grad, atol=1e-4, rtol=1e-3)


def test_public_linear_surrogate_lif_packed_forward_auto_matches_dense_checkpoint() -> None:
    from myelin.kernels import linear_surrogate_lif_forward, linear_surrogate_lif_packed_forward
    from myelin.triton import unpack_spikes_triton

    torch.manual_seed(139)
    inputs = torch.rand((6, 3, 13), device="cuda")
    weight = (torch.rand((13, 35), device="cuda") - 0.5) * 0.02
    bias = (torch.rand((35,), device="cuda") - 0.5) * 0.01
    params = LIFParams(tau_mem=8.0, threshold=0.7, reset=-0.2)

    expected_state, expected_spikes = linear_surrogate_lif_forward(
        inputs,
        weight,
        bias,
        params,
        surrogate="fast_sigmoid",
        surrogate_slope=5.0,
        backend="triton",
        checkpoint_size=3,
        block_b=16,
        block_n=16,
        block_f=16,
    )
    actual_state, packed_spikes = linear_surrogate_lif_packed_forward(
        inputs,
        weight,
        bias,
        params,
        surrogate="fast_sigmoid",
        surrogate_slope=5.0,
        backend="auto",
        checkpoint_size=3,
        block_b=16,
        block_f=16,
    )
    actual_spikes = unpack_spikes_triton(packed_spikes, dtype=inputs.dtype)

    assert packed_spikes.original_shape == tuple(expected_spikes.shape)
    assert torch.equal(actual_spikes, expected_spikes)
    assert torch.allclose(actual_state.membrane, expected_state.membrane, atol=1e-4, rtol=1e-3)


def test_two_layer_packed_hidden_rate_matches_composed_dense_gradients() -> None:
    from myelin.kernels import (
        linear_surrogate_lif_forward,
        linear_surrogate_lif_rate_forward,
        two_layer_surrogate_lif_rate_packed_hidden_forward,
    )

    torch.manual_seed(151)
    inputs = torch.rand((6, 3, 13), device="cuda", requires_grad=True)
    ref_inputs = inputs.detach().clone().requires_grad_()
    hidden_weight = ((torch.rand((13, 35), device="cuda") - 0.5) * 0.04).requires_grad_()
    ref_hidden_weight = hidden_weight.detach().clone().requires_grad_()
    hidden_bias = ((torch.rand((35,), device="cuda") - 0.5) * 0.01).requires_grad_()
    ref_hidden_bias = hidden_bias.detach().clone().requires_grad_()
    output_weight = ((torch.rand((35, 5), device="cuda") - 0.5) * 0.04).requires_grad_()
    ref_output_weight = output_weight.detach().clone().requires_grad_()
    output_bias = ((torch.rand((5,), device="cuda") - 0.5) * 0.01).requires_grad_()
    ref_output_bias = output_bias.detach().clone().requires_grad_()
    params = LIFParams(tau_mem=8.0, threshold=0.7, reset=-0.2)
    target = torch.linspace(0.05, 0.45, steps=15, device="cuda").reshape(3, 5)

    _hidden_state, hidden_spikes = linear_surrogate_lif_forward(
        ref_inputs,
        ref_hidden_weight,
        ref_hidden_bias,
        params,
        surrogate="fast_sigmoid",
        surrogate_slope=5.0,
        backend="triton",
        checkpoint_size=3,
        block_b=16,
        block_n=16,
        block_f=16,
    )
    _output_state, expected_rates = linear_surrogate_lif_rate_forward(
        hidden_spikes,
        ref_output_weight,
        ref_output_bias,
        params,
        surrogate="fast_sigmoid",
        surrogate_slope=5.0,
        backend="triton",
        checkpoint_size=3,
        reduction="none",
        block_b=16,
        block_n=16,
        block_f=16,
    )
    expected_loss = (expected_rates - target).square().mean()
    expected_loss.backward()

    actual_rates = two_layer_surrogate_lif_rate_packed_hidden_forward(
        inputs,
        hidden_weight,
        hidden_bias,
        output_weight,
        output_bias,
        params,
        surrogate="fast_sigmoid",
        surrogate_slope=5.0,
        checkpoint_size=3,
        block_b=16,
        block_n=16,
        block_f=16,
    )
    actual_loss = (actual_rates - target).square().mean()
    actual_loss.backward()

    assert torch.allclose(actual_rates, expected_rates.detach())
    assert inputs.grad is not None
    assert ref_inputs.grad is not None
    assert hidden_weight.grad is not None
    assert ref_hidden_weight.grad is not None
    assert hidden_bias.grad is not None
    assert ref_hidden_bias.grad is not None
    assert output_weight.grad is not None
    assert ref_output_weight.grad is not None
    assert output_bias.grad is not None
    assert ref_output_bias.grad is not None
    assert torch.allclose(inputs.grad, ref_inputs.grad, atol=1e-4, rtol=1e-3)
    assert torch.allclose(hidden_weight.grad, ref_hidden_weight.grad, atol=1e-4, rtol=1e-3)
    assert torch.allclose(hidden_bias.grad, ref_hidden_bias.grad, atol=1e-4, rtol=1e-3)
    assert torch.allclose(output_weight.grad, ref_output_weight.grad, atol=1e-4, rtol=1e-3)
    assert torch.allclose(output_bias.grad, ref_output_bias.grad, atol=1e-4, rtol=1e-3)


def test_public_linear_surrogate_lif_rate_forward_none_matches_dense_rates() -> None:
    from myelin.kernels import linear_surrogate_lif_forward, linear_surrogate_lif_rate_forward

    torch.manual_seed(139)
    inputs = torch.rand((6, 3, 13), device="cuda", requires_grad=True)
    ref_inputs = inputs.detach().clone().requires_grad_()
    weight = ((torch.rand((13, 35), device="cuda") - 0.5) * 0.02).requires_grad_()
    ref_weight = weight.detach().clone().requires_grad_()
    params = LIFParams(tau_mem=8.0, threshold=0.7, reset=-0.2)
    rate_weights = torch.linspace(0.1, 1.0, steps=35, device="cuda").reshape(1, 35)

    expected_state, expected_spikes = linear_surrogate_lif_forward(
        ref_inputs,
        ref_weight,
        None,
        params,
        surrogate="fast_sigmoid",
        surrogate_slope=5.0,
        backend="triton",
        checkpoint_size=3,
        block_b=16,
        block_n=16,
        block_f=16,
    )
    expected_rates = expected_spikes.mean(dim=0)
    expected_loss = (expected_rates * rate_weights).mean() + 0.1 * (
        expected_state.membrane.square().mean()
    )
    expected_loss.backward()

    actual_state, actual_rates = linear_surrogate_lif_rate_forward(
        inputs,
        weight,
        None,
        params,
        surrogate="fast_sigmoid",
        surrogate_slope=5.0,
        backend="triton",
        checkpoint_size=3,
        block_b=16,
        block_n=16,
        block_f=16,
        reduction="none",
    )
    actual_loss = (actual_rates * rate_weights).mean() + 0.1 * (
        actual_state.membrane.square().mean()
    )
    actual_loss.backward()

    assert torch.allclose(actual_rates, expected_rates.detach())
    assert torch.allclose(actual_state.membrane, expected_state.membrane.detach())
    assert inputs.grad is not None
    assert ref_inputs.grad is not None
    assert weight.grad is not None
    assert ref_weight.grad is not None
    assert torch.allclose(inputs.grad, ref_inputs.grad, atol=1e-4, rtol=1e-3)
    assert torch.allclose(weight.grad, ref_weight.grad, atol=1e-4, rtol=1e-3)


def test_public_linear_surrogate_lif_rate_triton_compile_matches_triton_weight_grad() -> None:
    from myelin.kernels import linear_surrogate_lif_rate_forward

    torch.manual_seed(140)
    inputs = torch.rand((6, 3, 16), device="cuda")
    weight = ((torch.rand((16, 32), device="cuda") - 0.5) * 0.02).requires_grad_()
    reference_weight = weight.detach().clone().requires_grad_()
    params = LIFParams(tau_mem=8.0, threshold=0.7, reset=-0.2)
    target = torch.full((3, 32), 0.05, device="cuda")

    expected_state, expected_rates = linear_surrogate_lif_rate_forward(
        inputs,
        reference_weight,
        None,
        params,
        surrogate="fast_sigmoid",
        surrogate_slope=5.0,
        backend="triton",
        checkpoint_size=3,
        block_b=16,
        block_n=16,
        block_f=16,
        reduction="none",
    )
    expected_loss = (expected_rates - target).square().mean()
    expected_loss = expected_loss + 0.1 * expected_state.membrane.square().mean()
    expected_loss.backward()

    actual_state, actual_rates = linear_surrogate_lif_rate_forward(
        inputs,
        weight,
        None,
        params,
        surrogate="fast_sigmoid",
        surrogate_slope=5.0,
        backend="triton_compile",
        checkpoint_size=3,
        block_b=16,
        block_n=16,
        block_f=16,
        reduction="none",
    )
    actual_loss = (actual_rates - target).square().mean()
    actual_loss = actual_loss + 0.1 * actual_state.membrane.square().mean()
    actual_loss.backward()

    assert torch.allclose(actual_rates, expected_rates.detach())
    assert torch.allclose(actual_state.membrane, expected_state.membrane.detach())
    assert weight.grad is not None
    assert reference_weight.grad is not None
    assert torch.allclose(weight.grad, reference_weight.grad, atol=1e-4, rtol=1e-3)


def test_public_linear_surrogate_lif_rate_triton_compile_matches_triton_bias_grad() -> None:
    from myelin.kernels import linear_surrogate_lif_rate_forward

    torch.manual_seed(145)
    inputs = torch.rand((6, 3, 16), device="cuda")
    weight = ((torch.rand((16, 32), device="cuda") - 0.5) * 0.02).requires_grad_()
    bias = ((torch.rand((32,), device="cuda") - 0.5) * 0.01).requires_grad_()
    reference_weight = weight.detach().clone().requires_grad_()
    reference_bias = bias.detach().clone().requires_grad_()
    params = LIFParams(tau_mem=8.0, threshold=0.7, reset=-0.2)
    target = torch.full((3, 32), 0.05, device="cuda")

    expected_state, expected_rates = linear_surrogate_lif_rate_forward(
        inputs,
        reference_weight,
        reference_bias,
        params,
        surrogate="fast_sigmoid",
        surrogate_slope=5.0,
        backend="triton",
        checkpoint_size=3,
        block_b=16,
        block_n=16,
        block_f=16,
        reduction="none",
    )
    expected_loss = (expected_rates - target).square().mean()
    expected_loss = expected_loss + 0.1 * expected_state.membrane.square().mean()
    expected_loss.backward()

    actual_state, actual_rates = linear_surrogate_lif_rate_forward(
        inputs,
        weight,
        bias,
        params,
        surrogate="fast_sigmoid",
        surrogate_slope=5.0,
        backend="triton_compile",
        checkpoint_size=3,
        block_b=16,
        block_n=16,
        block_f=16,
        reduction="none",
    )
    actual_loss = (actual_rates - target).square().mean()
    actual_loss = actual_loss + 0.1 * actual_state.membrane.square().mean()
    actual_loss.backward()

    assert torch.allclose(actual_rates, expected_rates.detach())
    assert torch.allclose(actual_state.membrane, expected_state.membrane.detach())
    assert weight.grad is not None
    assert bias.grad is not None
    assert reference_weight.grad is not None
    assert reference_bias.grad is not None
    assert torch.allclose(weight.grad, reference_weight.grad, atol=1e-4, rtol=1e-3)
    assert torch.allclose(bias.grad, reference_bias.grad, atol=1e-4, rtol=1e-3)


def test_public_linear_surrogate_lif_rate_triton_compile_rejects_unsupported_cases() -> None:
    from myelin.kernels import linear_surrogate_lif_rate_forward

    inputs = torch.rand((6, 3, 16), device="cuda")
    weight = torch.rand((16, 32), device="cuda")
    params = LIFParams()

    with pytest.raises(ValueError, match="surrogate='fast_sigmoid'"):
        linear_surrogate_lif_rate_forward(
            inputs,
            weight,
            None,
            params,
            surrogate="atan",
            backend="triton_compile",
        )
    with pytest.raises(ValueError, match="experimental.*hard_forward=True"):
        linear_surrogate_lif_rate_forward(
            inputs,
            weight,
            None,
            params,
            surrogate="fast_sigmoid",
            hard_forward=False,
            backend="triton_compile",
        )


def test_public_linear_surrogate_lif_rate_generated_backend_matches_triton_backend() -> None:
    from myelin.kernels import linear_surrogate_lif_rate_forward

    torch.manual_seed(144)
    inputs = torch.rand((6, 3, 13), device="cuda", requires_grad=True)
    generated_inputs = inputs.detach().clone().requires_grad_()
    weight = ((torch.rand((13, 35), device="cuda") - 0.5) * 0.02).requires_grad_()
    generated_weight = weight.detach().clone().requires_grad_()
    params = LIFParams(tau_mem=8.0, threshold=0.7, reset=-0.2)
    rate_weights = torch.linspace(0.1, 1.0, steps=35, device="cuda").reshape(1, 35)

    expected_state, expected_rates = linear_surrogate_lif_rate_forward(
        inputs,
        weight,
        None,
        params,
        surrogate="fast_sigmoid",
        surrogate_slope=5.0,
        backend="triton",
        checkpoint_size=3,
        block_b=16,
        block_n=16,
        block_f=16,
        reduction="none",
    )
    expected_loss = (expected_rates * rate_weights).mean() + 0.1 * (
        expected_state.membrane.square().mean()
    )
    expected_loss.backward()

    actual_state, actual_rates = linear_surrogate_lif_rate_forward(
        generated_inputs,
        generated_weight,
        None,
        params,
        surrogate="fast_sigmoid",
        surrogate_slope=5.0,
        backend="triton_generated",
        checkpoint_size=3,
        block_b=16,
        block_n=16,
        block_f=16,
        reduction="none",
    )
    actual_loss = (actual_rates * rate_weights).mean() + 0.1 * (
        actual_state.membrane.square().mean()
    )
    actual_loss.backward()

    assert torch.allclose(actual_rates, expected_rates.detach())
    assert torch.allclose(actual_state.membrane, expected_state.membrane.detach())
    assert inputs.grad is not None
    assert generated_inputs.grad is not None
    assert weight.grad is not None
    assert generated_weight.grad is not None
    assert torch.allclose(generated_inputs.grad, inputs.grad, atol=1e-4, rtol=1e-3)
    assert torch.allclose(generated_weight.grad, weight.grad, atol=1e-4, rtol=1e-3)


def test_linear_surrogate_lif_rate_module_triton_matches_functional_rate_path() -> None:
    from myelin.kernels import linear_surrogate_lif_rate_forward
    from myelin.modules import LinearSurrogateLIFRate

    torch.manual_seed(141)
    inputs = torch.rand((6, 3, 13), device="cuda", requires_grad=True)
    ref_inputs = inputs.detach().clone().requires_grad_()
    params = LIFParams(tau_mem=8.0, threshold=0.7, reset=-0.2)
    layer = LinearSurrogateLIFRate(
        13,
        35,
        params,
        surrogate=fast_sigmoid_surrogate,
        surrogate_slope=5.0,
        backend="triton",
        checkpoint_size=3,
        reduction="none",
    ).to(device="cuda")
    ref_weight = layer.synapse.weight.detach().clone().requires_grad_()
    assert layer.synapse.bias is not None
    ref_bias = layer.synapse.bias.detach().clone().requires_grad_()
    rate_weights = torch.linspace(0.1, 1.0, steps=35, device="cuda").reshape(1, 35)

    _expected_state, expected_rates = linear_surrogate_lif_rate_forward(
        ref_inputs,
        ref_weight,
        ref_bias,
        params,
        surrogate="fast_sigmoid",
        surrogate_slope=5.0,
        backend="triton",
        checkpoint_size=3,
        block_b=16,
        block_n=16,
        block_f=16,
        reduction="none",
    )
    expected_loss = (expected_rates * rate_weights).mean()
    expected_loss.backward()

    actual_rates = layer(inputs)
    actual_loss = (actual_rates * rate_weights).mean()
    actual_loss.backward()

    assert torch.allclose(actual_rates, expected_rates.detach())
    assert inputs.grad is not None
    assert ref_inputs.grad is not None
    assert layer.synapse.weight.grad is not None
    assert ref_weight.grad is not None
    assert layer.synapse.bias.grad is not None
    assert ref_bias.grad is not None
    assert torch.allclose(inputs.grad, ref_inputs.grad, atol=1e-4, rtol=1e-3)
    assert torch.allclose(layer.synapse.weight.grad, ref_weight.grad, atol=1e-4, rtol=1e-3)
    assert torch.allclose(layer.synapse.bias.grad, ref_bias.grad, atol=1e-4, rtol=1e-3)


def test_linear_surrogate_lif_packed_module_triton_matches_functional_packed_path() -> None:
    from myelin.kernels import linear_surrogate_lif_packed_forward
    from myelin.modules import LinearSurrogateLIFPacked
    from myelin.triton import unpack_spikes_triton

    torch.manual_seed(146)
    inputs = torch.rand((6, 3, 13), device="cuda")
    params = LIFParams(tau_mem=8.0, threshold=0.7, reset=-0.2)
    layer = LinearSurrogateLIFPacked(
        13,
        35,
        params,
        surrogate=fast_sigmoid_surrogate,
        surrogate_slope=5.0,
        backend="triton",
        checkpoint_size=3,
    ).to(device="cuda")
    assert layer.synapse.bias is not None

    _expected_state, expected_packed = linear_surrogate_lif_packed_forward(
        inputs,
        layer.synapse.weight,
        layer.synapse.bias,
        params,
        surrogate="fast_sigmoid",
        surrogate_slope=5.0,
        backend="triton",
        checkpoint_size=3,
        block_b=16,
        block_f=16,
    )
    actual_packed = layer(inputs)

    assert actual_packed.original_shape == expected_packed.original_shape
    assert torch.equal(actual_packed.data, expected_packed.data)
    assert torch.equal(
        unpack_spikes_triton(actual_packed, dtype=inputs.dtype),
        unpack_spikes_triton(expected_packed, dtype=inputs.dtype),
    )


def test_linear_surrogate_lif_rate_module_generated_backend_matches_triton_rate_path() -> None:
    from myelin.kernels import linear_surrogate_lif_rate_forward
    from myelin.modules import LinearSurrogateLIFRate

    torch.manual_seed(145)
    inputs = torch.rand((6, 3, 13), device="cuda", requires_grad=True)
    generated_inputs = inputs.detach().clone().requires_grad_()
    params = LIFParams(tau_mem=8.0, threshold=0.7, reset=-0.2)
    layer = LinearSurrogateLIFRate(
        13,
        35,
        params,
        surrogate=fast_sigmoid_surrogate,
        surrogate_slope=5.0,
        backend="triton_generated",
        checkpoint_size=3,
        reduction="none",
    ).to(device="cuda")
    weight = layer.synapse.weight.detach().clone().requires_grad_()
    assert layer.synapse.bias is not None
    bias = layer.synapse.bias.detach().clone().requires_grad_()
    rate_weights = torch.linspace(0.1, 1.0, steps=35, device="cuda").reshape(1, 35)

    _expected_state, expected_rates = linear_surrogate_lif_rate_forward(
        inputs,
        weight,
        bias,
        params,
        surrogate="fast_sigmoid",
        surrogate_slope=5.0,
        backend="triton",
        checkpoint_size=3,
        block_b=16,
        block_n=16,
        block_f=16,
        reduction="none",
    )
    expected_loss = (expected_rates * rate_weights).mean()
    expected_loss.backward()

    actual_rates = layer(generated_inputs)
    actual_loss = (actual_rates * rate_weights).mean()
    actual_loss.backward()

    assert torch.allclose(actual_rates, expected_rates.detach())
    assert inputs.grad is not None
    assert generated_inputs.grad is not None
    assert weight.grad is not None
    assert layer.synapse.weight.grad is not None
    assert bias.grad is not None
    assert layer.synapse.bias.grad is not None
    assert torch.allclose(generated_inputs.grad, inputs.grad, atol=1e-4, rtol=1e-3)
    assert torch.allclose(layer.synapse.weight.grad, weight.grad, atol=1e-4, rtol=1e-3)
    assert torch.allclose(layer.synapse.bias.grad, bias.grad, atol=1e-4, rtol=1e-3)


def test_linear_surrogate_lif_rate_module_auto_backend_runs_triton_on_cuda() -> None:
    from myelin.modules import LinearSurrogateLIFRate

    torch.manual_seed(142)
    inputs = torch.rand((6, 3, 13), device="cuda", requires_grad=True)
    layer = LinearSurrogateLIFRate(
        13,
        35,
        surrogate=fast_sigmoid_surrogate,
        surrogate_slope=5.0,
        backend="auto",
        checkpoint_size=3,
        reduction="none",
    ).to(device="cuda")

    rates = layer(inputs)
    loss = rates.square().mean()
    loss.backward()

    assert rates.shape == (3, 35)
    assert inputs.grad is not None
    assert layer.synapse.weight.grad is not None


def test_backend_lif_forward_torch_fallback_matches_reference_on_cpu() -> None:
    from myelin.kernels import lif_forward

    inputs = torch.rand((3, 2, 4))
    initial = LIFState(membrane=torch.zeros((2, 4)))
    params = LIFParams(tau_mem=8.0, threshold=0.7, reset=0.0)

    expected_state, expected_spikes = lif_unroll(inputs, initial, params)
    actual_state, actual_spikes = lif_forward(inputs, initial, params, backend="auto")

    assert torch.allclose(actual_state.membrane, expected_state.membrane)
    assert torch.equal(actual_spikes, expected_spikes)


def test_triton_surrogate_lif_recompute_matches_store_all() -> None:
    """The recompute backward must produce the same fwd + grads as the store-all path."""
    from myelin.autograd import triton_surrogate_lif_recompute_function
    from myelin.kernels import surrogate_lif_forward

    torch.manual_seed(7)
    params = LIFParams(tau_mem=2.0, threshold=1.0, reset=0.0)
    base_x = torch.randn((6, 3, 8), device="cuda")
    base_v = torch.zeros((3, 8), device="cuda")

    def run(use_recompute: bool):
        x = base_x.clone().requires_grad_(True)
        v = base_v.clone().requires_grad_(True)
        if use_recompute:
            _, spikes = triton_surrogate_lif_recompute_function(
                x, LIFState(membrane=v), params, surrogate="atan", surrogate_slope=2.0
            )
        else:
            _, spikes = surrogate_lif_forward(
                x,
                LIFState(membrane=v),
                params,
                surrogate="atan",
                surrogate_slope=2.0,
                backend="triton",
            )
        spikes.square().sum().backward()
        return spikes.detach(), x.grad, v.grad

    spikes_store, gx_store, gv_store = run(False)
    spikes_rec, gx_rec, gv_rec = run(True)
    assert torch.equal(spikes_store, spikes_rec)
    assert gx_store is not None and gx_rec is not None
    assert torch.allclose(gx_store, gx_rec, atol=1e-5, rtol=1e-4)
    assert torch.allclose(gv_store, gv_rec, atol=1e-5, rtol=1e-4)


def test_spiking_sequence_lif_fused_dispatch_matches_kernel() -> None:
    """SpikingSequenceLIF's fused path must route to the Triton kernel correctly.

    Validates the dispatch wiring (time-major movedim + param/surrogate mapping)
    against a direct kernel call. Kernel-vs-loop correctness is covered separately;
    bit-comparing the fused path to the loop is fragile because the hard reset
    cascades fp rounding differences at threshold crossings.
    """
    from myelin.kernels import surrogate_lif_forward
    from myelin.language import SpikingSequenceLIF

    torch.manual_seed(3)
    x = torch.randn((4, 20, 16), device="cuda")  # [B, T, C]
    params = LIFParams(tau_mem=2.0, threshold=1.0, reset=0.0)
    currents = x.movedim(1, 0).contiguous()
    initial = LIFState(membrane=currents.new_zeros(currents.shape[1:]))

    for recompute in (False, True):
        lif = SpikingSequenceLIF(
            tau=2.0, threshold=1.0, surrogate_slope=2.0, recompute=recompute
        ).cuda()
        spikes = lif(x)
        _, ref = surrogate_lif_forward(
            currents, initial, params, surrogate="atan", surrogate_slope=2.0, backend="triton"
        )
        assert torch.equal(spikes, ref.movedim(0, 1))
