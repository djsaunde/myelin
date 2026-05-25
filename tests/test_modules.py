from __future__ import annotations

import pytest
import torch

from spiker.baselines import compiled_available
from spiker.functional import alif_unroll, lif_unroll, surrogate_alif_unroll, surrogate_lif_unroll
from spiker.kernels import two_layer_surrogate_lif_rate_recompute_forward
from spiker.losses import SpikeRateLoss
from spiker.modules import (
    ALIFCell,
    CustomNeuronCell,
    CustomSurrogateNeuronCell,
    IzhikevichCell,
    LIFCell,
    LinearCustomSurrogateNeuron,
    LinearCustomSurrogateNeuronRate,
    LinearLIF,
    LinearSurrogateLIF,
    LinearSurrogateLIFPacked,
    LinearSurrogateLIFRate,
    LinearSynapse,
    RateReadoutClassifier,
    SurrogateALIFCell,
    SurrogateLIFCell,
    TimeUnroll,
    fast_sigmoid_surrogate,
    hard_surrogate_spike,
)
from spiker.neurons import (
    ALIFParams,
    ALIFState,
    IzhikevichParams,
    IzhikevichState,
    LIFParams,
    LIFState,
)


def build_test_lif_ir():
    from spiker.dsl import NeuronBuilder, where

    builder = NeuronBuilder("custom_lif")
    membrane = builder.state("membrane")
    current = builder.input("input_current")
    decay = builder.param("decay")
    threshold = builder.param("threshold")
    reset = builder.param("reset")
    pre_reset = membrane * decay + current
    did_spike = pre_reset.ge(threshold)
    return builder.build(
        next_state={"membrane": where(did_spike, reset, pre_reset)},
        outputs={"spike": where(did_spike, 1.0, 0.0)},
    )


def test_primitive_linear_synapse_and_lif_unroll_match_functional_reference() -> None:
    inputs = torch.rand((4, 2, 3))
    params = LIFParams(tau_mem=10.0, threshold=1.0, reset=0.0)
    synapse = LinearSynapse(3, 5)
    unroll = TimeUnroll(LIFCell(params))

    currents = synapse(inputs)
    state, spikes = unroll(currents)
    expected_state, expected_spikes = lif_unroll(
        currents,
        LIFState(membrane=torch.zeros((2, 5))),
        params,
    )

    assert torch.allclose(state.membrane, expected_state.membrane)
    assert torch.equal(spikes, expected_spikes)


def test_linear_lif_convenience_wrapper_matches_primitive_path() -> None:
    inputs = torch.rand((4, 2, 3))
    params = LIFParams(tau_mem=10.0, threshold=1.0, reset=0.0)
    primitive_synapse = LinearSynapse(3, 5)
    primitive = TimeUnroll(LIFCell(params))
    wrapper = LinearLIF(3, 5, params)
    wrapper.synapse.weight.data.copy_(primitive_synapse.weight)
    assert primitive_synapse.bias is not None
    assert wrapper.synapse.bias is not None
    wrapper.synapse.bias.data.copy_(primitive_synapse.bias)

    primitive_state, primitive_spikes = primitive(primitive_synapse(inputs))
    wrapper_state, wrapper_spikes = wrapper(inputs)

    assert torch.allclose(wrapper_state.membrane, primitive_state.membrane)
    assert torch.equal(wrapper_spikes, primitive_spikes)


def test_time_unroll_auto_backend_matches_reference_on_cpu() -> None:
    inputs = torch.rand((4, 2, 3))
    params = LIFParams(tau_mem=12.0, threshold=0.8, reset=-0.1)
    unroll = TimeUnroll(LIFCell(params), backend="auto")

    state, spikes = unroll(inputs)
    expected_state, expected_spikes = lif_unroll(
        inputs,
        LIFState(membrane=torch.zeros((2, 3))),
        params,
    )

    assert torch.allclose(state.membrane, expected_state.membrane)
    assert torch.equal(spikes, expected_spikes)


def test_time_unroll_alif_auto_backend_matches_reference_on_cpu() -> None:
    inputs = torch.rand((4, 2, 3))
    params = ALIFParams(
        tau_mem=12.0,
        tau_adaptation=30.0,
        threshold=0.8,
        reset=-0.1,
        beta=0.5,
    )
    unroll = TimeUnroll(ALIFCell(params), backend="auto")

    state, spikes = unroll(inputs)
    expected_state, expected_spikes = alif_unroll(
        inputs,
        ALIFState(
            membrane=torch.zeros((2, 3)),
            adaptation=torch.zeros((2, 3)),
        ),
        params,
    )

    assert isinstance(state, ALIFState)
    assert torch.allclose(state.membrane, expected_state.membrane)
    assert torch.allclose(state.adaptation, expected_state.adaptation)
    assert torch.equal(spikes, expected_spikes)


def test_time_unroll_izhikevich_auto_backend_matches_reference_on_cpu() -> None:
    from spiker.functional import izhikevich_unroll

    inputs = torch.rand((4, 2, 3)) * 15.0
    params = IzhikevichParams()
    initial = IzhikevichState(
        voltage=torch.full((2, 3), -65.0),
        recovery=torch.full((2, 3), -13.0),
    )
    unroll = TimeUnroll(IzhikevichCell(params), backend="auto")

    state, spikes = unroll(inputs, initial)
    expected_state, expected_spikes = izhikevich_unroll(inputs, initial, params)

    assert isinstance(state, IzhikevichState)
    assert torch.allclose(state.voltage, expected_state.voltage)
    assert torch.allclose(state.recovery, expected_state.recovery)
    assert torch.equal(spikes, expected_spikes)


def test_custom_neuron_cell_unroll_matches_dsl_reference_on_cpu() -> None:
    from spiker.dsl import NeuronBuilder, evaluate_neuron_unroll, where

    builder = NeuronBuilder("module_custom_lif")
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
    params = {"decay": 0.9, "threshold": 0.7, "reset": -0.1}
    inputs = torch.rand((4, 2, 3))
    initial_state = {"membrane": torch.rand((2, 3)) * 0.2}

    unroll = TimeUnroll(CustomNeuronCell(ir, params), backend="auto")
    actual_state, actual_spikes = unroll(inputs, initial_state)
    expected_state, expected_spikes = evaluate_neuron_unroll(
        ir,
        inputs,
        initial_state,
        params,
    )

    assert torch.allclose(actual_state["membrane"], expected_state["membrane"])
    assert torch.equal(actual_spikes, expected_spikes)


def test_custom_neuron_cell_infers_zero_initial_state() -> None:
    from spiker.dsl import NeuronBuilder, where

    builder = NeuronBuilder("module_custom_lif_default_state")
    builder.state("membrane")
    current = builder.input("input_current")
    threshold = builder.param("threshold")
    spike = current.ge(threshold)
    ir = builder.build(
        next_state={"membrane": where(spike, 0.0, current)},
        outputs={"spike": where(spike, 1.0, 0.0)},
    )
    inputs = torch.full((2, 3, 4), 2.0)

    state, spikes = TimeUnroll(CustomNeuronCell(ir, {"threshold": 1.0}))(inputs)

    assert torch.equal(state["membrane"], torch.zeros((3, 4)))
    assert torch.equal(spikes, torch.ones((2, 3, 4)))


def test_custom_neuron_cell_rejects_mismatched_params_and_state() -> None:
    from spiker.dsl import NeuronBuilder, where

    builder = NeuronBuilder("module_custom_lif_validation")
    builder.state("membrane")
    current = builder.input("input_current")
    threshold = builder.param("threshold")
    spike = current.ge(threshold)
    ir = builder.build(
        next_state={"membrane": where(spike, 0.0, current)},
        outputs={"spike": where(spike, 1.0, 0.0)},
    )

    with pytest.raises(ValueError, match="params is missing values"):
        CustomNeuronCell(ir)
    with pytest.raises(ValueError, match="params contains undeclared values"):
        CustomNeuronCell(ir, {"threshold": 1.0, "extra": 0.0})

    unroll = TimeUnroll(CustomNeuronCell(ir, {"threshold": 1.0}))
    with pytest.raises(ValueError, match="undeclared state tensors"):
        unroll(
            torch.rand((2, 3, 4)),
            {
                "membrane": torch.zeros((3, 4)),
                "extra": torch.zeros((3, 4)),
            },
        )


def test_custom_neuron_cell_rejects_ir_outside_module_abi() -> None:
    from spiker.dsl import NeuronIR, const, input_, state

    stateless = NeuronIR(
        name="module_stateless",
        state=(),
        inputs=("input_current",),
        params=(),
        next_state={},
        outputs={"spike": const(0.0)},
    )
    with pytest.raises(ValueError, match="at least one state"):
        CustomNeuronCell(stateless)

    wrong_input = NeuronIR(
        name="module_wrong_input",
        state=("membrane",),
        inputs=("current",),
        params=(),
        next_state={"membrane": input_("current")},
        outputs={"spike": state("membrane")},
    )
    with pytest.raises(ValueError, match="one input named input_current"):
        CustomNeuronCell(wrong_input)

    extra_output = NeuronIR(
        name="module_extra_output",
        state=("membrane",),
        inputs=("input_current",),
        params=(),
        next_state={"membrane": input_("input_current")},
        outputs={"spike": state("membrane"), "rate": state("membrane")},
    )
    with pytest.raises(ValueError, match="exactly one output named spike"):
        CustomNeuronCell(extra_output)


def test_linear_lif_auto_backend_matches_torch_backend_on_cpu() -> None:
    inputs = torch.rand((4, 2, 3))
    params = LIFParams(tau_mem=12.0, threshold=0.8, reset=-0.1)
    torch_layer = LinearLIF(3, 5, params, backend="torch")
    auto_layer = LinearLIF(3, 5, params, backend="auto")
    auto_layer.synapse.weight.data.copy_(torch_layer.synapse.weight)
    assert torch_layer.synapse.bias is not None
    assert auto_layer.synapse.bias is not None
    auto_layer.synapse.bias.data.copy_(torch_layer.synapse.bias)

    torch_state, torch_spikes = torch_layer(inputs)
    auto_state, auto_spikes = auto_layer(inputs)

    assert torch.allclose(auto_state.membrane, torch_state.membrane)
    assert torch.equal(auto_spikes, torch_spikes)


def test_linear_synapse_bias_can_be_disabled() -> None:
    synapse = LinearSynapse(3, 5, bias=False)

    assert synapse.bias is None


def test_surrogate_lif_cell_backpropagates() -> None:
    inputs = torch.rand((4, 2, 3))
    synapse = LinearSynapse(3, 5)
    unroll = TimeUnroll(SurrogateLIFCell(surrogate=fast_sigmoid_surrogate))

    _state, spikes = unroll(synapse(inputs))
    loss = spikes.mean()
    loss.backward()

    assert spikes.shape == (4, 2, 5)
    assert synapse.weight.grad is not None
    assert synapse.weight.grad.shape == synapse.weight.shape


def test_surrogate_lif_cell_matches_functional_reference_with_reset() -> None:
    inputs = torch.rand((4, 2, 3))
    params = LIFParams(tau_mem=8.0, threshold=0.7, reset=-0.2)
    cell = SurrogateLIFCell(
        params,
        surrogate=fast_sigmoid_surrogate,
        surrogate_slope=5.0,
    )
    unroll = TimeUnroll(cell)
    initial = LIFState(membrane=torch.zeros((2, 3)))

    state, spikes = unroll(inputs, initial)
    expected_state, expected_spikes = surrogate_lif_unroll(
        inputs,
        initial,
        params,
        surrogate=fast_sigmoid_surrogate,
        surrogate_slope=5.0,
        hard_forward=True,
    )

    assert torch.allclose(state.membrane, expected_state.membrane)
    assert torch.equal(spikes.detach(), expected_spikes.detach())


def test_surrogate_alif_cell_matches_functional_reference_and_backpropagates() -> None:
    inputs = torch.rand((4, 2, 3), requires_grad=True)
    reference_inputs = inputs.detach().clone().requires_grad_(True)
    params = ALIFParams(tau_mem=10.0, tau_adaptation=5.0, threshold=0.8, reset=-0.1, beta=0.4)
    initial = ALIFState(
        membrane=torch.zeros((2, 3)),
        adaptation=torch.zeros((2, 3)),
    )
    reference_initial = ALIFState(
        membrane=initial.membrane.detach().clone(),
        adaptation=initial.adaptation.detach().clone(),
    )
    unroll = TimeUnroll(
        SurrogateALIFCell(params, surrogate=fast_sigmoid_surrogate, surrogate_slope=5.0),
        backend="torch",
    )

    state, spikes = unroll(inputs, initial)
    expected_state, expected_spikes = surrogate_alif_unroll(
        reference_inputs,
        reference_initial,
        params,
        surrogate=fast_sigmoid_surrogate,
        surrogate_slope=5.0,
        hard_forward=True,
    )
    loss = spikes.mean() + state.membrane.mean() + state.adaptation.mean()
    expected_loss = (
        expected_spikes.mean() + expected_state.membrane.mean() + expected_state.adaptation.mean()
    )
    loss.backward()
    expected_loss.backward()

    assert torch.allclose(state.membrane, expected_state.membrane)
    assert torch.allclose(state.adaptation, expected_state.adaptation)
    assert torch.equal(spikes.detach(), expected_spikes.detach())
    assert inputs.grad is not None
    assert reference_inputs.grad is not None
    assert torch.allclose(inputs.grad, reference_inputs.grad)
    assert torch.isfinite(inputs.grad).all()


def test_custom_surrogate_neuron_cell_matches_lif_surrogate_path() -> None:
    torch.manual_seed(9)
    params = LIFParams(tau_mem=8.0, threshold=0.7, reset=-0.2)
    inputs = torch.rand((5, 2, 3), requires_grad=True)
    reference_inputs = inputs.detach().clone().requires_grad_()
    initial = {"membrane": torch.zeros((2, 3))}
    reference_initial = LIFState(membrane=initial["membrane"].detach().clone())
    ir = build_test_lif_ir()

    custom_unroll = TimeUnroll(
        CustomSurrogateNeuronCell(
            ir,
            {"decay": params.decay, "threshold": params.threshold, "reset": params.reset},
            surrogate=fast_sigmoid_surrogate,
            surrogate_slope=5.0,
        ),
        backend="torch",
    )
    reference_unroll = TimeUnroll(
        SurrogateLIFCell(params, surrogate=fast_sigmoid_surrogate, surrogate_slope=5.0),
        backend="torch",
    )

    custom_state, custom_spikes = custom_unroll(inputs, initial)
    reference_state, reference_spikes = reference_unroll(reference_inputs, reference_initial)
    custom_loss = custom_spikes.mean()
    reference_loss = reference_spikes.mean()
    custom_loss.backward()
    reference_loss.backward()

    assert isinstance(custom_state, dict)
    assert isinstance(reference_state, LIFState)
    assert torch.allclose(custom_state["membrane"], reference_state.membrane)
    assert torch.equal(custom_spikes.detach(), reference_spikes.detach())
    assert inputs.grad is not None
    assert reference_inputs.grad is not None
    assert torch.allclose(inputs.grad, reference_inputs.grad)


def test_custom_surrogate_neuron_cell_rejects_unsupported_backward_ir() -> None:
    from spiker.dsl import alif_ir

    with pytest.raises(ValueError, match="ALIF generated backward plan is recognized"):
        CustomSurrogateNeuronCell(
            alif_ir(),
            {
                "decay": 0.9,
                "adaptation_decay": 0.95,
                "threshold": 1.0,
                "reset": 0.0,
                "beta": 0.5,
            },
        )


def test_hard_surrogate_spike_is_binary_and_backpropagates() -> None:
    centered = torch.tensor([-2.0, -0.1, 0.0, 1.0], requires_grad=True)

    spikes = hard_surrogate_spike(centered, fast_sigmoid_surrogate)
    spikes.sum().backward()

    assert torch.equal(spikes.detach(), torch.tensor([0.0, 0.0, 1.0, 1.0]))
    assert centered.grad is not None
    assert torch.all(centered.grad > 0)


def test_surrogate_lif_cell_defaults_to_hard_forward_spikes() -> None:
    inputs = torch.full((2, 3), 2.0)
    state = LIFState(membrane=torch.zeros((2, 3)))
    cell = SurrogateLIFCell()

    _state, spikes = cell(state, inputs)

    assert torch.equal(spikes.detach(), torch.ones_like(spikes))


def test_surrogate_lif_cell_can_use_smooth_forward_spikes() -> None:
    inputs = torch.zeros((2, 3))
    state = LIFState(membrane=torch.zeros((2, 3)))
    cell = SurrogateLIFCell(hard_forward=False)

    _state, spikes = cell(state, inputs)

    assert torch.all(spikes > 0)
    assert torch.all(spikes < 1)


def test_linear_surrogate_lif_convenience_wrapper_backpropagates() -> None:
    inputs = torch.rand((4, 2, 3))
    layer = LinearSurrogateLIF(3, 5, surrogate=fast_sigmoid_surrogate)

    loss = layer(inputs).mean()
    loss.backward()

    assert layer.synapse.weight.grad is not None
    assert layer.synapse.weight.grad.shape == layer.synapse.weight.shape


def test_linear_custom_surrogate_neuron_matches_linear_surrogate_lif() -> None:
    torch.manual_seed(19)
    inputs = torch.rand((5, 3, 4), requires_grad=True)
    reference_inputs = inputs.detach().clone().requires_grad_()
    params = LIFParams(tau_mem=8.0, threshold=0.7, reset=-0.2)
    custom_params = {"decay": params.decay, "threshold": params.threshold, "reset": params.reset}
    reference = LinearSurrogateLIF(
        4,
        6,
        params,
        surrogate=fast_sigmoid_surrogate,
        surrogate_slope=5.0,
    )
    custom = LinearCustomSurrogateNeuron(
        4,
        6,
        build_test_lif_ir(),
        custom_params,
        surrogate=fast_sigmoid_surrogate,
        surrogate_slope=5.0,
    )
    custom.synapse.weight.data.copy_(reference.synapse.weight)
    assert reference.synapse.bias is not None
    assert custom.synapse.bias is not None
    custom.synapse.bias.data.copy_(reference.synapse.bias)

    expected_spikes = reference(reference_inputs)
    expected_loss = expected_spikes.square().mean()
    expected_loss.backward()
    actual_spikes = custom(inputs)
    actual_loss = actual_spikes.square().mean()
    actual_loss.backward()

    assert torch.equal(actual_spikes.detach(), expected_spikes.detach())
    assert inputs.grad is not None
    assert reference_inputs.grad is not None
    assert custom.synapse.weight.grad is not None
    assert reference.synapse.weight.grad is not None
    assert custom.synapse.bias.grad is not None
    assert reference.synapse.bias.grad is not None
    assert torch.allclose(inputs.grad, reference_inputs.grad)
    assert torch.allclose(custom.synapse.weight.grad, reference.synapse.weight.grad)
    assert torch.allclose(custom.synapse.bias.grad, reference.synapse.bias.grad)


def test_linear_custom_surrogate_neuron_stream_synapse_matches_materialized() -> None:
    torch.manual_seed(20)
    inputs = torch.rand((6, 3, 4), requires_grad=True)
    streamed_inputs = inputs.detach().clone().requires_grad_()
    params = LIFParams(tau_mem=8.0, threshold=0.7, reset=-0.2)
    custom_params = {"decay": params.decay, "threshold": params.threshold, "reset": params.reset}
    materialized = LinearCustomSurrogateNeuron(
        4,
        6,
        build_test_lif_ir(),
        custom_params,
        surrogate=fast_sigmoid_surrogate,
        surrogate_slope=5.0,
        stream_synapse=False,
    )
    streamed = LinearCustomSurrogateNeuron(
        4,
        6,
        build_test_lif_ir(),
        custom_params,
        surrogate=fast_sigmoid_surrogate,
        surrogate_slope=5.0,
        stream_synapse=True,
        checkpoint_size=3,
    )
    streamed.synapse.weight.data.copy_(materialized.synapse.weight)
    assert materialized.synapse.bias is not None
    assert streamed.synapse.bias is not None
    streamed.synapse.bias.data.copy_(materialized.synapse.bias)

    materialized_spikes = materialized(inputs)
    materialized_loss = materialized_spikes.mean() + 0.1 * materialized_spikes.square().mean()
    materialized_loss.backward()
    streamed_spikes = streamed(streamed_inputs)
    streamed_loss = streamed_spikes.mean() + 0.1 * streamed_spikes.square().mean()
    streamed_loss.backward()

    assert torch.equal(streamed_spikes.detach(), materialized_spikes.detach())
    assert inputs.grad is not None
    assert streamed_inputs.grad is not None
    assert materialized.synapse.weight.grad is not None
    assert streamed.synapse.weight.grad is not None
    assert materialized.synapse.bias.grad is not None
    assert streamed.synapse.bias.grad is not None
    assert torch.allclose(streamed_inputs.grad, inputs.grad)
    assert torch.allclose(streamed.synapse.weight.grad, materialized.synapse.weight.grad)
    assert torch.allclose(streamed.synapse.bias.grad, materialized.synapse.bias.grad)


def test_linear_custom_surrogate_neuron_rate_matches_lif_rate() -> None:
    torch.manual_seed(21)
    inputs = torch.rand((6, 3, 4), requires_grad=True)
    reference_inputs = inputs.detach().clone().requires_grad_()
    params = LIFParams(tau_mem=8.0, threshold=0.7, reset=-0.2)
    custom_params = {"decay": params.decay, "threshold": params.threshold, "reset": params.reset}
    reference = LinearSurrogateLIFRate(
        4,
        6,
        params,
        surrogate=fast_sigmoid_surrogate,
        surrogate_slope=5.0,
        checkpoint_size=3,
        reduction="none",
    )
    custom = LinearCustomSurrogateNeuronRate(
        4,
        6,
        build_test_lif_ir(),
        custom_params,
        surrogate=fast_sigmoid_surrogate,
        surrogate_slope=5.0,
        checkpoint_size=3,
        reduction="none",
    )
    custom.synapse.weight.data.copy_(reference.synapse.weight)
    assert reference.synapse.bias is not None
    assert custom.synapse.bias is not None
    custom.synapse.bias.data.copy_(reference.synapse.bias)

    expected_rates = reference(reference_inputs)
    expected_loss = expected_rates.square().mean()
    expected_loss.backward()
    actual_rates = custom(inputs)
    actual_loss = actual_rates.square().mean()
    actual_loss.backward()

    assert torch.equal(actual_rates.detach(), expected_rates.detach())
    assert inputs.grad is not None
    assert reference_inputs.grad is not None
    assert custom.synapse.weight.grad is not None
    assert reference.synapse.weight.grad is not None
    assert custom.synapse.bias.grad is not None
    assert reference.synapse.bias.grad is not None
    assert torch.allclose(inputs.grad, reference_inputs.grad)
    assert torch.allclose(custom.synapse.weight.grad, reference.synapse.weight.grad)
    assert torch.allclose(custom.synapse.bias.grad, reference.synapse.bias.grad)


def test_two_layer_surrogate_lif_rate_recompute_matches_composed_rate_model() -> None:
    torch.manual_seed(22)
    inputs = torch.rand((6, 3, 4), requires_grad=True)
    reference_inputs = inputs.detach().clone().requires_grad_()
    params = LIFParams(tau_mem=8.0, threshold=0.7, reset=-0.2)
    hidden = LinearSurrogateLIF(
        4,
        5,
        params,
        surrogate=fast_sigmoid_surrogate,
        surrogate_slope=5.0,
        stream_synapse=True,
        checkpoint_size=3,
    )
    output = LinearSurrogateLIFRate(
        5,
        2,
        params,
        surrogate=fast_sigmoid_surrogate,
        surrogate_slope=5.0,
        checkpoint_size=3,
        reduction="none",
    )
    assert hidden.synapse.bias is not None
    assert output.synapse.bias is not None
    recompute_hidden_weight = hidden.synapse.weight.detach().clone().requires_grad_()
    recompute_hidden_bias = hidden.synapse.bias.detach().clone().requires_grad_()
    recompute_output_weight = output.synapse.weight.detach().clone().requires_grad_()
    recompute_output_bias = output.synapse.bias.detach().clone().requires_grad_()

    expected_hidden_spikes = hidden(reference_inputs)
    expected_rates = output(expected_hidden_spikes)
    expected_loss = expected_rates.square().mean()
    expected_loss.backward()

    actual_rates = two_layer_surrogate_lif_rate_recompute_forward(
        inputs,
        recompute_hidden_weight,
        recompute_hidden_bias,
        recompute_output_weight,
        recompute_output_bias,
        params,
        surrogate="fast_sigmoid",
        surrogate_slope=5.0,
        checkpoint_size=3,
    )
    actual_loss = actual_rates.square().mean()
    actual_loss.backward()

    assert torch.equal(actual_rates.detach(), expected_rates.detach())
    assert inputs.grad is not None
    assert reference_inputs.grad is not None
    assert hidden.synapse.weight.grad is not None
    assert hidden.synapse.bias.grad is not None
    assert output.synapse.weight.grad is not None
    assert output.synapse.bias.grad is not None
    assert recompute_hidden_weight.grad is not None
    assert recompute_hidden_bias.grad is not None
    assert recompute_output_weight.grad is not None
    assert recompute_output_bias.grad is not None
    assert torch.allclose(inputs.grad, reference_inputs.grad)
    assert torch.allclose(recompute_hidden_weight.grad, hidden.synapse.weight.grad)
    assert torch.allclose(recompute_hidden_bias.grad, hidden.synapse.bias.grad)
    assert torch.allclose(recompute_output_weight.grad, output.synapse.weight.grad)
    assert torch.allclose(recompute_output_bias.grad, output.synapse.bias.grad)


def test_linear_surrogate_lif_stream_synapse_matches_materialized_path() -> None:
    torch.manual_seed(10)
    inputs = torch.rand((5, 3, 4), requires_grad=True)
    materialized_inputs = inputs.detach().clone().requires_grad_()
    params = LIFParams(tau_mem=8.0, threshold=0.7, reset=-0.2)
    materialized = LinearSurrogateLIF(
        4,
        6,
        params,
        surrogate=fast_sigmoid_surrogate,
        surrogate_slope=5.0,
        stream_synapse=False,
    )
    streamed = LinearSurrogateLIF(
        4,
        6,
        params,
        surrogate=fast_sigmoid_surrogate,
        surrogate_slope=5.0,
        stream_synapse=True,
    )
    streamed.synapse.weight.data.copy_(materialized.synapse.weight)
    assert materialized.synapse.bias is not None
    assert streamed.synapse.bias is not None
    streamed.synapse.bias.data.copy_(materialized.synapse.bias)

    materialized_spikes = materialized(materialized_inputs)
    materialized_loss = materialized_spikes.square().mean()
    materialized_loss.backward()
    streamed_spikes = streamed(inputs)
    streamed_loss = streamed_spikes.square().mean()
    streamed_loss.backward()

    assert torch.allclose(streamed_spikes, materialized_spikes.detach())
    assert inputs.grad is not None
    assert materialized_inputs.grad is not None
    assert materialized.synapse.weight.grad is not None
    assert streamed.synapse.weight.grad is not None
    assert materialized.synapse.bias.grad is not None
    assert streamed.synapse.bias.grad is not None
    assert torch.allclose(inputs.grad, materialized_inputs.grad)
    assert torch.allclose(streamed.synapse.weight.grad, materialized.synapse.weight.grad)
    assert torch.allclose(streamed.synapse.bias.grad, materialized.synapse.bias.grad)


def test_linear_surrogate_lif_stream_synapse_uses_per_timestep_matmuls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = torch.rand((4, 2, 3))
    params = LIFParams(tau_mem=8.0, threshold=0.7, reset=-0.2)
    materialized = LinearSurrogateLIF(3, 5, params, stream_synapse=False)
    streamed = LinearSurrogateLIF(3, 5, params, stream_synapse=True)
    streamed.synapse.weight.data.copy_(materialized.synapse.weight)
    assert materialized.synapse.bias is not None
    assert streamed.synapse.bias is not None
    streamed.synapse.bias.data.copy_(materialized.synapse.bias)

    original_matmul = torch.matmul
    matmul_shapes: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def recording_matmul(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        matmul_shapes.append((tuple(left.shape), tuple(right.shape)))
        return original_matmul(left, right)

    monkeypatch.setattr(torch, "matmul", recording_matmul)

    materialized(inputs)
    materialized_shapes = list(matmul_shapes)
    matmul_shapes.clear()

    streamed(inputs)
    streamed_shapes = list(matmul_shapes)

    assert materialized_shapes == [((4, 2, 3), (3, 5))]
    assert streamed_shapes == [((2, 3), (3, 5))] * inputs.shape[0]


def test_linear_surrogate_lif_forward_dispatcher_matches_streamed_module() -> None:
    from spiker.kernels import linear_surrogate_lif_forward

    torch.manual_seed(12)
    inputs = torch.rand((5, 3, 4), requires_grad=True)
    module_inputs = inputs.detach().clone().requires_grad_()
    params = LIFParams(tau_mem=8.0, threshold=0.7, reset=-0.2)
    layer = LinearSurrogateLIF(
        4,
        6,
        params,
        surrogate=fast_sigmoid_surrogate,
        surrogate_slope=5.0,
        stream_synapse=True,
    )

    expected_spikes = layer(module_inputs)
    expected_loss = expected_spikes.square().mean()
    expected_loss.backward()

    state, actual_spikes = linear_surrogate_lif_forward(
        inputs,
        layer.synapse.weight.detach().clone().requires_grad_(),
        None
        if layer.synapse.bias is None
        else layer.synapse.bias.detach().clone().requires_grad_(),
        params,
        surrogate="fast_sigmoid",
        surrogate_slope=5.0,
        backend="torch",
    )
    actual_loss = actual_spikes.square().mean() + 0.01 * state.membrane.square().mean()
    actual_loss.backward()

    assert torch.equal(actual_spikes.detach(), expected_spikes.detach())
    assert state.membrane.shape == (3, 6)


def test_linear_surrogate_lif_packed_forward_torch_matches_dense_dispatcher() -> None:
    from spiker.kernels import linear_surrogate_lif_forward, linear_surrogate_lif_packed_forward
    from spiker.packing import unpack_spikes

    torch.manual_seed(121)
    inputs = torch.rand((5, 3, 4))
    weight = (torch.rand((4, 6)) - 0.5) * 0.02
    bias = (torch.rand((6,)) - 0.5) * 0.01
    params = LIFParams(tau_mem=8.0, threshold=0.7, reset=-0.2)

    expected_state, expected_spikes = linear_surrogate_lif_forward(
        inputs,
        weight,
        bias,
        params,
        surrogate="fast_sigmoid",
        surrogate_slope=5.0,
        backend="torch",
        checkpoint_size=3,
    )
    actual_state, packed_spikes = linear_surrogate_lif_packed_forward(
        inputs,
        weight,
        bias,
        params,
        surrogate="fast_sigmoid",
        surrogate_slope=5.0,
        backend="torch",
        checkpoint_size=3,
    )

    assert packed_spikes.original_shape == tuple(expected_spikes.shape)
    assert torch.equal(unpack_spikes(packed_spikes, dtype=inputs.dtype), expected_spikes)
    assert torch.allclose(actual_state.membrane, expected_state.membrane)


def test_linear_surrogate_lif_checkpoint_synapse_matches_streamed_path() -> None:
    torch.manual_seed(11)
    inputs = torch.rand((7, 3, 4), requires_grad=True)
    checkpointed_inputs = inputs.detach().clone().requires_grad_()
    params = LIFParams(tau_mem=8.0, threshold=0.7, reset=-0.2)
    streamed = LinearSurrogateLIF(
        4,
        6,
        params,
        surrogate=fast_sigmoid_surrogate,
        surrogate_slope=5.0,
        stream_synapse=True,
    )
    checkpointed = LinearSurrogateLIF(
        4,
        6,
        params,
        surrogate=fast_sigmoid_surrogate,
        surrogate_slope=5.0,
        stream_synapse=True,
        checkpoint_size=3,
    )
    checkpointed.synapse.weight.data.copy_(streamed.synapse.weight)
    assert streamed.synapse.bias is not None
    assert checkpointed.synapse.bias is not None
    checkpointed.synapse.bias.data.copy_(streamed.synapse.bias)

    streamed_spikes = streamed(inputs)
    streamed_loss = streamed_spikes.mean() + 0.1 * streamed_spikes.square().mean()
    streamed_loss.backward()
    checkpointed_spikes = checkpointed(checkpointed_inputs)
    checkpointed_loss = checkpointed_spikes.mean() + 0.1 * checkpointed_spikes.square().mean()
    checkpointed_loss.backward()

    assert torch.equal(checkpointed_spikes.detach(), streamed_spikes.detach())
    assert inputs.grad is not None
    assert checkpointed_inputs.grad is not None
    assert streamed.synapse.weight.grad is not None
    assert checkpointed.synapse.weight.grad is not None
    assert streamed.synapse.bias.grad is not None
    assert checkpointed.synapse.bias.grad is not None
    assert torch.allclose(checkpointed_inputs.grad, inputs.grad)
    assert torch.allclose(checkpointed.synapse.weight.grad, streamed.synapse.weight.grad)
    assert torch.allclose(checkpointed.synapse.bias.grad, streamed.synapse.bias.grad)


def test_linear_surrogate_lif_stream_synapse_rejects_smooth_forward() -> None:
    layer = LinearSurrogateLIF(3, 5, hard_forward=False, stream_synapse=True)

    with pytest.raises(ValueError, match="hard_forward=True"):
        layer(torch.rand((4, 2, 3)))


def test_linear_surrogate_lif_checkpoint_synapse_rejects_invalid_checkpoint_size() -> None:
    layer = LinearSurrogateLIF(3, 5, stream_synapse=True, checkpoint_size=0)

    with pytest.raises(ValueError, match="checkpoint_size"):
        layer(torch.rand((4, 2, 3)))


def test_linear_surrogate_lif_checkpoint_synapse_accepts_policy_size() -> None:
    torch.manual_seed(14)
    inputs = torch.rand((8, 2, 3))
    policy_layer = LinearSurrogateLIF(
        3,
        5,
        surrogate=fast_sigmoid_surrogate,
        stream_synapse=True,
        checkpoint_size="balanced",
    )
    explicit_layer = LinearSurrogateLIF(
        3,
        5,
        surrogate=fast_sigmoid_surrogate,
        stream_synapse=True,
        checkpoint_size=2,
    )
    explicit_layer.synapse.weight.data.copy_(policy_layer.synapse.weight)
    assert policy_layer.synapse.bias is not None
    assert explicit_layer.synapse.bias is not None
    explicit_layer.synapse.bias.data.copy_(policy_layer.synapse.bias)

    assert torch.equal(policy_layer(inputs), explicit_layer(inputs))


def test_linear_surrogate_lif_rate_module_matches_dense_spike_rates() -> None:
    torch.manual_seed(15)
    inputs = torch.rand((5, 3, 4), requires_grad=True)
    ref_inputs = inputs.detach().clone().requires_grad_()
    params = LIFParams(tau_mem=8.0, threshold=0.7, reset=-0.2)
    dense = LinearSurrogateLIF(
        4,
        6,
        params,
        surrogate=fast_sigmoid_surrogate,
        surrogate_slope=5.0,
        stream_synapse=True,
        checkpoint_size=3,
    )
    rate = LinearSurrogateLIFRate(
        4,
        6,
        params,
        surrogate=fast_sigmoid_surrogate,
        surrogate_slope=5.0,
        backend="torch",
        checkpoint_size=3,
        reduction="none",
    )
    rate.synapse.weight.data.copy_(dense.synapse.weight)
    assert dense.synapse.bias is not None
    assert rate.synapse.bias is not None
    rate.synapse.bias.data.copy_(dense.synapse.bias)

    expected_spikes = dense(ref_inputs)
    expected_rates = expected_spikes.mean(dim=0)
    expected_loss = expected_rates.square().mean()
    expected_loss.backward()

    actual_rates = rate(inputs)
    actual_loss = actual_rates.square().mean()
    actual_loss.backward()

    assert torch.equal(actual_rates.detach(), expected_rates.detach())
    assert inputs.grad is not None
    assert ref_inputs.grad is not None
    assert torch.allclose(inputs.grad, ref_inputs.grad)
    assert dense.synapse.weight.grad is not None
    assert rate.synapse.weight.grad is not None
    assert torch.allclose(rate.synapse.weight.grad, dense.synapse.weight.grad)


def test_linear_surrogate_lif_packed_module_matches_dense_spikes() -> None:
    from spiker.packing import unpack_spikes

    torch.manual_seed(151)
    inputs = torch.rand((5, 3, 4))
    params = LIFParams(tau_mem=8.0, threshold=0.7, reset=-0.2)
    dense = LinearSurrogateLIF(
        4,
        6,
        params,
        surrogate=fast_sigmoid_surrogate,
        surrogate_slope=5.0,
        stream_synapse=True,
        checkpoint_size=3,
    )
    packed = LinearSurrogateLIFPacked(
        4,
        6,
        params,
        surrogate=fast_sigmoid_surrogate,
        surrogate_slope=5.0,
        backend="torch",
        checkpoint_size=3,
    )
    packed.synapse.weight.data.copy_(dense.synapse.weight)
    assert dense.synapse.bias is not None
    assert packed.synapse.bias is not None
    packed.synapse.bias.data.copy_(dense.synapse.bias)

    expected_spikes = dense(inputs)
    packed_spikes = packed(inputs)

    assert packed_spikes.original_shape == tuple(expected_spikes.shape)
    assert torch.equal(unpack_spikes(packed_spikes, dtype=inputs.dtype), expected_spikes)


def test_linear_surrogate_lif_rate_module_can_return_scalar_mean() -> None:
    torch.manual_seed(16)
    inputs = torch.rand((5, 3, 4))
    params = LIFParams(tau_mem=8.0, threshold=0.7, reset=-0.2)
    dense = LinearSurrogateLIF(
        4,
        6,
        params,
        surrogate=fast_sigmoid_surrogate,
        surrogate_slope=5.0,
        stream_synapse=True,
        checkpoint_size=3,
    )
    rate = LinearSurrogateLIFRate(
        4,
        6,
        params,
        surrogate=fast_sigmoid_surrogate,
        surrogate_slope=5.0,
        backend="torch",
        checkpoint_size=3,
        reduction="mean",
    )
    rate.synapse.weight.data.copy_(dense.synapse.weight)
    assert dense.synapse.bias is not None
    assert rate.synapse.bias is not None
    rate.synapse.bias.data.copy_(dense.synapse.bias)

    assert torch.equal(rate(inputs), dense(inputs).mean())


def test_rate_readout_classifier_returns_class_logits() -> None:
    torch.manual_seed(23)
    inputs = torch.rand((5, 3, 4))
    model = RateReadoutClassifier(
        4,
        6,
        2,
        surrogate=fast_sigmoid_surrogate,
        surrogate_slope=5.0,
        backend="torch",
        checkpoint_size=3,
        dropout=0.1,
    )

    logits = model(inputs)

    assert logits.shape == (3, 2)


def test_rate_readout_classifier_training_reduces_tiny_loss() -> None:
    torch.manual_seed(24)
    inputs = torch.rand((8, 4, 3))
    targets = torch.tensor([0, 1, 0, 1])
    model = RateReadoutClassifier(
        3,
        8,
        2,
        surrogate=fast_sigmoid_surrogate,
        surrogate_slope=5.0,
        backend="torch",
        checkpoint_size=4,
    )
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    initial_loss = loss_fn(model(inputs), targets).detach()
    for _ in range(20):
        optimizer.zero_grad()
        loss = loss_fn(model(inputs), targets)
        loss.backward()
        optimizer.step()
    final_loss = loss_fn(model(inputs), targets).detach()

    assert final_loss < initial_loss


def test_linear_surrogate_lif_training_step_updates_weight() -> None:
    inputs = torch.rand((4, 2, 3))
    layer = LinearSurrogateLIF(3, 5, surrogate=fast_sigmoid_surrogate)
    loss_fn = SpikeRateLoss(target_rate=0.2)
    optimizer = torch.optim.SGD(layer.parameters(), lr=0.1)
    before = layer.synapse.weight.detach().clone()

    optimizer.zero_grad()
    loss = loss_fn(layer(inputs))
    loss.backward()
    optimizer.step()

    assert loss.ndim == 0
    assert not torch.equal(layer.synapse.weight, before)


def test_linear_surrogate_lif_training_reduces_spike_rate_loss() -> None:
    torch.manual_seed(0)
    inputs = torch.rand((8, 4, 3))
    layer = LinearSurrogateLIF(3, 6, surrogate=fast_sigmoid_surrogate)
    loss_fn = SpikeRateLoss(target_rate=0.2)
    optimizer = torch.optim.SGD(layer.parameters(), lr=1.0)

    initial_loss = loss_fn(layer(inputs)).detach()
    for _ in range(25):
        optimizer.zero_grad()
        loss = loss_fn(layer(inputs))
        loss.backward()
        optimizer.step()
    final_loss = loss_fn(layer(inputs)).detach()

    assert final_loss < initial_loss


@pytest.mark.extended
def test_linear_surrogate_lif_compiles() -> None:
    if not compiled_available():
        pytest.skip("torch.compile is not available in this PyTorch build")

    layer = LinearSurrogateLIF(3, 5)
    inputs = torch.rand((4, 2, 3))

    try:
        compiled_layer = torch.compile(layer, mode="reduce-overhead", fullgraph=True)
        spikes = compiled_layer(inputs)
    except Exception as exc:
        pytest.skip(
            f"torch.compile could not compile SurrogateDenseLIF: {type(exc).__name__}: {exc}"
        )

    assert spikes.shape == (4, 2, 5)


@pytest.mark.extended
def test_time_unroll_auto_backend_compiles() -> None:
    if not compiled_available():
        pytest.skip("torch.compile is not available in this PyTorch build")

    params = LIFParams(tau_mem=12.0, threshold=0.8, reset=-0.1)
    unroll = TimeUnroll(LIFCell(params), backend="auto")
    inputs = torch.rand((4, 2, 3))

    try:
        compiled_unroll = torch.compile(unroll, mode="reduce-overhead", fullgraph=True)
        state, spikes = compiled_unroll(inputs)
    except Exception as exc:
        pytest.skip(
            f"torch.compile could not compile TimeUnroll auto backend: {type(exc).__name__}: {exc}"
        )

    expected_state, expected_spikes = lif_unroll(
        inputs,
        LIFState(membrane=torch.zeros((2, 3))),
        params,
    )
    assert torch.allclose(state.membrane, expected_state.membrane)
    assert torch.equal(spikes, expected_spikes)


@pytest.mark.extended
def test_linear_surrogate_lif_stream_auto_backend_compiles() -> None:
    if not compiled_available():
        pytest.skip("torch.compile is not available in this PyTorch build")

    layer = LinearSurrogateLIF(
        3,
        5,
        surrogate=fast_sigmoid_surrogate,
        backend="auto",
        stream_synapse=True,
        checkpoint_size=2,
    )
    inputs = torch.rand((4, 2, 3))

    try:
        compiled_layer = torch.compile(layer, mode="reduce-overhead", fullgraph=True)
        spikes = compiled_layer(inputs)
    except Exception as exc:
        pytest.skip(
            f"torch.compile could not compile LinearSurrogateLIF stream auto backend: "
            f"{type(exc).__name__}: {exc}"
        )

    assert spikes.shape == (4, 2, 5)


@pytest.mark.extended
def test_linear_surrogate_lif_rate_auto_backend_compiles() -> None:
    if not compiled_available():
        pytest.skip("torch.compile is not available in this PyTorch build")

    torch.manual_seed(21)
    params = LIFParams(tau_mem=8.0, threshold=0.4, reset=-0.2)
    eager_layer = LinearSurrogateLIFRate(
        4,
        6,
        params,
        surrogate=fast_sigmoid_surrogate,
        surrogate_slope=5.0,
        backend="auto",
        checkpoint_size=3,
        reduction="none",
    )
    compiled_layer = LinearSurrogateLIFRate(
        4,
        6,
        params,
        surrogate=fast_sigmoid_surrogate,
        surrogate_slope=5.0,
        backend="auto",
        checkpoint_size=3,
        reduction="none",
    )
    compiled_layer.synapse.weight.data.copy_(eager_layer.synapse.weight)
    assert eager_layer.synapse.bias is not None
    assert compiled_layer.synapse.bias is not None
    compiled_layer.synapse.bias.data.copy_(eager_layer.synapse.bias)

    inputs = (torch.rand((5, 3, 4)) * 1.5).requires_grad_()
    compiled_inputs = inputs.detach().clone().requires_grad_()

    eager_rates = eager_layer(inputs)
    eager_loss = eager_rates.square().mean()
    eager_loss.backward()

    try:
        compiled = torch.compile(compiled_layer, mode="reduce-overhead", fullgraph=True)
        compiled_rates = compiled(compiled_inputs)
        compiled_loss = compiled_rates.square().mean()
        compiled_loss.backward()
    except Exception as exc:
        pytest.skip(
            f"torch.compile could not compile LinearSurrogateLIFRate auto backend: "
            f"{type(exc).__name__}: {exc}"
        )

    assert torch.equal(compiled_rates.detach(), eager_rates.detach())
    assert inputs.grad is not None
    assert compiled_inputs.grad is not None
    assert eager_layer.synapse.weight.grad is not None
    assert compiled_layer.synapse.weight.grad is not None
    assert eager_layer.synapse.bias.grad is not None
    assert compiled_layer.synapse.bias.grad is not None
    assert torch.allclose(compiled_inputs.grad, inputs.grad)
    assert torch.allclose(compiled_layer.synapse.weight.grad, eager_layer.synapse.weight.grad)
    assert torch.allclose(compiled_layer.synapse.bias.grad, eager_layer.synapse.bias.grad)


@pytest.mark.extended
def test_rate_readout_classifier_compiles() -> None:
    if not compiled_available():
        pytest.skip("torch.compile is not available in this PyTorch build")

    model = RateReadoutClassifier(
        4,
        6,
        2,
        surrogate=fast_sigmoid_surrogate,
        surrogate_slope=5.0,
        backend="torch",
        checkpoint_size=3,
    )
    inputs = torch.rand((5, 3, 4))

    try:
        compiled_model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
        logits = compiled_model(inputs)
    except Exception as exc:
        pytest.skip(
            f"torch.compile could not compile RateReadoutClassifier: {type(exc).__name__}: {exc}"
        )

    assert logits.shape == (3, 2)
