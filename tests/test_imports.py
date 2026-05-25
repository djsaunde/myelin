from __future__ import annotations


def test_base_package_imports_without_cuda_extras() -> None:
    import spiker

    assert spiker.LIFParams
    assert spiker.LIFState
    assert spiker.ALIFParams
    assert spiker.ALIFState
    assert spiker.ALIFCell
    assert spiker.IzhikevichParams
    assert spiker.IzhikevichState
    assert spiker.IzhikevichCell
    assert spiker.LoweredNeuron
    assert spiker.NeuronBackwardPlan
    assert spiker.NeuronIR
    assert spiker.SurrogateBuilder
    assert spiker.CustomSurrogateNeuronCell
    assert spiker.SurrogateALIFCell
    assert spiker.alif_ir
    assert spiker.alif_step
    assert spiker.alif_unroll
    assert spiker.alif_forward
    assert spiker.surrogate_alif_forward
    assert spiker.surrogate_alif_unroll
    assert spiker.izhikevich_forward
    assert spiker.izhikevich_ir
    assert spiker.izhikevich_step
    assert spiker.izhikevich_unroll
    assert spiker.lif_ir
    assert spiker.compile_surrogate_derivative_ir
    assert spiker.lower_neuron_to_ssa
    assert spiker.plan_generated_backward_ir
    assert spiker.validate_expr
    assert spiker.validate_generated_backward_ir
    assert spiker.validate_neuron_ir
    assert spiker.validate_surrogate_derivative_ir
    assert spiker.render_triton_step_body
    assert spiker.render_surrogate_derivative_ir_body
    assert "superspike" in spiker.SURROGATE_NAMES
    assert spiker.superspike_surrogate
    assert spiker.superspike_surrogate_derivative_expr
    assert spiker.lif_step
    assert spiker.lif_unroll
    assert spiker.LinearCustomSurrogateNeuron
    assert spiker.LinearCustomSurrogateNeuronRate
    assert spiker.linear_surrogate_lif_forward
    assert spiker.linear_surrogate_lif_packed_forward
    assert spiker.LinearSurrogateLIFPacked
    assert spiker.LinearSurrogateLIFRate
    assert spiker.SpikeGPTModelType
    assert spiker.SpikeGPTPreset
    assert spiker.SamplingMode
    assert spiker.recommended_checkpoint_size(100) == 25


def test_fused_lif_unroll_auto_backend_runs_on_cpu() -> None:
    import torch

    from spiker.functional import lif_unroll
    from spiker.kernels import fused_lif_unroll
    from spiker.neurons import LIFParams, LIFState

    inputs = torch.rand((3, 2, 4))
    initial = LIFState(membrane=torch.zeros((2, 4)))
    params = LIFParams()

    expected_state, expected_spikes = lif_unroll(inputs, initial, params)
    actual_state, actual_spikes = fused_lif_unroll(inputs, initial, params)

    assert torch.allclose(actual_state.membrane, expected_state.membrane)
    assert torch.equal(actual_spikes, expected_spikes)


def test_lif_forward_explicit_torch_backend_does_not_warn() -> None:
    import warnings

    import torch

    from spiker.kernels import lif_forward
    from spiker.neurons import LIFParams, LIFState

    inputs = torch.rand((3, 2, 4))
    initial = LIFState(membrane=torch.zeros((2, 4)))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        lif_forward(inputs, initial, LIFParams(), backend="torch")

    assert len(caught) == 0
