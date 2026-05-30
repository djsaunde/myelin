from __future__ import annotations


def test_base_package_imports_without_cuda_extras() -> None:
    import myelin

    assert myelin.LIFParams
    assert myelin.LIFState
    assert myelin.ALIFParams
    assert myelin.ALIFState
    assert myelin.ALIFCell
    assert myelin.IzhikevichParams
    assert myelin.IzhikevichState
    assert myelin.IzhikevichCell
    assert myelin.LoweredNeuron
    assert myelin.NeuronBackwardPlan
    assert myelin.NeuronIR
    assert myelin.SurrogateBuilder
    assert myelin.CustomSurrogateNeuronCell
    assert myelin.SurrogateALIFCell
    assert myelin.alif_ir
    assert myelin.alif_step
    assert myelin.alif_unroll
    assert myelin.alif_forward
    assert myelin.surrogate_alif_forward
    assert myelin.surrogate_alif_unroll
    assert myelin.izhikevich_forward
    assert myelin.izhikevich_ir
    assert myelin.izhikevich_step
    assert myelin.izhikevich_unroll
    assert myelin.lif_ir
    assert myelin.compile_surrogate_derivative_ir
    assert myelin.lower_neuron_to_ssa
    assert myelin.plan_generated_backward_ir
    assert myelin.validate_expr
    assert myelin.validate_generated_backward_ir
    assert myelin.validate_neuron_ir
    assert myelin.validate_surrogate_derivative_ir
    assert myelin.render_triton_step_body
    assert myelin.render_surrogate_derivative_ir_body
    assert "superspike" in myelin.SURROGATE_NAMES
    assert myelin.superspike_surrogate
    assert myelin.superspike_surrogate_derivative_expr
    assert myelin.lif_step
    assert myelin.lif_unroll
    assert myelin.LinearCustomSurrogateNeuron
    assert myelin.LinearCustomSurrogateNeuronRate
    assert myelin.linear_surrogate_lif_forward
    assert myelin.linear_surrogate_lif_packed_forward
    assert myelin.LinearSurrogateLIFPacked
    assert myelin.LinearSurrogateLIFRate
    assert myelin.SpikeGPTModelType
    assert myelin.SpikeGPTPreset
    assert myelin.SamplingMode
    assert myelin.recommended_checkpoint_size(100) == 25


def test_fused_lif_unroll_auto_backend_runs_on_cpu() -> None:
    import torch

    from myelin.functional import lif_unroll
    from myelin.kernels import fused_lif_unroll
    from myelin.neurons import LIFParams, LIFState

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

    from myelin.kernels import lif_forward
    from myelin.neurons import LIFParams, LIFState

    inputs = torch.rand((3, 2, 4))
    initial = LIFState(membrane=torch.zeros((2, 4)))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        lif_forward(inputs, initial, LIFParams(), backend="torch")

    assert len(caught) == 0
