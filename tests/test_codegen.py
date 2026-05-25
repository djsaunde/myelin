from __future__ import annotations

import pytest
import torch

from spiker._optional import has_triton
from spiker.codegen import (
    compile_surrogate_derivative_ir,
    lower_neuron_to_ssa,
    render_lif_surrogate_backward_step_body,
    render_lowered_neuron,
    render_surrogate_derivative_body,
    render_surrogate_derivative_ir_body,
    render_triton_step_body,
)
from spiker.dsl import (
    NeuronIR,
    SurrogateBuilder,
    alif_ir,
    input_,
    lif_ir,
    surrogate_derivative_expr,
)
from spiker.neurons import LIFParams, LIFState, lif_step
from spiker.surrogates import surrogate_derivative


def test_lif_ir_lowers_to_shared_ssa_temporaries() -> None:
    lowered = lower_neuron_to_ssa(lif_ir())

    assert lowered.name == "lif"
    assert lowered.next_state == {"membrane": "tmp3"}
    assert lowered.outputs == {"spike": "tmp4"}
    assert [assignment.expression for assignment in lowered.assignments] == [
        "(membrane * decay)",
        "(tmp0 + input_current)",
        "(tmp1 >= threshold)",
        "torch.where(tmp2, reset, tmp1)",
        "torch.where(tmp2, 1.0, 0.0)",
    ]


def test_lowering_rejects_invalid_direct_ir_before_rendering() -> None:
    invalid = NeuronIR(
        name="bad",
        state=("membrane",),
        params=(),
        inputs=(),
        next_state={"membrane": input_("missing")},
        outputs={"spike": input_("missing")},
    )

    with pytest.raises(ValueError, match="undeclared input"):
        lower_neuron_to_ssa(invalid)


def test_lif_ir_lowers_to_triton_where_dialect() -> None:
    rendered = render_lowered_neuron(lower_neuron_to_ssa(lif_ir(), dialect="triton"))

    assert "tmp2 = (tmp1 >= threshold)" in rendered
    assert "tl.where(tmp2, reset, tmp1)" in rendered
    assert "tl.where(tmp2, 1.0, 0.0)" in rendered


def test_lif_ir_renders_triton_step_body_with_kernel_names() -> None:
    body = render_triton_step_body(
        lower_neuron_to_ssa(lif_ir(), dialect="triton"),
        variable_map={
            "input_current": "input_current",
            "membrane": "membrane",
            "decay": "decay",
            "threshold": "threshold",
            "reset": "reset",
        },
        output_map={"spike": "spike"},
        indent="",
    )

    assert body.splitlines() == [
        "tmp0 = (membrane * decay)",
        "tmp1 = (tmp0 + input_current)",
        "tmp2 = (tmp1 >= threshold)",
        "tmp3 = tl.where(tmp2, reset, tmp1)",
        "tmp4 = tl.where(tmp2, 1.0, 0.0)",
        "membrane = tmp3",
        "spike = tmp4",
    ]


def test_generated_lif_forward_kernel_source_contains_template() -> None:
    if not has_triton():
        pytest.skip("generated Triton source helper requires Triton imports")
    from spiker.triton.generated import render_lif_forward_kernel_source

    source = render_lif_forward_kernel_source()

    assert "for t in range(timesteps):" in source
    assert "input_current = tl.load" in source
    assert "membrane = tmp3" in source
    assert "tl.store(spikes_ptr" in source


def test_triton_step_body_variable_substitution_is_token_aware() -> None:
    body = render_triton_step_body(
        lower_neuron_to_ssa(lif_ir(), dialect="triton"),
        variable_map={"reset": "reset_value"},
        indent="",
    )

    assert "threshold" in body
    assert "threset_valuehold" not in body
    assert "reset_value" in body


def test_rendered_lif_python_fragment_executes_like_reference_step() -> None:
    params = LIFParams(tau_mem=10.0, threshold=1.0, reset=-0.25)
    state = LIFState(membrane=torch.tensor([[0.5, 0.0]]))
    input_current = torch.tensor([[0.6, 0.2]])
    namespace = {
        "torch": torch,
        "membrane": state.membrane,
        "input_current": input_current,
        "decay": params.decay,
        "threshold": params.threshold,
        "reset": params.reset,
    }
    code = render_lowered_neuron(lower_neuron_to_ssa(lif_ir()))

    exec(code, namespace)  # noqa: S102 - tests generated trusted local code.
    expected_state, expected_spike = lif_step(state, input_current, params)

    assert torch.allclose(namespace["next_membrane"], expected_state.membrane)
    assert torch.equal(namespace["out_spike"], expected_spike)


def test_alif_ir_lowers_with_two_state_outputs_and_shared_spike() -> None:
    lowered = lower_neuron_to_ssa(alif_ir())

    assert lowered.name == "alif"
    assert set(lowered.next_state) == {"membrane", "adaptation"}
    expressions = [assignment.expression for assignment in lowered.assignments]
    adaptation_expr = next(
        assignment.expression
        for assignment in lowered.assignments
        if assignment.target == lowered.next_state["adaptation"]
    )
    assert lowered.outputs["spike"] in adaptation_expr
    assert "(beta * adaptation)" in expressions
    assert "torch.where" in "\n".join(expressions)


def test_surrogate_derivative_expr_renders_python_and_triton() -> None:
    triangular_ir = NeuronIR(
        name="surrogate",
        state=(),
        params=(),
        inputs=("centered",),
        next_state={},
        outputs={"derivative": surrogate_derivative_expr("triangular", input_("centered"))},
    )
    sigmoid_ir = NeuronIR(
        name="surrogate",
        state=(),
        params=(),
        inputs=("centered",),
        next_state={},
        outputs={"derivative": surrogate_derivative_expr("sigmoid", input_("centered"))},
    )
    multi_gaussian_ir = NeuronIR(
        name="surrogate",
        state=(),
        params=(),
        inputs=("centered",),
        next_state={},
        outputs={"derivative": surrogate_derivative_expr("multi_gaussian", input_("centered"))},
    )

    python_rendered = render_lowered_neuron(lower_neuron_to_ssa(triangular_ir))
    triton_rendered = render_lowered_neuron(lower_neuron_to_ssa(triangular_ir, dialect="triton"))
    sigmoid_python_rendered = render_lowered_neuron(lower_neuron_to_ssa(sigmoid_ir))
    sigmoid_triton_rendered = render_lowered_neuron(
        lower_neuron_to_ssa(sigmoid_ir, dialect="triton")
    )
    multi_gaussian_python_rendered = render_lowered_neuron(lower_neuron_to_ssa(multi_gaussian_ir))
    multi_gaussian_triton_rendered = render_lowered_neuron(
        lower_neuron_to_ssa(multi_gaussian_ir, dialect="triton")
    )

    assert "torch.abs(centered)" in python_rendered
    assert "tl.abs(centered)" in triton_rendered
    assert "torch.sign(centered)" in python_rendered
    assert "tl.where(centered > 0.0" in triton_rendered
    assert "torch.sigmoid(centered)" in sigmoid_python_rendered
    assert "tl.sigmoid(centered)" in sigmoid_triton_rendered
    assert "torch.exp(" in multi_gaussian_python_rendered
    assert "tl.exp(" in multi_gaussian_triton_rendered
    assert "out_derivative" in python_rendered


def test_surrogate_derivative_body_executes_like_reference_derivative() -> None:
    centered = torch.tensor([[-1.5, -0.25, 0.0, 0.75]])
    namespace = {"torch": torch, "centered_value": centered}
    body = render_surrogate_derivative_body(
        "fast_sigmoid",
        dialect="python",
        centered_name="centered_value",
        output_name="derivative",
    )

    exec(body, namespace)  # noqa: S102 - tests generated trusted local code.

    assert torch.allclose(namespace["derivative"], surrogate_derivative(centered, "fast_sigmoid"))


def test_surrogate_derivative_body_renders_triton_loop_fragment_names() -> None:
    body = render_surrogate_derivative_body(
        "sigmoid",
        dialect="triton",
        centered_name="pre_reset_minus_threshold",
        output_name="surrogate_grad",
        indent="    ",
    )

    assert "tl.sigmoid(pre_reset_minus_threshold)" in body
    assert "surrogate_grad =" in body
    assert all(line.startswith("    ") for line in body.splitlines())


def test_custom_surrogate_derivative_body_renders_param_map() -> None:
    builder = SurrogateBuilder("wide_fast")
    centered = builder.centered()
    width = builder.param("width")
    ir = builder.build(0.5 / (1.0 + (centered / width).abs()).square())

    body = render_surrogate_derivative_ir_body(
        ir,
        dialect="triton",
        centered_name="centered_value",
        output_name="surrogate_grad",
        param_map={"width": "surrogate_width"},
    )

    assert "centered_value" in body
    assert "surrogate_width" in body
    assert "surrogate_grad =" in body


def test_compile_surrogate_derivative_ir_executes_reusable_callable() -> None:
    builder = SurrogateBuilder("wide_fast")
    centered_expr = builder.centered()
    width = builder.param("width")
    ir = builder.build(0.5 / (1.0 + (centered_expr / width).abs()).square())
    centered = torch.tensor([[-2.0, 0.0, 2.0]])

    compiled = compile_surrogate_derivative_ir(ir)
    actual = compiled(centered, {"width": 2.0})
    expected = 0.5 / (1.0 + (centered / 2.0).abs()).square()

    assert torch.allclose(actual, expected)


def test_compile_surrogate_derivative_ir_reuses_cached_callable() -> None:
    builder = SurrogateBuilder("wide_fast")
    centered_expr = builder.centered()
    width = builder.param("width")
    ir = builder.build(0.5 / (1.0 + (centered_expr / width).abs()).square())

    same_builder = SurrogateBuilder("wide_fast")
    same_centered_expr = same_builder.centered()
    same_width = same_builder.param("width")
    same_ir = same_builder.build(0.5 / (1.0 + (same_centered_expr / same_width).abs()).square())

    other_builder = SurrogateBuilder("narrow_fast")
    other_centered_expr = other_builder.centered()
    other_width = other_builder.param("width")
    other_ir = other_builder.build(0.5 / (1.0 + (other_centered_expr / other_width).abs()).square())

    compiled = compile_surrogate_derivative_ir(ir)

    assert compile_surrogate_derivative_ir(ir) is compiled
    assert compile_surrogate_derivative_ir(same_ir) is compiled
    assert compile_surrogate_derivative_ir(other_ir) is not compiled


def test_lif_surrogate_backward_step_body_executes_like_reference_recurrence() -> None:
    pre_reset = torch.tensor([[0.2, 0.8, 1.3]])
    spike = torch.tensor([[0.0, 1.0, 1.0]])
    grad_membrane = torch.tensor([[0.1, -0.2, 0.3]])
    grad_spike = torch.tensor([[0.4, 0.0, -0.1]])
    threshold = 0.7
    reset = -0.2
    decay = 0.9
    surrogate_slope = 5.0
    namespace = {
        "torch": torch,
        "pre_reset": pre_reset,
        "spike": spike,
        "grad_membrane": grad_membrane,
        "grad_spike": grad_spike,
        "threshold": threshold,
        "reset": reset,
        "decay": decay,
        "surrogate_slope": surrogate_slope,
    }

    body = render_lif_surrogate_backward_step_body("fast_sigmoid", dialect="python")
    exec(body, namespace)  # noqa: S102 - tests generated trusted local code.

    centered = surrogate_slope * (pre_reset - threshold)
    d_spike_d_membrane = surrogate_slope * surrogate_derivative(centered, "fast_sigmoid")
    expected_grad_pre_reset = grad_membrane * (
        (1.0 - spike) + (reset - pre_reset) * d_spike_d_membrane
    )
    expected_grad_pre_reset = expected_grad_pre_reset + grad_spike * d_spike_d_membrane
    expected_grad_membrane = expected_grad_pre_reset * decay

    assert torch.allclose(namespace["grad_pre_reset"], expected_grad_pre_reset)
    assert torch.allclose(namespace["grad_membrane"], expected_grad_membrane)


def test_lif_surrogate_backward_step_body_renders_triton_names() -> None:
    body = render_lif_surrogate_backward_step_body(
        "sigmoid",
        dialect="triton",
        pre_reset_name="pre",
        spike_name="spk",
        grad_membrane_name="gm",
        grad_spike_name="gs",
        grad_pre_reset_name="gpre",
        next_grad_membrane_name="gm_next",
        indent="    ",
    )

    assert "tl.sigmoid" in body
    assert "pre - threshold" in body
    assert "gpre = gm *" in body
    assert "gm_next = gpre * decay" in body
    assert all(line.startswith("    ") for line in body.splitlines())
