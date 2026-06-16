from __future__ import annotations

import pytest
import torch

from myelin.dsl import (
    Expr,
    NeuronBackwardPlan,
    NeuronBuilder,
    NeuronIRValidationReport,
    SurrogateBuilder,
    alif_ir,
    analyze_neuron_ir,
    evaluate_expr,
    evaluate_neuron,
    evaluate_neuron_unroll,
    evaluate_surrogate_derivative,
    input_,
    izhikevich_ir,
    lif_ir,
    plan_generated_backward_ir,
    surrogate_derivative_expr,
    validate_expr,
    validate_generated_backward_ir,
    validate_generated_forward_ir,
    validate_neuron_ir,
    validate_surrogate_derivative_ir,
    where,
)
from myelin.dsl import (
    param as _Param,
)
from myelin.dsl import (
    state as _State,
)
from myelin.functional import lif_unroll
from myelin.neurons import (
    ALIFParams,
    ALIFState,
    IzhikevichParams,
    IzhikevichState,
    LIFParams,
    LIFState,
    alif_step,
    izhikevich_step,
    lif_step,
)
from myelin.surrogates import SURROGATE_NAMES, surrogate_derivative


@pytest.mark.parametrize(
    "ir, name, state, inputs, params",
    [
        (
            lif_ir(),
            "lif",
            ("membrane",),
            ("input_current",),
            ("decay", "threshold", "reset"),
        ),
        (
            alif_ir(),
            "alif",
            ("membrane", "adaptation"),
            ("input_current",),
            ("decay", "adaptation_decay", "threshold", "reset", "beta"),
        ),
        (
            izhikevich_ir(),
            "izhikevich",
            ("voltage", "recovery"),
            ("input_current",),
            (
                "recovery_decay",
                "recovery_coupling",
                "reset_voltage",
                "recovery_jump",
                "threshold",
                "dt",
                "voltage_square_coeff",
                "voltage_coeff",
                "voltage_bias",
            ),
        ),
    ],
)
def test_neuron_ir_has_expected_boundary(ir, name, state, inputs, params) -> None:
    assert ir.name == name
    assert ir.state == state
    assert ir.inputs == inputs
    assert ir.params == params
    assert set(ir.next_state) == set(state)
    assert set(ir.outputs) == {"spike"}


@pytest.mark.parametrize(
    "ir, reference_step, state, input_current, params",
    [
        (
            lif_ir(),
            lif_step,
            LIFState(membrane=torch.tensor([[0.5, 0.0]])),
            torch.tensor([[0.6, 0.2]]),
            LIFParams(tau_mem=10.0, threshold=1.0, reset=-0.25),
        ),
        (
            alif_ir(),
            alif_step,
            ALIFState(
                membrane=torch.tensor([[0.5, 0.0]]),
                adaptation=torch.tensor([[0.1, 0.0]]),
            ),
            torch.tensor([[0.7, 1.1]]),
            ALIFParams(tau_mem=10.0, tau_adaptation=5.0, threshold=1.0, reset=0.0, beta=0.5),
        ),
        (
            izhikevich_ir(),
            izhikevich_step,
            IzhikevichState(
                voltage=torch.tensor([[29.0, -65.0]]),
                recovery=torch.tensor([[0.0, -13.0]]),
            ),
            torch.tensor([[20.0, 0.0]]),
            IzhikevichParams(),
        ),
    ],
)
def test_neuron_ir_evaluator_matches_reference_step(
    ir, reference_step, state, input_current, params
) -> None:
    state_values = {field: getattr(state, field) for field in ir.state}
    param_values = {name: getattr(params, name) for name in ir.params}

    next_state, outputs = evaluate_neuron(
        ir,
        state_values=state_values,
        input_values={"input_current": input_current},
        param_values=param_values,
    )
    expected_state, expected_spike = reference_step(state, input_current, params)

    for field in ir.state:
        assert torch.allclose(next_state[field], getattr(expected_state, field))
    assert torch.equal(outputs["spike"], expected_spike)


def test_lif_ir_evaluator_unroll_matches_reference_unroll() -> None:
    inputs = torch.tensor(
        [
            [[0.6, 0.1]],
            [[0.6, 0.9]],
            [[0.1, 0.2]],
        ]
    )
    params = LIFParams(tau_mem=10.0, threshold=1.0, reset=0.0)
    initial = LIFState(membrane=torch.zeros((1, 2)))
    ir = lif_ir()
    membrane = initial.membrane
    spikes = []

    for input_current in inputs:
        next_state, outputs = evaluate_neuron(
            ir,
            state_values={"membrane": membrane},
            input_values={"input_current": input_current},
            param_values={
                "decay": params.decay,
                "threshold": params.threshold,
                "reset": params.reset,
            },
        )
        membrane = next_state["membrane"]
        spikes.append(outputs["spike"])

    expected_state, expected_spikes = lif_unroll(inputs, initial, params)

    assert torch.allclose(membrane, expected_state.membrane)
    assert torch.equal(torch.stack(spikes), expected_spikes)


def test_lif_ir_unroll_helper_matches_reference_unroll() -> None:
    inputs = torch.tensor(
        [
            [[0.6, 0.1]],
            [[0.6, 0.9]],
            [[0.1, 0.2]],
        ]
    )
    params = LIFParams(tau_mem=10.0, threshold=1.0, reset=0.0)
    initial = LIFState(membrane=torch.zeros((1, 2)))

    final_state, spikes = evaluate_neuron_unroll(
        lif_ir(),
        inputs,
        {"membrane": initial.membrane},
        {
            "decay": params.decay,
            "threshold": params.threshold,
            "reset": params.reset,
        },
    )
    expected_state, expected_spikes = lif_unroll(inputs, initial, params)

    assert torch.allclose(final_state["membrane"], expected_state.membrane)
    assert torch.equal(spikes, expected_spikes)


def test_surrogate_derivative_expr_matches_reference_derivatives() -> None:
    centered = torch.tensor([[-2.0, -0.25, 0.0, 0.5, 3.0]])

    for name in SURROGATE_NAMES:
        expr = surrogate_derivative_expr(name, input_("centered"))
        actual = evaluate_expr(
            expr,
            state_values={},
            input_values={"centered": centered},
            param_values={},
        )
        expected = surrogate_derivative(centered, name)

        assert isinstance(actual, torch.Tensor)
        assert torch.allclose(actual, expected)


def test_surrogate_builder_authors_parameterized_derivative_ir() -> None:
    builder = SurrogateBuilder("wide_fast")
    centered = builder.centered()
    width = builder.param("width")

    ir = builder.build(0.5 / (1.0 + (centered / width).abs()).square())
    values = torch.tensor([[-2.0, 0.0, 2.0]])
    actual = evaluate_surrogate_derivative(ir, values, {"width": 2.0})
    expected = 0.5 / (1.0 + (values / 2.0).abs()).square()

    validate_surrogate_derivative_ir(ir)
    assert ir.name == "wide_fast_surrogate_derivative"
    assert ir.inputs == ("centered",)
    assert ir.params == ("width",)
    assert tuple(ir.outputs) == ("derivative",)
    assert torch.allclose(actual, expected)


def test_surrogate_builder_rejects_conflicting_param_name() -> None:
    builder = SurrogateBuilder("custom")

    with pytest.raises(ValueError, match="centered"):
        builder.param("centered")


def test_neuron_builder_authors_lif_like_ir_without_manual_metadata() -> None:
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

    assert ir.name == "custom_lif"
    assert ir.state == ("membrane",)
    assert ir.inputs == ("input_current",)
    assert ir.params == ("decay", "threshold", "reset")

    state_value = torch.tensor([[0.5, 0.0]])
    input_current = torch.tensor([[0.6, 0.2]])
    params = LIFParams(tau_mem=10.0, threshold=1.0, reset=-0.25)
    next_state, outputs = evaluate_neuron(
        ir,
        state_values={"membrane": state_value},
        input_values={"input_current": input_current},
        param_values={
            "decay": params.decay,
            "threshold": params.threshold,
            "reset": params.reset,
        },
    )
    expected_state, expected_spike = lif_step(LIFState(membrane=state_value), input_current, params)

    assert torch.allclose(next_state["membrane"], expected_state.membrane)
    assert torch.equal(outputs["spike"], expected_spike)


def test_analyze_neuron_ir_reports_backend_readiness() -> None:
    ir = lif_ir()

    report = analyze_neuron_ir(ir)

    assert isinstance(report, NeuronIRValidationReport)
    assert report.is_valid
    assert report.errors == ()
    assert report.warnings == ()
    assert report.generated_forward_errors == ()
    assert report.supports_unroll_api
    assert report.supports_generated_forward
    assert report.can_use_generated_forward
    assert report.supports_generated_backward
    assert report.can_use_generated_backward
    assert report.generated_backward_errors == ()
    assert report.generated_backward_plan == NeuronBackwardPlan(
        kind="lif_hard_reset",
        is_implemented=True,
        state_name="membrane",
        input_name="input_current",
        output_name="spike",
        decay_param="decay",
        threshold_param="threshold",
        reset_param="reset",
        saved_values=("pre_reset_membrane", "spike"),
    )
    assert plan_generated_backward_ir(ir) == report.generated_backward_plan
    validate_generated_backward_ir(ir)


def test_builder_authored_lif_reports_generated_backward_readiness() -> None:
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

    report = analyze_neuron_ir(ir)

    assert report.supports_generated_forward
    assert report.supports_generated_backward
    assert report.generated_backward_errors == ()
    plan = plan_generated_backward_ir(ir, context="CustomNeuronCell")
    assert plan.kind == "lif_hard_reset"
    assert plan.is_implemented
    assert plan.saved_values == ("pre_reset_membrane", "spike")
    validate_generated_backward_ir(ir, context="CustomNeuronCell")

    non_lif = NeuronBuilder("non_lif")
    value = non_lif.state("membrane")
    input_current = non_lif.input("input_current")
    non_lif.param("decay")
    non_lif.param("threshold")
    non_lif.param("reset")
    non_lif_ir = non_lif.build(
        next_state={"membrane": value + input_current},
        outputs={"spike": where(value.ge(0.0), 1.0, 0.0)},
    )
    non_lif_report = analyze_neuron_ir(non_lif_ir)

    assert not non_lif_report.supports_generated_backward
    assert non_lif_report.generated_backward_plan is None
    assert any(
        "hard-reset LIF state update" in error for error in non_lif_report.generated_backward_errors
    )


def test_alif_reports_planned_generated_backward_contract() -> None:
    ir = alif_ir()

    report = analyze_neuron_ir(ir)

    assert report.supports_generated_forward
    assert not report.supports_generated_backward
    assert report.generated_backward_errors == (
        "ALIF generated backward plan is recognized, but its kernel implementation is not "
        "available yet",
    )
    assert report.generated_backward_plan == NeuronBackwardPlan(
        kind="alif_adaptive_threshold",
        is_implemented=False,
        state_name="membrane",
        input_name="input_current",
        output_name="spike",
        decay_param="decay",
        threshold_param="threshold",
        reset_param="reset",
        saved_values=("pre_reset_membrane", "adaptive_threshold", "spike", "adaptation"),
        adaptation_state_name="adaptation",
        adaptation_decay_param="adaptation_decay",
        beta_param="beta",
    )
    plan = plan_generated_backward_ir(ir, allow_unimplemented=True)
    assert plan == report.generated_backward_plan

    with pytest.raises(ValueError, match="ALIF generated backward plan is recognized"):
        plan_generated_backward_ir(ir)


def test_analyze_neuron_ir_collects_actionable_errors_without_raising() -> None:
    from myelin.dsl import NeuronIR, param, state

    ir = NeuronIR(
        name="bad-name",
        state=("membrane", "membrane"),
        inputs=("current",),
        params=("threshold",),
        next_state={
            "membrane": state("missing_state"),
            "adaptation": input_("missing_input") + param("missing_param"),
        },
        outputs={"spike-rate": Expr("add", args=(state("membrane"),))},
    )

    report = analyze_neuron_ir(ir)

    assert not report.is_valid
    assert not report.supports_unroll_api
    assert not report.supports_generated_forward
    assert not report.supports_generated_backward
    assert report.generated_forward_errors
    assert report.generated_backward_errors
    assert any("neuron IR name must be a valid identifier" in error for error in report.errors)
    assert any("duplicate state declaration" in error for error in report.errors)
    assert any("undeclared state names: adaptation" in error for error in report.errors)
    assert any("missing_state" in error for error in report.errors)
    assert any("missing_input" in error for error in report.errors)
    assert any("missing_param" in error for error in report.errors)
    assert any("output name must be a valid identifier" in error for error in report.errors)
    assert any("expects 2 args" in error for error in report.errors)


def test_analyze_neuron_ir_warns_about_non_default_public_unroll_boundary() -> None:
    from myelin.dsl import NeuronIR, input_, state

    ir = NeuronIR(
        name="custom_readout",
        state=("value",),
        inputs=("input_a", "input_b"),
        params=(),
        next_state={"value": state("value") + input_("input_a")},
        outputs={"rate": state("value")},
    )

    report = analyze_neuron_ir(ir)

    assert report.is_valid
    assert not report.supports_unroll_api
    assert not report.supports_generated_forward
    assert not report.supports_generated_backward
    assert "requires one input named input_current" in report.generated_forward_errors
    assert "requires exactly one output named spike" in report.generated_forward_errors
    assert any("exactly one input" in warning for warning in report.warnings)
    assert any("output named 'spike'" in warning for warning in report.warnings)


def test_analyze_neuron_ir_reports_generated_forward_abi_errors() -> None:
    from myelin.dsl import NeuronIR, const, input_, state

    stateless = NeuronIR(
        name="custom_stateless",
        state=(),
        inputs=("input_current",),
        params=(),
        next_state={},
        outputs={"spike": const(0.0)},
    )
    stateless_report = analyze_neuron_ir(stateless)
    assert stateless_report.is_valid
    assert stateless_report.supports_unroll_api
    assert not stateless_report.supports_generated_forward
    assert not stateless_report.supports_generated_backward
    assert stateless_report.generated_forward_errors == ("requires at least one state",)

    wrong_input = NeuronIR(
        name="custom_wrong_input",
        state=("membrane",),
        inputs=("current",),
        params=(),
        next_state={"membrane": input_("current")},
        outputs={"spike": state("membrane")},
    )
    wrong_input_report = analyze_neuron_ir(wrong_input)
    assert wrong_input_report.is_valid
    assert wrong_input_report.supports_unroll_api
    assert not wrong_input_report.supports_generated_forward
    assert wrong_input_report.generated_forward_errors == (
        "requires one input named input_current",
    )

    extra_output = NeuronIR(
        name="custom_extra_output",
        state=("membrane",),
        inputs=("input_current",),
        params=(),
        next_state={"membrane": input_("input_current")},
        outputs={"spike": state("membrane"), "rate": state("membrane")},
    )
    extra_output_report = analyze_neuron_ir(extra_output)
    assert extra_output_report.is_valid
    assert extra_output_report.supports_unroll_api
    assert not extra_output_report.supports_generated_forward
    assert extra_output_report.generated_forward_errors == (
        "requires exactly one output named spike",
    )

    with pytest.raises(
        ValueError,
        match=r"^CustomNeuronCell currently requires exactly one output named spike$",
    ):
        validate_generated_forward_ir(extra_output, context="CustomNeuronCell")


def test_neuron_builder_rejects_undeclared_references() -> None:
    builder = NeuronBuilder("bad")
    membrane = builder.state("membrane")

    with pytest.raises(ValueError, match="undeclared input"):
        builder.build(
            next_state={"membrane": membrane + input_("undeclared_current")},
            outputs={"spike": membrane},
        )

    with pytest.raises(ValueError, match="undeclared state"):
        builder.build(next_state={"adaptation": membrane}, outputs={"spike": membrane})


def test_neuron_builder_rejects_duplicate_declarations() -> None:
    builder = NeuronBuilder("bad")
    builder.state("membrane")

    with pytest.raises(ValueError, match="duplicate state"):
        builder.state("membrane")

    with pytest.raises(ValueError, match="duplicate neuron symbol"):
        builder.param("membrane")


def test_neuron_builder_rejects_names_that_codegen_cannot_render() -> None:
    with pytest.raises(ValueError, match="neuron name must be a valid identifier"):
        NeuronBuilder("bad-name")

    builder = NeuronBuilder("bad_symbols")
    with pytest.raises(ValueError, match="state name must be a valid identifier"):
        builder.state("membrane-voltage")

    with pytest.raises(ValueError, match="param name must be a valid identifier"):
        builder.param("class")


def _make_bad_ir(
    *,
    name: str = "bad",
    state: tuple[str, ...] = ("value",),
    inputs: tuple[str, ...] = (),
    params: tuple[str, ...] = (),
    next_state: dict[str, Expr] | None = None,
    outputs: dict[str, Expr] | None = None,
):
    from myelin.dsl import NeuronIR

    return NeuronIR(
        name=name,
        state=state,
        inputs=inputs,
        params=params,
        next_state=next_state if next_state is not None else {"value": _State("value")},
        outputs=outputs if outputs is not None else {"spike": _State("value")},
    )


@pytest.mark.parametrize(
    "override, expected_match",
    [
        (
            {"name": "bad-name"},
            "neuron IR name must be a valid identifier",
        ),
        (
            {
                "state": ("value-with-hyphen",),
                "next_state": {"value-with-hyphen": _State("value-with-hyphen")},
                "outputs": {"spike": _State("value-with-hyphen")},
            },
            "state name must be a valid identifier",
        ),
        (
            {"outputs": {"spike-rate": _State("value")}},
            "output name must be a valid identifier",
        ),
        (
            {
                "params": ("value",),
                "outputs": {"spike": _Param("value")},
            },
            "duplicate neuron symbol",
        ),
        (
            {
                "state": ("membrane",),
                "next_state": {"membrane": input_("current")},
                "outputs": {"spike": _State("membrane")},
            },
            "undeclared input",
        ),
        (
            {
                "state": ("membrane", "adaptation"),
                "next_state": {"membrane": _State("membrane")},
                "outputs": {"spike": _State("membrane")},
            },
            "missing state updates.*adaptation",
        ),
    ],
)
def test_validate_neuron_ir_rejects_direct_invalid_ir(override, expected_match) -> None:
    with pytest.raises(ValueError, match=expected_match):
        validate_neuron_ir(_make_bad_ir(**override))


def test_validate_expr_rejects_malformed_direct_exprs() -> None:
    with pytest.raises(ValueError, match="expects 2 args"):
        validate_expr(Expr("add", args=(input_("x"),)))

    with pytest.raises(ValueError, match="must not define name or args"):
        validate_expr(Expr("const", value=1.0, args=(input_("x"),)))


def test_evaluate_neuron_unroll_rejects_invalid_boundary() -> None:
    inputs = torch.rand((3, 2, 4))
    ir = lif_ir()
    params = {"decay": 0.9, "threshold": 1.0, "reset": 0.0}

    with pytest.raises(ValueError, match="inputs must be shaped"):
        evaluate_neuron_unroll(ir, inputs[:, :, 0], {"membrane": torch.zeros((2, 4))}, params)

    with pytest.raises(ValueError, match="must match inputs.shape"):
        evaluate_neuron_unroll(ir, inputs, {"membrane": torch.zeros((2, 3))}, params)

    with pytest.raises(ValueError, match="missing values"):
        evaluate_neuron_unroll(
            ir,
            inputs,
            {"membrane": torch.zeros((2, 4))},
            {"decay": 0.9, "threshold": 1.0},
        )


def test_validate_neuron_ir_rejects_malformed_expression_before_evaluation() -> None:
    from myelin.dsl import NeuronIR, state

    malformed = NeuronIR(
        name="bad",
        state=("membrane",),
        inputs=(),
        params=(),
        next_state={"membrane": Expr("add", args=(state("membrane"),))},
        outputs={"spike": state("membrane")},
    )

    try:
        evaluate_neuron(
            malformed,
            state_values={"membrane": torch.zeros((1, 2))},
            input_values={},
            param_values={},
        )
    except ValueError as exc:
        assert "expects 2 args" in str(exc)
    else:  # pragma: no cover - assertion clarity.
        raise AssertionError("expected malformed expression error")
