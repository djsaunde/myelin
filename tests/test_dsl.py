from __future__ import annotations

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


def test_lif_ir_has_expected_boundary() -> None:
    ir = lif_ir()

    assert ir.name == "lif"
    assert ir.state == ("membrane",)
    assert ir.inputs == ("input_current",)
    assert ir.params == ("decay", "threshold", "reset")
    assert set(ir.next_state) == {"membrane"}
    assert set(ir.outputs) == {"spike"}


def test_lif_ir_evaluator_matches_reference_step() -> None:
    state = LIFState(membrane=torch.tensor([[0.5, 0.0]]))
    input_current = torch.tensor([[0.6, 0.2]])
    params = LIFParams(tau_mem=10.0, threshold=1.0, reset=-0.25)
    ir = lif_ir()

    next_state, outputs = evaluate_neuron(
        ir,
        state_values={"membrane": state.membrane},
        input_values={"input_current": input_current},
        param_values={
            "decay": params.decay,
            "threshold": params.threshold,
            "reset": params.reset,
        },
    )
    expected_state, expected_spike = lif_step(state, input_current, params)

    assert torch.allclose(next_state["membrane"], expected_state.membrane)
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


def test_alif_ir_has_expected_boundary() -> None:
    ir = alif_ir()

    assert ir.name == "alif"
    assert ir.state == ("membrane", "adaptation")
    assert ir.inputs == ("input_current",)
    assert ir.params == ("decay", "adaptation_decay", "threshold", "reset", "beta")
    assert set(ir.next_state) == {"membrane", "adaptation"}
    assert set(ir.outputs) == {"spike"}


def test_alif_ir_evaluator_matches_reference_step() -> None:
    state = ALIFState(
        membrane=torch.tensor([[0.5, 0.0]]),
        adaptation=torch.tensor([[0.1, 0.0]]),
    )
    input_current = torch.tensor([[0.7, 1.1]])
    params = ALIFParams(tau_mem=10.0, tau_adaptation=5.0, threshold=1.0, reset=0.0, beta=0.5)
    ir = alif_ir()

    next_state, outputs = evaluate_neuron(
        ir,
        state_values={"membrane": state.membrane, "adaptation": state.adaptation},
        input_values={"input_current": input_current},
        param_values={
            "decay": params.decay,
            "adaptation_decay": params.adaptation_decay,
            "threshold": params.threshold,
            "reset": params.reset,
            "beta": params.beta,
        },
    )
    expected_state, expected_spike = alif_step(state, input_current, params)

    assert torch.allclose(next_state["membrane"], expected_state.membrane)
    assert torch.allclose(next_state["adaptation"], expected_state.adaptation)
    assert torch.equal(outputs["spike"], expected_spike)


def test_izhikevich_ir_has_expected_boundary() -> None:
    ir = izhikevich_ir()

    assert ir.name == "izhikevich"
    assert ir.state == ("voltage", "recovery")
    assert ir.inputs == ("input_current",)
    assert ir.params == (
        "recovery_decay",
        "recovery_coupling",
        "reset_voltage",
        "recovery_jump",
        "threshold",
        "dt",
        "voltage_square_coeff",
        "voltage_coeff",
        "voltage_bias",
    )
    assert set(ir.next_state) == {"voltage", "recovery"}
    assert set(ir.outputs) == {"spike"}


def test_izhikevich_ir_evaluator_matches_reference_step() -> None:
    state = IzhikevichState(
        voltage=torch.tensor([[29.0, -65.0]]),
        recovery=torch.tensor([[0.0, -13.0]]),
    )
    input_current = torch.tensor([[20.0, 0.0]])
    params = IzhikevichParams()
    ir = izhikevich_ir()

    next_state, outputs = evaluate_neuron(
        ir,
        state_values={"voltage": state.voltage, "recovery": state.recovery},
        input_values={"input_current": input_current},
        param_values={
            "recovery_decay": params.recovery_decay,
            "recovery_coupling": params.recovery_coupling,
            "reset_voltage": params.reset_voltage,
            "recovery_jump": params.recovery_jump,
            "threshold": params.threshold,
            "dt": params.dt,
            "voltage_square_coeff": params.voltage_square_coeff,
            "voltage_coeff": params.voltage_coeff,
            "voltage_bias": params.voltage_bias,
        },
    )
    expected_state, expected_spike = izhikevich_step(state, input_current, params)

    assert torch.allclose(next_state["voltage"], expected_state.voltage)
    assert torch.allclose(next_state["recovery"], expected_state.recovery)
    assert torch.equal(outputs["spike"], expected_spike)


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

    try:
        builder.param("centered")
    except ValueError as exc:
        assert "centered" in str(exc)
    else:
        raise AssertionError("expected centered param conflict to fail")


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

    try:
        plan_generated_backward_ir(ir)
    except ValueError as exc:
        assert "ALIF generated backward plan is recognized" in str(exc)
    else:  # pragma: no cover - assertion clarity.
        raise AssertionError("expected unimplemented ALIF generated-backward error")


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

    try:
        validate_generated_forward_ir(extra_output, context="CustomNeuronCell")
    except ValueError as exc:
        assert str(exc) == ("CustomNeuronCell currently requires exactly one output named spike")
    else:  # pragma: no cover - assertion clarity.
        raise AssertionError("expected generated forward ABI error")


def test_neuron_builder_rejects_undeclared_references() -> None:
    builder = NeuronBuilder("bad")
    membrane = builder.state("membrane")

    try:
        builder.build(
            next_state={"membrane": membrane + input_("undeclared_current")},
            outputs={"spike": membrane},
        )
    except ValueError as exc:
        assert "undeclared input" in str(exc)
    else:  # pragma: no cover - assertion clarity.
        raise AssertionError("expected undeclared input error")

    try:
        builder.build(next_state={"adaptation": membrane}, outputs={"spike": membrane})
    except ValueError as exc:
        assert "undeclared state" in str(exc)
    else:  # pragma: no cover - assertion clarity.
        raise AssertionError("expected undeclared state error")


def test_neuron_builder_rejects_duplicate_declarations() -> None:
    builder = NeuronBuilder("bad")
    builder.state("membrane")

    try:
        builder.state("membrane")
    except ValueError as exc:
        assert "duplicate state" in str(exc)
    else:  # pragma: no cover - assertion clarity.
        raise AssertionError("expected duplicate state error")

    try:
        builder.param("membrane")
    except ValueError as exc:
        assert "duplicate neuron symbol" in str(exc)
    else:  # pragma: no cover - assertion clarity.
        raise AssertionError("expected cross-kind duplicate error")


def test_neuron_builder_rejects_names_that_codegen_cannot_render() -> None:
    try:
        NeuronBuilder("bad-name")
    except ValueError as exc:
        assert "neuron name must be a valid identifier" in str(exc)
    else:  # pragma: no cover - assertion clarity.
        raise AssertionError("expected invalid neuron name error")

    builder = NeuronBuilder("bad_symbols")
    try:
        builder.state("membrane-voltage")
    except ValueError as exc:
        assert "state name must be a valid identifier" in str(exc)
    else:  # pragma: no cover - assertion clarity.
        raise AssertionError("expected invalid state name error")

    try:
        builder.param("class")
    except ValueError as exc:
        assert "param name must be a valid identifier" in str(exc)
    else:  # pragma: no cover - assertion clarity.
        raise AssertionError("expected keyword param name error")


def test_validate_neuron_ir_rejects_direct_invalid_ir() -> None:
    from myelin.dsl import NeuronIR, param, state

    invalid_name = NeuronIR(
        name="bad-name",
        state=("value",),
        inputs=(),
        params=(),
        next_state={"value": state("value")},
        outputs={"spike": state("value")},
    )
    try:
        validate_neuron_ir(invalid_name)
    except ValueError as exc:
        assert "neuron IR name must be a valid identifier" in str(exc)
    else:  # pragma: no cover - assertion clarity.
        raise AssertionError("expected invalid IR name error")

    invalid_symbol = NeuronIR(
        name="bad",
        state=("value-with-hyphen",),
        inputs=(),
        params=(),
        next_state={"value-with-hyphen": state("value-with-hyphen")},
        outputs={"spike": state("value-with-hyphen")},
    )
    try:
        validate_neuron_ir(invalid_symbol)
    except ValueError as exc:
        assert "state name must be a valid identifier" in str(exc)
    else:  # pragma: no cover - assertion clarity.
        raise AssertionError("expected invalid state symbol error")

    invalid_output = NeuronIR(
        name="bad",
        state=("value",),
        inputs=(),
        params=(),
        next_state={"value": state("value")},
        outputs={"spike-rate": state("value")},
    )
    try:
        validate_neuron_ir(invalid_output)
    except ValueError as exc:
        assert "output name must be a valid identifier" in str(exc)
    else:  # pragma: no cover - assertion clarity.
        raise AssertionError("expected invalid output name error")

    duplicate_symbol = NeuronIR(
        name="bad",
        state=("value",),
        inputs=(),
        params=("value",),
        next_state={"value": state("value")},
        outputs={"spike": param("value")},
    )
    try:
        validate_neuron_ir(duplicate_symbol)
    except ValueError as exc:
        assert "duplicate neuron symbol" in str(exc)
    else:  # pragma: no cover - assertion clarity.
        raise AssertionError("expected duplicate symbol error")

    undeclared_reference = NeuronIR(
        name="bad",
        state=("membrane",),
        inputs=(),
        params=(),
        next_state={"membrane": input_("current")},
        outputs={"spike": state("membrane")},
    )
    try:
        evaluate_neuron(
            undeclared_reference,
            state_values={"membrane": torch.zeros((1, 2))},
            input_values={},
            param_values={},
        )
    except ValueError as exc:
        assert "undeclared input" in str(exc)
    else:  # pragma: no cover - assertion clarity.
        raise AssertionError("expected undeclared input error")

    missing_state_update = NeuronIR(
        name="bad",
        state=("membrane", "adaptation"),
        inputs=(),
        params=(),
        next_state={"membrane": state("membrane")},
        outputs={"spike": state("membrane")},
    )
    try:
        validate_neuron_ir(missing_state_update)
    except ValueError as exc:
        assert "missing state updates" in str(exc)
        assert "adaptation" in str(exc)
    else:  # pragma: no cover - assertion clarity.
        raise AssertionError("expected missing state update error")


def test_validate_expr_rejects_malformed_direct_exprs() -> None:
    try:
        validate_expr(Expr("add", args=(input_("x"),)))
    except ValueError as exc:
        assert "expects 2 args" in str(exc)
    else:  # pragma: no cover - assertion clarity.
        raise AssertionError("expected arity error")

    try:
        validate_expr(Expr("const", value=1.0, args=(input_("x"),)))
    except ValueError as exc:
        assert "must not define name or args" in str(exc)
    else:  # pragma: no cover - assertion clarity.
        raise AssertionError("expected malformed const error")


def test_evaluate_neuron_unroll_rejects_invalid_boundary() -> None:
    inputs = torch.rand((3, 2, 4))
    ir = lif_ir()
    params = {"decay": 0.9, "threshold": 1.0, "reset": 0.0}

    try:
        evaluate_neuron_unroll(ir, inputs[:, :, 0], {"membrane": torch.zeros((2, 4))}, params)
    except ValueError as exc:
        assert "inputs must be shaped" in str(exc)
    else:  # pragma: no cover - assertion clarity.
        raise AssertionError("expected input shape error")

    try:
        evaluate_neuron_unroll(ir, inputs, {"membrane": torch.zeros((2, 3))}, params)
    except ValueError as exc:
        assert "must match inputs.shape" in str(exc)
    else:  # pragma: no cover - assertion clarity.
        raise AssertionError("expected state shape error")

    try:
        evaluate_neuron_unroll(
            ir,
            inputs,
            {"membrane": torch.zeros((2, 4))},
            {"decay": 0.9, "threshold": 1.0},
        )
    except ValueError as exc:
        assert "missing values" in str(exc)
    else:  # pragma: no cover - assertion clarity.
        raise AssertionError("expected missing params error")


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
