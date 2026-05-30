"""Minimal neuron DSL and IR.

The DSL starts intentionally small: pointwise expressions over named state,
input, and parameter values. This gives us a concrete IR to target before adding
Triton emission.
"""

from __future__ import annotations

import keyword
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, TypeAlias

import torch

ExprKind: TypeAlias = Literal[
    "const",
    "input",
    "param",
    "state",
    "abs",
    "add",
    "div",
    "exp",
    "ge",
    "lt",
    "mul",
    "sign",
    "sigmoid",
    "square",
    "sub",
    "where",
]
Value: TypeAlias = torch.Tensor | float
GeneratedBackwardKind: TypeAlias = Literal["lif_hard_reset", "alif_adaptive_threshold"]

LEAF_EXPR_KINDS = {"const", "input", "param", "state"}
UNARY_EXPR_KINDS = {"abs", "exp", "sign", "sigmoid", "square"}
BINARY_EXPR_KINDS = {"add", "div", "ge", "lt", "mul", "sub"}
TERNARY_EXPR_KINDS = {"where"}


def _is_valid_symbol(name: str) -> bool:
    return name.isidentifier() and not keyword.iskeyword(name)


@dataclass(frozen=True)
class Expr:
    """Pointwise expression node in the neuron IR."""

    kind: ExprKind
    name: str | None = None
    value: float | None = None
    args: tuple[Expr, ...] = ()

    def __add__(self, other: Expr | float) -> Expr:
        return add(self, as_expr(other))

    def __radd__(self, other: Expr | float) -> Expr:
        return add(as_expr(other), self)

    def __sub__(self, other: Expr | float) -> Expr:
        return sub(self, as_expr(other))

    def __rsub__(self, other: Expr | float) -> Expr:
        return sub(as_expr(other), self)

    def __mul__(self, other: Expr | float) -> Expr:
        return mul(self, as_expr(other))

    def __rmul__(self, other: Expr | float) -> Expr:
        return mul(as_expr(other), self)

    def __truediv__(self, other: Expr | float) -> Expr:
        return div(self, as_expr(other))

    def __rtruediv__(self, other: Expr | float) -> Expr:
        return div(as_expr(other), self)

    def abs(self) -> Expr:
        return abs_(self)

    def exp(self) -> Expr:
        return exp(self)

    def square(self) -> Expr:
        return square(self)

    def ge(self, other: Expr | float) -> Expr:
        return ge(self, as_expr(other))

    def lt(self, other: Expr | float) -> Expr:
        return lt(self, as_expr(other))

    def sign(self) -> Expr:
        return sign(self)

    def sigmoid(self) -> Expr:
        return sigmoid(self)


@dataclass(frozen=True)
class NeuronIR:
    """Restricted state-update graph for one neuron population."""

    name: str
    state: tuple[str, ...]
    params: tuple[str, ...]
    inputs: tuple[str, ...]
    next_state: Mapping[str, Expr]
    outputs: Mapping[str, Expr]


@dataclass(frozen=True)
class NeuronBackwardPlan:
    """Recognized generated-backward contract for a custom neuron IR.

    This is intentionally more specific than ``supports_generated_backward``:
    callers can inspect the recognized recurrence shape and bind the required
    saved values without re-matching expression trees themselves.
    """

    kind: GeneratedBackwardKind
    is_implemented: bool
    state_name: str
    input_name: str
    output_name: str
    decay_param: str
    threshold_param: str
    reset_param: str
    saved_values: tuple[str, ...]
    adaptation_state_name: str | None = None
    adaptation_decay_param: str | None = None
    beta_param: str | None = None


@dataclass(frozen=True)
class NeuronIRValidationReport:
    """Structured diagnostics for a ``NeuronIR``."""

    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    supports_unroll_api: bool
    supports_generated_forward: bool
    generated_forward_errors: tuple[str, ...] = ()
    supports_generated_backward: bool = False
    generated_backward_errors: tuple[str, ...] = (
        "requires hard-reset LIF-shaped custom IR; richer custom neuron backward "
        "codegen is not implemented",
    )
    generated_backward_plan: NeuronBackwardPlan | None = None

    @property
    def is_valid(self) -> bool:
        return not self.errors

    @property
    def can_use_generated_forward(self) -> bool:
        """Return whether the IR fits the current generic Triton forward ABI."""

        return self.supports_generated_forward

    @property
    def can_use_generated_backward(self) -> bool:
        """Return whether the IR fits a generated backward ABI."""

        return self.supports_generated_backward


class NeuronBuilder:
    """Small helper for authoring custom pointwise neuron IRs.

    The builder records declarations in insertion order while returning the same
    expression nodes used by the lower-level DSL. This keeps custom neurons from
    having to duplicate state/input/parameter name tuples by hand.
    """

    def __init__(self, name: str) -> None:
        if not name:
            raise ValueError("neuron name must be non-empty")
        if not _is_valid_symbol(name):
            raise ValueError(f"neuron name must be a valid identifier: {name}")
        self.name = name
        self._state_names: list[str] = []
        self._input_names: list[str] = []
        self._param_names: list[str] = []

    def state(self, name: str) -> Expr:
        self._declare(name, self._state_names, "state")
        return state(name)

    def input(self, name: str) -> Expr:
        self._declare(name, self._input_names, "input")
        return input_(name)

    def param(self, name: str) -> Expr:
        self._declare(name, self._param_names, "param")
        return param(name)

    def build(
        self,
        *,
        next_state: Mapping[str, Expr],
        outputs: Mapping[str, Expr],
    ) -> NeuronIR:
        """Build a validated ``NeuronIR`` from declared expressions."""

        ir = NeuronIR(
            name=self.name,
            state=tuple(self._state_names),
            params=tuple(self._param_names),
            inputs=tuple(self._input_names),
            next_state=dict(next_state),
            outputs=dict(outputs),
        )
        validate_neuron_ir(ir)
        return ir

    def _declare(self, name: str, names: list[str], kind: str) -> None:
        if not name:
            raise ValueError(f"{kind} name must be non-empty")
        if not _is_valid_symbol(name):
            raise ValueError(f"{kind} name must be a valid identifier: {name}")
        if name in names:
            raise ValueError(f"duplicate {kind} declaration: {name}")
        all_names = set(self._state_names) | set(self._input_names) | set(self._param_names)
        if name in all_names:
            raise ValueError(f"duplicate neuron symbol declaration: {name}")
        names.append(name)


class SurrogateBuilder:
    """Helper for authoring pointwise surrogate derivative IRs.

    Custom surrogate derivatives are expressed as pure pointwise functions of a
    centered membrane value and optional scalar parameters. This mirrors the
    restriction used by generated backward kernels.
    """

    def __init__(self, name: str) -> None:
        if not name:
            raise ValueError("surrogate name must be non-empty")
        if not _is_valid_symbol(name):
            raise ValueError(f"surrogate name must be a valid identifier: {name}")
        self.name = name
        self._param_names: list[str] = []

    def centered(self) -> Expr:
        return input_("centered")

    def param(self, name: str) -> Expr:
        if not name:
            raise ValueError("param name must be non-empty")
        if not _is_valid_symbol(name):
            raise ValueError(f"param name must be a valid identifier: {name}")
        if name == "centered":
            raise ValueError("param name conflicts with surrogate input: centered")
        if name in self._param_names:
            raise ValueError(f"duplicate param declaration: {name}")
        self._param_names.append(name)
        return param(name)

    def build(self, derivative: Expr) -> NeuronIR:
        """Build a validated surrogate derivative IR."""

        ir = NeuronIR(
            name=f"{self.name}_surrogate_derivative",
            state=(),
            params=tuple(self._param_names),
            inputs=("centered",),
            next_state={},
            outputs={"derivative": derivative},
        )
        validate_surrogate_derivative_ir(ir)
        return ir


def const(value: float) -> Expr:
    return Expr("const", value=float(value))


def state(name: str) -> Expr:
    return Expr("state", name=name)


def param(name: str) -> Expr:
    return Expr("param", name=name)


def input_(name: str) -> Expr:
    return Expr("input", name=name)


def add(left: Expr, right: Expr) -> Expr:
    return Expr("add", args=(left, right))


def sub(left: Expr, right: Expr) -> Expr:
    return Expr("sub", args=(left, right))


def mul(left: Expr, right: Expr) -> Expr:
    return Expr("mul", args=(left, right))


def div(left: Expr, right: Expr) -> Expr:
    return Expr("div", args=(left, right))


def abs_(value: Expr) -> Expr:
    return Expr("abs", args=(value,))


def exp(value: Expr) -> Expr:
    return Expr("exp", args=(value,))


def square(value: Expr) -> Expr:
    return Expr("square", args=(value,))


def ge(left: Expr, right: Expr) -> Expr:
    return Expr("ge", args=(left, right))


def lt(left: Expr, right: Expr) -> Expr:
    return Expr("lt", args=(left, right))


def sign(value: Expr) -> Expr:
    return Expr("sign", args=(value,))


def sigmoid(value: Expr) -> Expr:
    return Expr("sigmoid", args=(value,))


def where(condition: Expr, true_value: Expr | float, false_value: Expr | float) -> Expr:
    return Expr("where", args=(condition, as_expr(true_value), as_expr(false_value)))


def as_expr(value: Expr | float) -> Expr:
    if isinstance(value, Expr):
        return value
    return const(value)


def evaluate_expr(
    expr: Expr,
    *,
    state_values: Mapping[str, torch.Tensor],
    input_values: Mapping[str, torch.Tensor],
    param_values: Mapping[str, float],
) -> Value:
    """Evaluate an expression with PyTorch semantics."""

    if expr.kind == "const":
        if expr.value is None:
            raise ValueError("const expression is missing value")
        return expr.value
    if expr.kind == "state":
        if expr.name is None:
            raise ValueError("state expression is missing name")
        return state_values[expr.name]
    if expr.kind == "input":
        if expr.name is None:
            raise ValueError("input expression is missing name")
        return input_values[expr.name]
    if expr.kind == "param":
        if expr.name is None:
            raise ValueError("param expression is missing name")
        return param_values[expr.name]

    values = tuple(
        evaluate_expr(
            arg,
            state_values=state_values,
            input_values=input_values,
            param_values=param_values,
        )
        for arg in expr.args
    )
    if expr.kind == "add":
        return values[0] + values[1]
    if expr.kind == "sub":
        return values[0] - values[1]
    if expr.kind == "mul":
        return values[0] * values[1]
    if expr.kind == "div":
        return values[0] / values[1]
    if expr.kind == "abs":
        if not isinstance(values[0], torch.Tensor):
            raise TypeError("abs operand must evaluate to a tensor")
        return torch.abs(values[0])
    if expr.kind == "exp":
        if not isinstance(values[0], torch.Tensor):
            raise TypeError("exp operand must evaluate to a tensor")
        return torch.exp(values[0])
    if expr.kind == "square":
        return values[0] * values[0]
    if expr.kind == "ge":
        return values[0] >= values[1]
    if expr.kind == "lt":
        return values[0] < values[1]
    if expr.kind == "sign":
        if not isinstance(values[0], torch.Tensor):
            raise TypeError("sign operand must evaluate to a tensor")
        return torch.sign(values[0])
    if expr.kind == "sigmoid":
        if not isinstance(values[0], torch.Tensor):
            raise TypeError("sigmoid operand must evaluate to a tensor")
        return torch.sigmoid(values[0])
    if expr.kind == "where":
        condition, true_value, false_value = values
        if not isinstance(condition, torch.Tensor):
            raise TypeError("where condition must evaluate to a tensor")
        return torch.where(condition, true_value, false_value)
    raise ValueError(f"unsupported expression kind: {expr.kind}")


def evaluate_neuron(
    ir: NeuronIR,
    *,
    state_values: Mapping[str, torch.Tensor],
    input_values: Mapping[str, torch.Tensor],
    param_values: Mapping[str, float],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Evaluate one neuron IR step and return state/output dictionaries."""

    validate_neuron_ir(ir)
    next_state = {
        name: _as_tensor(
            evaluate_expr(
                expr,
                state_values=state_values,
                input_values=input_values,
                param_values=param_values,
            )
        )
        for name, expr in ir.next_state.items()
    }
    outputs = {
        name: _as_tensor(
            evaluate_expr(
                expr,
                state_values=state_values,
                input_values=input_values,
                param_values=param_values,
            )
        )
        for name, expr in ir.outputs.items()
    }
    return next_state, outputs


def evaluate_neuron_unroll(
    ir: NeuronIR,
    inputs: torch.Tensor,
    initial_state: Mapping[str, torch.Tensor],
    param_values: Mapping[str, float],
    *,
    input_name: str | None = None,
    output_name: str = "spike",
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Evaluate a time-major custom neuron IR over ``[T, B, N]`` inputs.

    This is the CPU/PyTorch correctness oracle for user-authored pointwise
    neurons. The v0 helper supports the common single-input/single-spike-output
    case used by the generated Triton forward launcher.
    """

    validate_neuron_ir(ir)
    if inputs.ndim != 3:
        raise ValueError(f"inputs must be shaped [T, B, N]; got {tuple(inputs.shape)}")
    if input_name is None:
        if len(ir.inputs) != 1:
            raise ValueError("input_name is required when IR has multiple inputs")
        input_name = ir.inputs[0]
    if input_name not in ir.inputs:
        raise ValueError(f"input_name is not declared by IR: {input_name}")
    if output_name not in ir.outputs:
        raise ValueError(f"output_name is not defined by IR: {output_name}")

    expected_state_shape = inputs.shape[1:]
    state_values = {}
    for state_name in ir.state:
        if state_name not in initial_state:
            raise ValueError(f"initial_state is missing state tensor: {state_name}")
        state_tensor = initial_state[state_name]
        if state_tensor.shape != expected_state_shape:
            raise ValueError(f"initial_state.{state_name} must match inputs.shape[1:]")
        if state_tensor.device != inputs.device:
            raise ValueError(f"initial_state.{state_name} must be on the same device as inputs")
        state_values[state_name] = state_tensor

    missing_params = set(ir.params) - set(param_values)
    if missing_params:
        raise ValueError(f"param_values is missing values: {sorted(missing_params)}")

    outputs = []
    for input_current in inputs:
        state_values, step_outputs = evaluate_neuron(
            ir,
            state_values=state_values,
            input_values={input_name: input_current},
            param_values=param_values,
        )
        outputs.append(step_outputs[output_name])

    return state_values, torch.stack(outputs)


def evaluate_surrogate_derivative(
    ir: NeuronIR,
    centered: torch.Tensor,
    param_values: Mapping[str, float] | None = None,
) -> torch.Tensor:
    """Evaluate a custom surrogate derivative IR with PyTorch semantics."""

    validate_surrogate_derivative_ir(ir)
    outputs = evaluate_neuron(
        ir,
        state_values={},
        input_values={"centered": centered},
        param_values={} if param_values is None else param_values,
    )[1]
    return outputs["derivative"]


def _as_tensor(value: Value) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError("neuron IR outputs must evaluate to tensors")
    return value


def validate_neuron_ir(ir: NeuronIR) -> None:
    """Validate the structural constraints required by evaluators and codegen."""

    report = analyze_neuron_ir(ir)
    if report.errors:
        raise ValueError(report.errors[0])


def validate_generated_forward_ir(
    ir: NeuronIR,
    *,
    context: str = "generated forward kernels",
) -> None:
    """Validate the current generic generated-forward ABI for custom neurons."""

    report = analyze_neuron_ir(ir)
    if report.errors:
        raise ValueError(report.errors[0])
    if report.generated_forward_errors:
        raise ValueError(f"{context} currently {report.generated_forward_errors[0]}")


def validate_generated_backward_ir(
    ir: NeuronIR,
    *,
    context: str = "generated backward kernels",
) -> None:
    """Validate the current generic generated-backward ABI for custom neurons."""

    report = analyze_neuron_ir(ir)
    if report.errors:
        raise ValueError(report.errors[0])
    if report.generated_backward_errors:
        raise ValueError(f"{context} currently {report.generated_backward_errors[0]}")


def plan_generated_backward_ir(
    ir: NeuronIR,
    *,
    context: str = "generated backward kernels",
    allow_unimplemented: bool = False,
) -> NeuronBackwardPlan:
    """Return the generated-backward plan recognized for a custom neuron IR."""

    report = analyze_neuron_ir(ir)
    if report.errors:
        raise ValueError(report.errors[0])
    if report.generated_backward_plan is not None and (
        allow_unimplemented or report.generated_backward_plan.is_implemented
    ):
        return report.generated_backward_plan
    if report.generated_backward_errors:
        raise ValueError(f"{context} currently {report.generated_backward_errors[0]}")
    raise ValueError(f"{context} did not produce a backward plan")


def validate_surrogate_derivative_ir(ir: NeuronIR) -> None:
    """Validate the v0 custom surrogate derivative ABI."""

    validate_neuron_ir(ir)
    if ir.state:
        raise ValueError("surrogate derivative IR must not declare state")
    if ir.inputs != ("centered",):
        raise ValueError("surrogate derivative IR requires one input named centered")
    if ir.next_state:
        raise ValueError("surrogate derivative IR must not update state")
    if tuple(ir.outputs) != ("derivative",):
        raise ValueError("surrogate derivative IR requires exactly one output named derivative")


def analyze_neuron_ir(ir: NeuronIR) -> NeuronIRValidationReport:
    """Return validation diagnostics without raising.

    ``validate_neuron_ir`` is still the strict gate used by evaluators and
    codegen. This helper is for user-facing checks before attempting codegen or
    module dispatch.
    """

    errors: list[str] = []
    warnings: list[str] = []

    if not ir.name:
        errors.append("neuron IR name must be non-empty")
    elif not _is_valid_symbol(ir.name):
        errors.append(f"neuron IR name must be a valid identifier: {ir.name}")
    if not ir.outputs:
        errors.append("neuron IR must define at least one output")
    _collect_unique_name_errors(ir.state, "state", errors)
    _collect_unique_name_errors(ir.inputs, "input", errors)
    _collect_unique_name_errors(ir.params, "param", errors)
    all_names = (*ir.state, *ir.inputs, *ir.params)
    _collect_unique_name_errors(all_names, "neuron symbol", errors)

    unknown_next_state = set(ir.next_state) - set(ir.state)
    if unknown_next_state:
        names = ", ".join(sorted(unknown_next_state))
        errors.append(f"next_state contains undeclared state names: {names}")
    missing_next_state = set(ir.state) - set(ir.next_state)
    if missing_next_state:
        names = ", ".join(sorted(missing_next_state))
        errors.append(f"next_state is missing state updates: {names}")
    _collect_output_name_errors(ir.outputs, errors)

    _collect_expr_reference_errors(
        ir.next_state,
        state_names=ir.state,
        input_names=ir.inputs,
        param_names=ir.params,
        errors=errors,
    )
    _collect_expr_reference_errors(
        ir.outputs,
        state_names=ir.state,
        input_names=ir.inputs,
        param_names=ir.params,
        errors=errors,
    )
    if len(ir.inputs) != 1:
        warnings.append(
            "generated forward currently supports exactly one input; pass input_name for "
            "PyTorch unroll when multiple inputs are declared"
        )
    if "spike" not in ir.outputs:
        warnings.append("generated forward and TimeUnroll defaults expect an output named 'spike'")
    supports_unroll_api = not errors and len(ir.inputs) == 1 and "spike" in ir.outputs
    generated_forward_errors = _collect_generated_forward_errors(ir)
    supports_generated_forward = not errors and not generated_forward_errors
    generated_backward_errors, generated_backward_plan = _analyze_generated_backward(ir)
    supports_generated_backward = not errors and not generated_backward_errors
    return NeuronIRValidationReport(
        errors=tuple(errors),
        warnings=tuple(warnings),
        supports_unroll_api=supports_unroll_api,
        supports_generated_forward=supports_generated_forward,
        generated_forward_errors=tuple(generated_forward_errors),
        supports_generated_backward=supports_generated_backward,
        generated_backward_errors=tuple(generated_backward_errors),
        generated_backward_plan=None if errors else generated_backward_plan,
    )


def _collect_generated_forward_errors(ir: NeuronIR) -> list[str]:
    errors: list[str] = []
    if not ir.state:
        errors.append("requires at least one state")
    if ir.inputs != ("input_current",):
        errors.append("requires one input named input_current")
    if tuple(ir.outputs) != ("spike",):
        errors.append("requires exactly one output named spike")
    return errors


def _analyze_generated_backward(ir: NeuronIR) -> tuple[list[str], NeuronBackwardPlan | None]:
    alif_errors, alif_plan = _analyze_alif_generated_backward(ir)
    if alif_plan is not None or _is_alif_backward_candidate(ir):
        return alif_errors, alif_plan
    errors, plan = _analyze_lif_generated_backward(ir)
    if plan is not None or _is_lif_backward_candidate(ir):
        return errors, plan
    return errors, plan


def _is_lif_backward_candidate(ir: NeuronIR) -> bool:
    return (
        "membrane" in ir.state
        or "decay" in ir.params
        or "reset" in ir.params
        or "threshold" in ir.params
    )


def _is_alif_backward_candidate(ir: NeuronIR) -> bool:
    return (
        "membrane" in ir.state
        and "adaptation" in ir.state
        and "adaptation_decay" in ir.params
        and "beta" in ir.params
    )


def _analyze_lif_generated_backward(
    ir: NeuronIR,
) -> tuple[list[str], NeuronBackwardPlan | None]:
    errors: list[str] = []
    if ir.state != ("membrane",):
        errors.append("requires exactly one state named membrane")
    if ir.inputs != ("input_current",):
        errors.append("requires one input named input_current")
    if set(ir.params) != {"decay", "threshold", "reset"}:
        errors.append("requires params decay, threshold, and reset")
    if tuple(ir.outputs) != ("spike",):
        errors.append("requires exactly one output named spike")
    if errors:
        return errors, None

    membrane = state("membrane")
    current = input_("input_current")
    decay = param("decay")
    threshold = param("threshold")
    reset = param("reset")
    pre_reset = membrane * decay + current
    spike = pre_reset.ge(threshold)

    next_membrane = ir.next_state.get("membrane")
    output_spike = ir.outputs.get("spike")
    if next_membrane is None or not _expr_equal(next_membrane, where(spike, reset, pre_reset)):
        errors.append(
            "requires hard-reset LIF state update: "
            "membrane = where(pre_reset >= threshold, reset, pre_reset)"
        )
    if output_spike is None or not _expr_equal(output_spike, where(spike, 1.0, 0.0)):
        errors.append("requires spike output: spike = where(pre_reset >= threshold, 1.0, 0.0)")
    if errors:
        return errors, None
    return errors, NeuronBackwardPlan(
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


def _analyze_alif_generated_backward(
    ir: NeuronIR,
) -> tuple[list[str], NeuronBackwardPlan | None]:
    errors: list[str] = []
    if ir.state != ("membrane", "adaptation"):
        errors.append("requires states membrane and adaptation")
    if ir.inputs != ("input_current",):
        errors.append("requires one input named input_current")
    if set(ir.params) != {"decay", "adaptation_decay", "threshold", "reset", "beta"}:
        errors.append("requires params decay, adaptation_decay, threshold, reset, and beta")
    if tuple(ir.outputs) != ("spike",):
        errors.append("requires exactly one output named spike")
    if errors:
        return errors, None

    membrane = state("membrane")
    adaptation = state("adaptation")
    current = input_("input_current")
    decay = param("decay")
    adaptation_decay = param("adaptation_decay")
    threshold = param("threshold")
    reset = param("reset")
    beta = param("beta")
    pre_reset = membrane * decay + current
    adaptive_threshold = threshold + beta * adaptation
    did_spike = pre_reset.ge(adaptive_threshold)
    spike = where(did_spike, 1.0, 0.0)

    next_membrane = ir.next_state.get("membrane")
    next_adaptation = ir.next_state.get("adaptation")
    output_spike = ir.outputs.get("spike")
    if next_membrane is None or not _expr_equal(next_membrane, where(did_spike, reset, pre_reset)):
        errors.append(
            "requires ALIF membrane update: "
            "membrane = where(pre_reset >= threshold + beta * adaptation, reset, pre_reset)"
        )
    if next_adaptation is None or not _expr_equal(
        next_adaptation,
        adaptation * adaptation_decay + spike,
    ):
        errors.append(
            "requires ALIF adaptation update: adaptation = adaptation * adaptation_decay + spike"
        )
    if output_spike is None or not _expr_equal(output_spike, spike):
        errors.append(
            "requires ALIF spike output: "
            "spike = where(pre_reset >= threshold + beta * adaptation, 1.0, 0.0)"
        )
    if errors:
        return errors, None
    return [
        "ALIF generated backward plan is recognized, but its kernel implementation is not "
        "available yet"
    ], NeuronBackwardPlan(
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


def _expr_equal(left: Expr, right: Expr) -> bool:
    if left.kind != right.kind:
        return False
    if left.name != right.name:
        return False
    if left.value != right.value:
        return False
    if len(left.args) != len(right.args):
        return False
    return all(
        _expr_equal(left_arg, right_arg)
        for left_arg, right_arg in zip(left.args, right.args, strict=True)
    )


def _validate_unique_names(names: tuple[str, ...], kind: str) -> None:
    errors: list[str] = []
    _collect_unique_name_errors(names, kind, errors)
    if errors:
        raise ValueError(errors[0])


def _collect_unique_name_errors(names: tuple[str, ...], kind: str, errors: list[str]) -> None:
    seen: set[str] = set()
    for name in names:
        if not name:
            errors.append(f"{kind} name must be non-empty")
        elif not _is_valid_symbol(name):
            errors.append(f"{kind} name must be a valid identifier: {name}")
        elif name in seen:
            errors.append(f"duplicate {kind} declaration: {name}")
        seen.add(name)


def _validate_output_names(outputs: Mapping[str, Expr]) -> None:
    errors: list[str] = []
    _collect_output_name_errors(outputs, errors)
    if errors:
        raise ValueError(errors[0])


def _collect_output_name_errors(outputs: Mapping[str, Expr], errors: list[str]) -> None:
    seen: set[str] = set()
    for name in outputs:
        if not name:
            errors.append("output name must be non-empty")
        elif not _is_valid_symbol(name):
            errors.append(f"output name must be a valid identifier: {name}")
        elif name in seen:
            errors.append(f"duplicate output declaration: {name}")
        seen.add(name)


def _validate_expr_references(
    expressions: Mapping[str, Expr],
    *,
    state_names: tuple[str, ...],
    input_names: tuple[str, ...],
    param_names: tuple[str, ...],
) -> None:
    errors: list[str] = []
    _collect_expr_reference_errors(
        expressions,
        state_names=state_names,
        input_names=input_names,
        param_names=param_names,
        errors=errors,
    )
    if errors:
        raise ValueError(errors[0])


def _collect_expr_reference_errors(
    expressions: Mapping[str, Expr],
    *,
    state_names: tuple[str, ...],
    input_names: tuple[str, ...],
    param_names: tuple[str, ...],
    errors: list[str],
) -> None:
    state_set = set(state_names)
    input_set = set(input_names)
    param_set = set(param_names)
    for output_name, expr in expressions.items():
        try:
            validate_expr(expr)
            references = _iter_expr_references(expr)
        except ValueError as exc:
            errors.append(f"{output_name}: {exc}")
            continue
        for kind, name in references:
            if kind == "state" and name not in state_set:
                errors.append(f"{output_name} references undeclared state: {name}")
            if kind == "input" and name not in input_set:
                errors.append(f"{output_name} references undeclared input: {name}")
            if kind == "param" and name not in param_set:
                errors.append(f"{output_name} references undeclared param: {name}")


def _iter_expr_references(expr: Expr) -> tuple[tuple[ExprKind, str], ...]:
    references: list[tuple[ExprKind, str]] = []
    if expr.kind in {"state", "input", "param"}:
        if expr.name is None:
            raise ValueError(f"{expr.kind} expression is missing name")
        references.append((expr.kind, expr.name))
    for arg in expr.args:
        references.extend(_iter_expr_references(arg))
    return tuple(references)


def validate_expr(expr: Expr) -> None:
    """Validate expression-node structure before evaluation or rendering."""

    if expr.kind == "const":
        if expr.value is None:
            raise ValueError("const expression is missing value")
        if expr.name is not None or expr.args:
            raise ValueError("const expression must not define name or args")
        return

    if expr.kind in {"input", "param", "state"}:
        if expr.name is None:
            raise ValueError(f"{expr.kind} expression is missing name")
        if expr.value is not None or expr.args:
            raise ValueError(f"{expr.kind} expression must not define value or args")
        return

    if expr.name is not None or expr.value is not None:
        raise ValueError(f"{expr.kind} expression must not define name or value")

    expected_arity = _expr_arity(expr.kind)
    if len(expr.args) != expected_arity:
        raise ValueError(
            f"{expr.kind} expression expects {expected_arity} args; got {len(expr.args)}"
        )
    for arg in expr.args:
        validate_expr(arg)


def _expr_arity(kind: ExprKind) -> int:
    if kind in UNARY_EXPR_KINDS:
        return 1
    if kind in BINARY_EXPR_KINDS:
        return 2
    if kind in TERNARY_EXPR_KINDS:
        return 3
    if kind in LEAF_EXPR_KINDS:
        return 0
    raise ValueError(f"unsupported expression kind: {kind}")


def fast_sigmoid_surrogate_derivative_expr(centered: Expr) -> Expr:
    """Build the fast-sigmoid surrogate derivative expression."""

    return 0.5 / (1.0 + centered.abs()).square()


def sigmoid_surrogate_derivative_expr(centered: Expr) -> Expr:
    """Build the sigmoid surrogate derivative expression."""

    value = centered.sigmoid()
    return value * (1.0 - value)


def atan_surrogate_derivative_expr(centered: Expr) -> Expr:
    """Build the ATan surrogate derivative expression."""

    scaled = (torch.pi / 2.0) * centered
    return 0.5 / (1.0 + scaled.square())


def triangular_surrogate_derivative_expr(centered: Expr) -> Expr:
    """Build the triangular surrogate derivative expression."""

    return where(centered.abs().lt(1.0), 0.0 - centered.sign(), 0.0)


def superspike_surrogate_derivative_expr(centered: Expr) -> Expr:
    """Build the SuperSpike surrogate derivative expression."""

    return 1.0 / (1.0 + centered.abs()).square()


def multi_gaussian_surrogate_derivative_expr(centered: Expr) -> Expr:
    """Build a normalized three-component multi-Gaussian derivative expression."""

    inv_sqrt_2pi = 0.3989422804014327
    center = 0.6 * (inv_sqrt_2pi / 0.5) * (-0.5 * (centered / 0.5).square()).exp()
    right = 0.2 * inv_sqrt_2pi * (-0.5 * (centered - 1.0).square()).exp()
    left = 0.2 * inv_sqrt_2pi * (-0.5 * (centered + 1.0).square()).exp()
    return center + right + left


def surrogate_derivative_expr(name: str, centered: Expr) -> Expr:
    """Build a built-in surrogate derivative expression by stable name."""

    if name == "fast_sigmoid":
        return fast_sigmoid_surrogate_derivative_expr(centered)
    if name == "sigmoid":
        return sigmoid_surrogate_derivative_expr(centered)
    if name == "atan":
        return atan_surrogate_derivative_expr(centered)
    if name == "triangular":
        return triangular_surrogate_derivative_expr(centered)
    if name == "superspike":
        return superspike_surrogate_derivative_expr(centered)
    if name == "multi_gaussian":
        return multi_gaussian_surrogate_derivative_expr(centered)
    raise ValueError(f"unsupported surrogate derivative: {name}")


def lif_ir() -> NeuronIR:
    """Build the v0 LIF neuron IR."""

    membrane = state("membrane")
    current = input_("input_current")
    decay = param("decay")
    threshold = param("threshold")
    reset = param("reset")
    pre_reset = membrane * decay + current
    spike = pre_reset.ge(threshold)
    return NeuronIR(
        name="lif",
        state=("membrane",),
        params=("decay", "threshold", "reset"),
        inputs=("input_current",),
        next_state={"membrane": where(spike, reset, pre_reset)},
        outputs={"spike": where(spike, 1.0, 0.0)},
    )


def alif_ir() -> NeuronIR:
    """Build the v0 adaptive LIF neuron IR."""

    membrane = state("membrane")
    adaptation = state("adaptation")
    current = input_("input_current")
    decay = param("decay")
    adaptation_decay = param("adaptation_decay")
    threshold = param("threshold")
    reset = param("reset")
    beta = param("beta")
    pre_reset = membrane * decay + current
    adaptive_threshold = threshold + beta * adaptation
    spike = pre_reset.ge(adaptive_threshold)
    spike_float = where(spike, 1.0, 0.0)
    return NeuronIR(
        name="alif",
        state=("membrane", "adaptation"),
        params=("decay", "adaptation_decay", "threshold", "reset", "beta"),
        inputs=("input_current",),
        next_state={
            "membrane": where(spike, reset, pre_reset),
            "adaptation": adaptation * adaptation_decay + spike_float,
        },
        outputs={"spike": spike_float},
    )


def izhikevich_ir() -> NeuronIR:
    """Build the v0 Izhikevich-style neuron IR."""

    voltage = state("voltage")
    recovery = state("recovery")
    current = input_("input_current")
    recovery_decay = param("recovery_decay")
    recovery_coupling = param("recovery_coupling")
    reset_voltage = param("reset_voltage")
    recovery_jump = param("recovery_jump")
    threshold = param("threshold")
    dt = param("dt")
    voltage_square_coeff = param("voltage_square_coeff")
    voltage_coeff = param("voltage_coeff")
    voltage_bias = param("voltage_bias")

    voltage_delta = (
        voltage_square_coeff * voltage.square()
        + voltage_coeff * voltage
        + voltage_bias
        - recovery
        + current
    )
    pre_reset_voltage = voltage + dt * voltage_delta
    recovery_delta = recovery_decay * (recovery_coupling * voltage - recovery)
    pre_spike_recovery = recovery + dt * recovery_delta
    spike = pre_reset_voltage.ge(threshold)
    spike_float = where(spike, 1.0, 0.0)
    return NeuronIR(
        name="izhikevich",
        state=("voltage", "recovery"),
        params=(
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
        inputs=("input_current",),
        next_state={
            "voltage": where(spike, reset_voltage, pre_reset_voltage),
            "recovery": where(spike, pre_spike_recovery + recovery_jump, pre_spike_recovery),
        },
        outputs={"spike": spike_float},
    )
