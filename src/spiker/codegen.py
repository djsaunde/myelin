"""Small codegen helpers for neuron IR."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal

import torch

from spiker.dsl import (
    Expr,
    NeuronIR,
    input_,
    surrogate_derivative_expr,
    validate_neuron_ir,
    validate_surrogate_derivative_ir,
)

Dialect = Literal["python", "triton"]
SurrogateDerivativeFn = Callable[[torch.Tensor, Mapping[str, float]], torch.Tensor]
SurrogateDerivativeCacheKey = tuple[object, ...]

_SURROGATE_DERIVATIVE_COMPILE_CACHE: dict[
    SurrogateDerivativeCacheKey,
    SurrogateDerivativeFn,
] = {}


@dataclass(frozen=True)
class Assignment:
    """Single static-assignment statement."""

    target: str
    expression: str


@dataclass(frozen=True)
class LoweredNeuron:
    """SSA-like lowering of a neuron IR."""

    name: str
    assignments: tuple[Assignment, ...]
    next_state: dict[str, str]
    outputs: dict[str, str]


def lower_neuron_to_ssa(ir: NeuronIR, *, dialect: Dialect = "python") -> LoweredNeuron:
    """Lower a neuron IR into deterministic SSA-style assignments."""

    validate_neuron_ir(ir)
    assignments: list[Assignment] = []
    cache: dict[Expr, str] = {}

    def lower(expr: Expr) -> str:
        if expr.kind == "const":
            if expr.value is None:
                raise ValueError("const expression is missing value")
            return repr(float(expr.value))
        if expr.kind in {"state", "input", "param"}:
            if expr.name is None:
                raise ValueError(f"{expr.kind} expression is missing name")
            return expr.name
        if expr in cache:
            return cache[expr]

        lowered_args = tuple(lower(arg) for arg in expr.args)
        target = f"tmp{len(assignments)}"
        assignments.append(Assignment(target, render_expr(expr.kind, lowered_args, dialect)))
        cache[expr] = target
        return target

    next_state = {name: lower(expr) for name, expr in ir.next_state.items()}
    outputs = {name: lower(expr) for name, expr in ir.outputs.items()}
    return LoweredNeuron(
        name=ir.name,
        assignments=tuple(assignments),
        next_state=next_state,
        outputs=outputs,
    )


def render_expr(kind: str, args: tuple[str, ...], dialect: Dialect) -> str:
    """Render one non-leaf expression."""

    if kind == "add":
        return f"({args[0]} + {args[1]})"
    if kind == "sub":
        return f"({args[0]} - {args[1]})"
    if kind == "mul":
        return f"({args[0]} * {args[1]})"
    if kind == "div":
        return f"({args[0]} / {args[1]})"
    if kind == "abs":
        abs_name = "torch.abs" if dialect == "python" else "tl.abs"
        return f"{abs_name}({args[0]})"
    if kind == "exp":
        exp_name = "torch.exp" if dialect == "python" else "tl.exp"
        return f"{exp_name}({args[0]})"
    if kind == "square":
        return f"({args[0]} * {args[0]})"
    if kind == "ge":
        return f"({args[0]} >= {args[1]})"
    if kind == "lt":
        return f"({args[0]} < {args[1]})"
    if kind == "sign":
        if dialect == "python":
            return f"torch.sign({args[0]})"
        return f"tl.where({args[0]} > 0.0, 1.0, tl.where({args[0]} < 0.0, -1.0, 0.0))"
    if kind == "sigmoid":
        sigmoid_name = "torch.sigmoid" if dialect == "python" else "tl.sigmoid"
        return f"{sigmoid_name}({args[0]})"
    if kind == "where":
        where_name = "torch.where" if dialect == "python" else "tl.where"
        return f"{where_name}({args[0]}, {args[1]}, {args[2]})"
    raise ValueError(f"unsupported expression kind: {kind}")


def render_lowered_neuron(lowered: LoweredNeuron) -> str:
    """Render lowered assignments plus state/output bindings."""

    lines = [f"{assignment.target} = {assignment.expression}" for assignment in lowered.assignments]
    lines.extend(f"next_{name} = {value}" for name, value in lowered.next_state.items())
    lines.extend(f"out_{name} = {value}" for name, value in lowered.outputs.items())
    return "\n".join(lines)


def render_triton_step_body(
    lowered: LoweredNeuron,
    *,
    variable_map: dict[str, str] | None = None,
    output_map: dict[str, str] | None = None,
    indent: str = "        ",
) -> str:
    """Render a Triton loop-body fragment from lowered neuron IR.

    This produces local assignments only. Callers still own pointer loads/stores
    and loop structure.
    """

    names = {} if variable_map is None else dict(variable_map)
    outputs = {} if output_map is None else dict(output_map)

    def substitute(expression: str) -> str:
        rendered = expression
        for source, target in sorted(names.items(), key=lambda item: len(item[0]), reverse=True):
            rendered = re.sub(rf"\b{re.escape(source)}\b", target, rendered)
        return rendered

    lines = [
        f"{indent}{assignment.target} = {substitute(assignment.expression)}"
        for assignment in lowered.assignments
    ]
    for state_name, value in lowered.next_state.items():
        target = names.get(state_name, state_name)
        lines.append(f"{indent}{target} = {value}")
    for output_name, value in lowered.outputs.items():
        target = outputs.get(output_name, output_name)
        lines.append(f"{indent}{target} = {value}")
    return "\n".join(lines)


def render_surrogate_derivative_body(
    surrogate: str,
    *,
    dialect: Dialect,
    centered_name: str = "centered",
    output_name: str = "d_spike_d_centered",
    indent: str = "",
) -> str:
    """Render a local surrogate derivative fragment for backward kernels."""

    ir = NeuronIR(
        name=f"{surrogate}_surrogate_derivative",
        state=(),
        params=(),
        inputs=("centered",),
        next_state={},
        outputs={"derivative": surrogate_derivative_expr(surrogate, input_("centered"))},
    )
    return render_surrogate_derivative_ir_body(
        ir,
        dialect=dialect,
        centered_name=centered_name,
        output_name=output_name,
        indent=indent,
    )


def render_surrogate_derivative_ir_body(
    ir: NeuronIR,
    *,
    dialect: Dialect,
    centered_name: str = "centered",
    output_name: str = "d_spike_d_centered",
    param_map: dict[str, str] | None = None,
    indent: str = "",
) -> str:
    """Render a local derivative fragment from a custom surrogate derivative IR."""

    validate_surrogate_derivative_ir(ir)
    lowered = lower_neuron_to_ssa(ir, dialect=dialect)
    variable_map = {"centered": centered_name}
    if param_map is not None:
        variable_map.update(param_map)
    return render_triton_step_body(
        lowered,
        variable_map=variable_map,
        output_map={"derivative": output_name},
        indent=indent,
    )


def _expr_cache_key(expr: Expr) -> SurrogateDerivativeCacheKey:
    return (
        expr.kind,
        expr.name,
        expr.value,
        tuple(_expr_cache_key(arg) for arg in expr.args),
    )


def _surrogate_derivative_ir_cache_key(ir: NeuronIR) -> SurrogateDerivativeCacheKey:
    validate_surrogate_derivative_ir(ir)
    return (
        ir.name,
        ir.params,
        tuple(sorted((name, _expr_cache_key(expr)) for name, expr in ir.outputs.items())),
    )


def compile_surrogate_derivative_ir(ir: NeuronIR) -> SurrogateDerivativeFn:
    """Compile a surrogate derivative IR into a reusable Python callable."""

    cache_key = _surrogate_derivative_ir_cache_key(ir)
    cached = _SURROGATE_DERIVATIVE_COMPILE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    param_map = {name: f"params[{name!r}]" for name in ir.params}
    body = render_surrogate_derivative_ir_body(
        ir,
        dialect="python",
        centered_name="centered",
        output_name="derivative",
        param_map=param_map,
        indent="    ",
    )
    source = "def _compiled_surrogate_derivative(centered, params):\n"
    source += body + "\n"
    source += "    return derivative\n"
    namespace = {"torch": torch}
    exec(source, namespace)  # noqa: S102 - generated from validated local IR.
    compiled = namespace["_compiled_surrogate_derivative"]
    if not callable(compiled):
        raise TypeError("compiled surrogate derivative is not callable")
    _SURROGATE_DERIVATIVE_COMPILE_CACHE[cache_key] = compiled
    return compiled


def render_lif_surrogate_backward_step_body(
    surrogate: str,
    *,
    dialect: Dialect,
    pre_reset_name: str = "pre_reset",
    spike_name: str = "spike",
    grad_membrane_name: str = "grad_membrane",
    grad_spike_name: str = "grad_spike",
    threshold_name: str = "threshold",
    reset_name: str = "reset",
    decay_name: str = "decay",
    surrogate_slope_name: str = "surrogate_slope",
    grad_pre_reset_name: str = "grad_pre_reset",
    next_grad_membrane_name: str | None = None,
    indent: str = "",
) -> str:
    """Render one reverse-time surrogate LIF recurrence step."""

    next_grad_membrane = (
        grad_membrane_name if next_grad_membrane_name is None else next_grad_membrane_name
    )
    centered_name = "centered"
    d_centered_name = "d_spike_d_centered"
    d_membrane_name = "d_spike_d_membrane"
    derivative_body = render_surrogate_derivative_body(
        surrogate,
        dialect=dialect,
        centered_name=centered_name,
        output_name=d_centered_name,
        indent=indent,
    )
    lines = [
        f"{indent}{centered_name} = {surrogate_slope_name} * ({pre_reset_name} - {threshold_name})",
        derivative_body,
        f"{indent}{d_membrane_name} = {surrogate_slope_name} * {d_centered_name}",
        f"{indent}{grad_pre_reset_name} = {grad_membrane_name} * "
        f"((1.0 - {spike_name}) + ({reset_name} - {pre_reset_name}) * {d_membrane_name})",
        f"{indent}{grad_pre_reset_name} = {grad_pre_reset_name} + "
        f"{grad_spike_name} * {d_membrane_name}",
        f"{indent}{next_grad_membrane} = {grad_pre_reset_name} * {decay_name}",
    ]
    return "\n".join(lines)
