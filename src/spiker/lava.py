"""Optional Lava builders for spiker hardware export artifacts."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from spiker.hardware import (
    LavaDenseLIFSpec,
    QuantizedDenseLIFHardwareExport,
    read_quantized_hardware_export,
)


@dataclass(frozen=True)
class LavaDenseLIFProcesses:
    """Constructed Lava process pair for one dense LIF export."""

    dense: Any
    lif: Any
    weights: np.ndarray
    spec: LavaDenseLIFSpec
    quantized: QuantizedDenseLIFHardwareExport


def lava_available() -> bool:
    """Return whether Lava's process classes can be imported in this environment."""

    try:
        _load_lava_process_classes()
    except RuntimeError:
        return False
    return True


def build_lava_dense_lif_processes(
    spec: LavaDenseLIFSpec,
    *,
    quantized: QuantizedDenseLIFHardwareExport | None = None,
    base_dir: str | Path = ".",
    name_prefix: str = "spiker_dense_lif",
) -> LavaDenseLIFProcesses:
    """Instantiate Lava ``Dense`` and ``LIF`` processes from a Lava spec.

    Lava remains an optional dependency. If it is not installed, or if the
    installed Lava package is incompatible with the current Python runtime, this
    function raises ``RuntimeError`` with the underlying import failure attached.
    """

    dense_cls, lif_cls = _load_lava_process_classes()
    resolved_quantized = _resolve_quantized_export(spec, quantized, base_dir)
    _validate_spec_matches_quantized(spec, resolved_quantized)

    weights = np.asarray(resolved_quantized.weight, dtype=np.int32).T.copy()
    dense = dense_cls(
        weights=weights,
        name=f"{name_prefix}_dense",
        num_weight_bits=spec.weight_bits,
    )
    lif_params = spec.lava_lif_params
    shape = _expect_shape(lif_params["shape"])
    du = _expect_number(lif_params["du"], "du")
    dv = _expect_number(lif_params["dv"], "dv")
    bias_mant = _expect_int_list(lif_params["bias_mant"], "bias_mant")
    bias_exp = _expect_int(lif_params["bias_exp"], "bias_exp")
    vth = _expect_number(lif_params["vth"], "vth")
    lif = lif_cls(
        shape=tuple(shape),
        du=du,
        dv=dv,
        bias_mant=np.asarray(bias_mant, dtype=np.int32),
        bias_exp=bias_exp,
        vth=vth,
        name=f"{name_prefix}_lif",
    )
    dense.a_out.connect(lif.a_in)
    return LavaDenseLIFProcesses(
        dense=dense,
        lif=lif,
        weights=weights,
        spec=spec,
        quantized=resolved_quantized,
    )


def _load_lava_process_classes() -> tuple[type[Any], type[Any]]:
    try:
        dense_module = importlib.import_module("lava.proc.dense.process")
        lif_module = importlib.import_module("lava.proc.lif.process")
    except Exception as exc:  # noqa: BLE001 - optional integration should preserve root cause.
        raise RuntimeError(
            "Lava process classes could not be imported. Install a Python-compatible "
            "lava-nc package to instantiate Lava processes; the JSON export path does "
            "not require Lava."
        ) from exc

    dense_cls = getattr(dense_module, "Dense", None)
    lif_cls = getattr(lif_module, "LIF", None)
    if dense_cls is None or lif_cls is None:
        raise RuntimeError("Lava installation does not expose Dense and LIF process classes")
    return dense_cls, lif_cls


def _resolve_quantized_export(
    spec: LavaDenseLIFSpec,
    quantized: QuantizedDenseLIFHardwareExport | None,
    base_dir: str | Path,
) -> QuantizedDenseLIFHardwareExport:
    if quantized is not None:
        return quantized
    return read_quantized_hardware_export(Path(base_dir) / spec.quantized_export_path)


def _validate_spec_matches_quantized(
    spec: LavaDenseLIFSpec,
    quantized: QuantizedDenseLIFHardwareExport,
) -> None:
    if spec.input_size != quantized.input_size or spec.output_size != quantized.output_size:
        raise ValueError("Lava spec and quantized export shapes do not match")
    if spec.weight_bits != int(quantized.quantization["num_bits"]):
        raise ValueError("Lava spec weight_bits does not match quantized export num_bits")
    if spec.neuron != quantized.neuron:
        raise ValueError("Lava spec neuron parameters do not match quantized export")


def _expect_shape(value: object) -> list[int]:
    if not isinstance(value, list) or not value or not all(isinstance(item, int) for item in value):
        raise ValueError("Lava spec shape must be a non-empty integer list")
    return value


def _expect_number(value: object, field: str) -> float:
    if not isinstance(value, int | float):
        raise ValueError(f"Lava spec {field} must be a number")
    return float(value)


def _expect_int(value: object, field: str) -> int:
    if not isinstance(value, int):
        raise ValueError(f"Lava spec {field} must be an integer")
    return value


def _expect_int_list(value: object, field: str) -> list[int]:
    if not isinstance(value, list) or not all(isinstance(item, int) for item in value):
        raise ValueError(f"Lava spec {field} must be an integer list")
    return value
