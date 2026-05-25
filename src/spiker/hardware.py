"""Hardware-bridge export helpers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from spiker.neurons import LIFParams

HARDWARE_EXPORT_FORMAT = "spiker.dense_lif.v0"
QUANTIZED_HARDWARE_EXPORT_FORMAT = "spiker.dense_lif_quantized.v0"
PLACEMENT_EXPORT_FORMAT = "spiker.dense_lif_placement.v0"
HARDWARE_BUNDLE_FORMAT = "spiker.hardware_bundle.v0"
LOIHI2_DENSE_LIF_EXPORT_FORMAT = "spiker.loihi2_dense_lif_manifest.v0"
SPINNAKER2_DENSE_LIF_EXPORT_FORMAT = "spiker.spinnaker2_dense_lif_manifest.v0"
LAVA_DENSE_LIF_EXPORT_FORMAT = "spiker.lava_dense_lif_spec.v0"


@dataclass(frozen=True)
class DenseLIFHardwareExport:
    """JSON-serializable dense LIF layer export."""

    format: str
    layer_type: str
    input_size: int
    output_size: int
    weight: list[list[float]]
    bias: list[float] | None
    neuron: dict[str, float]
    timestep: dict[str, float]
    metadata: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


@dataclass(frozen=True)
class QuantizedDenseLIFHardwareExport:
    """JSON-serializable fixed-point dense LIF layer export."""

    format: str
    source_format: str
    layer_type: str
    input_size: int
    output_size: int
    weight: list[list[int]]
    bias: list[int] | None
    neuron: dict[str, float]
    timestep: dict[str, float]
    quantization: dict[str, int | float | str]
    metadata: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


@dataclass(frozen=True)
class DenseLIFPlacementCore:
    """One dense LIF placement tile for a target core."""

    core_id: int
    input_start: int
    input_end: int
    output_start: int
    output_end: int
    input_count: int
    output_count: int
    synapse_count: int
    requires_accumulator: bool


@dataclass(frozen=True)
class DenseLIFPlacementPlan:
    """JSON-serializable dense LIF placement/routing pre-plan."""

    format: str
    source_format: str
    target: str
    input_size: int
    output_size: int
    max_inputs_per_core: int
    max_outputs_per_core: int
    core_count: int
    total_synapse_count: int
    cores: list[DenseLIFPlacementCore]
    metadata: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


@dataclass(frozen=True)
class HardwareExportBundle:
    """Manifest tying together dense LIF hardware export artifacts."""

    format: str
    target: str
    dense_export_path: str
    quantized_export_path: str
    placement_plan_path: str
    formats: dict[str, str]
    summary: dict[str, int]
    metadata: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


@dataclass(frozen=True)
class Loihi2DenseLIFManifest:
    """Loihi 2-oriented manifest for a quantized dense LIF placement bundle."""

    format: str
    source_format: str
    target: str
    quantized_export_path: str
    placement_plan_path: str
    input_size: int
    output_size: int
    core_count: int
    total_synapse_count: int
    compartments_per_core: int
    axons_per_core: int
    weight_bits: int
    timestep_us: float
    neuron: dict[str, float]
    mapping: list[dict[str, int | bool]]
    notes: list[str]
    metadata: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


@dataclass(frozen=True)
class SpiNNaker2DenseLIFManifest:
    """SpiNNaker 2-oriented manifest for a quantized dense LIF placement bundle."""

    format: str
    source_format: str
    target: str
    quantized_export_path: str
    placement_plan_path: str
    input_size: int
    output_size: int
    core_count: int
    total_synapse_count: int
    neurons_per_core: int
    incoming_synapses_per_core: int
    weight_bits: int
    timestep_ms: float
    neuron: dict[str, float]
    mapping: list[dict[str, int | bool]]
    notes: list[str]
    metadata: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


@dataclass(frozen=True)
class LavaDenseLIFSpec:
    """Lava-oriented process spec for a quantized dense LIF layer."""

    format: str
    source_format: str
    target: str
    quantized_export_path: str
    input_size: int
    output_size: int
    weight_bits: int
    timestep_us: float
    process: dict[str, str]
    ports: dict[str, str]
    lava_lif_params: dict[str, float | int | list[int]]
    neuron: dict[str, float]
    quantization: dict[str, int | float | str]
    notes: list[str]
    metadata: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


def dense_lif_hardware_export_from_dict(data: Mapping[str, Any]) -> DenseLIFHardwareExport:
    """Parse and validate a dense LIF hardware export dictionary."""

    if data.get("format") != HARDWARE_EXPORT_FORMAT:
        raise ValueError(f"unsupported hardware export format: {data.get('format')!r}")
    if data.get("layer_type") != "dense_lif":
        raise ValueError(f"unsupported layer_type: {data.get('layer_type')!r}")

    input_size = _expect_int(data, "input_size")
    output_size = _expect_int(data, "output_size")
    weight = _expect_weight(data.get("weight"), input_size, output_size)
    raw_bias = data.get("bias")
    bias = None if raw_bias is None else _expect_vector(raw_bias, output_size, "bias")
    neuron = _expect_float_map(
        data.get("neuron"),
        required_keys=("tau_mem", "decay", "threshold", "reset"),
        field="neuron",
    )
    timestep = _expect_float_map(data.get("timestep"), required_keys=("dt",), field="timestep")
    if timestep["dt"] <= 0:
        raise ValueError("timestep.dt must be positive")
    metadata = _expect_metadata(data.get("metadata", {}))

    return DenseLIFHardwareExport(
        format=HARDWARE_EXPORT_FORMAT,
        layer_type="dense_lif",
        input_size=input_size,
        output_size=output_size,
        weight=weight,
        bias=bias,
        neuron=neuron,
        timestep=timestep,
        metadata=metadata,
    )


def quantized_dense_lif_hardware_export_from_dict(
    data: Mapping[str, Any],
) -> QuantizedDenseLIFHardwareExport:
    """Parse and validate a quantized dense LIF hardware export dictionary."""

    if data.get("format") != QUANTIZED_HARDWARE_EXPORT_FORMAT:
        raise ValueError(f"unsupported quantized hardware export format: {data.get('format')!r}")
    if data.get("source_format") != HARDWARE_EXPORT_FORMAT:
        raise ValueError(f"unsupported source_format: {data.get('source_format')!r}")
    if data.get("layer_type") != "dense_lif":
        raise ValueError(f"unsupported layer_type: {data.get('layer_type')!r}")

    input_size = _expect_int(data, "input_size")
    output_size = _expect_int(data, "output_size")
    quantization = _expect_quantization(data.get("quantization"))
    qmin = int(quantization["qmin"])
    qmax = int(quantization["qmax"])
    weight = _expect_int_matrix(data.get("weight"), input_size, output_size, qmin, qmax)
    raw_bias = data.get("bias")
    bias = (
        None if raw_bias is None else _expect_int_vector(raw_bias, output_size, qmin, qmax, "bias")
    )
    neuron = _expect_float_map(
        data.get("neuron"),
        required_keys=("tau_mem", "decay", "threshold", "reset"),
        field="neuron",
    )
    timestep = _expect_float_map(data.get("timestep"), required_keys=("dt",), field="timestep")
    metadata = _expect_metadata(data.get("metadata", {}))
    return QuantizedDenseLIFHardwareExport(
        format=QUANTIZED_HARDWARE_EXPORT_FORMAT,
        source_format=HARDWARE_EXPORT_FORMAT,
        layer_type="dense_lif",
        input_size=input_size,
        output_size=output_size,
        weight=weight,
        bias=bias,
        neuron=neuron,
        timestep=timestep,
        quantization=quantization,
        metadata=metadata,
    )


def dense_lif_placement_plan_from_dict(data: Mapping[str, Any]) -> DenseLIFPlacementPlan:
    """Parse and validate a dense LIF placement plan dictionary."""

    if data.get("format") != PLACEMENT_EXPORT_FORMAT:
        raise ValueError(f"unsupported placement export format: {data.get('format')!r}")
    if data.get("source_format") != QUANTIZED_HARDWARE_EXPORT_FORMAT:
        raise ValueError(f"unsupported source_format: {data.get('source_format')!r}")
    target = data.get("target")
    if not isinstance(target, str) or not target:
        raise ValueError("target must be a non-empty string")
    input_size = _expect_int(data, "input_size")
    output_size = _expect_int(data, "output_size")
    max_inputs_per_core = _expect_int(data, "max_inputs_per_core")
    max_outputs_per_core = _expect_int(data, "max_outputs_per_core")
    core_count = _expect_int(data, "core_count")
    total_synapse_count = _expect_int(data, "total_synapse_count")
    cores = _expect_placement_cores(
        data.get("cores"),
        input_size=input_size,
        output_size=output_size,
        max_inputs_per_core=max_inputs_per_core,
        max_outputs_per_core=max_outputs_per_core,
    )
    if len(cores) != core_count:
        raise ValueError("core_count must match the number of cores")
    if sum(core.synapse_count for core in cores) != total_synapse_count:
        raise ValueError("total_synapse_count must match core synapse counts")
    metadata = _expect_metadata(data.get("metadata", {}))
    return DenseLIFPlacementPlan(
        format=PLACEMENT_EXPORT_FORMAT,
        source_format=QUANTIZED_HARDWARE_EXPORT_FORMAT,
        target=target,
        input_size=input_size,
        output_size=output_size,
        max_inputs_per_core=max_inputs_per_core,
        max_outputs_per_core=max_outputs_per_core,
        core_count=core_count,
        total_synapse_count=total_synapse_count,
        cores=cores,
        metadata=metadata,
    )


def hardware_export_bundle_from_dict(data: Mapping[str, Any]) -> HardwareExportBundle:
    """Parse and validate a hardware export bundle manifest dictionary."""

    if data.get("format") != HARDWARE_BUNDLE_FORMAT:
        raise ValueError(f"unsupported hardware bundle format: {data.get('format')!r}")
    target = data.get("target")
    if not isinstance(target, str) or not target:
        raise ValueError("target must be a non-empty string")
    dense_export_path = _expect_nonempty_string(data, "dense_export_path")
    quantized_export_path = _expect_nonempty_string(data, "quantized_export_path")
    placement_plan_path = _expect_nonempty_string(data, "placement_plan_path")
    formats = _expect_format_map(data.get("formats"))
    summary = _expect_summary_map(data.get("summary"))
    metadata = _expect_metadata(data.get("metadata", {}))
    return HardwareExportBundle(
        format=HARDWARE_BUNDLE_FORMAT,
        target=target,
        dense_export_path=dense_export_path,
        quantized_export_path=quantized_export_path,
        placement_plan_path=placement_plan_path,
        formats=formats,
        summary=summary,
        metadata=metadata,
    )


def loihi2_dense_lif_manifest_from_dict(data: Mapping[str, Any]) -> Loihi2DenseLIFManifest:
    """Parse and validate a Loihi 2 dense LIF manifest dictionary."""

    if data.get("format") != LOIHI2_DENSE_LIF_EXPORT_FORMAT:
        raise ValueError(f"unsupported Loihi 2 manifest format: {data.get('format')!r}")
    if data.get("source_format") != QUANTIZED_HARDWARE_EXPORT_FORMAT:
        raise ValueError(f"unsupported source_format: {data.get('source_format')!r}")
    target = data.get("target")
    if target != "loihi2":
        raise ValueError("target must be 'loihi2'")
    input_size = _expect_int(data, "input_size")
    output_size = _expect_int(data, "output_size")
    core_count = _expect_int(data, "core_count")
    total_synapse_count = _expect_int(data, "total_synapse_count")
    compartments_per_core = _expect_int(data, "compartments_per_core")
    axons_per_core = _expect_int(data, "axons_per_core")
    weight_bits = _expect_int(data, "weight_bits")
    timestep_us = _expect_positive_number(data, "timestep_us")
    neuron = _expect_float_map(
        data.get("neuron"),
        required_keys=("tau_mem", "decay", "threshold", "reset"),
        field="neuron",
    )
    mapping = _expect_dense_lif_adapter_mapping(
        data.get("mapping"),
        input_size=input_size,
        output_size=output_size,
        core_count=core_count,
    )
    if sum(int(item["synapse_count"]) for item in mapping) != total_synapse_count:
        raise ValueError("total_synapse_count must match mapping synapse counts")
    notes = _expect_string_list(data.get("notes"))
    metadata = _expect_metadata(data.get("metadata", {}))
    return Loihi2DenseLIFManifest(
        format=LOIHI2_DENSE_LIF_EXPORT_FORMAT,
        source_format=QUANTIZED_HARDWARE_EXPORT_FORMAT,
        target="loihi2",
        quantized_export_path=_expect_nonempty_string(data, "quantized_export_path"),
        placement_plan_path=_expect_nonempty_string(data, "placement_plan_path"),
        input_size=input_size,
        output_size=output_size,
        core_count=core_count,
        total_synapse_count=total_synapse_count,
        compartments_per_core=compartments_per_core,
        axons_per_core=axons_per_core,
        weight_bits=weight_bits,
        timestep_us=timestep_us,
        neuron=neuron,
        mapping=mapping,
        notes=notes,
        metadata=metadata,
    )


def spinnaker2_dense_lif_manifest_from_dict(
    data: Mapping[str, Any],
) -> SpiNNaker2DenseLIFManifest:
    """Parse and validate a SpiNNaker 2 dense LIF manifest dictionary."""

    if data.get("format") != SPINNAKER2_DENSE_LIF_EXPORT_FORMAT:
        raise ValueError(f"unsupported SpiNNaker 2 manifest format: {data.get('format')!r}")
    if data.get("source_format") != QUANTIZED_HARDWARE_EXPORT_FORMAT:
        raise ValueError(f"unsupported source_format: {data.get('source_format')!r}")
    target = data.get("target")
    if target != "spinnaker2":
        raise ValueError("target must be 'spinnaker2'")
    input_size = _expect_int(data, "input_size")
    output_size = _expect_int(data, "output_size")
    core_count = _expect_int(data, "core_count")
    total_synapse_count = _expect_int(data, "total_synapse_count")
    neurons_per_core = _expect_int(data, "neurons_per_core")
    incoming_synapses_per_core = _expect_int(data, "incoming_synapses_per_core")
    weight_bits = _expect_int(data, "weight_bits")
    timestep_ms = _expect_positive_number(data, "timestep_ms")
    neuron = _expect_float_map(
        data.get("neuron"),
        required_keys=("tau_mem", "decay", "threshold", "reset"),
        field="neuron",
    )
    mapping = _expect_dense_lif_adapter_mapping(
        data.get("mapping"),
        input_size=input_size,
        output_size=output_size,
        core_count=core_count,
    )
    if sum(int(item["synapse_count"]) for item in mapping) != total_synapse_count:
        raise ValueError("total_synapse_count must match mapping synapse counts")
    notes = _expect_string_list(data.get("notes"))
    metadata = _expect_metadata(data.get("metadata", {}))
    return SpiNNaker2DenseLIFManifest(
        format=SPINNAKER2_DENSE_LIF_EXPORT_FORMAT,
        source_format=QUANTIZED_HARDWARE_EXPORT_FORMAT,
        target="spinnaker2",
        quantized_export_path=_expect_nonempty_string(data, "quantized_export_path"),
        placement_plan_path=_expect_nonempty_string(data, "placement_plan_path"),
        input_size=input_size,
        output_size=output_size,
        core_count=core_count,
        total_synapse_count=total_synapse_count,
        neurons_per_core=neurons_per_core,
        incoming_synapses_per_core=incoming_synapses_per_core,
        weight_bits=weight_bits,
        timestep_ms=timestep_ms,
        neuron=neuron,
        mapping=mapping,
        notes=notes,
        metadata=metadata,
    )


def lava_dense_lif_spec_from_dict(data: Mapping[str, Any]) -> LavaDenseLIFSpec:
    """Parse and validate a Lava dense LIF process spec dictionary."""

    if data.get("format") != LAVA_DENSE_LIF_EXPORT_FORMAT:
        raise ValueError(f"unsupported Lava spec format: {data.get('format')!r}")
    if data.get("source_format") != QUANTIZED_HARDWARE_EXPORT_FORMAT:
        raise ValueError(f"unsupported source_format: {data.get('source_format')!r}")
    target = data.get("target")
    if target != "lava":
        raise ValueError("target must be 'lava'")
    input_size = _expect_int(data, "input_size")
    output_size = _expect_int(data, "output_size")
    weight_bits = _expect_int(data, "weight_bits")
    timestep_us = _expect_positive_number(data, "timestep_us")
    process = _expect_string_map(
        data.get("process"),
        required_keys=("dense_class", "lif_class", "run_config_hint"),
        field="process",
    )
    ports = _expect_string_map(
        data.get("ports"),
        required_keys=("dense_input", "dense_output", "lif_input", "lif_output"),
        field="ports",
    )
    lava_lif_params = _expect_lava_lif_params(data.get("lava_lif_params"), output_size)
    neuron = _expect_float_map(
        data.get("neuron"),
        required_keys=("tau_mem", "decay", "threshold", "reset"),
        field="neuron",
    )
    quantization = _expect_quantization(data.get("quantization"))
    notes = _expect_string_list(data.get("notes"))
    metadata = _expect_metadata(data.get("metadata", {}))
    return LavaDenseLIFSpec(
        format=LAVA_DENSE_LIF_EXPORT_FORMAT,
        source_format=QUANTIZED_HARDWARE_EXPORT_FORMAT,
        target="lava",
        quantized_export_path=_expect_nonempty_string(data, "quantized_export_path"),
        input_size=input_size,
        output_size=output_size,
        weight_bits=weight_bits,
        timestep_us=timestep_us,
        process=process,
        ports=ports,
        lava_lif_params=lava_lif_params,
        neuron=neuron,
        quantization=quantization,
        notes=notes,
        metadata=metadata,
    )


def read_hardware_export(path: str | Path) -> DenseLIFHardwareExport:
    """Read a dense LIF hardware export JSON artifact."""

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError("hardware export JSON must contain an object")
    return dense_lif_hardware_export_from_dict(data)


def read_quantized_hardware_export(path: str | Path) -> QuantizedDenseLIFHardwareExport:
    """Read a quantized dense LIF hardware export JSON artifact."""

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError("quantized hardware export JSON must contain an object")
    return quantized_dense_lif_hardware_export_from_dict(data)


def read_dense_lif_placement_plan(path: str | Path) -> DenseLIFPlacementPlan:
    """Read a dense LIF placement plan JSON artifact."""

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError("placement plan JSON must contain an object")
    return dense_lif_placement_plan_from_dict(data)


def read_hardware_export_bundle(path: str | Path) -> HardwareExportBundle:
    """Read a hardware export bundle manifest JSON artifact."""

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError("hardware export bundle JSON must contain an object")
    return hardware_export_bundle_from_dict(data)


def read_loihi2_dense_lif_manifest(path: str | Path) -> Loihi2DenseLIFManifest:
    """Read a Loihi 2 dense LIF manifest JSON artifact."""

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError("Loihi 2 manifest JSON must contain an object")
    return loihi2_dense_lif_manifest_from_dict(data)


def read_spinnaker2_dense_lif_manifest(path: str | Path) -> SpiNNaker2DenseLIFManifest:
    """Read a SpiNNaker 2 dense LIF manifest JSON artifact."""

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError("SpiNNaker 2 manifest JSON must contain an object")
    return spinnaker2_dense_lif_manifest_from_dict(data)


def read_lava_dense_lif_spec(path: str | Path) -> LavaDenseLIFSpec:
    """Read a Lava dense LIF process spec JSON artifact."""

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError("Lava spec JSON must contain an object")
    return lava_dense_lif_spec_from_dict(data)


def export_dense_lif_layer(
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    params: LIFParams,
    *,
    dt: float = 1.0,
    metadata: dict[str, str] | None = None,
) -> DenseLIFHardwareExport:
    """Export dense weights and LIF params to a portable hardware-bridge schema."""

    _validate_dense_lif_export_inputs(weight, bias, dt)
    detached_weight = weight.detach().cpu().to(torch.float32)
    detached_bias = None if bias is None else bias.detach().cpu().to(torch.float32)
    input_size, output_size = detached_weight.shape
    return DenseLIFHardwareExport(
        format=HARDWARE_EXPORT_FORMAT,
        layer_type="dense_lif",
        input_size=input_size,
        output_size=output_size,
        weight=_tensor_2d_to_list(detached_weight),
        bias=None if detached_bias is None else _tensor_1d_to_list(detached_bias),
        neuron={
            "tau_mem": float(params.tau_mem),
            "decay": float(params.decay),
            "threshold": float(params.threshold),
            "reset": float(params.reset),
        },
        timestep={"dt": float(dt)},
        metadata=dict(metadata or {}),
    )


def plan_dense_lif_placement(
    export: QuantizedDenseLIFHardwareExport,
    *,
    max_inputs_per_core: int,
    max_outputs_per_core: int,
    target: str = "generic",
    metadata: dict[str, str] | None = None,
) -> DenseLIFPlacementPlan:
    """Tile a quantized dense LIF export into target-core placement ranges."""

    if export.format != QUANTIZED_HARDWARE_EXPORT_FORMAT:
        raise ValueError("export must be a quantized dense LIF hardware export")
    if max_inputs_per_core <= 0:
        raise ValueError("max_inputs_per_core must be positive")
    if max_outputs_per_core <= 0:
        raise ValueError("max_outputs_per_core must be positive")
    if not target:
        raise ValueError("target must be a non-empty string")

    input_ranges = _partition_range(export.input_size, max_inputs_per_core)
    output_ranges = _partition_range(export.output_size, max_outputs_per_core)
    requires_accumulator = len(input_ranges) > 1
    cores = []
    core_id = 0
    for output_start, output_end in output_ranges:
        for input_start, input_end in input_ranges:
            input_count = input_end - input_start
            output_count = output_end - output_start
            cores.append(
                DenseLIFPlacementCore(
                    core_id=core_id,
                    input_start=input_start,
                    input_end=input_end,
                    output_start=output_start,
                    output_end=output_end,
                    input_count=input_count,
                    output_count=output_count,
                    synapse_count=input_count * output_count,
                    requires_accumulator=requires_accumulator,
                )
            )
            core_id += 1

    return DenseLIFPlacementPlan(
        format=PLACEMENT_EXPORT_FORMAT,
        source_format=export.format,
        target=target,
        input_size=export.input_size,
        output_size=export.output_size,
        max_inputs_per_core=max_inputs_per_core,
        max_outputs_per_core=max_outputs_per_core,
        core_count=len(cores),
        total_synapse_count=sum(core.synapse_count for core in cores),
        cores=cores,
        metadata=dict(metadata or export.metadata),
    )


def export_linear_lif_module(
    module: torch.nn.Module,
    *,
    dt: float = 1.0,
    metadata: dict[str, str] | None = None,
) -> DenseLIFHardwareExport:
    """Export a ``LinearLIF``-compatible module to the dense LIF schema."""

    synapse = getattr(module, "synapse", None)
    unroll = getattr(module, "unroll", None)
    cell = None if unroll is None else getattr(unroll, "cell", None)
    params = None if cell is None else getattr(cell, "params", None)
    weight = None if synapse is None else getattr(synapse, "weight", None)
    bias = None if synapse is None else getattr(synapse, "bias", None)
    if not isinstance(weight, torch.Tensor) or not isinstance(params, LIFParams):
        raise TypeError("module must expose LinearLIF-style synapse weights and LIF params")
    if bias is not None and not isinstance(bias, torch.Tensor):
        raise TypeError("module bias must be a tensor or None")
    return export_dense_lif_layer(weight, bias, params, dt=dt, metadata=metadata)


def export_linear_lif_hardware_bundle(
    module: torch.nn.Module,
    directory: str | Path,
    *,
    dt: float = 1.0,
    num_bits: int = 8,
    max_inputs_per_core: int,
    max_outputs_per_core: int,
    target: str = "generic",
    prefix: str = "layer",
    metadata: dict[str, str] | None = None,
    indent: int | None = 2,
) -> HardwareExportBundle:
    """Write float, quantized, placement, and manifest artifacts for a LinearLIF layer."""

    if not prefix:
        raise ValueError("prefix must be a non-empty string")
    output_dir = Path(directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    dense_path = f"{prefix}.dense_lif.json"
    quantized_path = f"{prefix}.dense_lif_quantized.json"
    placement_path = f"{prefix}.dense_lif_placement.json"
    manifest_path = f"{prefix}.hardware_bundle.json"

    dense = export_linear_lif_module(module, dt=dt, metadata=metadata)
    quantized = quantize_dense_lif_export(dense, num_bits=num_bits)
    placement = plan_dense_lif_placement(
        quantized,
        max_inputs_per_core=max_inputs_per_core,
        max_outputs_per_core=max_outputs_per_core,
        target=target,
        metadata=metadata,
    )
    bundle = HardwareExportBundle(
        format=HARDWARE_BUNDLE_FORMAT,
        target=target,
        dense_export_path=dense_path,
        quantized_export_path=quantized_path,
        placement_plan_path=placement_path,
        formats={
            "dense_export": dense.format,
            "quantized_export": quantized.format,
            "placement_plan": placement.format,
        },
        summary={
            "input_size": dense.input_size,
            "output_size": dense.output_size,
            "num_bits": num_bits,
            "core_count": placement.core_count,
            "total_synapse_count": placement.total_synapse_count,
        },
        metadata=dict(metadata or {}),
    )

    write_hardware_export(dense, output_dir / dense_path, indent=indent)
    write_quantized_hardware_export(quantized, output_dir / quantized_path, indent=indent)
    write_dense_lif_placement_plan(placement, output_dir / placement_path, indent=indent)
    write_hardware_export_bundle(bundle, output_dir / manifest_path, indent=indent)
    return bundle


def export_loihi2_dense_lif_manifest(
    quantized: QuantizedDenseLIFHardwareExport,
    placement: DenseLIFPlacementPlan,
    *,
    quantized_export_path: str,
    placement_plan_path: str,
    compartments_per_core: int,
    axons_per_core: int,
    metadata: dict[str, str] | None = None,
) -> Loihi2DenseLIFManifest:
    """Build a Loihi 2-oriented manifest from quantized weights and placement.

    This is an adapter artifact, not an NxSDK/NxCore program. It preserves the
    validated generic exports while making the target constraints explicit for a
    later SDK-specific lowering pass.
    """

    if quantized.format != QUANTIZED_HARDWARE_EXPORT_FORMAT:
        raise ValueError("quantized must be a quantized dense LIF hardware export")
    if placement.format != PLACEMENT_EXPORT_FORMAT:
        raise ValueError("placement must be a dense LIF placement plan")
    if placement.source_format != quantized.format:
        raise ValueError("placement source_format must match quantized export format")
    if placement.target != "loihi2":
        raise ValueError("placement target must be 'loihi2'")
    if (
        quantized.input_size != placement.input_size
        or quantized.output_size != placement.output_size
    ):
        raise ValueError("quantized export and placement shape must match")
    if compartments_per_core <= 0:
        raise ValueError("compartments_per_core must be positive")
    if axons_per_core <= 0:
        raise ValueError("axons_per_core must be positive")
    for core in placement.cores:
        if core.output_count > compartments_per_core:
            raise ValueError("placement core output_count exceeds compartments_per_core")
        if core.input_count > axons_per_core:
            raise ValueError("placement core input_count exceeds axons_per_core")

    timestep_us = float(quantized.timestep["dt"]) * 1_000_000.0
    return Loihi2DenseLIFManifest(
        format=LOIHI2_DENSE_LIF_EXPORT_FORMAT,
        source_format=quantized.format,
        target="loihi2",
        quantized_export_path=quantized_export_path,
        placement_plan_path=placement_plan_path,
        input_size=quantized.input_size,
        output_size=quantized.output_size,
        core_count=placement.core_count,
        total_synapse_count=placement.total_synapse_count,
        compartments_per_core=compartments_per_core,
        axons_per_core=axons_per_core,
        weight_bits=int(quantized.quantization["num_bits"]),
        timestep_us=timestep_us,
        neuron=dict(quantized.neuron),
        mapping=[
            {
                "core_id": core.core_id,
                "input_start": core.input_start,
                "input_end": core.input_end,
                "output_start": core.output_start,
                "output_end": core.output_end,
                "synapse_count": core.synapse_count,
                "requires_accumulator": core.requires_accumulator,
            }
            for core in placement.cores
        ],
        notes=[
            "Adapter manifest only; SDK-specific Loihi 2 objects are not emitted.",
            "Weights and bias are in the referenced signed-symmetric quantized dense LIF export.",
        ],
        metadata=dict(metadata or quantized.metadata),
    )


def export_spinnaker2_dense_lif_manifest(
    quantized: QuantizedDenseLIFHardwareExport,
    placement: DenseLIFPlacementPlan,
    *,
    quantized_export_path: str,
    placement_plan_path: str,
    neurons_per_core: int,
    incoming_synapses_per_core: int,
    metadata: dict[str, str] | None = None,
) -> SpiNNaker2DenseLIFManifest:
    """Build a SpiNNaker 2-oriented manifest from quantized weights and placement.

    This adapter artifact preserves the generic export and placement contracts
    while making SpiNNaker-style per-core neuron and synapse constraints
    explicit for a later SDK-specific lowering pass.
    """

    if quantized.format != QUANTIZED_HARDWARE_EXPORT_FORMAT:
        raise ValueError("quantized must be a quantized dense LIF hardware export")
    if placement.format != PLACEMENT_EXPORT_FORMAT:
        raise ValueError("placement must be a dense LIF placement plan")
    if placement.source_format != quantized.format:
        raise ValueError("placement source_format must match quantized export format")
    if placement.target != "spinnaker2":
        raise ValueError("placement target must be 'spinnaker2'")
    if (
        quantized.input_size != placement.input_size
        or quantized.output_size != placement.output_size
    ):
        raise ValueError("quantized export and placement shape must match")
    if neurons_per_core <= 0:
        raise ValueError("neurons_per_core must be positive")
    if incoming_synapses_per_core <= 0:
        raise ValueError("incoming_synapses_per_core must be positive")
    for core in placement.cores:
        if core.output_count > neurons_per_core:
            raise ValueError("placement core output_count exceeds neurons_per_core")
        if core.synapse_count > incoming_synapses_per_core:
            raise ValueError("placement core synapse_count exceeds incoming_synapses_per_core")

    timestep_ms = float(quantized.timestep["dt"]) * 1_000.0
    return SpiNNaker2DenseLIFManifest(
        format=SPINNAKER2_DENSE_LIF_EXPORT_FORMAT,
        source_format=quantized.format,
        target="spinnaker2",
        quantized_export_path=quantized_export_path,
        placement_plan_path=placement_plan_path,
        input_size=quantized.input_size,
        output_size=quantized.output_size,
        core_count=placement.core_count,
        total_synapse_count=placement.total_synapse_count,
        neurons_per_core=neurons_per_core,
        incoming_synapses_per_core=incoming_synapses_per_core,
        weight_bits=int(quantized.quantization["num_bits"]),
        timestep_ms=timestep_ms,
        neuron=dict(quantized.neuron),
        mapping=[
            {
                "core_id": core.core_id,
                "input_start": core.input_start,
                "input_end": core.input_end,
                "output_start": core.output_start,
                "output_end": core.output_end,
                "synapse_count": core.synapse_count,
                "requires_accumulator": core.requires_accumulator,
            }
            for core in placement.cores
        ],
        notes=[
            "Adapter manifest only; SDK-specific SpiNNaker 2 objects are not emitted.",
            "Weights and bias are in the referenced signed-symmetric quantized dense LIF export.",
            "Dense input shards that require accumulation need target-side partial-sum handling.",
        ],
        metadata=dict(metadata or quantized.metadata),
    )


def export_lava_dense_lif_spec(
    quantized: QuantizedDenseLIFHardwareExport,
    *,
    quantized_export_path: str,
    metadata: dict[str, str] | None = None,
    require_zero_reset: bool = True,
) -> LavaDenseLIFSpec:
    """Build a Lava ``Dense`` + ``LIF`` process spec from a quantized dense LIF export.

    The spec is intentionally a Lava-facing handoff artifact, not an imported
    Lava object. This keeps Lava optional while making the exact process classes,
    ports, and parameter mapping explicit for a later executable builder.
    """

    if quantized.format != QUANTIZED_HARDWARE_EXPORT_FORMAT:
        raise ValueError("quantized must be a quantized dense LIF hardware export")
    reset = float(quantized.neuron["reset"])
    if require_zero_reset and reset != 0.0:
        raise ValueError("Lava stock LIF resets voltage to zero; export requires reset == 0.0")

    tau_mem = float(quantized.neuron["tau_mem"])
    threshold = float(quantized.neuron["threshold"])
    bias = [0 for _ in range(quantized.output_size)] if quantized.bias is None else quantized.bias
    timestep_us = float(quantized.timestep["dt"]) * 1_000_000.0
    return LavaDenseLIFSpec(
        format=LAVA_DENSE_LIF_EXPORT_FORMAT,
        source_format=quantized.format,
        target="lava",
        quantized_export_path=quantized_export_path,
        input_size=quantized.input_size,
        output_size=quantized.output_size,
        weight_bits=int(quantized.quantization["num_bits"]),
        timestep_us=timestep_us,
        process={
            "dense_class": "lava.proc.dense.process.Dense",
            "lif_class": "lava.proc.lif.process.LIF",
            "run_config_hint": (
                "Loihi2SimCfg or Loihi2HwCfg when available; CPU simulation otherwise."
            ),
        },
        ports={
            "dense_input": "Dense.s_in",
            "dense_output": "Dense.a_out",
            "lif_input": "LIF.a_in",
            "lif_output": "LIF.s_out",
        },
        lava_lif_params={
            "shape": [quantized.output_size],
            "du": 1.0,
            "dv": 1.0 / tau_mem,
            "bias_mant": bias,
            "bias_exp": 0,
            "vth": threshold,
        },
        neuron=dict(quantized.neuron),
        quantization=dict(quantized.quantization),
        notes=[
            (
                "Adapter spec only; executable Lava processes are not constructed unless "
                "Lava is installed."
            ),
            "Dense weights and optional bias are in the referenced quantized export.",
            "The mapping uses Lava LIF du=1.0 so current does not carry state between timesteps.",
            "The mapping uses dv=1/tau_mem to match spiker's membrane decay convention.",
            "Stock Lava LIF resets voltage to zero, so exact export requires reset == 0.0.",
        ],
        metadata=dict(metadata or quantized.metadata),
    )


def write_loihi2_dense_lif_manifest(
    manifest: Loihi2DenseLIFManifest,
    path: str | Path,
    *,
    indent: int | None = 2,
) -> None:
    """Write a Loihi 2 dense LIF manifest JSON artifact."""

    Path(path).write_text(manifest.to_json(indent=indent) + "\n", encoding="utf-8")


def write_spinnaker2_dense_lif_manifest(
    manifest: SpiNNaker2DenseLIFManifest,
    path: str | Path,
    *,
    indent: int | None = 2,
) -> None:
    """Write a SpiNNaker 2 dense LIF manifest JSON artifact."""

    Path(path).write_text(manifest.to_json(indent=indent) + "\n", encoding="utf-8")


def write_lava_dense_lif_spec(
    spec: LavaDenseLIFSpec,
    path: str | Path,
    *,
    indent: int | None = 2,
) -> None:
    """Write a Lava dense LIF process spec JSON artifact."""

    Path(path).write_text(spec.to_json(indent=indent) + "\n", encoding="utf-8")


def quantize_dense_lif_export(
    export: DenseLIFHardwareExport,
    *,
    num_bits: int = 8,
) -> QuantizedDenseLIFHardwareExport:
    """Convert a dense LIF float export to signed symmetric fixed-point integers."""

    qmin, qmax = _signed_symmetric_range(num_bits)
    weight_tensor = torch.tensor(export.weight, dtype=torch.float32)
    weight_scale = _symmetric_scale(weight_tensor, qmax)
    quantized_weight = _quantize_tensor(weight_tensor, scale=weight_scale, qmin=qmin, qmax=qmax)

    bias_scale = weight_scale
    quantized_bias = None
    if export.bias is not None:
        bias_tensor = torch.tensor(export.bias, dtype=torch.float32)
        bias_scale = _symmetric_scale(bias_tensor, qmax)
        quantized_bias = _quantize_tensor(bias_tensor, scale=bias_scale, qmin=qmin, qmax=qmax)

    return QuantizedDenseLIFHardwareExport(
        format=QUANTIZED_HARDWARE_EXPORT_FORMAT,
        source_format=export.format,
        layer_type=export.layer_type,
        input_size=export.input_size,
        output_size=export.output_size,
        weight=_int_tensor_2d_to_list(quantized_weight),
        bias=None if quantized_bias is None else _int_tensor_1d_to_list(quantized_bias),
        neuron=dict(export.neuron),
        timestep=dict(export.timestep),
        quantization={
            "scheme": "signed_symmetric_per_tensor",
            "num_bits": num_bits,
            "qmin": qmin,
            "qmax": qmax,
            "weight_scale": weight_scale,
            "bias_scale": bias_scale,
        },
        metadata=dict(export.metadata),
    )


def dequantize_dense_lif_export(
    export: QuantizedDenseLIFHardwareExport,
) -> DenseLIFHardwareExport:
    """Convert a quantized dense LIF export back to floating point values."""

    weight_scale = float(export.quantization["weight_scale"])
    bias_scale = float(export.quantization["bias_scale"])
    weight = (torch.tensor(export.weight, dtype=torch.float32) * weight_scale).tolist()
    bias = None
    if export.bias is not None:
        bias = (torch.tensor(export.bias, dtype=torch.float32) * bias_scale).tolist()
    return DenseLIFHardwareExport(
        format=HARDWARE_EXPORT_FORMAT,
        layer_type=export.layer_type,
        input_size=export.input_size,
        output_size=export.output_size,
        weight=_nested_float_list(weight),
        bias=None if bias is None else [float(value) for value in bias],
        neuron=dict(export.neuron),
        timestep=dict(export.timestep),
        metadata=dict(export.metadata),
    )


def write_hardware_export(
    export: DenseLIFHardwareExport,
    path: str | Path,
    *,
    indent: int | None = 2,
) -> None:
    """Write a hardware export JSON artifact."""

    Path(path).write_text(export.to_json(indent=indent) + "\n", encoding="utf-8")


def write_quantized_hardware_export(
    export: QuantizedDenseLIFHardwareExport,
    path: str | Path,
    *,
    indent: int | None = 2,
) -> None:
    """Write a quantized hardware export JSON artifact."""

    Path(path).write_text(export.to_json(indent=indent) + "\n", encoding="utf-8")


def write_dense_lif_placement_plan(
    plan: DenseLIFPlacementPlan,
    path: str | Path,
    *,
    indent: int | None = 2,
) -> None:
    """Write a dense LIF placement plan JSON artifact."""

    Path(path).write_text(plan.to_json(indent=indent) + "\n", encoding="utf-8")


def write_hardware_export_bundle(
    bundle: HardwareExportBundle,
    path: str | Path,
    *,
    indent: int | None = 2,
) -> None:
    """Write a hardware export bundle manifest JSON artifact."""

    Path(path).write_text(bundle.to_json(indent=indent) + "\n", encoding="utf-8")


def _validate_dense_lif_export_inputs(
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    dt: float,
) -> None:
    if weight.ndim != 2:
        raise ValueError(f"weight must be shaped [input, output]; got {tuple(weight.shape)}")
    if weight.shape[0] <= 0 or weight.shape[1] <= 0:
        raise ValueError("weight must have non-empty input and output dimensions")
    if not weight.is_floating_point():
        raise ValueError("weight must be floating point")
    if not torch.isfinite(weight).all():
        raise ValueError("weight must contain only finite values")
    if bias is not None:
        if bias.shape != (weight.shape[1],):
            raise ValueError("bias must be shaped [output]")
        if not bias.is_floating_point():
            raise ValueError("bias must be floating point")
        if not torch.isfinite(bias).all():
            raise ValueError("bias must contain only finite values")
    if dt <= 0:
        raise ValueError("dt must be positive")


def _tensor_2d_to_list(tensor: torch.Tensor) -> list[list[float]]:
    return [[float(value) for value in row] for row in tensor.tolist()]


def _tensor_1d_to_list(tensor: torch.Tensor) -> list[float]:
    return [float(value) for value in tensor.tolist()]


def _int_tensor_2d_to_list(tensor: torch.Tensor) -> list[list[int]]:
    return [[int(value) for value in row] for row in tensor.tolist()]


def _int_tensor_1d_to_list(tensor: torch.Tensor) -> list[int]:
    return [int(value) for value in tensor.tolist()]


def _nested_float_list(value: list[list[float]]) -> list[list[float]]:
    return [[float(item) for item in row] for row in value]


def _signed_symmetric_range(num_bits: int) -> tuple[int, int]:
    if num_bits < 2 or num_bits > 16:
        raise ValueError("num_bits must be between 2 and 16")
    qmax = (2 ** (num_bits - 1)) - 1
    return -qmax, qmax


def _symmetric_scale(tensor: torch.Tensor, qmax: int) -> float:
    max_abs = float(tensor.abs().max().item())
    if max_abs == 0.0:
        return 1.0
    return max_abs / qmax


def _quantize_tensor(
    tensor: torch.Tensor,
    *,
    scale: float,
    qmin: int,
    qmax: int,
) -> torch.Tensor:
    return torch.clamp(torch.round(tensor / scale), min=qmin, max=qmax).to(torch.int32)


def _partition_range(size: int, chunk_size: int) -> list[tuple[int, int]]:
    return [(start, min(start + chunk_size, size)) for start in range(0, size, chunk_size)]


def _expect_int(data: Mapping[str, Any], key: str) -> int:
    value = data.get(key)
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{key} must be a positive integer")
    return value


def _expect_weight(value: object, input_size: int, output_size: int) -> list[list[float]]:
    if not isinstance(value, list) or len(value) != input_size:
        raise ValueError("weight must be a list shaped [input_size, output_size]")
    return [_expect_vector(row, output_size, "weight row") for row in value]


def _expect_vector(value: object, size: int, field: str) -> list[float]:
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f"{field} must be a list of length {size}")
    vector = []
    for item in value:
        if not isinstance(item, int | float):
            raise ValueError(f"{field} must contain only numbers")
        vector.append(float(item))
    return vector


def _expect_float_map(
    value: object,
    *,
    required_keys: tuple[str, ...],
    field: str,
) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    result: dict[str, float] = {}
    for key in required_keys:
        item = value.get(key)
        if not isinstance(item, int | float):
            raise ValueError(f"{field}.{key} must be a number")
        result[key] = float(item)
    return result


def _expect_string_map(
    value: object,
    *,
    required_keys: tuple[str, ...],
    field: str,
) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    result: dict[str, str] = {}
    for key in required_keys:
        item = value.get(key)
        if not isinstance(item, str) or not item:
            raise ValueError(f"{field}.{key} must be a non-empty string")
        result[key] = item
    return result


def _expect_lava_lif_params(value: object, output_size: int) -> dict[str, float | int | list[int]]:
    if not isinstance(value, Mapping):
        raise ValueError("lava_lif_params must be an object")
    shape = value.get("shape")
    if not isinstance(shape, list) or shape != [output_size]:
        raise ValueError("lava_lif_params.shape must match [output_size]")
    du = value.get("du")
    dv = value.get("dv")
    bias_exp = value.get("bias_exp")
    vth = value.get("vth")
    bias_mant = value.get("bias_mant")
    if not isinstance(du, int | float) or du < 0:
        raise ValueError("lava_lif_params.du must be a non-negative number")
    if not isinstance(dv, int | float) or dv < 0:
        raise ValueError("lava_lif_params.dv must be a non-negative number")
    if not isinstance(bias_exp, int):
        raise ValueError("lava_lif_params.bias_exp must be an integer")
    if not isinstance(vth, int | float):
        raise ValueError("lava_lif_params.vth must be a number")
    if not isinstance(bias_mant, list) or len(bias_mant) != output_size:
        raise ValueError("lava_lif_params.bias_mant must be a list of length output_size")
    parsed_bias = []
    for item in bias_mant:
        if not isinstance(item, int):
            raise ValueError("lava_lif_params.bias_mant must contain only integers")
        parsed_bias.append(item)
    return {
        "shape": [output_size],
        "du": float(du),
        "dv": float(dv),
        "bias_mant": parsed_bias,
        "bias_exp": bias_exp,
        "vth": float(vth),
    }


def _expect_metadata(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError("metadata must be an object")
    metadata: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not isinstance(item, str):
            raise ValueError("metadata must contain only string keys and values")
        metadata[key] = item
    return metadata


def _expect_quantization(value: object) -> dict[str, int | float | str]:
    if not isinstance(value, Mapping):
        raise ValueError("quantization must be an object")
    scheme = value.get("scheme")
    if scheme != "signed_symmetric_per_tensor":
        raise ValueError(f"unsupported quantization scheme: {scheme!r}")
    num_bits = value.get("num_bits")
    qmin = value.get("qmin")
    qmax = value.get("qmax")
    weight_scale = value.get("weight_scale")
    bias_scale = value.get("bias_scale")
    if not isinstance(num_bits, int):
        raise ValueError("quantization.num_bits must be an integer")
    expected_qmin, expected_qmax = _signed_symmetric_range(num_bits)
    if qmin != expected_qmin or qmax != expected_qmax:
        raise ValueError("quantization qmin/qmax do not match num_bits")
    if not isinstance(weight_scale, int | float) or weight_scale <= 0:
        raise ValueError("quantization.weight_scale must be positive")
    if not isinstance(bias_scale, int | float) or bias_scale <= 0:
        raise ValueError("quantization.bias_scale must be positive")
    return {
        "scheme": scheme,
        "num_bits": num_bits,
        "qmin": expected_qmin,
        "qmax": expected_qmax,
        "weight_scale": float(weight_scale),
        "bias_scale": float(bias_scale),
    }


def _expect_int_matrix(
    value: object,
    rows: int,
    columns: int,
    qmin: int,
    qmax: int,
) -> list[list[int]]:
    if not isinstance(value, list) or len(value) != rows:
        raise ValueError("weight must be a list shaped [input_size, output_size]")
    return [_expect_int_vector(row, columns, qmin, qmax, "weight row") for row in value]


def _expect_int_vector(
    value: object,
    size: int,
    qmin: int,
    qmax: int,
    field: str,
) -> list[int]:
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f"{field} must be a list of length {size}")
    vector = []
    for item in value:
        if not isinstance(item, int):
            raise ValueError(f"{field} must contain only integers")
        if item < qmin or item > qmax:
            raise ValueError(f"{field} contains value outside quantized range")
        vector.append(item)
    return vector


def _expect_placement_cores(
    value: object,
    *,
    input_size: int,
    output_size: int,
    max_inputs_per_core: int,
    max_outputs_per_core: int,
) -> list[DenseLIFPlacementCore]:
    if not isinstance(value, list):
        raise ValueError("cores must be a list")
    cores = []
    for expected_core_id, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ValueError("each core must be an object")
        core = DenseLIFPlacementCore(
            core_id=_expect_nonnegative_int(item, "core_id"),
            input_start=_expect_nonnegative_int(item, "input_start"),
            input_end=_expect_int(item, "input_end"),
            output_start=_expect_nonnegative_int(item, "output_start"),
            output_end=_expect_int(item, "output_end"),
            input_count=_expect_int(item, "input_count"),
            output_count=_expect_int(item, "output_count"),
            synapse_count=_expect_int(item, "synapse_count"),
            requires_accumulator=_expect_bool(item, "requires_accumulator"),
        )
        if core.core_id != expected_core_id:
            raise ValueError("core_id values must be contiguous from zero")
        _validate_placement_core(
            core,
            input_size=input_size,
            output_size=output_size,
            max_inputs_per_core=max_inputs_per_core,
            max_outputs_per_core=max_outputs_per_core,
        )
        cores.append(core)
    if not cores:
        raise ValueError("cores must not be empty")
    return cores


def _validate_placement_core(
    core: DenseLIFPlacementCore,
    *,
    input_size: int,
    output_size: int,
    max_inputs_per_core: int,
    max_outputs_per_core: int,
) -> None:
    if core.input_start >= core.input_end or core.input_end > input_size:
        raise ValueError("core input range is invalid")
    if core.output_start >= core.output_end or core.output_end > output_size:
        raise ValueError("core output range is invalid")
    if core.input_count != core.input_end - core.input_start:
        raise ValueError("core input_count does not match input range")
    if core.output_count != core.output_end - core.output_start:
        raise ValueError("core output_count does not match output range")
    if core.input_count > max_inputs_per_core:
        raise ValueError("core input_count exceeds max_inputs_per_core")
    if core.output_count > max_outputs_per_core:
        raise ValueError("core output_count exceeds max_outputs_per_core")
    if core.synapse_count != core.input_count * core.output_count:
        raise ValueError("core synapse_count does not match tile shape")


def _expect_nonnegative_int(data: Mapping[str, Any], key: str) -> int:
    value = data.get(key)
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"{key} must be a non-negative integer")
    return value


def _expect_bool(data: Mapping[str, Any], key: str) -> bool:
    value = data.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean")
    return value


def _expect_positive_number(data: Mapping[str, Any], key: str) -> float:
    value = data.get(key)
    if not isinstance(value, int | float) or value <= 0:
        raise ValueError(f"{key} must be a positive number")
    return float(value)


def _expect_nonempty_string(data: Mapping[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _expect_format_map(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError("formats must be an object")
    required = {
        "dense_export": HARDWARE_EXPORT_FORMAT,
        "quantized_export": QUANTIZED_HARDWARE_EXPORT_FORMAT,
        "placement_plan": PLACEMENT_EXPORT_FORMAT,
    }
    result: dict[str, str] = {}
    for key, expected in required.items():
        item = value.get(key)
        if item != expected:
            raise ValueError(f"formats.{key} must be {expected!r}")
        result[key] = expected
    return result


def _expect_string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        raise ValueError("notes must be a list")
    notes = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError("notes must contain only strings")
        notes.append(item)
    return notes


def _expect_dense_lif_adapter_mapping(
    value: object,
    *,
    input_size: int,
    output_size: int,
    core_count: int,
) -> list[dict[str, int | bool]]:
    if not isinstance(value, list):
        raise ValueError("mapping must be a list")
    if len(value) != core_count:
        raise ValueError("mapping length must match core_count")
    mapping = []
    for expected_core_id, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ValueError("each mapping item must be an object")
        core_id = _expect_nonnegative_int(item, "core_id")
        input_start = _expect_nonnegative_int(item, "input_start")
        input_end = _expect_int(item, "input_end")
        output_start = _expect_nonnegative_int(item, "output_start")
        output_end = _expect_int(item, "output_end")
        synapse_count = _expect_int(item, "synapse_count")
        requires_accumulator = _expect_bool(item, "requires_accumulator")
        if core_id != expected_core_id:
            raise ValueError("mapping core_id values must be contiguous from zero")
        if input_start >= input_end or input_end > input_size:
            raise ValueError("mapping input range is invalid")
        if output_start >= output_end or output_end > output_size:
            raise ValueError("mapping output range is invalid")
        if synapse_count != (input_end - input_start) * (output_end - output_start):
            raise ValueError("mapping synapse_count does not match tile shape")
        mapping.append(
            {
                "core_id": core_id,
                "input_start": input_start,
                "input_end": input_end,
                "output_start": output_start,
                "output_end": output_end,
                "synapse_count": synapse_count,
                "requires_accumulator": requires_accumulator,
            }
        )
    return mapping


def _expect_summary_map(value: object) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise ValueError("summary must be an object")
    result: dict[str, int] = {}
    for key in ("input_size", "output_size", "num_bits", "core_count", "total_synapse_count"):
        item = value.get(key)
        if not isinstance(item, int) or item <= 0:
            raise ValueError(f"summary.{key} must be a positive integer")
        result[key] = item
    return result
