from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import spiker
from spiker.hardware import (
    HARDWARE_BUNDLE_FORMAT,
    HARDWARE_EXPORT_FORMAT,
    PLACEMENT_EXPORT_FORMAT,
    QUANTIZED_HARDWARE_EXPORT_FORMAT,
    SPINNAKER2_DENSE_LIF_EXPORT_FORMAT,
    dense_lif_hardware_export_from_dict,
    dense_lif_placement_plan_from_dict,
    dequantize_dense_lif_export,
    export_dense_lif_layer,
    export_linear_lif_hardware_bundle,
    export_linear_lif_module,
    export_spinnaker2_dense_lif_manifest,
    hardware_export_bundle_from_dict,
    plan_dense_lif_placement,
    quantize_dense_lif_export,
    quantized_dense_lif_hardware_export_from_dict,
    read_dense_lif_placement_plan,
    read_hardware_export,
    read_hardware_export_bundle,
    read_quantized_hardware_export,
    read_spinnaker2_dense_lif_manifest,
    spinnaker2_dense_lif_manifest_from_dict,
    write_dense_lif_placement_plan,
    write_hardware_export,
    write_hardware_export_bundle,
    write_quantized_hardware_export,
    write_spinnaker2_dense_lif_manifest,
)
from spiker.modules import LinearLIF
from spiker.neurons import LIFParams

pytestmark = pytest.mark.extended


def test_export_dense_lif_layer_round_trips_json(tmp_path: Path) -> None:
    weight = torch.tensor([[0.1, -0.2, 0.3], [0.4, 0.5, -0.6]])
    bias = torch.tensor([0.01, -0.02, 0.03])
    params = LIFParams(tau_mem=10.0, threshold=0.75, reset=-0.1)

    export = export_dense_lif_layer(
        weight,
        bias,
        params,
        dt=0.001,
        metadata={"target": "generic"},
    )
    parsed = dense_lif_hardware_export_from_dict(json.loads(export.to_json()))

    assert parsed.format == HARDWARE_EXPORT_FORMAT
    assert parsed.layer_type == "dense_lif"
    assert parsed.input_size == 2
    assert parsed.output_size == 3
    assert torch.allclose(
        torch.tensor(parsed.weight),
        torch.tensor([[0.1, -0.2, 0.3], [0.4, 0.5, -0.6]]),
    )
    assert parsed.bias == pytest.approx([0.01, -0.02, 0.03])
    assert parsed.neuron == {
        "tau_mem": 10.0,
        "decay": 0.9,
        "threshold": 0.75,
        "reset": -0.1,
    }
    assert parsed.timestep == {"dt": 0.001}
    assert parsed.metadata == {"target": "generic"}

    path = tmp_path / "layer.json"
    write_hardware_export(export, path)
    assert read_hardware_export(path) == parsed


def test_export_dense_lif_layer_supports_biasless() -> None:
    export = export_dense_lif_layer(
        torch.ones((2, 4)),
        None,
        LIFParams(),
        metadata={"bias": "none"},
    )

    assert export.bias is None
    assert export.input_size == 2
    assert export.output_size == 4
    assert export.metadata == {"bias": "none"}


def test_export_linear_lif_module_uses_module_weights() -> None:
    layer = LinearLIF(2, 3, LIFParams(tau_mem=5.0, threshold=2.0, reset=-1.0), bias=True)
    layer.synapse.weight.data.copy_(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    assert layer.synapse.bias is not None
    layer.synapse.bias.data.copy_(torch.tensor([0.1, 0.2, 0.3]))

    export = export_linear_lif_module(layer, dt=0.5)

    assert export.weight == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    assert export.bias == pytest.approx([0.1, 0.2, 0.3])
    assert export.neuron == {
        "tau_mem": 5.0,
        "decay": 0.8,
        "threshold": 2.0,
        "reset": -1.0,
    }
    assert export.timestep == {"dt": 0.5}


def test_quantize_dense_lif_export_round_trips_json(tmp_path: Path) -> None:
    export = export_dense_lif_layer(
        torch.tensor([[-1.0, -0.5, 0.0], [0.25, 0.5, 1.0]]),
        torch.tensor([-0.25, 0.0, 0.25]),
        LIFParams(tau_mem=8.0, threshold=0.75, reset=0.0),
        dt=0.002,
        metadata={"target": "fixed-point"},
    )

    quantized = quantize_dense_lif_export(export, num_bits=8)
    parsed = quantized_dense_lif_hardware_export_from_dict(json.loads(quantized.to_json()))

    assert parsed.format == QUANTIZED_HARDWARE_EXPORT_FORMAT
    assert parsed.source_format == HARDWARE_EXPORT_FORMAT
    assert parsed.weight == [[-127, -64, 0], [32, 64, 127]]
    assert parsed.bias == [-127, 0, 127]
    assert parsed.quantization == {
        "scheme": "signed_symmetric_per_tensor",
        "num_bits": 8,
        "qmin": -127,
        "qmax": 127,
        "weight_scale": pytest.approx(1.0 / 127.0),
        "bias_scale": pytest.approx(0.25 / 127.0),
    }
    assert parsed.neuron == export.neuron
    assert parsed.timestep == export.timestep
    assert parsed.metadata == {"target": "fixed-point"}

    path = tmp_path / "quantized.json"
    write_quantized_hardware_export(parsed, path)
    assert read_quantized_hardware_export(path) == parsed


def test_dequantize_dense_lif_export_approximates_float_export() -> None:
    export = export_dense_lif_layer(
        torch.tensor([[-0.3, 0.0, 0.3], [0.15, -0.15, 0.075]]),
        None,
        LIFParams(),
    )
    quantized = quantize_dense_lif_export(export, num_bits=4)
    dequantized = dequantize_dense_lif_export(quantized)

    scale = float(quantized.quantization["weight_scale"])
    assert torch.allclose(
        torch.tensor(dequantized.weight),
        torch.tensor(export.weight),
        atol=scale / 2.0,
    )
    assert dequantized.bias is None
    assert dequantized.neuron == export.neuron


def test_quantize_dense_lif_export_rejects_invalid_bit_width() -> None:
    export = export_dense_lif_layer(torch.ones((2, 3)), None, LIFParams())

    with pytest.raises(ValueError, match="num_bits"):
        quantize_dense_lif_export(export, num_bits=1)

    with pytest.raises(ValueError, match="num_bits"):
        quantize_dense_lif_export(export, num_bits=17)


def test_quantized_dense_lif_export_parser_rejects_out_of_range_values() -> None:
    data = quantize_dense_lif_export(
        export_dense_lif_layer(torch.ones((1, 1)), None, LIFParams()),
        num_bits=4,
    ).to_dict()
    data["weight"] = [[8]]

    with pytest.raises(ValueError, match="outside quantized range"):
        quantized_dense_lif_hardware_export_from_dict(data)


def test_plan_dense_lif_placement_round_trips_json(tmp_path: Path) -> None:
    quantized = quantize_dense_lif_export(
        export_dense_lif_layer(torch.ones((5, 7)), torch.ones(7), LIFParams()),
        num_bits=8,
    )

    plan = plan_dense_lif_placement(
        quantized,
        max_inputs_per_core=2,
        max_outputs_per_core=3,
        target="generic-test",
        metadata={"profile": "tiny"},
    )
    parsed = dense_lif_placement_plan_from_dict(json.loads(plan.to_json()))

    assert parsed.format == PLACEMENT_EXPORT_FORMAT
    assert parsed.source_format == QUANTIZED_HARDWARE_EXPORT_FORMAT
    assert parsed.target == "generic-test"
    assert parsed.input_size == 5
    assert parsed.output_size == 7
    assert parsed.core_count == 9
    assert parsed.total_synapse_count == 35
    assert parsed.metadata == {"profile": "tiny"}
    assert [
        (
            core.core_id,
            core.input_start,
            core.input_end,
            core.output_start,
            core.output_end,
            core.synapse_count,
            core.requires_accumulator,
        )
        for core in parsed.cores
    ] == [
        (0, 0, 2, 0, 3, 6, True),
        (1, 2, 4, 0, 3, 6, True),
        (2, 4, 5, 0, 3, 3, True),
        (3, 0, 2, 3, 6, 6, True),
        (4, 2, 4, 3, 6, 6, True),
        (5, 4, 5, 3, 6, 3, True),
        (6, 0, 2, 6, 7, 2, True),
        (7, 2, 4, 6, 7, 2, True),
        (8, 4, 5, 6, 7, 1, True),
    ]

    path = tmp_path / "placement.json"
    write_dense_lif_placement_plan(parsed, path)
    assert read_dense_lif_placement_plan(path) == parsed


def test_plan_dense_lif_placement_omits_accumulator_when_inputs_fit_one_core() -> None:
    quantized = quantize_dense_lif_export(
        export_dense_lif_layer(torch.ones((3, 5)), None, LIFParams(), metadata={"m": "n"}),
        num_bits=8,
    )

    plan = plan_dense_lif_placement(
        quantized,
        max_inputs_per_core=3,
        max_outputs_per_core=2,
    )

    assert plan.core_count == 3
    assert plan.metadata == {"m": "n"}
    assert all(not core.requires_accumulator for core in plan.cores)
    assert [core.output_count for core in plan.cores] == [2, 2, 1]


def test_plan_dense_lif_placement_rejects_invalid_limits_and_target() -> None:
    quantized = quantize_dense_lif_export(
        export_dense_lif_layer(torch.ones((2, 3)), None, LIFParams()),
        num_bits=8,
    )

    with pytest.raises(ValueError, match="max_inputs_per_core"):
        plan_dense_lif_placement(quantized, max_inputs_per_core=0, max_outputs_per_core=2)

    with pytest.raises(ValueError, match="max_outputs_per_core"):
        plan_dense_lif_placement(quantized, max_inputs_per_core=2, max_outputs_per_core=0)

    with pytest.raises(ValueError, match="target"):
        plan_dense_lif_placement(
            quantized, max_inputs_per_core=2, max_outputs_per_core=2, target=""
        )


def test_dense_lif_placement_parser_rejects_inconsistent_core_counts() -> None:
    quantized = quantize_dense_lif_export(
        export_dense_lif_layer(torch.ones((2, 3)), None, LIFParams()),
        num_bits=8,
    )
    data = plan_dense_lif_placement(
        quantized,
        max_inputs_per_core=2,
        max_outputs_per_core=2,
    ).to_dict()
    data["core_count"] = 99

    with pytest.raises(ValueError, match="core_count"):
        dense_lif_placement_plan_from_dict(data)


def test_export_linear_lif_hardware_bundle_writes_manifest_and_artifacts(tmp_path: Path) -> None:
    layer = LinearLIF(3, 5, LIFParams(tau_mem=12.0, threshold=0.8), bias=True)
    layer.synapse.weight.data.copy_(
        torch.tensor(
            [
                [0.1, -0.1, 0.2, -0.2, 0.3],
                [0.4, -0.4, 0.5, -0.5, 0.6],
                [0.7, -0.7, 0.8, -0.8, 0.9],
            ]
        )
    )
    assert layer.synapse.bias is not None
    layer.synapse.bias.data.copy_(torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4]))

    bundle = export_linear_lif_hardware_bundle(
        layer,
        tmp_path,
        dt=0.001,
        num_bits=6,
        max_inputs_per_core=2,
        max_outputs_per_core=3,
        target="generic-test",
        prefix="hidden",
        metadata={"run": "unit"},
    )
    parsed_bundle = read_hardware_export_bundle(tmp_path / "hidden.hardware_bundle.json")

    assert parsed_bundle == bundle
    assert parsed_bundle.format == HARDWARE_BUNDLE_FORMAT
    assert parsed_bundle.target == "generic-test"
    assert parsed_bundle.dense_export_path == "hidden.dense_lif.json"
    assert parsed_bundle.quantized_export_path == "hidden.dense_lif_quantized.json"
    assert parsed_bundle.placement_plan_path == "hidden.dense_lif_placement.json"
    assert parsed_bundle.formats == {
        "dense_export": HARDWARE_EXPORT_FORMAT,
        "quantized_export": QUANTIZED_HARDWARE_EXPORT_FORMAT,
        "placement_plan": PLACEMENT_EXPORT_FORMAT,
    }
    assert parsed_bundle.summary == {
        "input_size": 3,
        "output_size": 5,
        "num_bits": 6,
        "core_count": 4,
        "total_synapse_count": 15,
    }

    dense = read_hardware_export(tmp_path / parsed_bundle.dense_export_path)
    quantized = read_quantized_hardware_export(tmp_path / parsed_bundle.quantized_export_path)
    placement = read_dense_lif_placement_plan(tmp_path / parsed_bundle.placement_plan_path)
    assert dense.neuron["tau_mem"] == 12.0
    assert dense.timestep == {"dt": 0.001}
    assert quantized.quantization["num_bits"] == 6
    assert placement.core_count == 4
    assert all(path.exists() for path in tmp_path.iterdir())


def test_hardware_export_bundle_round_trips_json(tmp_path: Path) -> None:
    layer = LinearLIF(2, 2, LIFParams(), bias=False)
    bundle = export_linear_lif_hardware_bundle(
        layer,
        tmp_path,
        max_inputs_per_core=2,
        max_outputs_per_core=2,
        prefix="out",
    )
    parsed = hardware_export_bundle_from_dict(json.loads(bundle.to_json()))
    path = tmp_path / "copy.hardware_bundle.json"

    write_hardware_export_bundle(parsed, path)

    assert read_hardware_export_bundle(path) == parsed


def test_export_spinnaker2_dense_lif_manifest_round_trips_json(tmp_path: Path) -> None:
    dense = export_dense_lif_layer(
        torch.ones((5, 7)),
        torch.ones(7),
        LIFParams(tau_mem=12.0, threshold=0.75, reset=-0.1),
        dt=0.001,
        metadata={"run": "unit"},
    )
    quantized = quantize_dense_lif_export(dense, num_bits=8)
    placement = plan_dense_lif_placement(
        quantized,
        max_inputs_per_core=3,
        max_outputs_per_core=4,
        target="spinnaker2",
    )

    manifest = export_spinnaker2_dense_lif_manifest(
        quantized,
        placement,
        quantized_export_path="layer.dense_lif_quantized.json",
        placement_plan_path="layer.dense_lif_placement.json",
        neurons_per_core=4,
        incoming_synapses_per_core=12,
        metadata={"target": "spinnaker2"},
    )
    parsed = spinnaker2_dense_lif_manifest_from_dict(json.loads(manifest.to_json()))
    path = tmp_path / "layer.spinnaker2_manifest.json"

    write_spinnaker2_dense_lif_manifest(parsed, path)

    assert read_spinnaker2_dense_lif_manifest(path) == parsed
    assert parsed.format == SPINNAKER2_DENSE_LIF_EXPORT_FORMAT
    assert parsed.source_format == QUANTIZED_HARDWARE_EXPORT_FORMAT
    assert parsed.target == "spinnaker2"
    assert parsed.input_size == 5
    assert parsed.output_size == 7
    assert parsed.core_count == placement.core_count
    assert parsed.total_synapse_count == 35
    assert parsed.weight_bits == 8
    assert parsed.timestep_ms == pytest.approx(1.0)
    assert parsed.neuron == dense.neuron
    assert parsed.mapping[0] == {
        "core_id": 0,
        "input_start": 0,
        "input_end": 3,
        "output_start": 0,
        "output_end": 4,
        "synapse_count": 12,
        "requires_accumulator": True,
    }
    assert parsed.metadata == {"target": "spinnaker2"}


def test_export_spinnaker2_dense_lif_manifest_rejects_constraint_mismatch() -> None:
    dense = export_dense_lif_layer(torch.ones((4, 4)), None, LIFParams())
    quantized = quantize_dense_lif_export(dense, num_bits=8)
    placement = plan_dense_lif_placement(
        quantized,
        max_inputs_per_core=4,
        max_outputs_per_core=4,
        target="spinnaker2",
    )

    with pytest.raises(ValueError, match="neurons_per_core"):
        export_spinnaker2_dense_lif_manifest(
            quantized,
            placement,
            quantized_export_path="q.json",
            placement_plan_path="p.json",
            neurons_per_core=3,
            incoming_synapses_per_core=16,
        )

    with pytest.raises(ValueError, match="incoming_synapses_per_core"):
        export_spinnaker2_dense_lif_manifest(
            quantized,
            placement,
            quantized_export_path="q.json",
            placement_plan_path="p.json",
            neurons_per_core=4,
            incoming_synapses_per_core=15,
        )

    generic_placement = plan_dense_lif_placement(
        quantized,
        max_inputs_per_core=4,
        max_outputs_per_core=4,
        target="generic",
    )
    with pytest.raises(ValueError, match="target"):
        export_spinnaker2_dense_lif_manifest(
            quantized,
            generic_placement,
            quantized_export_path="q.json",
            placement_plan_path="p.json",
            neurons_per_core=4,
            incoming_synapses_per_core=16,
        )


def test_export_linear_lif_hardware_bundle_rejects_empty_prefix(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="prefix"):
        export_linear_lif_hardware_bundle(
            LinearLIF(2, 2, LIFParams()),
            tmp_path,
            max_inputs_per_core=2,
            max_outputs_per_core=2,
            prefix="",
        )


@pytest.mark.parametrize(
    ("weight", "bias", "dt", "match"),
    [
        (torch.ones((2, 3, 1)), None, 1.0, "weight must be shaped"),
        (torch.ones((0, 3)), None, 1.0, "non-empty"),
        (torch.ones((2, 3), dtype=torch.int64), None, 1.0, "floating point"),
        (torch.ones((2, 3)), torch.ones(2), 1.0, "bias must be shaped"),
        (torch.ones((2, 3)), None, 0.0, "dt must be positive"),
        (torch.tensor([[float("nan")]]), None, 1.0, "finite"),
    ],
)
def test_export_dense_lif_layer_rejects_invalid_inputs(
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    dt: float,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        export_dense_lif_layer(weight, bias, LIFParams(), dt=dt)


def test_export_linear_lif_module_rejects_non_lif_module() -> None:
    with pytest.raises(TypeError, match="LinearLIF-style"):
        export_linear_lif_module(torch.nn.Linear(2, 3))


def test_hardware_exports_are_public() -> None:
    assert spiker.HARDWARE_BUNDLE_FORMAT == HARDWARE_BUNDLE_FORMAT
    assert spiker.HARDWARE_EXPORT_FORMAT == HARDWARE_EXPORT_FORMAT
    assert spiker.PLACEMENT_EXPORT_FORMAT == PLACEMENT_EXPORT_FORMAT
    assert spiker.QUANTIZED_HARDWARE_EXPORT_FORMAT == QUANTIZED_HARDWARE_EXPORT_FORMAT
    assert spiker.SPINNAKER2_DENSE_LIF_EXPORT_FORMAT == SPINNAKER2_DENSE_LIF_EXPORT_FORMAT
    assert spiker.DenseLIFHardwareExport is not None
    assert spiker.DenseLIFPlacementCore is not None
    assert spiker.DenseLIFPlacementPlan is not None
    assert spiker.HardwareExportBundle is not None
    assert spiker.QuantizedDenseLIFHardwareExport is not None
    assert spiker.SpiNNaker2DenseLIFManifest is not None
    assert spiker.export_dense_lif_layer is export_dense_lif_layer
    assert spiker.export_linear_lif_module is export_linear_lif_module
    assert spiker.dense_lif_hardware_export_from_dict is dense_lif_hardware_export_from_dict
    assert spiker.dense_lif_placement_plan_from_dict is dense_lif_placement_plan_from_dict
    assert spiker.hardware_export_bundle_from_dict is hardware_export_bundle_from_dict
    assert (
        spiker.quantized_dense_lif_hardware_export_from_dict
        is quantized_dense_lif_hardware_export_from_dict
    )
    assert spiker.export_linear_lif_hardware_bundle is export_linear_lif_hardware_bundle
    assert spiker.export_spinnaker2_dense_lif_manifest is export_spinnaker2_dense_lif_manifest
    assert spiker.spinnaker2_dense_lif_manifest_from_dict is spinnaker2_dense_lif_manifest_from_dict
    assert spiker.plan_dense_lif_placement is plan_dense_lif_placement
    assert spiker.quantize_dense_lif_export is quantize_dense_lif_export
    assert spiker.dequantize_dense_lif_export is dequantize_dense_lif_export
    assert spiker.read_dense_lif_placement_plan is read_dense_lif_placement_plan
    assert spiker.read_hardware_export is read_hardware_export
    assert spiker.read_hardware_export_bundle is read_hardware_export_bundle
    assert spiker.read_quantized_hardware_export is read_quantized_hardware_export
    assert spiker.read_spinnaker2_dense_lif_manifest is read_spinnaker2_dense_lif_manifest
    assert spiker.write_dense_lif_placement_plan is write_dense_lif_placement_plan
    assert spiker.write_hardware_export is write_hardware_export
    assert spiker.write_hardware_export_bundle is write_hardware_export_bundle
    assert spiker.write_quantized_hardware_export is write_quantized_hardware_export
    assert spiker.write_spinnaker2_dense_lif_manifest is write_spinnaker2_dense_lif_manifest
