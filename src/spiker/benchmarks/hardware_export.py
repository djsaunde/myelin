"""Smoke benchmark for the hardware export bridge."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

import torch

from spiker.hardware import (
    DenseLIFHardwareExport,
    DenseLIFPlacementPlan,
    HardwareExportBundle,
    QuantizedDenseLIFHardwareExport,
    SpiNNaker2DenseLIFManifest,
    export_linear_lif_hardware_bundle,
    export_spinnaker2_dense_lif_manifest,
    plan_dense_lif_placement,
    read_dense_lif_placement_plan,
    read_hardware_export,
    read_quantized_hardware_export,
    read_spinnaker2_dense_lif_manifest,
    write_dense_lif_placement_plan,
    write_spinnaker2_dense_lif_manifest,
)
from spiker.modules import LinearLIF
from spiker.neurons import LIFParams


def initialize_deterministic_weights(layer: LinearLIF, *, seed: int) -> None:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    with torch.no_grad():
        layer.synapse.weight.copy_(
            (torch.rand(layer.synapse.weight.shape, generator=generator) - 0.5) * 0.2
        )
        if layer.synapse.bias is not None:
            layer.synapse.bias.copy_(
                (torch.rand(layer.synapse.bias.shape, generator=generator) - 0.5) * 0.02
            )


def write_export_artifacts(args: argparse.Namespace, output_dir: Path) -> dict[str, object]:
    params = LIFParams(tau_mem=args.tau_mem, threshold=args.threshold, reset=args.reset)
    layer = LinearLIF(args.in_features, args.out_features, params)
    initialize_deterministic_weights(layer, seed=args.seed)

    bundle = export_linear_lif_hardware_bundle(
        layer,
        output_dir,
        dt=args.dt,
        num_bits=args.num_bits,
        max_inputs_per_core=args.max_inputs_per_core,
        max_outputs_per_core=args.max_outputs_per_core,
        target="generic",
        prefix=args.prefix,
        metadata={"benchmark": "hardware_export"},
    )
    quantized = read_quantized_hardware_export(output_dir / bundle.quantized_export_path)

    spinnaker_placement = plan_dense_lif_placement(
        quantized,
        max_inputs_per_core=args.max_inputs_per_core,
        max_outputs_per_core=args.max_outputs_per_core,
        target="spinnaker2",
        metadata={"benchmark": "hardware_export", "adapter": "spinnaker2"},
    )
    spinnaker_placement_path = f"{args.prefix}.spinnaker2.dense_lif_placement.json"
    write_dense_lif_placement_plan(spinnaker_placement, output_dir / spinnaker_placement_path)
    spinnaker_manifest_path = f"{args.prefix}.spinnaker2_manifest.json"
    spinnaker_manifest = export_spinnaker2_dense_lif_manifest(
        quantized,
        spinnaker_placement,
        quantized_export_path=bundle.quantized_export_path,
        placement_plan_path=spinnaker_placement_path,
        neurons_per_core=args.spinnaker_neurons_per_core,
        incoming_synapses_per_core=args.spinnaker_incoming_synapses_per_core,
        metadata={"benchmark": "hardware_export", "adapter": "spinnaker2"},
    )
    write_spinnaker2_dense_lif_manifest(spinnaker_manifest, output_dir / spinnaker_manifest_path)

    return {
        "bundle": bundle,
        "bundle_path": f"{args.prefix}.hardware_bundle.json",
        "dense": read_hardware_export(output_dir / bundle.dense_export_path),
        "quantized": quantized,
        "placement": read_dense_lif_placement_plan(output_dir / bundle.placement_plan_path),
        "spinnaker_placement": read_dense_lif_placement_plan(output_dir / spinnaker_placement_path),
        "spinnaker_manifest": read_spinnaker2_dense_lif_manifest(
            output_dir / spinnaker_manifest_path
        ),
        "spinnaker_placement_path": spinnaker_placement_path,
        "spinnaker_manifest_path": spinnaker_manifest_path,
    }


def print_markdown(
    args: argparse.Namespace, output_dir: Path, artifacts: dict[str, object]
) -> None:
    bundle = cast(HardwareExportBundle, artifacts["bundle"])
    dense = cast(DenseLIFHardwareExport, artifacts["dense"])
    quantized = cast(QuantizedDenseLIFHardwareExport, artifacts["quantized"])
    placement = cast(DenseLIFPlacementPlan, artifacts["placement"])
    spinnaker_placement = cast(DenseLIFPlacementPlan, artifacts["spinnaker_placement"])
    spinnaker_manifest = cast(SpiNNaker2DenseLIFManifest, artifacts["spinnaker_manifest"])

    print("# Hardware Export Bridge Benchmark")
    print()
    print("Exports one deterministic `LinearLIF` through the generic and target handoff path.")
    print()
    print("## Environment")
    print()
    print(f"- `generated_utc`: `{datetime.now(UTC).isoformat(timespec='seconds')}`")
    print(f"- `torch`: `{torch.__version__}`")
    print(f"- `shape`: `F={args.in_features}, N={args.out_features}`")
    print(f"- `output_dir`: `{output_dir}`")
    print(f"- `num_bits`: `{args.num_bits}`")
    print(f"- `max_inputs_per_core`: `{args.max_inputs_per_core}`")
    print(f"- `max_outputs_per_core`: `{args.max_outputs_per_core}`")
    print()
    print("## Artifacts")
    print()
    print("| Artifact | Path | Format |")
    print("|---|---|---|")
    print(f"| bundle | `{artifacts['bundle_path']}` | {bundle.format} |")
    print(f"| dense_export | `{bundle.dense_export_path}` | {bundle.formats['dense_export']} |")
    print(
        f"| quantized_export | `{bundle.quantized_export_path}` | "
        f"{bundle.formats['quantized_export']} |"
    )
    print(
        f"| placement_plan | `{bundle.placement_plan_path}` | {bundle.formats['placement_plan']} |"
    )
    print(
        f"| spinnaker2_placement_plan | `{artifacts['spinnaker_placement_path']}` | "
        f"{spinnaker_placement.format} |"
    )
    print(
        f"| spinnaker2_manifest | `{artifacts['spinnaker_manifest_path']}` | "
        f"{spinnaker_manifest.format} |"
    )
    print()
    print("## Checks")
    print()
    print("| Check | Value |")
    print("|---|---:|")
    print(f"| dense_weight_rows | {len(dense.weight)} |")
    print(f"| quantized_weight_rows | {len(quantized.weight)} |")
    print(f"| generic_core_count | {placement.core_count} |")
    print(f"| spinnaker2_core_count | {spinnaker_manifest.core_count} |")
    print(f"| total_synapse_count | {bundle.summary['total_synapse_count']} |")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--prefix", default="linear_lif")
    parser.add_argument("--in-features", type=int, default=3)
    parser.add_argument("--out-features", type=int, default=5)
    parser.add_argument("--tau-mem", type=float, default=20.0)
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--reset", type=float, default=0.0)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--num-bits", type=int, default=8)
    parser.add_argument("--max-inputs-per-core", type=int, default=2)
    parser.add_argument("--max-outputs-per-core", type=int, default=3)
    parser.add_argument("--spinnaker-neurons-per-core", type=int, default=256)
    parser.add_argument("--spinnaker-incoming-synapses-per-core", type=int, default=65536)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.output_dir is None:
        with TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            artifacts = write_export_artifacts(args, output_dir)
            print_markdown(args, output_dir, artifacts)
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print_markdown(args, args.output_dir, write_export_artifacts(args, args.output_dir))


if __name__ == "__main__":
    main()
