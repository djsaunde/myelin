"""Export a LinearLIF layer through the generic hardware bridge."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from example_utils import print_model_summary

from spiker import (
    LIFParams,
    LinearLIF,
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


def print_artifact_table(
    output_dir: Path,
    bundle_path: str,
    bundle,
    adapters: list[tuple[str, str, str]],
) -> None:
    print("| Artifact | Path | Format |")
    print("|---|---|---|")
    print(f"| manifest | {output_dir / bundle_path} | {bundle.format} |")
    print(
        f"| dense_export | {output_dir / bundle.dense_export_path} | "
        f"{bundle.formats['dense_export']} |"
    )
    print(
        f"| quantized_export | {output_dir / bundle.quantized_export_path} | "
        f"{bundle.formats['quantized_export']} |"
    )
    print(
        f"| placement_plan | {output_dir / bundle.placement_plan_path} | "
        f"{bundle.formats['placement_plan']} |"
    )
    for label, adapter_path, adapter_format in adapters:
        print(f"| {label} | {output_dir / adapter_path} | {adapter_format} |")


def print_summary_table(bundle) -> None:
    print()
    print("| Field | Value |")
    print("|---|---:|")
    print(f"| input_size | {bundle.summary['input_size']} |")
    print(f"| output_size | {bundle.summary['output_size']} |")
    print(f"| num_bits | {bundle.summary['num_bits']} |")
    print(f"| core_count | {bundle.summary['core_count']} |")
    print(f"| total_synapse_count | {bundle.summary['total_synapse_count']} |")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("hardware_exports"))
    parser.add_argument("--prefix", default="linear_lif")
    parser.add_argument("--in-features", type=int, default=8)
    parser.add_argument("--out-features", type=int, default=12)
    parser.add_argument("--tau-mem", type=float, default=20.0)
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--reset", type=float, default=0.0)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--num-bits", type=int, default=8)
    parser.add_argument("--max-inputs-per-core", type=int, default=4)
    parser.add_argument("--max-outputs-per-core", type=int, default=6)
    parser.add_argument("--target", default="generic")
    parser.add_argument("--adapter", choices=("spinnaker2", "all"))
    parser.add_argument("--spinnaker-neurons-per-core", type=int, default=256)
    parser.add_argument("--spinnaker-incoming-synapses-per-core", type=int, default=65536)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    params = LIFParams(tau_mem=args.tau_mem, threshold=args.threshold, reset=args.reset)
    layer = LinearLIF(args.in_features, args.out_features, params)
    initialize_deterministic_weights(layer, seed=args.seed)

    print("## Model")
    print()
    print_model_summary(layer)
    print()

    target = args.target
    if args.adapter == "spinnaker2":
        target = "spinnaker2"
    bundle = export_linear_lif_hardware_bundle(
        layer,
        args.output_dir,
        dt=args.dt,
        num_bits=args.num_bits,
        max_inputs_per_core=args.max_inputs_per_core,
        max_outputs_per_core=args.max_outputs_per_core,
        target=target,
        prefix=args.prefix,
        metadata={"example": "export_hardware_bundle"},
    )
    bundle_path = f"{args.prefix}.hardware_bundle.json"

    dense = read_hardware_export(args.output_dir / bundle.dense_export_path)
    quantized = read_quantized_hardware_export(args.output_dir / bundle.quantized_export_path)
    placement = read_dense_lif_placement_plan(args.output_dir / bundle.placement_plan_path)
    adapter_artifacts: list[tuple[str, str, str]] = []
    if args.adapter in {"spinnaker2", "all"}:
        spinnaker_placement = placement
        spinnaker_placement_path = bundle.placement_plan_path
        if target != "spinnaker2":
            spinnaker_placement = plan_dense_lif_placement(
                quantized,
                max_inputs_per_core=args.max_inputs_per_core,
                max_outputs_per_core=args.max_outputs_per_core,
                target="spinnaker2",
                metadata={"example": "export_hardware_bundle", "adapter": "spinnaker2"},
            )
            spinnaker_placement_path = f"{args.prefix}.spinnaker2.dense_lif_placement.json"
            write_dense_lif_placement_plan(
                spinnaker_placement,
                args.output_dir / spinnaker_placement_path,
            )
            adapter_artifacts.append(
                (
                    "spinnaker2_placement_plan",
                    spinnaker_placement_path,
                    spinnaker_placement.format,
                )
            )
        adapter_path = f"{args.prefix}.spinnaker2_manifest.json"
        adapter = export_spinnaker2_dense_lif_manifest(
            quantized,
            spinnaker_placement,
            quantized_export_path=bundle.quantized_export_path,
            placement_plan_path=spinnaker_placement_path,
            neurons_per_core=args.spinnaker_neurons_per_core,
            incoming_synapses_per_core=args.spinnaker_incoming_synapses_per_core,
            metadata={"example": "export_hardware_bundle", "adapter": "spinnaker2"},
        )
        write_spinnaker2_dense_lif_manifest(adapter, args.output_dir / adapter_path)
        adapter_artifacts.append(
            (
                "spinnaker2_adapter_manifest",
                adapter_path,
                read_spinnaker2_dense_lif_manifest(args.output_dir / adapter_path).format,
            )
        )

    print("## Artifacts")
    print()
    print_artifact_table(args.output_dir, bundle_path, bundle, adapter_artifacts)
    print_summary_table(bundle)
    print()
    print("| Check | Value |")
    print("|---|---:|")
    print(f"| dense_weight_rows | {len(dense.weight)} |")
    print(f"| quantized_weight_rows | {len(quantized.weight)} |")
    print(f"| placement_cores | {len(placement.cores)} |")


if __name__ == "__main__":
    main()
