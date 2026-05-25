"""Instantiate a Lava Dense+LIF process pair from a spiker Lava export spec."""

from __future__ import annotations

import argparse
from pathlib import Path

from spiker.hardware import read_lava_dense_lif_spec, read_quantized_hardware_export
from spiker.lava import build_lava_dense_lif_processes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("spec", type=Path)
    parser.add_argument("--quantized-export", type=Path)
    args = parser.parse_args()

    spec = read_lava_dense_lif_spec(args.spec)
    quantized = (
        read_quantized_hardware_export(args.quantized_export)
        if args.quantized_export is not None
        else None
    )
    processes = build_lava_dense_lif_processes(
        spec,
        quantized=quantized,
        base_dir=args.spec.parent,
        name_prefix=args.spec.stem,
    )

    print("| Object | Value |")
    print("|---|---|")
    print(f"| dense | `{type(processes.dense).__module__}.{type(processes.dense).__name__}` |")
    print(f"| lif | `{type(processes.lif).__module__}.{type(processes.lif).__name__}` |")
    print(f"| weight_shape | `{tuple(processes.weights.shape)}` |")
    print(f"| input_size | {spec.input_size} |")
    print(f"| output_size | {spec.output_size} |")
    print(f"| weight_bits | {spec.weight_bits} |")


if __name__ == "__main__":
    main()
