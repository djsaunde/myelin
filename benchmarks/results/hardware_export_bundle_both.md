# Hardware Export Bridge Benchmark

Exports one deterministic `LinearLIF` through the generic and target handoff path.

## Environment

- `generated_utc`: `2026-05-24T19:27:30+00:00`
- `torch`: `2.12.0+cu130`
- `shape`: `F=3, N=5`
- `output_dir`: `benchmarks/results/hardware_exports`
- `num_bits`: `8`
- `max_inputs_per_core`: `2`
- `max_outputs_per_core`: `3`

## Artifacts

| Artifact | Path | Format |
|---|---|---|
| bundle | `linear_lif_both.hardware_bundle.json` | spiker.hardware_bundle.v0 |
| dense_export | `linear_lif_both.dense_lif.json` | spiker.dense_lif.v0 |
| quantized_export | `linear_lif_both.dense_lif_quantized.json` | spiker.dense_lif_quantized.v0 |
| placement_plan | `linear_lif_both.dense_lif_placement.json` | spiker.dense_lif_placement.v0 |
| lava_process_spec | `linear_lif_both.lava_dense_lif.json` | spiker.lava_dense_lif_spec.v0 |
| loihi2_placement_plan | `linear_lif_both.loihi2.dense_lif_placement.json` | spiker.dense_lif_placement.v0 |
| loihi2_manifest | `linear_lif_both.loihi2_manifest.json` | spiker.loihi2_dense_lif_manifest.v0 |
| spinnaker2_placement_plan | `linear_lif_both.spinnaker2.dense_lif_placement.json` | spiker.dense_lif_placement.v0 |
| spinnaker2_manifest | `linear_lif_both.spinnaker2_manifest.json` | spiker.spinnaker2_dense_lif_manifest.v0 |

## Checks

| Check | Value |
|---|---:|
| dense_weight_rows | 3 |
| quantized_weight_rows | 3 |
| generic_core_count | 4 |
| lava_weight_bits | 8 |
| loihi2_core_count | 4 |
| spinnaker2_core_count | 4 |
| total_synapse_count | 15 |
