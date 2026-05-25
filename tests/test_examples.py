from __future__ import annotations

import py_compile
import subprocess
import sys
from pathlib import Path

import pytest


def test_examples_compile() -> None:
    examples_dir = Path(__file__).resolve().parents[1] / "examples"

    for path in sorted(examples_dir.glob("*.py")):
        py_compile.compile(str(path), doraise=True)


def test_compile_policy_resolves_cpu_modes() -> None:
    examples_dir = Path(__file__).resolve().parents[1] / "examples"
    sys.path.insert(0, str(examples_dir))
    try:
        from example_utils import resolve_compile_policy
    finally:
        sys.path.pop(0)

    assert not resolve_compile_policy("auto", "cpu")
    assert resolve_compile_policy("on", "cpu")
    assert not resolve_compile_policy("off", "cpu")


@pytest.mark.extended
def test_mnist_rate_help_describes_backend_recommendation() -> None:
    example = Path(__file__).resolve().parents[1] / "examples" / "train_mnist_rate.py"

    result = subprocess.run(
        [sys.executable, str(example), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    normalized_stdout = " ".join(result.stdout.split())
    assert "auto resolves to triton on CUDA" in normalized_stdout
    assert "triton_compile is explicit, experimental" in normalized_stdout
    assert "memory, balanced, speed" in normalized_stdout


@pytest.mark.extended
def test_mnist_rate_ddp_example_runs_on_cpu() -> None:
    example = Path(__file__).resolve().parents[1] / "examples" / "train_mnist_rate_ddp.py"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node",
            "2",
            str(example),
            "--device",
            "cpu",
            "--compile",
            "off",
            "--backend",
            "torch",
            "--timesteps",
            "4",
            "--batch",
            "4",
            "--hidden",
            "8",
            "--epochs",
            "1",
            "--train-limit",
            "16",
            "--test-limit",
            "16",
            "--eval-batches",
            "1",
            "--log-every",
            "1",
            "--eval-every",
            "2",
            "--num-workers",
            "0",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert "world_size:2" in result.stdout
    assert "final_test_loss=" in result.stdout
    assert "final_test_accuracy=" in result.stdout


@pytest.mark.extended
def test_mnist_rate_fsdp2_example_runs_on_cpu() -> None:
    example = Path(__file__).resolve().parents[1] / "examples" / "train_mnist_rate_fsdp2.py"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node",
            "2",
            str(example),
            "--device",
            "cpu",
            "--compile",
            "off",
            "--backend",
            "torch",
            "--timesteps",
            "4",
            "--batch",
            "4",
            "--hidden",
            "8",
            "--epochs",
            "1",
            "--train-limit",
            "16",
            "--test-limit",
            "16",
            "--eval-batches",
            "1",
            "--log-every",
            "1",
            "--eval-every",
            "2",
            "--num-workers",
            "0",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert "world_size:2" in result.stdout
    assert "final_test_loss=" in result.stdout
    assert "final_test_accuracy=" in result.stdout


@pytest.mark.extended
def test_custom_neuron_dsl_example_runs_on_cpu() -> None:
    example = Path(__file__).resolve().parents[1] / "examples" / "custom_neuron_dsl.py"

    for variant in ("lif", "alif", "refractory_lif"):
        result = subprocess.run(
            [
                sys.executable,
                str(example),
                "--device",
                "cpu",
                "--variant",
                variant,
                "--timesteps",
                "8",
                "--batch",
                "2",
                "--neurons",
                "4",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        assert f"| variant | {variant} |" in result.stdout
        assert "| python_evaluator |" in result.stdout
        assert "| module_auto |" in result.stdout
        if variant == "lif":
            assert "## Surrogate Training Check" in result.stdout


@pytest.mark.extended
def test_custom_surrogate_online_example_runs_on_cpu() -> None:
    example = Path(__file__).resolve().parents[1] / "examples" / "custom_surrogate_online.py"

    result = subprocess.run(
        [
            sys.executable,
            str(example),
            "--device",
            "cpu",
            "--timesteps",
            "5",
            "--batch",
            "2",
            "--features",
            "3",
            "--neurons",
            "4",
            "--width",
            "1.5",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "| surrogate_params.width | 1.500000 |" in result.stdout
    assert "| grad_weight_max_error | 0.000000e+00 |" in result.stdout
    assert "| grad_bias_max_error | 0.000000e+00 |" in result.stdout


@pytest.mark.extended
def test_custom_lif_rate_training_example_runs_on_cpu() -> None:
    example = Path(__file__).resolve().parents[1] / "examples" / "train_custom_lif_rate.py"

    result = subprocess.run(
        [
            sys.executable,
            str(example),
            "--device",
            "cpu",
            "--timesteps",
            "8",
            "--batch",
            "2",
            "--features",
            "3",
            "--neurons",
            "4",
            "--steps",
            "6",
            "--log-every",
            "3",
            "--checkpoint-size",
            "3",
            "--target-rate",
            "0.2",
            "--lr",
            "1.0",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "LinearCustomSurrogateNeuronRate" not in result.stderr
    assert "| synapse.weight | 3x4 | True | 12 |" in result.stdout
    assert "| 6 |" in result.stdout
    assert "final_target_error=" in result.stdout


@pytest.mark.extended
def test_export_hardware_bundle_example_runs(tmp_path: Path) -> None:
    example = Path(__file__).resolve().parents[1] / "examples" / "export_hardware_bundle.py"

    result = subprocess.run(
        [
            sys.executable,
            str(example),
            "--output-dir",
            str(tmp_path),
            "--prefix",
            "unit",
            "--in-features",
            "3",
            "--out-features",
            "5",
            "--max-inputs-per-core",
            "2",
            "--max-outputs-per-core",
            "3",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "| manifest |" in result.stdout
    assert "| placement_cores | 4 |" in result.stdout
    assert (tmp_path / "unit.hardware_bundle.json").exists()
    assert (tmp_path / "unit.dense_lif.json").exists()
    assert (tmp_path / "unit.dense_lif_quantized.json").exists()
    assert (tmp_path / "unit.dense_lif_placement.json").exists()


@pytest.mark.extended
def test_export_hardware_bundle_example_writes_lava_adapter(tmp_path: Path) -> None:
    example = Path(__file__).resolve().parents[1] / "examples" / "export_hardware_bundle.py"

    result = subprocess.run(
        [
            sys.executable,
            str(example),
            "--output-dir",
            str(tmp_path),
            "--prefix",
            "unit",
            "--in-features",
            "3",
            "--out-features",
            "5",
            "--max-inputs-per-core",
            "2",
            "--max-outputs-per-core",
            "3",
            "--adapter",
            "lava",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "| lava_process_spec |" in result.stdout
    assert "spiker.lava_dense_lif_spec.v0" in result.stdout
    assert (tmp_path / "unit.lava_dense_lif.json").exists()


@pytest.mark.extended
def test_export_hardware_bundle_example_writes_spinnaker_adapter(tmp_path: Path) -> None:
    example = Path(__file__).resolve().parents[1] / "examples" / "export_hardware_bundle.py"

    result = subprocess.run(
        [
            sys.executable,
            str(example),
            "--output-dir",
            str(tmp_path),
            "--prefix",
            "unit",
            "--in-features",
            "3",
            "--out-features",
            "5",
            "--max-inputs-per-core",
            "2",
            "--max-outputs-per-core",
            "3",
            "--adapter",
            "spinnaker2",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "| spinnaker2_adapter_manifest |" in result.stdout
    assert "spiker.spinnaker2_dense_lif_manifest.v0" in result.stdout
    assert (tmp_path / "unit.spinnaker2_manifest.json").exists()


@pytest.mark.extended
def test_export_hardware_bundle_example_writes_both_adapters(tmp_path: Path) -> None:
    example = Path(__file__).resolve().parents[1] / "examples" / "export_hardware_bundle.py"

    result = subprocess.run(
        [
            sys.executable,
            str(example),
            "--output-dir",
            str(tmp_path),
            "--prefix",
            "unit",
            "--in-features",
            "3",
            "--out-features",
            "5",
            "--max-inputs-per-core",
            "2",
            "--max-outputs-per-core",
            "3",
            "--adapter",
            "both",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "| loihi2_adapter_manifest |" in result.stdout
    assert "| spinnaker2_adapter_manifest |" in result.stdout
    assert (tmp_path / "unit.hardware_bundle.json").exists()
    assert (tmp_path / "unit.loihi2_manifest.json").exists()
    assert (tmp_path / "unit.loihi2.dense_lif_placement.json").exists()
    assert (tmp_path / "unit.spinnaker2_manifest.json").exists()
    assert (tmp_path / "unit.spinnaker2.dense_lif_placement.json").exists()
