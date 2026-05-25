from __future__ import annotations

import argparse

import pytest
import torch

from spiker.benchmarks.checkpoint_size_sweep import (
    SweepResult,
    parse_checkpoint_sizes,
    pre_reset_scratch_bytes,
    rate_pareto_frontier,
)
from spiker.benchmarks.compile_inspect import summarize_output_code
from spiker.benchmarks.compile_triton_boundary import run_benchmark as run_compile_triton_boundary
from spiker.benchmarks.compile_triton_sweep import (
    parse_shape as parse_compile_triton_sweep_shape,
)
from spiker.benchmarks.compile_triton_sweep import (
    run_benchmark as run_compile_triton_sweep,
)
from spiker.benchmarks.currents_audit import (
    AuditResult,
    expected_checkpoint_scratch_bytes,
    expected_chunk_start_bytes,
    exposes_dense_spike_output,
    make_triton_checkpoint_rate_step,
)
from spiker.benchmarks.currents_boundary import BoundaryResult, current_bytes, format_ratio
from spiker.benchmarks.custom_neuron_module import (
    VARIANTS as CUSTOM_NEURON_VARIANTS,
)
from spiker.benchmarks.custom_neuron_module import (
    build_variant_ir,
    initial_state_for_variant,
    params_for_variant,
)
from spiker.benchmarks.custom_surrogate_training import (
    params_from_decay,
    run_benchmark,
    run_pairwise_comparisons,
)
from spiker.benchmarks.headline import (
    collect_headlines,
    parse_markdown_table,
    parse_scalar_metrics,
)
from spiker.benchmarks.mnist_compare import VARIANTS, parse_metrics, parse_variants, run_variant
from spiker.benchmarks.mnist_rate_matrix import (
    DEFAULT_SETTINGS,
    MatrixSetting,
    parse_setting,
    run_matrix,
)
from spiker.benchmarks.online_learning import (
    alif_online_custom_surrogate_step,
    alif_online_step,
    lif_online_custom_surrogate_step,
    lif_online_step,
)
from spiker.benchmarks.performance_frontier import run_benchmark as run_frontier_benchmark
from spiker.benchmarks.runner import PRESETS, make_runs
from spiker.benchmarks.scalar_loss_boundary import ScalarLossResult
from spiker.benchmarks.snntorch_matrix import (
    TimestepSetting,
    parse_timesteps,
)
from spiker.benchmarks.snntorch_matrix import (
    run_matrix as run_snntorch_matrix,
)
from spiker.benchmarks.training_breakdown import run_benchmark as run_training_breakdown
from spiker.benchmarks.two_layer_recompute import (
    BenchRow as TwoLayerBenchRow,
)
from spiker.benchmarks.two_layer_recompute import (
    Shape as TwoLayerShape,
)
from spiker.benchmarks.two_layer_recompute import (
    parse_shape as parse_two_layer_shape,
)
from spiker.benchmarks.two_layer_recompute import (
    run_benchmark as run_two_layer_recompute,
)
from spiker.benchmarks.two_layer_recompute import (
    summarize_result as summarize_two_layer_result,
)
from spiker.dsl import analyze_neuron_ir, evaluate_neuron_unroll
from spiker.neurons import LIFParams

pytestmark = pytest.mark.extended


def test_currents_boundary_result_reports_forward_increments() -> None:
    result = BoundaryResult(
        timesteps=4,
        batch=2,
        features=3,
        neurons=5,
        workload="Materialized currents",
        eager_forward_backward_seconds=2.0,
        compiled_forward_backward_seconds=0.5,
        eager_allocated_bytes=100,
        compiled_allocated_bytes=200,
        eager_forward_peak_bytes=260,
        compiled_forward_peak_bytes=240,
        eager_backward_peak_bytes=300,
        compiled_backward_peak_bytes=280,
        compile_warmup_seconds=1.0,
    )

    assert current_bytes(result) == 4 * 2 * 5 * 4
    assert result.speedup == 4.0
    assert result.eager_forward_increment_bytes == 160
    assert result.compiled_forward_increment_bytes == 40
    assert result.eager_forward_increment_ratio == 1.0
    assert result.compiled_forward_increment_ratio == 0.25


def test_currents_boundary_format_ratio_handles_missing_values() -> None:
    assert format_ratio(None) == ""
    assert format_ratio(1.234) == "1.23x"


def test_currents_audit_reports_expected_checkpoint_storage() -> None:
    assert (
        expected_chunk_start_bytes(
            timesteps=10,
            batch=2,
            neurons=5,
            checkpoint_size=4,
            dtype=torch.float32,
        )
        == 3 * 2 * 5 * 4
    )
    assert (
        expected_checkpoint_scratch_bytes(
            timesteps=10,
            batch=2,
            neurons=5,
            checkpoint_size=4,
            dtype=torch.float32,
        )
        == 4 * 2 * 5 * 4
    )


def test_currents_audit_result_reports_extra_over_dense_output() -> None:
    result = AuditResult(
        label="dense",
        split_forward_seconds=1.0,
        backward_seconds=2.0,
        allocated_bytes=100,
        forward_peak_bytes=260,
        backward_peak_bytes=320,
    )

    assert result.forward_increment_bytes == 160
    assert result.backward_increment_bytes == 220
    assert result.forward_extra_over_dense_output_bytes(120) == 40
    assert result.backward_extra_over_dense_output_bytes(120) == 100


def test_currents_audit_labels_dense_spike_output_variants() -> None:
    assert exposes_dense_spike_output("Triton checkpoint recompute")
    assert not exposes_dense_spike_output("torch.compile materialized graph")
    assert not exposes_dense_spike_output("Triton checkpoint rate output")


def test_scalar_loss_boundary_result_reports_peak_increment() -> None:
    result = ScalarLossResult(
        variant="unit",
        forward_backward_seconds=0.1,
        allocated_bytes=100,
        peak_bytes=180,
    )

    assert result.increment_bytes == 80


def test_currents_audit_rate_step_uses_public_rate_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_rate_forward(
        inputs: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        params: LIFParams,
        **kwargs: object,
    ) -> tuple[object, torch.Tensor]:
        calls.append({"bias": bias, **kwargs})

        class State:
            membrane = torch.zeros((inputs.shape[1], weight.shape[1]))

        return State(), weight.sum() * 0.0 + torch.tensor(0.5)

    import spiker.kernels as kernels

    monkeypatch.setattr(kernels, "linear_surrogate_lif_rate_forward", fake_rate_forward)
    step = make_triton_checkpoint_rate_step(
        surrogate="fast_sigmoid",
        surrogate_slope=5.0,
        checkpoint_size=2,
        backend="triton_generated",
    )
    inputs = torch.rand((3, 2, 4))
    weight = torch.rand((4, 5), requires_grad=True)

    loss = step(inputs, weight, LIFParams())
    loss.backward()

    assert calls == [
        {
            "bias": None,
            "surrogate": "fast_sigmoid",
            "surrogate_slope": 5.0,
            "hard_forward": True,
            "backend": "triton_generated",
            "checkpoint_size": 2,
            "reduction": "mean",
        }
    ]
    assert weight.grad is not None


def test_torch_rate_forward_does_not_call_dense_spike_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import spiker.kernels as kernels

    def fail_dense_forward(*_args: object, **_kwargs: object) -> tuple[object, torch.Tensor]:
        raise AssertionError("dense spike forward should not be used by torch rate backend")

    monkeypatch.setattr(kernels, "linear_surrogate_lif_forward", fail_dense_forward)
    inputs = torch.rand((4, 2, 3), requires_grad=True)
    weight = torch.rand((3, 5), requires_grad=True) * 0.01

    _state, rates = kernels.linear_surrogate_lif_rate_forward(
        inputs,
        weight,
        None,
        LIFParams(tau_mem=8.0, threshold=0.7, reset=-0.2),
        backend="torch",
        checkpoint_size=2,
        reduction="none",
    )
    loss = rates.square().mean()
    loss.backward()

    assert rates.shape == (2, 5)
    assert inputs.grad is not None


def test_online_learning_custom_surrogate_variants_match_builtin_alias() -> None:
    torch.manual_seed(0)
    inputs = torch.rand((3, 2, 4))
    weight = (torch.rand((4, 5)) - 0.5) * 0.02
    bias = torch.zeros(5)
    learning_signal = torch.rand((3, 2, 5)) - 0.5

    assert torch.allclose(
        lif_online_custom_surrogate_step(inputs, weight, bias, learning_signal),
        lif_online_step(inputs, weight, bias, learning_signal),
    )
    assert torch.allclose(
        alif_online_custom_surrogate_step(inputs, weight, bias, learning_signal),
        alif_online_step(inputs, weight, bias, learning_signal),
    )


def test_mnist_compare_parses_peak_cuda_memory_metric() -> None:
    metrics = parse_metrics(
        "\n".join(
            [
                "final_test_loss=2.123456",
                "final_test_accuracy=0.5000",
                "peak_cuda_memory_mb=123.456",
                "steady_state_average_step_ms=1.234",
            ]
        )
    )

    assert metrics["peak_cuda_memory_mb"] == 123.456


def test_mnist_compare_includes_generated_rate_variant() -> None:
    variants = parse_variants(["rate_generated"])

    assert variants[0].name == "rate_generated"
    assert variants[0].script == "train_mnist_rate.py"
    assert variants[0].extra_args == ("--backend", "triton_generated")


def test_mnist_compare_includes_triton_compile_rate_variant() -> None:
    variants = parse_variants(["rate_triton_compile"])

    assert variants[0].name == "rate_triton_compile"
    assert variants[0].script == "train_mnist_rate.py"
    assert variants[0].extra_args == ("--backend", "triton_compile")


def test_mnist_rate_matrix_parses_settings() -> None:
    assert parse_setting("small,10,128") == MatrixSetting("small", timesteps=10, hidden=128)
    assert DEFAULT_SETTINGS[0].name == "t10_h128"


def test_custom_surrogate_training_params_from_decay() -> None:
    params = params_from_decay(decay=0.75, threshold=0.8, reset=-0.1)

    assert params.tau_mem == 4.0
    assert params.threshold == 0.8
    assert params.reset == -0.1


def test_custom_surrogate_training_cpu_smoke() -> None:
    args = argparse.Namespace(
        timesteps=3,
        batch=2,
        features=3,
        neurons=4,
        device="cpu",
        checkpoint_size=25,
        linear_bias=0.25,
        decay=0.75,
        threshold=0.8,
        reset=-0.1,
        warmup=0,
        repeats=1,
        seed=0,
    )

    results = run_benchmark(args)

    assert [result.variant for result in results] == [
        "builtin",
        "custom_ir",
        "linear_builtin",
        "linear_custom_ir",
        "rate_builtin",
        "rate_custom_ir",
    ]
    assert [result.backend for result in results] == [
        "torch",
        "torch",
        "torch",
        "torch",
        "torch",
        "torch",
    ]

    pairwise = run_pairwise_comparisons(args)

    assert [(result.surface, result.backend) for result in pairwise] == [
        ("cell", "torch"),
        ("linear", "torch"),
        ("rate", "torch"),
    ]
    assert all(result.error is None for result in pairwise)
    assert all(result.loss_max_error == 0.0 for result in pairwise)
    assert all(result.input_grad_max_error == 0.0 for result in pairwise)


def test_performance_frontier_cpu_smoke_without_compile() -> None:
    args = argparse.Namespace(
        timesteps=3,
        batch=2,
        features=3,
        neurons=4,
        device="cpu",
        checkpoint_size=2,
        compile_mode="reduce-overhead",
        no_compile=True,
        surrogate="fast_sigmoid",
        surrogate_slope=5.0,
        warmup=0,
        repeats=1,
    )

    rows = run_frontier_benchmark(args)

    assert [row.contract for row in rows] == [
        "hard LIF forward",
        "hard LIF forward",
        "dense-output training",
        "dense-output training",
        "rate/scalar training",
        "rate/scalar training",
    ]
    assert rows[0].variant == "torch.compile"
    assert rows[0].error is None
    assert rows[1].variant == "Triton fused-time"
    assert rows[1].error is None
    assert rows[2].error == "variant was not produced"


def test_training_breakdown_cpu_smoke_without_compile() -> None:
    args = argparse.Namespace(
        timesteps=3,
        batch=2,
        features=3,
        neurons=4,
        device="cpu",
        checkpoint_size=2,
        compile_mode="reduce-overhead",
        no_compile=True,
        surrogate="fast_sigmoid",
        surrogate_slope=5.0,
        warmup=0,
        repeats=1,
        seed=0,
    )

    rows = run_training_breakdown(args)

    assert [row.component for row in rows] == [
        "dense projection",
        "full training",
        "checkpoint forward",
        "checkpoint forward",
        "checkpoint forward",
        "checkpoint backward",
        "checkpoint backward",
        "checkpoint backward",
        "checkpoint backward",
    ]
    assert rows[0].error is None
    assert rows[1].error == "compile disabled"


def test_compile_triton_boundary_cpu_smoke_reports_missing_backend() -> None:
    args = argparse.Namespace(
        timesteps=4,
        batch=2,
        features=3,
        neurons=5,
        device="cpu",
        checkpoint_size=2,
        compile_mode="reduce-overhead",
        matmul_precision="high",
        warmup=1,
        repeats=1,
        seed=0,
    )

    rows = run_compile_triton_boundary(args)

    assert rows[0].path == "compile-visible Triton op"
    assert rows[0].error == "CUDA and Triton are required"


def test_compile_triton_sweep_parses_shape() -> None:
    assert parse_compile_triton_sweep_shape("4,2,3,5") == (4, 2, 3, 5)


def test_compile_triton_sweep_cpu_smoke_reports_triton_missing() -> None:
    args = argparse.Namespace(
        shape=[(4, 2, 3, 5)],
        device="cpu",
        checkpoint_size=2,
        compile_mode="reduce-overhead",
        matmul_precision="high",
        warmup=1,
        repeats=1,
        seed=0,
    )

    rows = run_compile_triton_sweep(args)

    assert [row.path for row in rows] == [
        "regular torch.compile scalar-loss training",
        "existing Triton rate training",
        'torch.compile public backend="triton_compile"',
    ]
    assert rows[1].error == "CUDA and Triton are required"
    assert rows[2].error == "CUDA and Triton are required"
    assert all(row.error == "CUDA and Triton are required" for row in rows[2:])


def test_mnist_compare_threads_checkpoint_size_to_rate_variants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[list[str]] = []

    args = argparse.Namespace(
        data_dir="data",
        device="cpu",
        timesteps=4,
        encoding="poisson",
        batch=8,
        hidden=16,
        epochs=1,
        lr=0.001,
        surrogate_slope=5.0,
        matmul_precision="highest",
        log_every=100,
        eval_every=100,
        eval_batches=1,
        train_limit=32,
        test_limit=32,
        grad_clip=0.1,
        rate_checkpoint_size="memory",
        snntorch_beta=0.95,
        conv_synapse_init=None,
        compile=True,
        compile_spiker_only=False,
        smooth_forward=False,
    )

    def fake_run(command: list[str], **_kwargs: object) -> object:
        commands.append(command)

        class Completed:
            returncode = 0
            stdout = "final_test_loss=1.0\nfinal_test_accuracy=0.5\n"
            stderr = ""

        return Completed()

    monkeypatch.setattr("subprocess.run", fake_run)

    run_variant(args, VARIANTS["dense"])
    run_variant(args, VARIANTS["rate"])
    generated_result = run_variant(args, VARIANTS["rate_generated"])

    assert "--checkpoint-size" not in commands[0]
    assert commands[1][commands[1].index("--checkpoint-size") + 1] == "memory"
    assert "--compile" in commands[1]
    assert "--compile" in commands[2]
    assert generated_result.compiled


def test_mnist_rate_matrix_threads_settings_and_variants(monkeypatch: pytest.MonkeyPatch) -> None:
    commands: list[list[str]] = []
    args = argparse.Namespace(
        variant=["rate", "rate_triton_compile"],
        setting=[MatrixSetting("tiny", timesteps=4, hidden=16)],
        data_dir="data",
        device="cpu",
        encoding="poisson",
        batch=8,
        epochs=1,
        lr=0.001,
        surrogate_slope=5.0,
        matmul_precision="highest",
        log_every=100,
        eval_every=100,
        eval_batches=1,
        train_limit=32,
        test_limit=32,
        grad_clip=0.1,
        rate_checkpoint_size="memory",
        snntorch_beta=0.95,
        conv_synapse_init=None,
        compile=False,
        compile_spiker_only=True,
        smooth_forward=False,
    )

    def fake_run(command: list[str], **_kwargs: object) -> object:
        commands.append(command)

        class Completed:
            returncode = 0
            stdout = "final_test_loss=1.0\nfinal_test_accuracy=0.5\n"
            stderr = ""

        return Completed()

    monkeypatch.setattr("subprocess.run", fake_run)

    rows = run_matrix(args)

    assert [row.setting.name for row in rows] == ["tiny", "tiny"]
    assert [row.result.name for row in rows] == ["rate", "rate_triton_compile"]
    assert all(command[command.index("--timesteps") + 1] == "4" for command in commands)
    assert all(command[command.index("--hidden") + 1] == "16" for command in commands)
    assert commands[1][commands[1].index("--backend") + 1] == "triton_compile"


def test_headline_parses_scalar_metrics_and_tables() -> None:
    metrics = parse_scalar_metrics("final_test_accuracy=0.9551\npeak_cuda_memory_mb=145.002\n")
    table = parse_markdown_table(
        "\n".join(
            [
                "| Name | Value |",
                "|---|---:|",
                "| alpha | 1.0 |",
                "| beta | 2.0 |",
            ]
        ),
        "| Name |",
    )

    assert metrics["final_test_accuracy"] == 0.9551
    assert metrics["peak_cuda_memory_mb"] == 145.002
    assert table == [{"Name": "alpha", "Value": "1.0"}, {"Name": "beta", "Value": "2.0"}]


def test_headline_collects_saved_artifact_summary(tmp_path) -> None:
    (tmp_path / "mnist_rate_mlp_dropout_label_smoothing_5090.md").write_text(
        "\n".join(
            [
                "final_test_loss=1.544614",
                "final_test_accuracy=0.9551",
                "peak_cuda_memory_mb=145.002",
                "steady_state_average_step_ms=6.777",
            ]
        )
    )
    (tmp_path / "mnist_conv_dropout_label_smoothing_5090.md").write_text(
        "\n".join(
            [
                "final_test_loss=1.534343",
                "final_test_accuracy=0.9646",
                "peak_cuda_memory_mb=899.411",
                "steady_state_average_step_ms=86.496",
            ]
        )
    )
    (tmp_path / "mnist_rate_matrix_5090.md").write_text(
        "\n".join(
            [
                "| Setting | Variant | Final Test Acc | Peak CUDA MB |",
                "|---|---|---:|---:|",
                "| t25_h128 | rate | 0.8408 | 110.3 |",
                "| t25_h128 | rate_triton_compile | 0.8452 | 44.0 |",
            ]
        )
    )
    (tmp_path / "performance_frontier_5090.md").write_text(
        "\n".join(
            [
                "| Contract | Variant | Fwd+Bwd ms | Speedup vs Compile |",
                "|---|---|---:|---:|",
                "| hard LIF forward | Triton fused-time |  | 2.46x |",
                "| dense-output training | torch.compile materialized graph | 0.747 | 1.00x |",
                "| rate/scalar training | Triton checkpoint rate output | 0.697 | 1.07x |",
            ]
        )
    )
    (tmp_path / "compile_triton_boundary_5090.md").write_text(
        "\n".join(
            [
                "| Path | ms | Peak MB |",
                "|---|---:|---:|",
                "| torch.compile(public triton_compile rate training) | 0.511 | 11.6 |",
                "| torch.compile(public triton_compile rate training + bias) | 0.549 | 11.6 |",
            ]
        )
    )
    (tmp_path / "snntorch_matrix_5090.md").write_text(
        "\n".join(
            [
                "| T | Comparison | Steady Step Speedup | Peak Memory Ratio |",
                "|---:|---|---:|---:|",
                "| 50 | rate vs snntorch_dense | 50.42x | 1.23x |",
                "| 50 | conv vs snntorch_conv | 11.81x | 1.22x |",
            ]
        )
    )

    rows = collect_headlines(tmp_path)

    assert [row.status for row in rows] == ["ok", "ok", "ok", "ok", "ok", "ok"]
    assert "95.51%" in rows[0].headline
    assert "110.3 MB vs triton_compile 44.0 MB" in rows[2].headline
    assert "50.42x" in rows[3].headline
    assert "2.46x" in rows[4].headline
    assert "0.511 ms" in rows[5].headline


def test_snntorch_matrix_parses_timesteps() -> None:
    assert parse_timesteps("25") == TimestepSetting(25)


def test_snntorch_matrix_threads_timesteps_and_variants(monkeypatch: pytest.MonkeyPatch) -> None:
    commands: list[list[str]] = []
    args = argparse.Namespace(
        variant=["rate", "snntorch_dense"],
        timesteps=[TimestepSetting(10), TimestepSetting(25)],
        data_dir="data",
        device="cpu",
        encoding="poisson",
        batch=8,
        hidden=16,
        epochs=1,
        lr=0.001,
        grad_clip=0.1,
        rate_checkpoint_size="balanced",
        surrogate_slope=5.0,
        snntorch_beta=0.95,
        matmul_precision="highest",
        smooth_forward=False,
        compile=False,
        compile_spiker_only=True,
        conv_synapse_init="fan_in",
        log_every=100,
        eval_every=100,
        eval_batches=1,
        train_limit=32,
        test_limit=32,
    )

    def fake_run(command: list[str], **_kwargs: object) -> object:
        commands.append(command)

        class Completed:
            returncode = 0
            stdout = (
                "final_test_loss=1.0\n"
                "final_test_accuracy=0.5\n"
                "peak_cuda_memory_mb=10.0\n"
                "steady_state_average_step_ms=2.0\n"
            )
            stderr = ""

        return Completed()

    monkeypatch.setattr("subprocess.run", fake_run)

    rows = run_snntorch_matrix(args)

    assert [row.setting.timesteps for row in rows] == [10, 10, 25, 25]
    assert [row.result.name for row in rows] == [
        "rate",
        "snntorch_dense",
        "rate",
        "snntorch_dense",
    ]
    assert commands[0][commands[0].index("--timesteps") + 1] == "10"
    assert commands[2][commands[2].index("--timesteps") + 1] == "25"
    assert "--compile" in commands[0]
    assert "--compile" not in commands[1]


def test_two_layer_recompute_parses_shape() -> None:
    assert parse_two_layer_shape("10,2,3,4,5") == TwoLayerShape(10, 2, 3, 4, 5)


def test_two_layer_recompute_cpu_reports_missing_cuda() -> None:
    args = argparse.Namespace(
        device="cpu",
        shape=[TwoLayerShape(4, 2, 3, 5, 2)],
        checkpoint_size="balanced",
        warmup=1,
        repeats=1,
        seed=0,
    )

    rows = run_two_layer_recompute(args)

    assert rows[0].variant == "composed"
    assert rows[0].error == "CUDA is required"


def test_two_layer_recompute_summary_reports_negative_result() -> None:
    shape = TwoLayerShape(10, 2, 3, 4, 2)
    summary = summarize_two_layer_result(
        [
            TwoLayerBenchRow(shape, "composed", 1.0, 10.0),
            TwoLayerBenchRow(shape, "recompute", 2.0, 12.0),
        ]
    )

    assert "Whole-model PyTorch recompute was slower" in summary


def test_compile_inspect_summarizes_output_code(tmp_path) -> None:
    output_code = tmp_path / "output_code.py"
    output_code.write_text(
        "\n".join(
            [
                "@triton_heuristics.pointwise",
                "extern_kernels.mm(a, b, out=buf0)",
                "buf0 = empty_strided_cuda((4, 4), (4, 1), torch.float32)",
                "buf1 = buf0; del buf0  # reuse",
                "buf2 = reinterpret_tensor(buf1, (2, 2), (2, 1), 0)",
            ]
        )
    )

    stats = summarize_output_code((output_code,))

    assert stats["triton_jit_kernels"] == 1
    assert stats["extern_kernels_mm"] == 1
    assert stats["empty_strided_allocs"] == 1
    assert stats["reinterpret_tensor_calls"] == 1
    assert stats["explicit_del_calls"] == 1
    assert stats["reuse_assignments"] == 1
    assert stats["buf_mentions"] >= 3


def test_custom_neuron_benchmark_variants_fit_generated_forward_abi() -> None:
    for variant in CUSTOM_NEURON_VARIANTS:
        ir = build_variant_ir(variant)
        report = analyze_neuron_ir(ir)
        inputs = torch.rand((4, 2, 3))
        state = initial_state_for_variant(variant, batch=2, neurons=3, device="cpu")

        final_state, spikes = evaluate_neuron_unroll(
            ir,
            inputs,
            state,
            params_for_variant(variant),
        )

        assert report.supports_generated_forward
        assert report.generated_forward_errors == ()
        if variant == "lif":
            assert report.supports_generated_backward
            assert report.generated_backward_errors == ()
        elif variant == "alif":
            assert not report.supports_generated_backward
            assert report.generated_backward_plan is not None
            assert report.generated_backward_plan.kind == "alif_adaptive_threshold"
            assert not report.generated_backward_plan.is_implemented
            assert report.generated_backward_errors
        else:
            assert not report.supports_generated_backward
            assert report.generated_backward_plan is None
            assert report.generated_backward_errors
        assert tuple(final_state) == ir.state
        assert spikes.shape == inputs.shape


def test_checkpoint_size_sweep_parses_sizes() -> None:
    assert parse_checkpoint_sizes("5, 10,25") == (5, 10, 25)


def test_checkpoint_size_sweep_scratch_bytes_caps_at_timesteps() -> None:
    bytes_per_value = torch.tensor([], dtype=torch.float32).element_size()

    assert (
        pre_reset_scratch_bytes(
            checkpoint_size=20,
            timesteps=10,
            batch=2,
            neurons=3,
            dtype=torch.float32,
        )
        == 10 * 2 * 3 * bytes_per_value
    )


def test_checkpoint_size_sweep_rate_pareto_frontier() -> None:
    def result(
        checkpoint_size: int,
        variant: str,
        seconds: float,
        memory: int,
    ) -> SweepResult:
        return SweepResult(
            checkpoint_size=checkpoint_size,
            chunks=1,
            variant=variant,  # pyright: ignore[reportArgumentType]
            audit=AuditResult(
                label=variant,
                split_forward_seconds=None,
                backward_seconds=None,
                allocated_bytes=0,
                forward_peak_bytes=memory,
                backward_peak_bytes=memory,
            ),
            forward_backward_seconds_override=seconds,
        )

    frontier = rate_pareto_frontier(
        [
            result(10, "rate", 0.8, 12),
            result(25, "rate", 0.7, 20),
            result(5, "rate", 1.0, 30),
            result(25, "replay_rate", 3.0, 4),
            result(25, "dense", 0.6, 100),
        ]
    )

    assert [(item.checkpoint_size, item.variant) for item in frontier] == [
        (25, "replay_rate"),
        (10, "rate"),
        (25, "rate"),
    ]


def test_benchmark_runner_resolves_artifact_paths(tmp_path) -> None:
    specs = PRESETS["smoke"][:2]
    runs = make_runs(
        specs,
        device="cuda",
        output_dir=tmp_path,
        suffix="unit",
    )

    assert len(runs) == 2
    assert runs[0].command[:3] == (
        runs[0].command[0],
        "-m",
        "spiker.benchmarks.generated_forward",
    )
    assert runs[0].command[3:5] == ("--device", "cuda")
    assert runs[0].output_path == tmp_path / "generated_forward_smoke_unit.md"
    assert runs[1].output_path == tmp_path / "packed_forward_smoke_unit.md"


def test_benchmark_runner_supports_spec_device_overrides(tmp_path) -> None:
    specs = tuple(spec for spec in PRESETS["smoke"] if spec.name == "distributed_collectives_gloo")
    runs = make_runs(
        specs,
        device="cuda",
        output_dir=tmp_path,
        suffix="unit",
    )

    assert len(runs) == 1
    assert runs[0].command[3:5] == ("--device", "cpu")
    assert runs[0].output_path == tmp_path / "distributed_collectives_gloo_smoke_unit.md"


def test_benchmark_runner_includes_distributed_collective_specs() -> None:
    assert {"distributed_collectives_gloo", "distributed_collectives_nccl"} <= {
        spec.name for spec in PRESETS["smoke"]
    }
    assert {"distributed_collectives_gloo", "distributed_collectives_nccl"} <= {
        spec.name for spec in PRESETS["core"]
    }


def test_benchmark_runner_includes_scalar_loss_boundary_specs() -> None:
    assert "scalar_loss_boundary" in {spec.name for spec in PRESETS["smoke"]}
    assert "scalar_loss_boundary" in {spec.name for spec in PRESETS["core"]}


def test_benchmark_runner_includes_performance_frontier_specs() -> None:
    assert "performance_frontier" in {spec.name for spec in PRESETS["smoke"]}
    assert "performance_frontier" in {spec.name for spec in PRESETS["core"]}


def test_benchmark_runner_includes_training_breakdown_specs() -> None:
    assert "training_breakdown" in {spec.name for spec in PRESETS["smoke"]}
    assert "training_breakdown" in {spec.name for spec in PRESETS["core"]}


def test_benchmark_runner_includes_compile_triton_boundary_specs() -> None:
    assert "compile_triton_boundary" in {spec.name for spec in PRESETS["smoke"]}
    assert "compile_triton_boundary" in {spec.name for spec in PRESETS["core"]}


def test_benchmark_runner_includes_compile_triton_sweep_specs() -> None:
    assert "compile_triton_sweep" in {spec.name for spec in PRESETS["smoke"]}
    assert "compile_triton_sweep" in {spec.name for spec in PRESETS["core"]}


def test_benchmark_runner_includes_mnist_rate_matrix_core_spec() -> None:
    assert "mnist_rate_matrix" in {spec.name for spec in PRESETS["core"]}


def test_benchmark_runner_includes_headline_without_device_arg(tmp_path) -> None:
    specs = tuple(spec for spec in PRESETS["smoke"] if spec.name == "headline")
    runs = make_runs(
        specs,
        device="cuda",
        output_dir=tmp_path,
        suffix="unit",
    )

    assert len(runs) == 1
    assert "--device" not in runs[0].command
    assert runs[0].output_path == tmp_path / "headline_unit.md"


def test_benchmark_runner_includes_hardware_export_without_device_arg(tmp_path) -> None:
    specs = tuple(spec for spec in PRESETS["smoke"] if spec.name == "hardware_export")
    runs = make_runs(
        specs,
        device="cuda",
        output_dir=tmp_path,
        suffix="unit",
    )

    assert len(runs) == 1
    assert "--device" not in runs[0].command
    assert runs[0].output_path == tmp_path / "hardware_export_smoke_unit.md"


def test_distributed_collectives_benchmark_smoke() -> None:
    import subprocess
    import sys

    import torch.distributed as torch_dist

    if not torch_dist.is_available():
        pytest.skip("torch.distributed is not available")
    if not torch_dist.is_gloo_available():
        pytest.skip("torch.distributed gloo backend is not available")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "spiker.benchmarks.distributed_collectives",
            "--timesteps",
            "2",
            "--batch",
            "1",
            "--neurons",
            "35",
            "--warmup",
            "0",
            "--repeats",
            "1",
            "--timeout",
            "30",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "# Distributed Packed Collectives Benchmark" in result.stdout
    assert "| 0 |" in result.stdout
    assert "| 1 |" in result.stdout
