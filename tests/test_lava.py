from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import spiker
from spiker.hardware import (
    export_dense_lif_layer,
    export_lava_dense_lif_spec,
    quantize_dense_lif_export,
)
from spiker.lava import build_lava_dense_lif_processes, lava_available
from spiker.neurons import LIFParams


class FakePort:
    def __init__(self) -> None:
        self.connected_to: FakePort | None = None

    def connect(self, other: FakePort) -> None:
        self.connected_to = other


class FakeDense:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.a_out = FakePort()


class FakeLIF:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.a_in = FakePort()


def _make_quantized_and_spec():
    dense = export_dense_lif_layer(
        torch.tensor([[0.0, 0.5, -1.0], [1.0, -0.5, 0.25]]),
        torch.tensor([0.1, -0.2, 0.3]),
        LIFParams(tau_mem=8.0, threshold=0.75, reset=0.0),
        dt=0.001,
    )
    quantized = quantize_dense_lif_export(dense, num_bits=8)
    spec = export_lava_dense_lif_spec(
        quantized,
        quantized_export_path="layer.dense_lif_quantized.json",
    )
    return quantized, spec


def test_build_lava_dense_lif_processes_instantiates_and_wires_fake_lava(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import spiker.lava as lava_module

    def fake_import_module(name: str) -> object:
        if name == "lava.proc.dense.process":
            return SimpleNamespace(Dense=FakeDense)
        if name == "lava.proc.lif.process":
            return SimpleNamespace(LIF=FakeLIF)
        raise AssertionError(name)

    monkeypatch.setattr(lava_module.importlib, "import_module", fake_import_module)
    quantized, spec = _make_quantized_and_spec()

    processes = build_lava_dense_lif_processes(spec, quantized=quantized, name_prefix="unit")

    assert isinstance(processes.dense, FakeDense)
    assert isinstance(processes.lif, FakeLIF)
    assert processes.dense.a_out.connected_to is processes.lif.a_in
    assert processes.dense.kwargs["name"] == "unit_dense"
    assert processes.dense.kwargs["num_weight_bits"] == 8
    assert processes.weights.tolist() == torch.tensor(quantized.weight).t().tolist()
    assert processes.lif.kwargs["name"] == "unit_lif"
    assert processes.lif.kwargs["shape"] == (3,)
    assert processes.lif.kwargs["du"] == pytest.approx(1.0)
    assert processes.lif.kwargs["dv"] == pytest.approx(1.0 / 8.0)
    bias_mant = processes.lif.kwargs["bias_mant"]
    assert isinstance(bias_mant, np.ndarray)
    assert bias_mant.tolist() == quantized.bias
    assert processes.lif.kwargs["bias_exp"] == 0
    assert processes.lif.kwargs["vth"] == pytest.approx(0.75)


def test_build_lava_dense_lif_processes_rejects_mismatched_quantized_export(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import spiker.lava as lava_module

    monkeypatch.setattr(
        lava_module.importlib,
        "import_module",
        lambda name: SimpleNamespace(Dense=FakeDense, LIF=FakeLIF),
    )
    quantized, spec = _make_quantized_and_spec()
    mismatched = quantized.__class__(
        **{
            **quantized.to_dict(),
            "output_size": quantized.output_size + 1,
        }
    )

    with pytest.raises(ValueError, match="shapes"):
        build_lava_dense_lif_processes(spec, quantized=mismatched)


def test_lava_available_handles_import_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    import spiker.lava as lava_module

    def fail_import_module(_name: str) -> object:
        raise ValueError("broken Lava install")

    monkeypatch.setattr(lava_module.importlib, "import_module", fail_import_module)

    assert not lava_available()
    with pytest.raises(RuntimeError, match="could not be imported"):
        build_lava_dense_lif_processes(_make_quantized_and_spec()[1], quantized=None)


def test_lava_builder_api_is_public() -> None:
    assert spiker.LavaDenseLIFProcesses is not None
    assert spiker.build_lava_dense_lif_processes is build_lava_dense_lif_processes
    assert spiker.lava_available is lava_available
