from __future__ import annotations

import warnings

import pytest
import torch

from spiker import kernels


def test_cuda_torch_backend_warns_when_triton_is_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(kernels, "has_triton", lambda: True)
    inputs = torch.empty((1,))
    monkeypatch.setattr(type(inputs), "is_cuda", property(lambda self: True))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        kernels._warn_if_cuda_torch_backend(inputs, operation="lif_forward")

    assert len(caught) == 1
    assert issubclass(caught[0].category, RuntimeWarning)
    assert "backend='torch'" in str(caught[0].message)
    assert "backend='auto'" in str(caught[0].message)


def test_cuda_torch_backend_does_not_warn_when_triton_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(kernels, "has_triton", lambda: False)
    inputs = torch.empty((1,))
    monkeypatch.setattr(type(inputs), "is_cuda", property(lambda self: True))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        kernels._warn_if_cuda_torch_backend(inputs, operation="lif_forward")

    assert len(caught) == 0


def test_cpu_torch_backend_does_not_warn_when_triton_is_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(kernels, "has_triton", lambda: True)
    inputs = torch.empty((1,))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        kernels._warn_if_cuda_torch_backend(inputs, operation="lif_forward")

    assert len(caught) == 0
