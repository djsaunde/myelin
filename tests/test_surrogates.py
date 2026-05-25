from __future__ import annotations

import pytest
import torch

import spiker
from spiker.surrogates import (
    SURROGATE_NAMES,
    surrogate_derivative,
    surrogate_from_name,
    surrogate_name,
)


def test_registered_surrogates_roundtrip_through_public_helpers() -> None:
    centered = torch.tensor([[-2.0, -0.25, 0.0, 0.5, 3.0]])

    for name in SURROGATE_NAMES:
        fn = surrogate_from_name(name)

        assert surrogate_name(fn) == name
        assert getattr(spiker, f"{name}_surrogate") is fn
        assert surrogate_derivative(centered, name).shape == centered.shape


def test_surrogate_registry_rejects_unknown_callable() -> None:
    def custom_surrogate(centered: torch.Tensor) -> torch.Tensor:
        return centered

    with pytest.raises(ValueError, match="built-in surrogate functions"):
        surrogate_name(custom_surrogate)
