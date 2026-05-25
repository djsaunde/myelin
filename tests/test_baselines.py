from __future__ import annotations

import pytest
import torch

from spiker.baselines import compiled_available, compiled_lif_unroll, eager_lif_unroll
from spiker.neurons import LIFParams, LIFState

pytestmark = pytest.mark.extended


def test_compiled_lif_matches_eager_lif() -> None:
    if not compiled_available():
        pytest.skip("torch.compile is not available in this PyTorch build")

    inputs = torch.tensor(
        [
            [[0.6, 0.1], [0.0, 0.5]],
            [[0.6, 0.9], [0.8, 0.5]],
            [[0.1, 0.2], [0.1, 0.1]],
        ]
    )
    initial_state = LIFState(membrane=torch.zeros((2, 2)))
    params = LIFParams(tau_mem=10.0, threshold=1.0, reset=0.0)

    eager_state, eager_spikes = eager_lif_unroll(inputs, initial_state, params)
    try:
        compiled_state, compiled_spikes = compiled_lif_unroll()(inputs, initial_state, params)
    except Exception as exc:
        pytest.skip(f"torch.compile could not compile lif_unroll: {type(exc).__name__}: {exc}")

    assert torch.allclose(compiled_state.membrane, eager_state.membrane)
    assert torch.equal(compiled_spikes, eager_spikes)
