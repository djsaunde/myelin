from __future__ import annotations

import pytest

from myelin.neurons import (
    ALIFParams,
    ALIFState,
    IzhikevichParams,
    IzhikevichState,
    LIFParams,
    LIFState,
)


def test_lif_step_spikes_and_hard_resets() -> None:
    import torch

    from myelin.neurons import lif_step

    state = LIFState(membrane=torch.tensor([[0.5, 0.0]]))
    inputs = torch.tensor([[0.6, 0.2]])
    params = LIFParams(tau_mem=10.0, threshold=1.0, reset=-0.25)

    next_state, spike = lif_step(state, inputs, params)

    assert torch.equal(spike, torch.tensor([[1.0, 0.0]]))
    assert torch.allclose(next_state.membrane, torch.tensor([[-0.25, 0.2]]))


def test_lif_unroll_matches_repeated_steps() -> None:
    import torch

    from myelin.functional import lif_unroll
    from myelin.neurons import lif_step

    inputs = torch.tensor(
        [
            [[0.6, 0.1]],
            [[0.6, 0.9]],
            [[0.1, 0.2]],
        ]
    )
    initial_state = LIFState(membrane=torch.zeros((1, 2)))
    params = LIFParams(tau_mem=10.0, threshold=1.0, reset=0.0)

    expected_state = initial_state
    expected_spikes = []
    for step_input in inputs:
        expected_state, spike = lif_step(expected_state, step_input, params)
        expected_spikes.append(spike)

    final_state, spikes = lif_unroll(inputs, initial_state, params)

    assert torch.allclose(final_state.membrane, expected_state.membrane)
    assert torch.equal(spikes, torch.stack(expected_spikes))


def test_lif_shape_errors_are_explicit() -> None:
    import torch

    from myelin.functional import lif_unroll
    from myelin.neurons import lif_step

    params = LIFParams()
    with pytest.raises(ValueError, match="same shape"):
        lif_step(LIFState(membrane=torch.zeros((2, 3))), torch.zeros((2, 4)), params)

    with pytest.raises(ValueError, match=r"\[T, B, N\]"):
        lif_unroll(torch.zeros((2, 3)), LIFState(membrane=torch.zeros((3,))), params)


def test_alif_step_uses_adaptive_threshold_and_updates_adaptation() -> None:
    import torch

    from myelin.neurons import alif_step

    state = ALIFState(
        membrane=torch.tensor([[0.5, 0.0]]),
        adaptation=torch.tensor([[0.2, 0.0]]),
    )
    inputs = torch.tensor([[0.7, 0.8]])
    params = ALIFParams(
        tau_mem=10.0,
        tau_adaptation=5.0,
        threshold=1.0,
        reset=-0.25,
        beta=1.0,
    )

    next_state, spike = alif_step(state, inputs, params)

    assert torch.equal(spike, torch.tensor([[0.0, 0.0]]))
    assert torch.allclose(next_state.membrane, torch.tensor([[1.15, 0.8]]))
    assert torch.allclose(next_state.adaptation, torch.tensor([[0.16, 0.0]]))


def test_alif_step_spike_increments_adaptation() -> None:
    import torch

    from myelin.neurons import alif_step

    state = ALIFState(
        membrane=torch.tensor([[0.5, 0.0]]),
        adaptation=torch.tensor([[0.1, 0.0]]),
    )
    inputs = torch.tensor([[0.7, 1.1]])
    params = ALIFParams(tau_mem=10.0, tau_adaptation=5.0, threshold=1.0, reset=0.0, beta=0.5)

    next_state, spike = alif_step(state, inputs, params)

    assert torch.equal(spike, torch.tensor([[1.0, 1.0]]))
    assert torch.equal(next_state.membrane, torch.zeros_like(next_state.membrane))
    assert torch.allclose(next_state.adaptation, torch.tensor([[1.08, 1.0]]))


def test_izhikevich_step_spikes_and_resets() -> None:
    import torch

    from myelin.neurons import izhikevich_step

    state = IzhikevichState(
        voltage=torch.tensor([[29.0, -65.0]]),
        recovery=torch.tensor([[0.0, -13.0]]),
    )
    inputs = torch.tensor([[20.0, 0.0]])
    params = IzhikevichParams()

    next_state, spike = izhikevich_step(state, inputs, params)

    assert torch.equal(spike, torch.tensor([[1.0, 0.0]]))
    assert torch.allclose(next_state.voltage[:, :1], torch.tensor([[-65.0]]))
    assert torch.allclose(next_state.recovery[:, :1], torch.tensor([[8.116]]))
    assert torch.all(next_state.voltage[:, 1:] < params.threshold)


def test_izhikevich_unroll_matches_repeated_steps() -> None:
    import torch

    from myelin.functional import izhikevich_unroll
    from myelin.neurons import izhikevich_step

    inputs = torch.tensor(
        [
            [[20.0, 0.0]],
            [[5.0, 5.0]],
            [[0.0, 0.0]],
        ]
    )
    initial_state = IzhikevichState(
        voltage=torch.tensor([[29.0, -65.0]]),
        recovery=torch.tensor([[0.0, -13.0]]),
    )
    params = IzhikevichParams()

    expected_state = initial_state
    expected_spikes = []
    for step_input in inputs:
        expected_state, spike = izhikevich_step(expected_state, step_input, params)
        expected_spikes.append(spike)

    final_state, spikes = izhikevich_unroll(inputs, initial_state, params)

    assert torch.allclose(final_state.voltage, expected_state.voltage)
    assert torch.allclose(final_state.recovery, expected_state.recovery)
    assert torch.equal(spikes, torch.stack(expected_spikes))
