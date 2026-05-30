from __future__ import annotations

import torch

from myelin.losses import SpikeRateLoss, spike_rate_loss


def test_spike_rate_loss_matches_manual_mse() -> None:
    spikes = torch.tensor([0.0, 1.0, 1.0, 0.0])

    loss = spike_rate_loss(spikes, target_rate=0.25)

    assert torch.allclose(loss, torch.tensor((0.5 - 0.25) ** 2))


def test_spike_rate_loss_module_matches_function() -> None:
    spikes = torch.rand((4, 2, 3))

    assert torch.allclose(SpikeRateLoss(0.2)(spikes), spike_rate_loss(spikes, 0.2))
